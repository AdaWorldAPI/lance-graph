//! D-GSO-8 (P8): seal boundary.
//!
//! Plan: `.claude/plans/2026-10-06-global-sudoku-replayable-orchestration-v1.md`
//! §15 (Rubikon / seal as materialization governor) and §18 P8.
//!
//! Claim under test: many internal operations run without any durable write;
//! a seal happens only at one declared semantic boundary; and replaying the
//! persisted events from the prior seal reproduces the next seal exactly.
//!
//! # Reused, unchanged
//!
//! The seal is the shipped `persist_sink::DetachedCycleBatch::freeze`: it
//! stable-orders the casts by `stream_position`, coalesces same-row updates
//! (later position wins) and hashes the canonical content together with the
//! frame (`cycle`, `base_version`). The probe compares `batch_hash` and the
//! coalesced `image`.
//!
//! # The tie limit, and the choice made here
//!
//! `freeze` sorts stably, so casts with equal `stream_position` keep their
//! arrival order, and arrival order is not durable
//! (`.claude/knowledge/seal-vs-temporal-ordering-information.md` §2). The plan
//! offers three ways out; this probe takes the first: **the key is the
//! event's own sequence number, globally unique and persisted with the
//! event.** `apply` enforces that by refusing any event whose `seq` is not
//! greater than the last one (`a_reused_seq_is_refused`). `seal` also refuses
//! a batch with a tied key instead of sealing it, and
//! `tied_keys_make_the_seal_depend_on_arrival` shows on the real `freeze`
//! why that refusal is needed.
//!
//! # What this probe does not decide
//!
//! - **The ledger is local.** It keeps `(cycle, base_version, batch_hash)`
//!   and the frozen batch; no `WalSink` is driven, so publication, fencing and
//!   recovery are not exercised here.
//! - **One boundary kind.** The boundary is a declared event
//!   (`Event::Boundary`); which semantic events earn it (§15 lists several)
//!   is not decided.
//! - **The internal operation is a stand-in**: each event folds one byte into
//!   one row with a fixed rule.
//!
//! Run: `cargo run -p lance-graph-planner --example seal_boundary_probe`
//! Tests: `cargo test -p lance-graph-planner --example seal_boundary_probe`

use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_planner::persist_sink::{CycleFrame, CycleId, DetachedCycleBatch, SweepSlot};

/// Rows in the probe's state.
const ROWS: usize = 16;
/// The owner every landing is on behalf of.
const OWNER: u32 = 7;

/// One persisted event. `seq` is its durable, globally unique sequence number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    /// An internal operation: fold `value` into `row`.
    Internal { seq: u64, row: u8, value: u8 },
    /// The declared semantic boundary: seal what changed since the last seal.
    Boundary { seq: u64 },
}

/// The stand-in internal operation: a fixed, non-commutative fold.
const fn fold(current: u8, value: u8) -> u8 {
    current.rotate_left(3) ^ value
}

/// One sealed cycle as the ledger keeps it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Seal {
    cycle: CycleId,
    base_version: DatasetVersion,
    batch: DetachedCycleBatch,
}

/// Why an event was refused or a boundary did not seal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SealError {
    /// Two casts share a `stream_position`; their order would come from
    /// arrival, which is not durable.
    TiedKey(u64),
    /// An event's `seq` is not greater than the last accepted one. Row
    /// coalescing would hide a reused `seq` from `first_tie`, so the key's
    /// uniqueness is checked here, per event, before anything folds.
    StaleSeq(u64),
}

/// The in-memory working state plus the durable ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Engine {
    /// Working rows; changed freely, never durable by themselves.
    rows: [u8; ROWS],
    /// The rows as of the last seal.
    sealed_rows: [u8; ROWS],
    /// For each row changed since the last seal, the `seq` of its last change.
    last_change: [Option<u64>; ROWS],
    /// Durable record. The only thing a boundary writes to.
    ledger: Vec<Seal>,
    /// Internal operations applied since start (for the report).
    internal_ops: usize,
    /// The last accepted `seq`. Sequence numbers must strictly increase.
    last_seq: Option<u64>,
}

impl Engine {
    fn new() -> Self {
        Self {
            rows: [0; ROWS],
            sealed_rows: [0; ROWS],
            last_change: [None; ROWS],
            ledger: Vec::new(),
            internal_ops: 0,
            last_seq: None,
        }
    }

    /// The version a new seal builds on: the previous seal's successor, or 0.
    fn head(&self) -> DatasetVersion {
        DatasetVersion(self.ledger.len() as u64)
    }

    /// Apply one event. Internal events touch only working state; a boundary
    /// seals.
    fn apply(&mut self, event: Event) -> Result<(), SealError> {
        let seq = match event {
            Event::Internal { seq, .. } | Event::Boundary { seq } => seq,
        };
        if self.last_seq.is_some_and(|last| seq <= last) {
            return Err(SealError::StaleSeq(seq));
        }
        self.last_seq = Some(seq);
        match event {
            Event::Internal { seq, row, value } => {
                let r = usize::from(row) % ROWS;
                self.rows[r] = fold(self.rows[r], value);
                self.last_change[r] = Some(seq);
                self.internal_ops += 1;
                Ok(())
            }
            Event::Boundary { .. } => self.seal(),
        }
    }

    /// The casts a boundary would seal: one per row whose value differs from
    /// the last seal, keyed by the `seq` of its last change.
    fn casts(&self) -> Vec<SweepSlot> {
        let cycle = CycleId(self.ledger.len() as u64);
        (0..ROWS)
            .filter(|&r| self.rows[r] != self.sealed_rows[r])
            .map(|r| SweepSlot {
                cycle,
                stream_position: self.last_change[r].expect("a changed row has a last change"),
                owner: OWNER,
                row: r as u64,
                paired_move: None,
                payload: vec![self.rows[r]],
            })
            .collect()
    }

    /// Seal at the boundary. Nothing changed: no write. A tied key: refuse.
    fn seal(&mut self) -> Result<(), SealError> {
        self.seal_casts(self.casts())
    }

    fn seal_casts(&mut self, casts: Vec<SweepSlot>) -> Result<(), SealError> {
        if casts.is_empty() {
            return Ok(());
        }
        if let Some(k) = first_tie(&casts) {
            return Err(SealError::TiedKey(k));
        }
        let cycle = CycleId(self.ledger.len() as u64);
        let base_version = self.head();
        let batch = DetachedCycleBatch::freeze(CycleFrame::new(cycle, base_version), casts);
        self.sealed_rows = self.rows;
        self.last_change = [None; ROWS];
        self.ledger.push(Seal {
            cycle,
            base_version,
            batch,
        });
        Ok(())
    }

    /// Rebuild the state a seal left behind: the previous sealed rows with the
    /// seal's coalesced image written over them. The highest persisted key
    /// becomes `last_seq`, so a resumed engine still refuses a reused one.
    fn resume_after(prior: &[Seal]) -> Self {
        let mut e = Self::new();
        for s in prior {
            for (&row, payload) in &s.batch.image {
                e.sealed_rows[row as usize] = payload[0];
            }
            let top = s.batch.landings.iter().map(|l| l.stream_position).max();
            e.last_seq = e.last_seq.max(top);
            e.ledger.push(s.clone());
        }
        e.rows = e.sealed_rows;
        e
    }
}

/// The first `stream_position` shared by two casts, if any.
fn first_tie(casts: &[SweepSlot]) -> Option<u64> {
    let mut keys: Vec<u64> = casts.iter().map(|c| c.stream_position).collect();
    keys.sort_unstable();
    keys.windows(2).find(|w| w[0] == w[1]).map(|w| w[0])
}

/// A deterministic event stream: `n` internal events over a few rows, with a
/// boundary after every `every` internal events. Sequence numbers are unique.
fn stream(n: u64, every: u64) -> Vec<Event> {
    let mut v = Vec::new();
    let mut seq = 0;
    for i in 0..n {
        seq += 1;
        let row = ((i * 7 + 3) % 5) as u8;
        let value = (i.wrapping_mul(2_654_435_761) >> 13) as u8;
        v.push(Event::Internal { seq, row, value });
        if (i + 1) % every == 0 {
            seq += 1;
            v.push(Event::Boundary { seq });
        }
    }
    v
}

/// Split a stream into the event runs that end at each boundary.
fn runs(events: &[Event]) -> Vec<&[Event]> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, e) in events.iter().enumerate() {
        if matches!(e, Event::Boundary { .. }) {
            out.push(&events[start..=i]);
            start = i + 1;
        }
    }
    out
}

fn main() {
    let events = stream(1_000, 250);
    let mut engine = Engine::new();
    for &e in &events {
        engine.apply(e).expect("unique keys");
    }
    println!(
        "{} internal operations, {} seals",
        engine.internal_ops,
        engine.ledger.len()
    );
    for s in &engine.ledger {
        println!(
            "  cycle {} on base v{}: {} landings, hash {:016x}",
            s.cycle.0,
            s.base_version.0,
            s.batch.landings.len(),
            s.batch.batch_hash
        );
    }
    // Replay the last run from the seal before it.
    let last = engine.ledger.len() - 1;
    let mut replayed = Engine::resume_after(&engine.ledger[..last]);
    for &e in *runs(&events).last().expect("one run") {
        replayed.apply(e).expect("unique keys");
    }
    assert_eq!(replayed.ledger[last], engine.ledger[last]);
    println!("replay from the prior seal reproduced the last seal");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FAILS IF: an internal operation writes to the ledger, or a boundary
    /// writes more than one seal.
    #[test]
    fn only_a_boundary_writes() {
        let mut e = Engine::new();
        for ev in stream(999, 1_000) {
            e.apply(ev).unwrap();
        }
        assert_eq!(e.internal_ops, 999);
        assert!(e.ledger.is_empty(), "no boundary yet, so nothing durable");

        e.apply(Event::Boundary { seq: 10_000 }).unwrap();
        assert_eq!(e.ledger.len(), 1);
        // Rows 0..5 were touched; the seal holds each once, coalesced.
        assert_eq!(e.ledger[0].batch.image.len(), 5);
    }

    /// FAILS IF: a boundary with nothing changed since the last seal writes a
    /// seal anyway.
    #[test]
    fn an_unchanged_boundary_writes_nothing() {
        let mut e = Engine::new();
        e.apply(Event::Boundary { seq: 1 }).unwrap();
        assert!(e.ledger.is_empty());

        e.apply(Event::Internal {
            seq: 2,
            row: 1,
            value: 9,
        })
        .unwrap();
        e.apply(Event::Boundary { seq: 3 }).unwrap();
        e.apply(Event::Boundary { seq: 4 }).unwrap();
        assert_eq!(e.ledger.len(), 1, "the second boundary had nothing to seal");
    }

    /// FAILS IF: replaying the persisted events from the prior seal does not
    /// reproduce each next seal exactly (frame, landings, image, hash).
    #[test]
    fn replay_from_the_prior_seal_reproduces_the_next() {
        let events = stream(1_000, 125);
        let mut live = Engine::new();
        for &ev in &events {
            live.apply(ev).unwrap();
        }
        let runs = runs(&events);
        assert_eq!(live.ledger.len(), 8);
        assert_eq!(runs.len(), 8);

        for (k, run) in runs.iter().enumerate() {
            let mut replay = Engine::resume_after(&live.ledger[..k]);
            for &ev in *run {
                replay.apply(ev).unwrap();
            }
            assert_eq!(replay.ledger.len(), k + 1);
            assert_eq!(replay.ledger[k], live.ledger[k], "seal {k}");
            assert_eq!(
                replay.rows,
                live_rows_after(&events, k),
                "rows after seal {k}"
            );
        }
    }

    /// The working rows right after the `k`-th seal, by running the stream.
    fn live_rows_after(events: &[Event], k: usize) -> [u8; ROWS] {
        let mut e = Engine::new();
        for &ev in events {
            e.apply(ev).unwrap();
            if e.ledger.len() == k + 1 {
                return e.rows;
            }
        }
        unreachable!("stream has at least k + 1 seals")
    }

    /// FAILS IF: the seal does not bind its base, its content or its order:
    /// replaying from the wrong prior seal, dropping an event, or swapping two
    /// events on one row must each change the next seal.
    #[test]
    fn the_seal_binds_base_content_and_order() {
        let events = stream(400, 100);
        let mut live = Engine::new();
        for &ev in &events {
            live.apply(ev).unwrap();
        }
        let runs = runs(&events);
        let reference = &live.ledger[2];

        // Wrong base: replay run 2 on top of only the first seal.
        let mut wrong_base = Engine::resume_after(&live.ledger[..1]);
        for &ev in runs[2] {
            wrong_base.apply(ev).unwrap();
        }
        assert_ne!(
            wrong_base.ledger.last().unwrap().batch.batch_hash,
            reference.batch.batch_hash
        );

        // Dropped event.
        let mut dropped = Engine::resume_after(&live.ledger[..2]);
        for &ev in &runs[2][1..] {
            dropped.apply(ev).unwrap();
        }
        assert_ne!(
            dropped.ledger[2].batch.batch_hash,
            reference.batch.batch_hash
        );

        // Two operations on the same row, applied in the other order (the
        // fold does not commute). The values swap and the `seq`s stay put, so
        // the stream is still strictly increasing and is not refused.
        let mut swapped: Vec<Event> = runs[2].to_vec();
        let row_of = |e: &Event| match e {
            Event::Internal { row, .. } => Some(*row),
            Event::Boundary { .. } => None,
        };
        let value_of = |e: &Event| match e {
            Event::Internal { value, .. } => *value,
            Event::Boundary { .. } => unreachable!("only internal events swap"),
        };
        let i = 0;
        let j = (1..swapped.len())
            .find(|&j| row_of(&swapped[j]) == row_of(&swapped[i]))
            .unwrap();
        let (vi, vj) = (value_of(&swapped[i]), value_of(&swapped[j]));
        assert_ne!(vi, vj, "fixture: the two values must differ");
        for (k, v) in [(i, vj), (j, vi)] {
            if let Event::Internal { value, .. } = &mut swapped[k] {
                *value = v;
            }
        }
        let mut reordered = Engine::resume_after(&live.ledger[..2]);
        for ev in swapped {
            reordered.apply(ev).unwrap();
        }
        assert_ne!(reordered.ledger[2].batch.image, reference.batch.image);
    }

    /// FAILS IF: the seal depends on the order casts arrive in when their keys
    /// are unique. The freeze canonicalizes, so arrival order is irrelevant.
    #[test]
    fn unique_keys_make_the_seal_arrival_independent() {
        let mut e = Engine::new();
        for ev in stream(60, 1_000) {
            e.apply(ev).unwrap();
        }
        let casts = e.casts();
        let mut reversed = casts.clone();
        reversed.reverse();
        let frame = CycleFrame::new(CycleId(0), DatasetVersion(0));
        assert_eq!(
            DetachedCycleBatch::freeze(frame, casts).batch_hash,
            DetachedCycleBatch::freeze(frame, reversed).batch_hash
        );
    }

    /// FAILS IF: a reused `seq` is accepted. Both cases from review: a reuse
    /// on the same row (row coalescing hides it from `first_tie`) and a reuse
    /// across two seals (each batch alone has no tie). Also after a resume.
    #[test]
    fn a_reused_seq_is_refused() {
        // Same row, same seq: the second event folds nothing.
        let mut e = Engine::new();
        e.apply(Event::Internal {
            seq: 5,
            row: 2,
            value: 1,
        })
        .unwrap();
        assert_eq!(
            e.apply(Event::Internal {
                seq: 5,
                row: 2,
                value: 9
            }),
            Err(SealError::StaleSeq(5))
        );
        assert_eq!(e.rows[2], fold(0, 1), "the refused event did not fold");
        assert_eq!(e.internal_ops, 1);

        // Across seals: seq 5 is sealed, then reused after the boundary.
        e.apply(Event::Boundary { seq: 6 }).unwrap();
        assert_eq!(e.ledger.len(), 1);
        assert_eq!(
            e.apply(Event::Internal {
                seq: 5,
                row: 3,
                value: 4
            }),
            Err(SealError::StaleSeq(5))
        );

        // After a resume from that seal, the persisted key is still known.
        let mut resumed = Engine::resume_after(&e.ledger);
        assert_eq!(
            resumed.apply(Event::Internal {
                seq: 5,
                row: 3,
                value: 4
            }),
            Err(SealError::StaleSeq(5))
        );
        // Silence twin: a fresh, larger seq is accepted.
        resumed
            .apply(Event::Internal {
                seq: 7,
                row: 3,
                value: 4,
            })
            .unwrap();
    }

    /// The known limit, measured on the shipped `freeze`. FAILS IF: tied keys
    /// stop making the seal depend on arrival order (then the refusal would be
    /// unnecessary), or the probe seals a tied batch instead of refusing it.
    #[test]
    fn tied_keys_make_the_seal_depend_on_arrival() {
        let slot = |row: u64, byte: u8| SweepSlot {
            cycle: CycleId(0),
            stream_position: 42,
            owner: OWNER,
            row,
            paired_move: None,
            payload: vec![byte],
        };
        let frame = CycleFrame::new(CycleId(0), DatasetVersion(0));
        // Same row, same key, two arrival orders: the coalesced value differs.
        let a = DetachedCycleBatch::freeze(frame, vec![slot(3, 1), slot(3, 2)]);
        let b = DetachedCycleBatch::freeze(frame, vec![slot(3, 2), slot(3, 1)]);
        assert_ne!(a.image, b.image);
        // Different rows, same key: the landing order and hash differ.
        let c = DetachedCycleBatch::freeze(frame, vec![slot(1, 1), slot(2, 2)]);
        let d = DetachedCycleBatch::freeze(frame, vec![slot(2, 2), slot(1, 1)]);
        assert_ne!(c.batch_hash, d.batch_hash);

        let mut e = Engine::new();
        assert_eq!(
            e.seal_casts(vec![slot(1, 1), slot(2, 2)]),
            Err(SealError::TiedKey(42))
        );
        assert!(e.ledger.is_empty(), "a tied batch is refused, not sealed");
    }
}
