//! The post-write kanbanstep — the ordering core of the D-MBX-A6 persistence sink.
//!
//! This is the POST-write half of the fire-and-forget flow (the pre-write half is
//! [`crate::owner_adapter`]). It runs on the **independent persistence path**, not
//! the thinker: by the time the sink drains a cast, the thinker has already
//! reported (`BatchWriter::cast`) and moved on. There is no ack, no callback into
//! the thinker, no confirmation ledger.
//!
//! ## The ordering invariant (operator-ruled — the thing the falsifier proves)
//!
//! ```text
//! durable write FIRST  →  the new LanceVersion IS the SoA's latest state
//!                      →  ONLY THEN apply the cast's PAIRED move (the KanbanStep)
//! ```
//!
//! - **No successful write ⇒ no step.** A write that does not land leaves the
//!   owner's lifecycle untouched.
//! - **The step is the PAIRED move**, never a generic `next_phases().first()`.
//!   The move the thought cast with this write (`paired.to`) is what the owner
//!   advances to — a version appearing is not a licence to manufacture a generic
//!   forward-arc tick. (Contrast [`crate::batch_writer`]'s note and the generic
//!   `NextPhaseScheduler`, which is a *different*, legitimately-generic loop.)
//! - The owner's checked [`MailboxSoaOwner::try_advance_phase`] stays the SOLE
//!   mutator; an illegal paired edge is surfaced ([`PersistError::Illegal`]),
//!   never silently applied.
//!
//! ## The `DurableWrite` seam
//!
//! [`DurableWrite`] is the durable-append surface the sink drains into. Its
//! concrete implementation lives in a lance-having crate and wires Lance 7's
//! shipped MemWAL (`dataset::mem_wal::WalAppender::append(Vec<RecordBatch>)` via
//! `ShardWriter`) — it invents no persistence machinery. Keeping the durable
//! write behind this trait leaves the **ordering logic and its falsifier
//! lance-free and runnable without `protoc`**, which is the whole point of a
//! probe-first slice: prove the ordering before paying for the MemWAL build.

use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{KanbanMove, RubiconTransitionError};
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_contract::soa_view::MailboxSoaOwner;

/// A durable write that did not land (the WAL append failed / was fenced).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteFailed(pub String);

/// The durable-append seam the persistence sink drains a cast into.
///
/// The concrete impl (a lance-having crate) wires Lance 7's MemWAL
/// (`WalAppender::append`); this trait keeps the ordering core lance-free.
pub trait DurableWrite {
    /// Durably append the owner's live SoA backing state (read zero-copy at flush
    /// — the `bytes` are the resident `NodeRowPacket` view, never owned deltas) on
    /// behalf of `owner`.
    ///
    /// `Ok(version)` = the write landed and that `LanceVersion` is now the SoA's
    /// latest thinking state. `Err` = it did not land (⇒ the caller applies no
    /// lifecycle step).
    fn append(&mut self, owner: MailboxId, bytes: &[u8]) -> Result<DatasetVersion, WriteFailed>;
}

/// Why a persist-then-step cycle produced no lifecycle advance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistError {
    /// The durable write did not land — **no lifecycle step was applied** and the
    /// owner is untouched (the ordering invariant's negative half).
    Write(WriteFailed),
    /// The write landed, but the cast's paired move is not a legal Rubicon edge
    /// from the owner's current phase — surfaced, never silently applied.
    Illegal(RubiconTransitionError),
}

/// The post-write kanbanstep: durably write the owner's SoA state, then — and
/// **only on a successful write** — apply the cast's PAIRED move.
///
/// Returns:
/// - `Ok(Some(step))` — the write landed and the paired move was applied (the
///   `step` is the owner's own emitted [`KanbanMove`], carrying its live cycle);
/// - `Ok(None)` — the write landed but the cast carried no move (durable, no step);
/// - `Err(PersistError::Write)` — the write did not land (**no step**; owner untouched);
/// - `Err(PersistError::Illegal)` — the paired edge is illegal from the current
///   phase (surfaced; owner untouched).
///
/// The applied transition target is `paired.to` — the move the thought cast with
/// this write — never a scheduler-manufactured generic successor.
pub fn persist_then_step<W, O>(
    owner: &mut O,
    write: &mut W,
    paired: Option<KanbanMove>,
    bytes: &[u8],
) -> Result<Option<KanbanMove>, PersistError>
where
    W: DurableWrite,
    O: MailboxSoaOwner,
{
    // 1. Durable write FIRST. If it does not land, return without touching the
    //    owner's lifecycle — no successful write ⇒ no step.
    let _version = write
        .append(owner.mailbox_id(), bytes)
        .map_err(PersistError::Write)?;
    // 2. Post-write: apply the PAIRED move's target via the checked airgap. Never
    //    a generic `next_phases().first()` — `paired.to` is the thought's intent.
    match paired {
        Some(mv) => owner
            .try_advance_phase(mv.to)
            .map(Some)
            .map_err(PersistError::Illegal),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::collapse_gate::MailboxId as MId;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};
    use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};

    /// Minimal in-RAM owner (mirrors `kanban_actor::tests::TestBoard`).
    struct FakeOwner {
        id: MId,
        phase: KanbanColumn,
        cycle: u32,
    }
    impl MailboxSoaView for FakeOwner {
        fn mailbox_id(&self) -> MId {
            self.id
        }
        fn n_rows(&self) -> usize {
            0
        }
        fn w_slot(&self) -> u8 {
            0
        }
        fn current_cycle(&self) -> u32 {
            self.cycle
        }
        fn phase(&self) -> KanbanColumn {
            self.phase
        }
        fn energy(&self) -> &[f32] {
            &[]
        }
        fn edges_raw(&self) -> &[u64] {
            &[]
        }
        fn meta_raw(&self) -> &[u32] {
            &[]
        }
        fn entity_type(&self) -> &[u16] {
            &[]
        }
    }
    impl MailboxSoaOwner for FakeOwner {
        fn advance_phase(&mut self, to: KanbanColumn) -> KanbanMove {
            let from = self.phase;
            self.phase = to;
            self.cycle = self.cycle.wrapping_add(1);
            KanbanMove {
                mailbox: self.id,
                from,
                to,
                witness_chain_position: self.cycle,
                exec: ExecTarget::Native,
            }
        }
    }

    /// A `DurableWrite` whose success/failure and call-count are observable.
    struct FakeWal {
        succeed: bool,
        version: u64,
        calls: u32,
    }
    impl DurableWrite for FakeWal {
        fn append(&mut self, _owner: MId, _bytes: &[u8]) -> Result<DatasetVersion, WriteFailed> {
            self.calls += 1;
            if self.succeed {
                Ok(DatasetVersion(self.version))
            } else {
                Err(WriteFailed("wal fenced".into()))
            }
        }
    }

    fn owner(phase: KanbanColumn) -> FakeOwner {
        FakeOwner {
            id: 42,
            phase,
            cycle: 5,
        }
    }
    fn paired(to: KanbanColumn) -> KanbanMove {
        KanbanMove {
            mailbox: 42,
            from: KanbanColumn::Planning,
            to,
            witness_chain_position: 0,
            exec: ExecTarget::Elixir,
        }
    }

    #[test]
    fn no_successful_write_applies_no_step() {
        // The negative half of the ordering invariant: a failed write leaves the
        // owner's lifecycle UNTOUCHED.
        let mut o = owner(KanbanColumn::Planning);
        let mut w = FakeWal {
            succeed: false,
            version: 9,
            calls: 0,
        };
        let r = persist_then_step(
            &mut o,
            &mut w,
            Some(paired(KanbanColumn::CognitiveWork)),
            &[],
        );
        assert_eq!(
            r,
            Err(PersistError::Write(WriteFailed("wal fenced".into())))
        );
        assert_eq!(w.calls, 1, "the write WAS attempted");
        // Anti-vacuity: the owner did NOT advance — no write, no step.
        assert_eq!(
            o.phase(),
            KanbanColumn::Planning,
            "no step on a failed write"
        );
        assert_eq!(o.current_cycle(), 5, "cycle untouched");
    }

    #[test]
    fn successful_write_then_applies_the_paired_move_once() {
        let mut o = owner(KanbanColumn::Planning);
        let mut w = FakeWal {
            succeed: true,
            version: 43,
            calls: 0,
        };
        let step = persist_then_step(
            &mut o,
            &mut w,
            Some(paired(KanbanColumn::CognitiveWork)),
            &[],
        )
        .expect("write landed")
        .expect("a paired move was applied");
        assert_eq!(step.to, KanbanColumn::CognitiveWork, "the paired target");
        assert_eq!(
            o.phase(),
            KanbanColumn::CognitiveWork,
            "owner advanced once"
        );
        assert_eq!(o.current_cycle(), 6, "exactly one advance");
    }

    #[test]
    fn applies_the_paired_move_not_the_generic_successor() {
        // THE key falsifier. From `Evaluation`, the generic forward arc
        // (`next_phases().first()`) is `Commit`. But the thought cast a veto —
        // the free-won't `Evaluation → Prune`. The sink must apply the PAIRED
        // move (Prune), never the generic successor (Commit).
        assert_eq!(
            KanbanColumn::Evaluation.next_phases().first(),
            Some(&KanbanColumn::Commit),
            "precondition: generic successor is Commit",
        );
        let mut o = owner(KanbanColumn::Evaluation);
        let mut w = FakeWal {
            succeed: true,
            version: 44,
            calls: 0,
        };
        let step = persist_then_step(&mut o, &mut w, Some(paired(KanbanColumn::Prune)), &[])
            .expect("write landed")
            .expect("paired veto applied");
        assert_eq!(step.to, KanbanColumn::Prune, "the paired veto, not Commit");
        assert_eq!(o.phase(), KanbanColumn::Prune);
        assert_ne!(
            o.phase(),
            KanbanColumn::Commit,
            "the generic successor was NOT taken",
        );
    }

    #[test]
    fn a_landed_write_with_no_paired_move_is_a_durable_no_step() {
        // A write can land durably while carrying no lifecycle intent — durable,
        // but no step (non-vacuous: the write WAS attempted and succeeded).
        let mut o = owner(KanbanColumn::CognitiveWork);
        let mut w = FakeWal {
            succeed: true,
            version: 7,
            calls: 0,
        };
        let r = persist_then_step(&mut o, &mut w, None, &[]);
        assert_eq!(r, Ok(None), "durable write, no step");
        assert_eq!(w.calls, 1, "the write did land");
        assert_eq!(o.phase(), KanbanColumn::CognitiveWork, "phase unchanged");
    }

    #[test]
    fn an_illegal_paired_edge_is_surfaced_not_applied() {
        // The write lands, but the paired move is an illegal Rubicon edge from the
        // current phase (Planning → Evaluation skips CognitiveWork). Surfaced as
        // Illegal; the owner is NOT mutated (the checked airgap holds post-write).
        let mut o = owner(KanbanColumn::Planning);
        let mut w = FakeWal {
            succeed: true,
            version: 8,
            calls: 0,
        };
        let r = persist_then_step(&mut o, &mut w, Some(paired(KanbanColumn::Evaluation)), &[]);
        assert!(
            matches!(r, Err(PersistError::Illegal(_))),
            "illegal edge surfaced"
        );
        assert_eq!(
            o.phase(),
            KanbanColumn::Planning,
            "owner untouched on illegal edge"
        );
        assert_eq!(
            w.calls, 1,
            "the write still landed (durable) before the rejected step"
        );
    }
}
