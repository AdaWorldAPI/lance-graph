//! D-TGL-PUB step 3: the G10 × G11 publication lab, falsifiers L1–L6
//! (`.claude/plans/g10-g11-atomic-publication-lab-v1.md` §3). Lab only: no
//! production land, no `ThoughtCommitId` / `DurableRef` (EXO-148 stays
//! locked), no Lance. A branch revises one Codegraph source and carries DT64
//! route intents. It lands in the G10 order: prepare (Codegraph
//! `prepare_refresh` + `BindStore::check_route`), durable commit through the
//! lab `WalSink`, then an infallible in-memory publication. Recovery reads land
//! records from the durable image above a watermark, in canonical cycle order.

use std::fs;
use std::path::Path;

use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_dt64::{BindStore, ClassSchema, Extent, RouteVersion, SchemaRev};
use lance_graph_planner::persist_sink::{
    CommitError, CommitOutcome, CycleFrame, CycleId, DetachedCycleBatch, SweepSlot, WalSink,
};

use super::atomic::snapshot;
use super::lifecycle::{answers, dir, m_v1, m_v2, n_v1, n_v2, store};
use super::{Codegraph, PreparedRefresh};
use crate::lab_wal::LabWal;

const BASE: u32 = 0x0A1E_0001;
const C1: u32 = 0x0A1E_0002;
const C2: u32 = 0x0A1E_0003;
const P: u16 = 9;

fn run<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime")
        .block_on(f)
}

/// The DT64 side of W0: a base class and two derived ones, no routes yet.
fn binds() -> BindStore {
    let schema = || ClassSchema {
        schema_rev: SchemaRev(1),
        field_count: 4,
        extent: Extent::DR256_SQ_X64,
    };
    let mut b = BindStore::new();
    b.register(BASE, schema(), vec![]).expect("base");
    b.register_derived(BASE, C1, schema(), vec![]).expect("C1");
    b.register_derived(BASE, C2, schema(), vec![]).expect("C2");
    b
}

/// One land record: a source revision and route intents. No positions and
/// no `RouteVersion`: W0 assigns both when it publishes.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Record {
    krate: String,
    rows: String,
    routes: Vec<(u16, u32)>,
}

fn put(out: &mut Vec<u8>, b: &[u8]) {
    out.extend((b.len() as u32).to_le_bytes());
    out.extend(b);
}

fn take<'a>(b: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    let (head, tail) = (b.get(..n)?, b.get(n..)?);
    *b = tail;
    Some(head)
}

fn take_u32(b: &mut &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(take(b, 4)?.try_into().ok()?))
}

impl Record {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put(&mut out, self.krate.as_bytes());
        put(&mut out, self.rows.as_bytes());
        out.extend((self.routes.len() as u32).to_le_bytes());
        for &(p, c) in &self.routes {
            out.extend(p.to_le_bytes());
            out.extend(c.to_le_bytes());
        }
        out
    }

    fn decode(mut b: &[u8]) -> Option<Record> {
        let n = take_u32(&mut b)? as usize;
        let krate = String::from_utf8(take(&mut b, n)?.to_vec()).ok()?;
        let n = take_u32(&mut b)? as usize;
        let rows = String::from_utf8(take(&mut b, n)?.to_vec()).ok()?;
        let k = take_u32(&mut b)? as usize;
        let mut routes = Vec::with_capacity(k);
        for _ in 0..k {
            let p = u16::from_le_bytes(take(&mut b, 2)?.try_into().ok()?);
            routes.push((p, take_u32(&mut b)?));
        }
        b.is_empty().then_some(Record {
            krate,
            rows,
            routes,
        })
    }
}

/// A prepared land: the checked Codegraph change and the checked route
/// intents, plus the record to make durable.
struct Prepared {
    refresh: Option<PreparedRefresh>,
    routes: Vec<(u16, u32)>,
    record: Record,
}

/// The land record of `cycle`, read from the durable image. `scan_sealed`
/// carries no payload, so it is never the source.
fn record_of(wal: &LabWal, cycle: CycleId) -> Option<Record> {
    wal.image(cycle)
        .and_then(|img| img.get(&0))
        .and_then(|b| Record::decode(b))
}

/// W0: the Codegraph, the DT64 binding store, the durable head they reflect
/// and the watermark of the last land record applied.
struct W0 {
    graph: Codegraph,
    binds: BindStore,
    head: DatasetVersion,
    applied_through: Option<CycleId>,
}

impl W0 {
    fn open(base: &Path) -> W0 {
        W0 {
            graph: Codegraph::open(base).expect("open W0"),
            binds: binds(),
            head: DatasetVersion(0),
            applied_through: None,
        }
    }

    /// Prepare and check, changing nothing. (`&mut` only so the L1 disable
    /// can call `route`; the checks themselves read.)
    fn prepare(&mut self, record: Record) -> Prepared {
        let d = dir("land");
        fs::write(d.join(format!("{}.rows.tsv", record.krate)), &record.rows).expect("write");
        let refresh = self
            .graph
            .prepare_refresh(&d, &record.krate)
            .expect("codegraph prepare");
        for &(p, c) in &record.routes {
            self.binds.check_route(p, c).expect("route intent checks");
        }
        Prepared {
            refresh,
            routes: record.routes.clone(),
            record,
        }
    }

    /// The infallible publication. The writer is W0's sole mutator, so the
    /// Codegraph fence cannot be stale and checked routes cannot refuse.
    fn publish(&mut self, p: Prepared, cycle: CycleId, head: DatasetVersion) {
        if let Some(r) = p.refresh {
            self.graph
                .commit_refresh(r)
                .expect("sole mutator: not stale");
        }
        for (pos, class) in p.routes {
            self.binds.route(pos, class).expect("checked intent");
        }
        self.applied_through = Some(cycle);
        self.head = head;
    }

    /// The G10 order: prepare, durable commit, then publish. `crash` stops
    /// after the durable commit, before publication (boundary B4).
    fn land(
        &mut self,
        wal: &mut LabWal,
        cycle: u64,
        record: Record,
        crash: bool,
    ) -> Result<(), CommitError> {
        let p = self.prepare(record);
        let payload = p.record.encode();
        let out = durable(wal, cycle, self.head, payload)?;
        let head = match out {
            CommitOutcome::Committed { version, .. } => version,
            CommitOutcome::Reconciled { current_head, .. } => current_head,
            CommitOutcome::NoChange { head } => head,
        };
        if crash {
            return Ok(());
        }
        self.publish(p, CycleId(cycle), head);
        Ok(())
    }

    /// Recovery: every durable land record above the watermark, in canonical
    /// cycle order, re-prepared from the image and published.
    fn recover(&mut self, wal: &LabWal) {
        let mut frames = run(wal.timeline()).expect("timeline");
        frames.sort_by_key(|f| f.cycle);
        for f in frames {
            if self.applied_through.is_some_and(|w| f.cycle <= w) {
                continue;
            }
            let record = record_of(wal, f.cycle).expect("land record in the durable image");
            let p = self.prepare(record);
            self.publish(p, f.cycle, wal.head());
        }
    }

    /// The DT64 side, comparable: the version and `P`'s route at every version.
    fn routes(&self) -> (RouteVersion, Vec<Option<u32>>) {
        let v = self.binds.current_version();
        let history = (0..=v.0)
            .map(|i| self.binds.routed_at(RouteVersion(i), P).expect("routed_at"))
            .collect();
        (v, history)
    }
}

fn durable(
    wal: &mut LabWal,
    cycle: u64,
    base: DatasetVersion,
    payload: Vec<u8>,
) -> Result<CommitOutcome, CommitError> {
    let batch = DetachedCycleBatch::freeze(
        CycleFrame::new(CycleId(cycle), base),
        vec![SweepSlot {
            cycle: CycleId(cycle),
            stream_position: cycle,
            owner: 1,
            row: 0,
            paired_move: None,
            payload,
        }],
    );
    run(wal.commit_cycle(batch))
}

fn branch_m() -> Record {
    Record {
        krate: "m".into(),
        rows: m_v2(),
        routes: vec![(P, C1)],
    }
}

fn branch_n() -> Record {
    Record {
        krate: "n".into(),
        rows: n_v2(),
        routes: vec![(P, C2)],
    }
}

/// L1 + L5. Preparing a branch changes nothing in W0: Codegraph snapshot,
/// `RouteVersion`, `P`'s route. A discarded preparation leaves no trace.
/// Red if preparation applies routes eagerly (disable: `route` instead of
/// `check_route` in `prepare`).
#[test]
fn l1_l5_prepare_and_discard_leave_w0_and_routing_unchanged() {
    let base = store("w0", &m_v1(), &n_v1());
    let mut w = W0::open(&base);
    let (g, r) = (snapshot(&w.graph), w.routes());
    let claim = w
        .binds
        .claimed(C1, w.binds.current_version())
        .expect("claim");
    let p = w.prepare(branch_m());
    // Anti-vacuity: the discarded change is real.
    assert!(p.refresh.is_some() && !p.routes.is_empty());
    drop(p);
    assert_eq!(snapshot(&w.graph), g);
    assert_eq!(w.routes(), r);
    assert_eq!(
        w.binds
            .claimed(C1, w.binds.current_version())
            .expect("claim"),
        claim
    );
}

/// L2 + L5. A durable commit against a stale base is `Fenced`: nothing is
/// published, and the branch lands after re-preparing against the new head.
/// Red if publication runs before the durable commit (disable: publish
/// first, the order G10 forbids).
#[test]
fn l2_a_fenced_commit_publishes_nothing_and_the_branch_lands_after_reprepare() {
    let base = store("w0", &m_v1(), &n_v1());
    let mut w = W0::open(&base);
    let mut wal = LabWal::new(DatasetVersion(1)); // W0 believes head 0
    let (g, r) = (snapshot(&w.graph), w.routes());
    match w.land(&mut wal, 1, branch_m(), false) {
        Err(CommitError::Fenced { current_head }) => w.head = current_head,
        other => panic!("expected Fenced, got {other:?}"),
    }
    assert_eq!(snapshot(&w.graph), g, "nothing published");
    assert_eq!(w.routes(), r, "no route, no version");
    w.land(&mut wal, 1, branch_m(), false)
        .expect("lands after re-prepare");
    assert_eq!(
        w.binds.routed_at(w.binds.current_version(), P),
        Ok(Some(C1))
    );
}

/// L3 + L4. The commit is durable, then a crash before publication. Recovery
/// from the durable image restores exactly the confirmed state; a second
/// recovery changes nothing. Red if recovery reads `scan_sealed` (no
/// payload) instead of the image; red if the watermark is ignored (DT64
/// `route` is not idempotent: `RouteVersion` would tick again while the
/// Codegraph stays unchanged by its digest skip).
#[test]
fn l3_l4_recovery_restores_the_confirmed_state_once() {
    let base = store("w0", &m_v1(), &n_v1());
    let mut w = W0::open(&base);
    let mut wal = LabWal::new(DatasetVersion(0));
    let (g0, (v0, _)) = (snapshot(&w.graph), w.routes());
    w.land(&mut wal, 1, branch_m(), true).expect("durable");
    // B4: durable, not published.
    assert_eq!(snapshot(&w.graph), g0);
    assert_eq!(w.binds.current_version(), v0);

    w.recover(&wal);
    assert_ne!(snapshot(&w.graph), g0, "anti-vacuity: the graph moved");
    let r1 = w.routes();
    assert_eq!(r1.0, RouteVersion(v0.0 + 1), "exactly one tick");
    assert_eq!(w.binds.routed_at(r1.0, P), Ok(Some(C1)));
    let mut clean = Codegraph::open(&store("clean", &m_v2(), &n_v1())).expect("clean");
    assert_eq!(answers(&mut w.graph), answers(&mut clean));
    // Taken after the oracle comparison, whose queries fill the closure
    // caches; the second recovery must then change nothing at all.
    let g1 = snapshot(&w.graph);

    // L4: the watermark makes a second recovery a no-op.
    w.recover(&wal);
    assert_eq!(snapshot(&w.graph), g1);
    assert_eq!(w.routes(), r1);
}

/// L6. B5: W0 is lost and rebuilt from the durable base, then every land
/// record is replayed once in canonical cycle order, whatever the physical
/// order. The result equals the W0 that published both lands live. Two
/// records route `P` (C1 then C2), so order is observable. Red if recovery
/// trusts physical order (disable: drop the canonical sort).
#[test]
fn l6_a_rebuild_replays_every_record_once_in_canonical_order() {
    let base = store("w0", &m_v1(), &n_v1());
    let mut live = W0::open(&base);
    let mut wal = LabWal::new(DatasetVersion(0));
    live.land(&mut wal, 1, branch_m(), false).expect("land m");
    live.land(&mut wal, 2, branch_n(), false).expect("land n");
    assert_eq!(
        live.binds.routed_at(live.binds.current_version(), P),
        Ok(Some(C2))
    );

    wal.reverse_physical_order();
    let mut rebuilt = W0::open(&base);
    rebuilt.recover(&wal);
    assert_eq!(rebuilt.routes(), live.routes());
    assert_eq!(answers(&mut rebuilt.graph), answers(&mut live.graph));
    assert_eq!(rebuilt.head, live.head);
    assert_eq!(rebuilt.applied_through, Some(CycleId(2)));
}

/// The record codec round-trips and refuses trailing bytes, so a recovery
/// cannot silently read a truncated or padded record.
#[test]
fn a_land_record_round_trips_and_refuses_garbage() {
    let r = branch_m();
    let b = r.encode();
    assert_eq!(Record::decode(&b), Some(r));
    let mut padded = b.clone();
    padded.push(0);
    assert_eq!(Record::decode(&padded), None);
    assert_eq!(Record::decode(&b[..b.len() - 1]), None);
    assert_eq!(Record::decode(&[]), None);
}
