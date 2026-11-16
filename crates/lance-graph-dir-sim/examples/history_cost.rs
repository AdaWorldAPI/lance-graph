//! D-DIRSIM-DELTA-READ-1 cost probe: how read work grows with the number of
//! re-observations on one lineage.
//!
//! 100 users, one membership changed per re-read, N in {1, 16, 128, 1024}.
//! Run: `cargo run --release --example history_cost`.
//!
//! Counts are exact; latencies are warm medians on this machine and are
//! reported, not gated.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Instant;

use lance_graph_dir_sim::image::{RuleRegistry, StoredImage};
use lance_graph_dir_sim::log::{replay, LogState, RecordKind};
use lance_graph_dir_sim::{Observation, ObservedNode, VersionStore};
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::VersionId;

struct Counting;
thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|a| a.set(a.get() + 1));
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;
fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const EMPLOYEES: Guid128 = Guid128([0xE0; 16]);
const EXCHANGE: Guid128 = Guid128([0xEC; 16]);
const USERS: u8 = 100;

fn user(u: u8) -> Guid128 {
    let mut b = [0u8; 16];
    b[0] = 1;
    b[1] = u;
    Guid128(b)
}

/// Read `i`: every user in Employees; Exchange holds the users whose bit is
/// set in `exchange`.
fn read(exchange: &[bool]) -> Observation {
    let mut o = Observation {
        scope: SCOPE,
        nodes: vec![
            (EMPLOYEES, ObservedNode::group()),
            (EXCHANGE, ObservedNode::group()),
        ],
        members: Vec::new(),
    };
    for u in 0..USERS {
        let mail = format!("user{u}@example.test");
        o.nodes.push((user(u), ObservedNode::user(&mail, &mail)));
        o.members.push((user(u), EMPLOYEES));
        if exchange[usize::from(u)] {
            o.members.push((user(u), EXCHANGE));
        }
    }
    o
}

/// Median of `f`'s wall time over `runs`, in ns, and its allocations per call.
fn measure(runs: usize, mut f: impl FnMut()) -> (u128, u64) {
    f();
    let a0 = allocs();
    f();
    let per_call = allocs() - a0;
    let mut t: Vec<u128> = (0..runs)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_nanos()
        })
        .collect();
    t.sort_unstable();
    (t[runs / 2], per_call)
}

fn main() {
    println!(
        "{:>5} | {:>8} {:>8} | {:>9} {:>9} | {:>10} {:>6} | {:>10} {:>6} | {:>10} {:>6} | {:>10} {:>6} {:>5}",
        "N",
        "versions",
        "deltas",
        "log B",
        "B/reread",
        "range ns",
        "alloc",
        "member ns",
        "alloc",
        "diff ns",
        "alloc",
        "replay ns",
        "recs",
        "roots"
    );
    for n in [1usize, 16, 128, 1024] {
        let mut st = VersionStore::new();
        let mut log = LogState::new();
        let mut records = Vec::new();
        let mut exchange = vec![false; usize::from(USERS)];
        let save = |st: &VersionStore, log: &mut LogState, records: &mut Vec<_>| {
            if let Some(b) = log.batch(&StoredImage::from(st.image())).unwrap() {
                log.committed(&b).unwrap();
                records.extend(b.records);
            }
        };
        st.observe("ad", 0, read(&exchange)).unwrap();
        save(&st, &mut log, &mut records);
        let first_bytes: usize = records.iter().map(|r| r.body.len()).sum();
        let mut ids: Vec<VersionId> = vec![st.tag("observed").unwrap()];
        for i in 0..n {
            let u = i % usize::from(USERS);
            exchange[u] = !exchange[u];
            ids.push(st.reobserve("ad", i as i64 + 1, read(&exchange)).unwrap());
            save(&st, &mut log, &mut records);
        }
        let last = *ids.last().unwrap();
        let prev = ids[ids.len() - 2];
        let image = st.image();
        let versions = image.versions.len();
        let deltas: usize = image.versions.iter().map(|v| v.delta.len()).sum();
        let bytes: usize = records.iter().map(|r| r.body.len()).sum();
        let per_reread = (bytes - first_bytes) / n;
        let roots = records
            .iter()
            .filter(|r| r.kind == RecordKind::Root as u8)
            .count();

        // Historical change query: the stored delta of the last re-read.
        let (range_ns, range_alloc) = measure(201, || {
            let c: usize = st.history(prev, last).unwrap().map(|v| v.delta.len()).sum();
            assert_eq!(c, 1);
        });
        // Current-state, selected user: fold the lineage, then one lookup.
        let probe = user(((n - 1) % usize::from(USERS)) as u8);
        let (member_ns, member_alloc) = measure(51, || {
            let v = st.view(last).unwrap();
            std::hint::black_box(v.is_member(&probe, &EXCHANGE));
        });
        // Net change between the last two reads, through diff.
        let (diff_ns, diff_alloc) = measure(51, || {
            assert_eq!(st.diff(prev, last).unwrap().len(), 1);
        });
        // Cold recovery: replay the whole log and restore.
        let (replay_ns, _) = measure(11, || {
            let (img, _) = replay(records.clone()).unwrap();
            std::hint::black_box(VersionStore::restore_stored(img, &RuleRegistry::new()).unwrap());
        });
        println!(
            "{:>5} | {:>8} {:>8} | {:>9} {:>9} | {:>10} {:>6} | {:>10} {:>6} | {:>10} {:>6} | {:>10} {:>6} {:>5}",
            n,
            versions,
            deltas,
            bytes,
            per_reread,
            range_ns,
            range_alloc,
            member_ns,
            member_alloc,
            diff_ns,
            diff_alloc,
            replay_ns,
            records.len(),
            roots
        );
    }
}
