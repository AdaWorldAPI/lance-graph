//! Allocation instrumentation: a one-edge simulated mutation must cost work
//! proportional to the delta, not to the directory. Detects an accidentally
//! population-copying or row-exploding version path.
//!
//! Own test binary (the counting allocator is process-global).

use lance_graph_dir_sim::*;
use ogar_dir_core::Guid128;
use ogar_dir_sim::{EvidenceRef, RuleId};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        BYTES.fetch_add(l.size(), Ordering::Relaxed);
        // SAFETY: forwards to the system allocator with the caller's layout.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` was returned by `System.alloc` with this layout.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

fn guid(i: u32) -> Guid128 {
    let mut b = [0u8; 16];
    b[..4].copy_from_slice(&i.to_be_bytes());
    b[15] = 1;
    Guid128(b)
}

/// Bytes allocated by `simulate` of one membership for a directory of `n` users.
fn one_edge(n: u32) -> (usize, usize) {
    let (employees, exchange) = (guid(u32::MAX - 1), guid(u32::MAX));
    let mut obs = Observation::default();
    for i in 0..n {
        obs.nodes.push((
            guid(i),
            ObservedNode::user(&format!("u{i}@example.test"), &format!("u{i}@example.test")),
        ));
        obs.members.push((guid(i), employees));
    }
    obs.nodes.push((employees, ObservedNode::group()));
    obs.nodes.push((exchange, ObservedNode::group()));
    let mut st = VersionStore::new();
    let g0 = st.observe("lab", 0, obs).unwrap();
    let rule = GrantGroup {
        rule: RuleId {
            name: "ExchangeAccess",
            version: 1,
        },
        group: exchange,
        to: vec![guid(7)],
    };
    let ev = vec![EvidenceRef("REQ-1".into())];

    let before = BYTES.load(Ordering::Relaxed);
    let g1 = st.simulate(g0, &rule, &ev).unwrap();
    let simulate = BYTES.load(Ordering::Relaxed) - before;

    let before = BYTES.load(Ordering::Relaxed);
    let d = st.diff(g0, g1).unwrap();
    let diff = BYTES.load(Ordering::Relaxed) - before;
    assert_eq!(d.len(), 1);
    (simulate, diff)
}

#[test]
fn one_edge_mutation_is_delta_sized_not_population_sized() {
    let (s_small, d_small) = one_edge(1_000);
    let (s_large, d_large) = one_edge(100_000);
    eprintln!(
        "simulate: {s_small} B @1k users, {s_large} B @100k users; diff: {d_small} B / {d_large} B"
    );
    // A copy of even one u32 lane at 100k users is 400 KB; a node-bitmap is
    // 12.5 KB. Delta-sized work stays far below both and does not grow 100x.
    assert!(
        s_large < 4_096,
        "simulate allocated {s_large} B at 100k users"
    );
    assert!(d_large < 4_096, "diff allocated {d_large} B at 100k users");
    assert!(
        s_large <= s_small + 256 && d_large <= d_small + 256,
        "allocation grew with population"
    );
}
