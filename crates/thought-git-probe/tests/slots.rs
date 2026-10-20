//! D-TGB-1: the slot pool under its two starting policies, with squashing
//! along dependency chains before anything is evicted.

use thought_git_probe::slots::{Eviction, LIVE_SLOTS, Pressure, Slot, SlotPool, Stale};

/// Fill the pool with root thoughts (branched from main); `parents[t]` names
/// the earlier thought `t` branches from instead.
fn full(policy: Eviction, parents: &[(u32, u32)]) -> (SlotPool, Vec<Slot>) {
    let mut p = SlotPool::new(policy);
    let mut h: Vec<Slot> = Vec::new();
    for t in 0..LIVE_SLOTS as u32 {
        let from = parents
            .iter()
            .find(|&&(c, _)| c == t)
            .map(|&(_, parent)| h[parent as usize]);
        let (s, e) = p.branch(t, from).expect("live parent");
        assert!(e.is_none(), "pressure before the pool was full");
        h.push(s);
    }
    (p, h)
}

/// Without chains, FIFO evicts the oldest.
#[test]
fn fifo_evicts_the_oldest_root() {
    let (mut p, h) = full(Eviction::Fifo, &[]);
    for s in &h[..10] {
        p.touch(*s).unwrap();
    }
    let (_, e) = p.branch(100, None).unwrap();
    assert!(matches!(e, Some(Pressure::Evicted { thought: 0, .. })));
    assert_eq!(p.live(), LIVE_SLOTS);
}

/// Without chains, LeastUsed evicts the idlest; on the same workload FIFO
/// picks a different victim.
#[test]
fn least_used_evicts_the_idlest_root() {
    let (mut p, h) = full(Eviction::LeastUsed, &[]);
    for s in &h[..10] {
        p.touch(*s).unwrap();
    }
    let (_, e) = p.branch(100, None).unwrap();
    assert!(matches!(
        e,
        Some(Pressure::Evicted {
            thought: 10,
            uses: 0,
            ..
        })
    ));
}

/// The least-used thought has a live parent: it is squashed into it, not
/// evicted, and its handle goes stale.
#[test]
fn a_least_used_thought_in_a_chain_is_squashed_into_its_parent() {
    let (mut p, h) = full(Eviction::LeastUsed, &[(20, 5)]);
    for (t, s) in h.iter().enumerate() {
        if t != 20 {
            p.touch(*s).unwrap();
        }
    }
    let (_, e) = p.branch(100, None).unwrap();
    assert_eq!(
        e,
        Some(Pressure::Squashed {
            thought: 20,
            from: h[20],
            into: h[5],
        })
    );
    assert_eq!(p.thought(h[20]), Err(Stale(h[20])));
    assert_eq!(p.thought(h[5]), Ok(5));
}

/// The FIFO candidate has a live child but no parent: the child is squashed
/// into it, so the oldest thought survives and nothing is lost.
#[test]
fn a_candidate_with_a_child_absorbs_it() {
    let (mut p, h) = full(Eviction::Fifo, &[(1, 0)]);
    let (_, e) = p.branch(100, None).unwrap();
    assert_eq!(
        e,
        Some(Pressure::Squashed {
            thought: 1,
            from: h[1],
            into: h[0],
        })
    );
    assert_eq!(p.thought(h[0]), Ok(0));
}

/// Branching from the thought the policy would pick must not remove it.
#[test]
fn branching_from_the_candidate_keeps_it() {
    let (mut p, h) = full(Eviction::Fifo, &[]);
    let (s, e) = p.branch(100, Some(h[0])).unwrap();
    assert!(matches!(e, Some(Pressure::Evicted { thought: 1, .. })));
    assert_eq!(p.thought(h[0]), Ok(0));
    assert_eq!(p.thought(s), Ok(100));
}

/// Freeing makes room without pressure, and the freed handle is stale.
#[test]
fn a_freed_slot_is_reused_without_pressure() {
    let (mut p, h) = full(Eviction::LeastUsed, &[]);
    p.free(h[30]).unwrap();
    let (s, e) = p.branch(100, None).unwrap();
    assert!(e.is_none());
    assert_eq!(s.index, h[30].index);
    assert_eq!(p.thought(h[30]), Err(Stale(h[30])));
}
