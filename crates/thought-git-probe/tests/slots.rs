//! D-TGB-1: the slot pool under its two starting policies.

use thought_git_probe::slots::{Eviction, LIVE_SLOTS, SlotPool, Stale};

/// Fill the pool; return the handles in branch order.
fn full(policy: Eviction) -> (SlotPool, Vec<thought_git_probe::slots::Slot>) {
    let mut p = SlotPool::new(policy);
    let mut handles = Vec::new();
    for t in 0..LIVE_SLOTS as u32 {
        let (s, e) = p.branch(t);
        assert!(e.is_none(), "evicted before the pool was full");
        handles.push(s);
    }
    assert_eq!(p.live(), LIVE_SLOTS);
    (p, handles)
}

/// The 64th branch evicts; FIFO evicts the oldest thought.
#[test]
fn fifo_evicts_the_oldest() {
    let (mut p, h) = full(Eviction::Fifo);
    for s in &h[..10] {
        p.touch(*s).unwrap();
    }
    let (_, e) = p.branch(100);
    assert_eq!(e.expect("full pool evicts").thought, 0);
    assert_eq!(p.live(), LIVE_SLOTS);
}

/// LeastUsed keeps the busy old thoughts and evicts the first idle one.
/// Same workload as above, different victim: the policies are not the same.
#[test]
fn least_used_evicts_the_idlest() {
    let (mut p, h) = full(Eviction::LeastUsed);
    for s in &h[..10] {
        p.touch(*s).unwrap();
    }
    let (_, e) = p.branch(100);
    let e = e.expect("full pool evicts");
    assert_eq!(e.thought, 10, "oldest thought with zero uses");
    assert_eq!(e.uses, 0);
}

/// An evicted thought's old handle is refused, and its slot now holds the
/// new thought.
#[test]
fn an_evicted_handle_is_stale() {
    let (mut p, h) = full(Eviction::Fifo);
    let (new, e) = p.branch(100);
    let old = e.expect("evicts").slot;
    assert_eq!(old, h[0]);
    assert_eq!(new.index, old.index);
    assert_eq!(p.thought(old), Err(Stale(old)));
    assert_eq!(p.touch(old), Err(Stale(old)));
    assert_eq!(p.thought(new), Ok(100));
}

/// Freeing (squash, discard, land) makes room without evicting anyone.
#[test]
fn a_freed_slot_is_reused_without_eviction() {
    let (mut p, h) = full(Eviction::LeastUsed);
    p.free(h[30]).unwrap();
    let (s, e) = p.branch(100);
    assert!(e.is_none());
    assert_eq!(s.index, h[30].index);
    assert_eq!(p.thought(h[30]), Err(Stale(h[30])));
}
