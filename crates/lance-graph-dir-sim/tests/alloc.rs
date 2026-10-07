//! Allocation instrumentation: one simulated mutation of each kind
//! (membership add / remove, attribute, node create / delete) must cost work
//! proportional to the delta, not to the directory. Detects an accidentally
//! population-copying or row-exploding version path.
//!
//! Scales: 1,000, 16,384 and 65,536 users — the last is the population
//! bound ([`MAX_USERS`]). Larger directories are no longer representable.
//!
//! Own test binary (the counting allocator is process-global).

use lance_graph_dir_sim::*;
use ogar_dir_core::Guid128;
use ogar_dir_sim::{
    Attribute, Change, EvidenceRef, NodeKind, NodeState, RuleId, ValueId, VersionId,
};
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

const LONER: u32 = u32::MAX - 2;

/// A directory of `users` users — all but one in `employees`, plus a user
/// in no group — and an empty `exchange` group.
fn directory(users: usize) -> (VersionStore, VersionId) {
    let (employees, exchange) = (guid(u32::MAX - 1), guid(u32::MAX));
    let mut obs = Observation::default();
    for i in 0..(users - 1) as u32 {
        obs.nodes.push((
            guid(i),
            ObservedNode::user(&format!("u{i}@example.test"), &format!("u{i}@example.test")),
        ));
        obs.members.push((guid(i), employees));
    }
    obs.nodes.push((
        guid(LONER),
        ObservedNode::user("loner@example.test", "loner@example.test"),
    ));
    obs.nodes.push((employees, ObservedNode::group()));
    obs.nodes.push((exchange, ObservedNode::group()));
    let mut st = VersionStore::new();
    let g0 = st.observe("lab", 0, obs).unwrap();
    (st, g0)
}

struct Propose(Vec<Change>);
impl Rule for Propose {
    fn id(&self) -> RuleId {
        RuleId {
            name: "Propose",
            version: 1,
        }
    }
    fn propose(&self, _: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
        self.0.clone()
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    AddMembership,
    RemoveMembership,
    SetAttribute,
    CreateNode,
    DeleteNode,
}

/// The change for `op`. Its values are interned here, at ingress, outside
/// the measured window — as a real caller would before simulating.
fn change(op: Op, st: &mut VersionStore, g0: VersionId) -> Change {
    let (employees, exchange) = (guid(u32::MAX - 1), guid(u32::MAX));
    let mut v = |s: &str| -> Option<ValueId> { Some(st.intern(s)) };
    let (u7, renamed, new) = (
        v("u7@example.test"),
        v("renamed@example.test"),
        v("new@example.test"),
    );
    match op {
        Op::AddMembership => Change::AddMembership {
            user: guid(7),
            group: exchange,
        },
        Op::RemoveMembership => Change::RemoveMembership {
            user: guid(7),
            group: employees,
        },
        Op::SetAttribute => Change::SetAttribute {
            node: guid(7),
            attribute: Attribute::PrimarySmtp,
            from: u7,
            to: renamed,
        },
        Op::CreateNode => Change::CreateNode {
            node: guid(u32::MAX - 3),
            state: NodeState {
                kind: NodeKind::User,
                active: Some(true),
                upn: new,
                primary_smtp: new,
                dn: None,
            },
        },
        Op::DeleteNode => Change::DeleteNode {
            node: guid(LONER),
            state: st.view(g0).unwrap().node_state(&guid(LONER)).unwrap(),
        },
    }
}

/// Bytes allocated by `simulate` and by `diff` for one change of kind `op`
/// in a directory of `n` users (a create starts one below, so the version
/// it makes holds exactly `n`). The change itself is built outside the
/// measured window.
fn measure(op: Op, n: usize) -> (usize, usize) {
    let users = if matches!(op, Op::CreateNode) {
        n - 1
    } else {
        n
    };
    let (mut st, g0) = directory(users);
    let rule = Propose(vec![change(op, &mut st, g0)]);
    let ev = vec![EvidenceRef("REQ-1".into())];

    let before = BYTES.load(Ordering::Relaxed);
    let g1 = st.simulate(g0, &rule, &ev).unwrap();
    let simulate = BYTES.load(Ordering::Relaxed) - before;

    let before = BYTES.load(Ordering::Relaxed);
    let d = st.diff(g0, g1).unwrap();
    let diff = BYTES.load(Ordering::Relaxed) - before;
    assert_eq!(d.len(), 1, "{op:?}");
    (simulate, diff)
}

#[test]
fn one_mutation_of_each_kind_is_delta_sized_not_population_sized() {
    for op in [
        Op::AddMembership,
        Op::RemoveMembership,
        Op::SetAttribute,
        Op::CreateNode,
        Op::DeleteNode,
    ] {
        let (s1, d1) = measure(op, 1_000);
        let (s16, d16) = measure(op, 16_384);
        let (s64, d64) = measure(op, MAX_USERS);
        eprintln!(
            "{op:?}: simulate {s1} / {s16} / {s64} B, diff {d1} / {d16} / {d64} B \
             @ 1k / 16,384 / 65,536 users"
        );
        // A copy of one u32 lane at 65,536 users is 256 KB, a node bitmap
        // 8 KB, the u16 ordinal space itself 128 KB. Delta-sized work stays
        // below all of them.
        assert!(
            s64 < 8_192 && d64 < 8_192,
            "{op:?}: {s64} / {d64} B @65,536"
        );
        // Quadrupling a directory that is already a full tile changes nothing.
        assert!(
            s64 <= s16 + 256 && d64 <= d16 + 256,
            "{op:?}: allocation grew with population past 16,384"
        );
        // From 1k to 16,384 only the delete guard may grow, and only by its
        // `Count` program's scratch: one tile per slot, a tile capped at
        // `TILE_WORDS` (16,384 rows), so constant above that size.
        let tile_growth = if matches!(op, Op::DeleteNode) {
            4_096
        } else {
            256
        };
        assert!(
            s16 <= s1 + tile_growth && d16 <= d1 + tile_growth,
            "{op:?}: allocation grew with population from 1k to 16,384"
        );
    }
}
