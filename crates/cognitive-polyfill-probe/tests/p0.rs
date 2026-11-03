//! D-COGNITIVE-POLYFILL P0 falsifier. Numbering follows the brief's §5 list.
//!
//! The oracle is an independent nested loop over the borrowed fact slice. It
//! counts and tests existence; it never builds rows.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use cognitive_polyfill_probe::*;
use lance_graph_dt64::dr::{
    Focus, FrameRevisions, FrameStore, GuardRefusal, Guards, MapRead, MapRefusal, Scope,
};
use lance_graph_mask_risc::Value;

// --- allocation counter (check 12) -----------------------------------------

// Per thread, so parallel tests do not count into each other.
thread_local! { static BYTES: Cell<usize> = const { Cell::new(0) }; }
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = BYTES.try_with(|b| b.set(b.get() + l.size()));
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}
#[global_allocator]
static A: Counting = Counting;

// --- fixture ---------------------------------------------------------------

const R: u64 = 1000;
const S: u64 = 1001; // another relation name
const PRODUCT: u64 = 2000; // Narsese `(*, …)` functor
const A_: u64 = 10;
const B: u64 = 11;
const C: u64 = 12;
const D: u64 = 13;
const E: u64 = 14;
const Z1: u64 = 90;
const Z2: u64 = 91;
const IDS: [u64; 5] = [A_, B, C, D, E];
const FACTS: [(u64, u64); 4] = [(A_, B), (B, C), (C, D), (A_, E)];

fn hop(src: u64, dst: u64) -> Hop2 {
    Hop2 { rel: R, src, dst }
}

/// Independent oracle: Set (distinct witnesses) and Bag (derivations).
fn oracle(facts: &[(u64, u64)], src: u64, dst: u64, allowed: &dyn Fn(u64) -> bool) -> (u64, u64) {
    let mut set = 0u64;
    let mut bag = 0u64;
    for &y in &IDS {
        if !allowed(y) {
            continue;
        }
        let left = facts.iter().filter(|&&f| f == (src, y)).count() as u64;
        let right = facts.iter().filter(|&&f| f == (y, dst)).count() as u64;
        if left > 0 && right > 0 {
            set += 1;
        }
        bag += left * right;
    }
    (set, bag)
}

fn world(facts: &[(u64, u64)]) -> (MapTable, Frame, Relation) {
    let (t, f) = tile(1, &IDS);
    let r = Relation::ingest(R, &t, &f, facts).unwrap();
    (t, f, r)
}

use lance_graph_dt64::dr::{Frame, MapTable};

/// A selection in the relation tile's own coordinates, by global ids.
fn sel_in_rel(t: &MapTable, f: &Frame, ids: &[u64]) -> Routed {
    let m = t.map(f, TILE).unwrap();
    let mut mask = 0u64;
    for &g in ids {
        mask |= 1 << m.local(g).unwrap();
    }
    Routed { mask, dropped: 0 }
}

fn exists(facts: &[(u64, u64)], src: u64, dst: u64, w: &[u64]) -> bool {
    let (t, f, r) = world(facts);
    let mut p = Prepared::prepare(hop(src, dst), &r, &t, &f, Fold::Any).unwrap();
    p.run(&r, &sel_in_rel(&t, &f, w)) == Value::Bool(true)
}

// --- 1: equivalent answers on the common subset ----------------------------

/// MeTTa `(, (R A $y) (R $y C))`, MORK `(exec P (, (R A $y) (R $y C)) …)` and
/// the Atomese typed projection all hand-lower to this one pattern.
fn chain(src: u64, dst: u64) -> Pattern {
    Pattern {
        atoms: vec![
            Atom {
                rel: R,
                args: vec![Term::Ground(src), Term::Var(0)],
            },
            Atom {
                rel: R,
                args: vec![Term::Var(0), Term::Ground(dst)],
            },
        ],
    }
}

#[test]
fn c01_common_subset_agrees_with_oracle_exhaustively() {
    let subsets: [&[u64]; 5] = [&IDS, &[B], &[E], &[], &[B, E]];
    let mut checked = 0;
    for &src in &IDS {
        for &dst in &IDS {
            let h = lower(&chain(src, dst)).unwrap();
            assert_eq!(h, hop(src, dst));
            for w in subsets {
                let (t, f, r) = world(&FACTS);
                let sel = sel_in_rel(&t, &f, w);
                let (set, _) = oracle(&FACTS, src, dst, &|y| w.contains(&y));
                let mut any = Prepared::prepare(h, &r, &t, &f, Fold::Any).unwrap();
                let mut cnt = Prepared::prepare(h, &r, &t, &f, Fold::Count).unwrap();
                assert_eq!(
                    any.run(&r, &sel),
                    Value::Bool(set > 0),
                    "{src}->?->{dst} W={w:?}"
                );
                assert_eq!(cnt.run(&r, &sel), Value::Count(set as usize));
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 125);
    // anti-vacuity: the fixture has both answers
    assert!(exists(&FACTS, A_, C, &IDS));
    assert!(!exists(&FACTS, A_, D, &IDS));
}

// --- 2/3: support edge vs unrelated edge -----------------------------------

#[test]
fn c02_deleting_the_supporting_edge_flips_the_answer() {
    assert!(exists(&FACTS, A_, C, &IDS));
    let without: Vec<_> = FACTS.iter().copied().filter(|&e| e != (B, C)).collect();
    assert!(!exists(&without, A_, C, &IDS));
}

#[test]
fn c03_deleting_an_unrelated_edge_changes_nothing() {
    let without: Vec<_> = FACTS.iter().copied().filter(|&e| e != (C, D)).collect();
    for &(s, d) in &[(A_, C), (A_, D), (B, D)] {
        let before = exists(&FACTS, s, d, &IDS);
        let after = exists(&without, s, d, &IDS);
        if (s, d) == (B, D) {
            // B->C->D uses the deleted edge: the control that it is not inert
            assert!(before && !after);
        } else {
            assert_eq!(before, after, "{s}->?->{d}");
        }
    }
}

// --- 4/5: two selections, one global B at different local positions --------

#[test]
fn c04_two_selections_route_to_the_same_relation_coordinates() {
    let (t, f, r) = world(&FACTS); // B is local 1 here
    let (s1t, s1f) = tile(2, &[Z1, Z2, B]); // B is local 2
    let (s2t, s2f) = tile(3, &[B, Z1]); // B is local 0
    let s1 = Selection { mask: 1 << 2 };
    let s2 = Selection { mask: 1 << 0 };
    assert_ne!(s1.mask, s2.mask, "same identity, different local position");

    let r1 = s1.route(&s1t, &s1f, &t, &f).unwrap();
    let r2 = s2.route(&s2t, &s2f, &t, &f).unwrap();
    assert_eq!(r1.mask, r2.mask);
    assert_eq!(r1.mask, 1 << 1);

    let mut p = Prepared::prepare(hop(A_, C), &r, &t, &f, Fold::Any).unwrap();
    assert_eq!(p.run(&r, &r1), Value::Bool(true));
    assert_eq!(p.run(&r, &r2), Value::Bool(true));

    // Joining on raw local positions is a different, wrong answer: s1's bit 2
    // is C in the relation tile, not B.
    let raw = Routed {
        mask: s1.mask,
        dropped: 0,
    };
    assert_eq!(p.run(&r, &raw), Value::Bool(false));

    // A selected identity the relation tile does not hold is reported.
    let s3 = Selection {
        mask: (1 << 0) | (1 << 2),
    }; // Z1 and B
    assert_eq!(s3.route(&s1t, &s1f, &t, &f).unwrap().dropped, 1);
}

#[test]
fn c05_dr_round_trip() {
    let (t, f) = tile(1, &IDS);
    let m = t.map(&f, TILE).unwrap();
    for &g in &IDS {
        assert_eq!(m.global(m.local(g).unwrap()), Some(g));
    }
    assert_eq!(m.local(Z1), Err(MapRefusal::Unmapped));
}

// --- 6: wrong frame --------------------------------------------------------

#[test]
fn c06_foreign_frame_refuses_everywhere() {
    let (t, f, r) = world(&FACTS);
    let (_other, foreign) = tile(1, &IDS); // same content, another table
    assert!(matches!(
        Relation::ingest(R, &t, &foreign, &FACTS),
        Err(NativeRefusal::Read(MapRead::Stale(_)))
    ));
    assert!(matches!(
        Prepared::prepare(hop(A_, C), &r, &t, &foreign, Fold::Any),
        Err(NativeRefusal::Read(MapRead::Stale(_)))
    ));
    let (st, sf) = tile(2, &[B]);
    let s = Selection { mask: 1 };
    assert!(matches!(
        s.route(&st, &foreign, &t, &f),
        Err(NativeRefusal::SelectionRead(MapRead::Stale(_)))
    ));
    assert!(
        s.route(&st, &sf, &t, &f).is_ok(),
        "control: the right frames route"
    );
}

// --- 7: wrong relation / class, wrong scope --------------------------------

#[test]
fn c07_wrong_relation_and_wrong_scope_refuse() {
    let (t, f, r) = world(&FACTS);
    assert_eq!(
        Prepared::prepare(
            Hop2 {
                rel: S,
                src: A_,
                dst: C
            },
            &r,
            &t,
            &f,
            Fold::Any
        )
        .err(),
        Some(NativeRefusal::WrongRelation { bound: R, asked: S })
    );
    // Scope is dt64's own guard. NOTE: `Guards::check` takes a `FrameStore`,
    // and `MapTable` keeps its store private, so a MapTable frame cannot be
    // guard-checked: this guard runs on its own store (finding W-2).
    let store = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 0,
        coordinates: 0,
    });
    let g = Guards {
        frame: store.seal(),
        focus: Focus { revision: 0 },
        scope: Scope { id: 7 },
    };
    assert_eq!(
        g.check(&store, (Focus { revision: 0 }, Scope { id: 7 })),
        Ok(())
    );
    assert_eq!(
        g.check(&store, (Focus { revision: 0 }, Scope { id: 8 })),
        Err(GuardRefusal::ScopeChanged)
    );
}

// --- 8: Set versus Bag ------------------------------------------------------

#[test]
fn c08_duplicate_derivations_split_set_from_bag() {
    let facts = [(A_, B), (A_, B), (B, C), (A_, E), (E, C)];
    let (t, f, r) = world(&facts);
    let all = sel_in_rel(&t, &f, &IDS);
    let (set, bag) = oracle(&facts, A_, C, &|_| true);
    assert_eq!(
        (set, bag),
        (2, 3),
        "fixture: two witnesses, three derivations"
    );

    let mut cnt = Prepared::prepare(hop(A_, C), &r, &t, &f, Fold::Count).unwrap();
    assert_eq!(cnt.run(&r, &all), Value::Count(set as usize));

    let mut keep = Prepared::prepare(hop(A_, C), &r, &t, &f, Fold::Keep).unwrap();
    keep.run(&r, &all);
    let kept = keep.kept();
    assert_eq!(kept.count_ones() as u64, set);
    assert_eq!(r.bag_derivations(keep.src, keep.dst, kept), bag);
    assert_ne!(set, bag, "a Set fold cannot answer the Bag question");
}

// --- 9: NAL truth is not a Boolean edge -------------------------------------

/// A NAL-shaped belief `<s --> p>` with frequency and confidence codes.
#[derive(Clone, Copy)]
struct Belief {
    s: u64,
    p: u64,
    f: u8,
    c: u8,
}

#[test]
fn c09_a_belief_is_not_an_edge_without_a_projection_policy() {
    // Strong NEGATIVE evidence that B --> C: f = 0, c = 0.9.
    let beliefs = [
        Belief {
            s: A_,
            p: B,
            f: 255,
            c: 230,
        },
        Belief {
            s: B,
            p: C,
            f: 0,
            c: 230,
        },
    ];
    // Projection 1: "a belief row exists" -> edge.
    let existence: Vec<_> = beliefs.iter().map(|b| (b.s, b.p)).collect();
    // Projection 2: "f >= 0.5" -> edge.
    let thresholded: Vec<_> = beliefs
        .iter()
        .filter(|b| b.f >= 128)
        .map(|b| (b.s, b.p))
        .collect();
    let by_existence = exists(&existence, A_, C, &IDS);
    let by_truth = exists(&thresholded, A_, C, &IDS);
    assert!(
        by_existence && !by_truth,
        "the two projections disagree on the same beliefs"
    );
    // A revision that only raises confidence changes no Boolean edge under
    // either projection: the relational substrate cannot see revision.
    let revised = [
        beliefs[0],
        Belief {
            c: 250,
            ..beliefs[1]
        },
    ];
    assert_ne!(
        revised[1].c, beliefs[1].c,
        "anti-vacuity: the revision changed the truth value"
    );
    let thresholded2: Vec<_> = revised
        .iter()
        .filter(|b| b.f >= 128)
        .map(|b| (b.s, b.p))
        .collect();
    assert_eq!(thresholded, thresholded2);
}

// --- 10: unsupported shapes refuse -----------------------------------------

#[test]
fn c10_unsupported_patterns_refuse_with_a_reason() {
    let v = |i| Term::Var(i);
    let g = Term::Ground;
    let p = |atoms: Vec<Atom>| Pattern { atoms };
    let at = |rel, args| Atom { rel, args };
    // free endpoint: binding enumeration
    assert_eq!(
        lower(&p(vec![at(R, vec![v(1), v(0)]), at(R, vec![v(0), g(C)])])),
        Err(Refusal::FreeEndpoint)
    );
    // repeated variable: R($x, $x)
    assert_eq!(
        lower(&p(vec![at(R, vec![v(0), v(0)]), at(R, vec![v(0), g(C)])])),
        Err(Refusal::RepeatedVariable)
    );
    // structural: f(X, g(X)) / Narsese product <(*, A, $y) --> R>
    let nested = Term::Nested {
        functor: PRODUCT,
        args: vec![g(A_), v(0)],
    };
    assert_eq!(
        lower(&p(vec![at(R, vec![nested, g(C)]), at(R, vec![v(0), g(C)])])),
        Err(Refusal::NestedTerm)
    );
    // two relations
    assert_eq!(
        lower(&p(vec![at(R, vec![g(A_), v(0)]), at(S, vec![v(0), g(C)])])),
        Err(Refusal::TwoRelations)
    );
    // one atom
    assert_eq!(
        lower(&p(vec![at(R, vec![g(A_), v(0)])])),
        Err(Refusal::Arity)
    );
    // not a chain: R(A,$y) R(C,$y)
    assert_eq!(
        lower(&p(vec![at(R, vec![g(A_), v(0)]), at(R, vec![g(C), v(0)])])),
        Err(Refusal::FreeEndpoint)
    );
    // control: the chain lowers
    assert!(lower(&chain(A_, C)).is_ok());
}

// --- 11: the join is load-bearing -------------------------------------------

#[test]
fn c11_disabling_the_join_changes_the_answer() {
    let (t, f, r) = world(&FACTS);
    let all = sel_in_rel(&t, &f, &IDS);
    let mut p = Prepared::prepare(hop(A_, D), &r, &t, &f, Fold::Any).unwrap();
    assert_eq!(p.run(&r, &all), Value::Bool(false));
    let mut nj = Prepared::prepare(hop(A_, D), &r, &t, &f, Fold::Any)
        .unwrap()
        .without_join(Fold::Any);
    assert_eq!(nj.run(&r, &all), Value::Bool(true));
}

// --- 12: zero allocation on the hot path ------------------------------------

#[test]
fn c12_hot_run_allocates_nothing() {
    let (t, f, r) = world(&FACTS);
    let all = sel_in_rel(&t, &f, &IDS);
    let mut p = Prepared::prepare(hop(A_, C), &r, &t, &f, Fold::Count).unwrap();
    let _ = p.run(&r, &all); // warm
    let before = BYTES.with(|b| b.get());
    let mut n = 0usize;
    for _ in 0..1000 {
        if let Value::Count(c) = p.run(&r, &all) {
            n += c;
        }
    }
    let after = BYTES.with(|b| b.get());
    assert_eq!(n, 1000);
    assert_eq!(after - before, 0, "hot run allocated");
}
