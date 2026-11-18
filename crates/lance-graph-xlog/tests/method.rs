//! D-XLOG-W7-2 step 4b: a registered method executed over a cube relation's
//! lent lanes, prepared once per binding.

#[path = "support/method.rs"]
mod support;

use lance_graph_dt64::MethodId;
use lance_graph_xlog::cube::{Cube, CubeLanes};
use support::*;

/// 40 edges over ids 100..130, with repeats in both columns and one pair
/// held twice.
fn edges() -> Vec<(u32, u32)> {
    let mut e: Vec<(u32, u32)> = (0..39u32)
        .map(|i| (100 + (i * 7) % 13, 110 + (i * 5) % 19))
        .collect();
    e.push(e[3]);
    e
}

fn ids() -> Vec<u32> {
    (100..130).collect()
}

/// (method id, src, dst)
const METHODS: [(u32, u32, u32); 3] = [(1, 100, 110), (2, 108, 125), (3, 112, 111)];

/// FAILS IF: the method's count over the lent lanes differs from an edge
/// walk, or the guard's two fields are swapped.
#[test]
fn a_method_over_the_lent_lanes_equals_the_edge_walk() {
    let (t, e) = (table(&ids()), edges());
    let cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
    let w = world(
        METHODS
            .iter()
            .map(|&(id, s, d)| method(id, &t, s, d))
            .collect(),
    );
    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    let mut seen = Vec::new();
    for (id, src, dst) in METHODS {
        let r = w.store.method_ref(CLASS, MethodId(id)).unwrap();
        let mut p = Prepared::prepare(&w, &source, &t, &r).unwrap();
        let want = oracle(&e, src, dst);
        assert_eq!(p.run(), want, "method {id}");
        assert_eq!(p.run(), want, "a second run answers the same");
        // Swapping the two fields would give this instead; it must differ
        // for at least one method or the test could not see a swap.
        let swapped = e.iter().filter(|&&(a, b)| b == src && a != dst).count() as u64;
        seen.push((want, swapped));
    }
    assert!(seen.iter().any(|(want, _)| *want > 1), "{seen:?}");
    assert!(
        seen.iter().any(|(want, swapped)| want != swapped),
        "{seen:?}"
    );
    // The guard excludes rows: without `not to(dst)` one count would be larger.
    assert!(
        METHODS
            .iter()
            .any(|&(_, s, d)| oracle(&e, s, d) < e.iter().filter(|x| x.0 == s).count() as u64),
        "no method's second conjunct removes a row"
    );
}

/// FAILS IF: a preparation made after an edge was logged misses that edge.
/// (A preparation borrows the cube, so it cannot outlive the change.)
#[test]
fn a_logged_edge_needs_a_fresh_preparation() {
    let (t, mut e) = (table(&ids()), edges());
    let mut cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
    let (id, src, dst) = METHODS[0];
    let w = world(vec![method(id, &t, src, dst)]);
    let r = w.store.method_ref(CLASS, MethodId(id)).unwrap();
    let before = {
        let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
        let mut p = Prepared::prepare(&w, &source, &t, &r).unwrap();
        p.run()
    };
    assert_eq!(before, oracle(&e, src, dst));

    cube.add_edge(&t, 0, src, dst + 1).unwrap();
    e.push((src, dst + 1));
    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    let mut p = Prepared::prepare(&w, &source, &t, &r).unwrap();
    assert_eq!(p.run(), before + 1);
    assert_eq!(p.run(), oracle(&e, src, dst));
}

/// FAILS IF: a method the store does not hold, or a reference issued by
/// another store, is executed.
#[test]
fn a_method_the_store_does_not_hold_is_refused() {
    let (t, e) = (table(&ids()), edges());
    let cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
    let (id, src, dst) = METHODS[0];
    let w = world(vec![method(id, &t, src, dst)]);
    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    assert!(w.store.method_ref(CLASS, MethodId(99)).is_err());

    let other = world(vec![method(id, &t, src, dst + 1)]);
    let foreign = other.store.method_ref(CLASS, MethodId(id)).unwrap();
    assert!(matches!(
        Prepared::prepare(&w, &source, &t, &foreign),
        Err(MethodRefusal::Admit(_))
    ));
    // The silent twin: the store's own reference is admitted.
    let own = w.store.method_ref(CLASS, MethodId(id)).unwrap();
    assert!(Prepared::prepare(&w, &source, &t, &own).is_ok());
}
