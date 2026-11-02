//! XLOG and egglog reasoning over one relation.
//!
//! One `edge` relation is stored once, as a DR64 register under a sealed
//! frame. XLOG's compiled closure (`recursive::closure`) reads it for
//! directed reachability. egglog reads the same register for equality:
//! `(rule ((edge a b)) ((union a b)))`, so `a = b` exactly when `a` and `b`
//! are connected ignoring direction, and egglog returns a proof.
//!
//! What is shared:
//! - **Addresses.** Both engines name a node by its source id through one
//!   `AddressMap`; the register stores slots. The one conversion is
//!   register bits -> egglog fact text, counted below.
//! - **Witnesses.** The `edge` facts an egglog proof rests on are read back
//!   from the proof and checked against the same register bits, and against
//!   the closure's `support` masks. Neither engine's derivation is
//!   re-encoded as the other's.
//!
//! What is not shared: egglog keeps its own tables and union-find, and XLOG's
//! closure is not an equality. egglog's union-find is not replaced by masks.

#![allow(clippy::needless_range_loop)] // slot arithmetic reads clearer indexed

use std::collections::BTreeSet;

use dr_math_binding_probe::{
    Axis, AxisKind, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Scope,
};
use egglog::proof::{Justification, ProofId, ProofStore};
use egglog::{CommandOutput, EGraph};
use xlog_relcomp_probe::recursive::{closure, support};
use xlog_relcomp_probe::{AddressMap, SLOTS, register};

const N: Axis = Axis {
    class: 30,
    kind: AxisKind::Identity,
};
const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 2 };

fn sig() -> MethodSignature {
    MethodSignature::presence(40, 0, N, N)
}
fn revs(source: u64) -> FrameRevisions {
    FrameRevisions {
        source,
        relation_map: 1,
        coordinates: 1,
    }
}
fn guards(st: &FrameStore) -> Guards {
    Guards {
        frame: st.seal(),
        focus: FOCUS,
        scope: SCOPE,
    }
}

type Rows = [u64; SLOTS];

/// The one stored relation, and the map that names its slots.
struct Shared {
    map: AddressMap,
    edge: Rows,
    store: FrameStore,
}

fn shared(edges: &[(u32, u32)]) -> Shared {
    let map = AddressMap::new(N, edges.iter().flat_map(|&(a, b)| [a, b])).unwrap();
    let edge = register(edges, &map, &map).unwrap();
    Shared {
        map,
        edge,
        store: FrameStore::new(revs(1)),
    }
}

fn id(map: &AddressMap, slot: usize) -> u32 {
    map.ids(1 << slot)[0]
}

/// The conversion into egglog: one fact per set bit of the register, named
/// by source id. Read from the register, not from the source tuples.
fn egglog_facts(s: &Shared) -> String {
    let mut out = String::new();
    for u in 0..s.map.len() {
        for v in s.map.ids(s.edge[u]) {
            out.push_str(&format!("(edge (N {}) (N {v}))\n", id(&s.map, u)));
        }
    }
    out
}

const HEADER: &str = "(datatype Node (N i64))
(relation edge (Node Node))
(ruleset same)
(rule ((edge a b)) ((union a b)) :ruleset same)
";

/// egglog's proof of `(N x) = (N y)`, or `None` when it cannot prove it.
fn prove(s: &Shared, x: u32, y: u32) -> Option<(ProofStore, ProofId)> {
    let program = format!(
        "{HEADER}{}(run same 100)\n(prove (= (N {x}) (N {y})))\n",
        egglog_facts(s)
    );
    EGraph::new_with_proofs()
        .parse_and_run_program(None, &program)
        .ok()?
        .into_iter()
        .find_map(|o| match o {
            CommandOutput::ProveExists {
                proof_store,
                proof_id,
            } => Some((proof_store, proof_id)),
            _ => None,
        })
}

/// Whether egglog holds `(N x) = (N y)`, without proofs.
fn equal(s: &Shared, x: u32, y: u32) -> bool {
    let program = format!(
        "{HEADER}{}(run same 100)\n(check (= (N {x}) (N {y})))\n",
        egglog_facts(s)
    );
    EGraph::default()
        .parse_and_run_program(None, &program)
        .is_ok()
}

/// The `edge` facts a proof rests on: every proposition in the proof DAG
/// whose term is an `edge` atom, read as `(u, v)` source ids.
fn proof_edges(store: &ProofStore, root: ProofId) -> BTreeSet<(u32, u32)> {
    let mut out = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut todo = vec![root];
    while let Some(p) = todo.pop() {
        if !seen.insert(p) {
            continue;
        }
        let proof = store.get(p);
        for t in [proof.lhs(), proof.rhs()] {
            if let Some(e) = parse_edge(&store.term_dag().to_string(t)) {
                out.insert(e);
            }
        }
        match proof.justification() {
            Justification::Rule { premise_proofs, .. } => todo.extend(premise_proofs),
            Justification::MergeFn {
                old_proof,
                new_proof,
                ..
            } => todo.extend([*old_proof, *new_proof]),
            Justification::Trans(a, b) => todo.extend([*a, *b]),
            Justification::Sym(a) => todo.push(*a),
            Justification::Congr {
                proof, child_proof, ..
            } => todo.extend([*proof, *child_proof]),
            Justification::ContainerNormalize { proof } => todo.push(*proof),
            Justification::Fiat | Justification::Eval => {}
        }
    }
    out
}

fn parse_edge(t: &str) -> Option<(u32, u32)> {
    let rest = t.strip_prefix("(edge (N ")?;
    let (a, rest) = rest.split_once(") (N ")?;
    let b = rest.strip_suffix("))")?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// `E ∪ Eᵀ` over the same slots, computed from the register.
fn symmetric(edge: &Rows) -> Rows {
    let mut out = *edge;
    for (u, &w) in edge.iter().enumerate() {
        for v in 0..SLOTS {
            if w >> v & 1 == 1 {
                out[v] |= 1 << u;
            }
        }
    }
    out
}

fn closure_of(s: &Shared, rows: &Rows) -> Rows {
    closure(
        &s.store,
        &guards(&s.store),
        (FOCUS, SCOPE),
        sig(),
        rows,
        s.map.valid(),
    )
    .unwrap()
    .rows
}

const DIAMOND: &[(u32, u32)] = &[(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)];
/// Two components: {1,2,3,4,5} and {7,8}.
const TWO: &[(u32, u32)] = &[(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (8, 7)];

#[test]
fn egglog_equality_is_the_closure_of_the_symmetric_relation() {
    let s = shared(TWO);
    let sym = closure_of(&s, &symmetric(&s.edge));
    for x in 0..s.map.len() {
        for y in 0..s.map.len() {
            let want = sym[x] >> y & 1 == 1;
            assert_eq!(
                equal(&s, id(&s.map, x), id(&s.map, y)),
                want,
                "{} = {}",
                id(&s.map, x),
                id(&s.map, y)
            );
        }
    }
}

#[test]
fn every_directed_path_is_an_egglog_equality() {
    let s = shared(TWO);
    let path = closure_of(&s, &s.edge);
    for x in 0..s.map.len() {
        for y in s.map.ids(path[x]) {
            assert!(equal(&s, id(&s.map, x), y));
        }
    }
    // The converse fails: 5 = 1 holds, path(5, 1) does not.
    assert!(equal(&s, 5, 1));
    assert_eq!(
        path[s.map.slot(5).unwrap()] >> s.map.slot(1).unwrap() & 1,
        0
    );
}

#[test]
fn an_egglog_proof_rests_only_on_facts_in_the_shared_register() {
    let s = shared(DIAMOND);
    let (store, root) = prove(&s, 1, 5).expect("1 = 5 is provable");
    let used = proof_edges(&store, root);
    assert!(!used.is_empty(), "the proof names at least one edge fact");
    for &(u, v) in &used {
        let (su, sv) = (s.map.slot(u).unwrap(), s.map.slot(v).unwrap());
        assert_eq!(
            s.edge[su] >> sv & 1,
            1,
            "edge({u}, {v}) is a bit of the register"
        );
    }
    // The facts connect 1 and 5 ignoring direction.
    let mut reach = BTreeSet::from([1u32]);
    loop {
        let before = reach.len();
        for &(u, v) in &used {
            if reach.contains(&u) || reach.contains(&v) {
                reach.extend([u, v]);
            }
        }
        if reach.len() == before {
            break;
        }
    }
    assert!(reach.contains(&5));
}

#[test]
fn egglog_and_xlog_witnesses_meet_on_the_same_coordinates() {
    // For a directed answer, egglog's proof of the equality and XLOG's
    // support mask name facts in one coordinate space. Here every proof fact
    // is in the support, because the diamond has no edge off the 1 -> 5
    // routes.
    let s = shared(DIAMOND);
    let path = closure_of(&s, &s.edge);
    let (x, y) = (s.map.slot(1).unwrap(), s.map.slot(5).unwrap());
    let sup = support(&s.edge, &path, s.map.valid(), x, y);
    let (store, root) = prove(&s, 1, 5).unwrap();
    for (u, v) in proof_edges(&store, root) {
        let (su, sv) = (s.map.slot(u).unwrap(), s.map.slot(v).unwrap());
        assert_eq!(sup[su] >> sv & 1, 1, "edge({u}, {v}) supports path(1, 5)");
    }
}

#[test]
fn a_changed_fact_invalidates_both_readers_by_address() {
    let mut s = shared(DIAMOND);
    let old = guards(&s.store);
    let (store, root) = prove(&s, 1, 5).unwrap();
    let used = proof_edges(&store, root);
    // The source changes: 4 -> 5 is removed, the frame advances.
    let (s4, s5) = (s.map.slot(4).unwrap(), s.map.slot(5).unwrap());
    s.edge[s4] &= !(1 << s5);
    s.store.advance(revs(2));
    // XLOG: the old frame no longer binds.
    assert!(
        closure(
            &s.store,
            &old,
            (FOCUS, SCOPE),
            sig(),
            &s.edge,
            s.map.valid()
        )
        .is_err()
    );
    // egglog: the old proof names a fact that is no longer a bit, found by
    // address without re-running egglog.
    let stale: Vec<_> = used
        .iter()
        .filter(|&&(u, v)| s.edge[s.map.slot(u).unwrap()] >> s.map.slot(v).unwrap() & 1 == 0)
        .collect();
    assert_eq!(stale, [&(4, 5)]);
    // Re-running on the new frame agrees: 1 = 5 is gone, while 1 = 4 still
    // proves on the same program (so the program itself still runs).
    assert!(prove(&s, 1, 5).is_none());
    assert!(!equal(&s, 1, 5));
    assert!(prove(&s, 1, 4).is_some());
}

#[test]
fn report_conversion_and_work() {
    let s = shared(TWO);
    let facts = egglog_facts(&s);
    let path = closure(
        &s.store,
        &guards(&s.store),
        (FOCUS, SCOPE),
        sig(),
        &s.edge,
        s.map.valid(),
    )
    .unwrap();
    println!(
        "register: {} B, {} edges; egglog fact text: {} B; closure: {} rounds, {} row ORs",
        SLOTS * 8,
        TWO.len(),
        facts.len(),
        path.rounds,
        path.work.row_ors
    );
    assert_eq!(facts.lines().count(), TWO.len());
}
