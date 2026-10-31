//! Recursive reachability compiled by XLOG, evaluated on the DR64 register.
//!
//! Two references share no code with the evaluation:
//! 1. breadth-first search over the source tuples;
//! 2. a naive fixpoint walk of XLOG's own compiled rule bodies over tuples.

use std::collections::{BTreeSet, HashMap};

use dr_math_binding_probe::{
    Axis, AxisKind, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Refusal, Scope,
};
use xlog_core::RelId;
use xlog_ir::{ExecutionPlan, JoinType, ProjectExpr, RirNode};
use xlog_logic::{Compiler, parse_program};
use xlog_relcomp_probe::recursive::{admit_closure, affected, closure, support};
use xlog_relcomp_probe::{AddressMap, EvalRefusal, LowerRefusal, SLOTS, facts, register};

const N: Axis = Axis {
    class: 30,
    kind: AxisKind::Identity,
};
const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 2 };

fn sig() -> MethodSignature {
    MethodSignature::presence(40, 0, N, N)
}
fn store() -> FrameStore {
    FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    })
}
fn guards(st: &FrameStore) -> Guards {
    Guards {
        frame: st.seal(),
        focus: FOCUS,
        scope: SCOPE,
    }
}

const RULES: &str = "pred edge(u32, u32).
pred path(u32, u32).
path(X, Y) :- edge(X, Y).
path(X, Y) :- path(X, Z), edge(Z, Y).
?- path(X, Y).
";

fn program(edges: &[(u32, u32)], rules: &str) -> String {
    let mut s = String::new();
    for (a, b) in edges {
        s.push_str(&format!("edge({a}, {b}). "));
    }
    format!("{rules}\n{s}\n")
}

fn compile(src: &str) -> (ExecutionPlan, HashMap<String, RelId>) {
    let mut c = Compiler::new();
    let plan = c.compile(src).expect("compile");
    (plan, c.rel_ids().clone())
}

/// Reference 1: BFS from every node over the tuples.
fn bfs(edges: &[(u32, u32)]) -> BTreeSet<(u32, u32)> {
    let mut out = BTreeSet::new();
    let nodes: BTreeSet<u32> = edges.iter().flat_map(|&(a, b)| [a, b]).collect();
    for &s in &nodes {
        let mut seen = BTreeSet::new();
        let mut frontier = vec![s];
        while let Some(u) = frontier.pop() {
            for &(a, b) in edges {
                if a == u && seen.insert(b) {
                    frontier.push(b);
                }
            }
        }
        out.extend(seen.into_iter().map(|t| (s, t)));
    }
    out
}

/// Reference 2: XLOG's compiled bodies applied naively until nothing changes.
fn walk_plan(
    plan: &ExecutionPlan,
    rel: &HashMap<String, RelId>,
    edges: &[(u32, u32)],
) -> BTreeSet<(u32, u32)> {
    let mut db: HashMap<RelId, BTreeSet<Vec<u32>>> = HashMap::new();
    db.insert(
        rel["edge"],
        edges.iter().map(|&(a, b)| vec![a, b]).collect(),
    );
    db.insert(rel["path"], BTreeSet::new());
    loop {
        let mut grew = false;
        for r in plan
            .rules_by_scc
            .iter()
            .flatten()
            .filter(|r| r.head == "path")
        {
            let derived = eval(&r.body, &db);
            let tgt = db.get_mut(&rel["path"]).unwrap();
            for t in derived {
                grew |= tgt.insert(t);
            }
        }
        if !grew {
            break;
        }
    }
    db[&rel["path"]].iter().map(|t| (t[0], t[1])).collect()
}

fn eval(node: &RirNode, db: &HashMap<RelId, BTreeSet<Vec<u32>>>) -> BTreeSet<Vec<u32>> {
    match node {
        RirNode::Scan { rel } => db[rel].clone(),
        RirNode::ChainJoin {
            left,
            right,
            left_key,
            right_key,
            output_columns,
            ..
        } => project(
            &join(
                &eval(left, db),
                &eval(right, db),
                &[*left_key],
                &[*right_key],
            ),
            output_columns,
        ),
        RirNode::Project { input, columns } => match input.as_ref() {
            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type: JoinType::Inner,
            } => project(
                &join(&eval(left, db), &eval(right, db), left_keys, right_keys),
                columns,
            ),
            _ => panic!("oracle covers joins only"),
        },
        other => panic!("oracle does not cover {other:?}"),
    }
}

fn join(
    l: &BTreeSet<Vec<u32>>,
    r: &BTreeSet<Vec<u32>>,
    lk: &[usize],
    rk: &[usize],
) -> BTreeSet<Vec<u32>> {
    let mut out = BTreeSet::new();
    for a in l {
        for b in r {
            if lk.iter().zip(rk).all(|(i, j)| a[*i] == b[*j]) {
                out.insert(a.iter().chain(b).copied().collect());
            }
        }
    }
    out
}

fn project(rows: &BTreeSet<Vec<u32>>, cols: &[ProjectExpr]) -> BTreeSet<Vec<u32>> {
    rows.iter()
        .map(|t| {
            cols.iter()
                .map(|c| match c {
                    ProjectExpr::Column(i) => t[*i],
                    ProjectExpr::Computed(..) => panic!("columns only"),
                })
                .collect()
        })
        .collect()
}

struct Run {
    map: AddressMap,
    edge: [u64; SLOTS],
    path: [u64; SLOTS],
    rounds: usize,
}

fn run(edges: &[(u32, u32)]) -> Run {
    let src = program(edges, RULES);
    let (plan, rel) = compile(&src);
    let q = admit_closure(&plan, &rel).expect("admitted");
    assert_eq!(q.head, "path");
    let tuples = facts(&parse_program(&src).unwrap(), "edge").unwrap();
    let map = AddressMap::new(N, tuples.iter().flat_map(|&(a, b)| [a, b])).unwrap();
    let edge = register(&tuples, &map, &map).unwrap();
    let st = store();
    let c = closure(&st, &guards(&st), (FOCUS, SCOPE), sig(), &edge, map.valid()).unwrap();
    Run {
        map,
        edge,
        path: c.rows,
        rounds: c.rounds,
    }
}

fn pairs(r: &Run) -> BTreeSet<(u32, u32)> {
    let mut out = BTreeSet::new();
    for (i, &w) in r.path.iter().enumerate().take(r.map.len()) {
        let from = r.map.ids(1 << i)[0];
        for to in r.map.ids(w) {
            out.insert((from, to));
        }
    }
    out
}

const DIAMOND: &[(u32, u32)] = &[(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)];
const CYCLE: &[(u32, u32)] = &[(1, 2), (2, 3), (3, 1), (3, 4)];

#[test]
fn xlog_compiles_reachability_to_one_admitted_recursive_scc() {
    let (plan, rel) = compile(&program(DIAMOND, RULES));
    assert_eq!(plan.sccs.iter().filter(|s| s.is_recursive).count(), 1);
    let q = admit_closure(&plan, &rel).unwrap();
    assert_eq!(q.edge, rel["edge"]);
}

#[test]
fn closure_matches_bfs_and_a_walk_of_xlogs_own_plan() {
    for edges in [DIAMOND, CYCLE] {
        let r = run(edges);
        let src = program(edges, RULES);
        let (plan, rel) = compile(&src);
        assert_eq!(pairs(&r), bfs(edges), "{edges:?}");
        assert_eq!(pairs(&r), walk_plan(&plan, &rel, edges), "{edges:?}");
    }
}

#[test]
fn deterministic_differential_against_bfs() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    for case in 0..200 {
        let n = 2 + next() % 40;
        let m = next() % (3 * n);
        let edges: Vec<(u32, u32)> = (0..m).map(|_| (next() % n, next() % n)).collect();
        let mut dedup = Vec::new();
        for e in edges {
            if !dedup.contains(&e) {
                dedup.push(e);
            }
        }
        if dedup.is_empty() {
            continue;
        }
        assert_eq!(pairs(&run(&dedup)), bfs(&dedup), "case {case}");
    }
}

#[test]
fn a_cycle_terminates_and_reaches_itself() {
    let r = run(CYCLE);
    let p = pairs(&r);
    assert!(p.contains(&(1, 1)) && p.contains(&(3, 3)));
    assert!(!p.contains(&(4, 1)));
    // Semi-naive: rounds are bounded by the longest shortest path plus one.
    assert!(r.rounds <= 4, "rounds {}", r.rounds);
}

#[test]
fn support_names_both_alternative_derivations_and_nothing_else() {
    let r = run(DIAMOND);
    let s = |id| r.map.slot(id).unwrap();
    let sup = support(&r.edge, &r.path, r.map.valid(), s(1), s(5));
    let got: BTreeSet<(u32, u32)> = (0..r.map.len())
        .flat_map(|u| r.map.ids(sup[u]).into_iter().map(move |v| (u, v)))
        .map(|(u, v)| (r.map.ids(1 << u)[0], v))
        .collect();
    // Both branches of the diamond, and the shared tail.
    let want: BTreeSet<(u32, u32)> = DIAMOND.iter().copied().collect();
    assert_eq!(got, want);
    // A pair with one route uses only that route.
    let one = support(&r.edge, &r.path, r.map.valid(), s(2), s(5));
    assert_eq!(one[s(1)], 0, "1->2 is not on any route from 2");
    assert_eq!(one[s(3)], 0, "3->4 is not on any route from 2");
    // An answer that does not hold has no support.
    let none = support(&r.edge, &r.path, r.map.valid(), s(5), s(1));
    assert!(none.iter().all(|&w| w == 0));
}

#[test]
fn affected_is_every_answer_the_fact_feeds() {
    let r = run(DIAMOND);
    let s = |id| r.map.slot(id).unwrap();
    // Removing 3->4 and recomputing changes no answer (2->4 still routes).
    let without: Vec<_> = DIAMOND.iter().copied().filter(|&e| e != (3, 4)).collect();
    let before = bfs(DIAMOND);
    let after = bfs(&without);
    let lost: BTreeSet<_> = before.difference(&after).copied().collect();
    assert_eq!(lost, [(3, 4), (3, 5)].into_iter().collect());
    // `affected` over-approximates what a removal can change: every answer
    // with some derivation through 3->4.
    let aff = affected(&r.path, r.map.valid(), s(3), s(4));
    let fed: BTreeSet<(u32, u32)> = (0..r.map.len())
        .flat_map(|a| r.map.ids(aff[a]).into_iter().map(move |b| (a, b)))
        .map(|(a, b)| (r.map.ids(1 << a)[0], b))
        .collect();
    assert!(lost.is_subset(&fed));
    assert_eq!(
        fed,
        [(1, 4), (1, 5), (3, 4), (3, 5)]
            .into_iter()
            .collect::<BTreeSet<_>>()
    );
}

#[test]
fn refuses_non_linear_and_reordered_recursion() {
    let nonlinear = "pred edge(u32, u32).
pred path(u32, u32).
path(X, Y) :- edge(X, Y).
path(X, Y) :- path(X, Z), path(Z, Y).
?- path(X, Y).
";
    let (plan, rel) = compile(&program(DIAMOND, nonlinear));
    assert!(matches!(
        admit_closure(&plan, &rel),
        Err(LowerRefusal::UnsupportedNode(_))
    ));

    let mutual = "pred edge(u32, u32).
pred a(u32, u32).
pred b(u32, u32).
a(X, Y) :- edge(X, Y).
a(X, Y) :- b(X, Z), edge(Z, Y).
b(X, Y) :- a(X, Y).
?- a(X, Y).
";
    let (plan, rel) = compile(&program(DIAMOND, mutual));
    assert_eq!(
        admit_closure(&plan, &rel),
        Err(LowerRefusal::UnsupportedNode("mutual recursion"))
    );
}

#[test]
fn refuses_a_stale_frame() {
    let src = program(DIAMOND, RULES);
    let tuples = facts(&parse_program(&src).unwrap(), "edge").unwrap();
    let map = AddressMap::new(N, tuples.iter().flat_map(|&(a, b)| [a, b])).unwrap();
    let edge = register(&tuples, &map, &map).unwrap();
    let mut st = store();
    let old = guards(&st);
    st.advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    assert!(matches!(
        closure(&st, &old, (FOCUS, SCOPE), sig(), &edge, map.valid()),
        Err(EvalRefusal::Bind(Refusal::SourceChanged))
    ));
}
