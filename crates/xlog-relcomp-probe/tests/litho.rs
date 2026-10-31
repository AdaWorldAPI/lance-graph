//! D-LITHO-1: Gate B written out as the sequence of register operations it
//! is, and the first thing the register does not carry yet. The plan
//! (`.claude/plans/d-litho-photolithography-contract-v1.md`) is a mental
//! model; nothing here uses its vocabulary as types or names.
//!
//! Part 1 writes Gate B (`q(Y) :- r(1, Y), s(Y, 4)`) as explicit steps.
//! Each step is an existing call; the test asserts the sequence equals
//! `propagate` and both references, so the write-up cannot drift from the code.
//!
//! Part 2 is the bag-multiplicity fixture: duplicate derivation paths, with
//! per-Y counts checked by scalar enumeration and by a bag walk of XLOG's own
//! RIR. The Boolean register cannot carry them; the binder already says so
//! (`Refusal::BagNeedsMultiplicity`). One `#[ignore]`d test demands the counts
//! from the register path and fails today.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use dr_math_binding_probe::relcomp::{Chain, Rows, Work, compose};
use dr_math_binding_probe::{
    Axis, AxisKind, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand,
    Population, Refusal, Scope, bind_dr64,
};
use lance_graph_dt64::dr::{LocalMap, Width};
use xlog_core::{AggOp, RelId};
use xlog_ir::{CompareOp, ConstValue, Expr, JoinType, ProjectExpr, RirNode};
use xlog_logic::ast::Program;
use xlog_logic::{Compiler, parse_program};
use xlog_relcomp_probe::native::{admit_count, bag_count, nonzero};
use xlog_relcomp_probe::{
    AddressMap, EvalRefusal, LowerRefusal, SLOTS, admit, admit_program, facts, propagate, register,
};

const X: Axis = Axis {
    class: 10,
    kind: AxisKind::Identity,
};
const Y: Axis = Axis {
    class: 11,
    kind: AxisKind::Property,
};
const Z: Axis = Axis {
    class: 12,
    kind: AxisKind::Community,
};
const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 2 };

fn sig_r() -> MethodSignature {
    MethodSignature::presence(20, 0, X, Y)
}
fn sig_s() -> MethodSignature {
    MethodSignature::presence(21, 0, Y, Z)
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

struct Compiled {
    program: Program,
    rules: Vec<(String, RirNode)>,
    rel_ids: HashMap<String, RelId>,
}

fn compile(src: &str) -> Compiled {
    let program = parse_program(src).expect("parse");
    let mut c = Compiler::new();
    let plan = c.compile(src).expect("compile");
    let rules = plan
        .rules_by_scc
        .iter()
        .flatten()
        .map(|r| (r.head.clone(), r.body.clone()))
        .collect();
    Compiled {
        program,
        rules,
        rel_ids: c.rel_ids().clone(),
    }
}

fn rule<'a>(c: &'a Compiled, head: &str) -> &'a RirNode {
    &c.rules.iter().find(|(h, _)| h == head).expect("rule").1
}

fn edb(c: &Compiled) -> HashMap<RelId, Vec<Vec<u64>>> {
    let mut edb = HashMap::new();
    for (name, id) in &c.rel_ids {
        if let Ok(t) = facts(&c.program, name) {
            let rows = t
                .into_iter()
                .map(|(a, b)| vec![u64::from(a), u64::from(b)])
                .collect();
            edb.insert(*id, rows);
        }
    }
    edb
}

/// Bit `i` of the result is bit 0 of row `i`.
fn pack_column0(rows: &Rows) -> u64 {
    rows.iter()
        .enumerate()
        .fold(0, |m, (i, w)| m | ((w & 1) << i))
}

// ── Part 1: Gate B as explicit steps ──────────────────────────────────────

const GATE_B: &str = "pred r(u32, u32).
pred s(u32, u32).
pred q(u32).
r(1, 2). r(1, 3). r(9, 7).
s(2, 4). s(3, 5). s(7, 4).
q(Y) :- r(1, Y), s(Y, 4).
?- q(Y).
";

/// Gate B, step by step, each an existing call:
///
/// 1. fill and bind the R and S registers (`register`, `bind_dr64`);
/// 2. type the chain: one frame, focus and scope, R's columns are S's rows
///    (`Chain::new`);
/// 3. turn the bound constants into a row slot x (col0 == 1 on R) and a
///    one-column selector `C[z] = 1` (col1 == 4 on S);
/// 4. forward = the row `R[x]`;
/// 5. backward = the preimage of {z} under S (`compose` with the selector,
///    then column 0);
/// 6. answer = forward And backward, read out as ids.
///
/// Boolean throughout: one bit per (row, column), no counts.
#[test]
fn gate_b_as_explicit_steps_over_existing_calls() {
    let c = compile(GATE_B);
    admit_program(&c.program).unwrap();
    let q = admit(rule(&c, "q")).unwrap();
    let r = facts(&c.program, "r").unwrap();
    let s = facts(&c.program, "s").unwrap();

    // One 64-slot word per axis; the answer is read in the Y word.
    let xs = AddressMap::new(X, r.iter().map(|t| t.0).chain(q.left_bound)).unwrap();
    let ys = AddressMap::new(Y, r.iter().map(|t| t.1).chain(s.iter().map(|t| t.0))).unwrap();
    let zs = AddressMap::new(Z, s.iter().map(|t| t.1).chain(q.right_bound)).unwrap();

    // Registers: written once, then only borrowed.
    let rr = register(&r, &xs, &ys).unwrap();
    let ss = register(&s, &ys, &zs).unwrap();
    let st = store();
    let g = guards(&st);
    let bind = |sig, rows| {
        bind_dr64(
            &st,
            sig,
            sig,
            &g,
            (FOCUS, SCOPE),
            Operand {
                rows,
                validity: None,
            },
        )
    };
    let bound_r = bind(sig_r(), &rr).unwrap();
    let bound_s = bind(sig_s(), &ss).unwrap();

    // Typing: R's column domain is S's row domain, one frame, one focus.
    Chain::new(&st, vec![bound_r.clone(), bound_s.clone()]).unwrap();

    // The two bound constants, as a row slot and as a selector.
    let x = xs.slot(q.left_bound.unwrap()).unwrap();
    let mut selector = [0u64; SLOTS];
    selector[zs.slot(q.right_bound.unwrap()).unwrap()] = 1;

    // Forward and backward. The valid mask keeps padding bits out.
    let valid = ys.valid();
    let forward = bound_r.rows[x] & valid;
    let mut work = Work::default();
    let backward = pack_column0(&compose(bound_s.rows, &selector, &mut work)) & valid;

    let answer = ys.ids(forward & backward);

    assert_eq!(ys.ids(forward), vec![2, 3]);
    assert_eq!(ys.ids(backward), vec![2, 7]);
    assert_eq!(answer, vec![2]);
    assert_eq!(work.row_ors, 3, "one row OR per edge of S");

    // The steps equal `propagate`.
    let p = propagate(
        &st,
        &g,
        (FOCUS, SCOPE),
        (sig_r(), &rr),
        (sig_s(), &ss),
        Some(x),
        Some(zs.slot(4).unwrap()),
        &ys,
    )
    .unwrap();
    assert_eq!(
        (p.forward, p.backward, p.answer),
        (forward, backward, forward & backward)
    );
}

#[test]
fn binding_on_a_stale_frame_is_refused() {
    let rr = [0u64; SLOTS];
    let mut st = store();
    let g = guards(&st);
    st.advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    let e = bind_dr64(
        &st,
        sig_r(),
        sig_r(),
        &g,
        (FOCUS, SCOPE),
        Operand {
            rows: &rr,
            validity: None,
        },
    );
    assert_eq!(e.err(), Some(Refusal::SourceChanged));
}

// ── Part 2: bag multiplicity, the failing fixture ──────────────────────────

/// Y = 2 has two R witnesses (x = 1, 5) and two S witnesses (z = 4, 6), so
/// four derivation paths. Y = 3 and Y = 7 have one each. `qc` is the same body
/// with XLOG's own `count` aggregate.
const MULT: &str = "pred r(u32, u32).
pred s(u32, u32).
pred q(u32).
pred qc(u32, u64).
r(1, 2). r(5, 2). r(1, 3). r(9, 7).
s(2, 4). s(2, 6). s(3, 4). s(7, 8).
q(Y) :- r(X, Y), s(Y, Z).
qc(Y, count(X)) :- r(X, Y), s(Y, Z).
?- q(Y).
";

/// Per-Y derivation counts, by hand from the fixture.
fn expected() -> BTreeMap<u32, u64> {
    BTreeMap::from([(2, 4), (3, 1), (7, 1)])
}

/// Oracle 1: count every (x, y, z) witness of `r(X, Y), s(Y, Z)`.
fn scalar_bag(r: &[(u32, u32)], s: &[(u32, u32)]) -> BTreeMap<u32, u64> {
    let mut out = BTreeMap::new();
    for &(_, y) in r {
        for &(y2, _) in s {
            if y == y2 {
                *out.entry(y).or_insert(0) += 1;
            }
        }
    }
    out
}

/// Oracle 2: walk XLOG's compiled RIR over tuples with **bag** semantics:
/// `Project` keeps duplicates and `GroupBy` counts the rows of its input.
/// XLOG's executor documents the same reading: "the lowered group input is a
/// non-deduplicating projection of the join output, so both paths count
/// distinct full body bindings" (`xlog-runtime` `wcoj_dispatch.rs`, at the
/// pinned revision). `dedup` switches `Project` to set semantics, which is
/// what Gate B's oracle uses and what this fixture shows to be wrong here.
fn walk(node: &RirNode, edb: &HashMap<RelId, Vec<Vec<u64>>>, dedup: bool) -> Vec<Vec<u64>> {
    match node {
        RirNode::Scan { rel } => edb[rel].clone(),
        RirNode::Filter { input, predicate } => {
            let Expr::Compare {
                left,
                op: CompareOp::Eq,
                right,
            } = predicate
            else {
                panic!("oracle covers only == filters")
            };
            let (Expr::Column(c), Expr::Const(ConstValue::U32(v))) =
                (left.as_ref(), right.as_ref())
            else {
                panic!("oracle covers only column == u32")
            };
            walk(input, edb, dedup)
                .into_iter()
                .filter(|t| t[*c] == u64::from(*v))
                .collect()
        }
        RirNode::Join {
            left,
            right,
            left_keys,
            right_keys,
            join_type: JoinType::Inner,
        } => {
            let (l, r) = (walk(left, edb, dedup), walk(right, edb, dedup));
            let mut out = Vec::new();
            for a in &l {
                for b in &r {
                    if left_keys
                        .iter()
                        .zip(right_keys)
                        .all(|(i, j)| a[*i] == b[*j])
                    {
                        out.push(a.iter().chain(b).copied().collect());
                    }
                }
            }
            out
        }
        // Both readings are XLOG's: the promoted node and its fallback must
        // agree, as `admit` already requires.
        RirNode::ChainJoin { fallback, .. } => walk(fallback, edb, dedup),
        RirNode::Project { input, columns } => {
            let rows: Vec<Vec<u64>> = walk(input, edb, dedup)
                .into_iter()
                .map(|t| {
                    columns
                        .iter()
                        .map(|c| match c {
                            ProjectExpr::Column(i) => t[*i],
                            ProjectExpr::Computed(..) => panic!("oracle covers columns only"),
                        })
                        .collect()
                })
                .collect();
            if dedup {
                rows.into_iter()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect()
            } else {
                rows
            }
        }
        RirNode::GroupBy {
            input,
            key_cols,
            aggs,
        } => {
            assert_eq!(aggs.len(), 1, "oracle covers one aggregate");
            assert!(matches!(aggs[0].1, AggOp::Count), "oracle covers count");
            let mut groups: BTreeMap<Vec<u64>, u64> = BTreeMap::new();
            for t in walk(input, edb, dedup) {
                *groups
                    .entry(key_cols.iter().map(|k| t[*k]).collect())
                    .or_insert(0) += 1;
            }
            groups
                .into_iter()
                .map(|(mut k, n)| {
                    k.push(n);
                    k
                })
                .collect()
        }
        other => panic!("oracle does not cover {other:?}"),
    }
}

fn counts(rows: Vec<Vec<u64>>, y: usize, n: Option<usize>) -> BTreeMap<u32, u64> {
    let mut out = BTreeMap::new();
    for t in rows {
        let key = u32::try_from(t[y]).unwrap();
        *out.entry(key).or_insert(0) += n.map_or(1, |i| t[i]);
    }
    out
}

/// Demand shape of the fixture, stated so the counts cannot be read as a
/// claim about a larger domain.
#[test]
fn multiplicity_fixture_shape() {
    let c = compile(MULT);
    let r = facts(&c.program, "r").unwrap();
    let s = facts(&c.program, "s").unwrap();
    let ys = AddressMap::new(Y, r.iter().map(|t| t.1).chain(s.iter().map(|t| t.0))).unwrap();
    let xs = AddressMap::new(X, r.iter().map(|t| t.0)).unwrap();
    let zs = AddressMap::new(Z, s.iter().map(|t| t.1)).unwrap();
    // Domain: 3 Y ids in one 64-slot frame (one tile demanded), 3 X ids,
    // 3 Z ids. Density: 4 of 3 x 3 cells in R and in S.
    assert_eq!((xs.len(), ys.len(), zs.len()), (3, 3, 3));
    assert_eq!((r.len(), s.len()), (4, 4));
    // The largest count is |X| * |Z| restricted to one Y: here 2 * 2 = 4,
    // which needs three bits per Y in a counting register.
    assert_eq!(expected().values().max(), Some(&4));
}

#[test]
fn both_oracles_agree_on_the_per_y_counts() {
    let c = compile(MULT);
    let r = facts(&c.program, "r").unwrap();
    let s = facts(&c.program, "s").unwrap();
    let e = edb(&c);
    assert_eq!(scalar_bag(&r, &s), expected());
    // XLOG's own `q` plan, walked as a bag: one row per derivation.
    assert_eq!(counts(walk(rule(&c, "q"), &e, false), 0, None), expected());
    // XLOG's own `count` plan, walked as a bag: one row per Y with its count.
    assert_eq!(
        counts(walk(rule(&c, "qc"), &e, false), 0, Some(1)),
        expected()
    );
}

#[test]
fn set_semantics_loses_the_counts_the_fixture_is_about() {
    let c = compile(MULT);
    let e = edb(&c);
    // Deduplicating the projection counts distinct X per Y (2 for Y = 2),
    // not derivations (4). This is why Gate B's set-valued oracle cannot be
    // reused for a bag question.
    let set = counts(walk(rule(&c, "qc"), &e, true), 0, Some(1));
    assert_eq!(set, BTreeMap::from([(2, 2), (3, 1), (7, 1)]));
    assert_ne!(set, expected());
}

#[test]
fn the_boolean_register_answers_support_only() {
    let c = compile(MULT);
    let q = admit(rule(&c, "q")).unwrap();
    assert_eq!((q.left_bound, q.right_bound), (None, None));
    let r = facts(&c.program, "r").unwrap();
    let s = facts(&c.program, "s").unwrap();
    let xs = AddressMap::new(X, r.iter().map(|t| t.0)).unwrap();
    let ys = AddressMap::new(Y, r.iter().map(|t| t.1).chain(s.iter().map(|t| t.0))).unwrap();
    let zs = AddressMap::new(Z, s.iter().map(|t| t.1)).unwrap();
    let rr = register(&r, &xs, &ys).unwrap();
    let ss = register(&s, &ys, &zs).unwrap();
    let st = store();
    let p = propagate(
        &st,
        &guards(&st),
        (FOCUS, SCOPE),
        (sig_r(), &rr),
        (sig_s(), &ss),
        None,
        None,
        &ys,
    )
    .unwrap();
    // Support is right; one bit per cell cannot hold a count.
    assert_eq!(
        ys.ids(p.answer),
        expected().keys().copied().collect::<Vec<_>>()
    );
}

#[test]
fn the_count_rule_is_refused_not_answered() {
    let c = compile(MULT);
    // `admit` does not recognise the GroupBy plan; it must refuse, never
    // return the support mask in place of a count.
    let e = admit(rule(&c, "qc")).unwrap_err();
    assert_eq!(e, LowerRefusal::UnsupportedNode("GroupBy"));
}

#[test]
fn the_binder_refuses_a_bag_operand() {
    let rr = [0u64; SLOTS];
    let st = store();
    let bag = MethodSignature {
        population: Population::Bag,
        ..sig_r()
    };
    let e = bind_dr64(
        &st,
        bag,
        bag,
        &guards(&st),
        (FOCUS, SCOPE),
        Operand {
            rows: &rr,
            validity: None,
        },
    );
    assert_eq!(e.err(), Some(Refusal::BagNeedsMultiplicity));
}

/// The failing fixture. It demands the per-Y counts from the register path
/// and fails today: the path only carries support, so every derived Y reads
/// as 1. Run with `cargo test --test litho -- --ignored`.
#[test]
#[ignore = "D-LITHO-1: bag multiplicity is not carried by the Boolean DR64 register"]
fn the_register_path_carries_bag_counts() {
    let c = compile(MULT);
    let r = facts(&c.program, "r").unwrap();
    let s = facts(&c.program, "s").unwrap();
    let xs = AddressMap::new(X, r.iter().map(|t| t.0)).unwrap();
    let ys = AddressMap::new(Y, r.iter().map(|t| t.1).chain(s.iter().map(|t| t.0))).unwrap();
    let zs = AddressMap::new(Z, s.iter().map(|t| t.1)).unwrap();
    let rr = register(&r, &xs, &ys).unwrap();
    let ss = register(&s, &ys, &zs).unwrap();
    let st = store();
    let p: Result<_, EvalRefusal> = propagate(
        &st,
        &guards(&st),
        (FOCUS, SCOPE),
        (sig_r(), &rr),
        (sig_s(), &ss),
        None,
        None,
        &ys,
    );
    let read_out: BTreeMap<u32, u64> = ys
        .ids(p.unwrap().answer)
        .into_iter()
        .map(|y| (y, 1))
        .collect();
    assert_eq!(read_out, expected());
}

// ── Part 3: the bag count on the ndarray::simd polyfill ────────────────────

fn y_map(r: &[(u32, u32)], s: &[(u32, u32)]) -> LocalMap {
    let ys: BTreeSet<u64> = r
        .iter()
        .map(|t| u64::from(t.1))
        .chain(s.iter().map(|t| u64::from(t.0)))
        .collect();
    LocalMap::prepare(Width::W64, &ys.into_iter().collect::<Vec<_>>()).unwrap()
}

/// XLOG's own `count` rule is admitted on the native path and answered with
/// two keyed counts and a checked multiply; both oracles agree.
#[test]
fn the_native_path_carries_bag_counts() {
    let c = compile(MULT);
    let r = facts(&c.program, "r").unwrap();
    let s = facts(&c.program, "s").unwrap();
    let q = admit_count(rule(&c, "qc")).expect("the count plan is admitted");
    assert_eq!(q, admit(rule(&c, "q")).unwrap(), "same body as q");
    let mid = y_map(&r, &s);
    let got: BTreeMap<u32, u64> = nonzero(&mid, &bag_count(&q, &r, &s, &mid).unwrap())
        .into_iter()
        .map(|(y, n)| (u32::try_from(y).unwrap(), u64::try_from(n).unwrap()))
        .collect();
    assert_eq!(got, expected());
    assert_eq!(got, scalar_bag(&r, &s));
    assert_eq!(
        got,
        counts(walk(rule(&c, "qc"), &edb(&c), false), 0, Some(1))
    );
}

/// A bound end restricts both factors before the multiply.
#[test]
fn the_native_bag_count_respects_a_bound_end() {
    let src = MULT.replace(
        "qc(Y, count(X)) :- r(X, Y), s(Y, Z).",
        "qc(Y, count(X)) :- r(X, Y), s(Y, 4).",
    );
    let c = compile(&src);
    let r = facts(&c.program, "r").unwrap();
    let s = facts(&c.program, "s").unwrap();
    let q = admit_count(rule(&c, "qc")).unwrap();
    assert_eq!(q.right_bound, Some(4));
    let mid = y_map(&r, &s);
    let got: BTreeMap<u32, u64> = nonzero(&mid, &bag_count(&q, &r, &s, &mid).unwrap())
        .into_iter()
        .map(|(y, n)| (u32::try_from(y).unwrap(), u64::try_from(n).unwrap()))
        .collect();
    // Y = 2: two X times one Z (4); Y = 3: one times one; Y = 7 has no Z = 4.
    assert_eq!(got, BTreeMap::from([(2, 2), (3, 1)]));
    assert_eq!(
        got,
        counts(walk(rule(&c, "qc"), &edb(&c), false), 0, Some(1))
    );
}

/// Only `Count` keyed by the answer column is admitted.
#[test]
fn the_native_path_refuses_other_aggregates() {
    let c = compile(MULT);
    let RirNode::Project { input, columns } = rule(&c, "qc").clone() else {
        panic!("expected Project")
    };
    let RirNode::GroupBy {
        input: inner,
        key_cols,
        ..
    } = *input
    else {
        panic!("expected GroupBy")
    };
    let sum = RirNode::Project {
        input: Box::new(RirNode::GroupBy {
            input: inner,
            key_cols,
            aggs: vec![(1, AggOp::Sum)],
        }),
        columns,
    };
    assert_eq!(admit_count(&sum), Err(LowerRefusal::UnsupportedAggregate));
    assert_eq!(
        admit_count(rule(&c, "q")),
        Err(LowerRefusal::UnsupportedNode("ChainJoin")),
        "the support rule is not a count"
    );
}

/// A repeated base fact is one tuple, as in XLOG's fact loader: the count
/// does not change when `r(1, 2)` and `s(2, 4)` are declared twice.
#[test]
fn a_repeated_base_fact_is_not_a_second_witness() {
    let src = MULT
        .replace("r(1, 2).", "r(1, 2). r(1, 2).")
        .replace("s(2, 4).", "s(2, 4). s(2, 4).");
    let c = compile(&src);
    let r = facts(&c.program, "r").unwrap();
    let s = facts(&c.program, "s").unwrap();
    assert_eq!((r.len(), s.len()), (4, 4), "duplicates collapse");
    let q = admit_count(rule(&c, "qc")).unwrap();
    let mid = y_map(&r, &s);
    let got: BTreeMap<u32, u64> = nonzero(&mid, &bag_count(&q, &r, &s, &mid).unwrap())
        .into_iter()
        .map(|(y, n)| (u32::try_from(y).unwrap(), u64::try_from(n).unwrap()))
        .collect();
    assert_eq!(got, expected());
}
