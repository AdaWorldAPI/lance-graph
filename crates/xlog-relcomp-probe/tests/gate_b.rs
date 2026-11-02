//! Gate B: `q(Y) :- r(1, Y), s(Y, 4)` compiled by XLOG, lowered onto two
//! DR64 registers, answered by forward ∧ backward masks, and checked against
//! two references that share no code with the lowering:
//!
//! 1. scalar enumeration over the source tuples;
//! 2. a tuple-level walk of XLOG's own compiled RIR (Scan / Filter / Join /
//!    Project, as XLOG built it), which catches a lowering that reads the
//!    plan wrongly.
//!
//! XLOG's executor is CUDA-only, so (2) is the closest CPU stand-in for
//! "XLOG's current compiled query"; it is a test oracle, never an engine.

use std::collections::{BTreeSet, HashMap};

use dr_math_binding_probe::relcomp::ComposeRefusal;
use dr_math_binding_probe::{
    Axis, AxisKind, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Refusal, Scope,
};
use lance_graph_dt64::dr::{LocalMap, MapRefusal, Width};
use xlog_core::RelId;
use xlog_ir::{CompareOp, ConstValue, Expr, JoinType, ProjectExpr, RirNode};
use xlog_logic::ast::Program;
use xlog_logic::{Compiler, MagicSetStatus, parse_program, rewrite_magic_sets};
use xlog_relcomp_probe::native::{self, NativeRefusal};
use xlog_relcomp_probe::{
    AddressMap, ChainQuery, DomainRefusal, EvalRefusal, LowerRefusal, SLOTS, admit, admit_program,
    facts, propagate, register,
};

const FIXTURE: &str = "pred r(u32, u32).
pred s(u32, u32).
pred q(u32).
r(1, 2). r(1, 3). r(9, 7).
s(2, 4). s(3, 5). s(7, 4).
q(Y) :- r(1, Y), s(Y, 4).
?- q(Y).
";

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

fn name_of(c: &Compiled, id: RelId) -> String {
    c.rel_ids
        .iter()
        .find(|(_, v)| **v == id)
        .expect("rel id")
        .0
        .clone()
}

/// Oracle 1: plain enumeration over source tuples.
fn scalar(r: &[(u32, u32)], s: &[(u32, u32)], x: Option<u32>, z: Option<u32>) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    for &(a, b) in r {
        for &(c, d) in s {
            if b == c && x.is_none_or(|x| a == x) && z.is_none_or(|z| d == z) {
                out.insert(b);
            }
        }
    }
    out
}

/// Oracle 2: walk XLOG's compiled RIR over tuples, as XLOG built it.
fn walk(node: &RirNode, edb: &HashMap<RelId, Vec<Vec<u32>>>) -> BTreeSet<Vec<u32>> {
    match node {
        RirNode::Scan { rel } => edb[rel].iter().cloned().collect(),
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
            walk(input, edb)
                .into_iter()
                .filter(|t| t[*c] == *v)
                .collect()
        }
        RirNode::Join {
            left,
            right,
            left_keys,
            right_keys,
            join_type: JoinType::Inner,
        } => {
            let (l, r) = (walk(left, edb), walk(right, edb));
            let mut out = BTreeSet::new();
            for a in &l {
                for b in &r {
                    if left_keys
                        .iter()
                        .zip(right_keys)
                        .all(|(i, j)| a[*i] == b[*j])
                    {
                        out.insert(a.iter().chain(b).copied().collect());
                    }
                }
            }
            out
        }
        // XLOG's chain-join semantics: join left.col[left_key] =
        // right.col[right_key], then project the concatenated tuple.
        RirNode::ChainJoin {
            left,
            right,
            left_key,
            right_key,
            output_columns,
            ..
        } => walk(
            &RirNode::Project {
                input: Box::new(RirNode::Join {
                    left: left.clone(),
                    right: right.clone(),
                    left_keys: vec![*left_key],
                    right_keys: vec![*right_key],
                    join_type: JoinType::Inner,
                }),
                columns: output_columns.clone(),
            },
            edb,
        ),
        RirNode::Project { input, columns } => walk(input, edb)
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
            .collect(),
        other => panic!("oracle does not cover {other:?}"),
    }
}

fn xlog_oracle(c: &Compiled) -> BTreeSet<u32> {
    let mut edb = HashMap::new();
    for (name, id) in &c.rel_ids {
        if let Ok(t) = facts(&c.program, name) {
            edb.insert(*id, t.into_iter().map(|(a, b)| vec![a, b]).collect());
        }
    }
    walk(rule(c, "q"), &edb).into_iter().map(|t| t[0]).collect()
}

/// Everything the native path needs for one compiled chain query.
struct Prepared {
    q: ChainQuery,
    r: Vec<(u32, u32)>,
    s: Vec<(u32, u32)>,
    xs: AddressMap,
    ys: AddressMap,
    zs: AddressMap,
}

fn prepare(c: &Compiled) -> Result<Prepared, LowerRefusal> {
    admit_program(&c.program)?;
    let q = admit(rule(c, "q"))?;
    let r = facts(&c.program, &name_of(c, q.left))?;
    let s = facts(&c.program, &name_of(c, q.right))?;
    let xs = AddressMap::new(X, r.iter().map(|t| t.0).chain(q.left_bound)).unwrap();
    let ys = AddressMap::new(Y, r.iter().map(|t| t.1).chain(s.iter().map(|t| t.0))).unwrap();
    let zs = AddressMap::new(Z, s.iter().map(|t| t.1).chain(q.right_bound)).unwrap();
    Ok(Prepared {
        q,
        r,
        s,
        xs,
        ys,
        zs,
    })
}

fn answer(p: &Prepared) -> Result<(BTreeSet<u32>, xlog_relcomp_probe::Propagation), EvalRefusal> {
    let st = store();
    let g = guards(&st);
    let rr = register(&p.r, &p.xs, &p.ys).map_err(EvalRefusal::Domain)?;
    let ss = register(&p.s, &p.ys, &p.zs).map_err(EvalRefusal::Domain)?;
    let x =
        p.q.left_bound
            .map(|v| p.xs.slot(v))
            .transpose()
            .map_err(EvalRefusal::Domain)?;
    let z =
        p.q.right_bound
            .map(|v| p.zs.slot(v))
            .transpose()
            .map_err(EvalRefusal::Domain)?;
    let prop = propagate(
        &st,
        &g,
        (FOCUS, SCOPE),
        (sig_r(), &rr),
        (sig_s(), &ss),
        x,
        z,
        &p.ys,
    )?;
    // The native path is the execution path; the register is its
    // differential. Every mask must agree, not only the answer.
    let mid = local(&p.ys);
    let n = native::support(&p.q, &p.r, &p.s, &mid).expect("native support");
    let ids = |m: &[u64]| -> Vec<u32> {
        native::ids(&mid, m)
            .into_iter()
            .map(|g| u32::try_from(g).unwrap())
            .collect()
    };
    assert_eq!(ids(&n.forward), p.ys.ids(prop.forward), "forward");
    assert_eq!(ids(&n.backward), p.ys.ids(prop.backward), "backward");
    assert_eq!(ids(&n.answer), p.ys.ids(prop.answer), "answer");
    Ok((p.ys.ids(prop.answer).into_iter().collect(), prop))
}

/// DR's local map for the Y axis, in the same ascending order the register's
/// `AddressMap` assigns.
fn local(ys: &AddressMap) -> LocalMap {
    let globals: Vec<u64> = ys.ids(ys.valid()).into_iter().map(u64::from).collect();
    LocalMap::prepare(Width::W256, &globals).unwrap()
}

#[test]
fn xlog_compiles_the_fixture_to_the_admitted_shape() {
    let c = compile(FIXTURE);
    let q = admit(rule(&c, "q")).expect("admitted");
    assert_eq!(name_of(&c, q.left), "r");
    assert_eq!(name_of(&c, q.right), "s");
    assert_eq!((q.left_bound, q.right_bound), (Some(1), Some(4)));
}

#[test]
fn forward_backward_and_answer_match_both_references() {
    let c = compile(FIXTURE);
    let p = prepare(&c).expect("lowers");
    let (got, prop) = answer(&p).expect("answers");

    // The intermediate masks, as source ids.
    assert_eq!(p.ys.ids(prop.forward), vec![2, 3]);
    assert_eq!(p.ys.ids(prop.backward), vec![2, 7]);
    assert_eq!(got, BTreeSet::from([2]));

    // Oracle 1.
    assert_eq!(got, scalar(&p.r, &p.s, Some(1), Some(4)));

    // Oracle 2: XLOG's own plan, walked over the same facts.
    assert_eq!(got, xlog_oracle(&c));

    // Work: one row read forward; 64 rows read backward, one row OR per
    // set bit of S (three edges).
    assert_eq!(prop.words_read, 1 + SLOTS as u64);
    assert_eq!(prop.work.row_ors, 3);
}

/// Load-bearing: each arm alone answers a different, larger set. If the
/// backward compose (or the AND) were removed, the fixture's answer would
/// change, and this test is what notices.
#[test]
fn both_arms_are_load_bearing() {
    let c = compile(FIXTURE);
    let (_, prop) = answer(&prepare(&c).unwrap()).unwrap();
    assert_ne!(prop.forward, prop.answer, "forward alone over-selects");
    assert_ne!(prop.backward, prop.answer, "backward alone over-selects");
    assert_eq!(prop.forward & prop.backward, prop.answer);
}

#[test]
fn unbound_ends_still_match_scalar() {
    let src = FIXTURE.replace("q(Y) :- r(1, Y), s(Y, 4).", "q(Y) :- r(X, Y), s(Y, Z).");
    let c = compile(&src);
    let p = prepare(&c).expect("lowers");
    assert!(
        matches!(rule(&c, "q"), RirNode::ChainJoin { .. }),
        "XLOG promotes the unbound chain"
    );
    assert_eq!((p.q.left_bound, p.q.right_bound), (None, None));
    let (got, _) = answer(&p).unwrap();
    assert_eq!(got, scalar(&p.r, &p.s, None, None));
    assert_eq!(got, BTreeSet::from([2, 3, 7]));
    assert_eq!(got, xlog_oracle(&c));
}

/// Deterministic differential over dense, sparse, skewed and empty relations.
#[test]
fn differential_against_scalar_enumeration() {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for case in 0..300 {
        let density = [0u64, 1, 4, 16, 48, 64][case % 6];
        let mut draw = |n: usize| -> Vec<(u32, u32)> {
            (0..n)
                .filter_map(|_| {
                    let (a, b) = ((next() % 64) as u32 * 7 + 1, (next() % 64) as u32 * 3);
                    (next() % 64 < density).then_some((a, b))
                })
                .collect()
        };
        let r = draw(512);
        let s: Vec<(u32, u32)> = draw(512).into_iter().map(|(a, b)| (b, a)).collect();
        let xb = r.first().map(|t| t.0);
        let zb = s.last().map(|t| t.1);
        let p = Prepared {
            q: ChainQuery {
                left: RelId(0),
                left_bound: xb,
                right: RelId(1),
                right_bound: zb,
            },
            xs: AddressMap::new(X, r.iter().map(|t| t.0).chain(xb)).unwrap(),
            ys: AddressMap::new(Y, r.iter().map(|t| t.1).chain(s.iter().map(|t| t.0))).unwrap(),
            zs: AddressMap::new(Z, s.iter().map(|t| t.1).chain(zb)).unwrap(),
            r,
            s,
        };
        let (got, _) = answer(&p).unwrap();
        assert_eq!(got, scalar(&p.r, &p.s, xb, zb), "case {case}");
    }
}

/// XLOG orders joins from statistics: with a large, unselective S it plans
/// `s ⋈ r` instead of `r ⋈ s` for the SAME rule. Admission must accept both
/// orders and land on the same query; the answer must not depend on it.
#[test]
fn join_order_chosen_by_xlog_statistics_does_not_change_the_answer() {
    let mut src = String::from("pred r(u32, u32).\npred s(u32, u32).\npred q(u32).\n");
    for b in 0..64 {
        src += &format!("r(1, {b}).\n");
    }
    for a in 0..64 {
        for c in (0..64).filter(|&c| c != 4) {
            src += &format!("s({a}, {c}).\n");
        }
    }
    src += "s(5, 4).\nq(Y) :- r(1, Y), s(Y, 4).\n?- q(Y).\n";
    let c = compile(&src);
    let RirNode::Project { input, .. } = rule(&c, "q") else {
        panic!("expected Project")
    };
    let RirNode::Join { left_keys, .. } = input.as_ref() else {
        panic!("expected Join")
    };
    assert_eq!(left_keys, &vec![0], "XLOG picked s ⋈ r for this data");

    let p = prepare(&c).expect("the swapped order lowers");
    assert_eq!(name_of(&c, p.q.left), "r");
    assert_eq!(name_of(&c, p.q.right), "s");
    let (got, _) = answer(&p).unwrap();
    assert_eq!(got, BTreeSet::from([5]));
    assert_eq!(got, scalar(&p.r, &p.s, Some(1), Some(4)));
    assert_eq!(got, xlog_oracle(&c));
}

// ── magic sets ─────────────────────────────────────────────────────────────

/// XLOG's magic-set rewrite targets bound RECURSIVE queries only. On this
/// one-step query `auto` declines with its own reason and `on` is an error:
/// arm 2 of Gate B is "not eligible", reported, not forced.
#[test]
fn magic_sets_decline_a_nonrecursive_query() {
    let auto = format!("#pragma magic_sets = auto\n{FIXTURE}");
    let rw = rewrite_magic_sets(&parse_program(&auto).unwrap()).unwrap();
    assert_eq!(rw.report.status, MagicSetStatus::Declined);
    assert_eq!(
        rw.report.declined_reasons,
        vec!["no bound recursive query eligible for magic_sets".to_string()]
    );
    let on = format!("#pragma magic_sets = on\n{FIXTURE}");
    assert!(rewrite_magic_sets(&parse_program(&on).unwrap()).is_err());
    // And the declined program still compiles to the same admitted shape.
    let c = compile(&auto);
    assert_eq!(admit(rule(&c, "q")), admit(rule(&compile(FIXTURE), "q")));
}

// ── refusals ───────────────────────────────────────────────────────────────

#[test]
fn refuses_reversed_orientation_without_an_inverse() {
    // s(4, Y): the join reads s's column 1, which needs s⁻¹.
    let c = compile(&FIXTURE.replace("s(Y, 4)", "s(4, Y)"));
    assert!(matches!(
        admit(rule(&c, "q")),
        Err(LowerRefusal::NeedsInverse { .. })
    ));
}

#[test]
fn refuses_negation_unknown_is_not_false() {
    let c = compile(&FIXTURE.replace(", s(Y, 4).", ", not s(Y, 4)."));
    assert_eq!(
        admit_program(&c.program),
        Err(LowerRefusal::NonPositiveBody)
    );
    assert!(admit(rule(&c, "q")).is_err());
}

#[test]
fn refuses_probabilistic_programs_even_when_the_rir_is_deterministic() {
    let src = FIXTURE.replace("r(1, 2).", "0.5::r(1, 2).");
    let program = parse_program(&src).unwrap();
    assert_eq!(admit_program(&program), Err(LowerRefusal::Probabilistic));
}

#[test]
fn refuses_comparisons_and_non_integer_facts() {
    let c = compile(&FIXTURE.replace(", s(Y, 4).", ", s(Y, Z), Z != 5."));
    assert_eq!(
        admit_program(&c.program),
        Err(LowerRefusal::NonPositiveBody)
    );

    let src = FIXTURE.replace("r(9, 7).", "r(9, -7).");
    let program = parse_program(&src).unwrap();
    assert!(matches!(
        facts(&program, "r"),
        Err(LowerRefusal::FactTerm { .. })
    ));
}

/// Equal numeric ids on two axes are unrelated. Binding S with its rows on
/// X instead of Y (same slot numbers, different domain) is refused by the
/// EXISTING chain typing, not by anything new here.
#[test]
fn refuses_cross_domain_alias() {
    let st = store();
    let g = guards(&st);
    let ys = AddressMap::new(Y, [2, 3, 7]).unwrap();
    let rows = [0u64; SLOTS];
    let aliased = MethodSignature::presence(21, 0, X, Z);
    let e = propagate(
        &st,
        &g,
        (FOCUS, SCOPE),
        (sig_r(), &rows),
        (aliased, &rows),
        Some(0),
        Some(0),
        &ys,
    );
    assert_eq!(
        e,
        Err(EvalRefusal::Chain(ComposeRefusal::DomainMismatch { at: 0 }))
    );
    // Reading the answer in an axis the left operand does not produce.
    let zs = AddressMap::new(Z, [4]).unwrap();
    let e = propagate(
        &st,
        &g,
        (FOCUS, SCOPE),
        (sig_r(), &rows),
        (sig_s(), &rows),
        Some(0),
        Some(0),
        &zs,
    );
    assert_eq!(e, Err(EvalRefusal::AxisMismatch));
}

#[test]
fn refuses_a_chain_join_that_disagrees_with_its_fallback() {
    let c = compile(&FIXTURE.replace("q(Y) :- r(1, Y), s(Y, 4).", "q(Y) :- r(X, Y), s(Y, Z)."));
    let mut body = rule(&c, "q").clone();
    let RirNode::ChainJoin { fallback, .. } = &mut body else {
        panic!("expected ChainJoin")
    };
    // Corrupt only the fallback: bind its left end.
    let RirNode::Project { input, .. } = fallback.as_mut() else {
        panic!()
    };
    let RirNode::Join { left, .. } = input.as_mut() else {
        panic!()
    };
    let scan = std::mem::replace(left.as_mut(), RirNode::Unit);
    **left = RirNode::Filter {
        input: Box::new(scan),
        predicate: Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Eq,
            right: Box::new(Expr::Const(ConstValue::U32(1))),
        },
    };
    assert_eq!(admit(&body), Err(LowerRefusal::ChainJoinDisagrees));
    assert!(
        admit(rule(&c, "q")).is_ok(),
        "the uncorrupted plan is admitted"
    );
}

/// A scanned predicate that a rule also defines has tuples no fact lists.
/// Reading only its facts would under-answer silently: here the rule derives
/// r(1, 7), and s(7, 4) holds, so the true answer is {2, 7}, not {2}.
#[test]
fn refuses_a_scanned_relation_that_a_rule_derives() {
    let src = FIXTURE.replace(
        "pred q(u32).",
        "pred q(u32).\npred t(u32, u32).\nt(1, 7).\nr(X, Y) :- t(X, Y).",
    );
    let c = compile(&src);
    // The query still compiles to the admitted two-scan shape...
    assert!(admit(rule(&c, "q")).is_ok());
    // ...so the refusal has to come from the relation, not the plan.
    assert_eq!(
        facts(&c.program, "r"),
        Err(LowerRefusal::DerivedRelation {
            predicate: "r".to_string()
        })
    );
    assert!(matches!(
        prepare(&c),
        Err(LowerRefusal::DerivedRelation { .. })
    ));
    // A relation defined by facts only is still read.
    assert_eq!(facts(&c.program, "s").map(|v| v.len()), Ok(3));
}

#[test]
fn refuses_a_stale_frame() {
    let mut st = store();
    let g = guards(&st);
    st.advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    let ys = AddressMap::new(Y, [2]).unwrap();
    let rows = [0u64; SLOTS];
    let e = propagate(
        &st,
        &g,
        (FOCUS, SCOPE),
        (sig_r(), &rows),
        (sig_s(), &rows),
        Some(0),
        Some(0),
        &ys,
    );
    assert_eq!(e, Err(EvalRefusal::Bind(Refusal::SourceChanged)));
}

// ── Gate C: 64-slot boundary ───────────────────────────────────────────────

#[test]
fn domain_boundary_63_64_65_and_tail_bits() {
    let m63 = AddressMap::new(Y, 0..63).unwrap();
    assert_eq!(m63.valid() >> 63, 0, "slot 63 is not assigned at n = 63");
    let m64 = AddressMap::new(Y, 0..64).unwrap();
    assert_eq!(m64.valid(), u64::MAX);
    assert_eq!(
        AddressMap::new(Y, 0..65),
        Err(DomainRefusal::Overflow { needed: 65 })
    );
    // Non-contiguous source ids map to dense slots and back.
    let sparse = AddressMap::new(Y, [u32::MAX, 7, 1 << 31]).unwrap();
    assert_eq!(sparse.ids(0b111), vec![7, 1 << 31, u32::MAX]);
    assert_eq!(sparse.slot(8), Err(DomainRefusal::Unmapped(8)));

    // A stray bit past the domain in a register never leaks into a mask.
    let st = store();
    let g = guards(&st);
    let mut r = [0u64; SLOTS];
    let mut s = [0u64; SLOTS];
    r[0] = 1 | (1 << 63); // bit 63 is outside the 63-slot Y axis
    s[0] = 1;
    s[63] = 1;
    let p = propagate(
        &st,
        &g,
        (FOCUS, SCOPE),
        (sig_r(), &r),
        (sig_s(), &s),
        Some(0),
        Some(0),
        &m63,
    )
    .unwrap();
    assert_eq!(p.forward, 1);
    assert_eq!(p.backward, 1);
}

// ── the native path past one 64-slot tile ─────────────────────────────────

/// The register holds one 64-slot tile; the native path answers over DR's
/// 256-wide map with the same functions, and still agrees with scalar
/// enumeration.
#[test]
fn the_native_path_answers_past_one_tile() {
    let r: Vec<(u32, u32)> = (0..200).map(|y| (y % 3, y * 5)).collect();
    let s: Vec<(u32, u32)> = (0..200)
        .filter(|y| y % 7 != 0)
        .map(|y| (y * 5, y % 4))
        .collect();
    let ys: BTreeSet<u32> = r.iter().map(|t| t.1).chain(s.iter().map(|t| t.0)).collect();
    assert_eq!(
        AddressMap::new(Y, ys.iter().copied()),
        Err(DomainRefusal::Overflow { needed: 200 })
    );
    let globals: Vec<u64> = ys.iter().map(|&y| u64::from(y)).collect();
    let mid = LocalMap::prepare(Width::W256, &globals).unwrap();
    for (xb, zb) in [
        (None, None),
        (Some(1), None),
        (None, Some(2)),
        (Some(2), Some(3)),
    ] {
        let q = ChainQuery {
            left: RelId(0),
            left_bound: xb,
            right: RelId(1),
            right_bound: zb,
        };
        let got: BTreeSet<u32> =
            native::ids(&mid, &native::support(&q, &r, &s, &mid).unwrap().answer)
                .into_iter()
                .map(|g| u32::try_from(g).unwrap())
                .collect();
        let want = scalar(&r, &s, xb, zb);
        assert!(!want.is_empty(), "the case selects something");
        assert_eq!(got, want, "bounds {xb:?} {zb:?}");
    }
}

/// A Y outside DR's map is refused, never dropped.
#[test]
fn the_native_path_refuses_an_unmapped_y() {
    let mid = LocalMap::prepare(Width::W64, &[2, 3]).unwrap();
    let q = ChainQuery {
        left: RelId(0),
        left_bound: None,
        right: RelId(1),
        right_bound: None,
    };
    assert_eq!(
        native::support(&q, &[(1, 2), (1, 9)], &[(2, 4)], &mid).err(),
        Some(NativeRefusal::Domain(MapRefusal::Unmapped))
    );
}
