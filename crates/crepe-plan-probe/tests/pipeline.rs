//! Tests A–M of the spike brief. Every answer is compared against the
//! independent nested-loop reference over the UNOPTIMIZED rules, and where the
//! shape allows, against crepe's own evaluator, datafrog and MORK.

use std::collections::BTreeMap;

use crepe_plan_probe::logic::{Labels, RelId, Rule, resolve};
use crepe_plan_probe::mork_api::{Expr, MorkRefusal, QueryMulti, admit, query_multi, render};
use crepe_plan_probe::native::{
    NativeRefusal, SideKey, Store, distinct_centers, fingerprint, has_foreign_hop, joint, prepare,
    star,
};
use crepe_plan_probe::optimize::{Demand, Options, ResultKind, Rewrite};
use crepe_plan_probe::pipeline::{Outcome, PipelineRefusal, evaluate};
use crepe_plan_probe::plan::Plan;
use crepe_plan_probe::programs::{family, family_noisy, refused, star3, two_way};
use crepe_plan_probe::{logic, reference};
use lance_graph_mask_risc::{MaskOp, Operand, Program, Terminal};
use lance_graph_relalg::{Measure, Rel, normalize};
use mork_pathmap_probe::join::{JoinRefusal, JoinTable};
use mork_pathmap_probe::{Backend, JoinInput, RelationView, Semantics, bind, capability};

const SET: Semantics = Semantics::Set;
const BAG: Semantics = Semantics::Bag;

/// The family fixture. `Allowed(2)` is stored twice: a bag of facts, so bag
/// answers through `y = 2` are counted twice.
struct Family {
    parent: Vec<[u32; 2]>,
    allowed: Vec<[u32; 2]>,
    sponsor: Vec<[u32; 2]>,
}

fn family_data() -> Family {
    let mut parent = vec![
        [17, 1],
        [17, 2],
        [17, 3],
        [1, 10],
        [1, 11],
        [2, 10],
        [2, 12],
        [3, 13],
        [5, 1],
        [4, 2],
    ];
    // Unrelated families: they enlarge the shared domain, never the answer.
    for i in 0..200u32 {
        parent.push([1000 + i, 2000 + i]);
        parent.push([2000 + i, 3000 + i]);
    }
    Family {
        parent,
        allowed: vec![[1, 1], [2, 2], [2, 2], [2000, 2000]],
        sponsor: vec![[20, 20], [21, 21]],
    }
}

fn resolved(plan: &Plan) -> (Labels, Vec<Rule>) {
    let labels = Labels::of(plan);
    let rules = resolve(plan, &labels)
        .into_iter()
        .map(|r| r.expect("admitted rule"))
        .collect();
    (labels, rules)
}

fn store<'a>(labels: &Labels, f: &'a Family, gens: [u64; 3]) -> Store<'a> {
    Store {
        relations: vec![
            (labels.id("Parent").unwrap(), &f.parent, gens[0]),
            (labels.id("Allowed").unwrap(), &f.allowed, gens[1]),
            (labels.id("Sponsor").unwrap(), &f.sponsor, gens[2]),
        ],
    }
}

/// Raw rows of a relation at its declared arity, for the reference.
fn raw<'a>(labels: &'a Labels, store: &'a Store<'a>) -> impl Fn(RelId) -> Vec<Vec<u64>> + 'a {
    move |rel| {
        let (rows, _) = store.view(rel);
        rows.iter()
            .map(|r| {
                r[..labels.arity(rel)]
                    .iter()
                    .map(|&v| u64::from(v))
                    .collect()
            })
            .collect()
    }
}

fn answer_demand(labels: &Labels, x: u64) -> Demand {
    Demand {
        goal: labels.id("Answer").unwrap(),
        bound: vec![Some(x), None],
    }
}

fn reference_bag(
    labels: &Labels,
    rules: &[Rule],
    store: &Store<'_>,
    demand: &Demand,
) -> BTreeMap<Vec<u64>, u64> {
    reference::eval(rules, demand.goal, &demand.bound, &raw(labels, store))
}

fn as_set(bag: &BTreeMap<Vec<u64>, u64>) -> Vec<(Vec<u64>, u64)> {
    bag.keys().map(|k| (k.clone(), 1)).collect()
}

fn bindings(o: Outcome) -> Vec<(Vec<u64>, u64)> {
    match o {
        Outcome::Bindings(b) => b,
        other => panic!("expected bindings, got {other:?}"),
    }
}

/// crepe's own evaluator on the same rules, filtered to `x`.
fn crepe_oracle(f: &Family, x: u32) -> Vec<(Vec<u64>, u64)> {
    use family::oracle::{Allowed, Answer, Crepe, Parent, Sponsor};
    let mut rt = Crepe::new();
    rt.extend(f.parent.iter().map(|&[a, b]| Parent(a, b)));
    rt.extend(f.allowed.iter().map(|&[a, _]| Allowed(a)));
    rt.extend(f.sponsor.iter().map(|&[a, _]| Sponsor(a)));
    let (answers,) = rt.run();
    let mut out: Vec<(Vec<u64>, u64)> = answers
        .into_iter()
        .filter(|Answer(a, _)| *a == x)
        .map(|Answer(a, b)| (vec![u64::from(a), u64::from(b)], 1))
        .collect();
    out.sort();
    out
}

/// datafrog, for the family rule only (the sponsor branch never answers for
/// `x = 17`): `Answer(x, z) <- Parent(x, y), Parent(y, z), Allowed(y)`.
fn datafrog_family(f: &Family, x: u32) -> Vec<(Vec<u64>, u64)> {
    use datafrog::Relation;
    let by_y: Relation<(u32, u32)> = f.parent.iter().map(|&[a, b]| (b, a)).collect();
    let by_parent: Relation<(u32, u32)> = f.parent.iter().map(|&[a, b]| (a, b)).collect();
    let allowed: Relation<(u32, ())> = f.allowed.iter().map(|&[a, _]| (a, ())).collect();
    let xz: Relation<(u32, (u32, u32))> =
        Relation::from_join(&by_y, &by_parent, |&y, &x, &z| (y, (x, z)));
    let ans: Relation<(u32, u32)> = Relation::from_join(&xz, &allowed, |_, &(x, z), _| (x, z));
    ans.elements
        .iter()
        .filter(|(a, _)| *a == x)
        .map(|&(a, b)| (vec![u64::from(a), u64::from(b)], 1))
        .collect()
}

// ---------------------------------------------------------------------------
// A. Basic equivalence
// ---------------------------------------------------------------------------

#[test]
fn a_answers_equal_every_oracle() {
    let f = family_data();
    let (labels, rules) = resolved(&family::crepe_plan());
    let st = store(&labels, &f, [1, 1, 1]);
    let d = answer_demand(&labels, 17);
    let want = as_set(&reference_bag(&labels, &rules, &st, &d));
    assert_eq!(
        want,
        vec![(vec![17, 10], 1), (vec![17, 11], 1), (vec![17, 12], 1)],
        "fixture answer"
    );
    for opts in [Options::ALL, Options::NONE] {
        let (o, _) = evaluate(&rules, &d, ResultKind::Bindings(SET), opts, &st).unwrap();
        assert_eq!(bindings(o), want, "native, {opts:?}");
    }
    assert_eq!(crepe_oracle(&f, 17), want, "unmodified crepe!");
    assert_eq!(datafrog_family(&f, 17), want, "datafrog");
}

// ---------------------------------------------------------------------------
// B. Binding propagation
// ---------------------------------------------------------------------------

#[test]
fn b_the_bound_constant_reaches_the_body_and_the_dead_branch_is_dropped() {
    let f = family_data();
    let (labels, rules) = resolved(&family::crepe_plan());
    let st = store(&labels, &f, [1, 1, 1]);
    let d = answer_demand(&labels, 17);
    let (on, rep_on) = evaluate(&rules, &d, ResultKind::Bindings(SET), Options::ALL, &st).unwrap();
    let (off, rep_off) =
        evaluate(&rules, &d, ResultKind::Bindings(SET), Options::NONE, &st).unwrap();

    // Same answer.
    assert_eq!(on, off);
    // Plan with propagation: one rule, the constant in its first pattern, the
    // sponsor branch dropped for its head constant 5.
    assert!(
        rep_on
            .optimized
            .rewrites
            .contains(&Rewrite::DroppedRule { rule: 1, pos: 0 })
    );
    assert!(rep_on.optimized.rewrites.contains(&Rewrite::BoundHeadVar {
        rule: 0,
        pos: 0,
        value: 17
    }));
    assert_eq!(rep_on.optimized.rules.len(), 1);
    let text = render(&rep_on.patterns[0], |r| labels.name(r));
    assert_eq!(
        text,
        "(, (Parent 17 $1) (Parent $1 $2) (Allowed $1)) -> (Answer 17 $2)"
    );
    assert_eq!(rep_on.stars[0].sides[0].key.restrict, Some(17));
    assert_eq!(rep_on.optimized.post_filter, vec![None, None]);
    // Without it: both rules evaluated, the demand applied to the output.
    assert_eq!(rep_off.optimized.rules.len(), 2);
    assert_eq!(rep_off.optimized.post_filter, vec![Some(17), None]);
    assert_eq!(rep_off.stars[0].sides[0].key.restrict, None);
    // The plan shape is cheaper with it: fewer rows laid out, fewer programs.
    assert!(
        rep_on.rows_prepared < rep_off.rows_prepared,
        "{} vs {}",
        rep_on.rows_prepared,
        rep_off.rows_prepared
    );
    assert!(rep_on.programs_run < rep_off.programs_run);
    // Each rule runs exactly one mask-risc program, even when bindings are read.
    for rep in [&rep_on, &rep_off] {
        assert_eq!(rep.programs_run, rep.optimized.rules.len());
        assert_eq!(rep.programs.len(), rep.optimized.rules.len());
    }
}

/// The other demand keeps the sponsor branch and specializes the family rule
/// to `x = 5`.
#[test]
fn b_a_matching_head_constant_keeps_its_branch() {
    let f = family_data();
    let (labels, rules) = resolved(&family::crepe_plan());
    let st = store(&labels, &f, [1, 1, 1]);
    let d = answer_demand(&labels, 5);
    let (o, rep) = evaluate(&rules, &d, ResultKind::Bindings(SET), Options::ALL, &st).unwrap();
    assert_eq!(rep.optimized.rules.len(), 2);
    assert_eq!(
        bindings(o),
        as_set(&reference_bag(&labels, &rules, &st, &d))
    );
    assert_eq!(
        crepe_oracle(&f, 5),
        as_set(&reference_bag(&labels, &rules, &st, &d))
    );
}

// ---------------------------------------------------------------------------
// C. Algebraic elimination, gated on the semantics
// ---------------------------------------------------------------------------

#[test]
fn c_duplicate_atoms_and_rules_are_removed_only_for_set_answers() {
    let f = family_data();
    let (labels, rules) = resolved(&family_noisy::crepe_plan());
    let st = store(&labels, &f, [1, 1, 1]);
    let d = answer_demand(&labels, 17);

    let (set, rep_set) =
        evaluate(&rules, &d, ResultKind::Bindings(SET), Options::ALL, &st).unwrap();
    assert!(
        rep_set
            .optimized
            .rewrites
            .contains(&Rewrite::DuplicateAtom { rule: 0, atom: 3 })
    );
    assert!(
        rep_set
            .optimized
            .rewrites
            .contains(&Rewrite::DuplicateRule { rule: 1 })
    );
    assert_eq!(
        rep_set.optimized.rules.len(),
        1,
        "rule 1 equals rule 0; rule 2 is dead"
    );
    assert_eq!(
        bindings(set),
        as_set(&reference_bag(&labels, &rules, &st, &d))
    );

    // Under bag semantics both rewrites would change the counts, so neither runs.
    let (bag, rep_bag) =
        evaluate(&rules, &d, ResultKind::Bindings(BAG), Options::ALL, &st).unwrap();
    assert!(!rep_bag.optimized.rewrites.iter().any(|r| matches!(
        r,
        Rewrite::DuplicateAtom { .. } | Rewrite::DuplicateRule { .. }
    )));
    let want = reference_bag(&labels, &rules, &st, &d)
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(bindings(bag), want);
    // The duplicate Allowed(b) squares the Allowed(2) multiplicity: the counts
    // really are different from the set.
    assert!(want.iter().any(|(_, m)| *m > 1));
}

// ---------------------------------------------------------------------------
// D. Coordinate identity
// ---------------------------------------------------------------------------

#[test]
fn d_a_side_from_another_domain_is_not_anded() {
    let f = family_data();
    let (labels, rules) = resolved(&family::crepe_plan());
    let st = store(&labels, &f, [1, 1, 1]);
    let q = query_multi(&rules[0]);
    let plan = star(&q).unwrap();
    let mut prep = prepare(&plan, &st);

    // Same schema, other ordinal domain: Parent keyed by column 0 over a
    // domain built from a different relation.
    let other_rows: Vec<[u32; 2]> = (0..50).map(|i| [i * 7, i]).collect();
    let other = JoinTable::prepare_star(&[(
        RelationView {
            rows: &other_rows,
            generation: 1,
        },
        0,
    )]);
    let foreign = other.sides[0].clone();
    let key = prep.keys[1];
    assert_eq!(
        prep.graft(1, foreign, key, fingerprint(&other)),
        Err(NativeRefusal::DomainMismatch { side: 1 })
    );

    // A table prepared for another plan's sides is no witness for this one.
    let other_plan = star(&query_multi(&rules[1])).unwrap();
    assert_eq!(
        joint(&prep, &other_plan, &[1], Measure::Count, None).map(|_| ()),
        Err(NativeRefusal::NoCoordinateWitness)
    );
}

// ---------------------------------------------------------------------------
// E. Generation staleness
// ---------------------------------------------------------------------------

#[test]
fn e_a_stale_generation_and_a_value_outside_the_domain_are_refused() {
    let f = family_data();
    let (labels, rules) = resolved(&family::crepe_plan());
    let st = store(&labels, &f, [1, 1, 1]);
    let plan = star(&query_multi(&rules[0])).unwrap();
    let mut prep = prepare(&plan, &st);
    assert!(distinct_centers(&prep, &plan, &[1, 1, 1]).is_ok());
    // One generation per side: a short vector does not skip the check.
    for short in [&[][..], &[1, 1][..]] {
        assert_eq!(
            distinct_centers(&prep, &plan, short),
            Err(NativeRefusal::Generations {
                sides: 3,
                given: short.len()
            })
        );
    }
    assert_eq!(
        distinct_centers(&prep, &plan, &[1, 2, 1]),
        Err(NativeRefusal::Table(JoinRefusal::Stale {
            side: 1,
            bound: 1,
            current: 2
        }))
    );
    // A reseal of Allowed that names a y the domain never had.
    let resealed: Vec<[u32; 2]> = vec![[1, 1], [999_999, 999_999]];
    let r = prep.table.replace_side(
        2,
        RelationView {
            rows: &resealed,
            generation: 2,
        },
        0,
    );
    assert_eq!(
        r,
        Err(JoinRefusal::OutsideDomain {
            side: 2,
            y: 999_999
        })
    );
}

// ---------------------------------------------------------------------------
// F. Bag multiplicity
// ---------------------------------------------------------------------------

#[test]
fn f_set_and_bag_answers_are_not_confused() {
    let f = family_data();
    let (labels, rules) = resolved(&family::crepe_plan());
    let st = store(&labels, &f, [1, 1, 1]);
    let d = answer_demand(&labels, 17);
    let bag = reference_bag(&labels, &rules, &st, &d);
    let derivations: u64 = bag.values().sum();
    // (17,10) through y = 1 once and through y = 2 twice (Allowed(2) twice):
    // 3 derivations; (17,11) 1; (17,12) 2.
    assert_eq!(
        bag.into_iter().collect::<Vec<_>>(),
        vec![(vec![17, 10], 3), (vec![17, 11], 1), (vec![17, 12], 2)]
    );
    for opts in [Options::ALL, Options::NONE] {
        let count = |k| evaluate(&rules, &d, k, opts, &st).unwrap().0;
        assert_eq!(count(ResultKind::Count(BAG)), Outcome::Count(derivations));
        assert_eq!(count(ResultKind::Count(SET)), Outcome::Count(3));
        assert_eq!(count(ResultKind::Exists), Outcome::Exists(true));
    }
    // MORK stores sets: asked for a bag it refuses, never answers 1s.
    assert!(capability(Backend::Mork, BAG).is_err());
    assert!(capability(Backend::PathMapZipper, BAG).is_err());
}

// ---------------------------------------------------------------------------
// G. Unsupported patterns are refused by name
// ---------------------------------------------------------------------------

#[test]
fn g_unsupported_shapes_are_refused_not_flattened() {
    let plan = refused::crepe_plan();
    let labels = Labels::of(&plan);
    let resolved = resolve(&plan, &labels);
    let by_goal = |name: &str| -> Vec<_> {
        plan.rules
            .iter()
            .zip(&resolved)
            .filter(|(r, _)| r.goal.relation == name)
            .map(|(_, res)| res.clone())
            .collect()
    };
    assert_eq!(by_goal("Neg"), vec![Err(logic::Refusal::Negation)]);
    let reach = by_goal("Reach");
    assert!(reach[0].is_ok());
    assert_eq!(reach[1], Err(logic::Refusal::Recursive));

    let pattern = |name: &str| query_multi(by_goal(name)[0].as_ref().unwrap());
    assert!(matches!(
        admit(&pattern("Loop")),
        Err(MorkRefusal::RepeatedVarInAtom { pattern: 0, .. })
    ));
    assert_eq!(admit(&pattern("Tri")), Err(MorkRefusal::Cyclic));
    // A chain is acyclic, so MORK's shape admits it; the native star does not.
    let chain = pattern("Chain");
    assert_eq!(admit(&chain), Ok(()));
    assert!(matches!(star(&chain), Err(NativeRefusal::NoSharedVariable)));

    // A nested sub-expression is refused, never flattened.
    let mut nested: QueryMulti = pattern("Chain");
    nested.patterns[0][2] = Expr::Nested(vec![Expr::Int(1), Expr::Int(2)]);
    assert_eq!(
        admit(&nested),
        Err(MorkRefusal::NestedPattern { pattern: 0 })
    );
}

/// The pipeline surfaces a refusal; it does not answer.
#[test]
fn g_the_pipeline_refuses_a_chain() {
    let plan = refused::crepe_plan();
    let labels = Labels::of(&plan);
    let rules: Vec<Rule> = resolve(&plan, &labels)
        .into_iter()
        .filter_map(Result::ok)
        .collect();
    let e = labels.id("E").unwrap();
    let rows: Vec<[u32; 2]> = vec![[1, 2], [2, 3], [3, 4]];
    let st = Store {
        relations: vec![(e, &rows, 1), (labels.id("B").unwrap(), &rows, 1)],
    };
    let d = Demand {
        goal: labels.id("Chain").unwrap(),
        bound: vec![None, None],
    };
    let r = evaluate(&rules, &d, ResultKind::Bindings(SET), Options::ALL, &st);
    assert!(matches!(r, Err(PipelineRefusal::Native { .. })));
}

// ---------------------------------------------------------------------------
// H. No forbidden physical operation
// ---------------------------------------------------------------------------

#[test]
fn h_the_programs_have_no_foreign_hop() {
    let f = family_data();
    let (labels, rules) = resolved(&family::crepe_plan());
    let st = store(&labels, &f, [1, 1, 1]);
    for kind in [
        ResultKind::Exists,
        ResultKind::Count(BAG),
        ResultKind::Bindings(SET),
    ] {
        for d in [answer_demand(&labels, 17), answer_demand(&labels, 5)] {
            let (_, rep) = evaluate(&rules, &d, kind, Options::ALL, &st).unwrap();
            assert!(!rep.programs.is_empty(), "a program ran");
            for p in &rep.programs {
                assert!(!has_foreign_hop(p), "{p:?}");
                assert!(
                    p.ops.iter().all(|op| matches!(op, MaskOp::And { .. })),
                    "only plane ANDs: {p:?}"
                );
            }
        }
    }
    // The detector can fire.
    let gather = Program::new(
        vec![MaskOp::Gather {
            lane: 0,
            foreign: 0,
            under: None,
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    assert!(has_foreign_hop(&gather));
}

// ---------------------------------------------------------------------------
// I. Common-coordinate star
// ---------------------------------------------------------------------------

#[test]
fn i_the_star_mask_is_the_intersection_of_all_three_planes() {
    let plan = star3::crepe_plan();
    let (labels, rules) = resolved(&plan);
    let r: Vec<[u32; 2]> = (0..300).map(|v| [v, v]).collect();
    let s: Vec<[u32; 2]> = (0..300).filter(|v| v % 2 == 0).map(|v| [v, v]).collect();
    let q: Vec<[u32; 2]> = (0..300).filter(|v| v % 3 == 0).map(|v| [v, v]).collect();
    let st = Store {
        relations: vec![
            (labels.id("R").unwrap(), &r, 1),
            (labels.id("S").unwrap(), &s, 1),
            (labels.id("Q").unwrap(), &q, 1),
        ],
    };
    let sp = star(&query_multi(&rules[0])).unwrap();
    assert_eq!(sp.sides.len(), 3);
    let prep = prepare(&sp, &st);
    let n = distinct_centers(&prep, &sp, &[1, 1, 1]).unwrap();
    let all_three = (0..300).filter(|v| v % 6 == 0).count() as u64;
    let two_only = (0..300).filter(|v| v % 2 == 0).count() as u64;
    assert_eq!(n, all_three);
    assert_ne!(n, two_only, "the third plane must restrict");
}

// ---------------------------------------------------------------------------
// J. Reference parity, including MORK
// ---------------------------------------------------------------------------

#[test]
fn j_two_way_join_equals_mork_and_the_reference() {
    let plan = two_way::crepe_plan();
    let (labels, rules) = resolved(&plan);
    let r: Vec<[u32; 2]> = (0..120u32).map(|i| [i, i % 17]).collect();
    let s: Vec<[u32; 2]> = (0..90u32).map(|i| [i % 23, 500 + i]).collect();
    let st = Store {
        relations: vec![
            (labels.id("R").unwrap(), &r, 1),
            (labels.id("S").unwrap(), &s, 1),
        ],
    };
    let d = Demand {
        goal: labels.id("T").unwrap(),
        bound: vec![None, None, None],
    };
    let want_bag = reference::eval(&rules, d.goal, &d.bound, &raw(&labels, &st));
    let (set, _) = evaluate(&rules, &d, ResultKind::Bindings(SET), Options::ALL, &st).unwrap();
    let (bag, _) = evaluate(&rules, &d, ResultKind::Bindings(BAG), Options::ALL, &st).unwrap();
    assert_eq!(bindings(set.clone()), as_set(&want_bag));
    assert_eq!(
        bindings(bag),
        want_bag.clone().into_iter().collect::<Vec<_>>()
    );

    let input = JoinInput {
        r: RelationView {
            rows: &r,
            generation: 1,
        },
        s: RelationView {
            rows: &s,
            generation: 1,
        },
    };
    let mut mork = bind(Backend::Mork, &input, SET).expect("MORK builds");
    let mork_rows: Vec<(Vec<u64>, u64)> = mork
        .answers(&input)
        .expect("MORK answers")
        .rows
        .keys()
        .map(|t| (t.iter().map(|&v| u64::from(v)).collect(), 1))
        .collect();
    assert_eq!(bindings(set), mork_rows, "MORK");
}

// ---------------------------------------------------------------------------
// K. No runtime container in the optimized path
// ---------------------------------------------------------------------------

#[test]
fn k_the_native_path_names_no_crepe_runtime_container() {
    for (name, src) in [
        ("native.rs", include_str!("../src/native.rs")),
        ("pipeline.rs", include_str!("../src/pipeline.rs")),
        ("optimize.rs", include_str!("../src/optimize.rs")),
        ("mork_api.rs", include_str!("../src/mork_api.rs")),
    ] {
        for banned in ["HashMap", "HashSet", "crepe!", "Crepe::", "oracle::"] {
            assert!(!src.contains(banned), "{name} names {banned}");
        }
    }
}

// ---------------------------------------------------------------------------
// L. Deterministic lowering
// ---------------------------------------------------------------------------

#[test]
fn l_renamed_and_noisy_rules_give_the_same_plan() {
    let f = family_data();
    let (la, ra) = resolved(&family::crepe_plan());
    let (lb, rb) = resolved(&family_noisy::crepe_plan());
    let sa = store(&la, &f, [1, 1, 1]);
    let sb = store(&lb, &f, [1, 1, 1]);
    let (_, a) = evaluate(
        &ra,
        &answer_demand(&la, 17),
        ResultKind::Bindings(SET),
        Options::ALL,
        &sa,
    )
    .unwrap();
    let (_, b) = evaluate(
        &rb,
        &answer_demand(&lb, 17),
        ResultKind::Bindings(SET),
        Options::ALL,
        &sb,
    )
    .unwrap();
    // Both label tables sort the same names, so the ids agree.
    assert_eq!(la.id("Parent"), lb.id("Parent"));
    let canon = |r: &crepe_plan_probe::pipeline::Report| -> Vec<Rule> {
        r.optimized
            .rules
            .iter()
            .map(|(_, r)| r.canonical())
            .collect()
    };
    // The noisy rule's atoms are in another order; compare as sets of atoms.
    let atoms = |rs: Vec<Rule>| -> Vec<Vec<logic::Atom>> {
        rs.into_iter()
            .map(|r| {
                let mut b = r.body;
                b.sort();
                b
            })
            .collect()
    };
    assert_eq!(atoms(canon(&a)).len(), atoms(canon(&b)).len());
    assert_eq!(a.programs, b.programs, "same mask-risc programs");
    assert_eq!(a.stars.len(), b.stars.len());
}

// ---------------------------------------------------------------------------
// M. The H14d absorption case
// ---------------------------------------------------------------------------

/// `normalize` still misses `¬(x ∧ (x ∨ y)) = ¬x` (H14d, 2026-10-16) — but
/// only when `x` is itself an `And` or `Or`. Measured here: with `x` a plane or
/// a comparison it reduces correctly; with `x = p0 ∧ p2` De Morgan and
/// flattening run first and the absorption no longer matches. A relalg defect,
/// pinned here, not fixed in the optimizer. This spike never builds the shape:
/// it only emits an `And` of planes (no `Not`, no `Or`).
#[test]
fn m_the_known_absorption_gap_is_untouched_and_not_produced() {
    let absorbed = |x: Rel| {
        let e = Rel::Not(Box::new(Rel::And(vec![
            x.clone(),
            Rel::Or(vec![x.clone(), Rel::Plane(1)]),
        ])));
        normalize(&e) == normalize(&Rel::Not(Box::new(x)))
    };
    assert!(absorbed(Rel::Plane(0)), "leaf x reduces");
    assert!(
        !absorbed(Rel::And(vec![Rel::Plane(0), Rel::Plane(2)])),
        "compound x: fixed upstream? update this pin"
    );
    assert!(
        !absorbed(Rel::Or(vec![Rel::Plane(0), Rel::Plane(2)])),
        "compound x: fixed upstream? update this pin"
    );
    // What this spike produces: duplicates collapse, nothing widens.
    let ours = Rel::And(vec![Rel::Plane(0), Rel::Plane(1), Rel::Plane(0)]);
    assert_eq!(
        normalize(&ours),
        Rel::And(vec![Rel::Plane(0), Rel::Plane(1)])
    );
}

#[test]
fn keys_name_relation_and_column() {
    // A side key is equality on all four fields; two sides of the same
    // relation with other key columns are different coordinates.
    let a = SideKey {
        rel: RelId(0),
        key_col: 1,
        restrict: None,
        unary: false,
    };
    let b = SideKey { key_col: 0, ..a };
    assert_ne!(a, b);
}

// ---------------------------------------------------------------------------
// N. Semijoin reduction before preparation
// ---------------------------------------------------------------------------

const NO_SEMIJOIN: Options = Options {
    semijoin: false,
    ..Options::ALL
};

/// A restricted side names the only centers that can answer; the other sides
/// are laid out for those centers only. Same answers, set and bag.
#[test]
fn n_semijoin_prepares_only_the_restricted_centers() {
    let f = family_data();
    let (labels, rules) = resolved(&family::crepe_plan());
    let st = store(&labels, &f, [1, 1, 1]);
    let d = answer_demand(&labels, 17);
    for kind in [
        ResultKind::Bindings(SET),
        ResultKind::Bindings(BAG),
        ResultKind::Count(BAG),
        ResultKind::Exists,
    ] {
        let (on, rep_on) = evaluate(&rules, &d, kind, Options::ALL, &st).unwrap();
        let (off, rep_off) = evaluate(&rules, &d, kind, NO_SEMIJOIN, &st).unwrap();
        assert_eq!(on, off, "{kind:?}");
        // Parent(17, y) has y in {1, 2, 3}: 3 rows; Parent(y, z) for those y:
        // 5 rows; Allowed(y): 3 rows. Without the reduction every Parent and
        // Allowed row is laid out.
        assert_eq!(rep_on.rows_prepared, 3 + 5 + 3, "{kind:?}");
        assert_eq!(
            rep_off.rows_prepared,
            3 + f.parent.len() + f.allowed.len(),
            "{kind:?}"
        );
    }
    let want = reference_bag(&labels, &rules, &st, &d);
    let (bag, _) = evaluate(&rules, &d, ResultKind::Bindings(BAG), Options::ALL, &st).unwrap();
    assert_eq!(bindings(bag), want.into_iter().collect::<Vec<_>>());
}

/// No restricted side: nothing to reduce by, so the layout is unchanged.
#[test]
fn n_semijoin_without_a_restricted_side_changes_nothing() {
    let f = family_data();
    let (labels, rules) = resolved(&family::crepe_plan());
    let st = store(&labels, &f, [1, 1, 1]);
    let d = answer_demand(&labels, 17);
    let only_semijoin = Options {
        semijoin: true,
        ..Options::NONE
    };
    let kind = ResultKind::Bindings(SET);
    let (a, ra) = evaluate(&rules, &d, kind, only_semijoin, &st).unwrap();
    let (b, rb) = evaluate(&rules, &d, kind, Options::NONE, &st).unwrap();
    assert_eq!(a, b);
    assert_eq!(ra.rows_prepared, rb.rows_prepared);
}

/// A restricted side with no rows empties the star; the reference agrees.
#[test]
fn n_semijoin_with_an_empty_restricted_side_answers_nothing() {
    let f = family_data();
    let (labels, rules) = resolved(&family::crepe_plan());
    let st = store(&labels, &f, [1, 1, 1]);
    let d = answer_demand(&labels, 999);
    let (on, rep) = evaluate(&rules, &d, ResultKind::Bindings(BAG), Options::ALL, &st).unwrap();
    assert_eq!(bindings(on), Vec::<(Vec<u64>, u64)>::new());
    assert!(reference_bag(&labels, &rules, &st, &d).is_empty());
    assert_eq!(rep.rows_prepared, 0);
}

/// Two restricted sides: only the centers both have survive.
#[test]
fn n_semijoin_intersects_two_restricted_sides() {
    let plan = two_way::crepe_plan();
    let (labels, rules) = resolved(&plan);
    // R(1, y): y in {10, 10, 11, 12}; S(y, 9): y in {10, 11, 13}.
    let mut r: Vec<[u32; 2]> = vec![[1, 10], [1, 10], [1, 11], [1, 12], [2, 10]];
    let mut s: Vec<[u32; 2]> = vec![[10, 9], [11, 9], [11, 8], [12, 7], [13, 9]];
    for i in 0..50u32 {
        r.push([100 + i, 200 + i]);
        s.push([200 + i, 300 + i]);
    }
    let st = Store {
        relations: vec![
            (labels.id("R").unwrap(), &r, 1),
            (labels.id("S").unwrap(), &s, 1),
        ],
    };
    let d = Demand {
        goal: labels.id("T").unwrap(),
        bound: vec![Some(1), None, Some(9)],
    };
    let want = reference::eval(&rules, d.goal, &d.bound, &raw(&labels, &st));
    assert_eq!(
        want.clone().into_iter().collect::<Vec<_>>(),
        vec![(vec![1, 10, 9], 2), (vec![1, 11, 9], 1)],
        "fixture answer"
    );
    let kind = ResultKind::Bindings(BAG);
    let (on, rep_on) = evaluate(&rules, &d, kind, Options::ALL, &st).unwrap();
    let (off, rep_off) = evaluate(&rules, &d, kind, NO_SEMIJOIN, &st).unwrap();
    assert_eq!(bindings(on.clone()), want.into_iter().collect::<Vec<_>>());
    assert_eq!(on, off);
    // y = 12 and y = 13 are each on one side only: dropped before layout.
    assert_eq!(rep_on.rows_prepared, 3 + 2);
    assert_eq!(rep_off.rows_prepared, 4 + 3);
}
