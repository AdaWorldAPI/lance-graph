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

/// `¬(x ∧ (x ∨ y)) = ¬x` (H14d, 2026-10-16). Until D-CMR-1 `normalize` missed
/// it when `x` was itself an `And` or `Or`: De Morgan and flattening ran first
/// and absorption no longer matched. Fixed upstream in
/// `lance_graph_contract::set_algebra::normalize` (absorption against
/// flattened siblings and sub-junctions), which relalg now delegates to; this
/// pin flipped with it. Not fixed in the optimizer. This spike never builds the
/// shape: it only emits an `And` of planes (no `Not`, no `Or`).
#[test]
fn m_the_known_absorption_gap_is_closed_and_not_produced() {
    let absorbed = |x: Rel| {
        let e = Rel::Not(Box::new(Rel::And(vec![
            x.clone(),
            Rel::Or(vec![x.clone(), Rel::Plane(1)]),
        ])));
        normalize(&e) == normalize(&Rel::Not(Box::new(x)))
    };
    assert!(absorbed(Rel::Plane(0)), "leaf x reduces");
    assert!(
        absorbed(Rel::And(vec![Rel::Plane(0), Rel::Plane(2)])),
        "compound x (And) must reduce since D-CMR-1"
    );
    assert!(
        absorbed(Rel::Or(vec![Rel::Plane(0), Rel::Plane(2)])),
        "compound x (Or) must reduce since D-CMR-1"
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

// ---------------------------------------------------------------------------
// S. Stage 3 over Lance 13 (plan §2d, predictions P1–P4)
// ---------------------------------------------------------------------------

mod stage3 {
    use super::*;
    use crepe_plan_probe::lance_store::{LanceStore, SideIo, membership_mask, prepare_lance};
    use crepe_plan_probe::pipeline::evaluate_lance;

    async fn lance_family(labels: &Labels, f: &Family) -> (tempfile::TempDir, LanceStore) {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap().to_string();
        let st = LanceStore::write(
            &uri,
            &[
                (labels.id("Parent").unwrap(), "parent", &f.parent),
                (labels.id("Allowed").unwrap(), "allowed", &f.allowed),
                (labels.id("Sponsor").unwrap(), "sponsor", &f.sponsor),
            ],
        )
        .await
        .unwrap();
        (dir, st)
    }

    fn kinds() -> [ResultKind; 5] {
        [
            ResultKind::Bindings(SET),
            ResultKind::Bindings(BAG),
            ResultKind::Count(SET),
            ResultKind::Count(BAG),
            ResultKind::Exists,
        ]
    }

    /// P1: the Lance path answers what the in-memory path and the reference
    /// answer, for every option set and result kind, on both demands.
    #[tokio::test(flavor = "multi_thread")]
    async fn p1_lance_answers_equal_the_memory_path_and_the_reference() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let mem = store(&labels, &f, [1, 1, 1]);
        let (_dir, lance) = lance_family(&labels, &f).await;
        for x in [17, 5, 999] {
            let d = answer_demand(&labels, x);
            let want = reference_bag(&labels, &rules, &mem, &d);
            for opts in [Options::ALL, NO_SEMIJOIN, Options::NONE] {
                for kind in kinds() {
                    let (m, _) = evaluate(&rules, &d, kind, opts, &mem).unwrap();
                    let (l, _, _) = evaluate_lance(&rules, &d, kind, opts, &lance)
                        .await
                        .unwrap();
                    assert_eq!(l, m, "x={x} {opts:?} {kind:?}");
                }
                let (bag, _, _) =
                    evaluate_lance(&rules, &d, ResultKind::Bindings(BAG), opts, &lance)
                        .await
                        .unwrap();
                assert_eq!(bindings(bag), want.clone().into_iter().collect::<Vec<_>>());
            }
        }
    }

    /// P2: every masked read scans and returns exactly `|R|` rows.
    #[tokio::test(flavor = "multi_thread")]
    async fn p2_every_masked_read_scans_exactly_the_mask() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let (_dir, lance) = lance_family(&labels, &f).await;
        let d = answer_demand(&labels, 17);
        let (_, _, io) =
            evaluate_lance(&rules, &d, ResultKind::Bindings(SET), Options::ALL, &lance)
                .await
                .unwrap();
        let masked: Vec<(usize, &SideIo)> = io
            .iter()
            .flatten()
            .filter_map(|s| s.mask_rows.map(|r| (r, s)))
            .collect();
        // Parent(17, y), Parent(y, z) and Allowed(y) are all read under a mask.
        assert_eq!(masked.len(), 3, "{io:?}");
        for (r, s) in &masked {
            assert_eq!(s.rows.rows_scanned, *r, "{s:?}");
            assert_eq!(s.rows.rows_returned, *r, "{s:?}");
        }
        // Not vacuous: the masks are much smaller than the tables.
        let total: usize = masked.iter().map(|(r, _)| r).sum();
        assert_eq!(total, 3 + 5 + 3);
        assert!(
            masked
                .iter()
                .any(|(_, s)| s.mask_column.unwrap().rows_returned == f.parent.len())
        );
    }

    /// P3: the rows handed to `prepare_star` equal the in-memory semijoin's.
    #[tokio::test(flavor = "multi_thread")]
    async fn p3_prepared_rows_equal_the_memory_path() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let mem = store(&labels, &f, [1, 1, 1]);
        let (_dir, lance) = lance_family(&labels, &f).await;
        let d = answer_demand(&labels, 17);
        for opts in [Options::ALL, NO_SEMIJOIN] {
            let kind = ResultKind::Bindings(BAG);
            let (_, rm) = evaluate(&rules, &d, kind, opts, &mem).unwrap();
            let (_, rl, _) = evaluate_lance(&rules, &d, kind, opts, &lance)
                .await
                .unwrap();
            assert_eq!(rl.rows_prepared, rm.rows_prepared, "{opts:?}");
        }
        // Two restricted sides, intersected.
        let plan = two_way::crepe_plan();
        let (labels, rules) = resolved(&plan);
        let r: Vec<[u32; 2]> = vec![[1, 10], [1, 10], [1, 11], [1, 12], [2, 10], [7, 70]];
        let s: Vec<[u32; 2]> = vec![[10, 9], [11, 9], [11, 8], [12, 7], [13, 9], [70, 1]];
        let dir = tempfile::tempdir().unwrap();
        let lance = LanceStore::write(
            dir.path().to_str().unwrap(),
            &[
                (labels.id("R").unwrap(), "r", &r),
                (labels.id("S").unwrap(), "s", &s),
            ],
        )
        .await
        .unwrap();
        let mem = Store {
            relations: vec![
                (labels.id("R").unwrap(), &r, 1),
                (labels.id("S").unwrap(), &s, 1),
            ],
        };
        let d = Demand {
            goal: labels.id("T").unwrap(),
            bound: vec![Some(1), None, Some(9)],
        };
        let kind = ResultKind::Bindings(BAG);
        let (om, rm) = evaluate(&rules, &d, kind, Options::ALL, &mem).unwrap();
        let (ol, rl, _) = evaluate_lance(&rules, &d, kind, Options::ALL, &lance)
            .await
            .unwrap();
        assert_eq!(ol, om);
        assert_eq!((rl.rows_prepared, rm.rows_prepared), (5, 5));
    }

    /// P4: a write that adds a version makes a table prepared before it stale.
    #[tokio::test(flavor = "multi_thread")]
    async fn p4_an_append_makes_the_prepared_table_stale() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let (_dir, lance) = lance_family(&labels, &f).await;
        let d = answer_demand(&labels, 17);
        let planned =
            crepe_plan_probe::pipeline::plan(&rules, &d, ResultKind::Bindings(SET), Options::ALL)
                .unwrap();
        let star = &planned.stars[0];
        let (prep, _) = prepare_lance(star, &lance, true).await.unwrap();
        let before = lance.generations(star).await.unwrap();
        assert!(joint(&prep, star, &before, Measure::Count, None).is_ok());
        let parent = labels.id("Parent").unwrap();
        let v = lance.append(parent, &[[17, 4]]).await.unwrap();
        let after = lance.generations(star).await.unwrap();
        assert_eq!(after[0], v);
        assert!(matches!(
            joint(&prep, star, &after, Measure::Count, None),
            Err(NativeRefusal::Table(JoinRefusal::Stale { side: 0, .. }))
        ));
    }

    /// The semijoin mask is built through relalg for any number of centers,
    /// past the 4 that fit before relalg released consumed slots.
    #[test]
    fn the_membership_mask_is_exact_for_any_number_of_centers() {
        let values: Vec<u32> = (0..1_000u32).map(|i| (i * 37) % 500).collect();
        for k in [0usize, 1, 4, 5, 64, 200, 500] {
            let centers: Vec<u32> = (0..k as u32).map(|c| c * 3).collect();
            let got = membership_mask(&values, &centers).unwrap();
            for (i, v) in values.iter().enumerate() {
                let bit = got[i / 64] >> (i % 64) & 1 == 1;
                assert_eq!(bit, centers.contains(v), "k={k} i={i}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// B. Arm B: the addressed semijoin over attested lanes (plan §2e)
// ---------------------------------------------------------------------------

mod arm_b {
    use super::*;
    use crepe_plan_probe::addressed::{AddressedRefusal, AddressedRelation, attest, key};
    use crepe_plan_probe::addressed::{Lookup, Read, ReadPolicy};
    use crepe_plan_probe::lance_store::{LanceStore, SideIo};
    use crepe_plan_probe::pipeline::{
        PipelineRefusal, evaluate_addressed, evaluate_addressed_with, evaluate_lance,
    };
    use lance_graph_contract::ordered_lane::WitnessError;

    struct Fixture {
        _dir: tempfile::TempDir,
        lance: LanceStore,
        lanes: Vec<AddressedRelation>,
    }

    async fn fixture(labels: &Labels, f: &Family) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let rels = [
            (labels.id("Parent").unwrap(), "parent", f.parent.as_slice()),
            (
                labels.id("Allowed").unwrap(),
                "allowed",
                f.allowed.as_slice(),
            ),
            (
                labels.id("Sponsor").unwrap(),
                "sponsor",
                f.sponsor.as_slice(),
            ),
        ];
        let lance = LanceStore::write(dir.path().to_str().unwrap(), &rels)
            .await
            .unwrap();
        let mut lanes = Vec::new();
        for (rel, _, _) in rels {
            lanes.push(attest(&lance, rel).await.unwrap());
        }
        Fixture {
            _dir: dir,
            lance,
            lanes,
        }
    }

    fn kinds() -> [ResultKind; 5] {
        [
            ResultKind::Bindings(SET),
            ResultKind::Bindings(BAG),
            ResultKind::Count(SET),
            ResultKind::Count(BAG),
            ResultKind::Exists,
        ]
    }

    /// The written key order is the numeric order of the rows: `key(a, b)`
    /// compares as `(a, b)`.
    #[test]
    fn the_key_orders_as_its_pair() {
        let pairs = [
            [0u32, 0],
            [0, 1],
            [0, u32::MAX],
            [1, 0],
            [65_535, 7],
            [65_536, 0],
            [u32::MAX, 0],
        ];
        for w in pairs.windows(2) {
            assert!(
                key(w[0][0], w[0][1]).cmp_numeric_projection(&key(w[1][0], w[1][1]))
                    == std::cmp::Ordering::Less,
                "{w:?}"
            );
        }
    }

    /// B1: Arm B answers what the memory path, Arm A and the reference
    /// answer, for every result kind, on a hit, a sponsor branch and a miss.
    #[tokio::test(flavor = "multi_thread")]
    async fn b1_addressed_answers_equal_memory_arm_a_and_the_reference() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let mem = store(&labels, &f, [1, 1, 1]);
        let fx = fixture(&labels, &f).await;
        for x in [17, 5, 999] {
            let d = answer_demand(&labels, x);
            let want = reference_bag(&labels, &rules, &mem, &d);
            for kind in kinds() {
                let (m, _) = evaluate(&rules, &d, kind, Options::ALL, &mem).unwrap();
                let (a, _, _) = evaluate_lance(&rules, &d, kind, Options::ALL, &fx.lance)
                    .await
                    .unwrap();
                let (b, _, _, _) =
                    evaluate_addressed(&rules, &d, kind, Options::ALL, &fx.lance, &fx.lanes)
                        .await
                        .unwrap();
                assert_eq!(b, m, "x={x} {kind:?}");
                assert_eq!(b, a, "x={x} {kind:?}");
            }
            let (bag, _, _, _) = evaluate_addressed(
                &rules,
                &d,
                ResultKind::Bindings(BAG),
                Options::ALL,
                &fx.lance,
                &fx.lanes,
            )
            .await
            .unwrap();
            assert_eq!(bindings(bag), want.into_iter().collect::<Vec<_>>());
        }
    }

    /// B2: no hidden scan. Every side Arm B reads scans exactly the rows of
    /// its located ranges, and those are the rows Arm A keeps; Arm A also
    /// reads the whole mask column, Arm B reads no column at all.
    #[tokio::test(flavor = "multi_thread")]
    async fn b2_lance_scans_exactly_the_located_ranges() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let fx = fixture(&labels, &f).await;
        let d = answer_demand(&labels, 17);
        let kind = ResultKind::Bindings(SET);
        let (_, _, io_a) = evaluate_lance(&rules, &d, kind, Options::ALL, &fx.lance)
            .await
            .unwrap();
        let (_, _, io_b, lookups) =
            evaluate_addressed(&rules, &d, kind, Options::ALL, &fx.lance, &fx.lanes)
                .await
                .unwrap();
        let sides_b: Vec<&SideIo> = io_b.iter().flatten().collect();
        let looks: Vec<_> = lookups.iter().flatten().collect();
        assert_eq!(sides_b.len(), 3, "{io_b:?}");
        for (s, l) in sides_b.iter().zip(&looks) {
            assert!(s.mask_column.is_none(), "{s:?}");
            assert_eq!(s.rows.rows_scanned, l.rows_in_ranges, "{s:?} {l:?}");
            assert_eq!(s.rows.rows_returned, l.rows_in_ranges, "{s:?} {l:?}");
        }
        // Parent(17, y): 3 rows. Parent(y, z) for y in {1, 2, 3}: 5. Allowed(y): 3.
        let scanned_b: usize = sides_b.iter().map(|s| s.rows.rows_scanned).sum();
        assert_eq!(scanned_b, 3 + 5 + 3);
        let scanned_a: usize = io_a
            .iter()
            .flatten()
            .map(|s| s.rows.rows_scanned + s.mask_column.map_or(0, |c| c.rows_scanned))
            .sum();
        // Not vacuous: Arm A reads every mask column in full.
        assert_eq!(scanned_a, 3 + 5 + 3 + 2 * f.parent.len() + f.allowed.len());
        assert!(scanned_b * 10 < scanned_a, "{scanned_b} vs {scanned_a}");
    }

    /// B3: a demand that matches nothing reads nothing.
    #[tokio::test(flavor = "multi_thread")]
    async fn b3_a_miss_reads_no_rows() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let fx = fixture(&labels, &f).await;
        let d = answer_demand(&labels, 999);
        let (out, _, io, _) = evaluate_addressed(
            &rules,
            &d,
            ResultKind::Count(BAG),
            Options::ALL,
            &fx.lance,
            &fx.lanes,
        )
        .await
        .unwrap();
        assert_eq!(out, Outcome::Count(0));
        let scanned: usize = io.iter().flatten().map(|s| s.rows.rows_scanned).sum();
        assert_eq!(scanned, 0, "{io:?}");
    }

    /// B4: a high-degree center. Every row of the hub is read once, and the
    /// answer equals the memory path.
    #[tokio::test(flavor = "multi_thread")]
    async fn b4_a_hub_is_read_once_per_row() {
        let mut f = family_data();
        for h in 0..500u32 {
            f.parent.push([7, 10_000 + h]);
            f.parent.push([10_000 + h, 9]);
            f.allowed.push([10_000 + h, 10_000 + h]);
        }
        let (labels, rules) = resolved(&family::crepe_plan());
        let mem = store(&labels, &f, [1, 1, 1]);
        let fx = fixture(&labels, &f).await;
        let d = answer_demand(&labels, 7);
        let kind = ResultKind::Count(BAG);
        let (m, _) = evaluate(&rules, &d, kind, Options::ALL, &mem).unwrap();
        let (b, _, io, lookups) =
            evaluate_addressed(&rules, &d, kind, Options::ALL, &fx.lance, &fx.lanes)
                .await
                .unwrap();
        assert_eq!(b, m);
        assert_eq!(b, Outcome::Count(500));
        let scanned: usize = io.iter().flatten().map(|s| s.rows.rows_scanned).sum();
        assert_eq!(scanned, 3 * 500);
        // The two unrestricted sides each sought the 500 distinct centers.
        let sought: Vec<usize> = lookups.iter().flatten().map(|l| l.keys).collect();
        assert_eq!(sought, vec![1, 500, 500]);
    }

    /// B5: a demanded key repeated by the restricted side is sought once, and
    /// the bag multiplicity of the stored facts survives.
    #[tokio::test(flavor = "multi_thread")]
    async fn b5_a_repeated_center_is_sought_once_and_multiplicity_is_kept() {
        let mut f = family_data();
        // Parent(17, 2) twice: center 2 appears twice on the restricted side.
        f.parent.push([17, 2]);
        let (labels, rules) = resolved(&family::crepe_plan());
        let mem = store(&labels, &f, [1, 1, 1]);
        let fx = fixture(&labels, &f).await;
        let d = answer_demand(&labels, 17);
        let kind = ResultKind::Bindings(BAG);
        let (m, _) = evaluate(&rules, &d, kind, Options::ALL, &mem).unwrap();
        let (b, _, _, lookups) =
            evaluate_addressed(&rules, &d, kind, Options::ALL, &fx.lance, &fx.lanes)
                .await
                .unwrap();
        assert_eq!(b, m);
        let sought: Vec<usize> = lookups.iter().flatten().map(|l| l.keys).collect();
        assert_eq!(sought, vec![1, 3, 3], "centers {{1, 2, 3}}, each once");
        // Not vacuous: z = 10 comes once through y = 1 and 2 x 2 times through
        // y = 2 (Parent(17, 2) twice, Allowed(2) twice); z = 12 only through
        // y = 2. Without the duplicate fact this is 1 + 2 + 2 = 5.
        let Outcome::Bindings(rows) = b else { panic!() };
        let mult = |z: u64| rows.iter().find(|(t, _)| t[1] == z).map_or(0, |(_, n)| *n);
        assert_eq!((mult(10), mult(12)), (5, 4), "{rows:?}");
    }

    /// B6: a table written out of key order gets no witness.
    #[tokio::test(flavor = "multi_thread")]
    async fn b6_an_unsorted_table_is_refused_at_attest() {
        const R0: crepe_plan_probe::logic::RelId = crepe_plan_probe::logic::RelId(0);
        let dir = tempfile::tempdir().unwrap();
        let rows = [[5u32, 1], [1, 2], [3, 3]];
        let lance = LanceStore::write_as_given(dir.path().to_str().unwrap(), &[(R0, "r", &rows)])
            .await
            .unwrap();
        assert!(matches!(
            attest(&lance, R0).await,
            Err(AddressedRefusal::Witness(WitnessError::NotSorted {
                first_inversion_at: 1
            }))
        ));
        // Silence twin: the same rows in key order are attested.
        let dir2 = tempfile::tempdir().unwrap();
        let sorted = LanceStore::write(dir2.path().to_str().unwrap(), &[(R0, "r", &rows)])
            .await
            .unwrap();
        assert!(attest(&sorted, R0).await.is_ok());
    }

    /// B7: an append after attest makes the lane stale; re-attesting serves
    /// the new rows.
    #[tokio::test(flavor = "multi_thread")]
    async fn b7_an_append_makes_the_lane_stale() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let fx = fixture(&labels, &f).await;
        let d = answer_demand(&labels, 17);
        let kind = ResultKind::Count(BAG);
        let parent = labels.id("Parent").unwrap();
        fx.lance.append(parent, &[[999_999, 1]]).await.unwrap();
        let got = evaluate_addressed(&rules, &d, kind, Options::ALL, &fx.lance, &fx.lanes).await;
        assert!(
            matches!(&got, Err(PipelineRefusal::Addressed(s)) if s.starts_with("Stale")),
            "{got:?}"
        );
        // Re-attesting is refused too: the append landed in a second
        // fragment, so lane position is no longer row address. The arm needs
        // the table compacted (or rewritten) before it can address it again.
        assert!(matches!(
            attest(&fx.lance, parent).await,
            Err(AddressedRefusal::NotContiguous { rel }) if rel == parent
        ));
    }

    /// B8: a side whose restriction or demand is on the second column is
    /// refused by name; the lane is ordered by the first.
    #[tokio::test(flavor = "multi_thread")]
    async fn b8_a_demand_on_the_second_column_is_refused() {
        let plan = two_way::crepe_plan();
        let (labels, rules) = resolved(&plan);
        let r: Vec<[u32; 2]> = vec![[1, 10], [1, 11], [2, 10]];
        let s: Vec<[u32; 2]> = vec![[10, 9], [11, 8]];
        let dir = tempfile::tempdir().unwrap();
        let (rr, ss) = (labels.id("R").unwrap(), labels.id("S").unwrap());
        let lance = LanceStore::write(
            dir.path().to_str().unwrap(),
            &[(rr, "r", &r), (ss, "s", &s)],
        )
        .await
        .unwrap();
        let lanes = vec![
            attest(&lance, rr).await.unwrap(),
            attest(&lance, ss).await.unwrap(),
        ];
        let goal = labels.id("T").unwrap();
        let kind = ResultKind::Count(BAG);
        // S(y, 9): the constant is on S's second column.
        let d = Demand {
            goal,
            bound: vec![Some(1), None, Some(9)],
        };
        let got = evaluate_addressed(&rules, &d, kind, Options::ALL, &lance, &lanes).await;
        assert!(
            matches!(&got, Err(PipelineRefusal::Addressed(e)) if e.starts_with("NotLeadingColumn")),
            "{got:?}"
        );
        // Silence twin: the constant on R's first column only is served.
        let d = Demand {
            goal,
            bound: vec![Some(1), None, None],
        };
        let got = evaluate_addressed(&rules, &d, kind, Options::ALL, &lance, &lanes).await;
        assert_eq!(got.unwrap().0, Outcome::Count(2));
    }

    /// B10: a demand constant above `u32::MAX` matches nothing on every path
    /// (no stored value is that large), instead of being refused.
    #[tokio::test(flavor = "multi_thread")]
    async fn b10_a_constant_above_u32_matches_nothing() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let mem = store(&labels, &f, [1, 1, 1]);
        let fx = fixture(&labels, &f).await;
        let d = answer_demand(&labels, u64::from(u32::MAX) + 1);
        for kind in kinds() {
            let (m, _) = evaluate(&rules, &d, kind, Options::ALL, &mem).unwrap();
            let (a, _, _) = evaluate_lance(&rules, &d, kind, Options::ALL, &fx.lance)
                .await
                .unwrap();
            let (b, _, io, _) =
                evaluate_addressed(&rules, &d, kind, Options::ALL, &fx.lance, &fx.lanes)
                    .await
                    .unwrap();
            assert_eq!(a, m, "{kind:?}");
            assert_eq!(b, m, "{kind:?}");
            let scanned: usize = io.iter().flatten().map(|s| s.rows.rows_scanned).sum();
            assert_eq!(scanned, 0, "{kind:?}");
        }
        // Not vacuous: the memory path answers nothing here.
        let (m, _) = evaluate(&rules, &d, ResultKind::Count(BAG), Options::ALL, &mem).unwrap();
        assert_eq!(m, Outcome::Count(0));
    }

    /// B11: with nothing bound there are no centers, every side is read in
    /// full, and a side keyed by its second column is served.
    #[tokio::test(flavor = "multi_thread")]
    async fn b11_an_unbound_demand_reads_every_side_in_full() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let mem = store(&labels, &f, [1, 1, 1]);
        let fx = fixture(&labels, &f).await;
        let d = Demand {
            goal: labels.id("Answer").unwrap(),
            bound: vec![None, None],
        };
        for kind in kinds() {
            let (m, _) = evaluate(&rules, &d, kind, Options::ALL, &mem).unwrap();
            let (b, _, _, _) =
                evaluate_addressed(&rules, &d, kind, Options::ALL, &fx.lance, &fx.lanes)
                    .await
                    .unwrap();
            assert_eq!(b, m, "{kind:?}");
        }
        // Not vacuous: Parent(x, y) is keyed by its second column here.
        let planned =
            crepe_plan_probe::pipeline::plan(&rules, &d, ResultKind::Count(BAG), Options::ALL)
                .unwrap();
        assert!(
            planned
                .stars
                .iter()
                .any(|s| s.sides.iter().any(|side| side.key.key_col == 1))
        );
    }

    /// B12: a lane attested on one table is refused against another table
    /// with the same relation id and version.
    /// B13: every read policy answers what the exact read answers, for every
    /// result kind, on a hit, a sponsor branch and a miss. The bag kinds see
    /// the duplicate `Allowed(2, 2)` row, so a coalesced or full read that
    /// dropped or doubled a row would show.
    #[tokio::test(flavor = "multi_thread")]
    async fn b13_every_read_policy_answers_like_the_exact_read() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let fx = fixture(&labels, &f).await;
        let policies = [
            ReadPolicy::FULL,
            ReadPolicy::DEFAULT,
            ReadPolicy {
                coalesce_gap: 1,
                full_permille: u32::MAX,
            },
            ReadPolicy {
                coalesce_gap: u32::MAX,
                full_permille: u32::MAX,
            },
        ];
        for x in [17, 5, 999] {
            let d = answer_demand(&labels, x);
            for kind in kinds() {
                let (exact, _, _, _) = evaluate_addressed_with(
                    &rules,
                    &d,
                    kind,
                    Options::ALL,
                    &fx.lance,
                    &fx.lanes,
                    ReadPolicy::EXACT,
                )
                .await
                .unwrap();
                for p in policies {
                    let (got, _, _, _) = evaluate_addressed_with(
                        &rules,
                        &d,
                        kind,
                        Options::ALL,
                        &fx.lance,
                        &fx.lanes,
                        p,
                    )
                    .await
                    .unwrap();
                    assert_eq!(got, exact, "x={x} {kind:?} {p:?}");
                }
            }
        }
    }

    /// B14: the switch can fire and can stay silent on the same demand. The
    /// centers of `x = 17` cover a few rows of a 400-row `Parent`: below a
    /// 100-permille threshold the side reads ranges; at 1 permille it reads
    /// the table in full, and Lance then scans every row of it.
    #[tokio::test(flavor = "multi_thread")]
    async fn b14_the_full_read_fires_on_dense_demand_and_not_on_sparse() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let fx = fixture(&labels, &f).await;
        let d = answer_demand(&labels, 17);
        let kind = ResultKind::Count(BAG);
        let run = |p: ReadPolicy| {
            evaluate_addressed_with(&rules, &d, kind, Options::ALL, &fx.lance, &fx.lanes, p)
        };
        let (sparse, _, sparse_io, sparse_l) = run(ReadPolicy {
            coalesce_gap: 0,
            full_permille: 100,
        })
        .await
        .unwrap();
        let (dense, _, dense_io, dense_l) = run(ReadPolicy {
            coalesce_gap: 0,
            full_permille: 1,
        })
        .await
        .unwrap();
        assert_eq!(dense, sparse);
        // Sides that sought keys. `Allowed` has 4 rows, so its 3 centers
        // cover 750 permille and it reads in full under both policies;
        // `Parent` has 400 rows and is the side the threshold separates.
        let sought = |l: &[Vec<Lookup>]| -> Vec<Lookup> {
            l.iter().flatten().filter(|l| l.keys > 0).copied().collect()
        };
        let (s, d) = (sought(&sparse_l), sought(&dense_l));
        assert!(!s.is_empty(), "the fixture must have centers to seek");
        assert!(
            s.iter().any(|l| l.read == Read::Ranges),
            "sparse demand reads ranges on the large side: {s:?}"
        );
        assert!(
            d.iter().all(|l| l.read == Read::Full && l.ranges_read == 0),
            "dense demand reads every sought side in full: {d:?}"
        );
        let scanned = |io: &[Vec<SideIo>]| -> usize {
            io.iter().flatten().map(|s| s.rows.rows_scanned).sum()
        };
        assert!(
            scanned(&dense_io) > scanned(&sparse_io),
            "a full read scans more rows than the ranges: {} vs {}",
            scanned(&dense_io),
            scanned(&sparse_io)
        );
    }

    /// B15: a coalesced read hands Lance fewer ranges than the exact read and
    /// gives the memory path's answer. Without `Parent(17, 2)` the centers are
    /// `{1, 3}`, and key 2 sits between them, undemanded, on both `Parent` and
    /// `Allowed` (where it is the duplicate `[2, 2]` pair), so the merged read
    /// covers it. This fixture does not show the post-read drop: rows of key 2
    /// find no partner in the join, so the answer is the same with or without
    /// it (disable run). B13 and B14 are what pin the drop.
    #[tokio::test(flavor = "multi_thread")]
    async fn b15_coalescing_hands_lance_fewer_ranges_for_the_same_answer() {
        let mut f = family_data();
        f.parent.retain(|r| *r != [17, 2]);
        f.allowed = vec![[1, 1], [2, 2], [2, 2], [3, 3], [2000, 2000]];
        let (labels, rules) = resolved(&family::crepe_plan());
        let mem = store(&labels, &f, [1, 1, 1]);
        let fx = fixture(&labels, &f).await;
        let d = answer_demand(&labels, 17);
        for kind in kinds() {
            let (m, _) = evaluate(&rules, &d, kind, Options::ALL, &mem).unwrap();
            let run = |p: ReadPolicy| {
                evaluate_addressed_with(&rules, &d, kind, Options::ALL, &fx.lance, &fx.lanes, p)
            };
            let (exact, _, _, el) = run(ReadPolicy::EXACT).await.unwrap();
            let (merged, _, _, ml) = run(ReadPolicy {
                coalesce_gap: u32::MAX,
                full_permille: u32::MAX,
            })
            .await
            .unwrap();
            assert_eq!(exact, m, "{kind:?}");
            assert_eq!(merged, m, "{kind:?}");
            let exact_ranges: usize = el.iter().flatten().map(|l| l.ranges_read).sum();
            let merged_ranges: usize = ml.iter().flatten().map(|l| l.ranges_read).sum();
            assert!(
                merged_ranges < exact_ranges,
                "{merged_ranges} vs {exact_ranges}"
            );
            assert!(ml.iter().flatten().any(|l| l.read == Read::Coalesced));
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn b12_a_lane_from_another_dataset_is_refused() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let one = fixture(&labels, &f).await;
        let mut g = family_data();
        g.parent[0] = [17, 2];
        let two = fixture(&labels, &g).await;
        let d = answer_demand(&labels, 17);
        let kind = ResultKind::Count(BAG);
        // Same relation ids and versions on both stores.
        for (l1, l2) in one.lanes.iter().zip(&two.lanes) {
            assert_eq!((l1.rel, l1.lane.version()), (l2.rel, l2.lane.version()));
        }
        let got = evaluate_addressed(&rules, &d, kind, Options::ALL, &two.lance, &one.lanes).await;
        assert!(
            matches!(&got, Err(PipelineRefusal::Addressed(e)) if e.starts_with("OtherDataset")),
            "{got:?}"
        );
        // Silence twin: each store with its own lanes is served.
        assert!(
            evaluate_addressed(&rules, &d, kind, Options::ALL, &two.lance, &two.lanes)
                .await
                .is_ok()
        );
    }

    /// B9: the arm adds no program and no foreign hop.
    #[tokio::test(flavor = "multi_thread")]
    async fn b9_the_addressed_arm_adds_no_program() {
        let f = family_data();
        let (labels, rules) = resolved(&family::crepe_plan());
        let mem = store(&labels, &f, [1, 1, 1]);
        let fx = fixture(&labels, &f).await;
        let d = answer_demand(&labels, 17);
        let kind = ResultKind::Bindings(SET);
        let (_, rm) = evaluate(&rules, &d, kind, Options::ALL, &mem).unwrap();
        let (_, rb, _, _) =
            evaluate_addressed(&rules, &d, kind, Options::ALL, &fx.lance, &fx.lanes)
                .await
                .unwrap();
        assert_eq!(rb.programs.len(), rm.programs.len());
        assert_eq!(rb.rows_prepared, rm.rows_prepared);
    }
}
