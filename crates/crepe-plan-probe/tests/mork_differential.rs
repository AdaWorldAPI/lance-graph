//! MORK as the differential oracle for every admitted `query_multi` pattern
//! (D-MPS-9). Each fixture's rules are mapped by `mork_api::to_oracle`, run in
//! one MORK `Space` through `mork_pathmap_probe::boundary::mork_rules`, and the
//! distinct head tuples are compared as SETS with the independent nested-loop
//! reference. MORK stores sets: bag semantics stay with the reference, and no
//! count is ever invented for MORK.
//!
//! Run from the crate directory (own workspace root):
//!   cd crates/crepe-plan-probe && cargo test --test mork_differential

use std::collections::BTreeSet;

use crepe_plan_probe::logic::{Atom, Labels, RelId, Rule, Term, VarId, resolve};
use crepe_plan_probe::mork_api::{MorkRefusal, admit, query_multi, to_oracle};
use crepe_plan_probe::native::Store;
use crepe_plan_probe::plan::Plan;
use crepe_plan_probe::programs::{family, family_noisy, refused, star3, two_way};
use crepe_plan_probe::reference;
use mork_pathmap_probe::boundary::{OracleCost, OracleFacts, mork_rules};

// ---------------------------------------------------------------------------
// Fixtures (small helpers copied from tests/pipeline.rs; they are test code)
// ---------------------------------------------------------------------------

/// The family fixture. `Allowed(2)` is stored twice: a bag of facts, which
/// MORK collapses (its set semantics) and the reference counts twice.
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

/// Raw rows of a relation at its declared arity, for the reference.
fn raw<'a>(labels: &'a Labels, store: &'a Store) -> impl Fn(RelId) -> Vec<Vec<u64>> + 'a {
    move |rel| {
        let v = store.view(rel);
        (0..v.len())
            .map(|i| {
                v.cols[..labels.arity(rel)]
                    .iter()
                    .map(|c| u64::from(c[i]))
                    .collect()
            })
            .collect()
    }
}

/// The same rows with each relation deduplicated: what a MORK `Space` holds.
fn distinct<'a>(labels: &'a Labels, store: &'a Store) -> impl Fn(RelId) -> Vec<Vec<u64>> + 'a {
    let rows = raw(labels, store);
    move |rel| {
        rows(rel)
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

struct Fixture {
    name: &'static str,
    labels: Labels,
    rules: Vec<Rule>,
    store: Store,
    goal: RelId,
}

impl Fixture {
    fn goal_rules(&self) -> Vec<&Rule> {
        self.rules
            .iter()
            .filter(|r| r.head.rel == self.goal)
            .collect()
    }

    fn demand(&self) -> Vec<Option<u64>> {
        vec![None; self.goal_rules()[0].head.terms.len()]
    }
}

fn family_store(labels: &Labels, f: &Family) -> Store {
    Store::from_rows(vec![
        (labels.id("Parent").unwrap(), &f.parent, 1),
        (labels.id("Allowed").unwrap(), &f.allowed, 1),
        (labels.id("Sponsor").unwrap(), &f.sponsor, 1),
    ])
}

fn two_way_rows() -> (Vec<[u32; 2]>, Vec<[u32; 2]>) {
    let r = (0..120u32).map(|i| [i, i % 17]).collect();
    let s = (0..90u32).map(|i| [i % 23, 500 + i]).collect();
    (r, s)
}

fn fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();

    let f = family_data();
    let (labels, rules) = resolved(&family::crepe_plan());
    let store = family_store(&labels, &f);
    out.push(Fixture {
        name: "family",
        goal: labels.id("Answer").unwrap(),
        labels,
        rules,
        store,
    });

    let (labels, rules) = resolved(&family_noisy::crepe_plan());
    let store = family_store(&labels, &f);
    out.push(Fixture {
        name: "family_noisy",
        goal: labels.id("Answer").unwrap(),
        labels,
        rules,
        store,
    });

    let (r, s) = two_way_rows();
    let (labels, rules) = resolved(&two_way::crepe_plan());
    let store = Store::from_rows(vec![
        (labels.id("R").unwrap(), &r, 1),
        (labels.id("S").unwrap(), &s, 1),
    ]);
    out.push(Fixture {
        name: "two_way",
        goal: labels.id("T").unwrap(),
        labels,
        rules,
        store,
    });

    // Unary relations are stored as `[v, v]` rows.
    let (labels, rules) = resolved(&star3::crepe_plan());
    let r: Vec<[u32; 2]> = (0..50u32).map(|v| [v, v]).collect();
    let s: Vec<[u32; 2]> = (25..80u32).map(|v| [v, v]).collect();
    let q: Vec<[u32; 2]> = (0..100u32).filter(|v| v % 3 == 0).map(|v| [v, v]).collect();
    let store = Store::from_rows(vec![
        (labels.id("R").unwrap(), &r, 1),
        (labels.id("S").unwrap(), &s, 1),
        (labels.id("Q").unwrap(), &q, 1),
    ]);
    out.push(Fixture {
        name: "star3",
        goal: labels.id("Hit").unwrap(),
        labels,
        rules,
        store,
    });

    // A constant in the BODY: `C(z) <- R(7, y), S(y, z)` over the two_way data,
    // built by hand with `Atom`/`Term` (no `crepe_plan!` module). The head
    // relation id is not in the label table; nothing here resolves it by name.
    let (r, s) = two_way_rows();
    let (labels, _) = resolved(&two_way::crepe_plan());
    let (rel_r, rel_s) = (labels.id("R").unwrap(), labels.id("S").unwrap());
    let goal = RelId(99);
    let (y, z) = (Term::Var(VarId(0)), Term::Var(VarId(1)));
    let rule = Rule {
        head: Atom {
            rel: goal,
            terms: vec![z],
        },
        body: vec![
            Atom {
                rel: rel_r,
                terms: vec![Term::Const(7), y],
            },
            Atom {
                rel: rel_s,
                terms: vec![y, z],
            },
        ],
    };
    let store = Store::from_rows(vec![(rel_r, &r, 1), (rel_s, &s, 1)]);
    out.push(Fixture {
        name: "body_constant",
        labels,
        rules: vec![rule],
        store,
        goal,
    });

    // A three-atom chain: acyclic, so the MORK shape admits it, but the native
    // star refuses it. MORK answers it; the reference checks MORK.
    let plan = refused::crepe_plan();
    let labels = Labels::of(&plan);
    let rules: Vec<Rule> = resolve(&plan, &labels)
        .into_iter()
        .filter_map(Result::ok)
        .collect();
    let e: Vec<[u32; 2]> = (0..30u32).map(|i| [i, (i * 7 + 3) % 30]).collect();
    let store = Store::from_rows(vec![(labels.id("E").unwrap(), &e, 1)]);
    out.push(Fixture {
        name: "chain",
        goal: labels.id("Chain").unwrap(),
        labels,
        rules,
        store,
    });

    out
}

/// The oracle's input relations: every relation the store holds, at its
/// declared arity.
fn oracle_facts(fx: &Fixture) -> Vec<OracleFacts<'_>> {
    fx.store
        .relations
        .iter()
        .map(|(rel, view)| OracleFacts {
            rel: rel.0,
            arity: fx.labels.arity(*rel),
            view,
        })
        .collect()
}

/// The reference's distinct head tuples, narrowed to `u32` (MORK's symbols).
fn reference_set(fx: &Fixture) -> BTreeSet<Vec<u32>> {
    reference::eval(&fx.rules, fx.goal, &fx.demand(), &raw(&fx.labels, &fx.store))
        .keys()
        .map(|row| {
            row.iter()
                .map(|&v| u32::try_from(v).expect("fixture values fit u32"))
                .collect()
        })
        .collect()
}

fn mork_set(fx: &Fixture) -> (BTreeSet<Vec<u32>>, OracleCost) {
    let oracle_rules: Vec<_> = fx
        .goal_rules()
        .into_iter()
        .map(|r| to_oracle(r).expect("admitted by the logical interface"))
        .collect();
    let (set, cost) = mork_rules(&oracle_facts(fx), &oracle_rules)
        .unwrap_or_else(|e| panic!("{}: MORK refused: {e:?}", fx.name));
    (set.rows, cost)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// FAILS IF: any admitted pattern, run by MORK, derives a head tuple the
/// independent reference does not (or misses one it does). The set must also
/// be non-empty, so the equality cannot hold of two empty answers.
#[test]
fn mork_sets_equal_the_reference_sets() {
    for fx in fixtures() {
        let want = reference_set(&fx);
        let (got, _) = mork_set(&fx);
        assert!(!want.is_empty(), "{}: vacuous fixture", fx.name);
        assert_eq!(got, want, "{}", fx.name);
    }
}

/// `family_noisy` renames variables and duplicates an atom and a rule; its
/// answer set is `family`'s.
#[test]
fn family_noisy_gives_the_same_mork_set_as_family() {
    let all = fixtures();
    let by = |name: &str| all.iter().find(|f| f.name == name).expect("fixture");
    let (family_set, _) = mork_set(by("family"));
    let (noisy_set, _) = mork_set(by("family_noisy"));
    assert!(!family_set.is_empty());
    assert_eq!(noisy_set, family_set);
}

/// The body constant selects: only `R(7, y)` rows feed the answer, so the
/// answer is a strict subset of the unrestricted join's `z` column.
#[test]
fn a_body_constant_restricts_the_mork_answer() {
    let all = fixtures();
    let fx = all.iter().find(|f| f.name == "body_constant").unwrap();
    let (got, _) = mork_set(fx);
    // y = 7 (R(7, 7)); S rows with i % 23 == 7 are i = 7, 30, 53, 76.
    let want: BTreeSet<Vec<u32>> = [507u32, 530, 553, 576].iter().map(|&z| vec![z]).collect();
    assert_eq!(got, want);
}

/// The number of `query_multi` effect callbacks equals the number of body
/// matches over the DISTINCT facts: the reference's derivation count with each
/// relation deduplicated first. MORK holds a set, so a duplicated fact row
/// (`Allowed(2)` twice in `family`) matches once.
///
/// FAILS IF: the unification-only enumeration is not one callback per match,
/// or a duplicate fact reaches MORK as a second match.
#[test]
fn mork_match_count_equals_reference_derivations_over_distinct_facts() {
    for fx in fixtures() {
        let (_, cost) = mork_set(&fx);
        let want: u64 = reference::eval(
            &fx.rules,
            fx.goal,
            &fx.demand(),
            &distinct(&fx.labels, &fx.store),
        )
        .values()
        .sum();
        assert!(want > 0, "{}: vacuous fixture", fx.name);
        assert_eq!(cost.matches, want, "{}", fx.name);
    }
}

/// Over the BAG rows (duplicates kept) the reference counts more derivations
/// than MORK matched in `family`: the duplicate `Allowed(2)` is the difference.
/// Without this the distinct-facts test could pass with the dedup being a no-op.
#[test]
fn the_duplicate_allowed_row_is_what_separates_bag_from_set_matches() {
    let all = fixtures();
    let fx = all.iter().find(|f| f.name == "family").unwrap();
    let (_, cost) = mork_set(fx);
    let bag: u64 = reference::eval(&fx.rules, fx.goal, &fx.demand(), &raw(&fx.labels, &fx.store))
        .values()
        .sum();
    assert!(bag > cost.matches, "bag {bag} must exceed set {}", cost.matches);
}

/// `to_oracle` refuses exactly what `admit` refuses, and returns that refusal
/// unchanged. The chain is acyclic, so `admit` (and so `to_oracle`) admits it.
#[test]
fn to_oracle_returns_admits_refusals_unchanged() {
    let plan = refused::crepe_plan();
    let labels = Labels::of(&plan);
    let resolved = resolve(&plan, &labels);
    let rule_of = |name: &str| -> Rule {
        plan.rules
            .iter()
            .zip(&resolved)
            .find(|(r, _)| r.goal.relation == name)
            .and_then(|(_, res)| res.clone().ok())
            .unwrap_or_else(|| panic!("{name} resolves"))
    };
    for name in ["Chain", "Tri", "Loop"] {
        let rule = rule_of(name);
        assert_eq!(
            to_oracle(&rule).map(|_| ()),
            admit(&query_multi(&rule)),
            "{name}"
        );
    }
    assert!(to_oracle(&rule_of("Chain")).is_ok(), "chain is acyclic");
    assert_eq!(
        to_oracle(&rule_of("Tri")).map(|_| ()),
        Err(MorkRefusal::Cyclic)
    );
    assert!(matches!(
        to_oracle(&rule_of("Loop")),
        Err(MorkRefusal::RepeatedVarInAtom { pattern: 0, .. })
    ));
}

/// A constant that does not fit the oracle's `u32` symbols is refused by name,
/// never truncated.
#[test]
fn a_constant_beyond_u32_is_refused() {
    let (labels, _) = resolved(&two_way::crepe_plan());
    let rule = Rule {
        head: Atom {
            rel: RelId(99),
            terms: vec![Term::Var(VarId(0))],
        },
        body: vec![Atom {
            rel: labels.id("R").unwrap(),
            terms: vec![Term::Const(1 << 40), Term::Var(VarId(0))],
        }],
    };
    assert_eq!(
        to_oracle(&rule).map(|_| ()),
        Err(MorkRefusal::ConstOutOfRange { value: 1 << 40 })
    );
}
