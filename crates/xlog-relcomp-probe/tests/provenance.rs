//! X6 provenance and evidence roots.
//!
//! Provenance is the original source root, the frame version, the XLOG
//! program hash, and the CE64 declared reading. Evidence-root uniqueness is
//! counted separately from derivation multiplicity: two consumers of one
//! root are one independent observation, however many distinct derivations
//! the bag algebra holds.
//!
//! Independent support is the set of minimal root environments that entail
//! an answer (an ATMS label, PosBool provenance), not the lineage of roots
//! the derivations touch. `label` and `derivations` here are a cold, test-only
//! oracle over tiny fixtures: they enumerate paths and build owned sets of
//! environments, which the fold never does. They are not a precedent for a
//! label type in a query path; production root-disjoint support belongs to
//! `lance_graph_planner::nars::support::DerivationLedger`.

use std::collections::BTreeSet;

use causal_edge::edge::CausalEdge64;
use dr_math_binding_probe::{
    Axis, AxisKind, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Scope,
};
use egglog::proof::{Justification, ProofId, ProofStore};
use egglog::{CommandOutput, EGraph};
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::epistemic_state5::{
    Certification3, Epi5Declarations, Epi5Gen, Epi5Reading, EpistemicState5, Topology2,
};
use lance_graph_contract::rail_geometry::RailAxis;
use xlog_relcomp_probe::recursive::{closure, trace};
use xlog_relcomp_probe::{AddressMap, SLOTS, register};

const N: Axis = Axis {
    class: 30,
    kind: AxisKind::Identity,
};
const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 2 };
const XLOG_VIEW: ClassId = 0x7F01;
const RAIL: RailAxis = RailAxis::Taxonomy;

const RULES: &str = "pred edge(u32, u32).
pred path(u32, u32).
path(X, Y) :- edge(X, Y).
path(X, Y) :- path(X, Z), edge(Z, Y).
?- path(X, Y).
";

const DIAMOND: &[(u32, u32)] = &[(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)];

const BASE: EpistemicState5 =
    EpistemicState5::new(Epi5Gen::V1, Topology2::Direct, Certification3::Open);
const COMPOSED: EpistemicState5 =
    EpistemicState5::new(Epi5Gen::V1, Topology2::IndirectKnown, Certification3::Open);

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// One observer's facts: who reported them, and what was reported.
struct Source<'a> {
    observer: &'a str,
    facts: &'a [(u32, u32)],
}

/// The original source root: the observer and the set of facts it reported.
/// The facts are sorted and deduplicated first, so re-serializing the same
/// report in another order is the same root, never a new observation.
fn root(s: &Source<'_>) -> u64 {
    let mut facts = s.facts.to_vec();
    facts.sort_unstable();
    facts.dedup();
    let mut bytes = s.observer.as_bytes().to_vec();
    bytes.push(0);
    for (a, b) in facts {
        bytes.extend_from_slice(&a.to_le_bytes());
        bytes.extend_from_slice(&b.to_le_bytes());
    }
    fnv1a64(&bytes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Provenance {
    root: u64,
    version: FrameRevisions,
    /// FNV-1a over the consumer's program text (XLOG rules, or egglog rules);
    /// never over the facts, which the root already identifies.
    program_hash: u64,
    /// The CE64 declared reading. Declared for XLOG results only; an egglog
    /// equality has no declared CE64 reading, so it is `None`.
    reading: Option<EpistemicState5>,
}

fn declared() -> Epi5Declarations {
    let mut d = Epi5Declarations::new();
    d.declare(XLOG_VIEW, RAIL, Epi5Reading::default());
    d
}

/// The declared reading of an XLOG result whose trace has `trace_len` facts,
/// stamped into a CE64 edge and read back through the declaration.
fn reading(trace_len: usize) -> EpistemicState5 {
    let state = if trace_len > 1 { COMPOSED } else { BASE };
    let e = CausalEdge64(0x0000_00AB_CD12_3456).with_epistemic_raw5(state.raw());
    declared()
        .project_state5(
            XLOG_VIEW,
            RAIL,
            Epi5Gen::V1,
            e.epistemic_raw5(),
            EdgeProvenance::V2Stamped,
        )
        .unwrap()
}

fn xlog_provenance(
    s: &Source<'_>,
    rules: &str,
    version: FrameRevisions,
    trace_len: usize,
) -> Provenance {
    Provenance {
        root: root(s),
        version,
        program_hash: fnv1a64(rules.as_bytes()),
        reading: Some(reading(trace_len)),
    }
}

fn egglog_provenance(s: &Source<'_>, version: FrameRevisions) -> Provenance {
    Provenance {
        root: root(s),
        version,
        program_hash: fnv1a64(HEADER.as_bytes()),
        reading: None,
    }
}

fn revs(source: u64) -> FrameRevisions {
    FrameRevisions {
        source,
        relation_map: 1,
        coordinates: 1,
    }
}

/// Every simple path from `x` to `y` over the edge tuples, as its facts.
fn derivations(edges: &[(u32, u32)], x: u32, y: u32) -> Vec<Vec<(u32, u32)>> {
    fn go(
        edges: &[(u32, u32)],
        at: u32,
        y: u32,
        seen: &mut Vec<u32>,
        path: &mut Vec<(u32, u32)>,
        out: &mut Vec<Vec<(u32, u32)>>,
    ) {
        for &(a, b) in edges {
            if a != at || seen.contains(&b) {
                continue;
            }
            path.push((a, b));
            if b == y {
                out.push(path.clone());
            }
            seen.push(b);
            go(edges, b, y, seen, path, out);
            seen.pop();
            path.pop();
        }
    }
    let mut out = Vec::new();
    go(edges, x, y, &mut vec![x], &mut Vec::new(), &mut out);
    out
}

struct Reg {
    map: AddressMap,
    edge: [u64; SLOTS],
}

fn reg(edges: &[(u32, u32)]) -> Reg {
    let map = AddressMap::new(N, edges.iter().flat_map(|&(a, b)| [a, b])).unwrap();
    let edge = register(edges, &map, &map).unwrap();
    Reg { map, edge }
}

/// The XLOG consumer: closure over the register, trace of `path(x, y)`.
fn xlog_trace_len(r: &Reg, x: u32, y: u32) -> usize {
    let st = FrameStore::new(revs(1));
    let g = Guards {
        frame: st.seal(),
        focus: FOCUS,
        scope: SCOPE,
    };
    let sig = MethodSignature::presence(40, 0, N, N);
    let c = closure(&st, &g, (FOCUS, SCOPE), sig, &r.edge, r.map.valid()).unwrap();
    let (sx, sy) = (r.map.slot(x).unwrap(), r.map.slot(y).unwrap());
    assert_eq!(c.rows[sx] >> sy & 1, 1, "path({x}, {y}) holds");
    trace(&r.edge, r.map.valid(), sx, sy).unwrap().len()
}

const HEADER: &str = "(datatype Node (N i64))
(relation edge (Node Node))
(ruleset same)
(rule ((edge a b)) ((union a b)) :ruleset same)
";

/// egglog's proof of `(N x) = (N y)` over the register's facts.
fn prove(r: &Reg, x: u32, y: u32) -> Option<(ProofStore, ProofId)> {
    let mut facts = String::new();
    for u in 0..r.map.len() {
        for v in r.map.ids(r.edge[u]) {
            facts.push_str(&format!("(edge (N {}) (N {v}))\n", r.map.ids(1 << u)[0]));
        }
    }
    let program = format!("{HEADER}{facts}(run same 100)\n(prove (= (N {x}) (N {y})))\n");
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

const A: &str = "observer-a";
const B: &str = "observer-b";

#[test]
fn provenance_changes_when_and_only_when_its_inputs_change() {
    let src = Source {
        observer: A,
        facts: DIAMOND,
    };
    let base = xlog_provenance(&src, RULES, revs(1), 3);

    // A one-character rule change moves only the program hash.
    let rules2 = RULES.replace("?- path(X, Y).", "?- path(X, Z).");
    let p = xlog_provenance(&src, &rules2, revs(1), 3);
    assert_ne!(p.program_hash, base.program_hash);
    assert_eq!(
        (p.root, p.version, p.reading),
        (base.root, base.version, base.reading)
    );

    // The frame advances: only the version moves.
    let p = xlog_provenance(&src, RULES, revs(2), 3);
    assert_ne!(p.version, base.version);
    assert_eq!(
        (p.root, p.program_hash, p.reading),
        (base.root, base.program_hash, base.reading)
    );

    // One fact changes: only the root moves. The program hash covers the
    // rules, never the facts.
    let mut changed = DIAMOND.to_vec();
    changed[4] = (4, 6);
    let p = xlog_provenance(
        &Source {
            observer: A,
            facts: &changed,
        },
        RULES,
        revs(1),
        3,
    );
    assert_ne!(p.root, base.root);
    assert_eq!(
        (p.program_hash, p.version, p.reading),
        (base.program_hash, base.version, base.reading)
    );

    // Trace length 1 is Direct, longer is IndirectKnown; only the reading moves.
    let direct = xlog_provenance(&src, RULES, revs(1), 1);
    assert_eq!(direct.reading, Some(BASE));
    assert_eq!(base.reading, Some(COMPOSED));
    assert_eq!(
        (direct.root, direct.version, direct.program_hash),
        (base.root, base.version, base.program_hash)
    );
}

#[test]
fn readings_follow_the_trace_of_the_real_closure() {
    let r = reg(DIAMOND);
    // Fails if a base fact reads as composed or a composed edge as direct.
    assert_eq!(reading(xlog_trace_len(&r, 4, 5)), BASE);
    assert_eq!(reading(xlog_trace_len(&r, 1, 5)), COMPOSED);
}

#[test]
fn reordering_a_report_is_not_a_new_observation() {
    // Fails if the root depends on how the facts were serialized: a reordered
    // copy of one report would then count as independent evidence.
    let reordered: Vec<(u32, u32)> = DIAMOND.iter().rev().copied().collect();
    let duplicated: Vec<(u32, u32)> = DIAMOND.iter().chain(DIAMOND).copied().collect();
    let a = root(&Source {
        observer: A,
        facts: DIAMOND,
    });
    assert_eq!(
        a,
        root(&Source {
            observer: A,
            facts: &reordered
        })
    );
    assert_eq!(
        a,
        root(&Source {
            observer: A,
            facts: &duplicated
        })
    );
    // The same facts from another observer are a different root.
    assert_ne!(
        a,
        root(&Source {
            observer: B,
            facts: DIAMOND
        })
    );
}

type Env = BTreeSet<u64>;

/// The minimal sets of roots that entail `path(x, y)`: the ATMS label, or
/// PosBool provenance, over source roots. A fact's label is one singleton
/// environment per source that reports it. A derivation's label is the
/// pairwise union of its facts' labels, and the answer's label is the union
/// over derivations. Every environment that contains another is dropped
/// (absorption: `A + A·B = A`).
fn label(derivs: &[Vec<(u32, u32)>], sources: &[Source<'_>]) -> BTreeSet<Env> {
    let mut answer: BTreeSet<Env> = BTreeSet::new();
    for d in derivs {
        let mut envs: BTreeSet<Env> = [Env::new()].into();
        for f in d {
            let reporters: Vec<u64> = sources
                .iter()
                .filter(|s| s.facts.contains(f))
                .map(root)
                .collect();
            assert!(!reporters.is_empty(), "every fact has an observer");
            envs = envs
                .iter()
                .flat_map(|e| {
                    reporters.iter().map(move |&r| {
                        let mut e = e.clone();
                        e.insert(r);
                        e
                    })
                })
                .collect();
        }
        answer.extend(envs);
    }
    let all = answer.clone();
    answer.retain(|e| !all.iter().any(|o| o != e && o.is_subset(e)));
    answer
}

/// The roots any derivation touches: lineage. It is not independent support.
fn lineage(derivs: &[Vec<(u32, u32)>], sources: &[Source<'_>]) -> BTreeSet<u64> {
    derivs
        .iter()
        .flatten()
        .flat_map(|f| {
            sources
                .iter()
                .filter(move |s| s.facts.contains(f))
                .map(root)
        })
        .collect()
}

/// Independent supports: minimal environments that share no root.
fn disjoint(label: &BTreeSet<Env>) -> bool {
    let envs: Vec<&Env> = label.iter().collect();
    envs.iter()
        .enumerate()
        .all(|(i, a)| envs[i + 1..].iter().all(|b| a.is_disjoint(b)))
}

#[test]
fn derivation_multiplicity_is_not_independent_evidence() {
    // path(1, 5) has two derivations, 1->2->4->5 and 1->3->4->5.
    let derivs = derivations(DIAMOND, 1, 5);
    assert_eq!(derivs.len(), 2);
    let a_branch: &[(u32, u32)] = &[(1, 2), (2, 4), (4, 5)];
    let b_branch: &[(u32, u32)] = &[(1, 3), (3, 4)];
    let b_full: &[(u32, u32)] = &[(1, 3), (3, 4), (4, 5)];
    let src = |o, f| Source {
        observer: o,
        facts: f,
    };

    // Case 1: one observer reported every fact. Two derivations, one
    // environment {A}.
    let one = [src(A, DIAMOND)];
    assert_eq!(label(&derivs, &one).len(), 1);
    assert_eq!(lineage(&derivs, &one).len(), 1);

    // Case 2: B reported 1->3->4, but 4->5 only A reported. The derivations
    // rest on {A} and {A, B}; absorption leaves {A}. Lineage touches two
    // roots, yet B never supports path(1, 5) without A: one independent
    // support, not two.
    let two = [src(A, a_branch), src(B, b_branch)];
    assert_eq!(lineage(&derivs, &two).len(), 2);
    let l = label(&derivs, &two);
    assert_eq!(l.len(), 1);
    assert_eq!(l.iter().next().unwrap().len(), 1);

    // Case 3: B also reported 4->5. Now {A} and {B} each entail path(1, 5)
    // on their own and share no root: two independent supports.
    let three = [src(A, a_branch), src(B, b_full)];
    let l = label(&derivs, &three);
    assert_eq!(l.len(), 2);
    assert!(disjoint(&l));
    // Lineage counts two roots in both cases, so it cannot tell them apart.
    assert_eq!(lineage(&derivs, &three).len(), 2);
    // Two environments that share a root are not independent.
    let shared: BTreeSet<Env> = [[1u64, 2].into(), [2u64, 3].into()].into();
    assert!(!disjoint(&shared));
}

#[test]
fn two_consumers_of_one_root_are_one_observation() {
    let src = Source {
        observer: A,
        facts: DIAMOND,
    };
    let r = reg(DIAMOND);

    // Consumer 1: XLOG's closure and trace, with its declared CE64 reading.
    let xlog = xlog_provenance(&src, RULES, revs(1), xlog_trace_len(&r, 1, 5));
    // Consumer 2: an egglog proof over the same register. Its facts are bits
    // of that register, and it has no declared CE64 reading.
    let (store, proof) = prove(&r, 1, 5).expect("1 = 5 is provable");
    let used = proof_edges(&store, proof);
    assert!(!used.is_empty());
    assert!(used.iter().all(|f| DIAMOND.contains(f)));
    let egg = egglog_provenance(&src, revs(1));
    assert_eq!(egg.reading, None);
    assert_ne!(egg.program_hash, xlog.program_hash, "two programs");

    // Two consumers, two programs, one root: one independent observation.
    let roots: BTreeSet<u64> = [xlog.root, egg.root].into();
    assert_eq!(roots.len(), 1);
}

#[test]
fn unrelated_sources_never_collapse_to_one_root() {
    let b: &[(u32, u32)] = &[(7, 8), (8, 9)];
    let pa = root(&Source {
        observer: A,
        facts: DIAMOND,
    });
    let pb = root(&Source {
        observer: A,
        facts: b,
    });
    assert_ne!(pa, pb);
}
