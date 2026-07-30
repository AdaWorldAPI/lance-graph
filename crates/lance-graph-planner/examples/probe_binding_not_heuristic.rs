//! `probe_binding_not_heuristic` — W7: the falsifier W6 deferred. Prove the
//! `Locus::Antecedent` chip is placed by structural BINDING, not by a distance
//! heuristic — i.e. that the stored nibble carries bits a cheap algorithm
//! cannot recompute, so the chip bears load instead of decorating.
//!
//! **The operator's criterion (2026-07-30), which is this probe's spec:** a
//! store must be *effort on behalf of the thought after you*; bookkeeping that
//! a better algorithm makes redundant is "a joke to pretend it was thinking."
//! The pull-test follows: pull the chip — if a cheap resolver reconstructs the
//! same answer everywhere, the chip was decoration; if somewhere the cheap
//! resolver produces a *different, wrong* answer, the chip was load-bearing.
//!
//! **W6 bound fixture-supplied `(pronoun, antecedent)` pairs — it proved the
//! write path, not the resolution.** This probe supplies the two RESOLVERS and
//! makes them disagree on real KJV text:
//!
//! - **Heuristic** (the cheap baseline, deliberately strong): nearest
//!   preceding number-agreeing referential token. This is the resolver a
//!   "better algorithm makes it redundant" claim would deploy.
//! - **Binding** (structural): complement-clause subjects bind the MATRIX
//!   subject (the reflexive signature from `E-EYES-OPENED-PRINTS-BLIND-1`);
//!   main-clause subject pronouns bind the PREVIOUS main clause's subject,
//!   skipping relative-clause material, with an animacy check against the
//!   pronoun's own verb.
//!
//! Both resolvers read the same tagger-level token features (POS, number,
//! animacy, clause boundaries, relative spans, complementizers). Gold
//! annotations exist ONLY in the assert arms — neither resolver sees them,
//! which is exactly the honesty gap W6 left open.
//!
//! **The adversarial text is Gen 3:1**, where the relative clause *"which the
//! LORD God had made"* interposes between `serpent` and `he`: recency finds
//! `god` at in-window distance −4 (wrong); binding finds `serpent` at −17 —
//! **outside the ±8 chip range, so the binder ESCALATES rather than store
//! anything**. The chip's refusal is the load-bearing act: the cheap resolver
//! would have happily stored a well-formed, in-range, WRONG nibble. Gen 3:7
//! supplies the stay-silent half (both resolvers agree on both `they`s) plus
//! the chip-composition payoff: inner `they`@12 → matrix `they`@9 → `them`@4
//! resolves by following two stored nibbles — the chips COMPOSE, each carrying
//! its local warrant, never a cached far verdict.
//!
//! Escalation is side-band (a record, not row state): a `0` nibble alone reads
//! as *unbound*, indistinguishable from never-attempted — same honest boundary
//! W6 documented.
//!
//! Usage: `cargo run -p lance-graph-planner --example probe_binding_not_heuristic`

use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow};
use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::witness_fabric::WitnessLens;

/// Part-of-speech class at tagger granularity — the features both resolvers
/// share. Nothing here encodes an antecedent.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pos {
    /// Referential nominal — an antecedent candidate.
    Noun,
    /// Referential pronoun — a candidate AND (possibly) a resolvable site.
    Pron,
    /// Verb. `animate` on a verb token means "requires an animate subject".
    Verb,
    /// Everything else (particles, adjectives, conjunctions, complementizers).
    Other,
}

/// One fixture token: surface features only (what a POS tagger + chunker
/// emits). `gold` is the referee's answer — read ONLY by assert arms.
struct Tok {
    label: &'static str,
    pos: Pos,
    plural: bool,
    /// Noun/Pron: referent animacy. Verb: requires an animate subject.
    animate: bool,
    /// Token sits inside a relative-clause span ("which … made").
    in_relative: bool,
    /// Token is a conjunction opening a NEW main clause.
    new_main: bool,
    /// Token is a complementizer ("that") introducing a complement clause.
    comp: bool,
    /// Referee annotation for pronouns: the correct antecedent position.
    gold: Option<usize>,
}

impl Tok {
    fn t(label: &'static str) -> Self {
        Tok {
            label,
            pos: Pos::Other,
            plural: false,
            animate: false,
            in_relative: false,
            new_main: false,
            comp: false,
            gold: None,
        }
    }
    fn noun(label: &'static str, plural: bool, animate: bool) -> Self {
        Tok {
            pos: Pos::Noun,
            plural,
            animate,
            ..Tok::t(label)
        }
    }
    fn pron(label: &'static str, plural: bool, gold: usize) -> Self {
        Tok {
            pos: Pos::Pron,
            plural,
            animate: true,
            gold: Some(gold),
            ..Tok::t(label)
        }
    }
    fn verb(label: &'static str, requires_animate: bool) -> Self {
        Tok {
            pos: Pos::Verb,
            animate: requires_animate,
            ..Tok::t(label)
        }
    }
    fn conj(label: &'static str) -> Self {
        Tok {
            new_main: true,
            ..Tok::t(label)
        }
    }
    fn rel(mut self) -> Self {
        self.in_relative = true;
        self
    }
    fn is_candidate(&self) -> bool {
        matches!(self.pos, Pos::Noun | Pos::Pron)
    }
}

/// Gen 3:1 (KJV, lowercased, trimmed): *"now the serpent was more subtil than
/// any beast of the field which the LORD God had made: and he said unto the
/// woman"*. The relative span covers "which … made". Gold: `he`@19 → serpent@2.
fn genesis_3_1() -> Vec<Tok> {
    vec![
        Tok::t("now"),                       // 0
        Tok::t("the"),                       // 1
        Tok::noun("serpent", false, true),   // 2
        Tok::verb("was", false),             // 3
        Tok::t("more"),                      // 4
        Tok::t("subtil"),                    // 5
        Tok::t("than"),                      // 6
        Tok::t("any"),                       // 7
        Tok::noun("beast", false, true),     // 8
        Tok::t("of"),                        // 9
        Tok::t("the"),                       // 10
        Tok::noun("field", false, false),    // 11
        Tok::t("which").rel(),               // 12
        Tok::t("the").rel(),                 // 13
        Tok::t("lord").rel(),                // 14
        Tok::noun("god", false, true).rel(), // 15
        Tok::t("had").rel(),                 // 16
        Tok::verb("made", false).rel(),      // 17
        Tok::conj("and"),                    // 18
        Tok::pron("he", false, 2),           // 19 → serpent@2 (d = -17)
        Tok::verb("said", true),             // 20
        Tok::t("unto"),                      // 21
        Tok::t("the"),                       // 22
        Tok::noun("woman", false, true),     // 23
    ]
}

/// Gen 3:7 (KJV, lowercased, trimmed): *"and the eyes of them both were
/// opened, and they knew that they were naked"*. Gold: matrix `they`@9 →
/// `them`@4 (the pair, not the eyes); inner `they`@12 → matrix `they`@9 (the
/// reflexive signature — knower == overt inner subject).
fn genesis_3_7() -> Vec<Tok> {
    vec![
        Tok::t("and"),                  // 0
        Tok::t("the"),                  // 1
        Tok::noun("eyes", true, false), // 2  (plural, inanimate)
        Tok::t("of"),                   // 3
        Tok::noun("them", true, true),  // 4  (referential pronoun-as-candidate)
        Tok::t("both"),                 // 5
        Tok::verb("were", false),       // 6
        Tok::verb("opened", false),     // 7
        Tok::conj("and"),               // 8
        Tok::pron("they", true, 4),     // 9  → them@4 (d = -5)
        Tok::verb("knew", true),        // 10 (knowing requires an animate subject)
        Tok {
            comp: true,
            ..Tok::t("that")
        }, // 11
        Tok::pron("they", true, 9),     // 12 → matrix they@9 (d = -3)
        Tok::verb("were", false),       // 13
        Tok::t("naked"),                // 14
    ]
}

/// The cheap baseline: nearest preceding number-agreeing referential token.
/// Deliberately ignores clause structure and animacy — that IS its cheapness.
fn heuristic_resolve(toks: &[Tok], p: usize) -> Option<usize> {
    (0..p)
        .rev()
        .find(|&i| toks[i].is_candidate() && toks[i].plural == toks[p].plural)
}

/// Main-clause id per token (increments at each `new_main` conjunction).
fn clause_of(toks: &[Tok], i: usize) -> usize {
    toks[..=i].iter().filter(|t| t.new_main).count()
}

/// The subject of main clause `k`: its first candidate token outside any
/// relative span.
fn subject_of_clause(toks: &[Tok], k: usize) -> Option<usize> {
    (0..toks.len())
        .find(|&i| clause_of(toks, i) == k && toks[i].is_candidate() && !toks[i].in_relative)
}

/// The structural resolver. Two rules, both computed from features:
///
/// - **R1 (complement subject):** a pronoun immediately after a
///   complementizer binds the subject of the clause whose verb takes the
///   complement (the matrix subject) — the reflexive signature.
/// - **R2 (subject continuity, animacy-checked):** a subject pronoun opening
///   a main clause binds the previous main clause's subject — unless the
///   pronoun's own verb requires an animate subject and that subject is
///   inanimate, in which case it binds the nearest preceding number-agreeing
///   ANIMATE candidate.
fn structural_resolve(toks: &[Tok], p: usize) -> Option<usize> {
    // R1: complement-clause subject → matrix subject.
    if p > 0 && toks[p - 1].comp {
        let matrix_verb = (0..p - 1).rev().find(|&i| toks[i].pos == Pos::Verb)?;
        return subject_of_clause(toks, clause_of(toks, matrix_verb));
    }
    // R2: subject pronoun of main clause k → subject of clause k-1.
    let k = clause_of(toks, p);
    if k == 0 {
        return None;
    }
    let prev_subj = subject_of_clause(toks, k - 1)?;
    let own_verb = (p + 1..toks.len()).find(|&i| toks[i].pos == Pos::Verb);
    let needs_animate = own_verb.is_some_and(|v| toks[v].animate);
    if needs_animate && !toks[prev_subj].animate {
        // Animacy repair: nearest preceding number-agreeing animate candidate.
        return (0..p).rev().find(|&i| {
            toks[i].is_candidate() && toks[i].plural == toks[p].plural && toks[i].animate
        });
    }
    Some(prev_subj)
}

/// W6's binder semantics, verbatim: in-range displacement binds the nibble;
/// `0` or out-of-`±8` escalates — the row stays unbound, never clamped.
fn bind(rows: &mut [NodeRow], pronoun: usize, target: usize) -> Result<i8, isize> {
    let d = target as isize - pronoun as isize;
    if d == 0 || !(-8..=7).contains(&d) {
        return Err(d);
    }
    let facet = CausalWitnessFacet::ZERO.with(Locus::Antecedent, d as i8);
    WitnessLens::write_register(&mut rows[pronoun], &facet);
    Ok(d as i8)
}

fn fresh_rows(n: usize) -> Vec<NodeRow> {
    (0..n)
        .map(|i| NodeRow {
            key: NodeGuid::local(i as u32),
            edges: EdgeBlock::default(),
            value: [0u8; 480],
        })
        .collect()
}

fn main() {
    let mut green = true;
    let mut gate = |name: &str, pass: bool, detail: String| {
        println!("[{}] {name} — {detail}", if pass { "PASS" } else { "FAIL" });
        green &= pass;
    };

    // ── Fixture A: Gen 3:1 — the divergence text ─────────────────────────────
    let a = genesis_3_1();
    let he = 19;
    let h_a = heuristic_resolve(&a, he);
    let s_a = structural_resolve(&a, he);

    // B1 — can-fire: the resolvers genuinely disagree, and gold sides with
    // binding. Falsifier: agreement, or the heuristic being right.
    gate(
        "B1 divergence",
        h_a == Some(15) && s_a == Some(2) && a[he].gold == Some(2) && h_a != s_a,
        format!(
            "heuristic(he@19) = {:?} ({}), binding = {:?} ({}), gold = serpent@2",
            h_a,
            h_a.map_or("-", |i| a[i].label),
            s_a,
            s_a.map_or("-", |i| a[i].label),
        ),
    );

    // B2 — the temptation is real, and the chip refuses it. The heuristic's
    // wrong answer FITS the ±8 window (d = -4, storable); the structural
    // answer does not (d = -17) — so the binder must escalate, leaving the
    // nibble unbound rather than storing either a clamped or a cheap value.
    let mut rows_a = fresh_rows(a.len());
    let heuristic_d = 15_isize - he as isize;
    let bound = bind(&mut rows_a, he, s_a.unwrap());
    let lens_a = WitnessLens::new(&rows_a);
    let nibble = lens_a.at(he).map(|f| f.at(Locus::Antecedent));
    gate(
        "B2 escalate-not-clamp",
        (-8..=7).contains(&heuristic_d) && bound == Err(-17) && nibble == Some(0),
        format!(
            "heuristic d={heuristic_d} (in-window, storable-but-wrong), binding d=-17 → escalated, nibble={nibble:?}"
        ),
    );

    // ── Fixture B: Gen 3:7 — the stay-silent text + chip composition ─────────
    let b = genesis_3_7();
    let (matrix, inner) = (9, 12);
    let mut rows_b = fresh_rows(b.len());
    let mut agree = true;
    let mut displacements = Vec::new();
    for &p in &[matrix, inner] {
        let h = heuristic_resolve(&b, p);
        let s = structural_resolve(&b, p);
        agree &= h == s && s == b[p].gold;
        if let Some(t) = s {
            if let Ok(d) = bind(&mut rows_b, p, t) {
                displacements.push(d);
            }
        }
    }

    // B3 — can-stay-silent: on the reflexive verse both resolvers agree with
    // gold and with each other; the divergence gate does not fire on
    // everything (a guard that always fires carries no information). The two
    // chips hold distinct nonzero displacements (anti-vacuity).
    gate(
        "B3 stay-silent",
        agree && displacements == vec![-5, -3],
        format!("both they@9/they@12 agree across resolvers; chips = {displacements:?}"),
    );

    // B4 — the pull-test, both directions. WITH chips: the inner pronoun's
    // ultimate referent resolves by following two stored nibbles
    // (they@12 → they@9 → them@4) — each chip a local warrant, composed, no
    // far verdict cached anywhere. WITHOUT chips (heuristic-only
    // reconstruction over both fixtures): Gen 3:7 survives, Gen 3:1 comes
    // back wrong — the chip carried exactly the bits recency cannot recompute.
    let lens_b = WitnessLens::new(&rows_b);
    let hop1 = lens_b
        .at(inner)
        .and_then(|f| f.resolves_to(Locus::Antecedent, inner, lens_b.len()));
    let hop2 = hop1.and_then(|p| {
        lens_b
            .at(p)
            .and_then(|f| f.resolves_to(Locus::Antecedent, p, lens_b.len()))
    });
    let heuristic_wrong_somewhere = heuristic_resolve(&a, he) != a[he].gold;
    gate(
        "B4 pull-test",
        hop1 == Some(matrix) && hop2 == Some(4) && heuristic_wrong_somewhere,
        format!(
            "chip chain they@12 → {:?} → {:?} (gold them@4); heuristic-only reconstruction fails on 3:1",
            hop1, hop2
        ),
    );

    println!();
    if green {
        println!("ALL GATES GREEN");
    } else {
        println!("GATES FAILED");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe's own gates, runnable under `cargo test --example`.
    #[test]
    fn divergence_fires_on_3_1_and_stays_silent_on_3_7() {
        let a = genesis_3_1();
        assert_eq!(heuristic_resolve(&a, 19), Some(15), "recency finds god");
        assert_eq!(structural_resolve(&a, 19), Some(2), "binding finds serpent");
        assert_eq!(a[19].gold, Some(2));

        let b = genesis_3_7();
        for p in [9, 12] {
            assert_eq!(heuristic_resolve(&b, p), structural_resolve(&b, p));
            assert_eq!(structural_resolve(&b, p), b[p].gold);
        }
    }

    /// Deleting the animacy check must break 3:7's matrix resolution (the
    /// rule is load-bearing, not decoration): without it, R2 returns the
    /// previous clause's subject verbatim — `eyes`@2, which is wrong.
    #[test]
    fn animacy_check_is_load_bearing() {
        let b = genesis_3_7();
        let k = clause_of(&b, 9);
        let bare_r2 = subject_of_clause(&b, k - 1);
        assert_eq!(bare_r2, Some(2), "bare subject-continuity picks eyes@2");
        assert_ne!(bare_r2, b[9].gold, "…which is NOT the gold answer");
        assert_eq!(structural_resolve(&b, 9), b[9].gold, "the check repairs it");
    }

    /// The relative-clause skip must be load-bearing on 3:1: if `god`@15
    /// counted as a main-clause subject, R2 would bind it. The fixture pins
    /// that `god` is inside the relative span and the clause-0 subject is
    /// `serpent`.
    #[test]
    fn relative_span_skip_is_load_bearing() {
        let a = genesis_3_1();
        assert!(a[15].in_relative, "god sits inside the relative clause");
        assert_eq!(
            subject_of_clause(&a, 0),
            Some(2),
            "clause-0 subject = serpent"
        );
    }

    /// Out-of-window escalation leaves the nibble unbound — never a clamp.
    #[test]
    fn escalation_leaves_nibble_unbound() {
        let a = genesis_3_1();
        let mut rows = fresh_rows(a.len());
        assert_eq!(bind(&mut rows, 19, 2), Err(-17));
        let lens = WitnessLens::new(&rows);
        assert_eq!(lens.at(19).map(|f| f.at(Locus::Antecedent)), Some(0));
    }
}
