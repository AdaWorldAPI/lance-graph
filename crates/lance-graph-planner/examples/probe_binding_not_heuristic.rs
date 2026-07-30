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
//! annotations are TYPE-SEPARATED from resolver input (CodeRabbit hardening on
//! this PR): fixtures return `(Vec<Tok>, gold pairs)`, and `Tok` carries no
//! gold field at all — a future `toks[p].gold` shortcut is a compile error,
//! not a convention. This closes the honesty gap W6 left open structurally.
//!
//! **Three fixtures, three roles** (the third added after codex's P1 on this
//! PR, which caught that the original two left the pull-test's stored state
//! heuristic-reconstructible):
//!
//! - **Gen 3:1** — the refusal text: recency finds `god` at in-window distance
//!   −4 (wrong); binding finds `serpent` at −17 — **outside the ±8 chip range,
//!   so the binder ESCALATES rather than store anything**. The refusal is a
//!   choice, not a range limitation: B2 proves the binder ACCEPTS the tempting
//!   wrong target on scratch rows.
//! - **A constructed in-range interposition** (*"the man which the boy saw
//!   slept, and he smiled"* — built English, labelled as such, not KJV): both
//!   answers are in-window, they DIVERGE (recency → `boy`@−4 wrong, binding →
//!   `man`@−7 gold), and the structural answer BINDS — so the stored nibble
//!   itself differs from what recency would reconstruct. This is the chip that
//!   carries bits the cheap algorithm cannot recompute, IN STORED STATE.
//! - **Gen 3:7** — the stay-silent half (both resolvers agree on both `they`s)
//!   plus the chip-composition payoff: inner `they`@12 → matrix `they`@9 →
//!   `them`@4 resolves by following two stored nibbles — chips COMPOSE, each
//!   carrying its local warrant, never a cached far verdict.
//!
//! **Honesty note on the relative span (codex P2 on this PR):** in head-first
//! English the subject precedes its relative clause, so forward subject
//! selection excludes `god` by ORDER — the `!in_relative` filter is redundant
//! for these fixtures' subject picks. What actually bears load on 3:1 is
//! CLAUSE SEGMENTATION (delete the `and` boundary and resolution fails
//! entirely — asserted as a counterfactual test), and the span filter's
//! discriminative power is proven on a synthetic clause where order does NOT
//! protect it (`relative_span_filter_can_fire`).
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
/// emits). **Deliberately contains NO gold field** — the referee's answers
/// live in the separate gold list a fixture returns, so the resolvers (which
/// receive only `&[Tok]`) cannot reach them even by a future shortcut.
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
}

/// The referee's answer sheet: `(pronoun_pos, gold_antecedent_pos)` pairs.
/// Read ONLY by assert arms; never passed to a resolver.
type Gold = Vec<(usize, usize)>;

/// Look up the gold antecedent for a pronoun position.
fn gold_of(gold: &Gold, p: usize) -> Option<usize> {
    gold.iter().find(|&&(q, _)| q == p).map(|&(_, t)| t)
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
    fn pron(label: &'static str, plural: bool) -> Self {
        Tok {
            pos: Pos::Pron,
            plural,
            animate: true,
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
/// woman"*. The relative span covers "which … made". Gold (returned separately,
/// never inside `Tok`): `he`@19 → serpent@2.
fn genesis_3_1() -> (Vec<Tok>, Gold) {
    let toks = vec![
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
        Tok::pron("he", false),              // 19 → serpent@2 (d = -17)
        Tok::verb("said", true),             // 20
        Tok::t("unto"),                      // 21
        Tok::t("the"),                       // 22
        Tok::noun("woman", false, true),     // 23
    ];
    (toks, vec![(19, 2)])
}

/// Gen 3:7 (KJV, lowercased, trimmed): *"and the eyes of them both were
/// opened, and they knew that they were naked"*. Gold (returned separately,
/// never inside `Tok`): matrix `they`@9 → `them`@4 (the pair, not the eyes);
/// inner `they`@12 → matrix `they`@9 (the reflexive signature — knower ==
/// overt inner subject).
fn genesis_3_7() -> (Vec<Tok>, Gold) {
    let toks = vec![
        Tok::t("and"),                  // 0
        Tok::t("the"),                  // 1
        Tok::noun("eyes", true, false), // 2  (plural, inanimate)
        Tok::t("of"),                   // 3
        Tok::noun("them", true, true),  // 4  (referential pronoun-as-candidate)
        Tok::t("both"),                 // 5
        Tok::verb("were", false),       // 6
        Tok::verb("opened", false),     // 7
        Tok::conj("and"),               // 8
        Tok::pron("they", true),        // 9  → them@4 (d = -5)
        Tok::verb("knew", true),        // 10 (knowing requires an animate subject)
        Tok {
            comp: true,
            ..Tok::t("that")
        }, // 11
        Tok::pron("they", true),        // 12 → matrix they@9 (d = -3)
        Tok::verb("were", false),       // 13
        Tok::t("naked"),                // 14
    ];
    (toks, vec![(9, 4), (12, 9)])
}

/// A constructed in-range interposition (built English, NOT KJV — labelled
/// honestly): *"the man which the boy saw slept, and he smiled"*. Added for
/// codex's P1: both resolvers' answers are in-window and DIVERGE — recency →
/// `boy`@4 (d = −4, wrong), binding → `man`@1 (d = −7, gold) — and the
/// structural answer BINDS, so the STORED nibble itself differs from what
/// recency would reconstruct.
fn interposed_in_range() -> (Vec<Tok>, Gold) {
    let toks = vec![
        Tok::t("the"),                       // 0
        Tok::noun("man", false, true),       // 1
        Tok::t("which").rel(),               // 2
        Tok::t("the").rel(),                 // 3
        Tok::noun("boy", false, true).rel(), // 4
        Tok::verb("saw", true).rel(),        // 5
        Tok::verb("slept", true),            // 6
        Tok::conj("and"),                    // 7
        Tok::pron("he", false),              // 8 → man@1 (d = -7)
        Tok::verb("smiled", true),           // 9
    ];
    (toks, vec![(8, 1)])
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
    let (a, gold_a) = genesis_3_1();
    let he = 19;
    let h_a = heuristic_resolve(&a, he);
    let s_a = structural_resolve(&a, he);

    // B1 — can-fire: the resolvers genuinely disagree, and gold sides with
    // binding. Falsifier: agreement, or the heuristic being right.
    gate(
        "B1 divergence",
        h_a == Some(15) && s_a == Some(2) && gold_of(&gold_a, he) == Some(2) && h_a != s_a,
        format!(
            "heuristic(he@19) = {:?} ({}), binding = {:?} ({}), gold = serpent@2",
            h_a,
            h_a.map_or("-", |i| a[i].label),
            s_a,
            s_a.map_or("-", |i| a[i].label),
        ),
    );

    // B2 — the temptation is real, and the chip refuses it. Proven at the
    // BINDER, not by re-deriving its range predicate (CodeRabbit hardening):
    // binding the heuristic's target into scratch rows must SUCCEED — the
    // binder accepts and stores the wrong-but-in-window nibble (-4) — while
    // binding the structural answer (d = -17) must escalate, leaving the real
    // rows unbound. The chip's refusal is a choice, not a range limitation.
    let mut rows_tempted = fresh_rows(a.len());
    let tempted = bind(&mut rows_tempted, he, h_a.unwrap());
    let tempted_nibble = WitnessLens::new(&rows_tempted)
        .at(he)
        .map(|f| f.at(Locus::Antecedent));
    let mut rows_a = fresh_rows(a.len());
    let bound = bind(&mut rows_a, he, s_a.unwrap());
    let lens_a = WitnessLens::new(&rows_a);
    let nibble = lens_a.at(he).map(|f| f.at(Locus::Antecedent));
    gate(
        "B2 escalate-not-clamp",
        tempted == Ok(-4)
            && tempted_nibble == Some(-4)
            && bound == Err(-17)
            && nibble == Some(0),
        format!(
            "binder ACCEPTS the heuristic's wrong target ({tempted:?}, nibble {tempted_nibble:?} on scratch rows) — storable-but-wrong is real; binding d=-17 → escalated, nibble={nibble:?}"
        ),
    );

    // ── Fixture B: Gen 3:7 — the stay-silent text + chip composition ─────────
    let (b, gold_b) = genesis_3_7();
    let (matrix, inner) = (9, 12);
    let mut rows_b = fresh_rows(b.len());
    let mut agree = true;
    let mut displacements = Vec::new();
    for &p in &[matrix, inner] {
        let h = heuristic_resolve(&b, p);
        let s = structural_resolve(&b, p);
        agree &= h == s && s == gold_of(&gold_b, p);
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

    // ── Fixture C: constructed in-range interposition — the divergent CHIP ───
    // (codex P1 on this PR: with only fixtures A and B, every STORED chip came
    // from the agreeing verse, so stored state was heuristic-reconstructible
    // and the pull-test could not establish its central claim.)
    let (c, gold_c) = interposed_in_range();
    let hep = 8;
    let h_c = heuristic_resolve(&c, hep);
    let s_c = structural_resolve(&c, hep);
    let mut rows_c = fresh_rows(c.len());
    let chip_c = bind(&mut rows_c, hep, s_c.unwrap());
    let stored_c = WitnessLens::new(&rows_c)
        .at(hep)
        .map(|f| f.at(Locus::Antecedent));
    let heuristic_offset = h_c.map(|t| t as i8 - hep as i8);

    // B4 — the pull-test on a DIVERGENT stored chip: both answers in-window,
    // resolvers disagree, gold sides with binding, and the STORED nibble (-7)
    // differs from what recency would reconstruct (-4). Pull this chip and the
    // cheap resolver rebuilds the WRONG antecedent — the stored state itself
    // carries bits no better cheap algorithm recomputes.
    gate(
        "B4 divergent chip",
        h_c == Some(4)
            && s_c == Some(1)
            && gold_of(&gold_c, hep) == Some(1)
            && chip_c == Ok(-7)
            && stored_c == Some(-7)
            && heuristic_offset == Some(-4)
            && stored_c != heuristic_offset,
        format!(
            "recency → boy@4 (would store {heuristic_offset:?}), binding → man@1; STORED nibble {stored_c:?} ≠ heuristic reconstruction — divergence lives in stored state"
        ),
    );

    // B5 — chips COMPOSE: the inner pronoun's ultimate referent resolves by
    // following two stored nibbles (they@12 → they@9 → them@4) — each chip a
    // local warrant, no far verdict cached anywhere.
    let lens_b = WitnessLens::new(&rows_b);
    let hop1 = lens_b
        .at(inner)
        .and_then(|f| f.resolves_to(Locus::Antecedent, inner, lens_b.len()));
    let hop2 = hop1.and_then(|p| {
        lens_b
            .at(p)
            .and_then(|f| f.resolves_to(Locus::Antecedent, p, lens_b.len()))
    });
    gate(
        "B5 chips compose",
        hop1 == Some(matrix) && hop2 == Some(4),
        format!("chip chain they@12 → {hop1:?} → {hop2:?} (gold them@4) — two local warrants"),
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
        let (a, gold_a) = genesis_3_1();
        assert_eq!(heuristic_resolve(&a, 19), Some(15), "recency finds god");
        assert_eq!(structural_resolve(&a, 19), Some(2), "binding finds serpent");
        assert_eq!(gold_of(&gold_a, 19), Some(2));

        let (b, gold_b) = genesis_3_7();
        for p in [9, 12] {
            assert_eq!(heuristic_resolve(&b, p), structural_resolve(&b, p));
            assert_eq!(structural_resolve(&b, p), gold_of(&gold_b, p));
        }
    }

    /// Deleting the animacy check must break 3:7's matrix resolution (the
    /// rule is load-bearing, not decoration): without it, R2 returns the
    /// previous clause's subject verbatim — `eyes`@2, which is wrong.
    #[test]
    fn animacy_check_is_load_bearing() {
        let (b, gold_b) = genesis_3_7();
        let k = clause_of(&b, 9);
        let bare_r2 = subject_of_clause(&b, k - 1);
        assert_eq!(bare_r2, Some(2), "bare subject-continuity picks eyes@2");
        assert_ne!(
            bare_r2,
            gold_of(&gold_b, 9),
            "…which is NOT the gold answer"
        );
        assert_eq!(
            structural_resolve(&b, 9),
            gold_of(&gold_b, 9),
            "the check repairs it"
        );
    }

    /// What ACTUALLY bears load on 3:1 is clause segmentation, not the span
    /// filter (codex P2: in head-first English `serpent`@2 precedes `god`@15,
    /// so subject selection excludes `god` by ORDER — clearing `in_relative`
    /// changes nothing on this fixture). The honest counterfactual: delete the
    /// clause boundary and structural resolution FAILS entirely.
    #[test]
    fn clause_segmentation_is_load_bearing() {
        let (mut a, gold_a) = genesis_3_1();
        assert_eq!(structural_resolve(&a, 19), gold_of(&gold_a, 19));
        a[18].new_main = false; // erase the "and" boundary
        assert_eq!(
            structural_resolve(&a, 19),
            None,
            "without segmentation the resolver has no previous clause to bind"
        );
        assert_ne!(None, gold_of(&gold_a, 19), "…and gold still exists");
    }

    /// The span filter's discriminative power, proven where ORDER does not
    /// protect it: a synthetic clause whose FIRST candidate sits inside a
    /// relative span. With the filter the subject is the second candidate;
    /// the unfiltered first-candidate pick differs. Can-fire, non-vacuous.
    #[test]
    fn relative_span_filter_can_fire() {
        let toks = vec![
            Tok::noun("ghost", false, true).rel(), // 0 — in-relative candidate FIRST
            Tok::noun("host", false, true),        // 1 — the real subject
        ];
        let filtered = subject_of_clause(&toks, 0);
        let unfiltered =
            (0..toks.len()).find(|&i| clause_of(&toks, i) == 0 && toks[i].is_candidate());
        assert_eq!(filtered, Some(1), "filter skips the in-relative candidate");
        assert_eq!(unfiltered, Some(0), "without it, the wrong token wins");
        assert_ne!(filtered, unfiltered, "the filter changes the outcome");
    }

    /// codex P1's demanded artifact: a DIVERGENT chip in stored state. The
    /// in-range interposition binds -7 while recency would reconstruct -4 —
    /// the stored nibble itself differs from the cheap reconstruction.
    #[test]
    fn divergent_chip_differs_from_heuristic_reconstruction() {
        let (c, gold_c) = interposed_in_range();
        let h = heuristic_resolve(&c, 8);
        let s = structural_resolve(&c, 8);
        assert_eq!(h, Some(4), "recency finds boy (wrong)");
        assert_eq!(s, Some(1), "binding finds man (gold)");
        assert_eq!(s, gold_of(&gold_c, 8));

        let mut rows = fresh_rows(c.len());
        assert_eq!(bind(&mut rows, 8, s.unwrap()), Ok(-7));
        let stored = WitnessLens::new(&rows)
            .at(8)
            .map(|f| f.at(Locus::Antecedent));
        assert_eq!(stored, Some(-7));
        let heuristic_reconstruction = h.map(|t| t as i8 - 8);
        assert_ne!(
            stored, heuristic_reconstruction,
            "the stored bit is exactly what recency cannot recompute"
        );
    }

    /// Out-of-window escalation leaves the nibble unbound — never a clamp.
    #[test]
    fn escalation_leaves_nibble_unbound() {
        let (a, _) = genesis_3_1();
        let mut rows = fresh_rows(a.len());
        assert_eq!(bind(&mut rows, 19, 2), Err(-17));
        let lens = WitnessLens::new(&rows);
        assert_eq!(lens.at(19).map(|f| f.at(Locus::Antecedent)), Some(0));
    }

    /// The binder ACCEPTS the heuristic's wrong-but-in-window target — the
    /// "storable-but-wrong" half of B2 proven at the binder itself, not by
    /// re-deriving its range predicate in the gate.
    #[test]
    fn binder_accepts_the_tempting_wrong_target() {
        let (a, gold_a) = genesis_3_1();
        let mut rows = fresh_rows(a.len());
        assert_eq!(bind(&mut rows, 19, 15), Ok(-4), "god@15 binds cleanly");
        let lens = WitnessLens::new(&rows);
        assert_eq!(lens.at(19).map(|f| f.at(Locus::Antecedent)), Some(-4));
        assert_ne!(
            Some(15),
            gold_of(&gold_a, 19),
            "…and it is the WRONG target"
        );
    }
}
