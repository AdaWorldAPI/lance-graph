//! `loci` — the adapter [`crate::wave`] declares missing: deriving
//! [`CausalWitnessFacet`] loci FROM the [`Spo`] triple stream.
//!
//! `wave.rs` states the gap twice, in its own words:
//!
//! > "Deriving loci FROM `Spo` triples is a separate adapter and is
//! > intentionally NOT bundled here, so this stays a clean resolver."
//!
//! > "**nothing in this repository populates a `WitnessStream` from text or
//! > from a real `NodeRow`.** Every `WitnessStream::new()` call site is
//! > `#[cfg(test)]` — there is no production adapter feeding real loci into
//! > this stream today."
//!
//! This module is that adapter. It reads the triple stream and writes ONE
//! locus — [`Locus::Antecedent`] — leaving all 23 others UNBOUND, because this
//! crate has grounded no evidence for them and a fabricated pointer is worse
//! than an absent one (`0` is the zero-fallback: *not consulted*).
//!
//! ## What decides the antecedent
//!
//! The candidate set for a pronoun subject is the non-pronoun subjects inside
//! the `-8` reach. Ranking uses **this crate's own trained meaning space** —
//! [`Cam96Space::similarity`] between a candidate's code and the pronoun
//! clause's PREDICATE code. That is selectional fit: of the referents in view,
//! which one is distributionally the kind of thing that does this verb.
//!
//! [`AnteRule::Recency`] (nearest candidate, ignoring meaning) is carried as
//! the declared BASELINE so the meaning-based rule is never reported without
//! the control it has to beat — the house pattern.
//!
//! ## Why the reach is 8 and not a tunable
//!
//! `E-ANAPHORA-BEYOND-I4-IS-A-BASIN-EDGE-1`: a locus nibble is a signed `i4`,
//! so it addresses `-8..=+7`, and that boundary is a TYPE boundary rather than
//! a size limit — inside it, position discriminates; beyond it, identity does,
//! so a wider pointer would still address a position when the discriminator
//! has changed kind. An antecedent further back is a BASIN EDGE, not an
//! overflow, so this adapter leaves the locus UNBOUND rather than clamping to
//! `-8` and naming the wrong event.
//!
//! ## Margin, and who decides
//!
//! When the best and second-best candidate score within [`MIN_MARGIN`], the
//! locus stays UNBOUND: the adapter refuses rather than committing on a
//! coin-flip, and `wave.rs`'s resolver then reads it as
//! `WaveGrounding::Unbound`. The adapter never invents an answer the evidence
//! did not separate.

use crate::space::{Cam96, Cam96Space};
use crate::spo::Spo;
use crate::wave::WitnessStream;
use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};

/// The `i4` locus reach — a TYPE boundary, not a tunable (see module doc).
pub const REACH: usize = 8;

/// Minimum separation between best and second-best candidate before the
/// adapter will bind a locus. Below this the evidence did not choose.
pub const MIN_MARGIN: f32 = 0.01;

/// Which rule ranks the candidate antecedents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnteRule {
    /// Nearest candidate wins. The declared BASELINE — carries no meaning.
    Recency,
    /// Highest [`Cam96Space::similarity`] between the candidate's code and the
    /// pronoun clause's PREDICATE code: selectional fit.
    SelectionalFit,
}

/// What the adapter did, so a caller never has to infer it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LociReport {
    /// Triples whose subject is a pronoun — a slot holding a pointer.
    pub pronoun_subjects: usize,
    /// …that had at least one candidate inside the reach.
    pub with_candidates: usize,
    /// …that bound `Locus::Antecedent`.
    pub bound: usize,
    /// …left UNBOUND because best and second-best were within `MIN_MARGIN`.
    pub refused_thin_margin: usize,
    /// …left UNBOUND because no candidate sat inside the `-8` reach (the
    /// antecedent is a BASIN EDGE, not an overflow).
    pub beyond_reach: usize,
}

/// Score one candidate under `rule`. Higher wins.
///
/// `Recency` scores by proximity alone. `SelectionalFit` reads the trained
/// space; a candidate or predicate without a code scores `f32::NEG_INFINITY`
/// so it can never win by default — an absent code is absent evidence.
fn score(
    rule: AnteRule,
    space: &Cam96Space,
    codes: &[Cam96],
    cand_subject: u16,
    predicate: u16,
    back: usize,
) -> f32 {
    match rule {
        // Nearest = highest. `back` is 1..=REACH.
        AnteRule::Recency => -(back as f32),
        AnteRule::SelectionalFit => {
            match (
                codes.get(cand_subject as usize),
                codes.get(predicate as usize),
            ) {
                (Some(c), Some(p)) => space.similarity(c, p),
                _ => f32::NEG_INFINITY,
            }
        }
    }
}

/// Derive loci from the triple stream and return the populated
/// [`WitnessStream`] plus what the derivation did.
///
/// One event per triple, at version = its stream index, so the append index IS
/// the `±8` offset space `wave.rs` resolves against.
///
/// `pronoun_by_id[id]` says whether vocabulary id `id` is a pronoun. An empty
/// slice means no pronoun information, and then nothing is bound — absent
/// evidence is not a licence to guess.
#[must_use]
pub fn loci_from_triples(
    space: &Cam96Space,
    codes: &[Cam96],
    pronoun_by_id: &[bool],
    triples: &[Spo],
    rule: AnteRule,
) -> (WitnessStream, LociReport) {
    let is_pronoun = |id: u16| pronoun_by_id.get(id as usize).copied().unwrap_or(false);
    let mut stream = WitnessStream::new();
    let mut rep = LociReport::default();

    for (i, t) in triples.iter().enumerate() {
        let mut reg = CausalWitnessFacet::ZERO;

        if is_pronoun(t.subject) {
            rep.pronoun_subjects += 1;

            // Candidates: non-pronoun subjects inside the -8 reach, nearest first.
            let mut scored: Vec<(usize, f32)> = (1..=REACH)
                .filter_map(|back| i.checked_sub(back).map(|j| (back, j)))
                .filter(|&(_, j)| !is_pronoun(triples[j].subject))
                .map(|(back, j)| {
                    (
                        back,
                        score(rule, space, codes, triples[j].subject, t.predicate, back),
                    )
                })
                .filter(|&(_, s)| s > f32::NEG_INFINITY)
                .collect();

            if scored.is_empty() {
                rep.beyond_reach += 1;
            } else {
                rep.with_candidates += 1;
                // Descending by score; ties keep the nearer candidate because
                // the source iteration is nearest-first and the sort is stable.
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let margin = if scored.len() >= 2 {
                    scored[0].1 - scored[1].1
                } else {
                    f32::INFINITY
                };
                if margin < MIN_MARGIN {
                    rep.refused_thin_margin += 1;
                } else {
                    let back = i8::try_from(scored[0].0).unwrap_or(i8::MAX);
                    // Negative: `-` is before / antecedent.
                    reg = reg.with(Locus::Antecedent, -back);
                    rep.bound += 1;
                }
            }
        }

        stream.push(i as u64, reg);
    }

    (stream, rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A codebook where similarity is controllable: code `[k;12]` reconstructs
    /// to a point whose distance from `[m;12]` grows with `|k-m|`.
    fn space() -> Cam96Space {
        Cam96Space::demo(4)
    }

    fn codes(n: usize) -> Vec<Cam96> {
        (0..n).map(|i| [(i * 7 % 251) as u8; 12]).collect()
    }

    fn mask(n: usize, ps: &[u16]) -> Vec<bool> {
        (0..n).map(|i| ps.contains(&(i as u16))).collect()
    }

    /// The guard FIRES: a pronoun subject one step after a real subject binds
    /// `Antecedent = -1`, and `wave`'s own resolver grounds it Causal.
    #[test]
    fn a_pronoun_subject_binds_and_the_wave_grounds_it() {
        let triples = vec![Spo::new(1, 100, 200), Spo::new(9, 101, 201)];
        let (stream, rep) = loci_from_triples(
            &space(),
            &codes(256),
            &mask(16, &[9]),
            &triples,
            AnteRule::Recency,
        );
        assert_eq!(rep.pronoun_subjects, 1);
        assert_eq!(rep.bound, 1);
        // The stream `wave.rs` owns resolves the locus this adapter wrote.
        assert_eq!(
            stream.ground_at(1, Locus::Antecedent, 99, 4),
            lance_graph_contract::witness_fabric::WaveGrounding::Causal
        );
        let r = stream
            .resolve_at(1, Locus::Antecedent, 99, 1)
            .expect("focal visible");
        assert_eq!(r.final_offset, Some(-1));
    }

    /// The guard STAYS SILENT on non-trivial input: a stream of real subjects
    /// binds nothing. Not an empty stream — an empty one would prove only that
    /// the loop handles emptiness.
    #[test]
    fn real_subjects_bind_nothing() {
        let triples = vec![
            Spo::new(1, 100, 200),
            Spo::new(2, 101, 201),
            Spo::new(3, 102, 202),
        ];
        let (stream, rep) = loci_from_triples(
            &space(),
            &codes(256),
            &mask(16, &[9]),
            &triples,
            AnteRule::Recency,
        );
        assert_eq!(triples.len(), 3, "anti-vacuity: there IS a stream");
        assert_eq!(rep.pronoun_subjects, 0);
        assert_eq!(rep.bound, 0);
        for pos in 0..3 {
            assert_eq!(
                stream.ground_at(pos, Locus::Antecedent, 99, 4),
                lance_graph_contract::witness_fabric::WaveGrounding::Unbound
            );
        }
    }

    /// Past the `i4` reach the locus stays UNBOUND rather than clamping to -8
    /// and naming the wrong event: the antecedent has become a BASIN EDGE.
    #[test]
    fn beyond_the_reach_is_unbound_not_clamped() {
        // One real subject, then 9 pronoun subjects. The 9th cannot see it.
        let mut triples = vec![Spo::new(1, 100, 200)];
        triples.extend((0..9).map(|i| Spo::new(9, 110 + i, 210 + i)));
        let (stream, rep) = loci_from_triples(
            &space(),
            &codes(256),
            &mask(16, &[9]),
            &triples,
            AnteRule::Recency,
        );
        assert_eq!(rep.pronoun_subjects, 9);
        assert_eq!(rep.bound, 8, "the eight that fit");
        assert_eq!(rep.beyond_reach, 1, "the ninth is a basin edge");
        assert_eq!(
            stream.ground_at(9, Locus::Antecedent, 99, 4),
            lance_graph_contract::witness_fabric::WaveGrounding::Unbound
        );
    }

    /// No pronoun information binds nothing — and the SAME input with a mask
    /// does bind, so this is a real difference and not an unbindable stream.
    #[test]
    fn absent_evidence_is_not_a_licence_to_guess() {
        let triples = vec![Spo::new(1, 100, 200), Spo::new(9, 101, 201)];
        let (_, bare) = loci_from_triples(&space(), &codes(256), &[], &triples, AnteRule::Recency);
        assert_eq!(bare.bound, 0);
        assert_eq!(bare.pronoun_subjects, 0);
        let (_, with) = loci_from_triples(
            &space(),
            &codes(256),
            &mask(16, &[9]),
            &triples,
            AnteRule::Recency,
        );
        assert_eq!(with.bound, 1);
    }

    /// Only `Antecedent` is written. A fabricated TEKAMOLO or Kausal pointer
    /// would be indistinguishable downstream from a grounded one.
    #[test]
    fn no_locus_but_antecedent_is_ever_bound() {
        let triples = vec![Spo::new(1, 100, 200), Spo::new(9, 101, 201)];
        let (stream, _) = loci_from_triples(
            &space(),
            &codes(256),
            &mask(16, &[9]),
            &triples,
            AnteRule::Recency,
        );
        let w = stream.window_range(lance_graph_contract::temporal_pov::VersionRange::new(0, 99));
        assert!(w.iter().any(|(_, f)| f.bound_count() == 1), "one is bound");
        for (_, f) in &w {
            for l in Locus::ALL {
                if l != Locus::Antecedent {
                    assert_eq!(f.at(l), 0, "{l:?} must stay unbound");
                }
            }
        }
    }

    /// The two rules are genuinely different: on an input where the nearest
    /// candidate is NOT the best selectional fit, they pick different offsets.
    /// Without this the baseline would be decorative.
    #[test]
    fn selectional_fit_can_disagree_with_recency() {
        let sp = space();
        let cds = codes(256);
        // predicate id 5; candidate A (id 3) nearer, candidate B (id 40) farther.
        // Whichever the trained space prefers, the two rules must be capable of
        // disagreeing — assert that they DO on at least one arrangement.
        let mut disagreed = false;
        for far in [20u16, 40, 60, 80, 100] {
            let triples = vec![
                Spo::new(far, 1, 2),
                Spo::new(3, 1, 2),
                Spo::new(9, 5, 2), // pronoun subject, predicate 5
            ];
            let m = mask(256, &[9]);
            let (a, _) = loci_from_triples(&sp, &cds, &m, &triples, AnteRule::Recency);
            let (b, _) = loci_from_triples(&sp, &cds, &m, &triples, AnteRule::SelectionalFit);
            let ra = a
                .resolve_at(2, Locus::Antecedent, 99, 1)
                .unwrap()
                .final_offset;
            let rb = b
                .resolve_at(2, Locus::Antecedent, 99, 1)
                .unwrap()
                .final_offset;
            if ra != rb {
                disagreed = true;
                break;
            }
        }
        assert!(
            disagreed,
            "the rules never disagreed — the baseline would be decorative"
        );
    }
}
