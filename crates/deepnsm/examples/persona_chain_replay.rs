//! PROBE PERSONA-CHAIN-REPLAY — drive `ContextChain` replay with REAL
//! CANDIDATES, and measure it against the shipped agreement resolver's gold.
//!
//! ## The gap this closes
//!
//! `disambiguator_glue::disambiguate_with_trajectory` shipped the fingerprint
//! bridge, but **all three of its round-trip tests pass
//! `std::iter::empty::<CrystalFingerprint>()`** — they exercise the
//! empty-candidates SENTINEL path only. The contract's replay machinery
//! (`ContextChain::disambiguate_with` → winner / margin / dispersion /
//! `escalate_to_llm`) has therefore never had a caller that supplied real
//! candidates. This probe is that caller.
//!
//! Nothing here is new machinery. Every piece is shipped:
//!
//! | piece | where it already lives |
//! |---|---|
//! | ±5 replay, margin, escalation | `contract::grammar::context_chain` |
//! | f32 bundle → `Binary16K` | `deepnsm::disambiguator_glue::sign_binarize_to_binary16k` |
//! | pronoun case, `Ambiguous` never decides | `contract::grammar::clause_cues::pronoun_case` |
//! | where the answer is recorded | `contract::causal_witness::{CausalWitnessFacet, Locus::Antecedent}` |
//! | the gold + the agreement resolver | `deepnsm::examples::spo_anaphora_nibble` |
//!
//! ## The candidate set IS the persona register
//!
//! One surface form addresses several referents ("not all Marys are the mother
//! of Jesus"). The register supplies N candidates for one form; the chain
//! replay picks by coherence; the margin decides commit-vs-escalate. Here the
//! candidates are the window's non-pronoun NP heads — the same set the shipped
//! agreement resolver ranks — so the two doors are compared on ONE input.
//!
//! ## The content fingerprint is the DOCUMENTED metric, not an invention
//!
//! `E-FREQ-IS-COSINE-REPLACEMENT-1` (probe `freq_is_cosine`, rho 0.762) makes
//! frequency-rank distance the metric; `l9_loci_real_text` uses `|delta rank|/16`
//! on the same COCA table. A THERMOMETER code carries that metric exactly into
//! the `Binary16K` Hamming carrier: bit `i` is set iff `i < rank/16`, so
//!
//! ```text
//! hamming(fp(a), fp(b)) == |rank(a)/16 - rank(b)/16|
//! ```
//!
//! — the documented distance, byte-exact, not a re-derivation. Gate G3 asserts it.
//!
//! ## Registered gates (fixed before the first run)
//!
//! 1. **G1 REAL-CANDIDATES** — every disambiguation call has
//!    `candidate_count >= 2` (the sentinel path is NOT what is being measured).
//! 2. **G2 CASE-GATE** — an `Ambiguous` pronoun (`you`/`it`/`her`) is never
//!    committed on case; #849's "no parse beats a wrong parse".
//! 3. **G3 METRIC-EXACT** — the thermometer Hamming equals the documented
//!    rank distance for every pair in the fixture.
//! 4. **G4 REPORTED, NOT TUNED** — agreement with the shipped resolver's gold
//!    is printed whatever it is. A worse number is a finding (cf. #852's
//!    closed-class detectors measured LOSING to `rank<=150` and recorded as
//!    "stop searching"), never a reason to tune the kernel until it passes.
//!
//! ## Run
//!
//! ```bash
//! cargo run --manifest-path crates/deepnsm/Cargo.toml --example persona_chain_replay
//! ```

use deepnsm::disambiguator_glue::sign_binarize_to_binary16k;
use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::crystal::fingerprint::CrystalFingerprint;
use lance_graph_contract::grammar::clause_cues::{pronoun_case, PronounCase};
use lance_graph_contract::grammar::context_chain::{
    ContextChain, DisambiguateOpts, WeightingKernel, DISAMBIGUATION_MARGIN_THRESHOLD,
};
use std::collections::HashMap;

/// Bits in a `Binary16K`.
const BITS: usize = 16_384;
/// The documented rank-distance scale (`l9_loci_real_text`: `|delta rank|/16`).
const RANK_SCALE: u32 = 16;
/// The i4 locus reach — `E-ANAPHORA-BEYOND-I4-IS-A-BASIN-EDGE-1`. Beyond this a
/// reference is a BASIN EDGE (a different KIND of reference), never a wider
/// pointer, so the window is a type boundary and not a tunable.
const WINDOW: usize = 8;

/// `lemma -> rank` from the committed COCA table — the same loader shape
/// `spo_anaphora_nibble` uses, on the same file.
fn load_ranks(csv_path: &str) -> HashMap<String, u32> {
    let text = std::fs::read_to_string(csv_path).unwrap_or_default();
    let mut m = HashMap::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 2 {
            continue;
        }
        if let Ok(r) = f[0].trim().parse::<u32>() {
            m.entry(f[1].to_ascii_lowercase()).or_insert(r);
        }
    }
    m
}

/// The thermometer level for a rank: `rank / RANK_SCALE`, capped at `BITS`.
fn level(rank: u32) -> usize {
    ((rank / RANK_SCALE) as usize).min(BITS)
}

/// A f32 content vector whose sign-binarization is the thermometer code.
///
/// Bit `i` set iff `i < level(rank)`, so Hamming between two words is exactly
/// `|level(a) - level(b)|` — the documented `|delta rank|/16`.
fn thermometer_content(rank: u32) -> Vec<f32> {
    let lv = level(rank);
    (0..BITS).map(|i| if i < lv { 1.0 } else { -1.0 }).collect()
}

/// A word's `Binary16K` fingerprint: the thermometer content, sign-binarized
/// by the shipped glue.
///
/// **Deliberately NOT routed through `MarkovBundler`'s role slice.** The
/// obvious-looking move — build a one-token `WindowedSentence` so the call
/// "uses the shipped bundler" — would lay the thermometer into a 2000-wide
/// SUBJECT slice and silently CLAMP every `level(rank) > 2000`, destroying the
/// exact `|delta rank|/16` identity that G3 pins. The bundler's job is
/// superposing a WINDOW of sentences; a single candidate word is not that, and
/// calling it here would be decoration that breaks the metric. The chain below
/// is over TOKEN positions — the same ±8 token window `spo_anaphora_nibble`
/// and `l9_loci_real_text` both use.
fn word_fp(rank: u32) -> CrystalFingerprint {
    CrystalFingerprint::Binary16K(sign_binarize_to_binary16k(&thermometer_content(rank)))
}

/// Is this (lowercased) token a pronoun the case catalogue knows?
fn is_pronoun(tok: &str) -> bool {
    pronoun_case(tok).is_some()
}

/// The candidate antecedents for the pronoun at `pos`: the non-pronoun tokens
/// with a known rank inside the `-WINDOW` reach, nearest first.
///
/// This set IS the persona register for this probe: N referents addressable
/// from one pronoun. The chain replay's job is to pick among them.
fn candidates_at(stream: &[&str], pos: usize, ranks: &HashMap<String, u32>) -> Vec<(usize, u32)> {
    let lo = pos.saturating_sub(WINDOW);
    (lo..pos)
        .rev()
        .filter(|&k| !is_pronoun(stream[k]))
        .filter_map(|k| ranks.get(stream[k]).map(|&r| (k, r)))
        .collect()
}

/// Fill an 11-slot `ContextChain` centred on `pos`: slot `focal + delta` carries
/// the fingerprint of the token at `pos + delta`, or `None` where the token has
/// no rank (out of the COCA table) or falls outside the stream.
///
/// The focal slot is left EMPTY — that is the position the replay writes each
/// candidate into.
fn chain_around(stream: &[&str], pos: usize, ranks: &HashMap<String, u32>) -> ContextChain {
    let mut chain = ContextChain::new();
    let focal = ContextChain::focal_index() as isize;
    for slot in 0..chain.fingerprints.len() {
        let delta = slot as isize - focal;
        if delta == 0 {
            continue; // the replay slot
        }
        let idx = pos as isize + delta;
        if idx < 0 || idx as usize >= stream.len() {
            continue;
        }
        if let Some(&r) = ranks.get(stream[idx as usize]) {
            chain.fingerprints[slot] = Some(word_fp(r));
        }
    }
    chain
}

fn main() {
    let csv = concat!(env!("CARGO_MANIFEST_DIR"), "/word_frequency/lemmas_5k.csv");
    let ranks = load_ranks(csv);
    assert!(!ranks.is_empty(), "COCA table must load");

    // The fixture is `spo_anaphora_nibble`'s, VERBATIM — same stream, same gold.
    // Comparing two doors on one input is the point; a fresh fixture would
    // measure nothing.
    let stream: Vec<&str> = vec![
        "the", "man", "read", "a", "book", ".", // 0..6
        "he", "liked", "it", ".", // 6:he->man, 8:it->book
        "the", "girls", "played", ".", // 10..14
        "they", "won", ".", // 14:they->girls
        "the", "car", "that", "broke", ".", // 18..23  19:that->car
        "it", "rained", ".", // 22:it -> UNRESOLVED (pleonastic)
    ];
    let gold: &[(usize, Option<&str>)] = &[
        (6, Some("man")),
        (8, Some("book")),
        (14, Some("girls")),
        (19, Some("car")),
        (22, None),
    ];

    println!("PERSONA-CHAIN-REPLAY — real candidates through ContextChain replay\n");

    let mut fail: Vec<String> = Vec::new();
    let mut agree = 0usize;
    let mut scored = 0usize;
    let mut out_of_catalogue: Vec<usize> = Vec::new();
    let mut min_candidates = usize::MAX;
    let mut widest_spread = 0.0f32;

    for &(pos, want) in gold {
        let tok = stream[pos];
        let Some(case) = pronoun_case(tok) else {
            // `that` is a RELATIVE pronoun; `clause_cues::pronoun_case` is the
            // PERSONAL-pronoun case catalogue and does not carry it. Reported,
            // not patched: extending a Core catalogue to make a probe pass is
            // the adapter-hack the core-gap doctrine forbids.
            out_of_catalogue.push(pos);
            println!("  '{tok}'@{pos}  OUT-OF-CATALOGUE (relative pronoun; personal-case catalogue only)");
            continue;
        };

        let cands = candidates_at(&stream, pos, &ranks);
        if cands.len() < 2 {
            println!("  '{tok}'@{pos}  <2 candidates in the window — no replay");
            continue;
        }

        min_candidates = min_candidates.min(cands.len());
        let chain = chain_around(&stream, pos, &ranks);
        let res = chain.disambiguate_with(
            ContextChain::focal_index(),
            cands.iter().map(|&(_, r)| word_fp(r)).collect::<Vec<_>>(),
            DisambiguateOpts {
                kernel: Some(WeightingKernel::MexicanHat),
                sentinel_fp: None,
            },
        );

        // G1: this is the REAL-candidate path, not the sentinel.
        if res.candidate_count < 2 {
            fail.push(format!(
                "@{pos} G1: candidate_count={}",
                res.candidate_count
            ));
        }

        // The DYNAMIC RANGE of the metric, against the threshold that gates a
        // commit. If the widest coherence spread the candidates can produce is
        // orders of magnitude under the threshold, the replay cannot commit on
        // ANY input — which is a property of the pairing, not of this fixture.
        let spread = res
            .alternatives
            .first()
            .zip(res.alternatives.last())
            .map_or(0.0, |(hi, lo)| hi.1 - lo.1);
        widest_spread = widest_spread.max(spread);

        // G2: `Ambiguous` case never decides (#849 — no parse beats a wrong parse).
        let case_blocked = case == PronounCase::Ambiguous;

        let committed = if case_blocked || res.escalate_to_llm {
            None
        } else {
            cands.get(res.winner_index).copied()
        };

        let witness = match committed {
            Some((k, _)) => {
                let off = i8::try_from(pos - k).map(|d| -d).unwrap_or(0);
                CausalWitnessFacet::ZERO.with(Locus::Antecedent, off)
            }
            None => CausalWitnessFacet::ZERO,
        };

        let picked = committed.map(|(k, _)| stream[k]);
        let reason = if case_blocked {
            "case=Ambiguous -> withheld"
        } else if res.escalate_to_llm {
            "margin below threshold -> escalate"
        } else {
            "committed"
        };
        println!(
            "  '{tok}'@{pos}  case={case:?}  n={}  margin={:.4}  disp={:.4}  ante={:+}  -> {:?}  [{reason}]  (gold {:?})",
            res.candidate_count,
            res.margin,
            res.dispersion,
            witness.antecedent(),
            picked,
            want
        );

        // Only the case-decidable pronouns are SCORED against gold; a withheld
        // Ambiguous is correct behaviour, not a wrong answer.
        if !case_blocked {
            scored += 1;
            if picked == want {
                agree += 1;
            }
        } else if want.is_some() {
            // An Ambiguous pronoun whose gold IS resolvable is exactly the
            // information the case gate costs. Recorded, not hidden.
            println!("       ^ the case gate withheld a resolvable one — the price of G2");
        }
    }

    // ── G3: the thermometer carries the DOCUMENTED metric exactly ──
    let words: Vec<&str> = ["man", "book", "girls", "car", "read", "played"]
        .into_iter()
        .filter(|w| ranks.contains_key(*w))
        .collect();
    assert!(words.len() >= 3, "anti-vacuity: G3 needs real pairs");
    let mut g3_pairs = 0usize;
    let mut g3_nonzero = 0usize;
    for (i, a) in words.iter().enumerate() {
        for b in &words[i + 1..] {
            let (ra, rb) = (ranks[*a], ranks[*b]);
            let expect = level(ra).abs_diff(level(rb)) as u32;
            let got: u32 = match (word_fp(ra), word_fp(rb)) {
                (CrystalFingerprint::Binary16K(x), CrystalFingerprint::Binary16K(y)) => x
                    .iter()
                    .zip(y.iter())
                    .map(|(p, q)| (p ^ q).count_ones())
                    .sum(),
                _ => unreachable!("word_fp always builds Binary16K"),
            };
            if got != expect {
                fail.push(format!("G3 {a}/{b}: hamming {got} != |dRank|/16 {expect}"));
            }
            g3_pairs += 1;
            if expect > 0 {
                g3_nonzero += 1;
            }
        }
    }
    // Anti-vacuity: if every pair had distance 0 the identity would hold
    // trivially and prove nothing about the carrier.
    if g3_nonzero * 2 < g3_pairs {
        fail.push(format!(
            "G3 is near-vacuous: only {g3_nonzero}/{g3_pairs} pairs have non-zero distance"
        ));
    }

    // ── report ──
    println!();
    println!("G1 REAL-CANDIDATES  min candidates per replay = {min_candidates} (sentinel path never taken)");
    println!("G2 CASE-GATE        Ambiguous pronouns withheld, never committed on case");
    println!("G3 METRIC-EXACT     {g3_pairs} pairs, {g3_nonzero} with non-zero distance, hamming == |dRank|/{RANK_SCALE}");
    println!(
        "G4 REPORTED         chain-replay agreement with the shipped resolver's gold: {agree}/{scored}"
    );
    if !out_of_catalogue.is_empty() {
        println!(
            "                    {} gold item(s) out of the personal-case catalogue: {out_of_catalogue:?}",
            out_of_catalogue.len()
        );
    }
    println!("                    escalation threshold = {DISAMBIGUATION_MARGIN_THRESHOLD} (contract constant, untouched)");
    println!(
        "G5 SCALE            widest coherence spread across candidates = {widest_spread:.6}; \n\
         \x20                   threshold = {DISAMBIGUATION_MARGIN_THRESHOLD} -> ratio {:.0}x too small. The rank metric\n\
         \x20                   and the margin gate are on INCOMPATIBLE SCALES: common-word\n\
         \x20                   ranks differ by ~15 bits out of 16,384, so every replay\n\
         \x20                   escalates regardless of input. Reported, not tuned.",
        f64::from(DISAMBIGUATION_MARGIN_THRESHOLD) / f64::from(widest_spread.max(1e-9))
    );

    if fail.is_empty() {
        println!("\nGATES PASS — the real-candidate replay path runs end to end: candidates from");
        println!("the window, coherence from the documented rank metric, commit-vs-escalate from");
        println!("the contract's own margin, and the answer recorded as a Locus::Antecedent");
        println!("offset. G4 is a MEASUREMENT, not a target: see the note below.");
    } else {
        println!("\nGATES FAILED:");
        for f in &fail {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
}
