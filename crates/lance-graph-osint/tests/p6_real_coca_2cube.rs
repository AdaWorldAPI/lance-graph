//! P6 · D-DNV-2 — the SPO → `CausalEdge64` → 2³ Pearl decomposition, driven by a
//! REAL deepnsm COCA FSM parse (not synthetic indices).
//!
//! The prior probes exercised the pieces on manufactured input: P2 packed PRNG
//! palette indices into `CausalEdge64`; P3b ran `all_projections` on hand-set
//! table cells. This probe closes the loop **end-to-end from text**:
//!
//! ```text
//!   real COCA text
//!     → Vocabulary::tokenize            (real COCA word ranks)
//!     → Parser::parse (6-state FSM)      → real SpoTriple (S, P, O ranks)
//!     → rank → centroid index            → CausalEdge64::pack_v2 (the edge)
//!     → SpoHead                          → SpoDistances::all_projections (2³)
//! ```
//!
//! The only stand-in is the distance palette — the trained centroid codebook is
//! future work (`E-V3-DEEPNSM-IS-THE-ENCODER-NOT-A-MIGRATION-1`); the S/P/O
//! **identity landing is real**, produced by the shipped FSM on real vocabulary.
//! DeepNSM → V3 D-DNV-2 (`.claude/plans/deepnsm-v3-convergence-v1.md`). Extends
//! #624 P2 (edge round-trip) and P3b (2³ amortization) onto real input.
//!
//! Integer-exact: the palette is a deterministic SplitMix64 fill, the parse is
//! deterministic, no seed entropy. Real COCA CSVs are committed under
//! `crates/deepnsm/word_frequency/`, so the load is CI-safe.

use causal_edge::{CausalEdge64, CausalMask, PlasticityState};
use deepnsm::parser::Parser;
use deepnsm::Vocabulary;
use lance_graph_planner::cache::nars_engine::{
    SpoDistances, SpoHead, ALL_MASKS, MASK_NONE, MASK_O, MASK_P, MASK_PO, MASK_S, MASK_SO, MASK_SPO,
};
use std::path::Path;

mod common;
use common::splitmix64;

/// A deterministic symmetric 256×256 palette — the codebook stand-in, same
/// shape p2/p3b use. Real trained centroids are future work (D-DNV-2 §stand-in).
/// Zero on the diagonal (a centroid is distance 0 from itself), positive off it.
fn synth_palette(seed: u64) -> Vec<u16> {
    let mut t = vec![0u16; 256 * 256];
    let mut s = seed;
    for a in 0..256usize {
        for b in (a + 1)..256usize {
            let v = 1 + (splitmix64(&mut s) % 60_000) as u16;
            t[a * 256 + b] = v;
            t[b * 256 + a] = v;
        }
    }
    t
}

fn nars() -> SpoDistances {
    SpoDistances {
        s_table: synth_palette(0x0700_0001),
        p_table: synth_palette(0x0700_0002),
        o_table: synth_palette(0x0700_0003),
    }
}

/// The COCA rank of the trained centroid the word lands on is future work; until
/// the codebook is trained, a word's rank maps to a centroid by its low byte.
/// Deterministic and total — the honest stand-in for `code = codebook(word)[k]`.
fn centroid(rank: u16) -> u8 {
    (rank & 0xFF) as u8
}

/// Bridge a packed edge into the distance engine's head — the SAME palette
/// indices, nothing else (the P2 join point, reused verbatim).
fn head_of(e: CausalEdge64) -> SpoHead {
    let mut h = SpoHead::zero();
    h.s_idx = e.s_idx();
    h.p_idx = e.p_idx();
    h.o_idx = e.o_idx();
    h
}

/// Parse real text through the shipped COCA FSM and return the first SPO triple
/// that carries an object, as `(subject, predicate, object)` real word ranks.
fn first_real_triple(vocab: &Vocabulary, text: &str) -> (u16, u16, u16) {
    let toks = vocab.tokenize(text);
    let parsed = Parser::new().parse(&toks);
    let t = parsed
        .triples
        .iter()
        .find(|t| t.has_object())
        .expect("a real COCA sentence yields at least one SPO triple with an object");
    (t.subject(), t.predicate(), t.object())
}

fn load_vocab() -> Vocabulary {
    // osint CARGO_MANIFEST_DIR is crates/lance-graph-osint; the COCA CSVs live
    // in the sibling deepnsm crate.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../deepnsm/word_frequency");
    Vocabulary::load(&dir).expect("load committed COCA word_frequency CSVs")
}

fn idx(m: u8) -> usize {
    ALL_MASKS.iter().position(|&x| x == m).unwrap()
}

/// P6a — a real COCA parse's S/P/O identity survives the `CausalEdge64` carrier
/// unchanged (extends P2's PRNG round-trip onto real, FSM-derived words).
#[test]
fn p6_real_coca_spo_round_trips_the_edge_carrier() {
    let vocab = load_vocab();
    let (s, p, o) = first_real_triple(&vocab, "The system resolves the reference.");
    let (si, pi, oi) = (centroid(s), centroid(p), centroid(o));

    let edge = CausalEdge64::pack_v2(
        si,
        pi,
        oi,
        200, // freq — Grok-grade truth ≈0.78, the COCA-wire stand-in
        163, // conf
        CausalMask::SPO,
        0,
        PlasticityState::ALL_FROZEN,
    );

    assert_eq!(edge.s_idx(), si, "real subject centroid round-trips");
    assert_eq!(edge.p_idx(), pi, "real predicate centroid round-trips");
    assert_eq!(edge.o_idx(), oi, "real object centroid round-trips");
    assert_eq!(edge.causal_mask(), CausalMask::SPO);
    assert_eq!(edge.frequency_u8(), 200);
    assert_eq!(edge.confidence_u8(), 163);
}

/// P6b — the 8 Pearl projections run over a head built from a REAL COCA parse,
/// and the 2³ ladder is monotone with the counterfactual (SPO) dominating
/// (extends P3b's hand-set-cell amortization onto real, FSM-derived heads).
#[test]
fn p6_real_coca_2cube_ladder_holds_on_a_real_parse() {
    let vocab = load_vocab();
    // Candidate = a real COCA FSM parse. Context = three high-frequency real
    // COCA words looked up by rank (guaranteed present in the top-4096 vocab) —
    // both heads are real COCA identities, no fragile second parse.
    let (cs, cp, co) = first_real_triple(&vocab, "The system resolves the reference.");
    let word_rank = |w: &str| {
        vocab
            .rank_of(w)
            .expect("common COCA word is in the top-4096 vocab")
    };
    let (xs, xp, xo) = (word_rank("time"), word_rank("people"), word_rank("world"));

    let cand = head_of(CausalEdge64::pack_v2(
        centroid(cs),
        centroid(cp),
        centroid(co),
        200,
        163,
        CausalMask::SPO,
        0,
        PlasticityState::ALL_FROZEN,
    ));
    let ctx = head_of(CausalEdge64::pack_v2(
        centroid(xs),
        centroid(xp),
        centroid(xo),
        200,
        163,
        CausalMask::SPO,
        0,
        PlasticityState::ALL_FROZEN,
    ));

    let d = nars();
    let proj = d.all_projections(&cand, &ctx);

    // Amortization + taxonomy: prior is 0; the counterfactual is the full sum.
    assert_eq!(proj[idx(MASK_NONE)], 0, "MASK_NONE = prior");
    assert_eq!(
        proj[idx(MASK_SPO)],
        proj[idx(MASK_S)] + proj[idx(MASK_P)] + proj[idx(MASK_O)],
        "MASK_SPO = counterfactual = the three marginals summed"
    );

    // The 2³ lattice is monotone: adding a plane never decreases the projection.
    assert!(proj[idx(MASK_NONE)] <= proj[idx(MASK_S)]);
    assert!(proj[idx(MASK_S)] <= proj[idx(MASK_SO)]);
    assert!(proj[idx(MASK_SO)] <= proj[idx(MASK_SPO)]);
    assert!(proj[idx(MASK_O)] <= proj[idx(MASK_PO)]);
    assert!(proj[idx(MASK_PO)] <= proj[idx(MASK_SPO)]);
    assert_eq!(
        proj[idx(MASK_SPO)],
        *proj.iter().max().unwrap(),
        "counterfactual dominates every sub-question on the real-derived head"
    );

    // The real FSM parse produced a genuine 3-role triple whose S and O landed on
    // distinct centroids, so Association (S+O) strictly exceeds its parts here —
    // the rung decomposition is doing real work, not collapsing to one number.
    if cand.s_idx != ctx.s_idx && cand.o_idx != ctx.o_idx {
        assert!(
            proj[idx(MASK_SO)] > proj[idx(MASK_S)],
            "Association adds the Object plane on top of the Subject plane"
        );
        assert!(
            proj[idx(MASK_SO)] > proj[idx(MASK_O)],
            "Association adds the Subject plane on top of the Object plane"
        );
    }
}
