//! `loci_recipe_relevance` — the 24 L9 loci × 34 NARS recipes relevance matrix.
//!
//! The L9 `TekamoloWindowBinding` facet (plan §2.9) is 24 signed nibbles, each a
//! **locus**: a signed context pointer that places ONE awareness dimension's
//! filler in the ±8 `temporal.rs` Markov window. 16 of the 24 are the
//! operator-named dimensions (grounded in shipped organs); 8 stay reserved-empty
//! (RESERVE-DON'T-RECLAIM — never padded to hit a count).
//!
//! This example answers the operator's question — *"the 24 pointers seem to have
//! a very high relative relevance"* — by **deriving**, per (locus, recipe) cell,
//! how relevant that awareness pointer is to that reasoning runbook. The score is
//! a DOCUMENTED STRUCTURAL HEURISTIC over the REAL [`RECIPES`] metadata (the
//! `substrate` / `name` strings + `mechanism` / `spo2cubed` / `bucket` / `tier`
//! fields) — a design-time relevance map, NOT a runtime measurement. Every rule
//! is spelled out in [`keyword_hits`] and [`field_bonus`] so the derivation is
//! auditable, not a guess.
//!
//! Row-sums rank the 24 loci by their spread across the 34 runbooks (which
//! pointers earn their nibble); column-sums rank the 34 runbooks by how much
//! window-awareness they consume. Both feed the rung-escalation order: a runbook
//! that lights many loci is one the escalation ladder must hand a rich A9 view.
//!
//! ```sh
//! cargo run -p lance-graph-contract --example loci_recipe_relevance
//! ```

use lance_graph_contract::recipes::{Bucket, Coverage, Mechanism, Recipe, Tier, RECIPES};

/// One L9 locus dimension — a named awareness pointer the A9 register can bind.
struct Locus {
    /// Slot index 0..24 in the `TekamoloWindowBinding` register.
    slot: usize,
    /// Operator-named dimension (plan §2.9), or `""` for a reserved-empty slot.
    name: &'static str,
    /// Which shipped organ grounds this pointer (§2.9 "Grounded in" column).
    grounded_in: &'static str,
    /// Lowercase substrings that, when present in a recipe's `substrate`/`name`,
    /// signal this locus is load-bearing for that runbook. Documented rule set.
    keywords: &'static [&'static str],
}

/// The 24-slot L9 catalogue: 16 named (plan §2.9 table) + 8 reserved-empty.
/// Reserved slots carry NO keywords and score 0 everywhere — they are held open
/// (RESERVE-DON'T-RECLAIM), never padded with a construct to reach 24.
const LOCI: [Locus; 24] = [
    Locus {
        slot: 0,
        name: "TEMPORAL",
        grounded_in: "role_keys::TEMPORAL_KEY",
        keywords: &["temporal", "markov", "granger", "time", "±5", "cascade"],
    },
    Locus {
        slot: 1,
        name: "KAUSAL",
        grounded_in: "role_keys::KAUSAL_KEY",
        keywords: &[
            "causal",
            "cause",
            "abduction",
            "reverse",
            "counterfactual",
            "granger",
            "intervention",
            "reason",
        ],
    },
    Locus {
        slot: 2,
        name: "MODAL",
        grounded_in: "role_keys::MODAL_KEY",
        keywords: &[
            "uncertain",
            "confidence",
            "calibration",
            "brier",
            "possibility",
            "meta-cognition",
            "skeptic",
        ],
    },
    Locus {
        slot: 3,
        name: "LOKAL",
        grounded_in: "role_keys::LOKAL_KEY",
        keywords: &[
            "context",
            "window",
            "bindspace",
            "episodic",
            "latent space",
            "cluster",
            "scaffold",
        ],
    },
    Locus {
        slot: 4,
        name: "S-meaning",
        grounded_in: "SPO plane (A1)",
        keywords: &["spo", "subject", "agent", "s_o", "0b111", "triple"],
    },
    Locus {
        slot: 5,
        name: "P-meaning",
        grounded_in: "SPO plane (A1)",
        keywords: &["spo", "predicate", "action", "relation", "rel", "bind"],
    },
    Locus {
        slot: 6,
        name: "O-meaning",
        grounded_in: "SPO plane (A1)",
        keywords: &["spo", "object", "patient", "s_o", "0b111"],
    },
    Locus {
        slot: 7,
        name: "antecedent",
        grounded_in: "MODIFIER/CONTEXT keys",
        keywords: &[
            "reframe",
            "analog",
            "mapping",
            "reference",
            "intent",
            "roleplay",
        ],
    },
    Locus {
        slot: 8,
        name: "basin-anchor",
        grounded_in: "part_of:is_a (L1)",
        keywords: &[
            "clam",
            "cluster",
            "decompos",
            "hierarch",
            "abstraction",
            "knowledge",
            "is_a",
        ],
    },
    Locus {
        slot: 9,
        name: "supported-by",
        grounded_in: "hi_chain",
        keywords: &[
            "cascade",
            "hierarch",
            "decompos",
            "evidence",
            "scaffold",
            "abstraction",
            "coarse",
        ],
    },
    Locus {
        slot: 10,
        name: "supports",
        grounded_in: "lo_chain",
        keywords: &[
            "fusion",
            "compose",
            "synthesis",
            "aggregate",
            "bundle",
            "expansion",
            "fuse",
        ],
    },
    Locus {
        slot: 11,
        name: "runbook-evidence",
        grounded_in: "RECIPES[34] / A8",
        keywords: &[
            "template", "prompt", "scaffold", "style", "td-learn", "q-value", "slots",
        ],
    },
    Locus {
        slot: 12,
        name: "qualia-reference",
        grounded_in: "QualiaColumn / i4-qualia",
        keywords: &[
            "staunen",
            "qualia",
            "temperature",
            "perturb",
            "noise",
            "wisdom",
            "texture",
        ],
    },
    Locus {
        slot: 13,
        name: "meaning-level",
        grounded_in: "rung-content ladder 0–4",
        keywords: &[
            "rung",
            "depth",
            "abstraction",
            "expand",
            "compress",
            "scaling",
            "level",
            "meta",
        ],
    },
    Locus {
        slot: 14,
        name: "quorum",
        grounded_in: "NARS freq·conf (A3)",
        keywords: &[
            "agreement",
            "vote",
            "consensus",
            "majority",
            "debate",
            "council",
            "raid",
            "ecc",
            "independent",
            "corrob",
        ],
    },
    Locus {
        slot: 15,
        name: "contradiction",
        grounded_in: "Staunen×Wisdom depth",
        keywords: &[
            "contradiction",
            "dissonance",
            "opposing",
            "adversar",
            "critique",
            "skeptic",
            "distortion",
            "reciprocal",
            "challenge",
            "negation",
        ],
    },
    // ── reserved-empty (RESERVE-DON'T-RECLAIM; never padded) ──
    Locus {
        slot: 16,
        name: "",
        grounded_in: "reserved",
        keywords: &[],
    },
    Locus {
        slot: 17,
        name: "",
        grounded_in: "reserved",
        keywords: &[],
    },
    Locus {
        slot: 18,
        name: "",
        grounded_in: "reserved",
        keywords: &[],
    },
    Locus {
        slot: 19,
        name: "",
        grounded_in: "reserved",
        keywords: &[],
    },
    Locus {
        slot: 20,
        name: "",
        grounded_in: "reserved",
        keywords: &[],
    },
    Locus {
        slot: 21,
        name: "",
        grounded_in: "reserved",
        keywords: &[],
    },
    Locus {
        slot: 22,
        name: "",
        grounded_in: "reserved",
        keywords: &[],
    },
    Locus {
        slot: 23,
        name: "",
        grounded_in: "reserved",
        keywords: &[],
    },
];

/// Documented rule #1 — keyword hits. Count DISTINCT locus keywords present in
/// the recipe's `substrate` + `name` (lowercased). Capped at 2 so no single
/// verbose substrate string dominates.
fn keyword_hits(locus: &Locus, r: &Recipe) -> u8 {
    if locus.keywords.is_empty() {
        return 0;
    }
    let hay = format!("{} {}", r.substrate.to_lowercase(), r.name.to_lowercase());
    let hits = locus.keywords.iter().filter(|k| hay.contains(**k)).count();
    hits.min(2) as u8
}

/// Documented rule #2 — structural field bonus (+1 each, from the recipe's typed
/// metadata, NOT string matching). Encodes the substrate invariants: SPO-2³
/// coverage makes the causality + S/P/O loci load-bearing; TruthAwareInference
/// makes the quorum/contradiction peers load-bearing; a Control bucket is where
/// the escalation-facing loci (meaning-level, runbook-evidence) matter.
fn field_bonus(locus: &Locus, r: &Recipe) -> u8 {
    let mut b = 0u8;
    match locus.name {
        "KAUSAL" | "S-meaning" | "P-meaning" | "O-meaning" => {
            b += match r.spo2cubed {
                Coverage::Covered => 2,
                Coverage::Partial => 1,
                Coverage::NotCovered => 0,
            };
        }
        "quorum" => {
            b += u8::from(matches!(
                r.mechanism,
                Mechanism::ParallelIndependence | Mechanism::TruthAwareInference
            ));
        }
        "contradiction" => {
            b += u8::from(matches!(
                r.mechanism,
                Mechanism::TruthAwareInference | Mechanism::StructuralDivergence
            ));
        }
        "meaning-level" | "runbook-evidence" => {
            b += u8::from(matches!(r.bucket, Bucket::Control));
        }
        "qualia-reference" => {
            // Staunen (surprise/temperature) rides the Gate bucket + the hardest tiers.
            b +=
                u8::from(matches!(r.bucket, Bucket::Gate) || matches!(r.tier, Tier::ExtremelyHard));
        }
        _ => {}
    }
    b
}

/// Cell relevance 0..=3 = clamp(keyword_hits + field_bonus).
fn cell(locus: &Locus, r: &Recipe) -> u8 {
    (keyword_hits(locus, r) + field_bonus(locus, r)).min(3)
}

fn main() {
    // slot integrity: every LOCI entry's declared slot equals its array index.
    for (i, l) in LOCI.iter().enumerate() {
        assert_eq!(
            l.slot, i,
            "locus '{}' slot must equal its register index",
            l.name
        );
    }

    // ── the 24 × 34 matrix ──
    let mut matrix = [[0u8; 34]; 24];
    for (li, locus) in LOCI.iter().enumerate() {
        for (ri, r) in RECIPES.iter().enumerate() {
            matrix[li][ri] = cell(locus, r);
        }
    }

    // ── header ──
    println!("L9 loci × 34 NARS recipes — structural relevance matrix");
    println!(
        "(cell 0..3 = documented keyword hits + typed field bonus over REAL RECIPES metadata)"
    );
    println!();
    print!("{:<18}", "locus \\ recipe");
    for r in RECIPES.iter() {
        print!("{:>4}", r.code);
    }
    println!("{:>7}", "ΣROW");
    println!("{}", "─".repeat(18 + 34 * 4 + 7));

    // ── rows + row-sums ──
    let mut row_sum = [0u32; 24];
    for (li, locus) in LOCI.iter().enumerate() {
        let disp = if locus.name.is_empty() {
            "· reserved ·"
        } else {
            locus.name
        };
        print!("{:<18}", disp);
        for &v in &matrix[li] {
            row_sum[li] += v as u32;
            print!(
                "{:>4}",
                if v == 0 {
                    ".".to_string()
                } else {
                    v.to_string()
                }
            );
        }
        println!("{:>7}", row_sum[li]);
    }

    // ── column-sums ──
    println!("{}", "─".repeat(18 + 34 * 4 + 7));
    print!("{:<18}", "ΣCOL");
    let mut col_sum = [0u32; 34];
    for (ri, cs) in col_sum.iter_mut().enumerate() {
        for row in &matrix {
            *cs += row[ri] as u32;
        }
        print!("{:>4}", *cs);
    }
    let grand: u32 = row_sum.iter().sum();
    println!("{:>7}", grand);

    // ── locus ranking by row-sum (the "which pointers are high relative relevance") ──
    println!();
    println!("── Loci ranked by relative relevance (row-sum across all 34 runbooks) ──");
    let mut ranked: Vec<(usize, u32)> = (0..24).map(|i| (i, row_sum[i])).collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (rank, (li, s)) in ranked.iter().enumerate() {
        if *s == 0 {
            continue; // reserved-empty loci omitted from the ranking
        }
        let l = &LOCI[*li];
        println!(
            "  {:>2}. {:<17} Σ={:>3}   ← {}",
            rank + 1,
            l.name,
            s,
            l.grounded_in
        );
    }

    // ── recipe ranking by column-sum (which runbooks consume the most window-awareness) ──
    println!();
    println!("── Runbooks ranked by window-awareness consumed (col-sum over 24 loci) ──");
    let mut rcol: Vec<(usize, u32)> = (0..34).map(|i| (i, col_sum[i])).collect();
    rcol.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (rank, (ri, s)) in rcol.iter().take(10).enumerate() {
        let r = &RECIPES[*ri];
        println!("  {:>2}. {:<5} Σ={:>3}   {}", rank + 1, r.code, s, r.name);
    }

    // ═══ registered gates ═══
    println!();
    println!("── gates ──");
    let mut all_green = true;

    // Gate 1: every one of the 16 NAMED loci is load-bearing for ≥1 runbook
    //         (no dead pointer — each named dimension earns its nibble).
    let named_all_live = (0..16).all(|i| row_sum[i] > 0);
    println!(
        "[{}] G1 every named locus is relevant to ≥1 runbook (no dead pointer)",
        if named_all_live { "PASS" } else { "FAIL" }
    );
    all_green &= named_all_live;

    // Gate 2: every RESERVED locus scores exactly 0 (held empty, never padded).
    let reserved_empty = (16..24).all(|i| row_sum[i] == 0);
    println!(
        "[{}] G2 all 8 reserved loci score 0 (RESERVE-DON'T-RECLAIM, not padded)",
        if reserved_empty { "PASS" } else { "FAIL" }
    );
    all_green &= reserved_empty;

    // Gate 3: the causality-trajectory minimum features (KAUSAL + the 3 SPO
    //         meaning loci) rank in the top half of the 16 named loci — the
    //         TEKAMOLO causality core must be high-relevance, not incidental.
    let named_sorted: Vec<usize> = {
        let mut v: Vec<(usize, u32)> = (0..16).map(|i| (i, row_sum[i])).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.into_iter().map(|(i, _)| i).collect()
    };
    let top_half: std::collections::HashSet<usize> = named_sorted.iter().take(8).copied().collect();
    // slots: 1=KAUSAL, 4=S, 5=P, 6=O
    let core = [1usize, 4, 5, 6];
    let core_in_top = core.iter().filter(|s| top_half.contains(s)).count();
    let core_ok = core_in_top >= 3;
    println!(
        "[{}] G3 causality core (KAUSAL+S/P/O) mostly top-half: {}/4 in top-8",
        if core_ok { "PASS" } else { "FAIL" },
        core_in_top
    );
    all_green &= core_ok;

    // Gate 4: the matrix is not degenerate — mean named-cell relevance in a sane
    //         band (neither all-zero nor saturated). Sanity, not a target.
    let named_cells: u32 = (0..16).map(|i| row_sum[i]).sum();
    let mean = named_cells as f64 / (16.0 * 34.0);
    let band_ok = (0.10..=1.50).contains(&mean);
    println!(
        "[{}] G4 mean named-cell relevance in band [0.10,1.50]: {:.3}",
        if band_ok { "PASS" } else { "FAIL" },
        mean
    );
    all_green &= band_ok;

    println!();
    println!(
        "{}",
        if all_green {
            "ALL GATES GREEN"
        } else {
            "GATE FAILURE"
        }
    );
    assert!(all_green, "loci×recipe relevance gates failed");
}
