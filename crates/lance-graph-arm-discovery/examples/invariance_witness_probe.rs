// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Invariance-witness probe — the THIRD meta axis the MIT causal-discovery arc hands
//! us (Peters-Bühlmann Invariant Causal Prediction; Uhler's interventional / multi-
//! domain causal representation learning): a CAUSAL mechanism `P(Y|X)` is STABLE across
//! environments; a CONFOUNDED association `X←Z→Y` SHIFTS when the environment changes
//! the hidden Z.
//!
//! Episodic basins ARE the environments here — used as SUPPORT structure, never as a
//! cause. The load-bearing caution: basins formed by the OUTCOME induce collider bias
//! (you "discover" spurious parents of Y). So this probe partitions on a CONTEXT feature
//! (Region), explicitly NOT on the target.
//!
//! Why a third axis is needed: entropy (Staunen↔Wisdom) and Granger (lagged drive) both
//! MISS a confounded pair. It reads HIGH confidence (LOW entropy → "Wisdom") because
//! `P(Y|X)` really is high marginally. Only invariance refuses it: condition on the
//! basin and the "reliable" mechanism falls apart (high cross-basin variance). This is
//! the MIT line's recurring thesis — predictive success is not causal validity.
//!
//! Claims, each a measured number:
//!
//! ```text
//! INV1  invariance separates direct-causal from confounded   variance gap + Spearman
//! INV2  invariance is NOT redundant with entropy             confounded = low H, high variance
//! INV3  invariance is a usable witness/orientation signal    direct rules pass, confounded refuted
//! ```
//!
//! cargo run --release --example invariance_witness_probe \
//!     --manifest-path crates/lance-graph-arm-discovery/Cargo.toml --features ndarray-simd

use lance_graph_arm_discovery::translator::DebugProjector;
use lance_graph_arm_discovery::{
    extract_rules, CandidateRule, CandidateTriple, Dataset, ExtractParams, FeatureSpec, Item,
    MatrixDistance, NARS_PERSONALITY_K,
};
use ndarray::hpc::entropy_ladder::nars_entropy;
use ndarray::hpc::reliability::spearman;

// Schema. f0 Region is the BASIN/context key (NOT a target). X1→Y1 is a DIRECT edge
// (stable mechanism). X2 and Y2 are both children of a hidden confounder Z whose rate
// p_z varies by Region — so X2⇒Y2 looks strong marginally but its mechanism SHIFTS.
// f0 Region:4  f1 X1:2  f2 Y1:2  f3 X2:2  f4 Y2:2
const CARD: [u32; 5] = [4, 2, 2, 2, 2];
const REGION: u32 = 0;
const X1: u32 = 1;
const Y1: u32 = 2;
const X2: u32 = 3;
const Y2: u32 = 4;
const N_BASINS: u32 = 4;
const P_Z_BY_REGION: [f64; 4] = [0.15, 0.45, 0.75, 0.95]; // hidden confounder rate per basin
const P_Y1_GIVEN_X1: f64 = 0.90; // the DIRECT, invariant mechanism
const P_FROM_Z: f64 = 0.90; // X2,Y2 each track Z with this fidelity (no X2→Y2 edge)

fn splitmix(s: &mut u64) -> f64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn bern(s: &mut u64, p: f64) -> u32 {
    u32::from(splitmix(s) < p)
}

/// Generate rows. Region is observed (the basin key); Z is hidden (never emitted).
fn build_corpus(seed: u64, n: usize) -> Vec<Vec<u32>> {
    let mut s = seed;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let region = (splitmix(&mut s) * f64::from(N_BASINS)) as u32 % N_BASINS;
        // Direct, basin-invariant edge X1 → Y1.
        let x1 = bern(&mut s, 0.5);
        let y1 = if bern(&mut s, P_Y1_GIVEN_X1) == 1 {
            x1
        } else {
            1 - x1
        };
        // Hidden confounder Z, rate set by the basin. X2 and Y2 are BOTH its children.
        let z = bern(&mut s, P_Z_BY_REGION[region as usize]);
        let x2 = if bern(&mut s, P_FROM_Z) == 1 {
            z
        } else {
            bern(&mut s, 0.5)
        };
        let y2 = if bern(&mut s, P_FROM_Z) == 1 {
            z
        } else {
            bern(&mut s, 0.5)
        };
        rows.push(vec![region, x1, y1, x2, y2]);
    }
    rows
}

/// Codebook: propose Y1 given X1 and Y2 given X2 (the two pairs we want tested).
fn build_oracle(spec: &FeatureSpec) -> MatrixDistance {
    let dim = spec.dim();
    let slot = |f: u32, c: u32| spec.slot(Item::new(f, c));
    let mut table = vec![25u32; dim * dim];
    let near = |t: &mut [u32], a: usize, b: usize, v: u32| {
        t[a * dim + b] = v;
        t[b * dim + a] = v;
    };
    for c in 0..2 {
        near(&mut table, slot(X1, c), slot(Y1, c), 2);
        near(&mut table, slot(X2, c), slot(Y2, c), 2);
    }
    MatrixDistance::new(spec, table)
}

/// Confidence `P(consequent | antecedent)` of a rule WITHIN one basin (region) — the
/// per-environment mechanism ICP tests for invariance. `None` if the antecedent is too
/// rare in that basin to estimate.
fn basin_confidence(rule: &CandidateRule, rows: &[Vec<u32>], region: u32) -> Option<f64> {
    let holds = |row: &[u32], items: &[Item]| {
        items
            .iter()
            .all(|it| row[it.feature as usize] == it.category)
    };
    let (mut ante, mut both) = (0u64, 0u64);
    for row in rows.iter().filter(|r| r[REGION as usize] == region) {
        if holds(row, &rule.antecedent) {
            ante += 1;
            both += u64::from(holds(row, &rule.consequent));
        }
    }
    (ante >= 20).then(|| both as f64 / ante as f64)
}

/// Cross-basin standard deviation of the mechanism — the invariance signal.
/// Low = stable mechanism (causal); high = shifts with context (confounded).
fn cross_basin_std(rule: &CandidateRule, rows: &[Vec<u32>]) -> Option<f64> {
    let confs: Vec<f64> = (0..N_BASINS)
        .filter_map(|r| basin_confidence(rule, rows, r))
        .collect();
    if confs.len() < 2 {
        return None;
    }
    let mean = confs.iter().sum::<f64>() / confs.len() as f64;
    let var = confs.iter().map(|c| (c - mean) * (c - mean)).sum::<f64>() / confs.len() as f64;
    Some(var.sqrt())
}

/// Ground-truth label from the planting: DIRECT (X1⇒Y1, invariant), CONFOUNDED
/// (X2⇒Y2, no edge — purely via hidden Z), or filler.
fn label(rule: &CandidateRule) -> &'static str {
    let a = &rule.antecedent[0];
    let o = &rule.consequent[0];
    if a.feature == X1 && o.feature == Y1 && a.category == o.category {
        "DIRECT (X1->Y1)"
    } else if a.feature == X2 && o.feature == Y2 && a.category == o.category {
        "CONFOUNDED (X2~Z~Y2)"
    } else {
        "filler"
    }
}

fn main() {
    println!("== Invariance-witness probe: basins as environments (ICP), the third meta axis ==\n");

    let spec = FeatureSpec::new(CARD.to_vec());
    // NOTE: run on the default codegen path. `-C target-cpu=native` currently SIGILLs in
    // ndarray's U64x8 popcount over RowMasks this large (TD-NDARRAY-SIMD-POPCNT-NATIVE);
    // the result is identical on the scalar path, which is what this probe uses.
    let rows = build_corpus(0x1A5E_5EED, 24_000);
    let data = Dataset::new(spec.clone(), rows.clone());
    let oracle = build_oracle(&spec);
    let params = ExtractParams {
        theta: u32::MAX,
        max_antecedent: 1,
        min_support_ppm: 10_000,
        min_confidence_ppm: 550_000,
    };
    let rules = extract_rules(&oracle, &data, &params);
    println!(
        "Aerial+ extracted {} candidate rules (1% support / 55% confidence).\n",
        rules.len()
    );

    // Restrict the measured population to the two PLANTED families (direct vs confounded);
    // fillers carry no ground-truth confoundedness label.
    let mut stds = Vec::new();
    let mut confoundedness = Vec::new();
    let mut shown: Vec<(String, f64, f64, f64)> = Vec::new(); // label, global f, H, cross-basin σ

    for rule in &rules {
        let lbl = label(rule);
        if lbl == "filler" {
            continue;
        }
        let Some(sigma) = cross_basin_std(rule, &rows) else {
            continue;
        };
        let triple: CandidateTriple =
            CandidateTriple::from_rule(rule, &DebugProjector::default(), NARS_PERSONALITY_K);
        let h = nars_entropy(f64::from(triple.f), f64::from(triple.c));
        let conf = if lbl.starts_with("DIRECT") { 0.0 } else { 1.0 };
        stds.push(sigma);
        confoundedness.push(conf);
        shown.push((lbl.to_string(), f64::from(triple.f), h, sigma));
    }

    println!("INV3  Per-rule mechanism (global) vs invariance across {N_BASINS} context-basins:");
    println!(
        "    {:<22} {:>7} {:>7}  {:>14}  {:>9}",
        "ground truth", "f", "H", "cross-basin σ", "verdict"
    );
    shown.sort_by(|a, b| a.0.cmp(&b.0).then(a.3.partial_cmp(&b.3).unwrap()));
    let mut printed = std::collections::HashMap::new();
    for (lbl, f, h, sigma) in &shown {
        let seen = printed.entry(lbl.clone()).or_insert(0);
        if *seen >= 2 {
            continue;
        }
        *seen += 1;
        let verdict = if *sigma < 0.05 { "INVARIANT" } else { "shifts" };
        println!("    {lbl:<22} {f:>7.3} {h:>7.2}  {sigma:>14.4}  {verdict:>9}");
    }

    // INV1: invariance separates the two populations.
    let inv1 = spearman(&stds, &confoundedness);
    let mean_for = |want: f64| {
        let v: Vec<f64> = stds
            .iter()
            .zip(&confoundedness)
            .filter(|(_, &c)| (c - want).abs() < 1e-9)
            .map(|(&s, _)| s)
            .collect();
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let (mean_direct, mean_conf) = (mean_for(0.0), mean_for(1.0));

    println!("\nINV1  Spearman(cross-basin σ, true confoundedness) = {inv1:+.3}   (→ +1: σ ranks confounding)");
    println!(
        "      mean cross-basin σ:  DIRECT {mean_direct:.4}   vs   CONFOUNDED {mean_conf:.4}   ({:.1}× wider)",
        mean_conf / mean_direct.max(1e-9)
    );

    // INV2: the confounded rule is LOW entropy (looks reliable) yet NON-invariant.
    let direct = shown.iter().find(|r| r.0.starts_with("DIRECT"));
    let confd = shown.iter().find(|r| r.0.starts_with("CONFOUNDED"));
    if let (Some(d), Some(c)) = (direct, confd) {
        println!("\nINV2  entropy can't see the confounder — both read confident (low H):");
        println!(
            "        DIRECT      H={:.2}  σ={:.4}  → causal, and invariance agrees",
            d.2, d.3
        );
        println!(
            "        CONFOUNDED  H={:.2}  σ={:.4}  → entropy says 'Wisdom', invariance says NO",
            c.2, c.3
        );
        println!(
            "      ΔH = {:.2} (entropy nearly equal) but Δσ = {:.4} (invariance separates) — a THIRD axis.",
            (d.2 - c.2).abs(),
            (c.3 - d.3).abs()
        );
    }

    println!("\nVERDICT:");
    println!("  • Invariance across context-basins is an orientation/refutation witness that entropy and");
    println!("    Granger cannot supply: the confounded X2~Z~Y2 pair reads LOW entropy (high marginal P(Y|X))");
    println!("    yet its mechanism SHIFTS across basins (σ {mean_conf:.4} vs {mean_direct:.4} for the direct edge) — ICP refuses it.");
    println!("  • Basins are SUPPORT, not cause: partitioned on Region (context), never on the target Y —");
    println!("    forming basins by the outcome would be collider bias and would manufacture spurious parents.");
    println!("  • Witness-arc upshot: store an INVARIANCE witness (σ below a floor ⇒ supports the edge; σ above");
    println!("    ⇒ REFUTES it). With precedence (Granger) + reliability (entropy) + invariance, the witness");
    println!("    pointer resolves THREE independent meta coordinates — none stored, all derived.");
}
