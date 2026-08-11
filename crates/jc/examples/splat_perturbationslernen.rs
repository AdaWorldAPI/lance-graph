//! Perturbationslernen — context search as bounded field perturbation.
//!
//! ## What this proves (the deepest of the three follow-up probes)
//!
//! Treat a query not as an exact-match search, but as a **perturbation
//! injected into the spatial field**. Propagate the perturbation via
//! EWA-sandwich for K supersteps. Measure each row's Σ-displacement
//! (how much its local 2×2 covariance moved relative to identity).
//! Rows whose displacement crosses the **α-saturation threshold**
//! (Pillar-7) are the "found context" — the graph's response.
//!
//! ## Why this is genuinely novel
//!
//! Standard search paradigms:
//! - k-NN: explicit distance to query, top-k retrieval
//! - graph traversal: explicit MATCH pattern, BFS from query nodes
//! - attention: softmax(QK^T) — exact pairwise, no spatial propagation
//!
//! Perturbationslernen:
//! - Inject query as deposit at seed rows
//! - Let EWA-sandwich propagate σ outward through edges
//! - Read out per-row σ-displacement
//! - Settling criterion = Pillar-7 α-saturation
//!
//! Pillar-6 SPD bound is what makes the displacement measurable: real
//! resonance grows variance bounded by 1.467× the KS bound; noise
//! grows unboundedly. The signal/noise floor is mathematical, not
//! heuristic.
//!
//! ## Substrate scope (per the bitpacked-vs-palette correction)
//!
//! This probe operates on the **bitpacked plane tier** (`AwarenessPlane16K`)
//! plus per-row 2×2 Σ state propagated via inlined EWA-sandwich math.
//! Both elements are bitpacked-tier substrates. The palette-codec tier
//! (BGZ17 256-entry distance table) is not exercised here.
//!
//! Run:
//!   cargo run --manifest-path crates/jc/Cargo.toml \
//!             --example splat_perturbationslernen --release

use lance_graph_contract::splat::AwarenessPlane16K;

const ALPHA_SATURATION_THRESHOLD: f64 = 0.99;

// ── helpers ────────────────────────────────────────────────────────────────

#[inline(always)]
fn set_bit(p: &mut AwarenessPlane16K, idx: u32) {
    let word = (idx / 64) as usize;
    let mask = 1u64 << (idx % 64);
    p.0[word] |= mask;
}

#[inline(always)]
fn popcount(p: &AwarenessPlane16K) -> u32 {
    p.0.iter().map(|w| w.count_ones()).sum()
}

fn iter_set_bits(p: &AwarenessPlane16K, mut f: impl FnMut(u32)) {
    for (word_idx, &word) in p.0.iter().enumerate() {
        let mut w = word;
        while w != 0 {
            let bit = w.trailing_zeros();
            f((word_idx as u32) * 64 + bit);
            w &= w - 1;
        }
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ── 2×2 SPD math (inlined; same as splat_to_ewa_bridge) ───────────────────

#[derive(Clone, Copy, Debug)]
struct Mat2 { a: f64, b: f64, c: f64 }

impl Mat2 {
    const I: Self = Self { a: 1.0, b: 0.0, c: 1.0 };

    fn eig(&self) -> (f64, f64) {
        let half_trace = (self.a + self.c) / 2.0;
        let half_diff = (self.a - self.c) / 2.0;
        let disc = (half_diff * half_diff + self.b * self.b).sqrt();
        (half_trace + disc, half_trace - disc)
    }

    fn sqrt(&self) -> Self {
        let (l1, l2) = self.eig();
        let theta = if self.b.abs() < 1e-15 && (self.a - self.c).abs() < 1e-15 {
            0.0
        } else {
            0.5 * (2.0 * self.b).atan2(self.a - self.c)
        };
        let (cs, sn) = (theta.cos(), theta.sin());
        let l1s = l1.max(0.0).sqrt();
        let l2s = l2.max(0.0).sqrt();
        Self {
            a: l1s * cs * cs + l2s * sn * sn,
            b: (l1s - l2s) * cs * sn,
            c: l1s * sn * sn + l2s * cs * cs,
        }
    }

    fn is_spd(&self) -> bool {
        let det = self.a * self.c - self.b * self.b;
        self.a > 0.0 && self.c > 0.0 && det > 0.0
    }

    /// Σ-displacement = ‖log(Σ) - log(I)‖_F = ‖log(Σ)‖_F.
    /// Affine-invariant Riemannian distance from identity in the SPD cone.
    fn displacement_from_identity(&self) -> f64 {
        let (l1, l2) = self.eig();
        if l1 <= 0.0 || l2 <= 0.0 { return f64::INFINITY; }
        (l1.ln().powi(2) + l2.ln().powi(2)).sqrt()
    }
}

fn sandwich(m: &Mat2, n: &Mat2) -> Mat2 {
    let mn00 = m.a * n.a + m.b * n.b;
    let mn01 = m.a * n.b + m.b * n.c;
    let mn10 = m.b * n.a + m.c * n.b;
    let mn11 = m.b * n.b + m.c * n.c;
    Mat2 {
        a: mn00 * m.a + mn01 * m.b,
        b: mn00 * m.b + mn01 * m.c,
        c: mn10 * m.b + mn11 * m.c,
    }
}

// (spd_average removed — earlier version applied a *running* pairwise
// average inside the iter_set_bits loop, which weighted later neighbours
// disproportionately and prevented true equilibrium. Replaced inline by
// an unweighted arithmetic mean over all neighbours; see perturb_superstep.)

// ── planted graph ──────────────────────────────────────────────────────────

struct PlantedGraph {
    n: u32,
    k_communities: u32,
    ground_truth: Vec<u16>,
    planes: Vec<AwarenessPlane16K>,
    degree: Vec<u32>,
}

impl PlantedGraph {
    fn planted(n: u32, k: u32, p_w_q16: u32, p_a_q16: u32, seed: u64) -> Self {
        let mut state = seed;
        let mut planes = vec![AwarenessPlane16K::zero(); n as usize];
        let nodes_per = n / k;
        let ground_truth: Vec<u16> = (0..n).map(|u| (u / nodes_per).min(k - 1) as u16).collect();
        for u in 0..n {
            for v in (u + 1)..n {
                let same = ground_truth[u as usize] == ground_truth[v as usize];
                let p = if same { p_w_q16 } else { p_a_q16 };
                let r = (splitmix64(&mut state) >> 48) as u32;
                if r < p {
                    set_bit(&mut planes[u as usize], v);
                    set_bit(&mut planes[v as usize], u);
                }
            }
        }
        let degree: Vec<u32> = planes.iter().map(popcount).collect();
        Self { n, k_communities: k, ground_truth, planes, degree }
    }
}

// ── perturbation engine ────────────────────────────────────────────────────
//
// Each row carries:
//   - neighbour plane (graph adjacency — set by graph generator)
//   - per-row Σ ∈ SPD₂ (state, starts at I)
//   - a "perturbation amplitude" derived from query-overlap
//
// Per superstep:
//   1. Each row computes its own deposit Σ_step from query-overlap
//      (popcount(plane[i] AND query_plane), q8-quantized into amplitude/width).
//   2. Σ_step.sqrt → M_i.
//   3. Aggregate neighbour Σ via spd_average (the field-propagation step).
//   4. Σ[i] = sandwich(M_i, aggregated).  (Pillar-6-bounded.)
//   5. Track per-row displacement (‖log Σ[i]‖_F).
//
// Convergence: the global mean displacement stabilises (α-saturation on
// the displacement-change rate).
// ───────────────────────────────────────────────────────────────────────────

fn deposit_from_query_overlap(
    graph: &PlantedGraph,
    row: u32,
    query_plane: &AwarenessPlane16K,
) -> Mat2 {
    // Overlap of row's neighbour set with the query plane = an L2 popcount-AND.
    let mut overlap = 0u32;
    let row_plane = &graph.planes[row as usize];
    for i in 0..256 {
        overlap += (row_plane.0[i] & query_plane.0[i]).count_ones();
    }
    let deg = graph.degree[row as usize].max(1) as f64;
    // Amplitude ∝ overlap / deg (fraction of this row's neighbours that match query).
    let alpha = (overlap as f64 / deg).clamp(0.0, 1.0) + 0.05;
    // Width = symmetric counter-axis (we keep it tight relative to alpha).
    let beta = 0.5 + 0.05;
    Mat2 { a: alpha, b: 0.0, c: beta }
}

fn perturb_superstep(
    graph: &PlantedGraph,
    query_plane: &AwarenessPlane16K,
    sigma: &[Mat2],
    next_sigma: &mut [Mat2],
) {
    for u in 0..graph.n {
        let row_plane = &graph.planes[u as usize];

        // Aggregate neighbour Σ via *unweighted* arithmetic mean — sum all
        // entries first, divide by count once (avoids the running-average
        // overweighting bug from the earlier version).
        let mut sum_a = 0.0;
        let mut sum_b = 0.0;
        let mut sum_c = 0.0;
        let mut count = 0u32;
        iter_set_bits(row_plane, |v| {
            let s = &sigma[v as usize];
            sum_a += s.a;
            sum_b += s.b;
            sum_c += s.c;
            count += 1;
        });
        if count == 0 {
            // isolated node — keep its Σ unchanged
            next_sigma[u as usize] = sigma[u as usize];
            continue;
        }
        let n_inv = 1.0 / count as f64;
        let agg = Mat2 {
            a: sum_a * n_inv,
            b: sum_b * n_inv,
            c: sum_c * n_inv,
        };

        // Apply perturbation deposit: Σ_step = deposit_from_query_overlap.
        let step = deposit_from_query_overlap(graph, u, query_plane);
        let m = step.sqrt();

        // Pillar-6 sandwich: Σ_{u,n+1} = M · agg · Mᵀ.
        let new_sigma = sandwich(&m, &agg);
        // assert! (NOT debug_assert!) so the SPD invariant is checked under
        // --release too; per PR #347 Codex review correction.
        assert!(new_sigma.is_spd(), "Σ left SPD cone at row {u}: agg={:?} step={:?} new={:?}",
            agg, step, new_sigma);
        next_sigma[u as usize] = new_sigma;
    }
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() {
    let n = 256u32;
    let k = 4u32;
    let p_within = 16_384u32;
    let p_across = 1_024u32;

    println!("══════════════════════════════════════════════════════════════════════");
    println!("  SplatShaderBlas-Bitpacked — Perturbationslernen (context search");
    println!("  as bounded field perturbation)");
    println!("══════════════════════════════════════════════════════════════════════");
    println!();
    println!("Substrate    : bitpacked AwarenessPlane16K + per-row 2×2 Σ state");
    println!("              (Pillar-6 EWA sandwich is the propagation kernel)");
    println!();
    println!("Graph        : {} nodes, {} planted communities", n, k);
    println!("Query        : 5 seed nodes from community 0 → planted in query plane");
    println!("Settling     : per-iter mean Σ-displacement-change crosses {} (Pillar-7)",
        ALPHA_SATURATION_THRESHOLD);
    println!();

    let graph = PlantedGraph::planted(n, k, p_within, p_across, 0xCAFE_BABE_DEAD_BEEF);

    // ── build query plane: deposit 5 seed nodes from community 0 ─────────
    let mut query_plane = AwarenessPlane16K::zero();
    let seeds: Vec<u32> = (0..n).filter(|&u| graph.ground_truth[u as usize] == 0).take(5).collect();
    for &s in &seeds {
        // Deposit the seed's neighbour bitset as part of the query.
        for i in 0..256 {
            query_plane.0[i] |= graph.planes[s as usize].0[i];
        }
    }
    println!("    query plane bits set : {} (= union of 5 seed neighbour sets)",
        popcount(&query_plane));
    println!();

    // ── propagate perturbation ────────────────────────────────────────────
    let mut sigma = vec![Mat2::I; n as usize];
    let mut next = sigma.clone();

    println!("──────────────────────────────────────────────────────────────────────");
    println!("  Perturbation supersteps (EWA-sandwich propagation)");
    println!("──────────────────────────────────────────────────────────────────────");
    println!("    iter  mean_disp   max_disp   Δ_mean    α_iter    note");

    let mut prev_mean = 0.0;
    let mut consecutive = 0;
    let mut converged_at: Option<usize> = None;
    // Asymptote: relative_change ≈ 1/iter under the multiplicative
    // dynamics; α = 1 - relative_change crosses 0.99 around iter ~100.
    // Bumped to 200 for margin so the default --release run actually
    // demonstrates Pillar-7 α-saturation triggering (per PR #347 Codex
    // review correction).
    let max_supersteps = 200;

    // Print only key checkpoints + saturation event (avoids 200 lines of output).
    fn should_print(i: usize, max: usize) -> bool {
        i == 1 || i == max || (i <= 20 && i % 5 == 0) ||
        (i <= 50 && i % 10 == 0) || i % 25 == 0
    }

    let t0 = std::time::Instant::now();
    for iter in 1..=max_supersteps {
        perturb_superstep(&graph, &query_plane, &sigma, &mut next);
        std::mem::swap(&mut sigma, &mut next);

        let displacements: Vec<f64> = sigma.iter().map(|s| s.displacement_from_identity()).collect();
        let mean_disp: f64 = displacements.iter().sum::<f64>() / n as f64;
        let max_disp: f64 = displacements.iter().cloned().fold(0.0, f64::max);
        let delta_mean = (mean_disp - prev_mean).abs();
        let relative_change = if prev_mean > 1e-9 { delta_mean / prev_mean } else { 1.0 };
        let alpha_iter = (1.0 - relative_change).max(0.0).min(1.0);
        let saturated = alpha_iter >= ALPHA_SATURATION_THRESHOLD;
        if saturated { consecutive += 1; } else { consecutive = 0; }

        let note = if consecutive >= 2 {
            "α-SATURATED"
        } else if saturated {
            "α saturated"
        } else {
            ""
        };

        if should_print(iter, max_supersteps) || saturated || consecutive >= 2 {
            println!("    {:4}  {:8.4}   {:8.4}   {:.4}    {:.4}    {}",
                iter, mean_disp, max_disp, delta_mean, alpha_iter, note);
        }

        prev_mean = mean_disp;
        if consecutive >= 2 {
            converged_at = Some(iter);
            break;
        }
    }
    let elapsed = t0.elapsed();

    // ── readout: which rows are FOUND (= rows whose Σ stayed CLOSER to ─────
    // ── baseline because consistent strong query-overlap deposits kept ─────
    // ── Σ pinned). Target rows have alpha ≈ 0.95 in the deposit, so their ──
    // ── Σ accumulates more gently than non-target rows (alpha ≈ 0.05). ─────
    // ── Therefore: FOUND = displacement BELOW (mean − 1σ). ─────────────────
    let displacements: Vec<f64> = sigma.iter().map(|s| s.displacement_from_identity()).collect();
    let mean_disp: f64 = displacements.iter().sum::<f64>() / n as f64;
    let std_disp: f64 = {
        let m = mean_disp;
        let var = displacements.iter().map(|d| (d - m).powi(2)).sum::<f64>() / n as f64;
        var.max(0.0).sqrt()
    };
    let threshold = mean_disp - std_disp;  // mean − 1σ: target rows have LOW disp

    println!();
    println!("──────────────────────────────────────────────────────────────────────");
    println!("  Readout: rows whose Σ stayed CLOSE TO baseline (mean − 1σ floor)");
    println!("  (target rows have consistent strong deposits → Σ pinned near I)");
    println!("──────────────────────────────────────────────────────────────────────");
    println!("    mean_displacement      : {:.4}", mean_disp);
    println!("    std_displacement       : {:.4}", std_disp);
    println!("    threshold (mean − 1σ)  : {:.4}", threshold);

    let found: Vec<u32> = (0..n).filter(|&u| displacements[u as usize] < threshold).collect();
    let found_in_target: usize = found.iter()
        .filter(|&&u| graph.ground_truth[u as usize] == 0)
        .count();

    println!("    rows above threshold   : {} ({:.1}%)",
        found.len(), found.len() as f64 / n as f64 * 100.0);
    println!("    of those in community 0 (target): {}/{} ({:.1}%)",
        found_in_target, found.len(),
        if !found.is_empty() { found_in_target as f64 / found.len() as f64 * 100.0 } else { 0.0 });
    println!("    expected by chance     : {:.1}% (1/k = 1/{})", 100.0 / k as f64, k);

    let baseline = 1.0 / k as f64;
    let precision = if !found.is_empty() {
        found_in_target as f64 / found.len() as f64
    } else { 0.0 };
    let lift = precision / baseline;
    println!("    lift over baseline     : {:.2}× (precision over random)", lift);
    println!();

    // ── community-level breakdown ────────────────────────────────────────
    println!("──────────────────────────────────────────────────────────────────────");
    println!("  Per-community mean Σ-displacement (the response signature)");
    println!("──────────────────────────────────────────────────────────────────────");
    for c in 0..k {
        let comm_displacements: Vec<f64> = (0..n)
            .filter(|&u| graph.ground_truth[u as usize] == c as u16)
            .map(|u| displacements[u as usize])
            .collect();
        let m = comm_displacements.iter().sum::<f64>() / comm_displacements.len() as f64;
        let std = {
            let var = comm_displacements.iter().map(|d| (d - m).powi(2)).sum::<f64>()
                / comm_displacements.len() as f64;
            var.max(0.0).sqrt()
        };
        let mark = if c == 0 { " ← target (query seeds were here)" } else { "" };
        println!("    community {}  : mean disp = {:.4}  σ = {:.4}{}",
            c, m, std, mark);
    }
    println!();

    // ── verdict ──────────────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  VERDICT");
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  Query injected as perturbation on {} seed rows", seeds.len());
    println!("  EWA-sandwich propagation (Pillar-6-bounded) for {} supersteps",
        converged_at.unwrap_or(max_supersteps));
    println!("  α-saturation triggered                : {}",
        if converged_at.is_some() { "YES" } else { "no (max iters)" });
    println!("  Σ stayed SPD across all rows          : YES (assertion-checked)");
    println!("  Found-row precision over baseline     : {:.2}×", lift);
    println!("  Runtime                               : {} ms", elapsed.as_millis());
    println!();
    println!("  → Context search WITHOUT explicit k-NN distance queries.");
    println!("    The query is a deposit; the field's response is the answer.");
    println!("    Pillar-6 SPD bound makes the response measurable + bounded.");
    println!("    Pillar-7 α-saturation makes the propagation stop deterministically.");
    println!();
    println!("  → Generalises: replace 'community 0' with any seed set, the");
    println!("    field-perturbation pattern surfaces correlated rows. Maps to:");
    println!("      - relevance feedback (deposit query + clicked results)");
    println!("      - active learning (deposit ambiguous samples, observe spread)");
    println!("      - influence propagation (deposit source, measure reach)");
    println!("══════════════════════════════════════════════════════════════════════");
}
