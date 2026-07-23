//! PROBE-CODEBOOK-44 — hierarchical 16-way codebook vs flat k-means.
//!
//! Retires the long-unrun Probe M1 ("CLAM 3-level 16-way tree on 256 centroids").
//! Two mechanical gates on a PLANTED 16×16 hierarchy (the correct hierarchy is
//! known by construction), so the mechanism is fully falsifiable synthetically:
//!
//!   GATE 1 — prefix == ancestry: the hierarchical codebook's byte code carries
//!            its coarse cluster in the top nibble (`code >> 4 == coarse`,
//!            mirroring `lance-graph-contract::hhtl::NiblePath`), recovering the
//!            planted coarse structure; the flat codebook does NOT.
//!   GATE 2 — fidelity ρ: Spearman ρ of pairwise inter-centroid distances,
//!            codebook reconstruction vs ground-truth, hierarchical vs flat.
//!
//! Deterministic SplitMix64 (seed 0x9E3779B97F4A7C15) — no clock, no rand.
//!
//! Run:  cargo run --manifest-path crates/bgz17/Cargo.toml --example probe_codebook_44

use bgz17::base17::Base17;
use bgz17::palette::{HierarchicalPalette, Palette};
use bgz17::BASE_DIM;

// ── deterministic RNG ────────────────────────────────────────────────────────

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn jitter(&mut self, range: i32) -> i16 {
        let span = (range as i64 * 2 + 1) as u64;
        ((self.next_u64() % span) as i64 - range as i64) as i16
    }
}

// ── planted 16×16 hierarchy ──────────────────────────────────────────────────

/// (samples, ground-truth leaf templates, planted coarse index per template).
/// Separation scales: coarse (~±4000) ≫ fine (~±400) ≫ sample jitter (~±40).
fn planted_hierarchy(per_leaf: usize) -> (Vec<Base17>, Vec<Base17>, Vec<usize>) {
    let mut rng = SplitMix64(0x9E37_79B9_7F4A_7C15);
    let mut samples = Vec::new();
    let mut templates = Vec::new();
    let mut template_coarse = Vec::new();
    for c in 0..16usize {
        let mut a = [0i16; BASE_DIM];
        for slot in a.iter_mut() {
            *slot = rng.jitter(4000);
        }
        for _f in 0..16usize {
            let mut leaf = a;
            for slot in leaf.iter_mut() {
                *slot = slot.saturating_add(rng.jitter(400));
            }
            templates.push(Base17 { dims: leaf });
            template_coarse.push(c);
            for _s in 0..per_leaf {
                let mut samp = leaf;
                for slot in samp.iter_mut() {
                    *slot = slot.saturating_add(rng.jitter(40));
                }
                samples.push(Base17 { dims: samp });
            }
        }
    }
    (samples, templates, template_coarse)
}

// ── GATE 1 helpers ───────────────────────────────────────────────────────────

fn planted_coarse_of(centroid: &Base17, templates: &[Base17], template_coarse: &[usize]) -> usize {
    let mut best = 0usize;
    let mut best_d = u32::MAX;
    for (i, t) in templates.iter().enumerate() {
        let d = centroid.l1(t);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    template_coarse[best]
}

fn top_nibble_purity(entries: &[Base17], templates: &[Base17], template_coarse: &[usize]) -> f64 {
    let mut group_labels: Vec<Vec<usize>> = vec![Vec::new(); 16];
    for (code, e) in entries.iter().enumerate() {
        let pc = planted_coarse_of(e, templates, template_coarse);
        group_labels[code >> 4].push(pc);
    }
    let (mut correct, mut total) = (0usize, 0usize);
    for labels in &group_labels {
        if labels.is_empty() {
            continue;
        }
        let mut counts = [0usize; 16];
        for &l in labels {
            counts[l] += 1;
        }
        correct += counts.iter().copied().max().unwrap_or(0);
        total += labels.len();
    }
    if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    }
}

// ── GATE 2: Spearman ρ (average-rank, tie-aware) ─────────────────────────────

fn average_ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        // average rank (1-based) for the tie block [i, j)
        let avg = ((i + 1 + j) as f64) / 2.0;
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        0.0
    } else {
        cov / (vx.sqrt() * vy.sqrt())
    }
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&average_ranks(x), &average_ranks(y))
}

/// Spearman ρ of pairwise inter-centroid distances: codebook reconstruction
/// (each template quantized to its nearest palette centroid) vs ground truth.
fn fidelity_rho(palette: &Palette, templates: &[Base17]) -> f64 {
    let codes: Vec<u8> = templates.iter().map(|t| palette.nearest(t)).collect();
    let n = templates.len();
    let mut gt = Vec::with_capacity(n * (n - 1) / 2);
    let mut recon = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            gt.push(templates[i].l1(&templates[j]) as f64);
            let ci = &palette.entries[codes[i] as usize];
            let cj = &palette.entries[codes[j] as usize];
            recon.push(ci.l1(cj) as f64);
        }
    }
    spearman(&gt, &recon)
}

fn main() {
    let (samples, templates, template_coarse) = planted_hierarchy(8);
    println!("PROBE-CODEBOOK-44  (retires Probe M1)");
    println!("======================================");
    println!(
        "planted synthetic: 16 coarse × 16 fine = {} templates, {} samples",
        templates.len(),
        samples.len()
    );
    println!("data label: SYNTHETIC (planted hierarchy) — real-data ρ (Jina centroids) PENDING\n");

    let hp = Palette::build_hierarchical(&samples, 20);
    let flat = Palette::build(&samples, 256, 20);

    // ── GATE 1 ────────────────────────────────────────────────────────────────
    let mut ancestry_ok = true;
    for code in 0u16..256 {
        let code = code as u8;
        if !HierarchicalPalette::coarse_is_ancestor_of(
            HierarchicalPalette::coarse_index(code),
            code,
        ) {
            ancestry_ok = false;
        }
    }
    let hier_purity = top_nibble_purity(&hp.leaves.entries, &templates, &template_coarse);
    let flat_purity = top_nibble_purity(&flat.entries, &templates, &template_coarse);

    // coarse == mean of children?
    let mut parent_ok = true;
    for c in 0..hp.coarse_len() {
        let children: Vec<&Base17> = hp.leaves.entries[c * 16..c * 16 + 16].iter().collect();
        // recompute mean the same way build does
        let mut sum = [0i64; BASE_DIM];
        for ch in &children {
            for (d, s) in sum.iter_mut().enumerate() {
                *s += ch.dims[d] as i64;
            }
        }
        let mut dims = [0i16; BASE_DIM];
        for d in 0..BASE_DIM {
            dims[d] = (sum[d] / children.len() as i64) as i16;
        }
        if hp.coarse[c] != (Base17 { dims }) {
            parent_ok = false;
        }
    }

    println!("GATE 1 — prefix == ancestry (structural)");
    println!("  leaves={}  coarse={}", hp.leaf_len(), hp.coarse_len());
    println!("  code>>4 == coarse ancestor for all 256 codes : {ancestry_ok}");
    println!("  coarse centroid == mean of its 16 children    : {parent_ok}");
    println!("  hierarchical top-nibble planted purity        : {hier_purity:.4}");
    println!("  flat        top-nibble planted purity         : {flat_purity:.4}");
    let gate1 = ancestry_ok && parent_ok && hier_purity >= 0.95 && flat_purity <= 0.5;
    println!(
        "  => hierarchy REAL (not decorative), flat FAILS ancestry : {}",
        if gate1 { "PASS" } else { "FAIL" }
    );

    // ── GATE 2 ────────────────────────────────────────────────────────────────
    let rho_hier = fidelity_rho(&hp.leaves, &templates);
    let rho_flat = fidelity_rho(&flat, &templates);
    println!("\nGATE 2 — fidelity ρ (Spearman, pairwise inter-centroid)  [SYNTHETIC]");
    println!("  hierarchical ρ : {rho_hier:.4}");
    println!("  flat k-means ρ : {rho_flat:.4}");
    println!("  anchors        : 0.9973 / 0.965 (canon; real-data anchors)");
    println!(
        "  clears 0.965   : hier={} flat={}",
        rho_hier >= 0.965,
        rho_flat >= 0.965
    );
    println!(
        "  clears 0.9973  : hier={} flat={}",
        rho_hier >= 0.9973,
        rho_flat >= 0.9973
    );
    let gate2 = rho_hier >= rho_flat - 1e-9; // hierarchy must not lose fidelity
    println!(
        "  => hierarchical ρ >= flat ρ (structure is free) : {}",
        if gate2 { "PASS" } else { "FAIL" }
    );

    // ── verdict ────────────────────────────────────────────────────────────────
    println!("\nVERDICT");
    if gate1 && gate2 {
        println!("  PROBE-CODEBOOK-44: MECHANISM PROVEN (synthetic).");
        println!("  Hierarchy is real (prefix==ancestry) and fidelity >= flat.");
        println!("  Probe M1: mechanism retired; REAL-DATA ρ on Jina centroids PENDING");
        println!("  (no safetensors/centroids on disk this run).");
    } else {
        println!("  PROBE-CODEBOOK-44: FAIL — see gate output above.");
    }
}
