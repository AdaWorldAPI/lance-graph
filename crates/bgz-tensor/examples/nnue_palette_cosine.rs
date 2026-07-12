//! D-PALETTE-NNUE (corrected) — does the CERTIFIED palette256 cosine-replacement
//! preserve the pairwise-cosine ranking of NNUE feature-transformer columns?
//!
//! Correction of the first cut (`stockfish-rs examples/palette_nnue.rs`), which
//! used a hand-rolled scalar k-means codebook and re-ran the whole eval
//! (materialized) — wrong tool, wrong methodology. This probe reuses the real
//! codec, `bgz_tensor::fisher_z::FisherZTable` (Fisher-z i8, per-family 3σ gamma;
//! certified ρ≥0.999 on 21 Qwen3-TTS roles / 256 Jina-v5 centroids), and measures
//! RANKING directly off the i8 table — no vector reconstruction. Fisher-z
//! (`arctanh`) is monotone, so it preserves cosine rank BY CONSTRUCTION; the only
//! rank-affecting step is the i8 3σ quantization, which is exactly what ρ here
//! measures.
//!
//! Input: the 256 FT columns (1024-dim f32) exported by
//! `stockfish-rs --example export_ft_columns` (32 640 off-diagonal pairs, the
//! Jina-v5 certification setup). Path from arg1 / `$FT_COLUMNS_OUT` / default tmp.
//!
//! Gate (anchor: certified Fisher-z lane ρ≥0.999): ρ_all ≥ 0.999 AND the hard
//! near-orthogonal cut ρ_mid ≥ 0.99 → GREEN (FT columns ARE a palette256 tenant:
//! one-table-read cosine similarity preserved). The **exit code IS the gate**:
//! 0 only on GREEN; non-zero on fenced OR insufficient (a missing hard cut is
//! withheld, never an auto-pass). The input-absent path exits 0 (CI-safe).
//!
//! Run: cargo run --manifest-path crates/bgz-tensor/Cargo.toml --release \
//!        --example nnue_palette_cosine -- /path/to/ft_columns.bin

use std::path::PathBuf;
use std::process::ExitCode;

use bgz_tensor::fisher_z::FisherZTable;

fn in_path() -> PathBuf {
    std::env::args()
        .nth(1)
        .or_else(|| std::env::var("FT_COLUMNS_OUT").ok())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/ft_columns.bin"))
}

fn read_columns(path: &PathBuf) -> Option<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 8 {
        return None;
    }
    let k = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + k * dim * 4 {
        return None;
    }
    let mut cols = Vec::with_capacity(k);
    let mut off = 8;
    for _ in 0..k {
        let mut row = Vec::with_capacity(dim);
        for _ in 0..dim {
            row.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        cols.push(row);
    }
    Some(cols)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len().min(b.len()) {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let d = (na * nb).sqrt();
    if d < 1e-15 {
        0.0
    } else {
        (dot / d) as f32
    }
}

fn ranks(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap());
    let mut r = vec![0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (xs[idx[j]] - xs[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        let avg = ((i + j - 1) as f64) / 2.0 + 1.0;
        for &k in &idx[i..j] {
            r[k] = avg;
        }
        i = j;
    }
    r
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (x, y) = (a[i] - ma, b[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da == 0.0 || db == 0.0 {
        0.0
    } else {
        num / (da.sqrt() * db.sqrt())
    }
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}

fn main() -> ExitCode {
    let path = in_path();
    let Some(cols) = read_columns(&path) else {
        println!(
            "nnue_palette_cosine: no FT-columns file at {} (run stockfish-rs export_ft_columns first) — skipping.",
            path.display()
        );
        return ExitCode::SUCCESS;
    };
    let k = cols.len();
    println!(
        "nnue_palette_cosine: {} FT columns (dim {}) loaded\n",
        k,
        cols[0].len()
    );

    // The REAL certified codec: fits per-family 3σ gamma from the actual pairwise
    // cosines and encodes the k×k table to i8 (Fisher-z). No reconstruction.
    let table = FisherZTable::build(&cols, k);

    // True vs palette-restored pairwise cosine, off-diagonal pairs.
    let mut t_all = Vec::new();
    let mut pal = Vec::new();
    let (mut mid_t, mut mid_p) = (Vec::new(), Vec::new());
    let mut mae = 0f64;
    let mut mx = 0f64;
    for i in 0..k {
        for j in (i + 1)..k {
            let t = cosine(&cols[i], &cols[j]) as f64;
            let p = table.lookup_f32(i as u8, j as u8) as f64;
            mae += (t - p).abs();
            mx = mx.max((t - p).abs());
            t_all.push(t);
            pal.push(p);
            // hard cut: near-orthogonal band |cos| ≤ 0.3, where Fisher-z stretch
            // is smallest and discrimination is hardest.
            if t.abs() <= 0.3 {
                mid_t.push(t);
                mid_p.push(p);
            }
        }
    }
    let npairs = t_all.len();
    mae /= npairs as f64;
    let rho_all = spearman(&t_all, &pal);
    let r_all = pearson(&t_all, &pal);
    let rho_mid = if mid_t.len() >= 8 {
        spearman(&mid_t, &mid_p)
    } else {
        f64::NAN
    };
    println!("codec: FisherZTable (Fisher-z i8, per-family 3σ gamma) — the certified palette256 cosine-replacement");
    println!(
        "       gamma z_min {:.4} z_range {:.4} | table {}×{} i8 = {} KB (one-table-read)",
        table.gamma.z_min,
        table.gamma.z_range,
        k,
        k,
        (k * k) / 1024
    );
    println!(
        "pairs: {} off-diagonal | cosine MAE {:.5}, max |Δ| {:.4} | Pearson r {:.5} | Spearman ρ {:.5}",
        npairs, mae, mx, r_all, rho_all
    );
    println!(
        "hard cut (|cos| ≤ 0.3, near-orthogonal): {} pairs | Spearman ρ {:.5}   ← the fine-discrimination bar\n",
        mid_t.len(),
        rho_mid
    );

    // The gate is explicit: ρ_all ≥ 0.999 AND the hard near-orthogonal cut
    // ρ_mid ≥ 0.99. A missing hard cut (too few near-orthogonal pairs → ρ_mid
    // undefined) is NOT a pass — GREEN is withheld. Exit code IS the gate: 0 only
    // on GREEN; non-zero on fenced or insufficient, so automation can distinguish
    // a certified tenant from one where the fine-discrimination bar failed or was
    // never measured. (The input-absent path above still exits 0 — CI-safe.)
    let hard_cut_ok = rho_mid.is_finite() && rho_mid >= 0.99;
    if rho_all >= 0.999 && hard_cut_ok {
        println!(
            "D-PALETTE-NNUE (corrected): GREEN — ρ_all {:.5} ≥ 0.999 and near-orthogonal ρ {:.5} ≥ 0.99.\n\
             The NNUE FT columns ARE a palette256 tenant: the certified Fisher-z cosine-replacement\n\
             preserves pairwise-cosine ranking (one-table-read similarity), no materialization.\n\
             The earlier scalar-k-means FENCE was an artifact of the wrong codec + eval-reconstruction.",
            rho_all, rho_mid
        );
        ExitCode::SUCCESS
    } else if !rho_mid.is_finite() {
        println!(
            "D-PALETTE-NNUE (corrected): INSUFFICIENT — the near-orthogonal hard cut had < 8 pairs\n\
             (ρ_mid undefined), so the gate (ρ_all ≥ 0.999 AND ρ_mid ≥ 0.99) was never fully exercised.\n\
             GREEN is withheld — supply an FT-column fixture with near-orthogonal pairs. (ρ_all = {:.5}.)",
            rho_all
        );
        ExitCode::FAILURE
    } else {
        let primary = rho_all.min(rho_mid);
        println!(
            "D-PALETTE-NNUE (corrected): AMBER/FENCED — min(ρ_all, ρ_mid) {:.5} below the gate. The\n\
             certified cosine-replacement does not clear the ρ_all≥0.999 / near-orthogonal-0.99 anchor\n\
             on these FT columns; the tenant holds only coarsely. A real measured result — the gate\n\
             returns non-zero so automation can distinguish it from GREEN.",
            primary
        );
        ExitCode::FAILURE
    }
}
