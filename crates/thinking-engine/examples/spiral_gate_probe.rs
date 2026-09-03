//! spiral_gate_probe — the D-TEH-3 fate probe for `spiral_segment`.
//!
//! Plan `thinking-engine-harvest-closure-v1` §1c: `spiral_segment` (8 bytes
//! per row: anfang / ende / stride / gamma, "51× compression") is promoted to
//! a codec home only through the certification battery. The full battery
//! needs the F32 cosine matrix re-derived from the model source
//! (certification-officer, Rule 7) — which is not on disk here. This probe
//! is the GATE in front of that expensive step: it fits the codec to the
//! four REAL baked tables in the tree and measures whether it can clear the
//! encoding-ecosystem floor at all. A KILL here is final (a codec that cannot
//! preserve a baked 256×256 table will not preserve its F32 parent); a PASS
//! unblocks the battery, it is not a certification.
//!
//! Tables (all real, all committed): jina-v3 u8 CDF, bge-m3 u8 CDF,
//! reranker u8 CDF, jina-v5 u8 CDF and jina-v5 i8 direct (`round(cos·127)`).
//! Values are mapped to [0, 1] (u8/255) or [−1, 1] (i8/127) and handed to
//! the codec as BF16 exactly as `SpiralTable::encode` does; the reference for
//! every metric is the ORIGINAL table value, so BF16 rounding is charged to
//! the codec (it stores BF16). The BF16-rounded input is reported as a second
//! reference so the fit error can be told apart from the storage error.
//!
//! PRE-REGISTERED (encoding-ecosystem.md: "any encoding below the naive u8
//! floor is worse than doing nothing"; the bgz-hhtl-d gate is Pearson ≥
//! 0.9980): PASS if for SOME `max_error` on EVERY table
//!   Pearson r ≥ 0.9980  AND  Spearman ρ ≥ 0.9980  AND
//!   bytes(spiral) ≤ bytes(u8 table) / 2      (it must beat what it replaces)
//! KILL otherwise → stays LAB, the 51× claim recorded as measured-false.
//!
//! Usage:
//!   cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!     --example spiral_gate_probe

use std::collections::HashSet;

use bgz_tensor::stacked_n::{bf16_to_f32, f32_to_bf16};
use jc::reliability::{cronbach_alpha, pearson, spearman};
use thinking_engine::spiral_segment::SpiralRow;

const N: usize = 256;
const MAX_ERRORS: &[f32] = &[0.005, 0.01, 0.02, 0.05];
const R_FLOOR: f64 = 0.9980;
const RHO_FLOOR: f64 = 0.9980;
const MIN_RATIO_VS_U8: f64 = 2.0;

enum Kind {
    U8,
    I8,
}

const TABLES: &[(&str, &str, Kind)] = &[
    (
        "jina-v3 u8",
        "crates/thinking-engine/data/jina-v3-hdr/distance_table_256x256.u8",
        Kind::U8,
    ),
    (
        "bge-m3 u8",
        "crates/thinking-engine/data/bge-m3-hdr/distance_table_256x256.u8",
        Kind::U8,
    ),
    (
        "reranker u8",
        "crates/thinking-engine/data/jina-reranker-v3-BF16-hdr/distance_table_256x256.u8",
        Kind::U8,
    ),
    (
        "jina-v5 u8",
        "crates/thinking-engine/data/jina-v5-codebook/distance_table_256x256.u8",
        Kind::U8,
    ),
    (
        "jina-v5 i8",
        "crates/thinking-engine/data/jina-v5-codebook/distance_table_256x256.i8",
        Kind::I8,
    ),
];

struct Row {
    /// Pearson r / Spearman ρ / Cronbach α vs the ORIGINAL table values.
    r: f64,
    rho: f64,
    alpha: f64,
    /// Same three vs the BF16-rounded input (isolates the fit from storage).
    r_bf16: f64,
    rho_bf16: f64,
    /// Spiral bytes for the whole table, and the two baselines.
    bytes: usize,
    ratio_vs_u8: f64,
    ratio_vs_bf16: f64,
    avg_segments: f64,
    max_abs_err: f64,
}

fn load(path: &str, kind: &Kind) -> Vec<f32> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(raw.len(), N * N, "{path}: not a 256×256 table");
    match kind {
        Kind::U8 => raw.iter().map(|&b| f32::from(b) / 255.0).collect(),
        Kind::I8 => raw.iter().map(|&b| f32::from(b as i8) / 127.0).collect(),
    }
}

fn probe(table: &[f32], max_error: f32) -> Row {
    let mut reference = Vec::with_capacity(N * (N - 1));
    let mut ref_bf16 = Vec::with_capacity(N * (N - 1));
    let mut decoded = Vec::with_capacity(N * (N - 1));
    let mut bytes = 0usize;
    let mut segments = 0usize;
    let mut max_abs_err = 0.0f64;
    for i in 0..N {
        let row: Vec<f32> = (0..N)
            .filter(|&j| j != i)
            .map(|j| table[i * N + j])
            .collect();
        let enc = SpiralRow::encode(&row, table[i * N + i], max_error);
        let dec = enc.decode(N - 1);
        bytes += enc.byte_size();
        segments += enc.segments.len();
        for (k, &v) in row.iter().enumerate() {
            let d = f64::from(dec[k]);
            let orig = f64::from(v);
            reference.push(orig);
            ref_bf16.push(f64::from(bf16_to_f32(f32_to_bf16(v))));
            decoded.push(d);
            max_abs_err = max_abs_err.max((d - orig).abs());
        }
    }
    let nan = f64::NAN;
    Row {
        r: pearson(&reference, &decoded).unwrap_or(nan),
        rho: spearman(&reference, &decoded).unwrap_or(nan),
        alpha: cronbach_alpha(&[reference.clone(), decoded.clone()]).unwrap_or(nan),
        r_bf16: pearson(&ref_bf16, &decoded).unwrap_or(nan),
        rho_bf16: spearman(&ref_bf16, &decoded).unwrap_or(nan),
        bytes,
        ratio_vs_u8: (N * N) as f64 / bytes as f64,
        ratio_vs_bf16: (N * N * 2) as f64 / bytes as f64,
        avg_segments: segments as f64 / N as f64,
        max_abs_err,
    }
}

fn main() {
    let dedup_set: HashSet<&str> = TABLES.iter().map(|(n, _, _)| *n).collect();
    assert_eq!(dedup_set.len(), TABLES.len());
    println!(
        "gate: r ≥ {R_FLOOR}, ρ ≥ {RHO_FLOOR}, spiral bytes ≤ u8 bytes / {MIN_RATIO_VS_U8} — on EVERY table at ONE max_error\n"
    );
    println!(
        "{:>12} | {:>9} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>6} | {:>8} | {:>7}",
        "table",
        "max_error",
        "r",
        "rho",
        "alpha",
        "r_bf16",
        "rho_bf16",
        "x u8",
        "x bf16",
        "seg/row",
        "max|e|"
    );
    // pass_at[k] = every table clears the gate at MAX_ERRORS[k].
    let mut pass_at = vec![true; MAX_ERRORS.len()];
    for (name, path, kind) in TABLES {
        let table = load(path, kind);
        for (k, &me) in MAX_ERRORS.iter().enumerate() {
            let t0 = std::time::Instant::now();
            let row = probe(&table, me);
            let ok = row.r >= R_FLOOR && row.rho >= RHO_FLOOR && row.ratio_vs_u8 >= MIN_RATIO_VS_U8;
            pass_at[k] &= ok;
            println!(
                "{:>12} | {:>9.3} | {:>7.4} | {:>7.4} | {:>7.4} | {:>7.4} | {:>7.4} | {:>7.2} | {:>6.2} | {:>8.2} | {:>7.4}  {} ({:.1}s)",
                name,
                me,
                row.r,
                row.rho,
                row.alpha,
                row.r_bf16,
                row.rho_bf16,
                row.ratio_vs_u8,
                row.ratio_vs_bf16,
                row.avg_segments,
                row.max_abs_err,
                if ok { "ok" } else { "--" },
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let passing: Vec<f32> = MAX_ERRORS
        .iter()
        .zip(&pass_at)
        .filter(|(_, p)| **p)
        .map(|(m, _)| *m)
        .collect();
    println!(
        "\nVERDICT: {}",
        if passing.is_empty() {
            "KILL (stays LAB) — no max_error clears the gate on every table".to_string()
        } else {
            format!("PASS (battery-eligible) at max_error {passing:?}")
        }
    );
}
