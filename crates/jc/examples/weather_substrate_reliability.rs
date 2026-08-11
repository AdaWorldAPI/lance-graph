//! Gate 1 of the weather-substrate verification battery (§9 of
//! `.claude/knowledge/weather-normalized-substrate.md`): is the palette256
//! representation **valid and reliable** by the workspace's own instruments?
//!
//! Runs `jc::reliability` — Pearson r, Spearman ρ, Cronbach α, ICC(2,1) and
//! ICC(3,1) — over `(truth, reconstructed)` pairs produced by the P1/P2 probes
//! on real ARCO-ERA5 fields. Deliberately Rust + `jc` rather than scipy: §9
//! rules that `jc` returns `Option<f64>` where the `ndarray::hpc` mirror
//! returns `0.0` sentinels, so a `0.0` from the mirror is ambiguous
//! (undefined-vs-zero) and `jc` is what a verdict cites.
//!
//! Input: `probes/weather-p1/jc_input.bin`, written by
//! `probes/weather-p1/export_for_jc.py`. Layout (all little-endian):
//! `i64 n_vars, i64 n_points`, then per variable `n_points × f64` truth
//! followed by `n_points × f64` reconstructed.
//!
//! ```text
//! cargo run -p jc --example weather_substrate_reliability -- probes/weather-p1/jc_input.bin
//! ```
//!
//! **What would make this FAIL** (the falsifiability rule — an assertion
//! implied by its own input is not a test): a quantizer that lost rank order
//! drops Spearman; one that lost scale drops ICC(2,1) (absolute agreement)
//! while leaving ICC(3,1) (consistency) high — the two ICC forms are reported
//! separately precisely so a scale shift cannot hide behind a consistency
//! number. The `--shuffle` control below is the can-it-fire half: it feeds
//! deliberately mismatched pairs and must collapse every statistic.

use jc::reliability::{cronbach_alpha, icc, pearson, spearman, IccForm};
use std::io::Read;

const VAR_NAMES: [&str; 3] = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
];

fn read_f64s(buf: &[u8], off: &mut usize, n: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let mut b = [0u8; 8];
        b.copy_from_slice(&buf[*off..*off + 8]);
        v.push(f64::from_le_bytes(b));
        *off += 8;
    }
    v
}

fn show(label: &str, truth: &[f64], recon: &[f64]) {
    let fmt = |o: Option<f64>| o.map_or_else(|| "  UNDEFINED".to_string(), |v| format!("{v:10.6}"));
    // Cronbach / ICC treat (truth, reconstructed) as 2 "raters" of each point.
    let paired: Vec<Vec<f64>> = truth.iter().zip(recon).map(|(&t, &r)| vec![t, r]).collect();
    // cronbach_alpha wants items (raters) as the outer dimension.
    let items = vec![truth.to_vec(), recon.to_vec()];
    println!(
        "{label:<32} r={} rho={} alpha={} icc2_1={} icc3_1={}",
        fmt(pearson(truth, recon)),
        fmt(spearman(truth, recon)),
        fmt(cronbach_alpha(&items)),
        fmt(icc(&paired, IccForm::Icc2_1)),
        fmt(icc(&paired, IccForm::Icc3_1)),
    );
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "probes/weather-p1/jc_input.bin".to_string());
    let shuffle = std::env::args().any(|a| a == "--shuffle");

    let mut buf = Vec::new();
    match std::fs::File::open(&path).and_then(|mut f| f.read_to_end(&mut buf)) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            eprintln!("regenerate with: python3 probes/weather-p1/export_for_jc.py");
            std::process::exit(2);
        }
    }

    if buf.len() < 16 {
        eprintln!("{path}: truncated header ({} bytes)", buf.len());
        std::process::exit(2);
    }
    let mut hdr = [0u8; 8];
    hdr.copy_from_slice(&buf[0..8]);
    let n_vars = i64::from_le_bytes(hdr) as usize;
    hdr.copy_from_slice(&buf[8..16]);
    let n_points = i64::from_le_bytes(hdr) as usize;
    let mut off = 16usize;
    let want = 16 + n_vars * n_points * 2 * 8;
    if buf.len() != want {
        eprintln!(
            "{path}: expected {want} bytes for {n_vars}×{n_points}, got {}",
            buf.len()
        );
        std::process::exit(2);
    }

    println!("weather-substrate gate 1 — jc::reliability over palette256 round-trip");
    println!("input: {path}  vars={n_vars}  points/var={n_points}");
    if shuffle {
        println!("MODE: --shuffle (negative control — every statistic MUST collapse)");
    }
    println!();

    let mut all_truth = Vec::new();
    let mut all_recon = Vec::new();
    for i in 0..n_vars {
        let truth = read_f64s(&buf, &mut off, n_points);
        let mut recon = read_f64s(&buf, &mut off, n_points);
        if shuffle {
            recon.rotate_left(n_points / 3 + 1); // deterministic mismatch
        }
        let name = VAR_NAMES.get(i).copied().unwrap_or("var");
        show(name, &truth, &recon);
        all_truth.extend_from_slice(&truth);
        all_recon.extend_from_slice(&recon);
    }
    println!();
    show("POOLED (shared floor)", &all_truth, &all_recon);
    println!(
        "\nnote: ICC(2,1) is absolute agreement, ICC(3,1) consistency — reported\n\
         separately so a scale shift cannot hide behind a consistency number."
    );
}
