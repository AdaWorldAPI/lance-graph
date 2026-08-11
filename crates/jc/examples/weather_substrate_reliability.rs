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

/// Read `n` little-endian `f64`s from `buf` starting at `*off`, advancing `off`.
///
/// The caller is responsible for bounds: `main` validates the total length
/// against the header (`16 + n_vars * n_points * 2 * 8`) before any call, so a
/// short buffer is rejected up front rather than panicking mid-parse.
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

/// Parse and VALIDATE the 16-byte header, returning `(n_vars, n_points, want)`.
///
/// Rejects, rather than trusting: a short buffer, a negative dimension (an
/// `i64 -1` would otherwise become a huge `usize`), an arithmetic overflow in
/// the expected size, and a length mismatch. Only after all four does any
/// payload byte get read — so a malformed file exits cleanly instead of
/// panicking or attempting a giant allocation.
fn parse_header(buf: &[u8]) -> Result<(usize, usize, usize), String> {
    if buf.len() < 16 {
        return Err(format!("truncated header ({} bytes, need 16)", buf.len()));
    }
    let mut hdr = [0u8; 8];
    hdr.copy_from_slice(&buf[0..8]);
    let n_vars_i = i64::from_le_bytes(hdr);
    hdr.copy_from_slice(&buf[8..16]);
    let n_points_i = i64::from_le_bytes(hdr);
    if n_vars_i < 0 || n_points_i < 0 {
        return Err(format!(
            "negative dimension in header (n_vars={n_vars_i}, n_points={n_points_i})"
        ));
    }
    let (n_vars, n_points) = (n_vars_i as usize, n_points_i as usize);
    let want = n_vars
        .checked_mul(n_points)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_mul(8))
        .and_then(|v| v.checked_add(16))
        .ok_or_else(|| format!("header dimensions overflow: {n_vars}×{n_points}"))?;
    if buf.len() != want {
        return Err(format!(
            "expected {want} bytes for {n_vars}×{n_points}, got {}",
            buf.len()
        ));
    }
    Ok((n_vars, n_points, want))
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

    let (n_vars, n_points, want) = match parse_header(&buf) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(2);
        }
    };
    let mut off = 16usize;
    debug_assert_eq!(buf.len(), want);

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a well-formed buffer for `n_vars × n_points` (payload zeroed).
    fn hdr(n_vars: i64, n_points: i64, payload: usize) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&n_vars.to_le_bytes());
        v.extend_from_slice(&n_points.to_le_bytes());
        v.resize(16 + payload, 0);
        v
    }

    #[test]
    fn valid_header_round_trips() {
        // 2 vars × 3 points × 2 series × 8 bytes = 96 payload bytes.
        let buf = hdr(2, 3, 96);
        assert_eq!(parse_header(&buf), Ok((2, 3, 112)));
    }

    #[test]
    fn truncated_header_is_rejected() {
        assert!(parse_header(&[0u8; 8]).is_err(), "8-byte header must fail");
        assert!(parse_header(&[]).is_err(), "empty buffer must fail");
    }

    #[test]
    fn truncated_payload_is_rejected() {
        // Header promises 96 payload bytes; supply 40.
        assert!(parse_header(&hdr(2, 3, 40)).is_err());
    }

    #[test]
    fn negative_dimensions_are_rejected() {
        // Without the sign check these become huge usizes and the length
        // comparison could be satisfied by a wrapped `want`.
        assert!(parse_header(&hdr(-1, 3, 96)).is_err(), "negative n_vars");
        assert!(parse_header(&hdr(2, -3, 96)).is_err(), "negative n_points");
    }

    #[test]
    fn overflowing_dimensions_are_rejected() {
        // n_vars × n_points × 2 × 8 overflows usize on 64-bit.
        let buf = hdr(i64::MAX, i64::MAX, 0);
        assert!(parse_header(&buf).is_err(), "overflow must not wrap");
    }

    #[test]
    fn read_f64s_reads_little_endian_and_advances() {
        let vals = [1.5f64, -2.25, 1e-3];
        let mut buf = Vec::new();
        for v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let mut off = 0usize;
        let got = read_f64s(&buf, &mut off, 3);
        assert_eq!(got, vals.to_vec());
        assert_eq!(off, 24, "offset must advance by 8 per value");
    }
}
