//! D-PFP-1 runner: both lenses, one results block each, one verdict line.
//! Spec: `.claude/plans/perturbationsfeld-probe-v1.md` §9; constants: `PREREG.md`.

use perturbationsfeld_probe::{run_lens, verdict, Outcome, Report, BGE_M3, JINA_V5};

/// Print one lens's results block.
fn print(r: &Report) {
    println!("── {} ── N = {}", r.lens, r.n);
    println!(
        "  stimuli valid (base) {} / 32, comparison-valid Q_valid {}",
        r.base_valid, r.q_valid
    );
    println!(
        "  positive control pair ({}, {}) overlap {} / 8",
        r.positive_pair.0, r.positive_pair.1, r.positive_inter
    );
    println!("  empirical null N0 (cross-stimulus top-set overlap) {:.4}  [uniform reference 8/256 = {:.4}]", r.n0, 8.0 / 256.0);
    println!(
        "  retention (Σ ids kept of 8): R_E {}  R_C {}  R_S {}  R_Em {}   (per stimulus: E {:.3}  C {:.3})",
        r.r_e,
        r.r_c,
        r.r_s,
        r.r_em,
        r.r_e as f64 / r.q_valid.max(1) as f64,
        r.r_c as f64 / r.q_valid.max(1) as f64
    );
    println!(
        "  ΔR = R_E − R_C over comparison-valid stimuli: {}  (margin ±{})",
        r.delta_r,
        2 * r.q_valid
    );
    println!(
        "  relabel sanity: D(E,S) {} over {} eligible (needs ≥ {})",
        r.d_es,
        r.sanity_eligible,
        2 * r.sanity_eligible
    );
    println!(
        "  mean |ids_C| {:.2}  mean |ids_E| {:.2}  top_k padding rate {:.3}  mean L1(E,C) {:.4}",
        r.mean_c, r.mean_e, r.padding_rate, r.l1_ec
    );
    if r.theta_c_inert {
        println!("  θ is decoration in the control path (C unchanged at 2θ and θ/2)");
    }
    match &r.outcome {
        Outcome::Invalid(why) => println!("  OUTCOME: INVALID — {}", why.join("; ")),
        o => println!("  OUTCOME: {}", o.label()),
    }
}

/// Run both lenses and print the combined verdict.
fn main() {
    let sha = std::process::Command::new("git")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["log", "-1", "--format=%h", "--", "PREREG.md"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    println!(
        "D-PFP-1 Perturbationsfeld probe — PREREG commit {}",
        if sha.is_empty() {
            "(uncommitted)"
        } else {
            &sha
        }
    );
    let p = run_lens(&JINA_V5);
    print(&p);
    let r = run_lens(&BGE_M3);
    print(&r);
    println!("VERDICT: {}", verdict(&p.outcome, &r.outcome));
}
