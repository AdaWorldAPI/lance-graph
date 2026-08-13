//! `D-WXS-7` / `D-WXS-8` — bar B6/B7, RUN.
//!
//! Reads the raw pair-distance arrays [`crate`]'s companion Python stage
//! (`probes/weather-p1/fidelity_probe_fetch.py` + `fidelity_probe_prep.py`)
//! wrote from REAL, live-fetched ARCO-ERA5 grid data at three real seasons,
//! and computes every ρ with [`jc::reliability::spearman`] — per bar B6's own
//! wording, *"computed with `jc::reliability::spearman`"*, not a Python
//! re-implementation. Rust's role here is exactly this: the metric. Fetching
//! real data over HTTP is Python's role (`weather-poc` stays zero-dep in its
//! default `[lib]` build; `jc` is a **dev-dependency only** — see
//! `Cargo.toml`'s comment on why that is safe, unlike `helix`).
//!
//! Bars, verbatim from `.claude/plans/weather-soa-bake-v1.md` sec 4 W3:
//!
//! **Bar B6 (`D-WXS-7`)** — the K×K pair (or its within-variable degenerate
//! twin where only one K variable is available), at each season:
//! - (a) primary: ρ ≥ 0.9996 at resolution 256
//! - (b) control: the shuffled-decode-table arm must score < 0.98
//! - (c) can-it-differ: the 16/64/256 ladder must be **strictly monotone**,
//!   checked BEFORE any verdict is read
//! - thesis-level KILL (reported, does not block): any arm below ρ ≈ 0.9
//!
//! **Bar B7 (`D-WXS-8`)** — every cross-variable pair at each season:
//! - primary: cross-UNIT pairs must reach ρ ≥ 0.9996 on the shared floor
//! - control: the per-variable floor must LOSE (lower ρ) on every cross-unit
//!   pair — a control that WINS refutes bar B7's premise
//! - stay-silent twin: within-variable, the shared floor must not cost
//!   resolution — |ρ_shared − ρ_pervar| ≤ 0.0001 AND zero empty buckets
//!
//! This example does not soften a failing bar. Every verdict below is
//! printed exactly as measured, including any FAIL.

use std::fs;
use std::path::Path;

/// Absolute at compile time via `CARGO_MANIFEST_DIR`, not relative to
/// whatever directory `cargo run` happens to be invoked from.
fn pair_dir() -> String {
    format!(
        "{}/../../probes/weather-p1/fixture/fidelity_pairs",
        env!("CARGO_MANIFEST_DIR")
    )
}
const SEASONS: [&str; 3] = ["winter", "spring", "summer"];
const PRIMARY_RHO: f64 = 0.9996;
const SHUFFLE_CEILING: f64 = 0.98;
const THESIS_KILL_FLOOR: f64 = 0.9;
const TWIN_TOLERANCE: f64 = 0.0001;

/// Read a flat little-endian `f64` array with no header — the format
/// `fidelity_probe_prep.py::write_f64` emits (`ndarray.astype('<f8').tofile`).
fn read_f64(path: &Path) -> Vec<f64> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(
        bytes.len() % 8,
        0,
        "{}: not a whole number of f64s",
        path.display()
    );
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn spearman_or_panic(a: &[f64], b: &[f64], what: &str) -> f64 {
    jc::reliability::spearman(a, b)
        .unwrap_or_else(|| panic!("{what}: jc::reliability::spearman returned None (length mismatch or non-finite input) — a data-prep bug, not a measurement"))
}

/// The `key:value` / `cross:key:unit_a:unit_b` sidecar
/// `fidelity_probe_prep.py` writes instead of JSON (see its own comment for
/// why — no parsing crate needed).
struct SeasonMeta {
    within_var: String,
    within_empty_buckets: u32,
    kxk_name: String,
    kxk_va: String,
    kxk_vb: String,
    cross_pairs: Vec<(String, String, String)>, // (pair_key, unit_a, unit_b)
}

fn read_meta(season: &str) -> SeasonMeta {
    let dir = pair_dir();
    let text = fs::read_to_string(format!("{dir}/{season}_meta.txt"))
        .unwrap_or_else(|e| panic!("read {season}_meta.txt: {e}"));
    let mut within_var = String::new();
    let mut within_empty_buckets = 0u32;
    let mut kxk_name = String::new();
    let mut kxk_va = String::new();
    let mut kxk_vb = String::new();
    let mut cross_pairs = Vec::new();
    for line in text.lines() {
        let mut parts = line.splitn(4, ':');
        let tag = parts.next().unwrap_or("");
        match tag {
            "within_var" => within_var = parts.next().unwrap_or("").to_string(),
            "within_empty_buckets" => {
                within_empty_buckets = parts.next().unwrap_or("0").parse().unwrap_or(0)
            }
            "kxk_name" => kxk_name = parts.next().unwrap_or("").to_string(),
            "kxk_va" => kxk_va = parts.next().unwrap_or("").to_string(),
            "kxk_vb" => kxk_vb = parts.next().unwrap_or("").to_string(),
            "cross" => {
                let key = parts.next().unwrap_or("").to_string();
                let ua = parts.next().unwrap_or("").to_string();
                let ub = parts.next().unwrap_or("").to_string();
                cross_pairs.push((key, ua, ub));
            }
            "" => {}
            other => panic!("{season}_meta.txt: unknown tag {other:?}"),
        }
    }
    SeasonMeta {
        within_var,
        within_empty_buckets,
        kxk_name,
        kxk_va,
        kxk_vb,
        cross_pairs,
    }
}

struct Verdict {
    name: String,
    pass: bool,
    detail: String,
}

fn main() {
    let mut all_verdicts: Vec<Verdict> = Vec::new();

    for season in SEASONS {
        println!("\n== {season} ==");
        let meta = read_meta(season);
        let dir = pair_dir();

        // ---- Bar B6: the K x K pair (or its within-variable degenerate) ----
        let truth = read_f64(Path::new(&format!("{dir}/{season}_kxk_truth.f64")));
        let rho16 = spearman_or_panic(
            &truth,
            &read_f64(Path::new(&format!("{dir}/{season}_kxk_code_L16.f64"))),
            "kxk L16",
        );
        let rho64 = spearman_or_panic(
            &truth,
            &read_f64(Path::new(&format!("{dir}/{season}_kxk_code_L64.f64"))),
            "kxk L64",
        );
        let rho256 = spearman_or_panic(
            &truth,
            &read_f64(Path::new(&format!("{dir}/{season}_kxk_code_L256.f64"))),
            "kxk L256",
        );
        let rho_shuffled = spearman_or_panic(
            &truth,
            &read_f64(Path::new(&format!("{dir}/{season}_kxk_code_shuffled.f64"))),
            "kxk shuffled",
        );
        println!(
            "  B6 pair: {} (va={} vb={})",
            meta.kxk_name, meta.kxk_va, meta.kxk_vb
        );
        println!("    rho(L16)={rho16:.6}  rho(L64)={rho64:.6}  rho(L256)={rho256:.6}  rho(shuffled)={rho_shuffled:.6}");

        let ladder_monotone = rho16 < rho64 && rho64 < rho256;
        let v_ladder = Verdict {
            name: format!("B6c[{season}] resolution ladder strictly monotone"),
            pass: ladder_monotone,
            detail: format!(
                "rho16={rho16:.6} rho64={rho64:.6} rho256={rho256:.6} (need rho16<rho64<rho256)"
            ),
        };
        println!(
            "    B6(c) ladder monotone: {} — {}",
            if v_ladder.pass { "PASS" } else { "FAIL" },
            v_ladder.detail
        );

        // Bar B6(a)/(b) are only meaningful once (c) has been checked, per
        // the plan's own ordering ("run BEFORE any verdict"). Still computed
        // and reported either way — a FAIL is reported, never hidden.
        let v_primary = Verdict {
            name: format!("B6a[{season}] primary rho256 >= {PRIMARY_RHO}"),
            pass: rho256 >= PRIMARY_RHO,
            detail: format!("rho256={rho256:.6}"),
        };
        let v_control = Verdict {
            name: format!("B6b[{season}] shuffled control < {SHUFFLE_CEILING}"),
            pass: rho_shuffled < SHUFFLE_CEILING,
            detail: format!("rho_shuffled={rho_shuffled:.6}"),
        };
        let v_thesis = Verdict {
            name: format!("B6-thesis-kill[{season}] every arm >= {THESIS_KILL_FLOOR}"),
            pass: [rho16, rho64, rho256]
                .iter()
                .all(|&r| r >= THESIS_KILL_FLOOR),
            detail: format!(
                "min(rho16,rho64,rho256)={:.6}",
                rho16.min(rho64).min(rho256)
            ),
        };
        for v in [&v_primary, &v_control, &v_thesis] {
            println!(
                "    {}: {} — {}",
                v.name,
                if v.pass { "PASS" } else { "FAIL" },
                v.detail
            );
        }
        all_verdicts.push(v_ladder);
        all_verdicts.push(v_primary);
        all_verdicts.push(v_control);
        all_verdicts.push(v_thesis);

        // ---- Bar B7: every cross-variable pair, shared vs per-variable ----
        for (key, unit_a, unit_b) in &meta.cross_pairs {
            let truth = read_f64(Path::new(&format!("{dir}/{season}_{key}_truth.f64")));
            let code_shared = read_f64(Path::new(&format!("{dir}/{season}_{key}_code_shared.f64")));
            let code_pervar = read_f64(Path::new(&format!("{dir}/{season}_{key}_code_pervar.f64")));
            let rho_shared = spearman_or_panic(&truth, &code_shared, &format!("{key} shared"));
            let rho_pervar = spearman_or_panic(&truth, &code_pervar, &format!("{key} pervar"));
            let same_unit = unit_a == unit_b;
            println!(
                "  B7 {key:55} units=({unit_a},{unit_b}) rho_shared={rho_shared:.6} rho_pervar={rho_pervar:.6} {}",
                if same_unit { "(same-unit, informational)" } else { "(cross-unit)" }
            );
            if !same_unit {
                all_verdicts.push(Verdict {
                    name: format!(
                        "B7-primary[{season}][{key}] cross-unit rho_shared >= {PRIMARY_RHO}"
                    ),
                    pass: rho_shared >= PRIMARY_RHO,
                    detail: format!("rho_shared={rho_shared:.6} units=({unit_a},{unit_b})"),
                });
                all_verdicts.push(Verdict {
                    name: format!("B7-control[{season}][{key}] per-variable floor LOSES to shared"),
                    pass: rho_pervar < rho_shared,
                    detail: format!("rho_shared={rho_shared:.6} rho_pervar={rho_pervar:.6}"),
                });
            }
        }

        // ---- stay-silent twin: within-variable, shared must not cost ----
        let truth_w = read_f64(Path::new(&format!("{dir}/{season}_within_truth.f64")));
        let code_w_shared = read_f64(Path::new(&format!("{dir}/{season}_within_code_shared.f64")));
        let code_w_pervar = read_f64(Path::new(&format!("{dir}/{season}_within_code_pervar.f64")));
        let rho_w_shared = spearman_or_panic(&truth_w, &code_w_shared, "within shared");
        let rho_w_pervar = spearman_or_panic(&truth_w, &code_w_pervar, "within pervar");
        let diff = (rho_w_shared - rho_w_pervar).abs();
        println!(
            "  B7 stay-silent twin: within-var={} rho_shared={rho_w_shared:.6} rho_pervar={rho_w_pervar:.6} diff={diff:.6} empty_buckets={}",
            meta.within_var, meta.within_empty_buckets
        );
        all_verdicts.push(Verdict {
            name: format!("B7-twin[{season}] |rho_shared - rho_pervar| <= {TWIN_TOLERANCE}"),
            pass: diff <= TWIN_TOLERANCE,
            detail: format!(
                "rho_shared={rho_w_shared:.6} rho_pervar={rho_w_pervar:.6} diff={diff:.6}"
            ),
        });
        all_verdicts.push(Verdict {
            name: format!("B7-twin[{season}] zero empty buckets under shared floor"),
            pass: meta.within_empty_buckets == 0,
            detail: format!("empty_buckets={}", meta.within_empty_buckets),
        });
    }

    println!("\n== VERDICT SUMMARY (every one, no filtering) ==");
    let mut n_pass = 0usize;
    for v in &all_verdicts {
        println!(
            "  [{}] {}  ({})",
            if v.pass { "PASS" } else { "FAIL" },
            v.name,
            v.detail
        );
        if v.pass {
            n_pass += 1;
        }
    }
    println!(
        "\n{n_pass}/{} verdicts pass ({} FAIL)",
        all_verdicts.len(),
        all_verdicts.len() - n_pass
    );

    // Minimal hand-rolled JSON write (no serde — matches the crate's
    // zero-dep-by-construction lib; this example already carries the one
    // acceptable dev-dependency, jc, and should not add a second).
    let mut json = String::from("{\n  \"verdicts\": [\n");
    for (i, v) in all_verdicts.iter().enumerate() {
        json.push_str(&format!(
            "    {{\"name\": {:?}, \"pass\": {}, \"detail\": {:?}}}{}\n",
            v.name,
            v.pass,
            v.detail,
            if i + 1 < all_verdicts.len() { "," } else { "" }
        ));
    }
    json.push_str("  ]\n}\n");
    fs::write(
        format!("{}/../fidelity_probe_results.json", pair_dir()),
        &json,
    )
    .expect("write fidelity_probe_results.json");
    println!("\nwrote probes/weather-p1/fixture/fidelity_probe_results.json");
}
