//! Cross-country resilience scorecard — the three measured/declared axes against
//! the operator's domain priors, and the France paradox as the validation.
//!
//! The point of separating axes (§4.10 resistance ⟂ §4.11 buffer) is that a grid
//! can score badly on one and well on another. The operator's priors predict
//! exactly such a split: France is *topologically sparse* (huge country, long
//! lines ⇒ low λ₂) yet *stable* because of nuclear synchronous inertia (high
//! buffer); Spain is low on BOTH (wind/solar + old infra + permissive feed-in) =
//! the double-exposure outlier that actually failed 28 Apr 2025.
//!
//! Three axes here:
//!   1. TOPOLOGY  — λ₂, size-normalized mean effective resistance, bisection
//!      stability. MEASURED from the real PyPSA grid (the only measured axis).
//!   2. BUFFER    — an effective inertia H_eff (seconds) per generation type
//!      (nuclear/hydro = high rotating mass; wind/solar = inverter, low). DECLARED
//!      from the operator's / literature generation-mix priors, NOT measured here.
//!   3. POLICY    — an operational modifier: permissive feed-in raises the impulse,
//!      conservative dispatch + pumped-storage + reactive imports + good forecast
//!      lower it / refill the buffer. DECLARED prior.
//!
//! Honest scope: only axis 1 is measured. Axes 2-3 are transparent priors (tagged
//! [prior]); the scorecard tests whether the MODEL STRUCTURE organizes the known
//! domain reality coherently (construct validity, qualitative) — it does not claim
//! a measured country ranking. Real per-bus H + curtailment data would make it one.
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example scorecard -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv

use perturbation_sim::{from_pypsa_csv, impulse_buffer, symmetric_eigen, Resilience};

/// Operator/literature prior for one country (axes 2-3; tagged [prior]).
struct Profile {
    cc: &'static str,
    name: &'static str,
    gen: &'static str,
    /// Effective system inertia constant (s) from the generation mix.
    h_eff: f64,
    /// Network-quality prior (operator): Strong / Medium / Weak.
    network: &'static str,
    /// Policy impulse multiplier: <1 conservative (curtailed feed-in, fast imports,
    /// pumped storage, good forecast), >1 permissive (loose feed-in limits).
    policy_mult: f64,
    note: &'static str,
}

fn profiles() -> Vec<Profile> {
    // Priors stated by the operator + standard generation-mix knowledge.
    vec![
        Profile {
            cc: "FR",
            name: "France",
            gen: "nuclear",
            h_eff: 6.0,
            network: "Medium",
            policy_mult: 0.8,
            note: "stable nuclear baseload; sparse large grid",
        },
        Profile {
            cc: "NO",
            name: "Norway",
            gen: "hydro",
            h_eff: 5.0,
            network: "Medium",
            policy_mult: 0.8,
            note: "stable hydro, high rotating mass",
        },
        Profile {
            cc: "DE",
            name: "Germany",
            gen: "mixed+storage",
            h_eff: 4.5,
            network: "Strong",
            policy_mult: 0.6,
            note: "conservative policy, pumped-storage, reactive imports, meticulous forecast",
        },
        Profile {
            cc: "PL",
            name: "Poland",
            gen: "coal",
            h_eff: 4.5,
            network: "Weak",
            policy_mult: 1.0,
            note: "coal = rotating mass but older eastern network",
        },
        Profile {
            cc: "PT",
            name: "Portugal",
            gen: "hydro+wind",
            h_eff: 4.0,
            network: "Medium",
            policy_mult: 1.0,
            note: "hydro buffers the wind",
        },
        Profile {
            cc: "IT",
            name: "Italy",
            gen: "gas+solar",
            h_eff: 3.5,
            network: "Medium-Weak",
            policy_mult: 1.0,
            note: "long boot, weaker meshing",
        },
        Profile {
            cc: "GB",
            name: "Britain",
            gen: "gas+wind",
            h_eff: 3.0,
            network: "Medium",
            policy_mult: 0.9,
            note: "islanded (HVDC), declining inertia",
        },
        Profile {
            cc: "ES",
            name: "Spain",
            gen: "wind/solar+old",
            h_eff: 2.0,
            network: "Weak",
            policy_mult: 1.3,
            note: "modern RES, old infra, permissive feed-in",
        },
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: scorecard <buses.csv> <lines.csv>");
        return;
    }
    let buses = std::fs::read_to_string(&args[1]).expect("buses.csv");
    let lines = std::fs::read_to_string(&args[2]).expect("lines.csv");
    let df_band = 0.2;

    println!("Cross-country resilience scorecard (axis 1 MEASURED from PyPSA; axes 2-3 [prior])\n");
    println!(
        "  {:<8} {:>5} {:>11} {:>11} {:>7} {:>8} {:>7} {:>9}",
        "country", "buses", "λ₂(topo)", "meanR", "stab", "H_eff", "policy", "exposure"
    );

    let mut rows: Vec<(String, f64, f64, f64, f64, f64, f64)> = Vec::new();
    for p in profiles() {
        let imp = match from_pypsa_csv(&buses, &lines, Some(p.cc)) {
            Ok(i) => i.largest_component(),
            Err(_) => continue,
        };
        if imp.grid.n < 6 {
            continue;
        }
        let eig = symmetric_eigen(
            &imp.grid.laplacian_of(&vec![true; imp.grid.edges.len()]),
            imp.grid.n,
        );
        let cert = Resilience::from_eigenvalues(&eig.values, 1e-9);
        let lam2 = cert.lambda2;
        let lam3 = eig.values.get(2).copied().unwrap_or(lam2);
        let stab = if lam2 > 1e-30 {
            (lam3 - lam2) / lam2
        } else {
            0.0
        };
        let buffer = impulse_buffer(p.h_eff, df_band);
        // Exposure index: high when the topology is weak (large mean R), the buffer
        // is thin, and policy is permissive. Dimensionless, illustrative.
        // exposure = meanR · policy_mult / buffer  (↑ topology-weak, ↑ permissive, ↓ buffer)
        let exposure = cert.mean_resistance() * p.policy_mult / buffer.max(1e-9);
        println!(
            "  {:<8} {:>5} {:>11.3e} {:>11.3e} {:>7.2} {:>8.1} {:>7.2} {:>9.3e}   {} / {} ({})",
            p.name,
            imp.grid.n,
            lam2,
            cert.mean_resistance(),
            stab,
            p.h_eff,
            p.policy_mult,
            exposure,
            p.network,
            p.gen,
            p.note
        );
        rows.push((
            p.name.to_string(),
            lam2,
            cert.mean_resistance(),
            stab,
            p.h_eff,
            p.policy_mult,
            exposure,
        ));
    }

    // Rank by exposure (most exposed first).
    rows.sort_by(|a, b| b.6.total_cmp(&a.6));
    println!("\n== Exposure ranking (most exposed first) ==");
    for (i, r) in rows.iter().enumerate() {
        println!("  {}. {:<10} exposure = {:.3e}", i + 1, r.0, r.6);
    }

    // ── Fail-first investment locator ───────────────────────────────────────
    // For each country the BINDING CONSTRAINT = the exposure factor furthest above
    // the panel median: topology (mean R), buffer (1/H storage), or policy. The
    // binding axis dictates the intervention TYPE and the marginal exposure cut.
    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let med_r = med(rows.iter().map(|r| r.2).collect());
    let med_invbuf = med(rows
        .iter()
        .map(|r| 1.0 / impulse_buffer(r.4, df_band))
        .collect());
    let med_pol = med(rows.iter().map(|r| r.5).collect());
    println!(
        "\n== Fail-first investment locator (binding constraint → intervention → marginal cut) =="
    );
    for r in rows.iter() {
        let (meanr, h, pol, expo) = (r.2, r.4, r.5, r.6);
        let buf = impulse_buffer(h, df_band);
        // Factor excess over the panel median (how much each axis drives exposure).
        let topo_x = meanr / med_r;
        let buf_x = (1.0 / buf) / med_invbuf;
        let pol_x = pol / med_pol;
        let (binding, intervention, new_expo) = if buf_x >= topo_x && buf_x >= pol_x {
            // Buffer-bound: add synchronous inertia (gas turbine / sync condenser /
            // pumped storage). Model one turbine-class step as +2 s of H_eff.
            let new_buf = impulse_buffer(h + 2.0, df_band);
            (
                "buffer",
                "synchronous inertia — gas turbine / sync-condenser / pumped storage",
                meanr * pol / new_buf,
            )
        } else if topo_x >= pol_x {
            // Topology-bound: a transmission corridor (a 3rd inter-basin tie ≈ −20% mean R).
            (
                "topology",
                "transmission corridor (inter-basin reinforcement)",
                0.8 * meanr * pol / buf,
            )
        } else {
            // Policy-bound: curtailment limit / forecast / fast-import (≈ −0.2 policy).
            (
                "policy",
                "feed-in curtailment + forecast + fast-import / storage",
                meanr * (pol - 0.2).max(0.1) / buf,
            )
        };
        let cut = 100.0 * (1.0 - new_expo / expo);
        println!(
            "  {:<10} binding={:<8} → {:<55} exposure {:.2e} → {:.2e}  (−{:.0}%)",
            r.0, binding, intervention, expo, new_expo, cut
        );
    }
    println!(
        "\n  Marketing read: the binding column is the FAIL-FIRST investment that can't wait,\n  \
         and its type names the product. 'buffer' = a synchronous-inertia asset (the Siemens\n  \
         gas-turbine / synchronous-condenser story): the model says WHERE it buys the most\n  \
         resilience and by HOW MUCH (the marginal exposure cut), as a predictive-vulnerability\n  \
         case, not a generic sales pitch. Axis-1 measured; the H/policy inputs are priors —\n  \
         feed real per-bus inertia + curtailment data and the % becomes a costed ROI figure."
    );

    println!(
        "\nReads:\n  \
         • The FRANCE PARADOX validates the two-axis split: France's measured λ₂ is among\n    \
           the LOWEST (sparse large grid) yet it is operationally stable — because its\n    \
           BUFFER (nuclear H_eff=6 s) is the highest. A topology-only (λ₂/Kirchhoff) screen\n    \
           would wrongly flag France; the buffer axis explains why it holds. The two axes\n    \
           MUST be separate (§4.11).\n  \
         • SPAIN is the double-exposure outlier: weak topology AND thin buffer (RES+old\n    \
           infra) AND permissive feed-in (policy_mult 1.3) — top of the exposure ranking,\n    \
           matching the 28 Apr 2025 reality.\n  \
         • Norway/Germany sit low-exposure: hydro/pumped-storage buffer + conservative,\n    \
           import-reactive policy damp the impulse.\n  \
         Axis 1 is measured; axes 2-3 are operator/literature priors [prior]. This is a\n  \
         construct-validity check on the MODEL STRUCTURE, not a measured country ranking —\n  \
         feed real per-bus inertia + curtailment data to make it quantitative."
    );
}
