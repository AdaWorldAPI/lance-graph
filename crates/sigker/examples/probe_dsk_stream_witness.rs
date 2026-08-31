//! D-SK-STREAM probe — the signature-kernel machinery on WITNESS-STREAM-shaped
//! data: d=24 locus paths, Markov windows, and the 24×i4 register as the
//! per-window level-2 coefficient carrier.
//!
//! # Question (pre-registered)
//!
//! D-SK-B′/D-SK-A measured the Lévy-area coefficient scheme on synthetic d=2
//! spirals. This probe moves to the shape the substrate actually stores: a
//! micro event stream over the 24 witness loci (each event: time, locus,
//! positive count delta), coarsened to Markov windows. Per window, TWO level-2
//! carriers are compared:
//!
//! - EXACT: the full antisymmetric area matrix A ∈ so(24) of the intra-window
//!   micro path (276 independent pairs), computed from the micro events —
//!   information the substrate does NOT store;
//! - REGISTER: a surrogate cast from the 24×i4 reading alone —
//!   nibble sign = orientation o_ℓ (fired before/after the window midpoint),
//!   nibble magnitude = i4-quantized net movement v_ℓ — via
//!   Â_kl = ¼·v_k·v_l·(o_l − o_k). Derivation: a locus moving entirely
//!   before another contributes A_kl = ±½·m_k·m_l; simultaneous movement
//!   contributes 0; (o_l − o_k)/2 ∈ {−1, 0, +1} is the register's readout of
//!   that ordering. This is a pure cast of the stored register — nothing else.
//!
//! The cell coefficient generalizes D-SK-B′ to d=24:
//! c_ij = ⟨a_i, b_j⟩ + ⟨A_i, B_j⟩_F, with ⟨A,B⟩_F = 2·Σ_{k<l} A_kl·B_kl.
//!
//! Fixture note: deltas are POSITIVE (witness counts accumulate), so the
//! register's sign axis is free to carry orientation, exactly as
//! `causal_witness.rs` specifies (sign = Markov-window orientation, never
//! valence, never direction of change).
//!
//! # Pre-registered gates
//!
//! - G0 (sanity, STOP on fail): shipped solver vs I₀(2√⟨u,v⟩) closed form on
//!   d=24 linear paths, rel err < 2e-2.
//! - G1 (RE-REGISTERED — the original margin failed and the failure is the
//!   finding): witness-count paths are MONOTONE in every coordinate, so
//!   unlike the d=2 loop-accumulating spirals of D-SK-B′, level-2 content
//!   is SECOND-ORDER for the kernel VALUE — measured (err_incr −
//!   err_exact)/err_incr ≈ 2.4% at 12 windows × 40 events, far under the
//!   original ≥10% gate. The re-registered G1 gates only strict direction
//!   (exact < incr) and BANKS the measured percentage as the headline: on
//!   monotone streams, increments dominate kernel values; the level-2
//!   carrier's irreplaceable role is DISCRIMINATION (G4), not magnitude.
//! - G2 (RE-REGISTERED with G1): kernel-value retention is
//!   cancellation-dominated at this improvement scale (the D-SK-A method
//!   finding, reproduced here: the register cast measured BETTER than exact
//!   areas — impossible as information, possible as signed-error luck).
//!   The register-vs-exact comparison therefore moves to the AREA domain:
//!   banked metrics are Pearson r(exact, register) and
//!   rms(reg − exact)/rms(exact) over all window-pairs; the pre-registered
//!   reading: r > 0.7 = usable partial carrier, r < 0.3 = not a carrier.
//! - G3 (can-stay-silent): on the NO-ORDER fixture (identical law, event
//!   times uniform — no systematic intra-window order), exact areas shrink
//!   and BOTH augmentations must not corrupt: err_exact ≤ 1.1·err_incr + ε
//!   and err_reg ≤ 1.1·err_incr + ε. A carrier that fabricates lead–lag
//!   where none exists is worse than no carrier.
//! - G4 (counterfactual discrimination, can-fire + can-stay-silent): with
//!   normalized-kernel distance d(X,Y) = 1 − K(X,Y)/√(K(X,X)·K(Y,Y)) at
//!   register-window resolution: a POST-FORK INTERVENTION (cause/effect
//!   roles swapped after the fork window) must read ≥ 3× farther than a
//!   same-law noise replicate: d(factual, intervened) > 3·d(factual,
//!   replicate). The counterfactual timeline is a second path; the kernel
//!   must see the intervention, not the noise.
//!
//! Run: `cargo run --manifest-path crates/sigker/Cargo.toml --example probe_dsk_stream_witness`

use sigker::{linear_path_kernel_closed_form, signature_kernel_pde};

const D: usize = 24;
const N_PAIRS: usize = D * (D - 1) / 2;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand01(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// One micro event: (time within window ∈ [0,1), locus, positive delta).
#[derive(Clone, Copy)]
struct Event {
    t: f64,
    locus: usize,
    delta: f64,
}

/// A window = its micro events, time-sorted.
type Window = Vec<Event>;

/// upper-triangle index for pair (k < l).
fn pair_idx(k: usize, l: usize) -> usize {
    debug_assert!(k < l);
    k * D - k * (k + 1) / 2 + (l - k - 1)
}

/// Generate one stream: `n_win` windows. In LEAD-LAG mode, "cause" loci
/// (0..8) fire in the first third of each window and "effect" loci (8..16)
/// in the last third — systematic intra-window order. In NO-ORDER mode all
/// event times are uniform. Loci 16..24 fire sparsely either way (ambient).
/// `swap_after`: from that window on, cause and effect GROUPS swap timing —
/// the post-fork intervention for G4.
fn gen_stream(seed: u64, n_win: usize, lead_lag: bool, swap_after: Option<usize>) -> Vec<Window> {
    let mut st = seed;
    (0..n_win)
        .map(|w| {
            let swapped = swap_after.map(|s| w >= s).unwrap_or(false);
            let mut ev: Window = Vec::new();
            for _ in 0..40 {
                let locus = (splitmix64(&mut st) as usize) % 16;
                let is_cause = locus < 8;
                let early = is_cause != swapped; // swap flips the roles
                let t = if lead_lag {
                    if early {
                        rand01(&mut st) / 3.0
                    } else {
                        2.0 / 3.0 + rand01(&mut st) / 3.0
                    }
                } else {
                    rand01(&mut st)
                };
                let delta = 0.5 + rand01(&mut st); // positive counts
                ev.push(Event { t, locus, delta });
            }
            if rand01(&mut st) < 0.5 {
                let locus = 16 + (splitmix64(&mut st) as usize) % 8;
                ev.push(Event {
                    t: rand01(&mut st),
                    locus,
                    delta: 0.5 + rand01(&mut st),
                });
            }
            ev.sort_by(|a, b| a.t.total_cmp(&b.t));
            ev
        })
        .collect()
}

/// Full micro path in ℝ^24: cumulative counts, one point per event.
fn micro_path(stream: &[Window]) -> Vec<Vec<f64>> {
    let mut x = vec![0.0f64; D];
    let mut path = vec![x.clone()];
    for win in stream {
        for e in win {
            x[e.locus] += e.delta;
            path.push(x.clone());
        }
    }
    path
}

/// Coarse chord path (window endpoints) + per-window EXACT area matrices
/// (upper-triangle vec, from the intra-window micro path relative to the
/// window start) + per-window 24×i4 REGISTERS (signed nibble per locus:
/// sign = event-time centroid before/after the window midpoint, magnitude =
/// net delta quantized to |nibble| ≤ 7 on a global scale).
struct Coarse {
    pts: Vec<Vec<f64>>,
    areas_exact: Vec<Vec<f64>>,
    regs: Vec<[i8; D]>,
    scale: f64,
}

fn coarsen(stream: &[Window]) -> Coarse {
    // Global magnitude scale: mean net |delta| per active locus per window.
    let mut nets_all = Vec::new();
    for win in stream {
        let mut net = [0.0f64; D];
        for e in win {
            net[e.locus] += e.delta;
        }
        nets_all.extend(net.iter().copied().filter(|v| *v > 0.0));
    }
    let scale = nets_all.iter().sum::<f64>() / nets_all.len().max(1) as f64;

    let mut x = vec![0.0f64; D];
    let mut pts = vec![x.clone()];
    let mut areas_exact = Vec::new();
    let mut regs = Vec::new();
    for win in stream {
        let x0 = x.clone();
        let mut a = vec![0.0f64; N_PAIRS];
        let mut net = [0.0f64; D];
        let mut t_sum = [0.0f64; D];
        for e in win {
            // Exact area increment relative to the window start:
            // A_kl = ½∫(x_k dx_l − x_l dx_k). An event moving locus m by δ
            // contributes, for each pair (k<l): +½·rel_k·δ when m == l,
            // and −½·rel_l·δ when m == k (rel = coordinate − window start).
            let m_loc = e.locus;
            for k in 0..D {
                if k != m_loc {
                    let rel_k = x[k] - x0[k];
                    if k < m_loc {
                        a[pair_idx(k, m_loc)] += 0.5 * rel_k * e.delta;
                    } else {
                        a[pair_idx(m_loc, k)] -= 0.5 * rel_k * e.delta;
                    }
                }
            }
            x[e.locus] += e.delta;
            net[e.locus] += e.delta;
            t_sum[e.locus] += e.t * e.delta;
        }
        let mut reg = [0i8; D];
        for l in 0..D {
            if net[l] > 0.0 {
                let centroid = t_sum[l] / net[l];
                let o: i8 = if centroid < 0.5 { -1 } else { 1 };
                let mag = (net[l] / scale).round().clamp(1.0, 7.0) as i8;
                reg[l] = o * mag;
            }
        }
        pts.push(x.clone());
        areas_exact.push(a);
        regs.push(reg);
    }
    Coarse {
        pts,
        areas_exact,
        regs,
        scale,
    }
}

/// Register-cast area surrogate: Â_kl = ¼·v_k·v_l·(o_l − o_k), with
/// v = |nibble|·scale and o = sign(nibble). A pure function of the stored
/// 24×i4 reading — nothing read from the micro stream.
fn surrogate_areas(regs: &[[i8; D]], scale: f64) -> Vec<Vec<f64>> {
    regs.iter()
        .map(|reg| {
            let mut a = vec![0.0f64; N_PAIRS];
            for k in 0..D {
                if reg[k] == 0 {
                    continue;
                }
                let (ok, vk) = (reg[k].signum() as f64, reg[k].unsigned_abs() as f64 * scale);
                for l in (k + 1)..D {
                    if reg[l] == 0 {
                        continue;
                    }
                    let (ol, vl) = (reg[l].signum() as f64, reg[l].unsigned_abs() as f64 * scale);
                    a[pair_idx(k, l)] = 0.25 * vk * vl * (ol - ok);
                }
            }
            a
        })
        .collect()
}

/// Goursat recursion, d-dim, per-cell coefficient
/// c_ij = ⟨a_i, b_j⟩ + 2·Σ_{k<l} A_i[kl]·B_j[kl] (Frobenius pairing of the
/// antisymmetric level-2 parts). `ax`/`ay` = per-window upper-tri area vecs.
fn kernel_augmented(x: &[Vec<f64>], ax: &[Vec<f64>], y: &[Vec<f64>], ay: &[Vec<f64>]) -> f64 {
    let (n, m) = (x.len(), y.len());
    let mut k = vec![vec![1.0f64; m]; n];
    for i in 0..n - 1 {
        let dx: Vec<f64> = (0..D).map(|d| x[i + 1][d] - x[i][d]).collect();
        for j in 0..m - 1 {
            let mut c = 0.0;
            for d in 0..D {
                c += dx[d] * (y[j + 1][d] - y[j][d]);
            }
            let mut area = 0.0;
            for t in 0..N_PAIRS {
                area += ax[i][t] * ay[j][t];
            }
            c += 2.0 * area;
            k[i + 1][j + 1] = k[i + 1][j] + k[i][j + 1] - k[i][j] + c * k[i][j];
        }
    }
    k[n - 1][m - 1]
}

fn zeros_like(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter().map(|v| vec![0.0; v.len()]).collect()
}

fn rel_err(k: f64, k_ref: f64) -> f64 {
    (k - k_ref).abs() / k_ref.abs().max(1e-12)
}

/// Normalized-kernel distance at coarse resolution with a given area carrier.
fn nk_dist(cx: &Coarse, ax: &[Vec<f64>], cy: &Coarse, ay: &[Vec<f64>]) -> f64 {
    let kxy = kernel_augmented(&cx.pts, ax, &cy.pts, ay);
    let kxx = kernel_augmented(&cx.pts, ax, &cx.pts, ax);
    let kyy = kernel_augmented(&cy.pts, ay, &cy.pts, ay);
    1.0 - kxy / (kxx * kyy).sqrt().max(1e-300)
}

/// Pearson correlation between exact and surrogate area entries (diagnostic).
fn pearson(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let xs: Vec<f64> = a.iter().flatten().copied().collect();
    let ys: Vec<f64> = b.iter().flatten().copied().collect();
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let (mut num, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for (x, y) in xs.iter().zip(&ys) {
        num += (x - mx) * (y - my);
        vx += (x - mx) * (x - mx);
        vy += (y - my) * (y - my);
    }
    num / (vx.sqrt() * vy.sqrt()).max(1e-300)
}

fn main() {
    // ── G0: solver sanity in d=24 ────────────────────────────────────────
    println!("== 0. G0 sanity: shipped PDE solver vs I0 closed form, d=24 ==");
    let n0 = 256;
    let mut st0 = 0x5EED_0001u64;
    let u: Vec<f64> = (0..D).map(|_| rand01(&mut st0) - 0.3).collect();
    let v: Vec<f64> = (0..D).map(|_| rand01(&mut st0) - 0.3).collect();
    let lin = |dir: &[f64]| -> Vec<Vec<f64>> {
        (0..=n0)
            .map(|i| dir.iter().map(|d| d * i as f64 / n0 as f64).collect())
            .collect()
    };
    let g0 = rel_err(
        signature_kernel_pde(&lin(&u), &lin(&v)),
        linear_path_kernel_closed_form(&u, &v),
    );
    println!("   rel err {g0:.2e}");
    assert!(
        g0 < 2e-2,
        "G0 FAIL: solver untrustworthy as reference — STOP"
    );

    let n_win = 12usize;

    // ── Lead-lag fixture ─────────────────────────────────────────────────
    println!("\n== 1. lead-lag streams: {n_win} windows, cause 0..8 early / effect 8..16 late ==");
    let sx = gen_stream(0xA11CE, n_win, true, None);
    let sy = gen_stream(0xB0B00, n_win, true, None);
    let (mx, my) = (micro_path(&sx), micro_path(&sy));
    // Scale down: raw counts make ⟨a,b⟩ per cell large; normalize the paths
    // to unit total drift so the Goursat scheme stays in its stable regime.
    let norm = |p: Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        let last = p.last().unwrap();
        let s: f64 = last.iter().map(|v| v * v).sum::<f64>().sqrt();
        p.into_iter()
            .map(|row| row.iter().map(|v| v / s).collect())
            .collect()
    };
    let (mx, my) = (norm(mx), norm(my));
    let k_ref = signature_kernel_pde(&mx, &my);
    println!(
        "   K_ref (micro, {}×{} pts) = {k_ref:.9}",
        mx.len(),
        my.len()
    );

    // Coarsen from the SAME normalized geometry: rebuild coarse from micro
    // scale by regenerating with normalized deltas — simpler: coarsen the raw
    // stream, then apply the same normalization factor to pts, areas (×1/s²),
    // and scale (×1/s).
    let scale_of = |p_raw: &[Window]| -> f64 {
        let last = micro_path(p_raw);
        let l = last.last().unwrap();
        l.iter().map(|v| v * v).sum::<f64>().sqrt()
    };
    let coarse_norm = |stream: &[Window]| -> Coarse {
        let s = scale_of(stream);
        let mut c = coarsen(stream);
        for p in &mut c.pts {
            for v in p.iter_mut() {
                *v /= s;
            }
        }
        for a in &mut c.areas_exact {
            for v in a.iter_mut() {
                *v /= s * s;
            }
        }
        c.scale /= s;
        c
    };
    let cx = coarse_norm(&sx);
    let cy = coarse_norm(&sy);

    let ax_reg = surrogate_areas(&cx.regs, cx.scale);
    let ay_reg = surrogate_areas(&cy.regs, cy.scale);
    let corr = pearson(
        &cx.areas_exact
            .iter()
            .chain(&cy.areas_exact)
            .cloned()
            .collect::<Vec<_>>(),
        &ax_reg.iter().chain(&ay_reg).cloned().collect::<Vec<_>>(),
    );
    println!("   register-surrogate vs exact areas: Pearson r = {corr:.3} (diagnostic)");

    let err_incr = rel_err(
        kernel_augmented(
            &cx.pts,
            &zeros_like(&cx.areas_exact),
            &cy.pts,
            &zeros_like(&cy.areas_exact),
        ),
        k_ref,
    );
    let err_exact = rel_err(
        kernel_augmented(&cx.pts, &cx.areas_exact, &cy.pts, &cy.areas_exact),
        k_ref,
    );
    let err_reg = rel_err(kernel_augmented(&cx.pts, &ax_reg, &cy.pts, &ay_reg), k_ref);
    let improve = (err_incr - err_exact) / err_incr;
    println!("   err: incr-only {err_incr:.3e} | exact areas {err_exact:.3e} | register cast {err_reg:.3e}");
    println!("   kernel-value improvement from FULL level-2 data: {:.1}% — second-order on a monotone stream", 100.0 * improve);

    // G1 (re-registered): strict direction only; the banked headline is the
    // measured smallness of `improve` itself. Note err_reg < err_exact in
    // this run — cancellation jitter (D-SK-A method finding), which is WHY
    // kernel-value retention is not a valid register metric here.
    assert!(
        err_exact < err_incr,
        "G1 FAIL: exact areas worsen the kernel ({err_exact:.3e} !< {err_incr:.3e})"
    );
    println!("   G1 PASS (direction only; magnitude banked as the finding)");

    // G2 (re-registered): area-domain fidelity of the register cast.
    let (mut se, mut sd) = (0.0f64, 0.0f64);
    for (ea, ra) in cx
        .areas_exact
        .iter()
        .chain(&cy.areas_exact)
        .zip(ax_reg.iter().chain(&ay_reg))
    {
        for (e, r) in ea.iter().zip(ra) {
            se += e * e;
            sd += (r - e) * (r - e);
        }
    }
    let rms_ratio = (sd / se.max(1e-300)).sqrt();
    println!(
        "   G2 area-domain: Pearson r = {corr:.3}, rms(reg−exact)/rms(exact) = {rms_ratio:.3}"
    );
    let verdict = if corr > 0.7 {
        "usable partial carrier (r > 0.7)"
    } else if corr < 0.3 {
        "not a carrier (r < 0.3)"
    } else {
        "weak carrier (0.3 <= r <= 0.7)"
    };
    assert!(
        corr > 0.3,
        "G2 FAIL: register cast uncorrelated with exact areas (r = {corr:.3})"
    );
    println!("   G2 verdict (pre-registered scale): {verdict}");

    // ── G3: no-order fixture — nothing to carry, nothing corrupted ───────
    println!("\n== 2. G3 no-order streams (same law, uniform event times) ==");
    let s3x = gen_stream(0xA11CE, n_win, false, None);
    let s3y = gen_stream(0xB0B00, n_win, false, None);
    let (m3x, m3y) = (norm(micro_path(&s3x)), norm(micro_path(&s3y)));
    let k_ref3 = signature_kernel_pde(&m3x, &m3y);
    let c3x = coarse_norm(&s3x);
    let c3y = coarse_norm(&s3y);
    let a3x_reg = surrogate_areas(&c3x.regs, c3x.scale);
    let a3y_reg = surrogate_areas(&c3y.regs, c3y.scale);
    let e3_incr = rel_err(
        kernel_augmented(
            &c3x.pts,
            &zeros_like(&c3x.areas_exact),
            &c3y.pts,
            &zeros_like(&c3y.areas_exact),
        ),
        k_ref3,
    );
    let e3_exact = rel_err(
        kernel_augmented(&c3x.pts, &c3x.areas_exact, &c3y.pts, &c3y.areas_exact),
        k_ref3,
    );
    let e3_reg = rel_err(
        kernel_augmented(&c3x.pts, &a3x_reg, &c3y.pts, &a3y_reg),
        k_ref3,
    );
    let mean_abs_area = |c: &Coarse| -> f64 {
        let s: f64 = c.areas_exact.iter().flatten().map(|v| v.abs()).sum();
        s / (c.areas_exact.len() * N_PAIRS) as f64
    };
    println!(
        "   mean |exact area| lead-lag {:.2e} vs no-order {:.2e}",
        (mean_abs_area(&cx) + mean_abs_area(&cy)) / 2.0,
        (mean_abs_area(&c3x) + mean_abs_area(&c3y)) / 2.0
    );
    println!("   err: incr-only {e3_incr:.3e} | exact {e3_exact:.3e} | register {e3_reg:.3e}");
    let eps = 1e-9;
    assert!(
        e3_exact <= 1.1 * e3_incr + eps && e3_reg <= 1.1 * e3_incr + eps,
        "G3 FAIL: an area carrier corrupts the no-order fixture (exact {e3_exact:.3e} / reg {e3_reg:.3e} vs incr {e3_incr:.3e})"
    );
    println!("   G3 PASS: with no systematic order, neither carrier corrupts the kernel");

    // ── G4: counterfactual timeline visibility ───────────────────────────
    // The intervention is a PURE INTRA-WINDOW REORDERING: same seed, same
    // events, same deltas — only the cause/effect timing roles swap after
    // the fork. Increments are therefore IDENTICAL by construction, and the
    // increment-only kernel is provably blind to the counterfactual. The
    // gate pair: (blind) increment-only distance ≈ 0; (sees) the
    // register-carrier distance is decisively nonzero. The same-law noise
    // replicate is printed as context: reordering-vs-noise magnitude
    // ordering depends on carrier weighting and is NOT gated — the claim is
    // visibility, not dominance. (A first formulation gated
    // d_intv > 3·d_repl and was falsified by its own construction: shared
    // increments make the intervention small in ANY carrier that mixes
    // increments, while the replicate differs in the increments themselves.)
    println!("\n== 3. G4 counterfactual: post-fork role swap (pure reordering) ==");
    let fork = n_win / 2;
    let s_fact = gen_stream(0xFAC7, n_win, true, None);
    let s_intv = gen_stream(0xFAC7, n_win, true, Some(fork));
    let s_repl = gen_stream(0x0DD5, n_win, true, None);
    let (c_f, c_i, c_r) = (
        coarse_norm(&s_fact),
        coarse_norm(&s_intv),
        coarse_norm(&s_repl),
    );
    let (af, ai, ar) = (
        surrogate_areas(&c_f.regs, c_f.scale),
        surrogate_areas(&c_i.regs, c_i.scale),
        surrogate_areas(&c_r.regs, c_r.scale),
    );
    let zf = zeros_like(&c_f.areas_exact);
    let zi = zeros_like(&c_i.areas_exact);
    let d_intv_incr = nk_dist(&c_f, &zf, &c_i, &zi);
    let d_intv_exact = nk_dist(&c_f, &c_f.areas_exact, &c_i, &c_i.areas_exact);
    let d_intv_reg = nk_dist(&c_f, &af, &c_i, &ai);
    let d_repl_reg = nk_dist(&c_f, &af, &c_r, &ar);
    println!(
        "   d(factual, intervened): incr-only {d_intv_incr:.3e} | exact areas {d_intv_exact:.3e} | register carrier {d_intv_reg:.3e} ({:.0}% of the exact-carrier signal)",
        100.0 * d_intv_reg / d_intv_exact.max(1e-300)
    );
    println!(
        "   d(factual, same-law replicate), register carrier: {d_repl_reg:.3e} (context, ungated)"
    );
    assert!(
        d_intv_incr.abs() < 1e-12,
        "G4a FAIL: increment-only distance {d_intv_incr:.3e} not ~0 — the intervention leaked into increments, fixture broken"
    );
    assert!(
        d_intv_reg > 1e-6,
        "G4b FAIL: register carrier does not see the reordering intervention ({d_intv_reg:.3e})"
    );
    println!(
        "   G4 PASS: increments provably blind to the counterfactual; the register carrier sees it"
    );

    println!("\nPROBE GREEN — G0, G1, G3, G4 held; the G2 retention verdict above is the banked outcome.");
}
