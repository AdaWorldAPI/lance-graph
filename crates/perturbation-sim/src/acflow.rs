//! AC power flow (**the fork rung**) — Newton–Raphson on the full π-model
//! (`R + jX + jB/2`), with bus voltages and reactive power. This is the rung
//! that can represent the **voltage-collapse** mechanism the DC cascade cannot
//! (the 28 Apr 2025 Iberian blackout was overvoltage-driven — see
//! `DATA_SOURCES.md §5`).
//!
//! Per METHODS §8 "iterations, not a limiting fork": this adds new types
//! (`AcSystem`/`AcBus`/`AcLine`) and a new solver; the DC field tier
//! (`graph`/`flow`/`cascade`/`basin`) is untouched. Everything is in per-unit.
//!
//! **Honest scope (PROTOTYPE):** polar Newton–Raphson with a flat start; PV
//! buses hold `|V|` fixed but Q-limit → PV/PQ *switching is not modelled* (no
//! reactive limits), no transformer taps, no shunts beyond line charging. It
//! gives bus voltages + a convergence flag, and `collapse_margin` (the loading
//! at which Newton–Raphson can no longer solve = the voltage-collapse nose) —
//! the genuine voltage-stability quantity, not a full continuation power flow.

/// Minimal complex number (no external dep).
#[derive(Debug, Clone, Copy)]
struct Cx {
    re: f64,
    im: f64,
}
impl Cx {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn add(self, o: Cx) -> Cx {
        Cx::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: Cx) -> Cx {
        Cx::new(self.re - o.re, self.im - o.im)
    }
    /// Reciprocal `1/z`.
    fn recip(self) -> Cx {
        let d = self.re * self.re + self.im * self.im;
        Cx::new(self.re / d, -self.im / d)
    }
}

/// Bus role in the power-flow problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusKind {
    /// Reference: `|V|` and angle fixed, P and Q free.
    Slack,
    /// Generator: P and `|V|` fixed, Q free (no Q-limit switching here).
    Pv,
    /// Load: P and Q fixed, `|V|` and angle free.
    Pq,
}

/// A bus in per-unit. `p`/`q` are net injections (generation − load).
#[derive(Debug, Clone, Copy)]
pub struct AcBus {
    pub kind: BusKind,
    pub p: f64,
    pub q: f64,
    /// Set-point voltage magnitude for Slack/PV (ignored for PQ; start = 1.0).
    pub v_set: f64,
}

/// A line π-model in per-unit: series `r + jx`, total shunt susceptance `b`
/// (split `b/2` each end).
#[derive(Debug, Clone, Copy)]
pub struct AcLine {
    pub from: usize,
    pub to: usize,
    pub r: f64,
    pub x: f64,
    pub b_shunt: f64,
}

/// An AC network.
#[derive(Debug, Clone)]
pub struct AcSystem {
    pub buses: Vec<AcBus>,
    pub lines: Vec<AcLine>,
}

/// Power-flow solution.
#[derive(Debug, Clone)]
pub struct PowerFlowResult {
    pub converged: bool,
    pub iterations: usize,
    /// Voltage magnitudes (per-unit).
    pub vmag: Vec<f64>,
    /// Voltage angles (radians).
    pub vang: Vec<f64>,
    /// Final max power mismatch (per-unit).
    pub max_mismatch: f64,
}

impl AcSystem {
    pub fn new(buses: Vec<AcBus>, lines: Vec<AcLine>) -> Self {
        Self { buses, lines }
    }

    fn n(&self) -> usize {
        self.buses.len()
    }

    /// Build the nodal admittance matrix `Y = G + jB` (returned as two row-major
    /// `n×n` real matrices).
    pub fn ybus(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.n();
        let mut y = vec![Cx::new(0.0, 0.0); n * n];
        for l in &self.lines {
            let series = Cx::new(l.r, l.x).recip(); // 1/(r+jx)
            let sh = Cx::new(0.0, l.b_shunt / 2.0);
            let (f, t) = (l.from, l.to);
            y[f * n + f] = y[f * n + f].add(series).add(sh);
            y[t * n + t] = y[t * n + t].add(series).add(sh);
            y[f * n + t] = y[f * n + t].sub(series);
            y[t * n + f] = y[t * n + f].sub(series);
        }
        let g = y.iter().map(|c| c.re).collect();
        let b = y.iter().map(|c| c.im).collect();
        (g, b)
    }

    /// Calculated `(P, Q)` injections at every bus for the given voltages.
    fn injections(&self, g: &[f64], b: &[f64], vmag: &[f64], vang: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.n();
        let mut p = vec![0.0; n];
        let mut q = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                let ang = vang[i] - vang[j];
                let (gij, bij) = (g[i * n + j], b[i * n + j]);
                let vij = vmag[i] * vmag[j];
                p[i] += vij * (gij * ang.cos() + bij * ang.sin());
                q[i] += vij * (gij * ang.sin() - bij * ang.cos());
            }
        }
        (p, q)
    }

    /// Solve the power flow by polar Newton–Raphson. Returns `converged=false`
    /// if it does not reach `tol` within `max_iter` (a voltage-collapse signal).
    pub fn solve(&self, max_iter: usize, tol: f64) -> PowerFlowResult {
        let n = self.n();
        let (g, b) = self.ybus();

        let mut vmag: Vec<f64> = self
            .buses
            .iter()
            .map(|bus| match bus.kind {
                BusKind::Slack | BusKind::Pv => bus.v_set,
                BusKind::Pq => 1.0,
            })
            .collect();
        let mut vang = vec![0.0; n];

        // Unknown sets: angles for every non-slack bus; magnitudes for PQ buses.
        let pv_pq: Vec<usize> = (0..n)
            .filter(|&i| self.buses[i].kind != BusKind::Slack)
            .collect();
        let pq: Vec<usize> = (0..n)
            .filter(|&i| self.buses[i].kind == BusKind::Pq)
            .collect();
        let (na, nv) = (pv_pq.len(), pq.len());
        let m = na + nv;
        if m == 0 {
            return PowerFlowResult {
                converged: true,
                iterations: 0,
                vmag,
                vang,
                max_mismatch: 0.0,
            };
        }

        let mut iterations = 0;
        let mut max_mismatch = f64::INFINITY;
        for it in 0..max_iter {
            iterations = it + 1;
            let (pc, qc) = self.injections(&g, &b, &vmag, &vang);

            // Mismatch vector f = [ΔP (pv_pq), ΔQ (pq)].
            let mut f = vec![0.0; m];
            for (r, &i) in pv_pq.iter().enumerate() {
                f[r] = self.buses[i].p - pc[i];
            }
            for (r, &i) in pq.iter().enumerate() {
                f[na + r] = self.buses[i].q - qc[i];
            }
            max_mismatch = f.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
            if max_mismatch < tol {
                return PowerFlowResult {
                    converged: true,
                    iterations,
                    vmag,
                    vang,
                    max_mismatch,
                };
            }

            // Polar Jacobian J (m×m), row-major.
            let mut jac = vec![0.0; m * m];
            // ∂P rows
            for (r, &i) in pv_pq.iter().enumerate() {
                for (c, &j) in pv_pq.iter().enumerate() {
                    jac[r * m + c] = dp_dtheta(i, j, n, &g, &b, &vmag, &vang, pc[i], qc[i]);
                }
                for (c, &j) in pq.iter().enumerate() {
                    jac[r * m + (na + c)] = dp_dv(i, j, n, &g, &b, &vmag, &vang, pc[i]);
                }
            }
            // ∂Q rows
            for (r, &i) in pq.iter().enumerate() {
                for (c, &j) in pv_pq.iter().enumerate() {
                    jac[(na + r) * m + c] = dq_dtheta(i, j, n, &g, &b, &vmag, &vang, pc[i], qc[i]);
                }
                for (c, &j) in pq.iter().enumerate() {
                    jac[(na + r) * m + (na + c)] = dq_dv(i, j, n, &g, &b, &vmag, &vang, qc[i]);
                }
            }

            if !solve_linear(&mut jac, &mut f, m) {
                return PowerFlowResult {
                    converged: false,
                    iterations,
                    vmag,
                    vang,
                    max_mismatch,
                };
            }
            // f now holds Δx = [Δθ, ΔV].
            for (r, &i) in pv_pq.iter().enumerate() {
                vang[i] += f[r];
            }
            for (r, &i) in pq.iter().enumerate() {
                vmag[i] += f[na + r];
            }
        }
        PowerFlowResult {
            converged: false,
            iterations,
            vmag,
            vang,
            max_mismatch,
        }
    }

    /// Voltage-collapse loading margin: scale every PQ bus's `p` and `q` by `λ`
    /// (stepping from 1.0) and return the largest `λ` that still solves — the
    /// distance to the voltage-stability nose. `step` is the `λ` increment.
    pub fn collapse_margin(&self, step: f64, max_lambda: f64) -> f64 {
        let mut last_ok = 0.0;
        let mut lambda = 1.0;
        while lambda <= max_lambda {
            let scaled = AcSystem {
                buses: self
                    .buses
                    .iter()
                    .map(|bus| {
                        if bus.kind == BusKind::Pq {
                            AcBus {
                                p: bus.p * lambda,
                                q: bus.q * lambda,
                                ..*bus
                            }
                        } else {
                            *bus
                        }
                    })
                    .collect(),
                lines: self.lines.clone(),
            };
            if scaled.solve(50, 1e-8).converged {
                last_ok = lambda;
                lambda += step;
            } else {
                break;
            }
        }
        last_ok
    }
}

// ── Jacobian entries (standard polar form) ───────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn dp_dtheta(
    i: usize,
    j: usize,
    n: usize,
    g: &[f64],
    b: &[f64],
    v: &[f64],
    a: &[f64],
    _p: f64,
    q: f64,
) -> f64 {
    if i == j {
        -q - b[i * n + i] * v[i] * v[i]
    } else {
        let ang = a[i] - a[j];
        v[i] * v[j] * (g[i * n + j] * ang.sin() - b[i * n + j] * ang.cos())
    }
}
#[allow(clippy::too_many_arguments)]
fn dp_dv(i: usize, j: usize, n: usize, g: &[f64], b: &[f64], v: &[f64], a: &[f64], p: f64) -> f64 {
    if i == j {
        p / v[i] + g[i * n + i] * v[i]
    } else {
        let ang = a[i] - a[j];
        v[i] * (g[i * n + j] * ang.cos() + b[i * n + j] * ang.sin())
    }
}
#[allow(clippy::too_many_arguments)]
fn dq_dtheta(
    i: usize,
    j: usize,
    n: usize,
    g: &[f64],
    b: &[f64],
    v: &[f64],
    a: &[f64],
    p: f64,
    _q: f64,
) -> f64 {
    if i == j {
        p - g[i * n + i] * v[i] * v[i]
    } else {
        let ang = a[i] - a[j];
        -v[i] * v[j] * (g[i * n + j] * ang.cos() + b[i * n + j] * ang.sin())
    }
}
#[allow(clippy::too_many_arguments)]
fn dq_dv(i: usize, j: usize, n: usize, g: &[f64], b: &[f64], v: &[f64], a: &[f64], q: f64) -> f64 {
    if i == j {
        q / v[i] - b[i * n + i] * v[i]
    } else {
        let ang = a[i] - a[j];
        v[i] * (g[i * n + j] * ang.sin() - b[i * n + j] * ang.cos())
    }
}

/// Dense general linear solve `A x = b` (Gaussian elimination, partial pivot).
/// `b` is overwritten with `x`. Returns false if `A` is singular.
fn solve_linear(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-13 {
            return false;
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        for r in (col + 1)..n {
            let factor = a[r * n + col] / d;
            if factor != 0.0 {
                for c in col..n {
                    a[r * n + c] -= factor * a[col * n + c];
                }
                b[r] -= factor * b[col];
            }
        }
    }
    for col in (0..n).rev() {
        let mut s = b[col];
        for c in (col + 1)..n {
            s -= a[col * n + c] * b[c];
        }
        b[col] = s / a[col * n + col];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_bus(load_p: f64, load_q: f64) -> AcSystem {
        AcSystem::new(
            vec![
                AcBus {
                    kind: BusKind::Slack,
                    p: 0.0,
                    q: 0.0,
                    v_set: 1.0,
                },
                AcBus {
                    kind: BusKind::Pq,
                    p: load_p,
                    q: load_q,
                    v_set: 1.0,
                },
            ],
            vec![AcLine {
                from: 0,
                to: 1,
                r: 0.02,
                x: 0.1,
                b_shunt: 0.02,
            }],
        )
    }

    #[test]
    fn no_load_is_flat() {
        // No load AND no line charging (b=0) ⇒ no current ⇒ exactly flat.
        // (With charging, zero load still lifts V slightly — the Ferranti effect.)
        let sys = AcSystem::new(
            vec![
                AcBus { kind: BusKind::Slack, p: 0.0, q: 0.0, v_set: 1.0 },
                AcBus { kind: BusKind::Pq, p: 0.0, q: 0.0, v_set: 1.0 },
            ],
            vec![AcLine { from: 0, to: 1, r: 0.02, x: 0.1, b_shunt: 0.0 }],
        );
        let r = sys.solve(50, 1e-10);
        assert!(r.converged);
        assert!((r.vmag[1] - 1.0).abs() < 1e-6);
        assert!(r.vang[1].abs() < 1e-6);
    }

    #[test]
    fn solution_is_self_consistent() {
        // After NR converges, recomputed injections must match the specified
        // load at the PQ bus (the residual is what NR drove to zero).
        let sys = two_bus(-0.5, -0.2); // consuming 0.5 + j0.2 pu
        let r = sys.solve(50, 1e-10);
        assert!(r.converged, "NR should converge for a feasible load");
        let (g, b) = sys.ybus();
        let (pc, qc) = sys.injections(&g, &b, &r.vmag, &r.vang);
        assert!((pc[1] - (-0.5)).abs() < 1e-7, "P balance");
        assert!((qc[1] - (-0.2)).abs() < 1e-7, "Q balance");
        assert!(r.vmag[1] < 1.0, "load bus sags below slack");
    }

    #[test]
    fn ybus_is_symmetric() {
        let sys = two_bus(-0.5, -0.2);
        let (g, b) = sys.ybus();
        let n = sys.buses.len();
        for i in 0..n {
            for j in 0..n {
                assert!((g[i * n + j] - g[j * n + i]).abs() < 1e-12);
                assert!((b[i * n + j] - b[j * n + i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn heavy_load_fails_to_converge() {
        // Past the nose of the PV curve, Newton–Raphson cannot solve — the
        // voltage-collapse signal the DC model cannot produce.
        let sys = two_bus(-50.0, -20.0); // absurd load for x=0.1 pu line
        let r = sys.solve(50, 1e-8);
        assert!(!r.converged, "voltage collapse: NR must diverge");
    }

    #[test]
    fn collapse_margin_exceeds_base_then_is_finite() {
        let sys = two_bus(-0.5, -0.2);
        let margin = sys.collapse_margin(0.25, 100.0);
        assert!(margin >= 1.0, "base case is solvable");
        assert!(margin < 100.0, "there is a finite collapse nose");
    }
}
