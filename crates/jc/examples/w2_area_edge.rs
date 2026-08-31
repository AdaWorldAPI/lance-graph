//! W2 pre-registration, part 2: where is the depth-infinity leg's real edge?
//!
//! The converse leg asks "is a non-tree loop distinguished from a tree-like
//! one?" Its hardest cases are NEAR-DEGENERATE triangles — a triangle whose
//! enclosed area tends to zero tends to a tree-like path, so its deviation
//! must tend to zero too. That is not a defect; it is the theorem's own
//! boundary meeting a finite grid.
//!
//! Sampling random triangles and taking the min is the wrong instrument: the
//! measured min moved from 1.53e-2 (25 pairs) to 1.03e-1 (12 pairs) purely
//! because fewer draws find fewer degenerate cases — a gate that gets EASIER
//! with a smaller sample. So the leg gates a CONTROLLED family instead, and
//! this sweep is where the controlled parameter's meaning is measured.
//!
//! Family: `[p0, p1, p2(h), p0]` with `p2(h) = midpoint(p0,p1) + h·n`, `n` a
//! unit normal. Enclosed area is `‖p1−p0‖·h/2`, so `h` tunes degeneracy
//! directly. h = 0 IS tree-like (out-and-back along the same line).

use sigker::signature_kernel_pde;

fn resample(corners: &[Vec<f64>], per_seg: usize) -> Vec<Vec<f64>> {
    let dim = corners[0].len();
    let mut out = vec![corners[0].clone()];
    for w in corners.windows(2) {
        for s in 1..=per_seg {
            let t = s as f64 / per_seg as f64;
            out.push(
                (0..dim)
                    .map(|a| w[0][a] + t * (w[1][a] - w[0][a]))
                    .collect(),
            );
        }
    }
    out
}
fn dev(path: &[Vec<f64>]) -> f64 {
    let k = signature_kernel_pde(path, path);
    (1.0 / k.sqrt() - 1.0).abs()
}

fn main() {
    const PER_SEG: usize = 1536;
    // A fixed unit-scale base segment and an orthogonal normal in R^3.
    let p0 = vec![0.0, 0.0, 0.0];
    let p1 = vec![1.0, 0.0, 0.0];
    let n = [0.0, 1.0, 0.0];

    println!(
        "{:>10} {:>12} {:>14} {:>12}",
        "h", "area", "deviation", "dev/area^2"
    );
    for &h in &[1.0f64, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0] {
        let p2 = vec![0.5 + h * n[0], h * n[1], h * n[2]];
        let tri = resample(&[p0.clone(), p1.clone(), p2, p0.clone()], PER_SEG);
        let d = dev(&tri);
        let area = 1.0 * h / 2.0;
        let q = if area > 0.0 {
            d / (area * area)
        } else {
            f64::NAN
        };
        println!("{h:>10.4} {area:>12.5} {d:>14.6e} {q:>12.4}");
    }
    // The tree-like reference at the same resolution: the artifact floor.
    let oab = resample(&[p0.clone(), p1.clone(), p0.clone()], PER_SEG);
    println!(
        "\nout-and-back (tree-like, h=0 exactly): deviation {:.6e}  <- artifact floor",
        dev(&oab)
    );
}
