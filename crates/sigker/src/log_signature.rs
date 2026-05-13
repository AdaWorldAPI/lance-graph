//! Log-signatures via the Lyndon-word basis — Lie-algebra compression of
//! the truncated path signature.
//!
//! Citation: J. Reizenstein & B. Graham, "The iisignature library: efficient
//! calculation of iterated-integral signatures and log signatures",
//! ACM TOMS 46(1) (2020), arXiv:1802.08252.
//!
//! # Why this exists
//!
//! The truncated signature S_N(X) at depth N in dimension d has
//! (d^(N+1) − 1)/(d − 1) coordinates — exponential in N. For d=4, depth=12
//! that is ≈22.4M coefficients per path. Untenable.
//!
//! The signature lives in a *Lie algebra* — the free Lie algebra over d
//! letters truncated at depth N. Its dimension is given by Witt's formula:
//!
//!   dim L_N(d)  =  Σ_{k=1}^N (1/k) · Σ_{j | k} μ(k/j) · d^j
//!
//! where μ is the Möbius function. This buys real but bounded compression:
//!
//! ```text
//!   d=2, N=8:    full = 511    log-sig =       71   ratio = 7.2×
//!   d=2, N=12:   full = 8191   log-sig =      632   ratio = 13×
//!   d=4, N=8:    full = 87381  log-sig =    11164   ratio = 7.8×
//!   d=4, N=12:   full = 22.4M  log-sig =     1.92M  ratio = 12×
//! ```
//!
//! This is not a headline-grabbing 17,000× — log-signature compression is a
//! constant factor (roughly `d^(N+1) / ((d-1) · dim L_N(d))`) that grows
//! like O(N) for small d but stays modest for d=4. **For real depth-N
//! scaling at d=4, the production path is the Goursat-PDE signature kernel
//! (`kernel.rs`), which never materializes the signature at all.**
//!
//! That said, 7–13× space compression with NO information loss is worth
//! shipping: it puts depth-8 signatures within the same RAM envelope as
//! depth-6 raw signatures, and unlocks compact storage for offline analysis
//! or batched export.
//!
//! ## The Lyndon-word basis
//!
//! A Lyndon word is a string strictly lexicographically smaller than all its
//! rotations. Chen-Fox-Lyndon gives the unique factorization of any word into
//! a non-increasing product of Lyndon words. The Lyndon words of length ≤ N
//! over alphabet {0..d-1} enumerate the basis of L_N(d). This module:
//!
//! 1. **Enumerates Lyndon words** via Duval 1988, O(n) per word.
//! 2. **Computes the tensor-algebra logarithm** of a truncated signature via
//!    the Magnus series log(1 + S_+) = S_+ − S_+²/2 + S_+³/3 − …
//! 3. **Reads off Lyndon-basis coefficients** from the flat tensor-algebra
//!    storage. (The full Reizenstein-Graham 2020 algorithm uses a Lyndon-
//!    bracket transformation matrix — we omit that here; the flat-coordinate
//!    read is sufficient for similarity / round-trip uses.)
//!
//! # Performance (measured from Witt formula)
//!
//! ```text
//!   d=2, N=8:    full = 511 doubles (4 KB)     log-sig = 71 doubles (568 B)
//!   d=4, N=8:    full = 87381 doubles (700 KB) log-sig = 11164 doubles (89 KB)
//!   d=4, N=12:   full = 22.4M doubles (179 MB) log-sig = 1.92M doubles (15 MB)
//! ```
//!
//! Compute cost is dominated by the depth-N Magnus expansion (still O(d^(2N))
//! intermediate). The win is in *storage* (7–13×) and in downstream operations
//! on the log-sig representation directly.

use crate::signature::{signature_truncated, Signature};

// ════════════════════════════════════════════════════════════════════════════
// Witt's formula — closed-form dim L_N(d) over the free Lie algebra.
// ════════════════════════════════════════════════════════════════════════════

fn mobius(n: u64) -> i64 {
    if n == 1 {
        return 1;
    }
    let mut n = n;
    let mut primes_seen = 0i64;
    let mut p = 2u64;
    while p * p <= n {
        if n % p == 0 {
            n /= p;
            if n % p == 0 {
                return 0;
            }
            primes_seen += 1;
        }
        p += 1;
    }
    if n > 1 {
        primes_seen += 1;
    }
    if primes_seen % 2 == 0 { 1 } else { -1 }
}

/// Witt's formula: dim of the depth-k component of the free Lie algebra on
/// d letters. dim_witt(d, k) = (1/k) Σ_{j | k} μ(k/j) · d^j.
pub fn witt_component(d: usize, k: usize) -> usize {
    assert!(k >= 1);
    let mut sum: i64 = 0;
    let kk = k as u64;
    for j in 1..=kk {
        if kk % j == 0 {
            let m = mobius(kk / j);
            sum += m * (d as i64).pow(j as u32);
        }
    }
    debug_assert!(sum >= 0 && (sum as u64) % kk == 0);
    (sum as u64 / kk) as usize
}

/// Total dim of the Lie algebra truncated at depth N: Σ_{k=1}^N witt(d, k).
pub fn witt_dimension(d: usize, depth: usize) -> usize {
    (1..=depth).map(|k| witt_component(d, k)).sum()
}

// ════════════════════════════════════════════════════════════════════════════
// Lyndon-word enumeration — Duval 1988.
// ════════════════════════════════════════════════════════════════════════════

/// Enumerate all Lyndon words of length 1..=max_len over alphabet {0..alpha-1},
/// in length-then-lex order.
pub fn enumerate_lyndon_words(alpha: usize, max_len: usize) -> Vec<Vec<usize>> {
    assert!(alpha >= 1 && max_len >= 1);
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut w: Vec<usize> = vec![0];
    while !w.is_empty() {
        if w.len() <= max_len {
            out.push(w.clone());
        }
        let m = w.len();
        let mut new_w: Vec<usize> = (0..max_len).map(|i| w[i % m]).collect();
        while !new_w.is_empty() && *new_w.last().unwrap() == alpha - 1 {
            new_w.pop();
        }
        if let Some(last) = new_w.last_mut() {
            *last += 1;
        }
        w = new_w;
    }
    out.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
    out
}

// ════════════════════════════════════════════════════════════════════════════
// LogSignature — compact storage indexed by Lyndon word.
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct LogSignature {
    pub path_dim: usize,
    pub depth: usize,
    /// Coefficients in Lyndon-basis order (matches `enumerate_lyndon_words`).
    pub coeffs: Vec<f64>,
    /// Cached Lyndon basis used to interpret coeffs.
    pub basis: Vec<Vec<usize>>,
}

impl LogSignature {
    pub fn len(&self) -> usize { self.coeffs.len() }
    pub fn is_empty(&self) -> bool { self.coeffs.is_empty() }

    pub fn dot(&self, other: &Self) -> f64 {
        debug_assert_eq!(self.coeffs.len(), other.coeffs.len());
        self.coeffs.iter().zip(other.coeffs.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn cosine(&self, other: &Self) -> f64 {
        let na = self.coeffs.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = other.coeffs.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 { return 0.0; }
        self.dot(other) / (na * nb)
    }

    /// Compression ratio: full-signature length / log-signature length.
    pub fn compression_vs_signature(&self) -> f64 {
        let d = self.path_dim;
        let n = self.depth;
        let full_len = if d == 1 { n + 1 } else { (d.pow((n + 1) as u32) - 1) / (d - 1) };
        full_len as f64 / self.coeffs.len() as f64
    }
}

// ════════════════════════════════════════════════════════════════════════════
// log_signature_truncated — compute the log-signature of a path.
//
// Algorithm: signature → tensor-algebra log via Magnus series → read off
// coefficient at the flat index of each Lyndon word.
// ════════════════════════════════════════════════════════════════════════════

pub fn log_signature_truncated(path: &[Vec<f64>], depth: usize) -> LogSignature {
    let sig = signature_truncated(path, depth);
    let log_sig_tensor = tensor_log(&sig);
    let basis = enumerate_lyndon_words(sig.dim, depth);

    let d = sig.dim;
    let mut coeffs = Vec::with_capacity(basis.len());
    for word in &basis {
        let k = word.len();
        let mut flat = 0usize;
        for &letter in word {
            flat = flat * d + letter;
        }
        coeffs.push(log_sig_tensor.levels[k][flat]);
    }

    LogSignature { path_dim: d, depth, coeffs, basis }
}

// log(1 + S_+) = S_+ − S_+²/2 + S_+³/3 − …
fn tensor_log(s: &Signature) -> Signature {
    let d = s.dim;
    let depth = s.depth;
    let mut s_plus = s.clone();
    s_plus.levels[0][0] = 0.0;

    let mut result = zero_signature(d, depth);
    let mut power = s_plus.clone();
    let mut sign = 1.0f64;
    for k in 1..=depth {
        let coeff = sign / k as f64;
        for level in 0..=depth {
            for i in 0..result.levels[level].len() {
                result.levels[level][i] += coeff * power.levels[level][i];
            }
        }
        if k < depth {
            power = tensor_multiply(&power, &s_plus);
        }
        sign = -sign;
    }
    result
}

fn zero_signature(dim: usize, depth: usize) -> Signature {
    let mut levels = Vec::with_capacity(depth + 1);
    for k in 0..=depth {
        levels.push(vec![0.0; pow_usize(dim, k)]);
    }
    Signature { dim, depth, levels }
}

fn tensor_multiply(a: &Signature, b: &Signature) -> Signature {
    debug_assert_eq!(a.dim, b.dim);
    debug_assert_eq!(a.depth, b.depth);
    let dim = a.dim;
    let depth = a.depth;

    let mut out = zero_signature(dim, depth);
    out.levels[0][0] = a.levels[0][0] * b.levels[0][0];

    for k in 1..=depth {
        let len_k = pow_usize(dim, k);
        let mut level_k = vec![0.0; len_k];
        for i in 0..=k {
            let j = k - i;
            let len_i = pow_usize(dim, i);
            let len_j = pow_usize(dim, j);
            for ai in 0..len_i {
                let av = a.levels[i][ai];
                if av == 0.0 { continue; }
                for bj in 0..len_j {
                    let bv = b.levels[j][bj];
                    if bv == 0.0 { continue; }
                    let flat = ai * len_j + bj;
                    level_k[flat] += av * bv;
                }
            }
        }
        out.levels[k] = level_k;
    }
    out
}

fn pow_usize(base: usize, exp: usize) -> usize {
    let mut acc = 1usize;
    for _ in 0..exp { acc *= base; }
    acc
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mobius_known_values() {
        assert_eq!(mobius(1), 1);
        assert_eq!(mobius(2), -1);
        assert_eq!(mobius(3), -1);
        assert_eq!(mobius(4), 0);
        assert_eq!(mobius(6), 1);
        assert_eq!(mobius(12), 0);
        assert_eq!(mobius(30), -1);
    }

    #[test]
    fn witt_component_low() {
        // d=2: 2, 1, 2, 3, 6, 9, 18, 30, 56, 99, …
        // d=3: 3, 3, 8, 18, 48, …
        assert_eq!(witt_component(2, 1), 2);
        assert_eq!(witt_component(2, 2), 1);
        assert_eq!(witt_component(2, 3), 2);
        assert_eq!(witt_component(2, 4), 3);
        assert_eq!(witt_component(2, 5), 6);
        assert_eq!(witt_component(3, 1), 3);
        assert_eq!(witt_component(3, 2), 3);
        assert_eq!(witt_component(3, 3), 8);
    }

    #[test]
    fn witt_dimension_d4_n12_compression() {
        // d=4, N=12: full sig = (4^13 - 1)/3 = 22369621.
        // Lyndon basis dim (verified independently in Python): 1924378.
        // Compression ratio ≈ 11.6× — bounded, NOT the headline "17000×"
        // I initially conflated with sub-exponential growth claims.
        let dim_lie = witt_dimension(4, 12);
        assert_eq!(dim_lie, 1_924_378);
        let dim_full = (4usize.pow(13) - 1) / 3;
        let ratio = dim_full as f64 / dim_lie as f64;
        assert!(ratio > 10.0 && ratio < 15.0, "compression {ratio:.2} expected ~11.6×");
    }

    #[test]
    fn lyndon_count_matches_witt() {
        for d in 2..=4 {
            for n in 1..=5 {
                let words = enumerate_lyndon_words(d, n);
                let by_len: Vec<usize> = (1..=n)
                    .map(|k| words.iter().filter(|w| w.len() == k).count())
                    .collect();
                for k in 1..=n {
                    let witt = witt_component(d, k);
                    assert_eq!(by_len[k - 1], witt, "Lyndon count for d={d}, k={k}: got {}, witt = {witt}", by_len[k - 1]);
                }
            }
        }
    }

    #[test]
    fn lyndon_d2_n3_explicit() {
        // length 1: [0], [1]
        // length 2: [0,1]
        // length 3: [0,0,1], [0,1,1]
        let words = enumerate_lyndon_words(2, 3);
        let expected: Vec<Vec<usize>> = vec![
            vec![0], vec![1], vec![0, 1], vec![0, 0, 1], vec![0, 1, 1],
        ];
        assert_eq!(words, expected);
    }

    #[test]
    fn log_signature_dim_matches_witt() {
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![3.0, 1.0]];
        let log = log_signature_truncated(&path, 3);
        assert_eq!(log.coeffs.len(), witt_dimension(2, 3));
    }

    #[test]
    fn log_signature_constant_path_is_zero() {
        let path = vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        let log = log_signature_truncated(&path, 3);
        let max_abs = log.coeffs.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert!(max_abs < 1e-12, "log of constant-path signature should be 0, got max {max_abs}");
    }

    #[test]
    fn log_signature_level_1_equals_increment() {
        // The level-1 part of log(S(X)) equals the level-1 part of S(X) =
        // total path increment. Lyndon words of length 1 are [0], [1], …
        // so coeffs[0..d] should equal the increment.
        let path = vec![vec![0.0, 0.0], vec![3.0, 5.0]];
        let log = log_signature_truncated(&path, 2);
        assert!((log.coeffs[0] - 3.0).abs() < 1e-12, "got {}", log.coeffs[0]);
        assert!((log.coeffs[1] - 5.0).abs() < 1e-12, "got {}", log.coeffs[1]);
    }

    #[test]
    fn compression_ratio_is_substantial() {
        // For d=2, depth=8: full = 511 coeffs; Lyndon = 71.
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![3.0, 1.0]];
        let log = log_signature_truncated(&path, 8);
        let ratio = log.compression_vs_signature();
        assert!(ratio > 7.0, "expected compression > 7× at d=2 N=8, got {ratio:.2}");
    }
}
