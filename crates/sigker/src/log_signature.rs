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
//! where μ is the Möbius function. This buys real but bounded compression
//! (every row below is asserted by `witt_table_matches_measured_ratios`):
//!
//! ```text
//!   d=2, N=8:    full =      511  log-sig =      71   ratio =  7.2×
//!   d=2, N=12:   full =     8191  log-sig =     747   ratio = 11.0×
//!   d=4, N=8:    full =    87381  log-sig =   11464   ratio =  7.6×
//!   d=4, N=12:   full = 22369621  log-sig = 1924378   ratio = 11.6×
//! ```
//!
//! **The ratio is depth-dependent, and the low-depth end is far below the
//! "7–13×" headline.** At d=4 the measured ladder is:
//!
//! ```text
//!   N =    2     3     4     5     6     7     8    12
//!   ρ = 2.1×  2.8×  3.8×  4.6×  5.7×  6.6×  7.6× 11.6×
//! ```
//!
//! So 7–13× is the *asymptotic* regime (N ≥ 8), not what a depth-3 caller
//! gets. This crate's `lib.rs` summary was corrected to say so rather than
//! quote the ceiling as if it were the typical case.
//!
//! This is not a headline-grabbing 17,000× — log-signature compression is a
//! constant factor (roughly `d^(N+1) / ((d-1) · dim L_N(d))`) that grows
//! like O(N) for small d but stays modest for d=4. **For real depth-N
//! scaling at d=4, the production path is the Goursat-PDE signature kernel
//! (`kernel.rs`), which never materializes the signature at all.**
//!
//! That said, bounded compression with NO information loss is worth
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
//! 3. **Projects onto the Lyndon basis** by standard factorization + a
//!    triangular peel (see below).
//!
//! ## Step 3 in detail — why a flat read is NOT the projection
//!
//! Each Lyndon word `w` names a basis element `P_w` of the free Lie algebra,
//! obtained by *standard factorization*: `w = u·v` where `v` is the longest
//! proper suffix of `w` that is itself Lyndon; then `P_w = [P_u, P_v]`, with
//! `P_a = a` for single letters. Expanded in the tensor algebra,
//!
//! ```text
//!   P_w  =  w  +  Σ_{x > w, |x| = |w|} a_{w,x} · x
//! ```
//!
//! — leading term exactly `w`, everything else lexicographically *later*
//! (Lothaire, *Combinatorics on Words*, Thm 5.3.1). The change of basis is
//! therefore **unitriangular**, and reading `log_tensor[flat(w)]` gives
//! `c_w + Σ_{u < w} c_u · a_{u,w}` — the Lyndon coordinate plus contamination
//! from every earlier basis element. An earlier draft of this module did that
//! flat read and called the result "the Lyndon-basis coefficients"; it is not.
//!
//! The fix is the standard triangular peel: walk the Lyndon words of a degree
//! in increasing lex order, take `c_w` from the *residual* rather than the
//! original, then subtract `c_w · P_w` from the residual. Because the log of a
//! group-like element is a Lie element, the residual must end at **exactly
//! zero across every word of every degree — including the non-Lyndon ones**.
//! That is the falsifier this module leans on: it can fail (and did, on the
//! shortest-suffix convention), so `log_of_signature_is_a_lie_element` is a
//! real test rather than a restatement of the code.
//!
//! # Performance (measured from Witt formula)
//!
//! ```text
//!   d=2, N=8:    full = 511 doubles (4 KB)     log-sig = 71 doubles (568 B)
//!   d=4, N=8:    full = 87381 doubles (700 KB) log-sig = 11464 doubles (90 KB)
//!   d=4, N=12:   full = 22.4M doubles (179 MB) log-sig = 1.92M doubles (15 MB)
//! ```
//!
//! Compute cost is dominated by the depth-N Magnus expansion (still O(d^(2N))
//! intermediate). The win is in *storage* (the depth-dependent ratio tabulated
//! above) and in downstream operations on the log-sig representation directly.

use crate::signature::{pow_usize, signature_truncated, tensor_multiply, Signature};

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
        if n.is_multiple_of(p) {
            n /= p;
            if n.is_multiple_of(p) {
                return 0;
            }
            primes_seen += 1;
        }
        p += 1;
    }
    if n > 1 {
        primes_seen += 1;
    }
    if primes_seen % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Witt's formula: dim of the depth-k component of the free Lie algebra on
/// d letters. dim_witt(d, k) = (1/k) Σ_{j | k} μ(k/j) · d^j.
pub fn witt_component(d: usize, k: usize) -> usize {
    assert!(k >= 1);
    let mut sum: i64 = 0;
    let kk = k as u64;
    for j in 1..=kk {
        if kk.is_multiple_of(j) {
            let m = mobius(kk / j);
            sum += m * (d as i64).pow(j as u32);
        }
    }
    debug_assert!(sum >= 0 && (sum as u64).is_multiple_of(kk));
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
    assert!(alpha >= 1);
    // max_len == 0 falls straight through to the empty vec: the generation
    // loop below starts at w = [0] (length 1 > 0), so the length filter
    // never pushes it, and the successor step's `(0..max_len)` range is
    // itself empty, immediately driving `w` to empty and ending the loop.
    // No word of length >= 1 fits inside a max length of 0, and depth 0 is
    // a real, supported case: `signature_truncated` returns the identity
    // element there, whose logarithm has no Lie components at all — an
    // empty Lyndon basis, not an error.
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

/// Is `w` a Lyndon word — strictly smaller than every proper rotation?
///
/// O(n²); n ≤ the truncation depth, so this is never hot.
pub fn is_lyndon(w: &[usize]) -> bool {
    if w.is_empty() {
        return false;
    }
    (1..w.len()).all(|r| {
        let rotated: Vec<usize> = w[r..].iter().chain(w[..r].iter()).copied().collect();
        w < rotated.as_slice()
    })
}

/// Standard factorization split point: `w = w[..s] · w[s..]` where `w[s..]` is
/// the **longest** proper suffix of `w` that is itself a Lyndon word.
///
/// The longest-suffix convention is what makes the basis triangular; the
/// shortest-suffix variant produces a spanning set with a different (and, as
/// written here, wrong) leading-term structure. `bracket_leading_term_is_the_word`
/// pins the convention.
fn standard_factorization_split(w: &[usize]) -> usize {
    debug_assert!(w.len() >= 2);
    // Scanning s upward returns the longest Lyndon suffix first.
    (1..w.len())
        .find(|&s| is_lyndon(&w[s..]))
        .expect("a Lyndon word of length ≥ 2 always has a proper Lyndon suffix")
}

/// Expand the standard bracketing `P_w` into the tensor algebra, as
/// `(flat_index_within_degree, coefficient)` pairs sorted by index.
///
/// `P_a = a` for a letter; `P_w = [P_u, P_v] = P_u⊗P_v − P_v⊗P_u` otherwise.
/// Flat indexing matches `signature.rs`: word `(i₁…i_k)` ↦ `Σ iₘ·d^(k−m)`.
pub fn bracket_expansion(word: &[usize], dim: usize) -> Vec<(usize, f64)> {
    if word.len() == 1 {
        return vec![(word[0], 1.0)];
    }
    let s = standard_factorization_split(word);
    let (u, v) = word.split_at(s);
    let pu = bracket_expansion(u, dim);
    let pv = bracket_expansion(v, dim);
    let block_v = pow_usize(dim, v.len());
    let block_u = pow_usize(dim, u.len());

    let mut acc: Vec<(usize, f64)> = Vec::with_capacity(2 * pu.len() * pv.len());
    for &(a, ca) in &pu {
        for &(b, cb) in &pv {
            acc.push((a * block_v + b, ca * cb)); // + u⊗v
            acc.push((b * block_u + a, -ca * cb)); // − v⊗u
        }
    }
    acc.sort_by_key(|&(i, _)| i);
    // Coalesce duplicate indices, dropping exact cancellations.
    let mut out: Vec<(usize, f64)> = Vec::with_capacity(acc.len());
    for (i, c) in acc {
        match out.last_mut() {
            Some((j, acc_c)) if *j == i => *acc_c += c,
            _ => out.push((i, c)),
        }
    }
    out.retain(|&(_, c)| c != 0.0);
    out
}

/// Flat index of a word within its own degree.
fn flat_index(word: &[usize], dim: usize) -> usize {
    word.iter().fold(0usize, |acc, &l| acc * dim + l)
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
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub fn dot(&self, other: &Self) -> f64 {
        debug_assert_eq!(self.coeffs.len(), other.coeffs.len());
        self.coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn cosine(&self, other: &Self) -> f64 {
        let na = self.coeffs.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = other.coeffs.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 {
            return 0.0;
        }
        self.dot(other) / (na * nb)
    }

    /// Rebuild the full tensor-algebra logarithm `Σ_w c_w · P_w` from the
    /// Lyndon coordinates.
    ///
    /// This is the inverse of [`project_onto_lyndon_basis`], and comparing its
    /// output against [`tensor_log`] of the original signature is what makes
    /// "NO information loss" a measured claim instead of a doc-comment.
    pub fn reconstruct_tensor_log(&self) -> Signature {
        let mut out = zero_signature(self.path_dim, self.depth);
        for (word, &c) in self.basis.iter().zip(self.coeffs.iter()) {
            if c == 0.0 {
                continue;
            }
            let k = word.len();
            for (idx, weight) in bracket_expansion(word, self.path_dim) {
                out.levels[k][idx] += c * weight;
            }
        }
        out
    }

    /// Compression ratio: full-signature length / log-signature length.
    pub fn compression_vs_signature(&self) -> f64 {
        let d = self.path_dim;
        let n = self.depth;
        let full_len = if d == 1 {
            n + 1
        } else {
            (d.pow((n + 1) as u32) - 1) / (d - 1)
        };
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
    let d = sig.dim;
    let basis = enumerate_lyndon_words(d, depth);
    let (coeffs, _residual) = project_onto_lyndon_basis(&tensor_log(&sig), &basis);
    LogSignature {
        path_dim: d,
        depth,
        coeffs,
        basis,
    }
}

/// Project a Lie element of the truncated tensor algebra onto the Lyndon basis,
/// returning `(coefficients, residual)`.
///
/// The peel walks `basis` in its natural degree-then-lex order, reading each
/// coefficient off the running residual and subtracting that multiple of `P_w`.
/// Unitriangularity (see the module header) makes this exact.
///
/// **The residual is the falsifier.** For a genuine Lie element — which
/// `tensor_log` of a signature always is — it comes back zero to floating-point
/// noise across *every* word, Lyndon or not. A nonzero residual means the basis,
/// the bracketing convention, or the input is wrong. Callers that want the check
/// use [`max_abs`]; `log_signature_truncated` discards it on the hot path.
pub fn project_onto_lyndon_basis(
    log_tensor: &Signature,
    basis: &[Vec<usize>],
) -> (Vec<f64>, Signature) {
    let d = log_tensor.dim;
    let mut residual = log_tensor.clone();
    let mut coeffs = Vec::with_capacity(basis.len());
    for word in basis {
        let k = word.len();
        let c = residual.levels[k][flat_index(word, d)];
        coeffs.push(c);
        if c != 0.0 {
            for (idx, weight) in bracket_expansion(word, d) {
                residual.levels[k][idx] -= c * weight;
            }
        }
    }
    (coeffs, residual)
}

/// Largest absolute entry of a truncated tensor — used to assert a residual is
/// zero, or to compare two tensors entrywise.
pub fn max_abs(s: &Signature) -> f64 {
    s.levels
        .iter()
        .flat_map(|l| l.iter())
        .fold(0.0f64, |m, x| m.max(x.abs()))
}

/// log(1 + S₊) = S₊ − S₊²/2 + S₊³/3 − … on the truncated tensor algebra.
///
/// Public because reconstructing it from a [`LogSignature`] and comparing is
/// how losslessness is demonstrated rather than asserted.
pub fn tensor_log(s: &Signature) -> Signature {
    let d = s.dim;
    let depth = s.depth;
    let mut s_plus = s.clone();
    s_plus.levels[0][0] = 0.0;

    let mut result = zero_signature(d, depth);
    let mut power = s_plus.clone();
    let mut sign = 1.0f64;
    for k in 1..=depth {
        let coeff = sign / k as f64;
        for (out_level, pow_level) in result.levels.iter_mut().zip(power.levels.iter()) {
            for (o, p) in out_level.iter_mut().zip(pow_level.iter()) {
                *o += coeff * p;
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

// `tensor_multiply` and `pow_usize` are reused from `signature.rs` rather than
// duplicated here — see the `pub(crate)` markers there.

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
        assert!(
            ratio > 10.0 && ratio < 15.0,
            "compression {ratio:.2} expected ~11.6×"
        );
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
                    assert_eq!(
                        by_len[k - 1],
                        witt,
                        "Lyndon count for d={d}, k={k}: got {}, witt = {witt}",
                        by_len[k - 1]
                    );
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
        let expected: Vec<Vec<usize>> =
            vec![vec![0], vec![1], vec![0, 1], vec![0, 0, 1], vec![0, 1, 1]];
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
        assert!(
            max_abs < 1e-12,
            "log of constant-path signature should be 0, got max {max_abs}"
        );
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
        assert!(
            ratio > 7.0,
            "expected compression > 7× at d=2 N=8, got {ratio:.2}"
        );
    }

    // ── the Lyndon-basis machinery ──────────────────────────────────────────

    #[test]
    fn is_lyndon_discriminates() {
        // Must both accept and reject non-trivially, or it carries no
        // information (CLAUDE.md § can-it-fire / can-it-stay-silent).
        for w in [
            &[0usize][..],
            &[0, 1],
            &[0, 0, 1],
            &[0, 1, 1],
            &[0, 1, 0, 1, 1],
        ] {
            assert!(is_lyndon(w), "{w:?} is Lyndon");
        }
        for w in [
            &[1usize, 0][..],
            &[0, 1, 0, 1],
            &[1, 1],
            &[0, 1, 0],
            &[1, 0, 0],
        ] {
            assert!(!is_lyndon(w), "{w:?} is not Lyndon");
        }
    }

    #[test]
    fn standard_factorization_picks_longest_lyndon_suffix() {
        // w = 001 → suffixes "01" (Lyndon) and "1" (Lyndon); longest wins.
        assert_eq!(standard_factorization_split(&[0, 0, 1]), 1); // 0 · 01
                                                                 // w = 011 → "11" is not Lyndon, "1" is.
        assert_eq!(standard_factorization_split(&[0, 1, 1]), 2); // 01 · 1
    }

    #[test]
    fn bracket_expansion_matches_hand_computation() {
        // P_{01} = [0,1] = 01 − 10. Flat (d=2): 01→1, 10→2.
        let p01 = bracket_expansion(&[0, 1], 2);
        assert_eq!(p01, vec![(1, 1.0), (2, -1.0)]);

        // P_{001} = [0, [0,1]] = 0⊗(01−10) − (01−10)⊗0
        //         = 001 − 010 − 010 + 100 = 001 − 2·010 + 100.
        // Flat (d=2): 001→1, 010→2, 100→4.
        let p001 = bracket_expansion(&[0, 0, 1], 2);
        assert_eq!(p001, vec![(1, 1.0), (2, -2.0), (4, 1.0)]);
    }

    #[test]
    fn bracket_leading_term_is_the_word() {
        // Unitriangularity: P_w's lex-smallest word is w itself, coefficient 1.
        // This is the property the triangular peel depends on; it FAILS under
        // the shortest-Lyndon-suffix convention, so it pins the choice.
        for d in 2..=3 {
            for word in enumerate_lyndon_words(d, 5) {
                let exp = bracket_expansion(&word, d);
                let (lead_idx, lead_coeff) = exp[0]; // sorted by flat index = lex
                assert_eq!(
                    lead_idx,
                    flat_index(&word, d),
                    "leading term of P_{word:?} (d={d}) is not the word itself"
                );
                assert!(
                    (lead_coeff - 1.0).abs() < 1e-12,
                    "leading coeff {lead_coeff}"
                );
            }
        }
    }

    // ── losslessness ────────────────────────────────────────────────────────

    #[test]
    fn log_of_signature_is_a_lie_element() {
        // THE falsifier. After peeling every Lyndon coordinate, the residual
        // must vanish across ALL d^k words of every degree — not merely the
        // Lyndon-indexed ones. A wrong bracketing convention, a missing basis
        // element, or a non-Lie input all leave a nonzero residual here.
        let path = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 2.0, -1.0],
            vec![3.0, 1.0, 0.5],
            vec![-2.0, 4.0, 2.0],
        ];
        for depth in 1..=5 {
            let sig = signature_truncated(&path, depth);
            let basis = enumerate_lyndon_words(3, depth);
            let (_, residual) = project_onto_lyndon_basis(&tensor_log(&sig), &basis);
            let r = max_abs(&residual);
            let scale = max_abs(&tensor_log(&sig)).max(1.0);
            assert!(
                r < 1e-9 * scale,
                "depth {depth}: residual {r:.3e} (scale {scale:.3e}) — the tensor log \
                 did not lie in the span of the Lyndon basis"
            );
        }
    }

    #[test]
    fn log_signature_round_trips_to_the_tensor_log() {
        // Lossless in the operational sense: the compressed coordinates
        // reconstruct the full tensor logarithm entrywise.
        let path = vec![
            vec![0.0, 0.0],
            vec![1.5, -2.0],
            vec![0.25, 3.0],
            vec![-1.0, 1.0],
        ];
        for depth in 1..=6 {
            let log = log_signature_truncated(&path, depth);
            let rebuilt = log.reconstruct_tensor_log();
            let truth = tensor_log(&signature_truncated(&path, depth));

            let mut diff = rebuilt.clone();
            for (dl, tl) in diff.levels.iter_mut().zip(truth.levels.iter()) {
                for (a, b) in dl.iter_mut().zip(tl.iter()) {
                    *a -= b;
                }
            }
            let err = max_abs(&diff);
            let scale = max_abs(&truth).max(1.0);
            assert!(
                err < 1e-9 * scale,
                "depth {depth}: round-trip error {err:.3e} (scale {scale:.3e})"
            );
        }
    }

    #[test]
    fn round_trip_would_catch_a_corrupted_coefficient() {
        // Anti-vacuity for the test above: perturbing ONE coordinate must
        // break the round-trip. Otherwise the comparison proves nothing.
        let path = vec![vec![0.0, 0.0], vec![1.5, -2.0], vec![0.25, 3.0]];
        let truth = tensor_log(&signature_truncated(&path, 4));
        let mut log = log_signature_truncated(&path, 4);
        log.coeffs[3] += 0.5;

        let mut diff = log.reconstruct_tensor_log();
        for (dl, tl) in diff.levels.iter_mut().zip(truth.levels.iter()) {
            for (a, b) in dl.iter_mut().zip(tl.iter()) {
                *a -= b;
            }
        }
        assert!(
            max_abs(&diff) > 0.1,
            "a corrupted coefficient slipped through"
        );
    }

    #[test]
    fn coefficient_count_equals_witt_sum_over_degrees() {
        // The storage claim: one scalar per Lyndon word, i.e. Σ_k witt(d,k).
        for d in 2..=4 {
            for depth in 1..=4 {
                let path: Vec<Vec<f64>> = (0..5)
                    .map(|t| (0..d).map(|i| (t * (i + 1)) as f64 * 0.37).collect())
                    .collect();
                let log = log_signature_truncated(&path, depth);
                let expected: usize = (1..=depth).map(|k| witt_component(d, k)).sum();
                assert_eq!(log.coeffs.len(), expected, "d={d} depth={depth}");
                assert_eq!(log.coeffs.len(), witt_dimension(d, depth));
            }
        }
    }

    #[test]
    fn witt_table_matches_measured_ratios() {
        // Pins every number quoted in the module header, including the two
        // that were WRONG before this module was ever compiled:
        //   d=4,N=8 was documented as 11164 (真 11464, ratio 7.6× not 7.8×)
        //   d=2,N=12 was documented as 632 / 13× (真 747 / 11.0×)
        assert_eq!(witt_dimension(2, 8), 71);
        assert_eq!(witt_dimension(2, 12), 747);
        assert_eq!(witt_dimension(4, 8), 11_464);
        assert_eq!(witt_dimension(4, 12), 1_924_378);

        let ratio = |d: usize, n: usize| {
            let full = (d.pow((n + 1) as u32) - 1) / (d - 1);
            full as f64 / witt_dimension(d, n) as f64
        };
        assert!((ratio(2, 8) - 7.20).abs() < 0.01);
        assert!((ratio(2, 12) - 10.97).abs() < 0.01);
        assert!((ratio(4, 8) - 7.62).abs() < 0.01);
        assert!((ratio(4, 12) - 11.62).abs() < 0.01);
    }

    #[test]
    fn compression_at_shallow_depth_is_far_below_the_headline() {
        // The honest correction: "7–13×" is asymptotic. At the depths
        // `depth_scaling.rs` actually sweeps (d=4, N=2..8) the low end is ~2×.
        // If someone "fixes" the docs back to a flat 7–13× claim, this fails.
        let path: Vec<Vec<f64>> = (0..8)
            .map(|t| (0..4).map(|i| (t as f64) * 0.3 + i as f64).collect())
            .collect();
        let shallow = log_signature_truncated(&path, 2).compression_vs_signature();
        assert!(
            shallow < 3.0,
            "d=4 depth=2 compression is {shallow:.2}×, not in the 7–13× band"
        );
        let deep = log_signature_truncated(&path, 6).compression_vs_signature();
        assert!(deep > shallow, "compression must grow with depth");
    }

    /// `signature_truncated` explicitly supports `depth == 0` (it returns the
    /// identity element — see its own early return). `log_signature_truncated`
    /// must stay consistent with that: the identity's logarithm has no Lie
    /// components, so this is a valid, empty log-signature, not an error.
    /// `enumerate_lyndon_words` used to assert `max_len >= 1` and panic here.
    #[test]
    fn depth_zero_is_the_empty_log_signature_not_a_panic() {
        let path = vec![vec![1.0, 2.0], vec![3.0, -1.0]];
        let log_sig = log_signature_truncated(&path, 0);
        assert_eq!(log_sig.depth, 0);
        assert!(log_sig.basis.is_empty());
        assert!(log_sig.coeffs.is_empty());
        assert_eq!(witt_dimension(log_sig.path_dim, 0), 0);
        assert_eq!(enumerate_lyndon_words(log_sig.path_dim, 0).len(), 0);
    }
}
