//! Shuffle product on the truncated tensor algebra.
//!
//! The shuffle product ⧢ on words is the sum over all interleavings of
//! two words preserving relative order. On the tensor algebra it is the
//! algebra dual to concatenation — the operation that makes the signature
//! a *character* of the shuffle Hopf algebra:
//!
//! ```text
//!   〈S(X), u ⧢ v〉  =  〈S(X), u〉 · 〈S(X), v〉
//! ```
//!
//! This is the deep reason signatures are universal feature maps: products
//! of signature coordinates are themselves signature coordinates (under ⧢).
//! For VSA practitioners this is the algebraic peer to the bind operation —
//! but commutative and exact, where bind is associative-but-noisy.
//!
//! # Implementation scope
//!
//! Implemented end-to-end on multi-indices represented as Vec<usize>. This
//! is sufficient for testing the shuffle-product identity and for the
//! kernel-via-Goursat solver in `kernel.rs`. It is NOT optimized for
//! production hot-paths — production users should prefer the kernel form
//! (which never materializes the shuffled signature) or randomized
//! signatures (which approximate the shuffle in fixed dimension).
//!
//! # Citation
//!
//! Reutenauer, "Free Lie Algebras", Oxford 1993, Ch. 1.

/// Shuffle product u ⧢ v on multi-indices over an alphabet of size `dim`.
/// Returns a Vec of (multi-index, coefficient) pairs.
///
/// For the words u = (u₁,…,u_p) and v = (v₁,…,v_q) the shuffle is the sum
/// over all (p+q)!/(p! q!) interleavings. We represent each interleaving as
/// a binary mask with p ones (positions taken from u) and q zeros.
pub fn shuffle_product(u: &[usize], v: &[usize]) -> Vec<(Vec<usize>, f64)> {
    let p = u.len();
    let q = v.len();
    if p == 0 {
        return vec![(v.to_vec(), 1.0)];
    }
    if q == 0 {
        return vec![(u.to_vec(), 1.0)];
    }

    let total_len = p + q;
    let mut interleavings: Vec<Vec<usize>> = Vec::new();

    // Enumerate all binary masks of length total_len with exactly p ones.
    // bit=1 ⇒ next from u; bit=0 ⇒ next from v.
    for mask in 0u64..(1u64 << total_len) {
        if mask.count_ones() as usize != p {
            continue;
        }
        let mut word = Vec::with_capacity(total_len);
        let mut iu = 0usize;
        let mut iv = 0usize;
        for bit_pos in 0..total_len {
            // Iterate from low bit to high bit — defines reading order.
            if (mask >> bit_pos) & 1 == 1 {
                word.push(u[iu]);
                iu += 1;
            } else {
                word.push(v[iv]);
                iv += 1;
            }
        }
        interleavings.push(word);
    }

    // Group identical words and accumulate coefficients (multinomial-like).
    interleavings.sort();
    let mut out: Vec<(Vec<usize>, f64)> = Vec::new();
    for w in interleavings {
        if let Some(last) = out.last_mut() {
            if last.0 == w {
                last.1 += 1.0;
                continue;
            }
        }
        out.push((w, 1.0));
    }
    out
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn count_total(shuffle: &[(Vec<usize>, f64)]) -> f64 {
        shuffle.iter().map(|(_, c)| c).sum()
    }

    #[test]
    fn shuffle_with_empty_is_identity() {
        let u = vec![1usize, 2, 3];
        let s = shuffle_product(&u, &[]);
        assert_eq!(s, vec![(vec![1, 2, 3], 1.0)]);
    }

    #[test]
    fn shuffle_count_is_multinomial() {
        // |u ⧢ v| = (p+q)! / (p! q!)  when u and v have no shared letters.
        // u = [1,2], v = [3,4]: count = 4!/(2!2!) = 6
        let u = vec![1usize, 2];
        let v = vec![3, 4];
        let s = shuffle_product(&u, &v);
        assert_eq!(count_total(&s) as usize, 6);
    }

    #[test]
    fn shuffle_is_commutative_in_count() {
        // u ⧢ v and v ⧢ u contain the same multiset of words.
        let u = vec![1usize, 2, 3];
        let v = vec![4, 5];
        let mut s_uv = shuffle_product(&u, &v);
        let mut s_vu = shuffle_product(&v, &u);
        s_uv.sort();
        s_vu.sort();
        assert_eq!(s_uv, s_vu);
    }

    #[test]
    fn shuffle_aa() {
        // [a] ⧢ [a] = 2 · [aa]
        let s = shuffle_product(&[7usize], &[7]);
        assert_eq!(s, vec![(vec![7, 7], 2.0)]);
    }

    #[test]
    fn shuffle_ab_with_a() {
        // [a, b] ⧢ [a] yields three interleavings: (a,a,b), (a,a,b) [from
        // putting the second a between], (a,b,a). The first two differ in
        // which 'a' came from where but are textually identical, so the
        // grouped count for (a,a,b) is 2 and for (a,b,a) is 1.
        let s = shuffle_product(&[1usize, 2], &[1]);
        let mut counts = std::collections::BTreeMap::new();
        for (w, c) in &s {
            *counts.entry(w.clone()).or_insert(0.0) += c;
        }
        assert_eq!(counts.get(&vec![1usize, 1, 2]).copied().unwrap_or(0.0), 2.0);
        assert_eq!(counts.get(&vec![1usize, 2, 1]).copied().unwrap_or(0.0), 1.0);
    }
}
