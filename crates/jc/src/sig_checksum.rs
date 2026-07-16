//! PROBE-SIG-CHECKSUM — depth-2 truncated signature as a replayable
//! trajectory digest (grades E-WH-TWO-SIDES-SIG-CHECKSUM-1 leg 2).
//!
//! Citation: B. Hambly & T. Lyons, "Uniqueness for the signature of a path
//! of bounded variation and the reduced path group", Annals of Mathematics,
//! Vol. 171, No. 1 (2010), 109-167. See `hambly_lyons.rs` (Pillar 11) for
//! the theorem statement this probe reuses operationally.
//!
//! # What this probe checks
//!
//! Hambly-Lyons Theorem 4 says the signature is a lossless digest *modulo
//! tree-like equivalence*: `S(X) = S(Y) ⟺ X ~ Y` (tree-quotient). This
//! probe asks whether that theorem behaves the way a "trajectory checksum"
//! needs it to behave in practice, on a single realistic path rather than
//! on synthetic out-and-back / triangle pairs in isolation (as Pillar 11
//! does):
//!
//! 1. **Replay identity** — hashing (signing) the same trajectory twice
//!    must give the same digest. Trivial in principle, but worth pinning
//!    down because `signature_truncated` accumulates via `tensor_multiply`
//!    (see `sigker::signature::tensor_multiply`), and floating-point
//!    accumulation order must be deterministic across two independent
//!    builds of an identical path for the digest to be replayable at all.
//! 2. **Tree-like edit invisibility** — inserting an out-and-back
//!    excursion (`A → B → A`) *in the middle* of an otherwise-normal path
//!    must leave the digest unchanged (within float tolerance). This is
//!    the operationally interesting case: Pillar 11 tests out-and-back as
//!    the *entire* path; this probe tests it as a *sub-path* spliced into
//!    a longer trajectory, relying on `tensor_multiply`'s associativity
//!    (Chen's identity) for the excursion's near-identity contribution to
//!    the middle of the product to vanish rather than to propagate.
//! 3. **Non-tree edit is caught** — permanently displacing one interior
//!    point of the path (a kink that does *not* backtrack — no A→B→A
//!    structure) changes the digest measurably. Level-1 (total
//!    displacement, endpoint − startpoint) is invariant under an interior
//!    displacement, so this specifically tests that level-2 (signed-area /
//!    curvature information) is doing the discriminating work.
//!
//! # Probe design
//!
//! - **Home:** `crates/jc/src/sig_checksum.rs`, gated by the same
//!   `hambly-lyons` feature as Pillar 11 (`sigker` workspace sibling).
//!   This module is registered in `lib.rs` but is **not** added to the
//!   pillar `prove()` registry — it is a probe, not a 12th pillar.
//! - **Path:** a single deterministic 2-D "trajectory" of `PATH_LEN = 64`
//!   points, built via a seeded SplitMix64 integer walk (steps in
//!   `{-1, 0, 1}²` per axis, cumulatively summed) mapped to `f64`
//!   coordinates — mirrors `hambly_lyons.rs`'s `splitmix64`/`rand_in`
//!   generator style, but accumulates a walk instead of drawing
//!   independent points.
//! - **Signature:** `sigker::signature_truncated` at `DEPTH = 2` in
//!   `DIM = 2` (the tensor-algebra path, not `signature_kernel_pde` —
//!   same rationale as Pillar 11: this probe is independent of the
//!   PR #350 PDE-form correction).
//! - **Distance:** reuses the Frobenius-across-all-levels
//!   `signature_distance` helper from `hambly_lyons.rs` (duplicated here
//!   since the original is private to that module).
//! - **Thresholds:** reused verbatim from `hambly_lyons.rs`'s own
//!   forward/converse legs — `FORWARD_TOLERANCE = 1e-9` (the zero-edit
//!   threshold) and `CONVERSE_THRESHOLD = 0.05` (the caught-edit
//!   threshold) — so this probe is graded on the exact same numeric bar
//!   Pillar 11 already certifies, not a bar invented for this probe.
//!
//! # Pass / kill
//!
//! - **PASS:** all three tests green (replay-identical, tree-edit
//!   invisible under `FORWARD_TOLERANCE`, non-tree edit caught above
//!   `CONVERSE_THRESHOLD`).
//! - **KILL:** test (b) or (c) fails at these thresholds — depth-2
//!   truncation is not enough for the checksum idea as stated (would need
//!   depth > 2, or the digest framing is wrong).

#[cfg(feature = "hambly-lyons")]
mod active {
    use sigker::signature::Signature;
    use sigker::signature_truncated;

    /// Path dimension (a 2-D trajectory).
    const DIM: usize = 2;
    /// Truncation depth — matches Pillar 11's `hambly_lyons::DEPTH`.
    const DEPTH: usize = 2;
    /// Number of points in the deterministic trajectory.
    const PATH_LEN: usize = 64;

    /// Reused verbatim from `hambly_lyons.rs`'s forward-leg tolerance —
    /// the zero-threshold below which two signatures are "the same digest".
    const FORWARD_TOLERANCE: f64 = 1e-9;
    /// Reused verbatim from `hambly_lyons.rs`'s converse-leg threshold —
    /// the bar a non-tree edit's digest distance must clear to be "caught".
    const CONVERSE_THRESHOLD: f64 = 0.05;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Frobenius distance across all signature levels. Duplicated from
    /// `hambly_lyons.rs::active::signature_distance` (private there) —
    /// identical implementation, kept local per the plan's edit-only
    /// constraint (do not modify `hambly_lyons.rs`).
    fn signature_distance(a: &Signature, b: &Signature) -> f64 {
        assert_eq!(a.dim, b.dim);
        assert_eq!(a.depth, b.depth);
        let mut acc = 0.0_f64;
        for (la, lb) in a.levels.iter().zip(b.levels.iter()) {
            for (xa, xb) in la.iter().zip(lb.iter()) {
                let d = xa - xb;
                acc += d * d;
            }
        }
        acc.sqrt()
    }

    /// Build the deterministic `PATH_LEN`-point 2-D trajectory: a seeded
    /// SplitMix64 integer walk (each axis steps by -1, 0, or +1 per point),
    /// cumulatively summed into `f64` coordinates. Same seed always
    /// reproduces the same path — that determinism is exactly what test
    /// (a) below is checking end-to-end.
    fn build_path(seed: u64) -> Vec<Vec<f64>> {
        let mut state = seed;
        let mut x = 0.0_f64;
        let mut y = 0.0_f64;
        let mut path = Vec::with_capacity(PATH_LEN);
        path.push(vec![x, y]);
        for _ in 1..PATH_LEN {
            let r = splitmix64(&mut state);
            let dx = ((r % 3) as i64 - 1) as f64;
            let dy = (((r >> 2) % 3) as i64 - 1) as f64;
            x += dx;
            y += dy;
            path.push(vec![x, y]);
        }
        path
    }

    /// Fixed seed for a reproducible probe trajectory.
    const PATH_SEED: u64 = 0xD161_1D3D_0A7A_7EC7u64;

    /// Insert an `A → B → A` excursion mid-path at index `i`: splice
    /// `[B, path[i]]` right after `path[i]`. `B` is an arbitrary point
    /// offset from `path[i]` — the excursion goes out to `B` and
    /// immediately returns, which is the canonical tree-like-equivalence
    /// generator (Hambly-Lyons: identifying a sub-path with its
    /// concatenated reverse collapses it to its start point).
    fn insert_tree_like_excursion(path: &[Vec<f64>], i: usize) -> Vec<Vec<f64>> {
        let anchor = path[i].clone();
        let b = vec![anchor[0] + 5.0, anchor[1] + 3.0];
        let mut edited = Vec::with_capacity(path.len() + 2);
        edited.extend_from_slice(&path[..=i]);
        edited.push(b);
        edited.push(anchor);
        edited.extend_from_slice(&path[i + 1..]);
        edited
    }

    /// Find the first interior index `i` (starting at `from`, scanning
    /// forward) whose neighbors `path[i-1]`, `path[i+1]` are not
    /// coincident — i.e. `path[i+1] - path[i-1] != (0, 0)`.
    ///
    /// Why this matters: at `DEPTH = 2` truncation, displacing only
    /// `path[i]` (endpoints of the whole trajectory untouched) changes the
    /// *total* signature by *exactly* the antisymmetric (signed-area) term
    /// `½·(off ⊗ combined − combined ⊗ off)` where
    /// `combined = path[i+1] − path[i-1]` — this is an algebraic identity
    /// of `tensor_multiply`'s associativity/bilinearity at this truncation
    /// depth, not an approximation, and it holds independent of where `i`
    /// sits in the path (the prefix/suffix segments contribute nothing at
    /// depth 2, since the edit's own level-0/level-1 contributions are
    /// exactly zero). That signed-area term is zero whenever `off` is
    /// parallel to `combined` — which includes the degenerate case
    /// `combined == (0, 0)`, where *no* offset produces a visible edit.
    /// Scanning for a non-degenerate `combined` makes the probe robust to
    /// which interior point of the fixed seeded path is chosen.
    fn find_displaceable_index(path: &[Vec<f64>], from: usize) -> usize {
        let n = path.len();
        for i in from..n - 1 {
            let dx = path[i + 1][0] - path[i - 1][0];
            let dy = path[i + 1][1] - path[i - 1][1];
            if dx != 0.0 || dy != 0.0 {
                return i;
            }
        }
        panic!("no displaceable interior index found in path of length {n}");
    }

    /// Permanently displace the interior point at index `i` to a new,
    /// non-collinear location — a kink that does *not* backtrack (no
    /// A→B→A structure: the path never returns to the original `path[i]`).
    /// Endpoints are untouched, so level-1 (total displacement) is
    /// invariant; only the level-2 (area/curvature) signature component
    /// should move. The offset is chosen **perpendicular** to
    /// `path[i+1] - path[i-1]` (see `find_displaceable_index`) so the
    /// depth-2 signed-area term is maximal rather than accidentally
    /// vanishing for a fixed-direction offset.
    fn displace_interior_point(path: &[Vec<f64>], i: usize) -> Vec<Vec<f64>> {
        let combined_x = path[i + 1][0] - path[i - 1][0];
        let combined_y = path[i + 1][1] - path[i - 1][1];
        debug_assert!(
            combined_x != 0.0 || combined_y != 0.0,
            "index {i} is degenerate for displacement — use find_displaceable_index"
        );
        // Rotate 90°: (x, y) -> (-y, x). Guaranteed non-parallel to
        // (combined_x, combined_y) whenever that vector is nonzero.
        const SCALE: f64 = 8.0;
        let offset_x = -combined_y * SCALE;
        let offset_y = combined_x * SCALE;

        let mut edited = path.to_vec();
        edited[i] = vec![path[i][0] + offset_x, path[i][1] + offset_y];
        edited
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// (a) replay_identity — building the same deterministic path twice
        /// and signing both must produce identical digests. Exact f64
        /// equality is expected (same ops, same order, same seed); if this
        /// ever proves flaky across toolchains, the fallback is
        /// `signature_distance(...) < 1e-12`, tighter than the module's own
        /// `FORWARD_TOLERANCE` — noted here rather than used, since exact
        /// equality held on this run (see the assert below).
        #[test]
        fn replay_identity() {
            let path_a = build_path(PATH_SEED);
            let path_b = build_path(PATH_SEED);
            assert_eq!(path_a, path_b, "path replay must be byte-identical");

            let sig_a = signature_truncated(&path_a, DEPTH);
            let sig_b = signature_truncated(&path_b, DEPTH);

            assert_eq!(sig_a.dim, sig_b.dim);
            assert_eq!(sig_a.depth, sig_b.depth);

            let mut max_abs_diff = 0.0_f64;
            for (la, lb) in sig_a.levels.iter().zip(sig_b.levels.iter()) {
                assert_eq!(la.len(), lb.len());
                for (&xa, &xb) in la.iter().zip(lb.iter()) {
                    // Exact f64 equality: same seed, same accumulation
                    // order, same floating-point operations both times.
                    assert_eq!(
                        xa.to_bits(),
                        xb.to_bits(),
                        "replay must be bit-identical per coefficient"
                    );
                    max_abs_diff = max_abs_diff.max((xa - xb).abs());
                }
            }

            let dist = signature_distance(&sig_a, &sig_b);
            eprintln!(
                "[PROBE-SIG-CHECKSUM] (a) replay_identity: max |Δcoeff| = {:.3e}, \
                 Frobenius distance = {:.3e} (exact bit-equality asserted above; \
                 fallback tolerance would have been 1e-12)",
                max_abs_diff, dist
            );
            assert_eq!(dist, 0.0, "replayed digests must be exactly identical");
        }

        /// (b) tree_like_edit_invisible — an out-and-back excursion spliced
        /// mid-path must leave the digest within `FORWARD_TOLERANCE` of the
        /// original (the module's own zero-threshold, reused from
        /// `hambly_lyons.rs`'s forward leg).
        #[test]
        fn tree_like_edit_invisible() {
            let path = build_path(PATH_SEED);
            let mid = PATH_LEN / 2;
            let edited = insert_tree_like_excursion(&path, mid);
            assert_eq!(edited.len(), path.len() + 2);

            let sig_orig = signature_truncated(&path, DEPTH);
            let sig_edited = signature_truncated(&edited, DEPTH);
            let dist = signature_distance(&sig_orig, &sig_edited);

            eprintln!(
                "[PROBE-SIG-CHECKSUM] (b) tree_like_edit_invisible: \
                 excursion at index {mid}, distance = {:.3e} (pass if < {:.0e})",
                dist, FORWARD_TOLERANCE
            );
            assert!(
                dist < FORWARD_TOLERANCE,
                "tree-like A→B→A excursion must be invisible to the digest: \
                 distance {dist:e} >= FORWARD_TOLERANCE {FORWARD_TOLERANCE:e}"
            );
        }

        /// (c) non_tree_edit_caught — permanently displacing one interior
        /// point (no backtrack, endpoints unchanged) must move the digest
        /// past `CONVERSE_THRESHOLD` (the module's own converse-leg
        /// threshold, reused from `hambly_lyons.rs`).
        #[test]
        fn non_tree_edit_caught() {
            let path = build_path(PATH_SEED);
            let mid = find_displaceable_index(&path, PATH_LEN / 2);
            let edited = displace_interior_point(&path, mid);
            assert_eq!(edited.len(), path.len());
            // Endpoints untouched — level-1 (total displacement) must be
            // invariant; the digest still has to move via level-2.
            assert_eq!(edited[0], path[0]);
            assert_eq!(edited[path.len() - 1], path[path.len() - 1]);

            let sig_orig = signature_truncated(&path, DEPTH);
            let sig_edited = signature_truncated(&edited, DEPTH);
            assert_eq!(sig_orig.dim, DIM);
            let dist = signature_distance(&sig_orig, &sig_edited);

            // Level-1 alone (total displacement, endpoint − startpoint) is
            // mathematically invariant under an interior-only displacement;
            // computed directly (not via `signature_distance`, which
            // requires matching `depth`) purely as a diagnostic confirming
            // the discriminating signal comes from level-2, not level-1.
            let level1_delta: f64 = sig_orig.levels[1]
                .iter()
                .zip(sig_edited.levels[1].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();

            eprintln!(
                "[PROBE-SIG-CHECKSUM] (c) non_tree_edit_caught: \
                 displaced index {mid}, distance = {:.4} (pass if > {:.2}); \
                 level-1-only delta = {:.3e} (expected ~0, invariant under \
                 an interior-only displacement)",
                dist, CONVERSE_THRESHOLD, level1_delta
            );
            assert!(
                dist > CONVERSE_THRESHOLD,
                "non-tree interior displacement must be caught by the digest: \
                 distance {dist} <= CONVERSE_THRESHOLD {CONVERSE_THRESHOLD}"
            );
        }
    }
}
