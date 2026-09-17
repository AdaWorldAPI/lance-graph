//! Four-arm probe for the per-axis facet LCP — does the PEEK shape or the
//! masked single-register shape actually win, and at which workload?
//!
//! ## Why this exists
//!
//! `EPIPHANIES.md` (16) `E-FORMAT-SLOT-FOLD-IS-THE-SAME-OP-AS-THE-VL-DESCENT-1`
//! records *"loop 12.5 ns → masked readout 5.8 ns for both axes"* on 64K random
//! pairs, and that number retired the `[u8; 6]` chain fold in favour of the
//! masked `u128` readout (`shared_axis`, `facet.rs`). **No harness for that
//! number is committed anywhere in this tree** — the sibling ndarray number in
//! the same entry names `examples/ternlogq_tail_descent_probe.rs`, this one
//! names nothing, and the entry's own status line says MEASURED for the ndarray
//! side and only SHIPPED for the contract side. So the comparison is not
//! reproducible and not falsifiable as recorded. This probe makes it both.
//!
//! ## The two open questions
//!
//! 1. **Was the baseline the fold, or the fold compiled badly?** A `[u8; 6]`
//!    pick off a `repr(C, align(16))` struct is six constant-offset byte reads —
//!    a PEEK. That is nearly free *when the bytes have addresses* (in memory, or
//!    in an xmm register a shuffle can index). Hoisted into a GPR pair it has no
//!    byte addressing and the pick degrades to scalar shift-and-mask. Arms A and
//!    B differ in exactly that: A takes the facet **by value** (`hi_chain()`
//!    consumes `self`), B reads **through `as_bytes()`** (the documented
//!    reinterpret no-op) at constant offsets.
//! 2. **Was the workload representative?** Uniformly random pairs diverge at
//!    tier 0 almost always, which is the *shortest* possible prefix. An LCP is
//!    asked about near neighbours, where the prefix is long. Every arm is
//!    therefore measured at each shared-prefix depth 0..=6, not just random.
//!
//! ## Arms (all four compute the same two numbers: shared hi- and lo-prefix)
//!
//! | arm | shape | readout |
//! |---|---|---|
//! | A `chain_loop`   | by-value `hi_chain()`/`lo_chain()` + byte loop — the shape #1244 retired | early-exit compare |
//! | B `peek_u64`     | `as_bytes()` + six constant-offset reads packed to `u64` | `xor`, `tzcnt`, `>> 3` — contiguous, **no mask, no −32** |
//! | C `masked_u128`  | the shipped `hi_distance`/`lo_distance` | `xor`, `& AXIS`, `tzcnt`, `− 32`, `/ 16` |
//! | D `peek_both`    | one `as_bytes()` pair, both axes from the same reads | as B, shared loads |
//!
//! D exists because A/B/C are each called twice when both axes are wanted, and
//! the recorded 5.8 ns for both axes (2.9 ns each) shows zero sharing.
//!
//! ## Falsifiability
//!
//! * **Oracle first.** Every arm is cross-checked against every other arm AND
//!   against the shipped `hi_distance`/`lo_distance` on every generated pair,
//!   before any timing. A mismatch aborts with the pair printed — no numbers are
//!   reported from a wrong arm.
//! * **Anti-vacuity on the workload.** The per-depth rows prove the depth knob
//!   binds: arm A's cost must rise with depth (its loop runs longer) or the
//!   generator is not actually producing longer prefixes. That is asserted.
//! * **Dead-code guard.** Inputs and the accumulated checksum go through
//!   `black_box`; the checksum is printed so the loop cannot be elided.
//! * **Arms are `#[inline(never)]`** so `cargo asm` can find them. Call overhead
//!   is therefore included, identically, in all four — it does not move the
//!   comparison, and it is what makes the asm question answerable:
//!   `cargo asm --example facet_axis_lcp_probe <arm>` on A vs B is expected to
//!   show scalar shift/mask for A and byte loads (or a shuffle) for B.
//!
//! This probe measures; it rules nothing. Whatever it prints is a measurement on
//! one machine and one toolchain, and belongs in the board entry as such.
//!
//! ## Run
//!
//! ```text
//! cargo run --release --example facet_axis_lcp_probe -p lance-graph-contract
//! ```

use lance_graph_contract::facet::FacetCascade;
use std::hint::black_box;
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic input generation (SplitMix64, the workspace's probe seed).
// ─────────────────────────────────────────────────────────────────────────────

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fill16(&mut self) -> [u8; 16] {
        let (a, b) = (self.next_u64().to_le_bytes(), self.next_u64().to_le_bytes());
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&a);
        out[8..].copy_from_slice(&b);
        out
    }
}

/// What kind of pair to generate.
#[derive(Clone, Copy)]
enum Workload {
    /// Two independent random facets — both axes diverge at tier 0 with
    /// probability (1 - 2^-8) ≈ 0.996. This is what the recorded 12.5/5.8 used.
    Random,
    /// `b` equals `a` except one byte flipped at tier `d` on BOTH axes, so the
    /// shared prefix is exactly `d` on each. `d == 6` ⇒ identical pair.
    Depth(u8),
}

impl Workload {
    fn label(self) -> String {
        match self {
            Workload::Random => "random".to_string(),
            Workload::Depth(6) => "identical".to_string(),
            Workload::Depth(d) => format!("depth {d}"),
        }
    }
}

fn make_pairs(w: Workload, n: usize, rng: &mut SplitMix64) -> Vec<(FacetCascade, FacetCascade)> {
    (0..n)
        .map(|_| {
            let ba = rng.fill16();
            let bb = match w {
                Workload::Random => rng.fill16(),
                Workload::Depth(d) => {
                    let mut bb = ba;
                    if d < 6 {
                        let t = d as usize;
                        bb[4 + 2 * t] ^= 0x80; // lo byte of tier d
                        bb[5 + 2 * t] ^= 0x80; // hi byte of tier d
                    }
                    bb
                }
            };
            (FacetCascade::from_bytes(&ba), FacetCascade::from_bytes(&bb))
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// The four arms. Each returns (shared_hi, shared_lo), both 0..=6.
// ─────────────────────────────────────────────────────────────────────────────

/// **Arm A** — the `[u8; 6]` chain fold that #1244 retired, verbatim: `hi_chain`
/// takes `self` by value, so the facet is copied out of memory before the pick.
#[inline(never)]
fn arm_a_chain_loop(a: &FacetCascade, b: &FacetCascade) -> (u8, u8) {
    fn looped(x: [u8; 6], y: [u8; 6]) -> u8 {
        let mut n = 0u8;
        while (n as usize) < 6 && x[n as usize] == y[n as usize] {
            n += 1;
        }
        n
    }
    (
        looped(a.hi_chain(), b.hi_chain()),
        looped(a.lo_chain(), b.lo_chain()),
    )
}

/// Pack one axis's six bytes, coarse→fine, into the low 48 bits of a `u64`.
/// `axis_off` is 1 for `hi` (bytes 5,7,…,15), 0 for `lo` (bytes 4,6,…,14).
#[inline(always)]
fn peek_axis(bytes: &[u8; 16], axis_off: usize) -> u64 {
    let mut v = 0u64;
    let mut t = 0;
    while t < 6 {
        v |= (bytes[4 + 2 * t + axis_off] as u64) << (8 * t);
        t += 1;
    }
    v
}

/// Shared prefix of two packed axes: contiguous bytes, so `tz >> 3` with no
/// mask and no offset correction — the classid was never picked up.
#[inline(always)]
fn shared_packed(x: u64) -> u8 {
    if x == 0 {
        6
    } else {
        (x.trailing_zeros() >> 3) as u8
    }
}

/// **Arm B** — the PEEK shape: read through `as_bytes()` (the documented
/// reinterpret no-op) at constant offsets, pack contiguous, one `tzcnt` each.
#[inline(never)]
fn arm_b_peek_u64(a: &FacetCascade, b: &FacetCascade) -> (u8, u8) {
    let (pa, pb) = (a.as_bytes(), b.as_bytes());
    (
        shared_packed(peek_axis(pa, 1) ^ peek_axis(pb, 1)),
        shared_packed(peek_axis(pa, 0) ^ peek_axis(pb, 0)),
    )
}

/// **Arm C** — the shipped masked `u128` readout, through the public API.
#[inline(never)]
fn arm_c_masked_u128(a: &FacetCascade, b: &FacetCascade) -> (u8, u8) {
    const fn axis_mask(axis_off: u32) -> u128 {
        let mut m = 0u128;
        let mut t = 0;
        while t < 6 {
            m |= 0xFF << (8 * (4 + 2 * t + axis_off));
            t += 1;
        }
        m
    }
    const fn shared(x: u128, mask: u128) -> u8 {
        let x = x & mask;
        if x == 0 {
            6
        } else {
            ((x.trailing_zeros() - 32) / 16) as u8
        }
    }
    let x = a.as_u128() ^ b.as_u128();
    (shared(x, axis_mask(1)), shared(x, axis_mask(0)))
}

/// **Arm D** — arm B with the two `as_bytes()` reads shared across both axes,
/// which is what a caller wanting locality on both hierarchies actually needs.
#[inline(never)]
fn arm_d_peek_both(a: &FacetCascade, b: &FacetCascade) -> (u8, u8) {
    let (pa, pb) = (a.as_bytes(), b.as_bytes());
    let (mut h, mut l) = (0u64, 0u64);
    let mut t = 0;
    while t < 6 {
        let (i_lo, i_hi) = (4 + 2 * t, 5 + 2 * t);
        h |= ((pa[i_hi] ^ pb[i_hi]) as u64) << (8 * t);
        l |= ((pa[i_lo] ^ pb[i_lo]) as u64) << (8 * t);
        t += 1;
    }
    (shared_packed(h), shared_packed(l))
}

type Arm = (&'static str, fn(&FacetCascade, &FacetCascade) -> (u8, u8));

const ARMS: [Arm; 4] = [
    ("A chain_loop", arm_a_chain_loop),
    ("B peek_u64", arm_b_peek_u64),
    ("C masked_u128", arm_c_masked_u128),
    ("D peek_both", arm_d_peek_both),
];

// ─────────────────────────────────────────────────────────────────────────────
// Oracle — every arm against every other arm, and against the shipped API.
// ─────────────────────────────────────────────────────────────────────────────

fn check_arms_agree(pairs: &[(FacetCascade, FacetCascade)], what: &str) {
    for (a, b) in pairs {
        let shipped = (6 - a.hi_distance(b), 6 - a.lo_distance(b));
        for (name, f) in ARMS {
            let got = f(a, b);
            assert_eq!(
                got,
                shipped,
                "ORACLE MISMATCH in {what}: arm {name} gave {got:?}, shipped \
                 hi_distance/lo_distance gave {shipped:?}\n  a = {:02x?}\n  b = {:02x?}",
                a.as_bytes(),
                b.as_bytes(),
            );
        }
    }
}

/// The depth knob must bind: a pair generated at depth `d` must actually have
/// shared prefix `d` on both axes. Without this the per-depth rows are theatre.
fn check_depth_knob_binds(rng: &mut SplitMix64) {
    for d in 0..=6u8 {
        let pairs = make_pairs(Workload::Depth(d), 64, rng);
        for (a, b) in &pairs {
            let got = arm_b_peek_u64(a, b);
            assert_eq!(
                got,
                (d, d),
                "depth knob does not bind: asked for depth {d}, measured {got:?}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Timing
// ─────────────────────────────────────────────────────────────────────────────

/// Min-of-`runs` ns/op. Min is the right statistic here: the true cost is a
/// floor and every perturbation (interrupt, migration, frequency dip) only adds.
fn time_arm(
    f: fn(&FacetCascade, &FacetCascade) -> (u8, u8),
    pairs: &[(FacetCascade, FacetCascade)],
    runs: usize,
) -> f64 {
    // Warm up: caches, branch predictors, frequency.
    let mut sink = 0u64;
    for _ in 0..2 {
        for (a, b) in pairs {
            let (h, l) = f(black_box(a), black_box(b));
            sink = sink.wrapping_add(h as u64).wrapping_add(l as u64);
        }
    }
    black_box(sink);

    let mut best = f64::MAX;
    for _ in 0..runs {
        let mut acc = 0u64;
        let t0 = Instant::now();
        for (a, b) in pairs {
            let (h, l) = f(black_box(a), black_box(b));
            acc = acc.wrapping_add(h as u64).wrapping_add(l as u64);
        }
        let ns = t0.elapsed().as_nanos() as f64 / pairs.len() as f64;
        black_box(acc);
        if ns < best {
            best = ns;
        }
    }
    best
}

fn main() {
    const N: usize = 65_536; // the entry's "64K random pairs"
    const RUNS: usize = 7;

    let mut rng = SplitMix64(0x9E37_79B9_7F4A_7C15);

    println!("facet per-axis LCP — four arms, {N} pairs, min of {RUNS} runs, ns/op (both axes)");
    println!("target: {}", std::env::var("RUSTFLAGS").unwrap_or_default());
    println!();

    // Gates before numbers.
    check_depth_knob_binds(&mut rng);
    println!("depth knob binds (0..=6 verified on 64 pairs each)");

    let workloads = [
        Workload::Random,
        Workload::Depth(0),
        Workload::Depth(1),
        Workload::Depth(2),
        Workload::Depth(3),
        Workload::Depth(4),
        Workload::Depth(5),
        Workload::Depth(6),
    ];

    let mut rows: Vec<(String, Vec<f64>)> = Vec::new();
    let mut arm_a_by_depth: Vec<f64> = Vec::new();

    for w in workloads {
        let pairs = make_pairs(w, N, &mut rng);
        check_arms_agree(&pairs, &w.label());
        let times: Vec<f64> = ARMS
            .iter()
            .map(|(_, f)| time_arm(*f, &pairs, RUNS))
            .collect();
        if let Workload::Depth(_) = w {
            arm_a_by_depth.push(times[0]);
        }
        rows.push((w.label(), times));
    }
    println!("oracle: all 4 arms agree with the shipped API on every workload");
    println!();

    // Table.
    print!("{:<12}", "workload");
    for (name, _) in ARMS {
        print!("{name:>16}");
    }
    println!();
    println!("{}", "-".repeat(12 + 16 * ARMS.len()));
    for (label, times) in &rows {
        print!("{label:<12}");
        for t in times {
            print!("{t:>16.2}");
        }
        println!();
    }
    println!();

    // Anti-vacuity: arm A is the only arm whose work depends on prefix length,
    // so its cost must rise from depth 0 to depth 5. If it does not, the
    // early-exit loop is not being compiled as one and the A/B contrast is not
    // measuring what this probe claims.
    let (a0, a5) = (arm_a_by_depth[0], arm_a_by_depth[5]);
    let binds = a5 > a0 * 1.05;
    println!(
        "arm A depth-0 {a0:.2} ns → depth-5 {a5:.2} ns ({:+.1}%) — the early-exit loop {}",
        (a5 - a0) / a0 * 100.0,
        if binds {
            "does depend on prefix length, as it must"
        } else {
            "does NOT depend on prefix length — READ THE ASM before trusting any row above"
        }
    );
    println!();
    // This is a gate, not a remark: a flat slope means the A/B contrast is not
    // measuring what this probe claims, so the table above must not be
    // recorded. Fail the process rather than let an inattentive run bank it.
    assert!(
        binds,
        "ANTI-VACUITY FAILED: arm A depth-5 ({a5:.2} ns) is not >5% above depth-0 \
         ({a0:.2} ns); the early-exit chain is not being compiled as one on this \
         target and every row above is invalid"
    );
    println!("next: cargo asm --example facet_axis_lcp_probe arm_a_chain_loop");
    println!("      cargo asm --example facet_axis_lcp_probe arm_b_peek_u64");
    println!("      A scalar shift/mask vs B byte-loads is the whole question.");
}
