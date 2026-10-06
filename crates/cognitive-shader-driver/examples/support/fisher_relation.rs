//! Shared support for the D-CTX probes (included with `#[path]`, not an
//! example of its own): the thread-local allocation counter, the Moore
//! offsets, and the borrowed Palette256 × Palette256 Fisher-Z relation.
//!
//! One copy, so the probes cannot drift apart. Each probe binary includes it
//! once; not every probe uses every item.
#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use bgz_tensor::fisher_z::{FamilyGamma, FisherZTable};
use cognitive_shader_driver::palette_perturbation::PaletteState;

// ── allocation counter (per thread, so parallel tests cannot pollute it) ────

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static ALLOC_BYTES: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

// SAFETY: a pure pass-through to `System`; the thread-local counters are the
// only addition and they never allocate (const-initialised `Cell<usize>`).
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        let _ = ALLOC_BYTES.try_with(|c| c.set(c.get() + layout.size()));
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// `(allocations, bytes)` made on this thread while `f` ran.
pub fn allocations_during<R>(f: impl FnOnce() -> R) -> (R, usize, usize) {
    let (n0, b0) = (ALLOCATIONS.with(Cell::get), ALLOC_BYTES.with(Cell::get));
    let r = f();
    let (n1, b1) = (ALLOCATIONS.with(Cell::get), ALLOC_BYTES.with(Cell::get));
    (r, n1 - n0, b1 - b0)
}

/// The eight Moore offsets, NW N NE W E SW S SE (the contract test's order).
pub const MOORE: [(i8, i8); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

// ── the relation carrier: borrowed, calibrated, never recomputed ───────────

/// One local relation. `Identity` comes from address equality, `Pair` from
/// the Fisher-Z table. Two bytes, register-resident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    Identity,
    Pair(i8),
}

/// A borrowed Palette256 × Palette256 Fisher-Z law: the calibrated i8 table,
/// its family gamma, and a generation identity computed once at borrow time.
#[derive(Clone, Copy, Debug)]
pub struct PairwiseFisherZ<'a> {
    pub entries: &'a [i8],
    pub gamma: FamilyGamma,
    pub generation: u64,
}

impl<'a> PairwiseFisherZ<'a> {
    /// Borrow a full 256 x 256 table. Panics on a smaller palette: every
    /// `u8` must have a row, so the hot read needs no range check.
    pub fn borrow(table: &'a FisherZTable) -> Self {
        assert_eq!(table.k, 256, "a Palette256 law needs k = 256");
        assert_eq!(table.entries.len(), 256 * 256);
        // FNV-1a over gamma then entries: the law's generation identity.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let gamma = table.gamma.to_le_bytes();
        let bytes = gamma
            .iter()
            .copied()
            .chain(table.entries.iter().map(|&v| v as u8));
        for b in bytes {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        Self {
            entries: &table.entries,
            gamma: table.gamma,
            generation: h,
        }
    }

    /// The local relation of two palette ordinals: one byte read, or identity.
    #[inline(always)]
    pub fn relation(&self, a: PaletteState, b: PaletteState) -> Relation {
        if a == b {
            Relation::Identity
        } else {
            Relation::Pair(self.entries[(a.0 as usize) << 8 | b.0 as usize])
        }
    }

    /// `z` for every i8 code, built once per generation. Affine in the code,
    /// so no transcendental is evaluated at all.
    pub fn z_decode(&self) -> [f32; 256] {
        core::array::from_fn(|i| {
            let code = i as u8 as i8;
            (f32::from(code) + 127.0) / 254.0 * self.gamma.z_range + self.gamma.z_min
        })
    }
}

// ── calibration fixture ─────────────────────────────────────────────────────

/// 256 deterministic representative rows (dim 16, SplitMix64), for the one
/// calibration call. This is calibration, not the hot path.
pub fn representatives(seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    (0..256)
        .map(|_| (0..16).map(|_| next()).collect())
        .collect()
}
