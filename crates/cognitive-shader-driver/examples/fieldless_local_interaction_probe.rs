//! D-CTX-0: fieldless local interaction probe.
//!
//! Question under test: does a cognitive texture have to exist as a field to
//! produce its local consequence, or are
//!
//! ```text
//! address + pairwise law + activation + register
//! ```
//!
//! enough? This probe answers it for one synchronous Moore step over a 4 x 4
//! resident tile, before any Gaussian or EWA rendering exists.
//!
//! # The three readings, kept apart
//!
//! - **Palette byte = point / needle.** `PaletteState(u8)` is the material at a
//!   position. It never carries geometry and never carries a distribution.
//! - **Position = geometry.** The tile is a [`Register128`] whose lane index is
//!   the [`Morton8x8`] code of `(x, y)`. A 4 x 4 grid is exactly the codes
//!   `0..16` (contract test `a_power_of_two_subgrid_is_a_code_prefix`), so the
//!   lane needs no row-major translation and a neighbour is
//!   `checked_offset(dx, dy)` kept only when `code < 16`.
//! - **Palette : Palette = relation / distribution.** The local relation is
//!   `R_local = FisherZ[palette(center), palette(neighbor)]`, read from the
//!   calibrated i8 table `bgz_tensor::fisher_z::FisherZTable` (the one
//!   calibrator; this file holds no `atanh`, no cosine and no `tanh`).
//!
//! The diagonal rule: when the two palette bytes are equal the answer is
//! *identity*, given by address equality. The distribution table is never
//! asked a needle question. Identity neighbours are counted separately and add
//! no Fisher-Z code; what an identity neighbour should contribute is a policy
//! this probe does not invent.
//!
//! # The update
//!
//! ```text
//! local update = activation[center] × Σ_{n ∈ Moore(center)} R_local(center, n)
//! ```
//!
//! `activation` is an external per-lane byte (the SPOFC / ONNX / energy reading,
//! orthogonal to `R_local`); `0` gates the lane. The result is added to a
//! resident per-lane strength register. The palette tile is never written, so
//! the step is synchronous by construction and the integer sum is commutative:
//! the visiting order cannot change the result (pinned below, not assumed).
//!
//! # Two implementations
//!
//! - **A, materialized oracle** (test-only shape): row-major geometry with plain
//!   integer bounds checks, a `Vec` of directed pairs, a `Vec` of relations read
//!   through `FisherZTable::lookup_i8`, a per-lane sum field, then activation.
//! - **B, fieldless:** center code → checked Morton neighbour → resident
//!   neighbour byte → one table read → immediate fold into one `LocalSum`.
//!   No pair list, no relation population, no field.
//!
//! Run: `cargo run -p cognitive-shader-driver --example fieldless_local_interaction_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example fieldless_local_interaction_probe`

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::mem::size_of;

use bgz_tensor::fisher_z::{FamilyGamma, FisherZTable};
use cognitive_shader_driver::palette_perturbation::PaletteState;
use lance_graph_contract::morton8x8::Morton8x8;
use lance_graph_contract::register128::Register128;

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
fn allocations_during<R>(f: impl FnOnce() -> R) -> (R, usize, usize) {
    let (n0, b0) = (ALLOCATIONS.with(Cell::get), ALLOC_BYTES.with(Cell::get));
    let r = f();
    let (n1, b1) = (ALLOCATIONS.with(Cell::get), ALLOC_BYTES.with(Cell::get));
    (r, n1 - n0, b1 - b0)
}

// ── geometry ───────────────────────────────────────────────────────────────

/// Grid side; the tile is the 16-code Morton prefix.
const SIDE: u8 = 4;
/// Lanes in the tile.
const LANES: usize = 16;

/// The eight Moore offsets, NW N NE W E SW S SE (the contract test's order).
const MOORE: [(i8, i8); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// The neighbour of `center` at `(dx, dy)` inside the 4 x 4 tile, or `None`.
#[inline(always)]
fn neighbor(center: Morton8x8, dx: i8, dy: i8) -> Option<Morton8x8> {
    center
        .checked_offset(dx, dy)
        .filter(|n| n.code() < LANES as u16)
}

// ── the relation carrier: borrowed, calibrated, never recomputed ───────────

/// One local relation. `Identity` comes from address equality, `Pair` from
/// the Fisher-Z table. Two bytes, register-resident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Relation {
    Identity,
    Pair(i8),
}

/// A borrowed Palette256 × Palette256 Fisher-Z law: the calibrated i8 table,
/// its family gamma, and a generation identity computed once at borrow time.
#[derive(Clone, Copy, Debug)]
struct PairwiseFisherZ<'a> {
    entries: &'a [i8],
    gamma: FamilyGamma,
    generation: u64,
}

impl<'a> PairwiseFisherZ<'a> {
    /// Borrow a full 256 x 256 table. Panics on a smaller palette: every
    /// `u8` must have a row, so the hot read needs no range check.
    fn borrow(table: &'a FisherZTable) -> Self {
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
    fn relation(&self, a: PaletteState, b: PaletteState) -> Relation {
        if a == b {
            Relation::Identity
        } else {
            Relation::Pair(self.entries[(a.0 as usize) << 8 | b.0 as usize])
        }
    }

    /// `z` for every i8 code, built once per generation. Affine in the code,
    /// so no transcendental is evaluated at all.
    fn z_decode(&self) -> [f32; 256] {
        core::array::from_fn(|i| {
            let code = i as u8 as i8;
            (f32::from(code) + 127.0) / 254.0 * self.gamma.z_range + self.gamma.z_min
        })
    }
}

// ── B: the fieldless path ──────────────────────────────────────────────────

/// The folded consequence of one center's Moore neighbourhood. Eight bytes,
/// register-resident; no neighbour or relation is retained.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LocalSum {
    /// Σ of the Fisher-Z i8 codes over the non-identity neighbours.
    code_sum: i32,
    /// Neighbours that contributed a Fisher-Z code.
    pairs: u8,
    /// Neighbours with the same palette byte (address answered, no read).
    identities: u8,
}

/// Fold one center: checked neighbour, resident byte, one read, accumulate.
#[inline]
fn fold_center(tile: &Register128, center: Morton8x8, law: &PairwiseFisherZ<'_>) -> LocalSum {
    let c = PaletteState(tile.0[center.code() as usize]);
    let mut acc = LocalSum::default();
    for &(dx, dy) in &MOORE {
        let Some(n) = neighbor(center, dx, dy) else {
            continue;
        };
        match law.relation(c, PaletteState(tile.0[n.code() as usize])) {
            Relation::Identity => acc.identities += 1,
            Relation::Pair(r) => {
                acc.code_sum += i32::from(r);
                acc.pairs += 1;
            }
        }
    }
    acc
}

/// One synchronous step: every activated lane adds
/// `activation × Σ R_local` to its resident strength. Lanes are visited in the
/// order given; a gated lane (activation `0`) is not visited at all.
fn step_in_order(
    tile: &Register128,
    activation: &Register128,
    strength: &mut [i32; LANES],
    law: &PairwiseFisherZ<'_>,
    order: impl Iterator<Item = u16>,
) {
    for code in order {
        let a = activation.0[code as usize];
        if a == 0 {
            continue;
        }
        let s = fold_center(tile, Morton8x8::from_code(code), law);
        strength[code as usize] += i32::from(a) * s.code_sum;
    }
}

/// The step in Morton code order.
fn step(
    tile: &Register128,
    activation: &Register128,
    strength: &mut [i32; LANES],
    law: &PairwiseFisherZ<'_>,
) {
    step_in_order(tile, activation, strength, law, 0..LANES as u16);
}

// ── A: the materialized oracle ─────────────────────────────────────────────

/// What the oracle had to build, in bytes.
#[derive(Clone, Copy, Debug, Default)]
struct OracleInventory {
    pair_bytes: usize,
    relation_bytes: usize,
    field_bytes: usize,
}

/// Row-major geometry with integer bounds checks, a pair list, a relation
/// population read through `FisherZTable::lookup_i8`, and a per-lane field.
/// `from_xy` only translates the row-major result back to Morton lanes.
fn oracle_step(
    tile: &Register128,
    activation: &Register128,
    strength: &[i32; LANES],
    table: &FisherZTable,
) -> ([i32; LANES], OracleInventory) {
    let side = i16::from(SIDE);
    let cell = |x: i16, y: i16| tile.0[Morton8x8::from_xy(x as u8, y as u8).code() as usize];

    let mut pairs: Vec<((i16, i16), (i16, i16))> = Vec::new();
    for y in 0..side {
        for x in 0..side {
            for &(dx, dy) in &MOORE {
                let (nx, ny) = (x + i16::from(dx), y + i16::from(dy));
                if (0..side).contains(&nx) && (0..side).contains(&ny) {
                    pairs.push(((x, y), (nx, ny)));
                }
            }
        }
    }
    let relations: Vec<Relation> = pairs
        .iter()
        .map(|&((x, y), (nx, ny))| {
            let (a, b) = (cell(x, y), cell(nx, ny));
            if a == b {
                Relation::Identity
            } else {
                Relation::Pair(table.lookup_i8(a, b))
            }
        })
        .collect();
    let mut field = [LocalSum::default(); LANES];
    for (&((x, y), _), rel) in pairs.iter().zip(&relations) {
        let s = &mut field[(y * side + x) as usize];
        match *rel {
            Relation::Identity => s.identities += 1,
            Relation::Pair(r) => {
                s.code_sum += i32::from(r);
                s.pairs += 1;
            }
        }
    }
    let mut out = *strength;
    for y in 0..side {
        for x in 0..side {
            let lane = Morton8x8::from_xy(x as u8, y as u8).code() as usize;
            let a = activation.0[lane];
            if a != 0 {
                out[lane] += i32::from(a) * field[(y * side + x) as usize].code_sum;
            }
        }
    }
    let inv = OracleInventory {
        pair_bytes: pairs.len() * size_of::<((i16, i16), (i16, i16))>(),
        relation_bytes: relations.len() * size_of::<Relation>(),
        field_bytes: field.len() * size_of::<LocalSum>(),
    };
    (out, inv)
}

// ── fixtures ───────────────────────────────────────────────────────────────

/// 256 deterministic representative rows (dim 16, SplitMix64), for the one
/// calibration call. This is calibration, not the hot path.
fn representatives(seed: u64) -> Vec<Vec<f32>> {
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

/// Write a row-major 4 x 4 grid into Morton lanes.
fn morton_tile(rows: [[u8; 4]; 4]) -> Register128 {
    let mut t = Register128::default();
    for (y, row) in rows.iter().enumerate() {
        for (x, &v) in row.iter().enumerate() {
            t.0[Morton8x8::from_xy(x as u8, y as u8).code() as usize] = v;
        }
    }
    t
}

/// Palette materials. (0,0)=(1,0)=7 and (2,0)=(2,1)=200 are equal
/// neighbours, so the identity path is exercised.
fn palette_fixture() -> Register128 {
    morton_tile([
        [7, 7, 200, 13],
        [42, 99, 200, 3],
        [255, 0, 64, 128],
        [17, 7, 31, 250],
    ])
}

/// External activation (SPOFC / energy reading). Zeros gate their lanes.
fn activation_fixture() -> Register128 {
    morton_tile([[3, 1, 0, 2], [5, 0, 4, 1], [2, 2, 0, 7], [1, 6, 3, 0]])
}

/// A non-zero starting strength, so "unchanged" is not "zero".
fn strength_fixture() -> [i32; LANES] {
    core::array::from_fn(|i| 100 * i as i32 - 700)
}

fn main() {
    let table = FisherZTable::build(&representatives(1), 256);
    let law = PairwiseFisherZ::borrow(&table);
    let (tile, act) = (palette_fixture(), activation_fixture());

    let mut b = strength_fixture();
    let ((), n_alloc, n_bytes) = allocations_during(|| step(&tile, &act, &mut b, &law));
    let (a, inv) = oracle_step(&tile, &act, &strength_fixture(), &table);

    let visits: u32 = (0..LANES as u16)
        .map(|c| {
            MOORE
                .iter()
                .filter(|&&(dx, dy)| neighbor(Morton8x8::from_code(c), dx, dy).is_some())
                .count() as u32
        })
        .sum();
    let identities: u32 = (0..LANES as u16)
        .map(|c| u32::from(fold_center(&tile, Morton8x8::from_code(c), &law).identities))
        .sum();

    // z readout for the interior lane (1,1): from the i32 register alone, and
    // through the once-built decode table. No transcendental either way.
    let z = law.z_decode();
    let center = Morton8x8::from_xy(1, 1);
    let s = fold_center(&tile, center, &law);
    let n = f32::from(s.pairs);
    let z_register =
        law.gamma.z_range / 254.0 * (s.code_sum as f32 + 127.0 * n) + n * law.gamma.z_min;
    let z_table: f32 = MOORE
        .iter()
        .filter_map(|&(dx, dy)| neighbor(center, dx, dy))
        .filter_map(|nb| {
            match law.relation(
                PaletteState(tile.0[center.code() as usize]),
                PaletteState(tile.0[nb.code() as usize]),
            ) {
                Relation::Pair(r) => Some(z[r as u8 as usize]),
                Relation::Identity => None,
            }
        })
        .sum();

    println!("D-CTX-0 fieldless local interaction, 4 x 4 Morton tile, one synchronous step");
    println!("  law generation           : {:#018x}", law.generation);
    println!(
        "  directed Moore visits    : {visits} ({identities} identity, {} Fisher-Z reads)",
        visits - identities
    );
    println!("  A == B                   : {}", a == b);
    println!("  z sum at (1,1)           : register {z_register:.4}, table {z_table:.4}");
    println!("  resident palette bytes   : {}", size_of::<Register128>());
    println!("  resident activation bytes: {}", size_of::<Register128>());
    println!("  resident strength bytes  : {}", size_of::<[i32; LANES]>());
    println!(
        "  borrowed law bytes       : {} + {} gamma",
        law.entries.len(),
        FamilyGamma::BYTE_SIZE
    );
    println!(
        "  B live relation state    : {} B (Relation), fold register {} B (LocalSum)",
        size_of::<Relation>(),
        size_of::<LocalSum>()
    );
    println!("  B heap during step       : {n_alloc} allocations, {n_bytes} B");
    println!(
        "  A materialized           : pairs {} B, relations {} B, field {} B",
        inv.pair_bytes, inv.relation_bytes, inv.field_bytes
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn law_and_table() -> FisherZTable {
        FisherZTable::build(&representatives(1), 256)
    }

    fn run_b(tile: &Register128, act: &Register128, law: &PairwiseFisherZ<'_>) -> [i32; LANES] {
        let mut s = strength_fixture();
        step(tile, act, &mut s, law);
        s
    }

    /// FAILS IF: the fieldless path computes anything other than the
    /// materialized oracle, on the fixture and on 64 random tiles.
    #[test]
    fn fieldless_equals_the_materialized_oracle() {
        let table = law_and_table();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, act) = (palette_fixture(), activation_fixture());
        let (oracle, _) = oracle_step(&tile, &act, &strength_fixture(), &table);
        assert_eq!(run_b(&tile, &act, &law), oracle);

        let mut s: u32 = 0x1234_5678;
        for _ in 0..64 {
            let mut byte = || {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 24) as u8
            };
            let tile = Register128(core::array::from_fn(|_| byte() % 6 * 40));
            let act = Register128(core::array::from_fn(|_| byte() % 4));
            let (oracle, _) = oracle_step(&tile, &act, &strength_fixture(), &table);
            assert_eq!(run_b(&tile, &act, &law), oracle);
        }
    }

    /// FAILS IF: the fieldless step touches the heap, or the counter cannot
    /// see an allocation (the oracle must register some).
    #[test]
    fn the_fieldless_step_allocates_nothing() {
        let table = law_and_table();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, act) = (palette_fixture(), activation_fixture());
        let mut s = strength_fixture();
        let ((), n, bytes) = allocations_during(|| step(&tile, &act, &mut s, &law));
        assert_eq!((n, bytes), (0, 0));

        let (_, n_oracle, bytes_oracle) =
            allocations_during(|| oracle_step(&tile, &act, &strength_fixture(), &table));
        assert!(n_oracle >= 2 && bytes_oracle > 0, "counter is blind");
    }

    /// FAILS IF: a checked Morton neighbour differs from row-major geometry,
    /// or the tile does not have 84 directed visits.
    #[test]
    fn morton_neighbours_match_row_major_geometry() {
        let mut visits = 0;
        for y in 0..SIDE {
            for x in 0..SIDE {
                let c = Morton8x8::from_xy(x, y);
                for &(dx, dy) in &MOORE {
                    let (nx, ny) = (i16::from(x) + i16::from(dx), i16::from(y) + i16::from(dy));
                    let on = (0..4).contains(&nx) && (0..4).contains(&ny);
                    let expect = on.then(|| Morton8x8::from_xy(nx as u8, ny as u8));
                    assert_eq!(neighbor(c, dx, dy), expect, "({x},{y})+({dx},{dy})");
                    visits += usize::from(on);
                }
            }
        }
        assert_eq!(visits, 84);
    }

    /// FAILS IF: changing one palette byte moves a lane that is not its
    /// neighbour, or moves a neighbour by anything other than
    /// `table[n, new] - table[n, old]`, or the edited lane itself does not move.
    ///
    /// The delta is predicted per neighbour instead of "must change": two
    /// different materials can quantise to the same i8 code against a given
    /// neighbour (measured here: palette 42 reads 99 and 150 identically), so a
    /// material change is not guaranteed to be visible to every neighbour.
    #[test]
    fn one_neighbour_byte_changes_exactly_its_neighbourhood() {
        let table = law_and_table();
        let law = PairwiseFisherZ::borrow(&table);
        let act = Register128([1; 16]);
        let tile = palette_fixture();
        let changed_at = Morton8x8::from_xy(1, 1);
        let (old, new) = (tile.0[changed_at.code() as usize], 150u8); // 150 equals nothing
        let mut edited = tile;
        edited.0[changed_at.code() as usize] = new;
        let (before, after) = (run_b(&tile, &act, &law), run_b(&edited, &act, &law));
        let mut visible = 0;
        for code in 0..LANES as u16 {
            let lane = Morton8x8::from_code(code);
            let delta = after[code as usize] - before[code as usize];
            let adjacent = MOORE
                .iter()
                .any(|&(dx, dy)| neighbor(lane, dx, dy) == Some(changed_at));
            if lane == changed_at {
                assert_ne!(delta, 0, "the edited lane must move");
            } else if adjacent {
                let p = tile.0[code as usize];
                let expect =
                    i32::from(table.lookup_i8(p, new)) - i32::from(table.lookup_i8(p, old));
                assert_eq!(delta, expect, "lane {code}");
                visible += usize::from(delta != 0);
            } else {
                assert_eq!(delta, 0, "non-neighbour lane {code} moved");
            }
        }
        // Anti-vacuity: most of the eight neighbours see the change.
        assert!(visible >= 6, "only {visible} of 8 neighbours saw the edit");
    }

    /// FAILS IF: asking the table its diagonal (a needle question) would give
    /// the same result as the pairwise read on this fixture.
    #[test]
    fn a_diagonal_lookup_mutant_is_caught() {
        let table = law_and_table();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, act) = (palette_fixture(), activation_fixture());
        let mut mutant = strength_fixture();
        for code in 0..LANES as u16 {
            let a = act.0[code as usize];
            if a == 0 {
                continue;
            }
            let c = tile.0[code as usize];
            let center = Morton8x8::from_code(code);
            let n = MOORE
                .iter()
                .filter(|&&(dx, dy)| neighbor(center, dx, dy).is_some())
                .count();
            mutant[code as usize] += i32::from(a) * n as i32 * i32::from(table.lookup_i8(c, c));
        }
        assert_ne!(run_b(&tile, &act, &law), mutant);
    }

    /// FAILS IF: a wrong neighbour (row-major arithmetic applied to Morton
    /// lanes) would give the same result as the checked Morton neighbour.
    #[test]
    fn a_wrong_neighbour_mutant_is_caught() {
        let table = law_and_table();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, act) = (palette_fixture(), activation_fixture());
        let mut mutant = strength_fixture();
        for code in 0..LANES as i16 {
            let a = act.0[code as usize];
            if a == 0 {
                continue;
            }
            let c = PaletteState(tile.0[code as usize]);
            let mut sum = 0i32;
            for &(dx, dy) in &MOORE {
                let n = code + i16::from(dx) + 4 * i16::from(dy);
                if (0..16).contains(&n) {
                    if let Relation::Pair(r) = law.relation(c, PaletteState(tile.0[n as usize])) {
                        sum += i32::from(r);
                    }
                }
            }
            mutant[code as usize] += i32::from(a) * sum;
        }
        assert_ne!(run_b(&tile, &act, &law), mutant);
    }

    /// FAILS IF: an identity neighbour reads the table diagonal. Poisoning the
    /// diagonal must leave B unchanged, and the fixture must contain identity
    /// neighbours for this to mean anything.
    #[test]
    fn identity_is_answered_by_the_address_never_by_the_diagonal() {
        let table = law_and_table();
        let mut poisoned = table.clone();
        for a in 0..256 {
            poisoned.entries[a * 256 + a] = -128;
        }
        let (tile, act) = (palette_fixture(), activation_fixture());
        let clean = PairwiseFisherZ::borrow(&table);
        let dirty = PairwiseFisherZ::borrow(&poisoned);
        assert_ne!(clean.generation, dirty.generation);
        assert_eq!(run_b(&tile, &act, &clean), run_b(&tile, &act, &dirty));

        let ids: u32 = (0..LANES as u16)
            .map(|c| u32::from(fold_center(&tile, Morton8x8::from_code(c), &clean).identities))
            .sum();
        assert_eq!(
            ids, 4,
            "two equal-neighbour pairs, each seen from both ends"
        );
    }

    /// FAILS IF: a gated lane (activation 0) changes, an all-zero activation
    /// moves anything, or no activated lane changes.
    #[test]
    fn gated_lanes_do_not_update() {
        let table = law_and_table();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, act) = (palette_fixture(), activation_fixture());
        let out = run_b(&tile, &act, &law);
        let start = strength_fixture();
        let mut moved = 0;
        for lane in 0..LANES {
            if act.0[lane] == 0 {
                assert_eq!(out[lane], start[lane], "gated lane {lane} moved");
            } else if out[lane] != start[lane] {
                moved += 1;
            }
        }
        let active = act.0.iter().filter(|&&a| a != 0).count();
        assert_eq!(
            moved, active,
            "every activated lane must move on this fixture"
        );
        assert_eq!(run_b(&tile, &Register128::default(), &law), start);
    }

    /// FAILS IF: visiting lanes in a different order changes the result. The
    /// law is a commutative integer sum over an unwritten tile, so no order is
    /// pinned; this checks it rather than assuming it.
    #[test]
    fn visit_order_cannot_change_the_result() {
        let table = law_and_table();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, act) = (palette_fixture(), activation_fixture());
        let morton: Vec<u16> = (0..16).collect();
        let reversed: Vec<u16> = (0..16).rev().collect();
        let row_major: Vec<u16> = (0..16)
            .map(|i| Morton8x8::from_xy(i % 4, i / 4).code())
            .collect();
        assert_ne!(morton, row_major, "the orders must actually differ");
        let mut outs = Vec::new();
        for order in [&morton, &reversed, &row_major] {
            let mut s = strength_fixture();
            step_in_order(&tile, &act, &mut s, &law, order.iter().copied());
            outs.push(s);
        }
        assert_eq!(outs[0], outs[1]);
        assert_eq!(outs[0], outs[2]);
    }

    /// FAILS IF: the integer register is not a sufficient statistic for the
    /// z-space sum: Σ z_decode[r] must follow from (code_sum, pairs) alone.
    #[test]
    fn the_integer_register_carries_the_z_sum() {
        let table = law_and_table();
        let law = PairwiseFisherZ::borrow(&table);
        let z = law.z_decode();
        let tile = palette_fixture();
        for code in 0..LANES as u16 {
            let center = Morton8x8::from_code(code);
            let s = fold_center(&tile, center, &law);
            let c = PaletteState(tile.0[code as usize]);
            let mut direct = 0f32;
            for &(dx, dy) in &MOORE {
                if let Some(n) = neighbor(center, dx, dy) {
                    if let Relation::Pair(r) =
                        law.relation(c, PaletteState(tile.0[n.code() as usize]))
                    {
                        direct += z[r as u8 as usize];
                    }
                }
            }
            let n = f32::from(s.pairs);
            let from_register =
                law.gamma.z_range / 254.0 * (s.code_sum as f32 + 127.0 * n) + n * law.gamma.z_min;
            assert!(
                (direct - from_register).abs() < 1e-4,
                "lane {code}: {direct} vs {from_register}"
            );
        }
    }

    /// FAILS IF: the same calibration does not replay to the same generation
    /// and result, or a different calibration keeps the generation.
    #[test]
    fn replay_and_generation_identity() {
        let (t1, t2, t3) = (
            law_and_table(),
            law_and_table(),
            FisherZTable::build(&representatives(2), 256),
        );
        let (l1, l2, l3) = (
            PairwiseFisherZ::borrow(&t1),
            PairwiseFisherZ::borrow(&t2),
            PairwiseFisherZ::borrow(&t3),
        );
        assert_eq!(l1.generation, l2.generation);
        assert_ne!(l1.generation, l3.generation);
        let (tile, act) = (palette_fixture(), activation_fixture());
        assert_eq!(run_b(&tile, &act, &l1), run_b(&tile, &act, &l2));
        assert_ne!(run_b(&tile, &act, &l1), run_b(&tile, &act, &l3));
    }
}
