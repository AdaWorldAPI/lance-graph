//! `D-WXS-2a` half A — a pure key-space comparison of three byte-order
//! arms for the ERA5 grid's `(HEEL, HIP)` axis bytes.
//!
//! Plan: `.claude/plans/weather-soa-bake-v1.md` §1.3a/§1.3b. **HALF A
//! ONLY** — pure key-space, no ERA5 data, no classid, no bake. It measures
//! whether re-ordering the four axis bytes of the shipped key
//! ([`weather_poc::key::encode_key`], bytes `4..8`) changes two locality
//! metrics: range count over a box, and neighbour distance in key order.
//! Half B (the ζ stencil under each layout) is gated on `D-WXS-9` and is
//! out of scope here.
//!
//! # The three arms
//!
//! Every arm is a function of the same four axis-byte inputs the shipped
//! key already carries at bytes `4..8` — `lat_tile = lat_idx >> 6`,
//! `lon_tile = lon_idx >> 6`, `lat_hip = lat_idx & 63`,
//! `lon_hip = lon_idx & 63` — obtained here by calling
//! [`weather_poc::key::encode_key`] itself (never re-derived by hand),
//! then packed into a `u32` whose ordinary numeric (big-endian) ordering
//! is BY CONSTRUCTION identical to lexicographic byte-array ordering.
//! That equivalence is why packing into a `u32` and sorting it is an
//! EXACT stand-in for comparing the real 16-byte keys, not an
//! approximation: `classid`, TWIG, and the tail are always zero in this
//! crate (see [`weather_poc::key`]'s module doc), so for any two grid
//! cells only bytes `4..8` can ever differ, and those four bytes are
//! exactly what each arm packs.
//!
//! * **SHIPPED** — `[lat_tile, lon_tile, lat_hip, lon_hip]`, exactly what
//!   `encode_key` emits at bytes `4..8` today. Obtained by literally
//!   calling `encode_key` and reading those four bytes back — this
//!   crate's `key.rs` is read-only per the brief, and this is the
//!   strongest way to guarantee the arm "matches what it actually does":
//!   zero re-derivation, zero drift risk if `key.rs`'s shift/mask ever
//!   changes.
//! * **MORTON** — the OGAR-canon reading, quoted directly in this
//!   workspace's `CLAUDE.md`: *"the x/y nibble-interleave = alternating-
//!   axis refinement (Morton in centroid space)."* Applied IDENTICALLY to
//!   both tiers (HEEL and HIP), which is what plan §1.3a's open item asks
//!   for. Each tier's two axis bytes are split into nibbles and
//!   interleaved axis-by-axis, coarsest nibble first, lat's nibble in the
//!   high half of each output byte:
//!   `[lat_hi:lon_hi, lat_lo:lon_lo]` per tier (see
//!   [`interleave_nibbles`]). The four MORTON output bytes are therefore
//!   `[heel_hi, heel_lo, hip_hi, hip_lo]`.
//!
//!   **This is ONE defensible spelling of "nibble-interleaved", chosen
//!   and stated up front because more than one exists.** The alternative
//!   is BIT-level Morton (interleave individual bits of lat/lon rather
//!   than 4-bit groups) — true Z-order, finer-grained locality. The
//!   OGAR passage's own wording ("nibble-interleaved", "a byte's nibbles
//!   are ancestry", "256 = 4⁴") is specifically about NIBBLE granularity
//!   matching a 4-ary centroid hierarchy (2 bits per quaternary tree
//!   level, so one nibble = 2 tree levels), which is the spelling
//!   implemented here. Bit-level Morton was NOT implemented — it is a
//!   different, finer interleave that the same OGAR passage does not
//!   describe for this tile shape.
//! * **CONTROL-BAD** — `[lon_hip, lat_hip, lon_tile, lat_tile]`, the
//!   SHIPPED byte order reversed. Deliberately locality-destroying, named
//!   directly in plan §1.3b's arm table as the control that must lose.
//!
//! # The two metrics (computed over key-order rank, plan §1.3b)
//!
//! 1. **Range count** — the number of maximal runs of CONSECUTIVE rank
//!    values among a box's own cells, computed EXACTLY by sorting the
//!    box's cells' own ranks and counting where consecutive-integer runs
//!    break. This has NO FALSE POSITIVES by construction: `pack` is a
//!    bijection over the whole grid for every arm (proven by
//!    `--selftest`'s "pack() is bijective" check, and argued in
//!    [`RankTable`]'s doc), so rank is dense over `0..total` with no
//!    gaps and no ties. Two box cells with adjacent ranks therefore have
//!    NO other cell — in the box or not — between them in key order, so
//!    a maximal run of consecutive ranks is EXACTLY that run's cells,
//!    never more. This is the opposite of "scan a superset then filter":
//!    nothing here ever reads a cell outside the box.
//! 2. **Neighbour locality** — the median `|rank(a) - rank(b)|` over the
//!    4-neighbourhood (`lat ± 1`, `lon ± 1`; longitude wraps, latitude is
//!    clipped at the poles rather than wrapped — plan §1.3 wrinkle 2),
//!    pooled across a deterministic sample of cells (every
//!    [`SAMPLE_STRIDE`]-th cell in row-major `cell_id` order; see that
//!    constant's doc for the exact rule and why that stride). This metric
//!    is GRID-WIDE, not box-scoped — its own plan definition never
//!    mentions a box — so it is reported once per arm, not once per
//!    (arm, box-kind) pair; the results table therefore has range-count
//!    stats broken out by box-kind and a single neighbour-locality-median
//!    line per arm underneath it.
//!
//! # Determinism
//!
//! Arms are iterated in the fixed order [`ALL_ARMS`], box-kinds in the
//! fixed order [`ALL_KINDS`], and the box set itself is a hardcoded
//! literal ([`box_set`]) always built and printed in the same order.
//! There is no randomness anywhere in this file — no `rand`, no
//! hash-map iteration exposed in output order (the one `HashSet`-shaped
//! check, bijectivity, uses a `Vec<bool>` presence table instead of a
//! set, precisely to avoid iteration-order questions).
//!
//! # Output
//!
//! A human-readable table and the three verdict lines always go to
//! stdout. Pass `--json` to ALSO write the same content as JSON to
//! `layout_probe.json` next to this file
//! (`{CARGO_MANIFEST_DIR}/examples/layout_probe.json`). Pass `--selftest`
//! to run the machinery self-checks in isolation and exit before the real
//! grid comparison runs.
//!
//! Per the brief: this file prints the three verdict lines and the
//! numbers that decided them. It does **not** print an overall verdict
//! sentence, and it does not predict or claim a result — running it and
//! reading the output is the orchestrator's job, not the author's.

#![forbid(unsafe_code)]

use weather_poc::key::{box_ranges, encode_key, LAT_COUNT, LON_COUNT};

// ── Arms ─────────────────────────────────────────────────────────────────

/// The three axis-byte-order arms under comparison. See the module doc
/// for exactly what each one packs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    /// `[lat_tile, lon_tile, lat_hip, lon_hip]` — what `encode_key` ships
    /// today.
    Shipped,
    /// Nibble-interleaved per tier, applied identically to HEEL and HIP.
    Morton,
    /// `[lon_hip, lat_hip, lon_tile, lat_tile]` — SHIPPED reversed.
    ControlBad,
}

/// Fixed iteration order for every arm-indexed loop and every printed
/// table in this file.
const ALL_ARMS: [Arm; 3] = [Arm::Shipped, Arm::Morton, Arm::ControlBad];

impl Arm {
    /// Human-readable name, used in the stdout table and verdict lines.
    fn name(self) -> &'static str {
        match self {
            Arm::Shipped => "SHIPPED",
            Arm::Morton => "MORTON",
            Arm::ControlBad => "CONTROL-BAD",
        }
    }

    /// `snake_case` name, used as a JSON object key.
    fn json_name(self) -> &'static str {
        match self {
            Arm::Shipped => "shipped",
            Arm::Morton => "morton",
            Arm::ControlBad => "control_bad",
        }
    }
}

/// Packs the four SHIPPED axis bytes for `(lat_idx, lon_idx)` — obtained
/// by calling the real [`encode_key`] and reading bytes `4..8` back, NOT
/// by re-deriving the `>> 6` / `& 63` shift/mask by hand. This is the
/// single source of axis-byte truth every arm below packs differently;
/// `classid` is passed as `0` and is irrelevant here (bytes `0..4` are
/// never part of any arm's packing).
fn shipped_axis_bytes(lat_idx: u16, lon_idx: u16) -> [u8; 4] {
    let key = encode_key(0, lat_idx, lon_idx);
    [key[4], key[5], key[6], key[7]]
}

/// Nibble-interleaves two axis bytes (`lat`, `lon`) belonging to the SAME
/// tier into two output bytes, coarsest nibble first, lat's nibble in the
/// high half of each output byte: `[lat_hi:lon_hi, lat_lo:lon_lo]`. See
/// the module doc's "MORTON" section for why this spelling of
/// "nibble-interleaved" was chosen. Fully invertible (hence a bijection
/// on `(lat, lon)` pairs), which is what licenses treating MORTON's rank
/// table as collision-free the same way SHIPPED's is.
fn interleave_nibbles(lat: u8, lon: u8) -> [u8; 2] {
    let lat_hi = lat >> 4;
    let lat_lo = lat & 0x0F;
    let lon_hi = lon >> 4;
    let lon_lo = lon & 0x0F;
    [(lat_hi << 4) | lon_hi, (lat_lo << 4) | lon_lo]
}

/// Interprets four bytes, most-significant first, as the `u32` whose
/// ordinary numeric ordering equals lexicographic byte-array ordering —
/// the equivalence the module doc's "exact, not approximate" claim rests
/// on.
fn be_u32(bytes: [u8; 4]) -> u32 {
    u32::from_be_bytes(bytes)
}

/// Packs `(lat_idx, lon_idx)` into `arm`'s key-order `u32`. All three
/// arms are bijections of the same underlying `(lat_tile, lon_tile,
/// lat_hip, lon_hip)` 4-tuple, which is itself a bijection of
/// `(lat_idx, lon_idx)` (proven by `weather_poc::key`'s own exhaustive
/// round-trip test) — so `pack` is collision-free over the whole grid for
/// every arm, independently re-verified here by `--selftest`.
fn pack(arm: Arm, lat_idx: u16, lon_idx: u16) -> u32 {
    let [lat_tile, lon_tile, lat_hip, lon_hip] = shipped_axis_bytes(lat_idx, lon_idx);
    match arm {
        Arm::Shipped => be_u32([lat_tile, lon_tile, lat_hip, lon_hip]),
        Arm::ControlBad => be_u32([lon_hip, lat_hip, lon_tile, lat_tile]),
        Arm::Morton => {
            let heel = interleave_nibbles(lat_tile, lon_tile);
            let hip = interleave_nibbles(lat_hip, lon_hip);
            be_u32([heel[0], heel[1], hip[0], hip[1]])
        }
    }
}

// ── Cell / rank plumbing ─────────────────────────────────────────────────

/// Row-major cell identifier: `lat_idx * LON_COUNT + lon_idx`. A plain
/// bijection over the grid domain, independent of any arm's byte order —
/// used only to index into per-cell tables, never compared for locality.
fn cell_id(lat_idx: u16, lon_idx: u16) -> u32 {
    lat_idx as u32 * LON_COUNT as u32 + lon_idx as u32
}

/// Inverse of [`cell_id`].
fn cell_from_id(cid: u32) -> (u16, u16) {
    let lat_idx = (cid / LON_COUNT as u32) as u16;
    let lon_idx = (cid % LON_COUNT as u32) as u16;
    (lat_idx, lon_idx)
}

/// A full-grid rank table for one arm: `rank[cell_id(lat, lon)]` is that
/// cell's 0-based position in the ascending sort of `pack(arm, lat,
/// lon)` over ALL 1,038,240 grid cells.
///
/// Built by an EXACT sort of the real generated keys (never an
/// approximate formula) — `pack` is a bijection (see [`pack`]'s doc), so
/// this sort produces exactly the integers `0..total`, once each, with no
/// ties. `--selftest` independently re-checks that claim via
/// [`RankTable::assert_bijective`] rather than merely asserting it in
/// prose.
struct RankTable {
    rank: Vec<u32>,
}

impl RankTable {
    /// Builds the rank table for `arm` by sorting all 1,038,240 packed
    /// keys.
    fn build(arm: Arm) -> Self {
        let total = LAT_COUNT as usize * LON_COUNT as usize;
        let mut entries: Vec<(u32, u32)> = Vec::with_capacity(total);
        for lat_idx in 0..LAT_COUNT {
            for lon_idx in 0..LON_COUNT {
                let packed = pack(arm, lat_idx, lon_idx);
                entries.push((packed, cell_id(lat_idx, lon_idx)));
            }
        }
        entries.sort_unstable_by_key(|&(packed, _)| packed);

        let mut rank = vec![0u32; total];
        for (r, &(_, cid)) in entries.iter().enumerate() {
            rank[cid as usize] = r as u32;
        }
        RankTable { rank }
    }

    /// This cell's key-order rank under the arm this table was built
    /// for.
    fn rank(&self, lat_idx: u16, lon_idx: u16) -> u32 {
        self.rank[cell_id(lat_idx, lon_idx) as usize]
    }

    /// Total number of cells this table covers (always 1,038,240 for the
    /// real grid).
    fn total(&self) -> usize {
        self.rank.len()
    }

    /// `--selftest`-only: asserts every rank value `0..total` is used
    /// EXACTLY once — i.e. that `pack` had no collisions across the
    /// whole grid for this arm. This is the machinery fact "range count
    /// is exact, not approximate" depends on; if this ever fails, the
    /// range-count metric silently stops being exact for that arm and
    /// every downstream number is suspect.
    fn assert_bijective(&self) {
        let mut seen = vec![false; self.rank.len()];
        for &r in &self.rank {
            let idx = r as usize;
            assert!(
                !seen[idx],
                "duplicate rank {r} -- pack() is not a bijection for this arm"
            );
            seen[idx] = true;
        }
    }
}

// ── Boxes ────────────────────────────────────────────────────────────────

/// The four box kinds `D-WXS-2a`'s pre-registered bar (plan §1.3b)
/// requires: "a box set that includes tile-aligned, non-tile-aligned,
/// seam-crossing and pole-adjacent boxes."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoxKind {
    /// Exactly one full 64×64 HEEL tile (interior tiles only — the
    /// ragged last tile on each axis is deliberately excluded here so
    /// this kind stays pure to its own definition; raggedness is not
    /// separately exercised by this probe).
    TileAligned,
    /// Arbitrary offsets and sizes, not aligned to any 64-wide tile
    /// boundary.
    NonTileAligned,
    /// Crosses the 0°/360° longitude seam (`lon_lo >= lon_hi`, decoded
    /// by [`box_ranges`] as a wrap).
    SeamCrossing,
    /// Touches `lat_idx == 0` or `lat_idx == LAT_COUNT - 1`.
    PoleAdjacent,
}

/// Fixed iteration order for every box-kind-indexed loop and printed
/// table.
const ALL_KINDS: [BoxKind; 4] = [
    BoxKind::TileAligned,
    BoxKind::NonTileAligned,
    BoxKind::SeamCrossing,
    BoxKind::PoleAdjacent,
];

impl BoxKind {
    /// Human-readable label, used in the stdout table.
    fn label(self) -> &'static str {
        match self {
            BoxKind::TileAligned => "tile-aligned",
            BoxKind::NonTileAligned => "non-tile-aligned",
            BoxKind::SeamCrossing => "seam-crossing",
            BoxKind::PoleAdjacent => "pole-adjacent",
        }
    }

    /// `snake_case` label, used as a JSON string value.
    fn json_name(self) -> &'static str {
        match self {
            BoxKind::TileAligned => "tile_aligned",
            BoxKind::NonTileAligned => "non_tile_aligned",
            BoxKind::SeamCrossing => "seam_crossing",
            BoxKind::PoleAdjacent => "pole_adjacent",
        }
    }
}

/// One box in the pre-registered box set: `lat_lo..lat_hi`,
/// `lon_lo..lon_hi`, in the same half-open convention
/// [`weather_poc::key::box_ranges`] expects (a wrapping box is expressed
/// as `lon_lo >= lon_hi`).
#[derive(Debug, Clone, Copy)]
struct ProbeBox {
    lat_lo: u16,
    lat_hi: u16,
    lon_lo: u16,
    lon_hi: u16,
}

/// The pre-registered box set (plan §1.3b). Hardcoded and deterministic —
/// printed verbatim by [`print_box_set`] so the exact geometry is
/// auditable, never regenerated per run. Chosen to include several boxes
/// of each kind so a median over the non-tile-aligned kind (the primary
/// bar's own metric) is meaningful, not a single sample.
fn box_set() -> Vec<(BoxKind, ProbeBox)> {
    vec![
        // Tile-aligned: exactly one full 64x64 HEEL tile, interior tiles
        // (lat_tile 1..=10 are full 64-row tiles; lon_tile 0..=21 are
        // full 64-col tiles). Four different positions across the grid.
        (
            BoxKind::TileAligned,
            ProbeBox {
                lat_lo: 128,
                lat_hi: 192,
                lon_lo: 0,
                lon_hi: 64,
            },
        ),
        (
            BoxKind::TileAligned,
            ProbeBox {
                lat_lo: 320,
                lat_hi: 384,
                lon_lo: 640,
                lon_hi: 704,
            },
        ),
        (
            BoxKind::TileAligned,
            ProbeBox {
                lat_lo: 512,
                lat_hi: 576,
                lon_lo: 1344,
                lon_hi: 1408,
            },
        ),
        (
            BoxKind::TileAligned,
            ProbeBox {
                lat_lo: 192,
                lat_hi: 256,
                lon_lo: 960,
                lon_hi: 1024,
            },
        ),
        // Non-tile-aligned: arbitrary offsets/sizes, several shapes
        // (small, medium, large, thin-tall, thin-wide, boundary-crossing,
        // a 1-column sliver).
        (
            BoxKind::NonTileAligned,
            ProbeBox {
                lat_lo: 10,
                lat_hi: 37,
                lon_lo: 5,
                lon_hi: 53,
            },
        ),
        (
            BoxKind::NonTileAligned,
            ProbeBox {
                lat_lo: 100,
                lat_hi: 233,
                lon_lo: 200,
                lon_hi: 410,
            },
        ),
        (
            BoxKind::NonTileAligned,
            ProbeBox {
                lat_lo: 50,
                lat_hi: 521,
                lon_lo: 33,
                lon_hi: 999,
            },
        ),
        (
            BoxKind::NonTileAligned,
            ProbeBox {
                lat_lo: 400,
                lat_hi: 405,
                lon_lo: 700,
                lon_hi: 1300,
            },
        ),
        (
            BoxKind::NonTileAligned,
            ProbeBox {
                lat_lo: 20,
                lat_hi: 680,
                lon_lo: 900,
                lon_hi: 905,
            },
        ),
        (
            // Deliberately offset by 1 from tile boundaries on both
            // axes (65/129, not 64/128) so it genuinely crosses two
            // tiles per axis without being tile-aligned.
            BoxKind::NonTileAligned,
            ProbeBox {
                lat_lo: 65,
                lat_hi: 129,
                lon_lo: 63,
                lon_hi: 129,
            },
        ),
        (
            BoxKind::NonTileAligned,
            ProbeBox {
                lat_lo: 200,
                lat_hi: 205,
                lon_lo: 1,
                lon_hi: 2,
            },
        ),
        (
            BoxKind::NonTileAligned,
            ProbeBox {
                lat_lo: 300,
                lat_hi: 450,
                lon_lo: 1100,
                lon_hi: 1150,
            },
        ),
        // Seam-crossing: lon_lo >= lon_hi, decoded by box_ranges as a
        // wrap around 0deg/360deg. Includes a minimal 2-column wrap.
        (
            BoxKind::SeamCrossing,
            ProbeBox {
                lat_lo: 100,
                lat_hi: 200,
                lon_lo: 1400,
                lon_hi: 40,
            },
        ),
        (
            BoxKind::SeamCrossing,
            ProbeBox {
                lat_lo: 300,
                lat_hi: 350,
                lon_lo: 1430,
                lon_hi: 10,
            },
        ),
        (
            // Minimal wrap: exactly one column on each side of the seam.
            BoxKind::SeamCrossing,
            ProbeBox {
                lat_lo: 500,
                lat_hi: 600,
                lon_lo: 1439,
                lon_hi: 1,
            },
        ),
        (
            BoxKind::SeamCrossing,
            ProbeBox {
                lat_lo: 50,
                lat_hi: 70,
                lon_lo: 1300,
                lon_hi: 100,
            },
        ),
        // Pole-adjacent: touches lat_idx == 0 or lat_idx == LAT_COUNT-1.
        // Includes the full-width polar row as a stress case.
        (
            BoxKind::PoleAdjacent,
            ProbeBox {
                lat_lo: 0,
                lat_hi: 30,
                lon_lo: 100,
                lon_hi: 200,
            },
        ),
        (
            BoxKind::PoleAdjacent,
            ProbeBox {
                lat_lo: 700,
                lat_hi: 721,
                lon_lo: 500,
                lon_hi: 600,
            },
        ),
        (
            // The entire north-pole row: one physical point repeated
            // LON_COUNT times (plan §1.3 wrinkle 2's pole degeneracy),
            // width 1440.
            BoxKind::PoleAdjacent,
            ProbeBox {
                lat_lo: 0,
                lat_hi: 1,
                lon_lo: 0,
                lon_hi: LON_COUNT,
            },
        ),
        (
            BoxKind::PoleAdjacent,
            ProbeBox {
                lat_lo: 715,
                lat_hi: 721,
                lon_lo: 900,
                lon_hi: 950,
            },
        ),
    ]
}

/// Enumerates every `(lat_idx, lon_idx)` cell in a (possibly
/// seam-crossing) box, by delegating the wrap decomposition to
/// [`box_ranges`] and flattening its non-wrapping pieces. A seam-crossing
/// box's two pieces never overlap (by `box_ranges`'s own contract), so
/// this never double-counts a cell.
fn box_cells(lat_lo: u16, lat_hi: u16, lon_lo: u16, lon_hi: u16) -> Vec<(u16, u16)> {
    let mut cells = Vec::new();
    for piece in box_ranges(lat_lo, lat_hi, lon_lo, lon_hi) {
        for lat_idx in piece.lat_lo..piece.lat_hi {
            for lon_idx in piece.lon_lo..piece.lon_hi {
                cells.push((lat_idx, lon_idx));
            }
        }
    }
    cells
}

/// Metric 1: the number of maximal runs of consecutive rank values among
/// `cells`' own ranks under `rank`. Exact, not approximate — see the
/// module doc and [`RankTable`]'s doc for why.
fn range_count(rank: &RankTable, cells: &[(u16, u16)]) -> usize {
    assert!(
        !cells.is_empty(),
        "range_count of an empty cell set is undefined"
    );
    let mut ranks: Vec<u32> = cells
        .iter()
        .map(|&(lat, lon)| rank.rank(lat, lon))
        .collect();
    ranks.sort_unstable();

    let mut runs = 1usize;
    for w in ranks.windows(2) {
        if w[1] != w[0] + 1 {
            runs += 1;
        }
    }
    runs
}

// ── Neighbour locality ───────────────────────────────────────────────────

/// Deterministic sample stride for the neighbour-locality metric: every
/// `SAMPLE_STRIDE`-th cell in row-major `cell_id` order (`0`,
/// `SAMPLE_STRIDE`, `2*SAMPLE_STRIDE`, ... up to `total`). `677` is prime
/// and does not divide `64` (the tile width), `1440` (the row width), or
/// `721` (the column height) — so successive sampled `cell_id`s do not
/// repeatedly land on the same phase within a tile or a row, which a
/// stride sharing a factor with 64 (e.g. any multiple of 64) would risk.
/// Yields `floor((1_038_240 - 1) / 677) + 1 = 1534` sample cells (the
/// "-1, +1" is because the first sample is index `0`, not `677`; verified
/// by hand, not `1_038_240 / 677 = 1533` which undercounts by exactly the
/// starting-at-zero term). The exact count is also printed at runtime and,
/// under `--json`, recorded, rather than trusted from this comment alone.
const SAMPLE_STRIDE: u32 = 677;

/// The deterministic sample of `cell_id`s the neighbour-locality metric
/// pools its distances over.
fn sample_cell_ids(total: u32) -> Vec<u32> {
    (0..total).step_by(SAMPLE_STRIDE as usize).collect()
}

/// Wraps `lon_idx + delta` (`delta` is `-1` or `1` at every call site)
/// around a cyclic axis of the given `width`, using Euclidean
/// (always-non-negative) modulo. At `width == 1` both `delta = -1` and
/// `delta = 1` wrap back to the only column that exists — see
/// `--selftest`'s degenerate-axis check.
fn wrap_lon(lon_idx: u16, delta: i32, width: u16) -> u16 {
    let w = width as i32;
    let l = lon_idx as i32;
    (((l + delta) % w + w) % w) as u16
}

/// Pushes `|rank(a) - rank(b)|` onto `out`.
fn push_distance(
    out: &mut Vec<u64>,
    rank: &RankTable,
    a_lat: u16,
    a_lon: u16,
    b_lat: u16,
    b_lon: u16,
) {
    let ra = rank.rank(a_lat, a_lon) as i64;
    let rb = rank.rank(b_lat, b_lon) as i64;
    out.push(ra.abs_diff(rb));
}

/// Metric 2's raw pooled sample: for every cell in `sample_ids`, the
/// `|rank(a) - rank(b)|` distance to each of its up-to-4 neighbours
/// (`lat ± 1` skipped, not wrapped, at the poles; `lon ± 1` always
/// wraps). Distances from all sampled cells are pooled into one flat
/// list — the median of THIS list is metric 2, per the module doc.
fn neighbour_distances(rank: &RankTable, sample_ids: &[u32]) -> Vec<u64> {
    let mut distances = Vec::new();
    for &cid in sample_ids {
        let (lat_idx, lon_idx) = cell_from_id(cid);

        // lat +/- 1: skip (never wrap) when off the grid -- plan §1.3
        // wrinkle 2, latitude has poles, not a seam.
        if lat_idx > 0 {
            push_distance(&mut distances, rank, lat_idx, lon_idx, lat_idx - 1, lon_idx);
        }
        if lat_idx + 1 < LAT_COUNT {
            push_distance(&mut distances, rank, lat_idx, lon_idx, lat_idx + 1, lon_idx);
        }

        // lon +/- 1: always wraps.
        let lon_minus = wrap_lon(lon_idx, -1, LON_COUNT);
        let lon_plus = wrap_lon(lon_idx, 1, LON_COUNT);
        push_distance(&mut distances, rank, lat_idx, lon_idx, lat_idx, lon_minus);
        push_distance(&mut distances, rank, lat_idx, lon_idx, lat_idx, lon_plus);
    }
    distances
}

// ── Medians ──────────────────────────────────────────────────────────────

/// Median of a non-empty `usize` slice (sorted in place); even-length
/// slices average the two middle values.
fn median_usize(values: &mut [usize]) -> f64 {
    assert!(!values.is_empty(), "median of an empty slice is undefined");
    values.sort_unstable();
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2] as f64
    } else {
        (values[n / 2 - 1] as f64 + values[n / 2] as f64) / 2.0
    }
}

/// Median of a non-empty `u64` slice (sorted in place); even-length
/// slices average the two middle values.
fn median_u64(values: &mut [u64]) -> f64 {
    assert!(!values.is_empty(), "median of an empty slice is undefined");
    values.sort_unstable();
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2] as f64
    } else {
        (values[n / 2 - 1] as f64 + values[n / 2] as f64) / 2.0
    }
}

// ── Report assembly ──────────────────────────────────────────────────────

/// Range-count statistics for one (arm, box-kind) pair.
struct BoxKindStats {
    kind: BoxKind,
    /// Each box's own range count, in [`box_set`]'s order (NOT sorted) —
    /// so the printed/JSON output stays auditable against the box list.
    per_box_range_counts: Vec<usize>,
    range_count_median: f64,
    range_count_max: usize,
}

/// Full report for one arm: range-count stats broken out by box-kind,
/// plus the single grid-wide neighbour-locality median.
struct ArmReport {
    arm: Arm,
    per_kind: Vec<BoxKindStats>,
    neighbour_locality_median: f64,
}

fn kind_stats(report: &ArmReport, kind: BoxKind) -> &BoxKindStats {
    report
        .per_kind
        .iter()
        .find(|k| k.kind == kind)
        .expect("every ArmReport carries all four box kinds")
}

fn arm_report(reports: &[ArmReport], arm: Arm) -> &ArmReport {
    reports
        .iter()
        .find(|r| r.arm == arm)
        .expect("reports always cover all three arms")
}

/// `--selftest`-only: looks up the already-built [`RankTable`] for `arm`
/// out of a `(Arm, RankTable)` list. A plain function (not a closure) so
/// its return-borrow lifetime is elided against `tables` the ordinary,
/// unambiguous way -- a closure doing the same lookup would tie its
/// return type to the closure's own captured-environment lifetime, which
/// is a well-known source of "cannot escape closure body" errors for
/// exactly this shape and was deliberately avoided here.
fn find_rank(tables: &[(Arm, RankTable)], arm: Arm) -> &RankTable {
    &tables
        .iter()
        .find(|(a, _)| *a == arm)
        .expect("tables always cover all three arms")
        .1
}

/// Builds the full report set: for each arm (fixed [`ALL_ARMS`] order),
/// builds its rank table once, then computes range-count stats per box
/// kind (fixed [`ALL_KINDS`] order, each box in [`box_set`]'s order) and
/// the grid-wide neighbour-locality median.
fn build_reports(boxes: &[(BoxKind, ProbeBox)], sample_ids: &[u32]) -> Vec<ArmReport> {
    let mut reports = Vec::with_capacity(ALL_ARMS.len());
    for &arm in &ALL_ARMS {
        let rank = RankTable::build(arm);

        let mut per_kind = Vec::with_capacity(ALL_KINDS.len());
        for &kind in &ALL_KINDS {
            let counts: Vec<usize> = boxes
                .iter()
                .filter(|(k, _)| *k == kind)
                .map(|(_, b)| {
                    let cells = box_cells(b.lat_lo, b.lat_hi, b.lon_lo, b.lon_hi);
                    range_count(&rank, &cells)
                })
                .collect();

            let max = counts
                .iter()
                .copied()
                .max()
                .expect("box_set has boxes of every kind");
            let mut sorted = counts.clone();
            let median = median_usize(&mut sorted);

            per_kind.push(BoxKindStats {
                kind,
                per_box_range_counts: counts,
                range_count_median: median,
                range_count_max: max,
            });
        }

        let mut distances = neighbour_distances(&rank, sample_ids);
        let neighbour_locality_median = median_u64(&mut distances);

        reports.push(ArmReport {
            arm,
            per_kind,
            neighbour_locality_median,
        });
    }
    reports
}

// ── Verdicts ─────────────────────────────────────────────────────────────

/// One of the three pre-registered verdicts (plan §1.3b): `name`,
/// whether it `pass`ed, and the `detail` string carrying the exact
/// numbers that decided it (printed on stdout and embedded in the JSON
/// output, never a bare PASS/FAIL with no numbers attached).
struct Verdict {
    name: &'static str,
    pass: bool,
    detail: String,
}

/// Computes the three pre-registered verdicts against ALREADY-BUILT
/// reports. Does not print or predict anything — a pure function of the
/// measured numbers.
fn compute_verdicts(reports: &[ArmReport]) -> Vec<Verdict> {
    let shipped = arm_report(reports, Arm::Shipped);
    let morton = arm_report(reports, Arm::Morton);
    let control = arm_report(reports, Arm::ControlBad);

    let shipped_nta = kind_stats(shipped, BoxKind::NonTileAligned);
    let morton_nta = kind_stats(morton, BoxKind::NonTileAligned);
    let control_nta = kind_stats(control, BoxKind::NonTileAligned);

    // Primary: MORTON beats SHIPPED on both metrics -- strictly fewer
    // ranges on the median non-tile-aligned box, AND strictly smaller
    // median neighbour distance.
    let primary_pass = morton_nta.range_count_median < shipped_nta.range_count_median
        && morton.neighbour_locality_median < shipped.neighbour_locality_median;
    let primary_detail = format!(
        "morton_rc_median={:.2} vs shipped_rc_median={:.2} (non-tile-aligned); \
         morton_nb_median={:.2} vs shipped_nb_median={:.2}",
        morton_nta.range_count_median,
        shipped_nta.range_count_median,
        morton.neighbour_locality_median,
        shipped.neighbour_locality_median,
    );

    // Control that can lose: CONTROL-BAD must be worse than BOTH
    // SHIPPED and MORTON on BOTH metrics.
    let control_pass = control_nta.range_count_median > shipped_nta.range_count_median
        && control_nta.range_count_median > morton_nta.range_count_median
        && control.neighbour_locality_median > shipped.neighbour_locality_median
        && control.neighbour_locality_median > morton.neighbour_locality_median;
    let control_detail = format!(
        "control_rc_median={:.2} vs shipped={:.2}/morton={:.2} (non-tile-aligned); \
         control_nb_median={:.2} vs shipped={:.2}/morton={:.2}",
        control_nta.range_count_median,
        shipped_nta.range_count_median,
        morton_nta.range_count_median,
        control.neighbour_locality_median,
        shipped.neighbour_locality_median,
        morton.neighbour_locality_median,
    );

    // Stay-silent twin (non-trivial): on EVERY tile-aligned box, SHIPPED
    // and MORTON must produce exactly one range each -- identical, and
    // checked box-by-box, not just on the median.
    let shipped_ta = kind_stats(shipped, BoxKind::TileAligned);
    let morton_ta = kind_stats(morton, BoxKind::TileAligned);
    let twin_pass = shipped_ta.per_box_range_counts.iter().all(|&c| c == 1)
        && morton_ta.per_box_range_counts.iter().all(|&c| c == 1);
    let twin_detail = format!(
        "shipped_tile_aligned_counts={:?}; morton_tile_aligned_counts={:?} \
         (every entry must be 1)",
        shipped_ta.per_box_range_counts, morton_ta.per_box_range_counts,
    );

    vec![
        Verdict {
            name: "primary",
            pass: primary_pass,
            detail: primary_detail,
        },
        Verdict {
            name: "control",
            pass: control_pass,
            detail: control_detail,
        },
        Verdict {
            name: "stay-silent twin",
            pass: twin_pass,
            detail: twin_detail,
        },
    ]
}

// ── Printing ─────────────────────────────────────────────────────────────

fn print_box_set(boxes: &[(BoxKind, ProbeBox)]) {
    println!(
        "-- box set (pre-registered, plan sec 1.3b, {} boxes) --",
        boxes.len()
    );
    for (kind, b) in boxes {
        println!(
            "  {:<18} lat {:>4}..{:<4} lon {:>4}..{:<4}",
            kind.label(),
            b.lat_lo,
            b.lat_hi,
            b.lon_lo,
            b.lon_hi
        );
    }
    println!();
}

fn print_table(reports: &[ArmReport]) {
    println!("== layout_probe: SHIPPED vs MORTON vs CONTROL-BAD (D-WXS-2a half A) ==");
    println!();
    println!(
        "{:<14} {:<18} {:>12} {:>10} {:>9}",
        "arm", "box-kind", "rc_median", "rc_max", "n_boxes"
    );
    for r in reports {
        for k in &r.per_kind {
            println!(
                "{:<14} {:<18} {:>12.2} {:>10} {:>9}",
                r.arm.name(),
                k.kind.label(),
                k.range_count_median,
                k.range_count_max,
                k.per_box_range_counts.len(),
            );
        }
    }
    println!();
    println!("{:<14} {:>26}", "arm", "neighbour_locality_median");
    for r in reports {
        println!("{:<14} {:>26.4}", r.arm.name(), r.neighbour_locality_median);
    }
    println!();
}

fn print_verdicts(verdicts: &[Verdict]) {
    for v in verdicts {
        println!(
            "{}: {} ({})",
            v.name,
            if v.pass { "PASS" } else { "FAIL" },
            v.detail
        );
    }
}

// ── JSON ─────────────────────────────────────────────────────────────────

/// Absolute path this file's `--json` mode writes to, resolved at compile
/// time from `CARGO_MANIFEST_DIR` so the output location does not depend
/// on the working directory `cargo run --example` happens to be invoked
/// from.
const JSON_OUTPUT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/layout_probe.json");

/// Hand-rolled JSON emission (this crate is zero-dep by construction, see
/// `Cargo.toml`) of exactly the same content [`print_table`] and
/// [`print_verdicts`] print to stdout. No escaping is needed anywhere in
/// this output: every string value is a fixed, quote-free label from this
/// file's own enums, and every numeric field is a plain integer or `f64`.
fn write_json(boxes: &[(BoxKind, ProbeBox)], reports: &[ArmReport], verdicts: &[Verdict]) {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"arms\": [\"shipped\", \"morton\", \"control_bad\"],\n");

    out.push_str("  \"sample_stride\": ");
    out.push_str(&SAMPLE_STRIDE.to_string());
    out.push_str(",\n");

    out.push_str("  \"box_set\": [\n");
    for (i, (kind, b)) in boxes.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"kind\": \"{}\", \"lat_lo\": {}, \"lat_hi\": {}, \"lon_lo\": {}, \"lon_hi\": {}}}{}\n",
            kind.json_name(),
            b.lat_lo,
            b.lat_hi,
            b.lon_lo,
            b.lon_hi,
            if i + 1 == boxes.len() { "" } else { "," }
        ));
    }
    out.push_str("  ],\n");

    out.push_str("  \"results\": {\n");
    for (ri, r) in reports.iter().enumerate() {
        out.push_str(&format!("    \"{}\": {{\n", r.arm.json_name()));
        out.push_str("      \"by_kind\": {\n");
        for (ki, k) in r.per_kind.iter().enumerate() {
            out.push_str(&format!(
                "        \"{}\": {{\"range_counts\": {:?}, \"median\": {:.4}, \"max\": {}}}{}\n",
                k.kind.json_name(),
                k.per_box_range_counts,
                k.range_count_median,
                k.range_count_max,
                if ki + 1 == r.per_kind.len() { "" } else { "," }
            ));
        }
        out.push_str("      },\n");
        out.push_str(&format!(
            "      \"neighbour_locality_median\": {:.4}\n",
            r.neighbour_locality_median
        ));
        out.push_str(&format!(
            "    }}{}\n",
            if ri + 1 == reports.len() { "" } else { "," }
        ));
    }
    out.push_str("  },\n");

    out.push_str("  \"verdicts\": [\n");
    for (i, v) in verdicts.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"name\": \"{}\", \"pass\": {}, \"detail\": \"{}\"}}{}\n",
            v.name,
            v.pass,
            v.detail,
            if i + 1 == verdicts.len() { "" } else { "," }
        ));
    }
    out.push_str("  ]\n");

    out.push_str("}\n");

    std::fs::write(JSON_OUTPUT_PATH, out)
        .unwrap_or_else(|e| panic!("failed to write {JSON_OUTPUT_PATH}: {e}"));
    println!("wrote {JSON_OUTPUT_PATH}");
}

// ── Selftest ─────────────────────────────────────────────────────────────

/// Machinery self-checks with known answers, run in isolation from the
/// real box-set comparison. Per the brief: if an assertion here is
/// expected to hold and fails, that is a finding about the instrument,
/// reported by letting the assertion panic -- never silently adjusted to
/// pass.
fn run_selftest() {
    println!("== layout_probe --selftest ==");

    // (0) MACHINERY: pack() is a bijection over the whole grid for every
    // arm. This is what licenses "consecutive rank => no cell in
    // between" as exact rather than approximate (module doc, metric 1;
    // RankTable's own doc). Built once per arm and kept for the checks
    // below, rather than rebuilt, so the ~1,038,240-element sort only
    // runs once per arm even in selftest mode.
    let mut tables: Vec<(Arm, RankTable)> = Vec::with_capacity(ALL_ARMS.len());
    for &arm in &ALL_ARMS {
        let rank = RankTable::build(arm);
        assert_eq!(
            rank.total(),
            1_038_240,
            "{}: grid cell count drifted from the plan",
            arm.name()
        );
        rank.assert_bijective();
        println!(
            "  [ok] {}: pack() is bijective over all 1,038,240 cells",
            arm.name()
        );
        tables.push((arm, rank));
    }

    // (1) A 1-cell box is exactly 1 range under every arm -- trivially:
    // a singleton set is one run regardless of byte order.
    for &arm in &ALL_ARMS {
        let cells = box_cells(300, 301, 300, 301);
        assert_eq!(cells.len(), 1, "expected exactly one cell");
        let rc = range_count(find_rank(&tables, arm), &cells);
        assert_eq!(
            rc,
            1,
            "{}: a 1-cell box must be exactly 1 range",
            arm.name()
        );
    }
    println!("  [ok] 1-cell box is exactly 1 range under every arm");

    // (2) A box covering the WHOLE grid is exactly 1 range under every
    // arm -- true for ANY bijective arm, independent of byte order: the
    // sorted ranks of ALL cells are simply 0..total, one contiguous run.
    for &arm in &ALL_ARMS {
        let cells = box_cells(0, LAT_COUNT, 0, LON_COUNT);
        assert_eq!(cells.len(), 1_038_240, "expected the whole grid");
        let rc = range_count(find_rank(&tables, arm), &cells);
        assert_eq!(
            rc,
            1,
            "{}: the whole-grid box must be exactly 1 range",
            arm.name()
        );
    }
    println!("  [ok] whole-grid box is exactly 1 range under every arm");

    // (3) A full longitude row is NOT 1 range under SHIPPED -- worked
    // out by hand (module doc + this comment), not assumed:
    //
    // SHIPPED's byte order is [lat_tile, lon_tile, lat_hip, lon_hip].
    // For a single row, lat_tile and lat_hip are BOTH fixed. The
    // SECOND-most-significant byte is lon_tile, which sorts BEFORE
    // lat_hip -- so cells are grouped first by lon_tile (shared across
    // ALL 64 rows of the HEEL tile), and only WITHIN one lon_tile group
    // are they further sorted by lat_hip (which row) then lon_hip
    // (position in the row). That means our row's own cells, for a
    // fixed lon_tile, form one contiguous slice inside that lon_tile's
    // group -- sandwiched between OTHER rows' cells that share the same
    // lon_tile. So the row scatters into exactly one run PER DISTINCT
    // lon_tile value it touches. A full row spans lon_tile 0..=22
    // (ceil(1440/64) = 23 distinct values, the last one ragged at 32
    // columns) => 23 runs, not 1.
    {
        let cells = box_cells(200, 201, 0, LON_COUNT);
        assert_eq!(cells.len(), 1440, "expected one full longitude row");
        let rc = range_count(find_rank(&tables, Arm::Shipped), &cells);
        assert_eq!(
            rc, 23,
            "a full longitude row must be 23 ranges under SHIPPED \
             (ceil(1440/64) distinct lon_tile values), NOT 1 -- see the \
             comment above this assertion for the byte-order argument"
        );
    }
    println!("  [ok] full longitude row under SHIPPED is 23 ranges (worked out, NOT 1)");

    // (4) The neighbour metric degenerates predictably at axis width 1:
    // wrap_lon's Euclidean modulo sends BOTH lon-1 and lon+1 back to the
    // only column that exists, so the "neighbour" is the cell itself --
    // a predictable distance of 0, never a panic or a nonsensical value.
    assert_eq!(
        wrap_lon(0, -1, 1),
        0,
        "width-1 axis: lon-1 must wrap to the only column"
    );
    assert_eq!(
        wrap_lon(0, 1, 1),
        0,
        "width-1 axis: lon+1 must wrap to the only column"
    );
    println!("  [ok] neighbour wrap degenerates predictably (self, distance 0) at axis width 1");

    println!("== all selftest assertions passed ==");
}

// ── main ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let selftest = args.iter().any(|a| a == "--selftest");
    let emit_json = args.iter().any(|a| a == "--json");

    if selftest {
        run_selftest();
        return;
    }

    let boxes = box_set();
    print_box_set(&boxes);

    let total = LAT_COUNT as u32 * LON_COUNT as u32;
    let sample_ids = sample_cell_ids(total);
    println!(
        "neighbour-locality sample: stride={SAMPLE_STRIDE}, n_samples={}",
        sample_ids.len()
    );
    println!();

    let reports = build_reports(&boxes, &sample_ids);
    print_table(&reports);

    let verdicts = compute_verdicts(&reports);
    print_verdicts(&verdicts);

    if emit_json {
        write_json(&boxes, &reports, &verdicts);
    }
}
