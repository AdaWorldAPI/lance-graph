//! Grid cell (lat_idx, lon_idx) <-> the canonical 16-byte `NodeGuid` key.
//!
//! Owner: the key worker (plan `.claude/plans/weather-soa-bake-v1.md` §6.2,
//! deliverable D-WXS-2). Grid: ERA5 0.25 degree, pole-inclusive latitude
//! (`lat_idx` 0..721), cyclic longitude (`lon_idx` 0..1440).
//!
//! # Byte layout
//!
//! This crate is deliberately zero-dep (`Cargo.toml`'s "ZERO-DEP BY
//! CONSTRUCTION" note) and therefore has no dependency on
//! `lance-graph-contract`'s `NodeGuid` type. [`encode_key`] instead emits a
//! raw `[u8; 16]` in the **same byte layout** `NodeGuid` uses (`CLAUDE.md`
//! § "CANON -- Minimal SoA node"), little-endian throughout:
//!
//! | bytes | content |
//! |---|---|
//! | `0..4`  | `classid: u32`, little-endian -- a **parameter this module accepts**; it is never composed, minted, or bit-manipulated here (`D-WXS-0`, the classid mint, is blocked on an OGAR-side decision -- plan §1.4) |
//! | `4`     | HEEL, lat axis: `(lat_idx >> 6) as u8` |
//! | `5`     | HEEL, lon axis: `(lon_idx >> 6) as u8` |
//! | `6`     | HIP, lat axis: `(lat_idx & 63) as u8` |
//! | `7`     | HIP, lon axis: `(lon_idx & 63) as u8` |
//! | `8..10` | TWIG -- `0u16`, dormant/reserved (zero-fallback: "not consulted", never "compacted") |
//! | `10..16`| untouched (all zero) -- the V1-legacy tail; this crate mints nothing there |
//!
//! **Agreement between this layout and the real `NodeGuid` (`canonical_node.rs`)
//! is NOT verified by this module.** That is a separate, cross-manifest parity
//! deliverable of the same shape as `D-WXS-12` (jc <-> ndarray) -- a measured
//! comparison run across both crates, not a dependency edge from this
//! (deliberately dependency-free) crate.
//!
//! # The three wrinkles this module implements (plan §1.3)
//!
//! 1. **Ragged tiles, never padded.** `721 = 11*64 + 17` and
//!    `1440 / 64 = 22.5`: the last tile on each axis is partial (17 lat rows,
//!    32 lon columns). Cells are addressed by index with half-open ranges;
//!    nothing is padded to a full tile, because a padded cell would be a
//!    cell that does not exist.
//! 2. **Longitude is cyclic, latitude is not.** A box that crosses the
//!    0deg/360deg seam is decomposed into two ranges by [`box_ranges`], never
//!    approximated as one. Latitude has no such wrap -- it has poles instead
//!    (a pole row is one physical point repeated `LON_COUNT` times, which is
//!    a downstream statistics concern, not a key-encoding one).
//! 3. **No `MailboxId`/`NiblePath` here.** That mapping is unresolved
//!    elsewhere in the workspace (`le-contract.md` §5) and this module does
//!    not lean on it.

/// Number of latitude rows in the ERA5 0.25 degree, pole-inclusive grid
/// (`720 + 1`). `lat_idx` ranges over `0..LAT_COUNT`.
///
/// `721 = 11 * 64 + 17`: the grid does not tile evenly at the 64-row tile
/// size, so the last latitude tile (HEEL = 11) is ragged, holding only the
/// 17 rows `704..721` (plan §1.3 wrinkle 1).
pub const LAT_COUNT: u16 = 721;

/// Number of longitude columns in the ERA5 0.25 degree grid. `lon_idx`
/// ranges over `0..LON_COUNT` and is cyclic (wraps modulo `LON_COUNT`).
///
/// `1440 / 64 = 22.5`: the last longitude tile (HEEL = 22) is ragged,
/// holding only the 32 columns `1408..1440` (plan §1.3 wrinkle 1).
pub const LON_COUNT: u16 = 1440;

/// Length in bytes of the canonical `NodeGuid` key this module emits.
pub const KEY_LEN: usize = 16;

/// The HEEL/HIP split point on each axis: `HEEL = idx >> TILE_SHIFT`,
/// `HIP = idx & TILE_MASK`. A 64-wide tile per axis.
const TILE_SHIFT: u16 = 6;

/// `(1 << TILE_SHIFT) - 1` -- the low-bits mask recovering the HIP (within
/// tile) component of a grid index.
const TILE_MASK: u16 = (1 << TILE_SHIFT) - 1;

/// Encodes a grid cell `(lat_idx, lon_idx)` into the canonical 16-byte key,
/// under the caller-supplied `classid`.
///
/// `classid` is stored verbatim as little-endian bytes -- this module never
/// composes, mints, or bit-manipulates it (`D-WXS-0` is blocked; see the
/// module doc's byte-layout table).
///
/// # Preconditions
///
/// `lat_idx < LAT_COUNT` and `lon_idx < LON_COUNT`. Violating this is a
/// caller bug: in debug builds it trips a `debug_assert`; in release builds
/// the HEEL/HIP split still runs (silently producing a key that will not
/// round-trip through [`decode_key`], since the out-of-range guard lives on
/// the decode side, not here). Callers that need to construct or inspect an
/// out-of-range key directly (e.g. to test rejection) should build the
/// `[u8; 16]` by hand rather than calling this function with an
/// out-of-range index.
pub fn encode_key(classid: u32, lat_idx: u16, lon_idx: u16) -> [u8; KEY_LEN] {
    debug_assert!(
        lat_idx < LAT_COUNT,
        "lat_idx {lat_idx} out of range (0..{LAT_COUNT})"
    );
    debug_assert!(
        lon_idx < LON_COUNT,
        "lon_idx {lon_idx} out of range (0..{LON_COUNT})"
    );

    let mut key = [0u8; KEY_LEN];
    key[0..4].copy_from_slice(&classid.to_le_bytes());
    key[4] = (lat_idx >> TILE_SHIFT) as u8;
    key[5] = (lon_idx >> TILE_SHIFT) as u8;
    key[6] = (lat_idx & TILE_MASK) as u8;
    key[7] = (lon_idx & TILE_MASK) as u8;
    // bytes 8..10 (TWIG) stay 0 -- dormant, reserved.
    // bytes 10..16 stay 0 -- V1-legacy tail; this crate mints nothing there.
    key
}

/// Decodes a canonical 16-byte key back into its grid cell `(lat_idx,
/// lon_idx)`, or `None` if the encoded cell falls outside the real grid
/// (`0..LAT_COUNT` x `0..LON_COUNT`).
///
/// The bound check is deliberately on the *reconstructed* index
/// (`heel << TILE_SHIFT | hip`), not on the HEEL/HIP bytes individually --
/// this is what makes an out-of-range cell (e.g. `lat_idx == LAT_COUNT`,
/// one past the last valid row) come back `None` instead of silently
/// wrapping or aliasing onto a real cell. `classid` and the TWIG/tail bytes
/// are not inspected; only the lat/lon axes are decoded.
pub fn decode_key(key: &[u8; KEY_LEN]) -> Option<(u16, u16)> {
    let heel_lat = key[4] as u16;
    let heel_lon = key[5] as u16;
    let hip_lat = key[6] as u16;
    let hip_lon = key[7] as u16;

    let lat_idx = (heel_lat << TILE_SHIFT) | hip_lat;
    let lon_idx = (heel_lon << TILE_SHIFT) | hip_lon;

    if lat_idx >= LAT_COUNT || lon_idx >= LON_COUNT {
        return None;
    }
    Some((lat_idx, lon_idx))
}

/// A rectangular, half-open, non-wrapping box of grid cells:
/// `lat_lo..lat_hi` and `lon_lo..lon_hi`.
///
/// The only place a longitude wrap is resolved is [`box_ranges`] -- every
/// `CellBox` it hands back already excludes the wrap; a `CellBox` value is
/// never itself wrapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellBox {
    /// Inclusive lower latitude index.
    pub lat_lo: u16,
    /// Exclusive upper latitude index.
    pub lat_hi: u16,
    /// Inclusive lower longitude index.
    pub lon_lo: u16,
    /// Exclusive upper longitude index.
    pub lon_hi: u16,
}

/// Splits a lat/lon box into the non-wrapping [`CellBox`] range(s) that
/// cover it (plan §1.3 wrinkle 2: "a box crossing the 0deg/360deg seam is
/// two ranges, never one").
///
/// `lat_lo..lat_hi` is passed through unchanged on every returned box --
/// latitude never wraps (it has poles, not a seam).
///
/// `lon_lo` and `lon_hi` are both grid indices in `0..=LON_COUNT`:
///
/// * If `lon_lo < lon_hi`, the box does not cross the seam and this returns
///   **exactly one** `CellBox` covering `lon_lo..lon_hi`.
/// * If `lon_lo >= lon_hi`, the box wraps (the caller is expressing "from
///   `lon_lo` around through 0deg to `lon_hi`") and this returns **exactly
///   two**: `lon_lo..LON_COUNT`, then `0..lon_hi`.
///
/// The return type is deliberate: a `Vec` (never an `Option<CellBox>`) so a
/// caller cannot silently drop the second range the way an optional-single-
/// range API would let them -- callers must handle however many ranges come
/// back, and `.len()` itself reports whether the box wrapped.
pub fn box_ranges(lat_lo: u16, lat_hi: u16, lon_lo: u16, lon_hi: u16) -> Vec<CellBox> {
    if lon_lo < lon_hi {
        vec![CellBox {
            lat_lo,
            lat_hi,
            lon_lo,
            lon_hi,
        }]
    } else {
        vec![
            CellBox {
                lat_lo,
                lat_hi,
                lon_lo,
                lon_hi: LON_COUNT,
            },
            CellBox {
                lat_lo,
                lat_hi,
                lon_lo: 0,
                lon_hi,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Primary bar (B1): every one of the 1,038,240 real ERA5 0.25-degree
    /// grid cells (`LAT_COUNT * LON_COUNT`) round-trips exactly through
    /// `encode_key` -> `decode_key`, AND no two distinct cells produce the
    /// same 16 bytes. This is **exhaustive, not sampled** -- ~1M iterations
    /// is fast and this is the primary falsifier for the whole module: any
    /// bug that makes two cells collide, or makes any single cell fail to
    /// round-trip, fails this test.
    #[test]
    fn all_grid_cells_round_trip_and_are_collision_free() {
        let classid = 0x1234_5678u32;
        let total = LAT_COUNT as usize * LON_COUNT as usize;
        assert_eq!(total, 1_038_240, "grid cell count drifted from the plan");

        let mut seen: HashSet<[u8; KEY_LEN]> = HashSet::with_capacity(total);
        for lat_idx in 0..LAT_COUNT {
            for lon_idx in 0..LON_COUNT {
                let key = encode_key(classid, lat_idx, lon_idx);
                assert_eq!(
                    decode_key(&key),
                    Some((lat_idx, lon_idx)),
                    "round-trip failed for ({lat_idx}, {lon_idx})"
                );
                assert!(
                    seen.insert(key),
                    "collision: ({lat_idx}, {lon_idx}) produced a key already seen"
                );
            }
        }
        assert_eq!(seen.len(), total, "fewer distinct keys than grid cells");
    }

    /// Control that can lose: exercises the ragged last tile on BOTH axes
    /// explicitly, using hardcoded bounds (704/721, 1408/1440) rather than
    /// `LAT_COUNT`/`LON_COUNT`. If the exhaustive sweep above ever had its
    /// bounds accidentally clamped down to the last *full* tile (704 lat
    /// rows, 1408 lon columns -- e.g. a copy-pasted `& !TILE_MASK`), this
    /// test would still catch it, because it does not share that bug's
    /// source of truth.
    #[test]
    fn ragged_last_tile_cells_round_trip_on_both_axes() {
        let classid = 0x0102_0304u32;
        for lat_idx in 704..721u16 {
            for lon_idx in 1408..1440u16 {
                let key = encode_key(classid, lat_idx, lon_idx);
                assert_eq!(
                    decode_key(&key),
                    Some((lat_idx, lon_idx)),
                    "ragged-tile cell ({lat_idx}, {lon_idx}) did not round-trip"
                );
            }
        }
    }

    /// Stay-silent-twin partner of the ragged-tile test above, on the
    /// rejection side: a cell exactly one past the grid on each axis must
    /// be rejected by `decode_key`, not silently wrapped or aliased onto a
    /// real cell. Keys are built by hand (not via `encode_key`, whose
    /// precondition these indices violate) so the test exercises
    /// `decode_key`'s own bound check in isolation.
    #[test]
    fn out_of_range_index_is_rejected_not_silently_wrapped() {
        let classid = 0xAABB_CCDDu32;

        // lat_idx = 721 = 11*64 + 17: heel=11 (valid tile), hip=17 (one
        // past that tile's 17 valid rows, 0..=16).
        let mut lat_over = [0u8; KEY_LEN];
        lat_over[0..4].copy_from_slice(&classid.to_le_bytes());
        lat_over[4] = 11; // lat heel
        lat_over[5] = 0; // lon heel (valid)
        lat_over[6] = 17; // lat hip -> reconstructs to 721
        lat_over[7] = 0; // lon hip
        assert_eq!(
            decode_key(&lat_over),
            None,
            "lat_idx=721 (one past LAT_COUNT) must be rejected"
        );

        // lon_idx = 1440 = 22*64 + 32: heel=22 (valid tile), hip=32 (one
        // past that tile's 32 valid columns, 0..=31).
        let mut lon_over = [0u8; KEY_LEN];
        lon_over[0..4].copy_from_slice(&classid.to_le_bytes());
        lon_over[4] = 0; // lat heel (valid)
        lon_over[5] = 22; // lon heel
        lon_over[6] = 0; // lat hip
        lon_over[7] = 32; // lon hip -> reconstructs to 1440
        assert_eq!(
            decode_key(&lon_over),
            None,
            "lon_idx=1440 (one past LON_COUNT) must be rejected"
        );

        // Positive control: the true last valid cell on both axes must
        // still decode fine -- proving the two rejections above are about
        // being genuinely out of range, not a blanket "anything near the
        // boundary fails" bug.
        let last_valid = encode_key(classid, 720, 1439);
        assert_eq!(decode_key(&last_valid), Some((720, 1439)));
    }

    /// `box_ranges` can fire: a box crossing the 0deg/360deg seam reports
    /// the wrap as exactly two ranges.
    #[test]
    fn box_ranges_seam_crossing_reports_exactly_two_ranges() {
        // lon 1400..1440, wrapping around to lon 0..40.
        let ranges = box_ranges(10, 20, 1400, 40);
        assert_eq!(ranges.len(), 2, "seam-crossing box must report the wrap");
        assert_eq!(
            ranges[0],
            CellBox {
                lat_lo: 10,
                lat_hi: 20,
                lon_lo: 1400,
                lon_hi: LON_COUNT,
            }
        );
        assert_eq!(
            ranges[1],
            CellBox {
                lat_lo: 10,
                lat_hi: 20,
                lon_lo: 0,
                lon_hi: 40,
            }
        );
    }

    /// `box_ranges` can stay silent: an ordinary box entirely inside
    /// `0..LON_COUNT` reports no wrap -- exactly one range, matching the
    /// input bounds verbatim. Paired with the can-fire test above so
    /// neither half of the wrap detector goes unchecked.
    #[test]
    fn box_ranges_non_wrapping_box_reports_exactly_one_range() {
        let ranges = box_ranges(10, 20, 200, 400);
        assert_eq!(ranges.len(), 1, "non-wrapping box must report no wrap");
        assert_eq!(
            ranges[0],
            CellBox {
                lat_lo: 10,
                lat_hi: 20,
                lon_lo: 200,
                lon_hi: 400,
            }
        );
    }
}
