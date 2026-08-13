//! The L4 lane: pack and unpack ONE 16-byte facet — 4-byte classid prefix +
//! 12-byte payload as `6 x (8:8)` palette pairs
//! (`.claude/v3/soa_layout/le-contract.md` §1/§2/§3's L4 row).
//!
//! Owner: the lane worker (plan `.claude/plans/weather-soa-bake-v1.md` §6.2,
//! deliverable — this file, `D-WXS-4`'s row-assembly companion). Consumes
//! [`crate::manifest`] (the ClassView-side field manifest) and
//! [`crate::floor`] (the calibrated linear quantiser); invents nothing new
//! architecturally.
//!
//! # Slot purity (le-contract.md §2) — restated in this module's own words
//!
//! **Labels and positions come from the ClassView, NEVER from a slot in the
//! payload.** A `NodeRow`'s value slab is dumb bytes; the class makes them
//! meaningful. This module is the code half of that split: it moves bytes
//! according to a [`crate::manifest::FieldManifest`] it was HANDED — it does
//! not know, and must never come to know, which ERA5 variable rides which
//! byte. There is no `match` on a variable name anywhere in this file, and
//! there must never be one. If a future change to this module needs to name
//! a specific ERA5 field to work correctly, that is the exact defect this
//! module exists to prevent — the mapping belongs in the manifest, not here.
//!
//! **The consequence, stated plainly: nothing this file writes ever encodes
//! what a byte means.** [`pack_facet`] and [`unpack_facet`] both take the
//! manifest and the calibrated floors as opaque parameters and follow
//! whatever they say. A caller who wants `v_component_of_wind @ 850 hPa` in
//! byte 5 today and byte 9 tomorrow only ever has to edit
//! `data/field_manifest_v1.tsv` — this module's behaviour follows without a
//! single line changing here. [`crate::manifest`]'s own bar B0 (mutating one
//! manifest entry must change the bytes the bake writes for at least one
//! cell) is exercised end-to-end through this module's tests, closing the
//! half `manifest.rs`'s own module docs left open ("the bake does not exist
//! yet").
//!
//! # The layout, byte-for-byte
//!
//! | bytes | content |
//! |---|---|
//! | `0..4` | `classid: u32`, little-endian — a caller-supplied parameter, never composed or bit-sliced here |
//! | `4..16` | the 96-bit payload: 6 pairs, `payload[2p]` = pair `p`'s LOW byte, `payload[2p+1]` = pair `p`'s HIGH byte |
//!
//! `u8:u8` means **two separate bytes** — a pair is never combined into a
//! `u16`, never byte-swapped, and the low/high assignment is fixed by the
//! manifest's own `PairByte::{Lo, Hi}` convention (`manifest.rs`'s "lo/hi
//! convention" doc section), never inferred from magnitude at pack or
//! unpack time.
//!
//! # A reserved slot is a slot the manifest has no row for
//!
//! `crate::manifest`'s own docs: "reserved slots are NOT rows... their
//! absence IS their meaning: dormant, expandable later without a layout
//! change." This module enforces the two halves of that on the wire:
//! [`pack_facet`] leaves an unresolved `(facet, pair, byte)` slot at its
//! initial `0`, and [`unpack_facet`] never emits an [`UnpackedField`] for
//! one — a reserved slot is *absent* from the unpacked result, never
//! present with the decoded value of a stray `0` byte (`bucket_center(0)`
//! is a real, plausible-looking number for most floors; reporting it for an
//! unoccupied slot would be exactly the corruption this split prevents).
//!
//! # Floor-version provenance is external, per le-contract.md §2.6 point 2
//!
//! The plan is explicit that a floor's `(lo, hi, floor_version)` is
//! **dataset metadata, aligned to the Lance version boundary** — never a
//! per-row byte, which would itself be a slot-purity break. This module
//! therefore does not stamp a version into the row. [`stamped_floor_versions`]
//! stands in for "what a reader would find in that dataset metadata": the
//! `floor_version` each occupied slot's floor carried at the moment the
//! facet was packed. [`unpack_facet`] takes that map as an explicit
//! parameter and refuses (via [`crate::floor::CalibratedFloor::decode`]) to
//! resolve any slot whose stamped version has drifted from the floor it was
//! actually handed — a version mismatch is a reported error, never a
//! silently wrong decoded value.
//!
//! # Scope — what this module is NOT
//!
//! This is one 16-byte facet. It is not the bake (`D-WXS-4`, blocked on the
//! `D-WXS-0` classid mint) and not the 512-byte row assembly (three facets
//! plus the key and edge block). Both are deferred to their own
//! deliverables; this module's job ends at one facet in, one facet out.

use std::collections::HashMap;

use crate::floor::CalibratedFloor;
use crate::manifest::{FieldManifest, ManifestEntry, PairByte};

/// Length in bytes of one L4 facet: 4-byte classid prefix + 12-byte payload
/// (le-contract.md §1).
pub const FACET_LEN: usize = 16;

/// Length in bytes of the classid prefix (le-contract.md §1, bytes `0..4`).
const CLASSID_LEN: usize = 4;

/// Number of `(8:8)` pairs in the L4 payload (le-contract.md §3, "6 x
/// (8:8)").
pub const PAIR_COUNT: u8 = 6;

const _: () = assert!(FACET_LEN == CLASSID_LEN + PAIR_COUNT as usize * 2);

/// The byte offset **within a 16-byte facet** that a given `(pair, byte)`
/// slot occupies.
///
/// This is a purely structural fact about the L4 layout (le-contract.md
/// §3: "every layout is exactly 12 bytes — 6x2... layouts differ only in
/// how the 96 payload bits subdivide") — it is not a label, a variable
/// name, or a display position, so exposing it does not break slot
/// purity: it answers "where is pair `p`'s low/high byte", never "what is
/// stored there".
///
/// # Panics
///
/// Panics if `pair >= `[`PAIR_COUNT`]. Every caller in this module reaches
/// this function only through a `0..PAIR_COUNT` loop, so the panic path is
/// unreachable in practice; it exists as a loud guard against a future
/// caller passing an out-of-range pair directly.
pub fn slot_offset(pair: u8, byte: PairByte) -> usize {
    assert!(
        pair < PAIR_COUNT,
        "pair {pair} out of range (0..{PAIR_COUNT})"
    );
    let byte_idx = match byte {
        PairByte::Lo => 0,
        PairByte::Hi => 1,
    };
    CLASSID_LEN + (pair as usize) * 2 + byte_idx
}

/// One resolved field read back out of a packed facet: the slot it
/// occupied, its manifest identity, and its dequantised value.
///
/// Only ever produced by [`unpack_facet`] for a slot the manifest actually
/// resolves — a reserved-zero slot never produces one of these (see the
/// module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct UnpackedField {
    /// The facet index this field was read from.
    pub facet: u8,
    /// The pair index within the facet.
    pub pair: u8,
    /// Which byte of the pair.
    pub byte: PairByte,
    /// The ERA5 variable name, copied from the manifest entry that
    /// resolved this slot (never inferred from the byte value itself).
    pub variable: String,
    /// The pressure level, copied from the manifest entry.
    pub level_hpa: Option<u16>,
    /// The dequantised value (`CalibratedFloor::bucket_center` of the raw
    /// byte, under the floor whose stamped version matched).
    pub value: f64,
}

/// Everything that can go wrong packing or unpacking an L4 facet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaneError {
    /// An occupied manifest slot names a `floor_id` that is not present in
    /// the `floors` map handed to [`pack_facet`], [`unpack_facet`], or
    /// [`stamped_floor_versions`].
    UnknownFloor {
        /// The facet the missing floor was needed for.
        facet: u8,
        /// The pair.
        pair: u8,
        /// The byte.
        byte: PairByte,
        /// The `floor_id` the manifest entry named.
        floor_id: String,
    },
    /// [`pack_facet`]'s `value_of` closure returned `None` for an occupied
    /// slot — every occupied slot in the target facet must have a value; a
    /// caller with a genuinely missing reading should not call `pack_facet`
    /// for that facet at all, or should thread a documented sentinel
    /// through its own value source rather than lean on the lane to invent
    /// a default.
    MissingValue {
        /// The facet.
        facet: u8,
        /// The pair.
        pair: u8,
        /// The byte.
        byte: PairByte,
        /// The variable the manifest names for this slot.
        variable: String,
        /// The level the manifest names for this slot.
        level_hpa: Option<u16>,
    },
    /// [`pack_facet`]'s `value_of` closure returned a **non-finite** reading
    /// (`NaN`, `+inf` or `-inf`) for an occupied slot.
    ///
    /// # Why this is a hard error and not a silent bucket
    ///
    /// This is **not** a defensive nicety — it closes a measured
    /// silent-corruption path, found by review on PR #948:
    ///
    /// * ARCO-ERA5 is **sparse by design**. `probes/weather-p1/README.md` §1
    ///   records the store's `fill_value: NaN`, that several variables 404 at
    ///   the arc's own fixture timestep, and that in Zarr v2 *a missing chunk
    ///   means all-`fill_value`* — so **a 404 is valid store semantics, not a
    ///   fetch failure**, and an all-`NaN` field is ordinary data an ingest
    ///   must expect.
    /// * The 404-ing list at that timestep includes `mean_sea_level_pressure`,
    ///   `10m_v_component_of_wind`, `surface_pressure`,
    ///   `total_column_water_vapour` and `total_cloud_cover` — **five
    ///   variables the W1 field set actually packs** (F0 pairs 0, 1, 3, 4).
    /// * [`CalibratedFloor::quantize`](crate::floor::CalibratedFloor::quantize)
    ///   maps non-finite input to a **valid-looking bucket, silently**
    ///   (measured: `NaN` → `0`, `-inf` → `0`, `+inf` → `255`; Rust's
    ///   float→int cast saturates and sends `NaN` to zero, and `f64::clamp`
    ///   propagates `NaN` rather than clamping it).
    ///
    /// Without this guard an entire missing field would be written as
    /// plausible low-bucket measurements and read back through
    /// [`CalibratedFloor::bucket_center`](crate::floor::CalibratedFloor::bucket_center)
    /// as ordinary numbers — the exact failure the reserved-slot rule
    /// (*"a reserved slot must not read back as a plausible number"*) exists
    /// to prevent, one level deeper and harder to see.
    ///
    /// The guard covers **all** non-finite values, not only `NaN`: `±inf`
    /// land on the rim buckets, which are legitimate saturation values and
    /// therefore just as indistinguishable from a real reading.
    NonFiniteValue {
        /// The facet.
        facet: u8,
        /// The pair.
        pair: u8,
        /// The byte.
        byte: PairByte,
        /// The variable the manifest names for this slot.
        variable: String,
        /// The level the manifest names for this slot.
        level_hpa: Option<u16>,
    },
    /// [`unpack_facet`]'s `stamped_versions` map has no entry for a
    /// `floor_id` an occupied slot needs — the caller's "dataset metadata"
    /// is incomplete for this facet.
    MissingStampedVersion {
        /// The facet.
        facet: u8,
        /// The pair.
        pair: u8,
        /// The byte.
        byte: PairByte,
        /// The `floor_id` with no stamped version.
        floor_id: String,
    },
    /// A slot's stamped `floor_version` (the version recorded at pack time,
    /// per the module docs' "dataset metadata" framing) does not match the
    /// `floor_version` the [`CalibratedFloor`] handed to [`unpack_facet`]
    /// actually carries. Detected via
    /// [`CalibratedFloor::decode`](crate::floor::CalibratedFloor::decode)
    /// returning `None` — never silently dequantised under the wrong
    /// `[lo, hi]` window.
    FloorVersionMismatch {
        /// The facet.
        facet: u8,
        /// The pair.
        pair: u8,
        /// The byte.
        byte: PairByte,
        /// The `floor_id` whose version drifted.
        floor_id: String,
        /// The version stamped in `stamped_versions`.
        expected: u64,
        /// The version the floor handed to `unpack_facet` actually has.
        found: u64,
    },
}

impl std::fmt::Display for LaneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaneError::UnknownFloor {
                facet,
                pair,
                byte,
                floor_id,
            } => write!(
                f,
                "facet={facet} pair={pair} byte={byte}: no floor for floor_id {floor_id:?}"
            ),
            LaneError::MissingValue {
                facet,
                pair,
                byte,
                variable,
                level_hpa,
            } => write!(
                f,
                "facet={facet} pair={pair} byte={byte}: no value supplied for {variable:?} at level {level_hpa:?}"
            ),
            LaneError::NonFiniteValue {
                facet,
                pair,
                byte,
                variable,
                level_hpa,
            } => write!(
                f,
                "facet={facet} pair={pair} byte={byte}: non-finite reading for {variable:?} at level {level_hpa:?} \
                 (an all-NaN field is valid ARCO-ERA5 store semantics for a missing chunk, never a bucket)"
            ),
            LaneError::MissingStampedVersion {
                facet,
                pair,
                byte,
                floor_id,
            } => write!(
                f,
                "facet={facet} pair={pair} byte={byte}: no stamped floor_version for floor_id {floor_id:?}"
            ),
            LaneError::FloorVersionMismatch {
                facet,
                pair,
                byte,
                floor_id,
                expected,
                found,
            } => write!(
                f,
                "facet={facet} pair={pair} byte={byte}: floor_id {floor_id:?} stamped version {expected} does not match floor's actual version {found}"
            ),
        }
    }
}

impl std::error::Error for LaneError {}

/// The six `(pair, byte)` slots of one facet, in a fixed, deterministic
/// order (`pair` ascending, `Lo` before `Hi` within a pair). Both
/// [`pack_facet`] and [`unpack_facet`] walk this same order, which is what
/// makes the manifest — not iteration order — the only thing either
/// function's output depends on.
fn all_slots() -> impl Iterator<Item = (u8, PairByte)> {
    (0..PAIR_COUNT).flat_map(|pair| [(pair, PairByte::Lo), (pair, PairByte::Hi)])
}

/// Packs ONE facet: for every occupied `(pair, byte)` slot the manifest
/// resolves against `facet`, quantises the value `value_of` supplies
/// (through the floor the manifest names) into that slot's byte; every
/// unresolved slot stays `0` (reserved-zero, see the module docs).
///
/// `classid` is written verbatim as little-endian bytes into `0..4` — this
/// function never composes, mints, or bit-slices it.
///
/// `value_of` is called once per occupied slot, in the fixed order
/// [`all_slots`] walks, with the [`ManifestEntry`] that slot resolved to.
/// It returns the raw `f64` reading for that field, or `None` if none is
/// available — which is reported as [`LaneError::MissingValue`], not
/// silently treated as zero (a `0.0` reading and a missing reading are
/// different facts, and conflating them would corrupt the bucket the
/// floor quantises `0.0` to).
///
/// # Errors
///
/// Returns [`LaneError::UnknownFloor`] if an occupied slot's `floor_id` is
/// not in `floors`, or [`LaneError::MissingValue`] if `value_of` returns
/// `None` for an occupied slot. Fails on the first such slot encountered
/// (in [`all_slots`] order); a facet with several bad slots reports the
/// first one, not all of them.
pub fn pack_facet<F>(
    classid: u32,
    facet: u8,
    manifest: &FieldManifest,
    floors: &HashMap<String, CalibratedFloor>,
    mut value_of: F,
) -> Result<[u8; FACET_LEN], LaneError>
where
    F: FnMut(&ManifestEntry) -> Option<f64>,
{
    let mut bytes = [0u8; FACET_LEN];
    bytes[0..CLASSID_LEN].copy_from_slice(&classid.to_le_bytes());

    for (pair, byte) in all_slots() {
        let Some(entry) = manifest.resolve_slot(facet, pair, byte) else {
            // Reserved-zero slot: no manifest entry, leave the byte at its
            // initial 0. This IS the slot's meaning (dormant, expandable),
            // never an omission to fill in.
            continue;
        };

        let floor = floors
            .get(&entry.floor_id)
            .ok_or_else(|| LaneError::UnknownFloor {
                facet,
                pair,
                byte,
                floor_id: entry.floor_id.clone(),
            })?;

        let value = value_of(entry).ok_or_else(|| LaneError::MissingValue {
            facet,
            pair,
            byte,
            variable: entry.variable.clone(),
            level_hpa: entry.level_hpa,
        })?;

        // A non-finite reading must NEVER reach `quantize`. See
        // `LaneError::NonFiniteValue` for the measured silent-corruption
        // path this guard closes.
        if !value.is_finite() {
            return Err(LaneError::NonFiniteValue {
                facet,
                pair,
                byte,
                variable: entry.variable.clone(),
                level_hpa: entry.level_hpa,
            });
        }

        bytes[slot_offset(pair, byte)] = floor.quantize(value);
    }

    Ok(bytes)
}

/// Unpacks ONE facet: for every `(pair, byte)` slot `manifest` resolves
/// against `facet`, dequantises the raw byte through the matching floor —
/// gated on `stamped_versions` agreeing with that floor's actual
/// `floor_version` — and returns it as an [`UnpackedField`]. A slot the
/// manifest does not resolve is skipped entirely: it produces no
/// `UnpackedField`, reserved or not, whatever raw byte value it holds.
///
/// # Errors
///
/// Returns [`LaneError::UnknownFloor`] if an occupied slot's `floor_id` is
/// not in `floors`, [`LaneError::MissingStampedVersion`] if it is not in
/// `stamped_versions`, or [`LaneError::FloorVersionMismatch`] if the
/// stamped version does not match the floor's own
/// [`CalibratedFloor::floor_version`](crate::floor::CalibratedFloor::floor_version)
/// (checked via [`CalibratedFloor::decode`](crate::floor::CalibratedFloor::decode),
/// never bypassed). Fails on the first such slot, in [`all_slots`] order.
pub fn unpack_facet(
    bytes: &[u8; FACET_LEN],
    facet: u8,
    manifest: &FieldManifest,
    floors: &HashMap<String, CalibratedFloor>,
    stamped_versions: &HashMap<String, u64>,
) -> Result<Vec<UnpackedField>, LaneError> {
    let mut out = Vec::new();

    for (pair, byte) in all_slots() {
        let Some(entry) = manifest.resolve_slot(facet, pair, byte) else {
            continue;
        };

        let floor = floors
            .get(&entry.floor_id)
            .ok_or_else(|| LaneError::UnknownFloor {
                facet,
                pair,
                byte,
                floor_id: entry.floor_id.clone(),
            })?;

        let expected_version = stamped_versions
            .get(&entry.floor_id)
            .copied()
            .ok_or_else(|| LaneError::MissingStampedVersion {
                facet,
                pair,
                byte,
                floor_id: entry.floor_id.clone(),
            })?;

        let raw = bytes[slot_offset(pair, byte)];
        let value =
            floor
                .decode(raw, expected_version)
                .ok_or_else(|| LaneError::FloorVersionMismatch {
                    facet,
                    pair,
                    byte,
                    floor_id: entry.floor_id.clone(),
                    expected: expected_version,
                    found: floor.floor_version(),
                })?;

        out.push(UnpackedField {
            facet,
            pair,
            byte,
            variable: entry.variable.clone(),
            level_hpa: entry.level_hpa,
            value,
        });
    }

    Ok(out)
}

/// Collects the `floor_version` every occupied slot of `facet` actually
/// used, keyed by `floor_id`.
///
/// This stands in for the "dataset metadata, aligned to the Lance version
/// boundary" the plan describes (module docs, "Floor-version provenance is
/// external") — in the real bake it would be read from that metadata; here
/// it is derived directly from the same `floors` map [`pack_facet`] was
/// given, so a caller can produce it immediately after packing and hand it
/// to [`unpack_facet`] as the "what was this written under" record.
///
/// # Errors
///
/// Returns [`LaneError::UnknownFloor`] under the same condition
/// [`pack_facet`] would have failed under — an occupied slot names a
/// `floor_id` not present in `floors`.
pub fn stamped_floor_versions(
    manifest: &FieldManifest,
    facet: u8,
    floors: &HashMap<String, CalibratedFloor>,
) -> Result<HashMap<String, u64>, LaneError> {
    let mut out = HashMap::new();

    for (pair, byte) in all_slots() {
        let Some(entry) = manifest.resolve_slot(facet, pair, byte) else {
            continue;
        };

        let floor = floors
            .get(&entry.floor_id)
            .ok_or_else(|| LaneError::UnknownFloor {
                facet,
                pair,
                byte,
                floor_id: entry.floor_id.clone(),
            })?;

        out.insert(entry.floor_id.clone(), floor.floor_version());
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::floor::calibrate;

    /// The exact manifest header, spelled literally rather than imported —
    /// `manifest::HEADER` is a private constant of that module (by design:
    /// it is validated internally, not a public API surface).
    const HEADER: &str = "facet\tpair\tbyte\tvariable\tlevel_hpa\tunit\tfloor_id";

    /// A small, self-contained fixture manifest for this module's tests:
    /// facet 0 only, three occupied slots spread across two pairs, leaving
    /// both a fully-reserved pair (2) and a half-reserved pair (1, hi byte)
    /// to exercise the reserved-slot behaviour without depending on the
    /// full committed W1 manifest.
    ///
    /// | facet | pair | byte | variable | floor_id |
    /// |---|---|---|---|---|
    /// | 0 | 0 | lo | fieldA | fa |
    /// | 0 | 0 | hi | fieldB | fb |
    /// | 0 | 1 | lo | fieldC | fa |
    /// | 0 | 1 | hi | *(reserved)* | |
    /// | 0 | 2..5 | * | *(fully reserved)* | |
    fn fixture_manifest() -> FieldManifest {
        let text = format!(
            "{HEADER}\n\
             0\t0\tlo\tfieldA\tsurface\tK\tfa\n\
             0\t0\thi\tfieldB\tsurface\tPa\tfb\n\
             0\t1\tlo\tfieldC\tsurface\tK\tfa\n"
        );
        FieldManifest::parse(&text).expect("fixture manifest must parse")
    }

    /// Deterministic evenly-spaced sample, mirroring `floor.rs`'s own test
    /// helper (private to that module, so re-declared here).
    fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        assert!(n >= 2, "linspace needs at least 2 points");
        (0..n)
            .map(|i| lo + (hi - lo) * i as f64 / (n as f64 - 1.0))
            .collect()
    }

    /// Two distinctly-ranged floors, keyed exactly as [`fixture_manifest`]'s
    /// entries name them (`"fa"`, `"fb"`).
    fn fixture_floors() -> HashMap<String, CalibratedFloor> {
        let mut floors = HashMap::new();
        floors.insert(
            "fa".to_string(),
            calibrate(&linspace(-100.0, 100.0, 10_000)).expect("fa calibrates"),
        );
        floors.insert(
            "fb".to_string(),
            calibrate(&linspace(900.0, 1_100.0, 10_000)).expect("fb calibrates"),
        );
        floors
    }

    /// A `value_of` closure over the three fixture fields, matched by
    /// variable name (the closure — supplied by the TEST, not by this
    /// module — is where variable-name knowledge is allowed to live; the
    /// module under test never inspects `entry.variable` itself to decide
    /// what to do, only to report it back to the caller).
    fn fixture_values(
        field_a: f64,
        field_b: f64,
        field_c: f64,
    ) -> impl FnMut(&ManifestEntry) -> Option<f64> {
        move |entry: &ManifestEntry| match entry.variable.as_str() {
            "fieldA" => Some(field_a),
            "fieldB" => Some(field_b),
            "fieldC" => Some(field_c),
            _ => None,
        }
    }

    // ── round-trip ───────────────────────────────────────────────────────

    #[test]
    fn round_trip_recovers_every_occupied_field_within_half_a_bucket() {
        let manifest = fixture_manifest();
        let floors = fixture_floors();

        let field_a = 12.3_f64;
        let field_b = 1_005.7_f64;
        let field_c = -42.0_f64;

        let bytes = pack_facet(
            0xDEAD_BEEF,
            0,
            &manifest,
            &floors,
            fixture_values(field_a, field_b, field_c),
        )
        .expect("pack must succeed");

        let stamped = stamped_floor_versions(&manifest, 0, &floors).expect("stamping must succeed");
        let unpacked =
            unpack_facet(&bytes, 0, &manifest, &floors, &stamped).expect("unpack must succeed");

        assert_eq!(unpacked.len(), 3, "exactly the three occupied slots");

        let expect_value = |variable: &str| match variable {
            "fieldA" => field_a,
            "fieldB" => field_b,
            "fieldC" => field_c,
            other => panic!("unexpected variable in unpacked output: {other}"),
        };

        for field in &unpacked {
            let entry = manifest
                .resolve_slot(field.facet, field.pair, field.byte)
                .expect("unpacked field must resolve back to a manifest entry");
            let floor = floors
                .get(&entry.floor_id)
                .expect("resolved entry's floor_id must be in the fixture floors map");
            let (lo, hi) = floor.bounds();
            let half_bucket = (hi - lo) / (2.0 * 256.0);
            let original = expect_value(&field.variable);
            let delta = (field.value - original).abs();
            assert!(
                delta <= half_bucket + 1e-9,
                "{}: round-trip error {delta} exceeds the +/-half-bucket bound {half_bucket}",
                field.variable
            );
        }
    }

    // ── reserved slots stay zero AND read as absent ─────────────────────

    #[test]
    fn reserved_slots_are_byte_zero_after_pack_and_absent_after_unpack() {
        let manifest = fixture_manifest();
        let floors = fixture_floors();

        let bytes = pack_facet(0x1, 0, &manifest, &floors, fixture_values(1.0, 2.0, 3.0))
            .expect("pack must succeed");

        // Fully-reserved pair (pair 2, both bytes) is byte-zero.
        assert_eq!(bytes[slot_offset(2, PairByte::Lo)], 0);
        assert_eq!(bytes[slot_offset(2, PairByte::Hi)], 0);
        // Half-reserved pair (pair 1, hi byte only) is also byte-zero.
        assert_eq!(bytes[slot_offset(1, PairByte::Hi)], 0);

        let stamped = stamped_floor_versions(&manifest, 0, &floors).expect("stamping must succeed");
        let unpacked =
            unpack_facet(&bytes, 0, &manifest, &floors, &stamped).expect("unpack must succeed");

        // Absence, not a decoded-0.0 entry: no UnpackedField at any
        // reserved slot.
        let has_slot =
            |pair: u8, byte: PairByte| unpacked.iter().any(|f| f.pair == pair && f.byte == byte);
        assert!(!has_slot(2, PairByte::Lo), "pair 2 lo must be absent");
        assert!(!has_slot(2, PairByte::Hi), "pair 2 hi must be absent");
        assert!(!has_slot(1, PairByte::Hi), "pair 1 hi must be absent");

        // The occupied slots ARE present, so the absence above is really a
        // filtered set and not an empty-everything bug.
        assert!(has_slot(0, PairByte::Lo), "pair 0 lo must be present");
        assert!(has_slot(0, PairByte::Hi), "pair 0 hi must be present");
        assert!(has_slot(1, PairByte::Lo), "pair 1 lo must be present");
    }

    // ── manifest is load-bearing (can-it-fire) ──────────────────────────

    #[test]
    fn moving_a_field_to_a_different_slot_changes_the_packed_bytes() {
        let manifest_a = fixture_manifest();
        // Same fields, but `fieldA` moves from (pair 0, lo) to the
        // previously fully-reserved (pair 2, lo) — a genuinely different
        // manifest, not a relabelling.
        let manifest_b_text = format!(
            "{HEADER}\n\
             0\t2\tlo\tfieldA\tsurface\tK\tfa\n\
             0\t0\thi\tfieldB\tsurface\tPa\tfb\n\
             0\t1\tlo\tfieldC\tsurface\tK\tfa\n"
        );
        let manifest_b =
            FieldManifest::parse(&manifest_b_text).expect("moved-slot manifest must parse");

        let floors = fixture_floors();
        let value_of = || fixture_values(12.3, 1_005.7, -42.0);

        let bytes_a = pack_facet(0x1, 0, &manifest_a, &floors, value_of())
            .expect("pack under manifest_a must succeed");
        let bytes_b = pack_facet(0x1, 0, &manifest_b, &floors, value_of())
            .expect("pack under manifest_b must succeed");

        assert_ne!(
            bytes_a, bytes_b,
            "moving fieldA to a different slot must change the packed bytes"
        );
        // Specifically: fieldA's quantised byte now lives at pair 2 lo
        // (manifest_a leaves that slot reserved-zero, manifest_b occupies
        // it), and the pair-0-lo slot it vacated is back to reserved-zero
        // under manifest_b.
        assert_eq!(bytes_a[slot_offset(2, PairByte::Lo)], 0);
        assert_eq!(bytes_b[slot_offset(0, PairByte::Lo)], 0);
        // Anti-vacuity: the relocated byte is a real, non-zero quantised
        // value -- not a coincidental all-zero match between the two facets.
        assert_ne!(bytes_a[slot_offset(0, PairByte::Lo)], 0);
        assert_eq!(
            bytes_b[slot_offset(2, PairByte::Lo)],
            bytes_a[slot_offset(0, PairByte::Lo)],
            "fieldA's quantised byte value should be identical, just relocated"
        );
    }

    // ── stay-silent twin: reordered-but-identical manifest ──────────────

    #[test]
    fn reordering_manifest_rows_does_not_change_the_packed_bytes() {
        let manifest_forward = fixture_manifest();

        let reordered_text = format!(
            "{HEADER}\n\
             0\t1\tlo\tfieldC\tsurface\tK\tfa\n\
             0\t0\thi\tfieldB\tsurface\tPa\tfb\n\
             0\t0\tlo\tfieldA\tsurface\tK\tfa\n"
        );
        let manifest_reordered =
            FieldManifest::parse(&reordered_text).expect("reordered manifest must parse");

        let floors = fixture_floors();
        let value_of = || fixture_values(12.3, 1_005.7, -42.0);

        let bytes_forward = pack_facet(0x1, 0, &manifest_forward, &floors, value_of())
            .expect("pack under forward-order manifest must succeed");
        let bytes_reordered = pack_facet(0x1, 0, &manifest_reordered, &floors, value_of())
            .expect("pack under reordered manifest must succeed");

        assert_eq!(
            bytes_forward, bytes_reordered,
            "a byte-different-but-semantically-identical manifest must pack identically"
        );
    }

    // ── the pair is two bytes, never combined or swapped ────────────────

    #[test]
    fn lo_and_hi_bytes_of_a_pair_land_at_their_own_index_never_swapped() {
        let manifest = fixture_manifest();
        let floors = fixture_floors();

        // fieldA (lo, floor "fa" over [-100, 100]) near its floor's low
        // bound; fieldB (hi, floor "fb" over [900, 1100]) near its floor's
        // high bound -- deliberately far apart so a swap or a u16-combine
        // is impossible to miss.
        let field_a = -95.0_f64;
        let field_b = 1_095.0_f64;
        let field_c = 0.0_f64;

        let bytes = pack_facet(
            0x1,
            0,
            &manifest,
            &floors,
            fixture_values(field_a, field_b, field_c),
        )
        .expect("pack must succeed");

        let expected_lo = floors["fa"].quantize(field_a);
        let expected_hi = floors["fb"].quantize(field_b);

        assert_eq!(
            bytes[slot_offset(0, PairByte::Lo)],
            expected_lo,
            "the lo byte must hold fieldA's own quantised value"
        );
        assert_eq!(
            bytes[slot_offset(0, PairByte::Hi)],
            expected_hi,
            "the hi byte must hold fieldB's own quantised value, not fieldA's"
        );
        // Anti-vacuity: the two really are different bytes at different
        // indices -- a swap-bug (writing lo into hi's slot or vice versa)
        // would make this assertion pass for the wrong reason if the two
        // expected values happened to coincide.
        assert_ne!(
            expected_lo, expected_hi,
            "fixture must produce genuinely distinct lo/hi bytes to be a real test"
        );
        assert_eq!(
            slot_offset(0, PairByte::Lo) + 1,
            slot_offset(0, PairByte::Hi)
        );
    }

    // ── floor-version mismatch is detected, never silently mis-decoded ──

    #[test]
    fn a_stamped_floor_version_mismatch_is_reported_not_silently_mis_decoded() {
        let manifest = fixture_manifest();
        let floors = fixture_floors();

        let bytes = pack_facet(
            0x1,
            0,
            &manifest,
            &floors,
            fixture_values(12.3, 1_005.7, -42.0),
        )
        .expect("pack must succeed");

        let mut stamped =
            stamped_floor_versions(&manifest, 0, &floors).expect("stamping must succeed");

        // Corrupt only floor "fa"'s stamped version (shared by fieldA and
        // fieldC); floor "fb" (fieldB) is untouched.
        let real_fa_version = stamped["fa"];
        let bogus_fa_version = real_fa_version.wrapping_add(1);
        assert_ne!(
            bogus_fa_version, real_fa_version,
            "fixture must construct a genuinely different version"
        );
        stamped.insert("fa".to_string(), bogus_fa_version);

        let err = unpack_facet(&bytes, 0, &manifest, &floors, &stamped)
            .expect_err("a stamped version drift must be reported, not silently decoded");

        match err {
            LaneError::FloorVersionMismatch {
                floor_id,
                expected,
                found,
                ..
            } => {
                assert_eq!(floor_id, "fa");
                assert_eq!(expected, bogus_fa_version);
                assert_eq!(found, real_fa_version);
            }
            other => panic!("expected FloorVersionMismatch, got {other:?}"),
        }

        // Positive control: restoring the correct stamped version decodes
        // cleanly again, proving the failure above was really about the
        // version and not some other defect in the fixture.
        stamped.insert("fa".to_string(), real_fa_version);
        let unpacked = unpack_facet(&bytes, 0, &manifest, &floors, &stamped)
            .expect("unpack with the correct stamped version must succeed");
        assert_eq!(unpacked.len(), 3);
    }

    // ── error paths this module owns ─────────────────────────────────────

    #[test]
    fn pack_reports_a_missing_value_rather_than_defaulting_to_zero() {
        let manifest = fixture_manifest();
        let floors = fixture_floors();

        // A value_of closure that always returns None -- every occupied
        // slot is "missing".
        let err = pack_facet(0x1, 0, &manifest, &floors, |_entry| None)
            .expect_err("a missing value must be reported, not silently zeroed");

        assert!(matches!(err, LaneError::MissingValue { pair: 0, .. }));
    }

    /// A non-finite reading is REJECTED, and the paired half proves the same
    /// slot packs fine when the reading is finite.
    ///
    /// This is the regression for the PR #948 review finding. ARCO-ERA5 is
    /// sparse by design: `probes/weather-p1/README.md` §1 records
    /// `fill_value: NaN` and that a 404 chunk means an all-`NaN` field is
    /// **valid store semantics**, at the arc's own fixture timestep, for five
    /// variables the W1 field set actually packs.
    ///
    /// The third assertion block is what makes this a real test rather than a
    /// restatement of the guard: it shows what the guard PREVENTS, by
    /// quantising the same non-finite values directly and observing that they
    /// land on ordinary, plausible buckets.
    #[test]
    fn pack_rejects_non_finite_readings_which_would_otherwise_become_plausible_buckets() {
        let manifest = fixture_manifest();
        let floors = fixture_floors();

        // ── can-fire: every flavour of non-finite is refused ──
        // Only ONE field is poisoned per run; the others stay finite, so a
        // guard that fired on everything would fail the twin below.
        for (name, bad) in [
            ("NaN", f64::NAN),
            ("+inf", f64::INFINITY),
            ("-inf", f64::NEG_INFINITY),
        ] {
            let err = pack_facet(0x1, 0, &manifest, &floors, move |entry| {
                if entry.variable == "fieldA" {
                    Some(bad)
                } else {
                    Some(1.0)
                }
            })
            .expect_err("a non-finite reading must be refused");
            assert!(
                matches!(err, LaneError::NonFiniteValue { ref variable, .. } if variable == "fieldA"),
                "{name} must be reported as NonFiniteValue for fieldA, got {err:?}"
            );
        }

        // ── stay-silent twin (non-trivial): the SAME slot, finite, packs ──
        let ok = pack_facet(
            0x1,
            0,
            &manifest,
            &floors,
            fixture_values(12.3, 1_005.7, -42.0),
        )
        .expect("finite readings at the same slots must pack cleanly");
        assert_ne!(
            ok, [0u8; FACET_LEN],
            "the finite pack must actually write something"
        );

        // ── what the guard PREVENTS (the anti-vacuity half) ──
        // Without the guard these values reach `quantize`, which maps them to
        // ordinary buckets with no signal that anything was wrong.
        let floor = floors.get("fa").expect("fixture floor fa");
        let (lo, hi) = floor.bounds();
        for (name, bad) in [
            ("NaN", f64::NAN),
            ("+inf", f64::INFINITY),
            ("-inf", f64::NEG_INFINITY),
        ] {
            let bucket = floor.quantize(bad);
            let decoded = floor.bucket_center(bucket);
            assert!(
                decoded >= lo && decoded <= hi,
                "{name} quantised to bucket {bucket}, decoding to {decoded} — inside [{lo}, {hi}], \
                 i.e. indistinguishable from a real reading. This is why the guard exists."
            );
        }
    }

    #[test]
    fn pack_reports_an_unknown_floor_rather_than_skipping_it() {
        let manifest = fixture_manifest();
        // Deliberately omit "fa" from the floors map.
        let mut floors = fixture_floors();
        floors.remove("fa");

        let err = pack_facet(0x1, 0, &manifest, &floors, fixture_values(1.0, 2.0, 3.0))
            .expect_err("an unresolvable floor_id must be reported");

        assert!(matches!(
            err,
            LaneError::UnknownFloor { floor_id, .. } if floor_id == "fa"
        ));
    }
}
