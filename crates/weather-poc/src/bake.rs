//! W1 weather bake assembly: one ERA5 cell -> canonical key + three L4 facets.
//!
//! This module deliberately stops at the zero-dependency ownership boundary.
//! The default `weather-poc` codec path does not depend on
//! `lance-graph-contract`, so it must not copy the canonical `NodeRow`
//! value-tail offset into a second crate. Instead it emits a
//! [`PackedWeatherCell`] containing the byte-exact key and the three W1 facets.
//! The opt-in canonical adapter places those facets into the live canonical
//! `NodeRow` using the contract-derived free-tail offset.
//!
//! That split is load-bearing: new append-only `ValueTenant`s may move the free
//! tail without changing the 512-byte ABI. A hard-coded row offset here would
//! silently overwrite a later tenant.
//!
//! # Streaming, not a million-object staging vector
//!
//! [`bake_timestep`] walks the 721 x 1440 grid and hands each packed cell to a
//! caller-provided sink immediately. It never allocates a `Vec` of 1,038,240
//! cells. The sink may assemble `NodeRow`s, feed Arrow batches, hash a receipt,
//! or stop on its own error.

use std::collections::HashMap;
use std::fmt;

use crate::floor::CalibratedFloor;
use crate::key::{encode_key, LAT_COUNT, LON_COUNT};
use crate::lane::{pack_facet, LaneError, FACET_LEN};
use crate::manifest::{FieldManifest, ManifestEntry};

/// Number of W1 L4 facets: F0 surface, F1 850 hPa, F2 500 hPa.
pub const W1_FACET_COUNT: usize = 3;

/// Bytes in the W1 weather extension image: three 16-byte L4 facets.
pub const W1_IMAGE_LEN: usize = W1_FACET_COUNT * FACET_LEN;

/// Number of real cells in one ERA5 0.25-degree global timestep.
pub const GLOBAL_CELL_COUNT: usize = LAT_COUNT as usize * LON_COUNT as usize;

const _: () = assert!(GLOBAL_CELL_COUNT == 1_038_240);
const _: () = assert!(W1_IMAGE_LEN == 48);

/// One zero-dependency W1 cell image, ready for the canonical-row adapter.
///
/// This is intentionally *not* a second `NodeRow` type. It carries only the
/// bytes `weather-poc` owns: the canonical key image plus the three weather
/// facets. Edge bytes, existing value tenants, and the current free-tail
/// offset remain the contract crate's responsibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedWeatherCell {
    /// Source latitude index, retained for receipts and sink diagnostics.
    pub lat_idx: u16,
    /// Source longitude index, retained for receipts and sink diagnostics.
    pub lon_idx: u16,
    /// Canonical 16-byte key image produced by [`encode_key`].
    pub key: [u8; 16],
    /// F0/F1/F2 L4 facets in facet-index order.
    pub facets: [[u8; FACET_LEN]; W1_FACET_COUNT],
}

impl PackedWeatherCell {
    /// Flatten the three W1 facets into their contiguous 48-byte extension
    /// image. This is useful for hashing/receipts and remains independent of
    /// where the live `NodeRow` contract places the extension.
    pub fn facet_image(&self) -> [u8; W1_IMAGE_LEN] {
        let mut out = [0u8; W1_IMAGE_LEN];
        for (facet, bytes) in self.facets.iter().enumerate() {
            let start = facet * FACET_LEN;
            out[start..start + FACET_LEN].copy_from_slice(bytes);
        }
        out
    }
}

/// Errors produced before a packed cell reaches the caller's sink.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BakeError {
    /// `0x0000_0000` belongs to the canonical bootstrap/default ladder and may
    /// never label a durable weather row.
    ZeroClassId,
    /// A caller asked to pack a coordinate outside the real 721 x 1440 grid.
    /// This is a hard error even in release builds. `encode_key` itself keeps a
    /// debug assertion because it is a low-level codec; the bake is the
    /// publication boundary and must not rely on debug-only validation.
    GridIndexOutOfRange {
        /// Requested latitude index.
        lat_idx: u16,
        /// Requested longitude index.
        lon_idx: u16,
    },
    /// Packing one of the three L4 facets failed.
    Lane(LaneError),
}

impl fmt::Display for BakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BakeError::ZeroClassId => f.write_str(
                "weather bake refuses classid 0x00000000: it is the canonical bootstrap/default class",
            ),
            BakeError::GridIndexOutOfRange { lat_idx, lon_idx } => write!(
                f,
                "weather bake cell ({lat_idx}, {lon_idx}) is outside the real grid 0..{LAT_COUNT} x 0..{LON_COUNT}"
            ),
            BakeError::Lane(err) => write!(f, "weather facet packing failed: {err}"),
        }
    }
}

impl std::error::Error for BakeError {}

impl From<LaneError> for BakeError {
    fn from(value: LaneError) -> Self {
        Self::Lane(value)
    }
}

/// Error from a full-grid streaming bake.
#[derive(Debug)]
pub enum BakeStreamError<E> {
    /// The weather cell could not be assembled.
    Bake(BakeError),
    /// The downstream sink rejected an otherwise valid packed cell.
    Sink(E),
}

impl<E: fmt::Display> fmt::Display for BakeStreamError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BakeStreamError::Bake(err) => err.fmt(f),
            BakeStreamError::Sink(err) => write!(f, "weather bake sink failed: {err}"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for BakeStreamError<E> {}

/// Pack one real grid cell under the W1 manifest.
///
/// `value_of` is the source boundary. It receives manifest entries rather
/// than numeric slot ordinals, so source adapters may resolve values however
/// they like while slot placement remains entirely manifest-owned.
///
/// # Errors
///
/// Refuses the bootstrap classid, rejects indices outside the real grid in all
/// build modes, and propagates all [`LaneError`] variants, including
/// missing/non-finite source values and unknown floors.
pub fn pack_cell<F>(
    classid: u32,
    lat_idx: u16,
    lon_idx: u16,
    manifest: &FieldManifest,
    floors: &HashMap<String, CalibratedFloor>,
    mut value_of: F,
) -> Result<PackedWeatherCell, BakeError>
where
    F: FnMut(&ManifestEntry) -> Option<f64>,
{
    if classid == 0 {
        return Err(BakeError::ZeroClassId);
    }
    if lat_idx >= LAT_COUNT || lon_idx >= LON_COUNT {
        return Err(BakeError::GridIndexOutOfRange { lat_idx, lon_idx });
    }

    let key = encode_key(classid, lat_idx, lon_idx);
    let mut facets = [[0u8; FACET_LEN]; W1_FACET_COUNT];
    for (facet_idx, facet_bytes) in facets.iter_mut().enumerate() {
        *facet_bytes = pack_facet(
            classid,
            facet_idx as u8,
            manifest,
            floors,
            |entry| value_of(entry),
        )?;
    }

    Ok(PackedWeatherCell {
        lat_idx,
        lon_idx,
        key,
        facets,
    })
}

/// Stream one complete global timestep through `sink`, one packed cell at a
/// time, in deterministic latitude-major / longitude-minor order.
///
/// The function returns the number of cells accepted by the sink. A successful
/// full-grid run therefore returns [`GLOBAL_CELL_COUNT`]. No million-cell
/// staging vector is created.
///
/// # Errors
///
/// Stops at the first source/packing error or sink error. The caller owns any
/// transactional semantics needed to ensure that a partial sink is not
/// published as a durable timestep.
pub fn bake_timestep<F, S, E>(
    classid: u32,
    manifest: &FieldManifest,
    floors: &HashMap<String, CalibratedFloor>,
    mut value_of: F,
    mut sink: S,
) -> Result<usize, BakeStreamError<E>>
where
    F: FnMut(u16, u16, &ManifestEntry) -> Option<f64>,
    S: FnMut(PackedWeatherCell) -> Result<(), E>,
{
    if classid == 0 {
        return Err(BakeStreamError::Bake(BakeError::ZeroClassId));
    }

    let mut written = 0usize;
    for lat_idx in 0..LAT_COUNT {
        for lon_idx in 0..LON_COUNT {
            let cell = pack_cell(
                classid,
                lat_idx,
                lon_idx,
                manifest,
                floors,
                |entry| value_of(lat_idx, lon_idx, entry),
            )
            .map_err(BakeStreamError::Bake)?;
            sink(cell).map_err(BakeStreamError::Sink)?;
            written += 1;
        }
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::floor::calibrate;
    use crate::manifest::PairByte;

    const HEADER: &str = "facet\tpair\tbyte\tvariable\tlevel_hpa\tunit\tfloor_id";
    const WEATHER_W1_CLASSID: u32 = 0x0401_0009;

    fn manifest(variable0: &str) -> FieldManifest {
        FieldManifest::parse(&format!(
            "{HEADER}\n\
             0\t0\tlo\t{variable0}\tsurface\tK\tf0\n\
             1\t0\tlo\tf1\t850\tK\tf1\n\
             2\t0\tlo\tf2\t500\tK\tf2\n"
        ))
        .expect("fixture manifest")
    }

    fn floors() -> HashMap<String, CalibratedFloor> {
        let sample: Vec<f64> = (0..10_000).map(|i| i as f64 / 100.0).collect();
        let floor = calibrate(&sample).expect("fixture floor");
        HashMap::from([
            ("f0".to_string(), floor.clone()),
            ("f1".to_string(), floor.clone()),
            ("f2".to_string(), floor),
        ])
    }

    fn value(entry: &ManifestEntry) -> Option<f64> {
        match entry.variable.as_str() {
            "a" => Some(10.0),
            "b" => Some(90.0),
            "f1" => Some(20.0),
            "f2" => Some(30.0),
            _ => None,
        }
    }

    #[test]
    fn zero_classid_is_refused_before_any_row_can_be_published() {
        let err = pack_cell(0, 0, 0, &manifest("a"), &floors(), value)
            .expect_err("bootstrap classid must be rejected");
        assert_eq!(err, BakeError::ZeroClassId);
    }

    #[test]
    fn out_of_grid_cell_is_a_release_grade_error_not_a_debug_assert() {
        let err = pack_cell(
            WEATHER_W1_CLASSID,
            LAT_COUNT,
            LON_COUNT - 1,
            &manifest("a"),
            &floors(),
            value,
        )
        .expect_err("latitude one-past-end must fail");
        assert_eq!(
            err,
            BakeError::GridIndexOutOfRange {
                lat_idx: LAT_COUNT,
                lon_idx: LON_COUNT - 1,
            }
        );

        let err = pack_cell(
            WEATHER_W1_CLASSID,
            LAT_COUNT - 1,
            LON_COUNT,
            &manifest("a"),
            &floors(),
            value,
        )
        .expect_err("longitude one-past-end must fail");
        assert!(matches!(err, BakeError::GridIndexOutOfRange { .. }));
    }

    #[test]
    fn one_cell_contains_key_and_exactly_three_class_prefixed_facets() {
        let cell = pack_cell(
            WEATHER_W1_CLASSID,
            720,
            1439,
            &manifest("a"),
            &floors(),
            value,
        )
        .expect("last real cell packs");

        assert_eq!(&cell.key[0..4], &WEATHER_W1_CLASSID.to_le_bytes());
        for facet in &cell.facets {
            assert_eq!(&facet[0..4], &WEATHER_W1_CLASSID.to_le_bytes());
            // Only pair 0 low is occupied in the fixture. Everything after it
            // is reserved-zero, proving the row image does not invent values.
            assert!(facet[5..].iter().all(|b| *b == 0));
        }
        assert_eq!(cell.facet_image().len(), 48);
    }

    #[test]
    fn manifest_mutation_is_load_bearing_end_to_end() {
        let a = pack_cell(
            WEATHER_W1_CLASSID,
            10,
            20,
            &manifest("a"),
            &floors(),
            value,
        )
        .expect("a packs");
        let b = pack_cell(
            WEATHER_W1_CLASSID,
            10,
            20,
            &manifest("b"),
            &floors(),
            value,
        )
        .expect("b packs");

        assert_ne!(
            a.facet_image(),
            b.facet_image(),
            "changing the manifest's field identity must change a written byte when the source values differ"
        );
        // The changed byte is exactly the slot the manifest resolves, not a
        // label or metadata byte smuggled into the payload.
        assert_eq!(manifest("a").resolve("a", None), Some((0, 0, PairByte::Lo)));
    }

    #[test]
    fn sink_failure_stops_the_stream_without_building_a_global_vec() {
        #[derive(Debug, PartialEq, Eq)]
        struct Stop;

        let mut seen = 0usize;
        let err = bake_timestep(
            WEATHER_W1_CLASSID,
            &manifest("a"),
            &floors(),
            |_lat, _lon, entry| value(entry),
            |_cell| {
                seen += 1;
                if seen == 3 {
                    Err(Stop)
                } else {
                    Ok(())
                }
            },
        )
        .expect_err("sink deliberately stops at cell 3");

        assert_eq!(seen, 3);
        match err {
            BakeStreamError::Sink(Stop) => {}
            other => panic!("expected sink stop, got {other:?}"),
        }
    }
}
