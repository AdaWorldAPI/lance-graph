//! Opt-in bridge from the zero-dependency W1 image to the live canonical
//! `lance-graph-contract::canonical_node::NodeRow`.
//!
//! The important property is what is *not* a constant here: the weather
//! extension offset. It is derived from the final descriptor in
//! `VALUE_TENANTS`, so append-only tenants added ahead of weather move this
//! extension automatically without a second source of truth.

use std::collections::HashMap;
use std::fmt;
use std::io::{self, Write};

use lance_graph_contract::canonical_node::{
    EdgeBlock, NodeGuid, NodeRow, NODE_ROW_STRIDE, VALUE_SLAB_LEN, VALUE_SLAB_ROW_OFFSET,
    VALUE_TENANTS,
};

use crate::bake::{bake_timestep, BakeError, BakeStreamError, PackedWeatherCell, W1_IMAGE_LEN};
use crate::floor::CalibratedFloor;
use crate::manifest::{FieldManifest, ManifestEntry};

/// Offset of the first currently unassigned byte **within `NodeRow::value`**.
///
/// Derived from the live append-only tenant table. Weather never owns a copied
/// absolute offset.
pub fn weather_value_offset() -> usize {
    let last = VALUE_TENANTS
        .last()
        .expect("canonical NodeRow must expose at least one value tenant");
    last.row_offset as usize + last.col_bytes_per_row() - VALUE_SLAB_ROW_OFFSET
}

/// Bytes still available in the canonical value slab after all named tenants.
pub fn weather_tail_capacity() -> usize {
    VALUE_SLAB_LEN - weather_value_offset()
}

/// A canonical-row assembly failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalRowError {
    /// The live contract has grown until W1's three facets no longer fit.
    InsufficientTail {
        /// W1 bytes required.
        needed: usize,
        /// Live free-tail capacity.
        available: usize,
    },
    /// `weather-poc::key` and the real `NodeGuid` no longer agree byte-for-byte.
    KeyContractDrift,
    /// A facet's classid prefix disagrees with the cell key's classid.
    FacetClassIdMismatch {
        /// Facet index 0..2.
        facet: usize,
        /// Classid carried by the cell key.
        key_classid: u32,
        /// Classid carried by the facet.
        facet_classid: u32,
    },
}

impl fmt::Display for CanonicalRowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CanonicalRowError::InsufficientTail { needed, available } => write!(
                f,
                "W1 weather image needs {needed} value bytes but the live NodeRow free tail has {available}"
            ),
            CanonicalRowError::KeyContractDrift => f.write_str(
                "weather-poc key bytes no longer agree with lance-graph-contract::NodeGuid",
            ),
            CanonicalRowError::FacetClassIdMismatch {
                facet,
                key_classid,
                facet_classid,
            } => write!(
                f,
                "weather facet {facet} classid 0x{facet_classid:08x} disagrees with key classid 0x{key_classid:08x}"
            ),
        }
    }
}

impl std::error::Error for CanonicalRowError {}

/// Error while streaming a whole timestep directly to a byte writer.
#[derive(Debug)]
pub enum CanonicalBakeError {
    /// Source/lane assembly failed.
    Bake(BakeError),
    /// Live canonical-row agreement or capacity failed.
    Canonical(CanonicalRowError),
    /// The output writer failed.
    Io(io::Error),
}

impl fmt::Display for CanonicalBakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CanonicalBakeError::Bake(err) => err.fmt(f),
            CanonicalBakeError::Canonical(err) => err.fmt(f),
            CanonicalBakeError::Io(err) => write!(f, "weather slab write failed: {err}"),
        }
    }
}

impl std::error::Error for CanonicalBakeError {}

fn key_as_node_guid(bytes: &[u8; 16]) -> Result<NodeGuid, CanonicalRowError> {
    let classid = u32::from_le_bytes(bytes[0..4].try_into().expect("4-byte classid"));
    let heel = u16::from_le_bytes([bytes[4], bytes[5]]);
    let hip = u16::from_le_bytes([bytes[6], bytes[7]]);
    let twig = u16::from_le_bytes([bytes[8], bytes[9]]);
    let family = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], 0]);
    let identity = u32::from_le_bytes([bytes[13], bytes[14], bytes[15], 0]);
    let guid = NodeGuid::new(classid, heel, hip, twig, family, identity);
    if guid.as_bytes() != bytes {
        return Err(CanonicalRowError::KeyContractDrift);
    }
    Ok(guid)
}

/// Assemble one [`PackedWeatherCell`] into the current canonical 512-byte row.
///
/// Existing value tenants are left zero. W1's three facets are placed only in
/// the currently free append-only tail, whose offset is derived at runtime from
/// the canonical descriptor table.
pub fn assemble_row(cell: &PackedWeatherCell) -> Result<NodeRow, CanonicalRowError> {
    let offset = weather_value_offset();
    let available = weather_tail_capacity();
    if W1_IMAGE_LEN > available {
        return Err(CanonicalRowError::InsufficientTail {
            needed: W1_IMAGE_LEN,
            available,
        });
    }

    let key = key_as_node_guid(&cell.key)?;
    let key_classid = key.classid();
    for (facet, bytes) in cell.facets.iter().enumerate() {
        let facet_classid = u32::from_le_bytes(bytes[0..4].try_into().expect("4-byte classid"));
        if facet_classid != key_classid {
            return Err(CanonicalRowError::FacetClassIdMismatch {
                facet,
                key_classid,
                facet_classid,
            });
        }
    }

    let mut value = [0u8; VALUE_SLAB_LEN];
    let image = cell.facet_image();
    value[offset..offset + W1_IMAGE_LEN].copy_from_slice(&image);

    Ok(NodeRow {
        key,
        edges: EdgeBlock::default(),
        value,
    })
}

/// Serialize one canonical row without an unsafe cast.
///
/// This makes the copy boundary explicit: key, edge block and value slab are
/// copied once into the 512-byte publication image. A later Arrow zero-copy
/// path may replace this only after measurement.
pub fn row_bytes(row: &NodeRow) -> [u8; NODE_ROW_STRIDE] {
    let mut out = [0u8; NODE_ROW_STRIDE];
    out[0..16].copy_from_slice(row.key.as_bytes());
    out[16..28].copy_from_slice(&row.edges.in_family);
    out[28..32].copy_from_slice(&row.edges.out_family);
    out[VALUE_SLAB_ROW_OFFSET..].copy_from_slice(&row.value);
    out
}

/// Stream a complete W1 timestep as canonical 512-byte rows into `writer`.
///
/// The function creates no global staging vector. It is the production-neutral
/// seam consumed by the existing Lance import/publication path: publication
/// must still be atomic at the dataset/version layer, so callers should write
/// to a temporary/unpublished target and only publish after this returns
/// successfully.
pub fn bake_timestep_to_writer<W, F>(
    classid: u32,
    manifest: &FieldManifest,
    floors: &HashMap<String, CalibratedFloor>,
    mut value_of: F,
    writer: &mut W,
) -> Result<usize, CanonicalBakeError>
where
    W: Write,
    F: FnMut(u16, u16, &ManifestEntry) -> Option<f64>,
{
    enum SinkError {
        Canonical(CanonicalRowError),
        Io(io::Error),
    }

    let result = bake_timestep(
        classid,
        manifest,
        floors,
        |lat, lon, entry| value_of(lat, lon, entry),
        |cell| {
            let row = assemble_row(&cell).map_err(SinkError::Canonical)?;
            writer
                .write_all(&row_bytes(&row))
                .map_err(SinkError::Io)
        },
    );

    match result {
        Ok(rows) => Ok(rows),
        Err(BakeStreamError::Bake(err)) => Err(CanonicalBakeError::Bake(err)),
        Err(BakeStreamError::Sink(SinkError::Canonical(err))) => {
            Err(CanonicalBakeError::Canonical(err))
        }
        Err(BakeStreamError::Sink(SinkError::Io(err))) => Err(CanonicalBakeError::Io(err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::pack_cell;
    use crate::floor::calibrate;

    const HEADER: &str = "facet\tpair\tbyte\tvariable\tlevel_hpa\tunit\tfloor_id";

    fn fixture_manifest() -> FieldManifest {
        FieldManifest::parse(&format!(
            "{HEADER}\n\
             0\t0\tlo\ta\tsurface\tK\tf\n\
             1\t0\tlo\tb\t850\tK\tf\n\
             2\t0\tlo\tc\t500\tK\tf\n"
        ))
        .expect("fixture manifest")
    }

    fn fixture_floors() -> HashMap<String, CalibratedFloor> {
        let sample: Vec<f64> = (0..10_000).map(|i| i as f64 / 100.0).collect();
        HashMap::from([(
            "f".to_string(),
            calibrate(&sample).expect("fixture floor"),
        )])
    }

    #[test]
    fn live_contract_places_weather_after_every_named_tenant() {
        let manifest = fixture_manifest();
        let floors = fixture_floors();
        let cell = pack_cell(0x0F01_0001, 720, 1439, &manifest, &floors, |entry| {
            match entry.variable.as_str() {
                "a" => Some(10.0),
                "b" => Some(20.0),
                "c" => Some(30.0),
                _ => None,
            }
        })
        .expect("cell packs");

        let row = assemble_row(&cell).expect("canonical row assembles");
        let offset = weather_value_offset();
        let image = cell.facet_image();

        assert_eq!(row.key.as_bytes(), &cell.key);
        assert!(row.edges.in_family.iter().all(|b| *b == 0));
        assert!(row.edges.out_family.iter().all(|b| *b == 0));
        assert!(row.value[..offset].iter().all(|b| *b == 0));
        assert_eq!(&row.value[offset..offset + W1_IMAGE_LEN], &image);
        assert!(row.value[offset + W1_IMAGE_LEN..].iter().all(|b| *b == 0));
        assert!(weather_tail_capacity() >= W1_IMAGE_LEN);
    }

    #[test]
    fn canonical_serialization_is_exactly_512_bytes_and_key_agrees() {
        let manifest = fixture_manifest();
        let floors = fixture_floors();
        let cell = pack_cell(0x0F01_0001, 123, 456, &manifest, &floors, |_| Some(42.0))
            .expect("cell packs");
        let row = assemble_row(&cell).expect("row");
        let bytes = row_bytes(&row);

        assert_eq!(bytes.len(), NODE_ROW_STRIDE);
        assert_eq!(&bytes[..16], &cell.key);
        assert_eq!(&bytes[VALUE_SLAB_ROW_OFFSET..], &row.value);
    }

    #[test]
    fn facet_classid_tamper_is_rejected() {
        let manifest = fixture_manifest();
        let floors = fixture_floors();
        let mut cell = pack_cell(0x0F01_0001, 1, 2, &manifest, &floors, |_| Some(42.0))
            .expect("cell packs");
        cell.facets[1][0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());

        assert!(matches!(
            assemble_row(&cell),
            Err(CanonicalRowError::FacetClassIdMismatch { facet: 1, .. })
        ));
    }
}
