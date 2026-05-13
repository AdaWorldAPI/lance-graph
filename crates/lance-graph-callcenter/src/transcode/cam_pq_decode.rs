//! Decode-on-read shim for persistent-SoA columns.
//!
//! Each outer-ontology column maps to a [`CodecRoute`] from
//! [`lance_graph_contract::cam`]:
//!
//! - `Passthrough` — exact-identity f32, zero-copy on read.
//! - `CamPq` — 6-byte product-quantised codes, decoded on read.
//! - `Skip` — no codec; raw bytes (Utf8, scalar columns).
//!
//! This module is the **dispatch contract** — domain-agnostic, reusable
//! across every ontology that lands a Lance dataset. The actual PQ math
//! lives in `lance_graph_contract::cam` and `lance_graph_planner::physical::cam_pq_scan`;
//! this trait only carries the read-side shape downstream consumers
//! interact with.

use lance_graph_contract::cam::CodecRoute;

use super::zerocopy::OuterColumn;

/// Pick the `CodecRoute` for an outer-ontology column.
///
/// Returns the column's declarative route copied straight from the
/// upstream `PropertySpec.codec_route`. This is round-2 of the dispatch
/// path: round-1 inferred the route by feeding the column name through
/// `lance_graph_contract::cam::route_tensor`, which is calibrated for
/// model-weight tensor names (`q_proj`, `lm_head`, …) and would
/// silently mis-classify document predicates (`patient_id`, `iban`, …).
///
/// The schema layer already declares each property's route; honouring
/// that field is correct by construction.
pub fn route_for_column(col: &OuterColumn) -> CodecRoute {
    col.codec_route
}

/// Read-side decoder. Implementations get one row's worth of bytes and
/// hand back the f32 representation the consumer expects.
pub trait CamPqDecoder: Send + Sync {
    fn decode_row(&self, encoded: &[u8], out: &mut [f32]) -> Result<usize, DecodeError>;
}

/// Errors from the decode path.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodeError {
    BadStride {
        expected: usize,
        got: usize,
    },
    OutputTooSmall {
        needed: usize,
        got: usize,
    },
    /// Codebook not registered for the column's route — caller should
    /// either register one or fall back to the [`PassthroughDecoder`].
    NoCodebook,
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodeError::BadStride { expected, got } => {
                write!(f, "bad stride: expected {expected} bytes, got {got}")
            }
            DecodeError::OutputTooSmall { needed, got } => {
                write!(f, "output buffer too small: need {needed}, got {got}")
            }
            DecodeError::NoCodebook => {
                write!(f, "no codebook registered for column's CAM-PQ route")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Trivial decoder that re-interprets encoded bytes as little-endian f32.
/// Used for `CodecRoute::Passthrough` / `Skip` columns where no codec
/// transform happened.
#[derive(Debug, Default, Clone, Copy)]
pub struct PassthroughDecoder;

impl CamPqDecoder for PassthroughDecoder {
    fn decode_row(&self, encoded: &[u8], out: &mut [f32]) -> Result<usize, DecodeError> {
        let needed = out.len() * 4;
        if encoded.len() < needed {
            return Err(DecodeError::BadStride {
                expected: needed,
                got: encoded.len(),
            });
        }
        for (i, chunk) in encoded.chunks_exact(4).take(out.len()).enumerate() {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(out.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcode::zerocopy::OuterSchema;
    use lance_graph_contract::ontology::Ontology;
    use lance_graph_contract::property::Schema;

    #[test]
    fn route_for_required_scalar_column_uses_property_spec_default() {
        // PropertySpec::required() defaults `codec_route` to
        // `CodecRoute::Passthrough` (Index regime). The transcode-layer
        // dispatcher must honour that — round-1's heuristic returned
        // Skip for scalars by guessing from the Arrow shape, which
        // diverged from the contract's own field. Round-2 just reads
        // `OuterColumn.codec_route`, so the result is whatever the
        // schema declared.
        let ont = Ontology::builder("T")
            .schema(Schema::builder("Patient").required("name").build())
            .build();
        let soa = OuterSchema::from_ontology(&ont, "Patient").unwrap();
        assert_eq!(
            route_for_column(&soa.columns[0]),
            CodecRoute::Passthrough,
            "required scalar default route should be Passthrough"
        );
    }

    #[test]
    fn passthrough_decoder_round_trips_le_f32() {
        let xs: [f32; 4] = [1.0, -2.5, 3.25, 0.0];
        let bytes: Vec<u8> = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut out = [0.0f32; 4];
        let n = PassthroughDecoder.decode_row(&bytes, &mut out).unwrap();
        assert_eq!(n, 4);
        assert_eq!(out, xs);
    }

    #[test]
    fn passthrough_decoder_rejects_short_input() {
        let mut out = [0.0f32; 4];
        let err = PassthroughDecoder
            .decode_row(&[0u8; 8], &mut out)
            .unwrap_err();
        assert!(matches!(
            err,
            DecodeError::BadStride {
                expected: 16,
                got: 8
            }
        ));
    }
}
