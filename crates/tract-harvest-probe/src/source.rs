//! Source evidence read from the raw ONNX protobuf, never from Tract's graph.
//!
//! Tract keeps node names and output labels, but it looks operators up by
//! `op_type` alone, deduplicates equal constants and synthesizes names. So
//! everything admission must not lose (domain, opset, initializer identity and
//! digest, external-data placement) is read here, from the bytes the model was
//! loaded from.

use sha2::{Digest, Sha256};
use tract_onnx::pb;

/// SHA-256 of a byte string, lower-case hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// One source node, as the protobuf states it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceNode {
    /// `NodeProto.name`; may be empty in real exports.
    pub name: String,
    pub op_type: String,
    /// `""` and `"ai.onnx"` are the default domain.
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<String>,
}

impl SourceNode {
    pub fn default_domain(&self) -> bool {
        self.domain.is_empty() || self.domain == "ai.onnx"
    }
}

/// Where an initializer's bytes are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    /// Inline bytes, with their SHA-256.
    Inline { sha256: String },
    /// External data: file, byte offset, byte length.
    External {
        location: String,
        offset: u64,
        length: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceInitializer {
    pub name: String,
    pub dims: Vec<i64>,
    /// `TensorProto.data_type` (1 = FLOAT, 6 = INT32, 7 = INT64).
    pub data_type: i32,
    pub placement: Placement,
}

/// What the model file says, independent of any importer.
#[derive(Debug, Clone)]
pub struct SourceInventory {
    pub model_sha256: String,
    pub ir_version: i64,
    /// `(domain, version)`, `""` normalized to `"ai.onnx"`.
    pub opsets: Vec<(String, i64)>,
    pub nodes: Vec<SourceNode>,
    pub initializers: Vec<SourceInitializer>,
}

impl SourceInventory {
    /// Read the inventory from model bytes. The protobuf is decoded with the
    /// same `prost` types Tract uses, but nothing of Tract's import runs.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, prost::DecodeError> {
        use prost::Message;
        let m = pb::ModelProto::decode(bytes)?;
        Ok(Self::from_proto(&m, sha256_hex(bytes)))
    }

    pub fn from_proto(m: &pb::ModelProto, model_sha256: String) -> Self {
        let g = m.graph.as_ref();
        let nodes = g
            .map(|g| {
                g.node
                    .iter()
                    .map(|n| SourceNode {
                        name: n.name.clone(),
                        op_type: n.op_type.clone(),
                        domain: n.domain.clone(),
                        inputs: n.input.clone(),
                        outputs: n.output.clone(),
                        attributes: n.attribute.iter().map(|a| a.name.clone()).collect(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let initializers = g
            .map(|g| g.initializer.iter().map(initializer).collect())
            .unwrap_or_default();
        let opsets = m
            .opset_import
            .iter()
            .map(|o| {
                let d = if o.domain.is_empty() {
                    "ai.onnx".to_string()
                } else {
                    o.domain.clone()
                };
                (d, o.version)
            })
            .collect();
        Self {
            model_sha256,
            ir_version: m.ir_version,
            opsets,
            nodes,
            initializers,
        }
    }

    /// The default-domain opset version, if declared.
    pub fn default_opset(&self) -> Option<i64> {
        self.opsets
            .iter()
            .find(|(d, _)| d == "ai.onnx")
            .map(|(_, v)| *v)
    }

    pub fn node_producing(&self, output: &str) -> Option<&SourceNode> {
        self.nodes
            .iter()
            .find(|n| n.outputs.iter().any(|o| o == output))
    }

    pub fn node_named(&self, name: &str) -> Option<&SourceNode> {
        self.nodes.iter().find(|n| n.name == name)
    }

    pub fn initializer(&self, name: &str) -> Option<&SourceInitializer> {
        self.initializers.iter().find(|i| i.name == name)
    }
}

fn initializer(t: &pb::TensorProto) -> SourceInitializer {
    let external = t.data_location == Some(pb::tensor_proto::DataLocation::External as i32);
    let placement = if external {
        let get = |k: &str| {
            t.external_data
                .iter()
                .find(|e| e.key == k)
                .map(|e| e.value.clone())
        };
        Placement::External {
            location: get("location").unwrap_or_default(),
            offset: get("offset").and_then(|v| v.parse().ok()).unwrap_or(0),
            length: get("length").and_then(|v| v.parse().ok()).unwrap_or(0),
        }
    } else {
        Placement::Inline {
            sha256: sha256_hex(&inline_bytes(t)),
        }
    };
    SourceInitializer {
        name: t.name.clone(),
        dims: t.dims.clone(),
        data_type: t.data_type,
        placement,
    }
}

/// The inline payload as stored: `raw_data` if present, else the typed field
/// re-encoded little-endian. Two encodings of the same values hash the same.
fn inline_bytes(t: &pb::TensorProto) -> Vec<u8> {
    if !t.raw_data.is_empty() {
        return t.raw_data.to_vec();
    }
    let mut v = Vec::new();
    t.float_data
        .iter()
        .for_each(|x| v.extend_from_slice(&x.to_le_bytes()));
    t.int32_data
        .iter()
        .for_each(|x| v.extend_from_slice(&x.to_le_bytes()));
    t.int64_data
        .iter()
        .for_each(|x| v.extend_from_slice(&x.to_le_bytes()));
    t.double_data
        .iter()
        .for_each(|x| v.extend_from_slice(&x.to_le_bytes()));
    v
}
