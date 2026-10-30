//! Source evidence read from the raw ONNX protobuf, never from Tract's graph.
//!
//! Tract keeps node names and output labels, but it looks operators up by
//! `op_type` alone, deduplicates equal constants and synthesizes names. So
//! everything admission must not lose (domain, opset, initializer identity and
//! digest, external-data placement) is read here, from the bytes the model was
//! loaded from.

use sha2::{Digest, Sha256};
use tract_onnx::pb;

use crate::value::{decode, Value};

/// SHA-256 of a byte string, lower-case hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// One attribute value, as the protobuf states it.
#[derive(Debug, Clone, PartialEq)]
pub enum Attr {
    Int(i64),
    Float(f32),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Str(String),
    /// A tensor attribute (a `Constant`'s value), decoded; `None` if the
    /// tensor's dtype has no [`Value`].
    Tensor(Option<Value>),
    /// Graph, sparse tensor, type proto or other: kept by name only.
    Other(i32),
}

impl Attr {
    fn from_proto(a: &pb::AttributeProto) -> Self {
        match a.r#type {
            1 => Attr::Float(a.f),
            2 => Attr::Int(a.i),
            3 => Attr::Str(String::from_utf8_lossy(&a.s).into_owned()),
            4 => Attr::Tensor(a.t.as_ref().and_then(decode)),
            6 => Attr::Floats(a.floats.clone()),
            7 => Attr::Ints(a.ints.clone()),
            t => Attr::Other(t),
        }
    }
}

/// One source node, as the protobuf states it.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceNode {
    /// `NodeProto.name`; may be empty in real exports.
    pub name: String,
    pub op_type: String,
    /// `""` and `"ai.onnx"` are the default domain.
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<String>,
    /// The attribute values, in source order.
    pub attrs: Vec<(String, Attr)>,
}

impl SourceNode {
    pub fn default_domain(&self) -> bool {
        self.domain.is_empty() || self.domain == "ai.onnx"
    }

    pub fn attr(&self, name: &str) -> Option<&Attr> {
        self.attrs.iter().find(|(n, _)| n == name).map(|(_, a)| a)
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

#[derive(Debug, Clone, PartialEq)]
pub struct SourceInitializer {
    pub name: String,
    pub dims: Vec<i64>,
    /// `TensorProto.data_type` (1 = FLOAT, 6 = INT32, 7 = INT64).
    pub data_type: i32,
    pub placement: Placement,
    /// The decoded value of an inline initializer; `None` for external data.
    pub value: Option<Value>,
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
    /// Graph inputs that are not initializers, in source order.
    pub inputs: Vec<String>,
    /// Graph outputs, in source order.
    pub outputs: Vec<String>,
    /// One SHA-256 per external `Constant` inlined after decoding, over its
    /// output name, dtype, shape and little-endian contents, in inlining
    /// order. The bytes were not part of `model_sha256`, so a plan's source
    /// revision includes these.
    pub inlined: Vec<String>,
}

/// SHA-256 over an inlined constant's output name, dtype, shape and
/// little-endian contents.
fn inlined_digest(output: &str, value: &Value) -> String {
    let mut b = Vec::new();
    let mut field = |f: &[u8]| {
        b.extend((f.len() as u64).to_le_bytes());
        b.extend_from_slice(f);
    };
    field(output.as_bytes());
    field(format!("{:?}", value.dtype()).as_bytes());
    let shape: Vec<u8> = value
        .shape()
        .iter()
        .flat_map(|d| (*d as u64).to_le_bytes())
        .collect();
    field(&shape);
    let data: Vec<u8> = match value {
        Value::F32(a) => a.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Value::I32(a) => a.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Value::I64(a) => a.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Value::Bool { words, .. } => words.iter().flat_map(|v| v.to_le_bytes()).collect(),
    };
    field(&data);
    sha256_hex(&b)
}

impl SourceInventory {
    /// Inline every external-data `Constant` listed in a manifest written by
    /// `tools/fetch_jina_constants.py` (`{"constants": [{output, dtype, dims,
    /// sha256, file}]}`), reading each file from `data_dir` and refusing a
    /// digest, length or dtype that does not match. Returns how many were
    /// inlined.
    pub fn inline_constants_from_manifest(
        &mut self,
        manifest: &std::path::Path,
        data_dir: &std::path::Path,
    ) -> Result<usize, String> {
        use sha2::{Digest, Sha256};
        let m: serde_json::Value = serde_json::from_slice(
            &std::fs::read(manifest).map_err(|e| format!("{}: {e}", manifest.display()))?,
        )
        .map_err(|e| e.to_string())?;
        let list = m["constants"]
            .as_array()
            .ok_or("manifest without constants")?;
        for c in list {
            let field = |k: &str| c[k].as_str().ok_or(format!("constant without {k}"));
            let (output, file, digest) = (field("output")?, field("file")?, field("sha256")?);
            let bytes = std::fs::read(data_dir.join(file)).map_err(|e| format!("{file}: {e}"))?;
            let got: String = Sha256::digest(&bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            if got != digest {
                return Err(format!("{file}: sha256 {got} != {digest}"));
            }
            let dims: Vec<usize> = c["dims"]
                .as_array()
                .ok_or("constant without dims")?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize).ok_or("bad dim"))
                .collect::<Result<_, _>>()?;
            let n: usize = dims.iter().product();
            let value = match field("dtype")? {
                "f32" if bytes.len() == 4 * n => Value::F32(
                    ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(&dims),
                        bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|b| f32::from_le_bytes(*b))
                            .collect(),
                    )
                    .map_err(|e| e.to_string())?,
                ),
                "i64" if bytes.len() == 8 * n => Value::I64(
                    ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(&dims),
                        bytes
                            .as_chunks::<8>()
                            .0
                            .iter()
                            .map(|b| i64::from_le_bytes(*b))
                            .collect(),
                    )
                    .map_err(|e| e.to_string())?,
                ),
                t => {
                    return Err(format!(
                        "{output}: {t} with {} bytes for {dims:?}",
                        bytes.len()
                    ))
                }
            };
            self.inline_external_constant(output, value)?;
        }
        Ok(list.len())
    }

    /// Supply the value of a `Constant` whose tensor lives in external data
    /// (the source decoder leaves it `None`). Refused unless `output` is the
    /// output of exactly one default-domain `Constant` with an undecoded
    /// tensor; the caller is responsible for the bytes (a pinned range).
    pub fn inline_external_constant(&mut self, output: &str, value: Value) -> Result<(), String> {
        let mut hits = self.nodes.iter_mut().filter(|n| {
            n.op_type == "Constant"
                && n.default_domain()
                && n.outputs.first().is_some_and(|o| o == output)
        });
        let (Some(node), None) = (hits.next(), hits.next()) else {
            return Err(format!("{output}: not the output of exactly one Constant"));
        };
        match node.attrs.iter_mut().find(|(n, _)| n == "value") {
            Some((_, a @ Attr::Tensor(None))) => {
                let digest = inlined_digest(output, &value);
                *a = Attr::Tensor(Some(value));
                self.inlined.push(digest);
                Ok(())
            }
            Some((_, Attr::Tensor(Some(_)))) => Err(format!("{output}: value is already decoded")),
            _ => Err(format!("{output}: no tensor value attribute")),
        }
    }

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
                        attrs: n
                            .attribute
                            .iter()
                            .map(|a| (a.name.clone(), Attr::from_proto(a)))
                            .collect(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let initializers: Vec<SourceInitializer> = g
            .map(|g| g.initializer.iter().map(initializer).collect())
            .unwrap_or_default();
        let inputs = g
            .map(|g| {
                g.input
                    .iter()
                    .filter(|i| !initializers.iter().any(|t| t.name == i.name))
                    .map(|i| i.name.clone())
                    .collect()
            })
            .unwrap_or_default();
        let outputs = g
            .map(|g| g.output.iter().map(|o| o.name.clone()).collect())
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
            inputs,
            outputs,
            inlined: vec![],
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
        value: decode(t),
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
