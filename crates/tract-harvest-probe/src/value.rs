//! Tensor values on Property positions, and the decoding of source tensors.
//!
//! `tensor(bool)` is the packed mask every `ndarray::simd` mask kernel uses
//! (bit `i % 64` of word `i / 64` is element `i`), the `lance-graph-onnx`
//! convention. A bool tensor in the source is decoded into that form here,
//! once, and the packing is the only conversion (falsifier 5 of #1802).

use ndarray::{ArrayD, IxDyn};
use tract_onnx::pb;

/// Element types a Property position can carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DType {
    F32,
    F64,
    F16,
    I32,
    I64,
    Bool,
    /// Any other `TensorProto.DataType`, by its number.
    Other(i32),
}

impl DType {
    /// From `TensorProto.DataType`.
    pub fn from_onnx(t: i32) -> Self {
        match t {
            1 => DType::F32,
            6 => DType::I32,
            7 => DType::I64,
            9 => DType::Bool,
            10 => DType::F16,
            11 => DType::F64,
            o => DType::Other(o),
        }
    }
}

/// A value bound to a Property position.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    F32(ArrayD<f32>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    /// Packed `tensor(bool)`: logical shape plus mask words.
    Bool {
        shape: Vec<usize>,
        words: Vec<u64>,
    },
}

impl Value {
    pub fn dtype(&self) -> DType {
        match self {
            Value::F32(_) => DType::F32,
            Value::I32(_) => DType::I32,
            Value::I64(_) => DType::I64,
            Value::Bool { .. } => DType::Bool,
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Value::F32(a) => a.shape().to_vec(),
            Value::I32(a) => a.shape().to_vec(),
            Value::I64(a) => a.shape().to_vec(),
            Value::Bool { shape, .. } => shape.clone(),
        }
    }

    /// Every element as `f32`, for reading a small constant (an exponent, an
    /// epsilon); `None` for a non-float value.
    pub fn f32s(&self) -> Option<Vec<f32>> {
        match self {
            Value::F32(a) => Some(a.iter().copied().collect()),
            _ => None,
        }
    }

    /// Every element as `i64` for an integer value (axes, indices).
    pub fn i64s(&self) -> Option<Vec<i64>> {
        match self {
            Value::I64(a) => Some(a.iter().copied().collect()),
            Value::I32(a) => Some(a.iter().map(|&v| i64::from(v)).collect()),
            _ => None,
        }
    }
}

fn le<const N: usize>(raw: &[u8]) -> impl Iterator<Item = [u8; N]> + '_ {
    raw.as_chunks::<N>().0.iter().copied()
}

/// Decode an inline `TensorProto` into a [`Value`]. `None` for external data
/// (its bytes are not in the protobuf) or a dtype no Property carries.
pub fn decode(t: &pb::TensorProto) -> Option<Value> {
    if t.data_location == Some(pb::tensor_proto::DataLocation::External as i32) {
        return None;
    }
    let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
    let raw = &t.raw_data;
    let arr = |n: usize| shape.iter().product::<usize>() == n;
    match DType::from_onnx(t.data_type) {
        DType::F32 => {
            let v: Vec<f32> = if raw.is_empty() {
                t.float_data.clone()
            } else {
                le::<4>(raw).map(f32::from_le_bytes).collect()
            };
            ArrayD::from_shape_vec(IxDyn(&shape), v)
                .ok()
                .map(Value::F32)
        }
        DType::I32 => {
            let v: Vec<i32> = if raw.is_empty() {
                t.int32_data.clone()
            } else {
                le::<4>(raw).map(i32::from_le_bytes).collect()
            };
            ArrayD::from_shape_vec(IxDyn(&shape), v)
                .ok()
                .map(Value::I32)
        }
        DType::I64 => {
            let v: Vec<i64> = if raw.is_empty() {
                t.int64_data.clone()
            } else {
                le::<8>(raw).map(i64::from_le_bytes).collect()
            };
            ArrayD::from_shape_vec(IxDyn(&shape), v)
                .ok()
                .map(Value::I64)
        }
        DType::Bool => {
            // ONNX stores bool as one byte per element (raw) or int32_data.
            let bits: Vec<bool> = if raw.is_empty() {
                t.int32_data.iter().map(|&v| v != 0).collect()
            } else {
                raw.iter().map(|&b| b != 0).collect()
            };
            if !arr(bits.len()) {
                return None;
            }
            let mut words = vec![0u64; bits.len().div_ceil(64)];
            for (i, b) in bits.iter().enumerate() {
                words[i / 64] |= u64::from(*b) << (i % 64);
            }
            Some(Value::Bool { shape, words })
        }
        _ => None,
    }
}
