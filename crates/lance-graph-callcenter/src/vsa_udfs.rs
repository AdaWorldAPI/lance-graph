//! RoleDB DataFusion VSA UDFs — DU-3.
//!
//! Five UDFs that make the internal fingerprint dataset queryable
//! as "DuckDB over roles." All operate on the L4/L5 speed-tier
//! fingerprint format: `FixedSizeBinary(2048)` = `[u64; 256]` = 16 Kbit.
//!
//! Precision note (§ 18 of callcenter-membrane-v1.md):
//! - These UDFs run at fingerprint precision (L4/L5 speed lane).
//! - They are correct for dispatch scoring, Hamming search, and Markov
//!   window queries over the hot internal dataset.
//! - Full algebraic unbind (role recovery, not just overlap scoring)
//!   requires Vsa10k BF16 from the L3 cold dataset — Phase B, deferred.
//!
//! SQL example:
//! ```sql
//! SELECT expert_id, vsa_unbind(2, fingerprint) AS n8n_overlap
//! FROM internal_dataset
//! WHERE round = (SELECT max(round) FROM internal_dataset)
//! ORDER BY n8n_overlap DESC
//! LIMIT 5;
//! ```
//!
//! Plan: `.claude/plans/unified-integration-v1.md` § DU-3
//!       `.claude/plans/callcenter-membrane-v1.md` §§ 15, 18

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, FixedSizeBinaryArray, FixedSizeBinaryBuilder, Float32Array,
    ListArray, UInt16Array, UInt32Array,
};
use arrow::buffer::{Buffer, OffsetBuffer};
use arrow::datatypes::{DataType, Field};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{ScalarUDF, ScalarUDFImpl, Signature, Volatility};
use datafusion::physical_plan::ColumnarValue;

// ── Constants ────────────────────────────────────────────────────────────────

/// Byte length of one `Fingerprint<256>` = `[u64; 256]`.
pub const FP_BYTES: i32 = 2048;
/// Word count of one fingerprint.
pub const FP_WORDS: usize = 256;
/// Number of ExternalRole variants (8) — determines per-role word slice.
const N_ROLES: usize = 8;
/// Words allocated per role slice: 256 / 8 = 32 words = 2048 bits.
const ROLE_SLICE_WORDS: usize = FP_WORDS / N_ROLES;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn bytes_to_words(bytes: &[u8]) -> [u64; FP_WORDS] {
    let mut words = [0u64; FP_WORDS];
    for (i, chunk) in bytes.chunks_exact(8).enumerate().take(FP_WORDS) {
        words[i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    words
}

fn words_to_bytes(words: &[u64; FP_WORDS]) -> [u8; FP_BYTES as usize] {
    let mut out = [0u8; FP_BYTES as usize];
    for (i, &w) in words.iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// Downcast to `FixedSizeBinaryArray` and validate byte length.
fn as_fp_array(col: &ArrayRef) -> DfResult<&FixedSizeBinaryArray> {
    let arr = col
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or_else(|| {
            DataFusionError::Execution("VSA UDF: expected FixedSizeBinary column".into())
        })?;
    if arr.value_length() != FP_BYTES {
        return Err(DataFusionError::Execution(format!(
            "VSA UDF: expected FixedSizeBinary({FP_BYTES}), got FixedSizeBinary({})",
            arr.value_length()
        )));
    }
    Ok(arr)
}

/// Materialise a ColumnarValue into an `ArrayRef`, broadcasting scalars to `len`.
fn to_array(cv: &ColumnarValue, len: usize) -> DfResult<ArrayRef> {
    match cv {
        ColumnarValue::Array(a) => Ok(Arc::clone(a)),
        ColumnarValue::Scalar(s) => s.to_array_of_size(len).map_err(|e| {
            DataFusionError::Execution(format!("VSA UDF scalar broadcast: {e}"))
        }),
    }
}

// ── Core operations ───────────────────────────────────────────────────────────

/// Overlap score of `role`-indexed slice in `fp`. Returns [0.0, 1.0].
///
/// Reads the word slice [role * ROLE_SLICE_WORDS .. (role+1) * ROLE_SLICE_WORDS]
/// and returns the set-bit fraction as a dispatch score.
///
/// CONJECTURE: this is a simplified overlap metric at fingerprint precision.
/// For algebraic unbind (role recovery), use Vsa10k BF16 at L3 — Phase B.
fn unbind_op(role: u8, fp_bytes: &[u8]) -> f32 {
    let words = bytes_to_words(fp_bytes);
    let role = (role as usize).min(N_ROLES - 1);
    let start = role * ROLE_SLICE_WORDS;
    let end = start + ROLE_SLICE_WORDS;
    let set_bits: u32 = words[start..end].iter().map(|w| w.count_ones()).sum();
    set_bits as f32 / (ROLE_SLICE_WORDS as f32 * 64.0)
}

/// XOR-bundle two fingerprints.
///
/// Phase A: XOR (single-writer delta merge, `MergeMode::Xor`).
/// Phase B: N-ary majority vote (`MergeMode::Bundle`, CK-safe) — deferred.
/// See I-SUBSTRATE-MARKOV in CLAUDE.md for why Bundle ≠ XOR.
fn bundle_op(a: &[u8], b: &[u8]) -> [u64; FP_WORDS] {
    let wa = bytes_to_words(a);
    let wb = bytes_to_words(b);
    let mut out = [0u64; FP_WORDS];
    for i in 0..FP_WORDS {
        out[i] = wa[i] ^ wb[i];
    }
    out
}

/// Popcount of XOR — number of differing bits between two fingerprints.
fn hamming_dist_op(a: &[u8], b: &[u8]) -> u32 {
    let wa = bytes_to_words(a);
    let wb = bytes_to_words(b);
    wa.iter().zip(wb.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Cyclic-rotate fingerprint words by `pos` positions.
///
/// Simulates the Markov braid offset: `braid_at(-5, fp)` retrieves
/// the fingerprint as it would appear at position HEAD-5 in the braid.
fn braid_at_op(pos: i32, fp_bytes: &[u8]) -> [u64; FP_WORDS] {
    let words = bytes_to_words(fp_bytes);
    let shift = pos.rem_euclid(FP_WORDS as i32) as usize;
    let mut out = [0u64; FP_WORDS];
    for i in 0..FP_WORDS {
        out[i] = words[(i + FP_WORDS - shift) % FP_WORDS];
    }
    out
}

/// Returns the indices of the `k` words with the most set bits (highest activation).
///
/// Each index is a u16 word-index [0, 255]. Caller maps index → persona coordinate
/// via the role-slice layout (see § 15 of callcenter-membrane-v1.md).
fn top_k_op(fp_bytes: &[u8], k: usize) -> Vec<u16> {
    let words = bytes_to_words(fp_bytes);
    let mut scored: Vec<(u32, u16)> = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.count_ones(), i as u16))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    scored.into_iter().take(k).map(|(_, idx)| idx).collect()
}

// ── UDF implementations ───────────────────────────────────────────────────────

// ─── vsa_unbind ──────────────────────────────────────────────────────────────

#[derive(Debug)]
struct UnbindUdf {
    signature: Signature,
}

impl datafusion::logical_expr::ScalarUDFImpl for UnbindUdf {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn name(&self) -> &str { "vsa_unbind" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> DfResult<DataType> { Ok(DataType::Float32) }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> DfResult<ColumnarValue> {
        let args = &args.args;
        if args.len() != 2 {
            return Err(DataFusionError::Execution(
                "vsa_unbind(role, fingerprint) requires 2 args".into(),
            ));
        }
        let len = match &args[0] {
            ColumnarValue::Array(a) => a.len(),
            ColumnarValue::Scalar(_) => match &args[1] {
                ColumnarValue::Array(a) => a.len(),
                ColumnarValue::Scalar(_) => 1,
            },
        };
        let role_arr = to_array(&args[0], len)?;
        let fp_arr = to_array(&args[1], len)?;

        let roles = role_arr.as_any().downcast_ref::<arrow::array::UInt8Array>()
            .ok_or_else(|| DataFusionError::Execution("vsa_unbind: arg 0 must be UInt8".into()))?;
        let fps = as_fp_array(&fp_arr)?;

        let results: Vec<f32> = (0..len)
            .map(|i| {
                if roles.is_null(i) || fps.is_null(i) {
                    f32::NAN
                } else {
                    unbind_op(roles.value(i), fps.value(i))
                }
            })
            .collect();

        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
    }
}

impl PartialEq for UnbindUdf { fn eq(&self, o: &Self) -> bool { self.name() == o.name() } }
impl Eq for UnbindUdf {}
impl std::hash::Hash for UnbindUdf {
    fn hash<H: std::hash::Hasher>(&self, s: &mut H) { self.name().hash(s); }
}

// ─── vsa_bundle ──────────────────────────────────────────────────────────────

#[derive(Debug)]
struct BundleUdf {
    signature: Signature,
}

impl datafusion::logical_expr::ScalarUDFImpl for BundleUdf {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn name(&self) -> &str { "vsa_bundle" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> DfResult<DataType> {
        Ok(DataType::FixedSizeBinary(FP_BYTES))
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> DfResult<ColumnarValue> {
        let args = &args.args;
        if args.len() != 2 {
            return Err(DataFusionError::Execution(
                "vsa_bundle(fp1, fp2) requires 2 args".into(),
            ));
        }
        let len = match &args[0] {
            ColumnarValue::Array(a) => a.len(),
            _ => 1,
        };
        let a_arr = as_fp_array(&to_array(&args[0], len)?)?
            .iter()
            .map(|v| v.map(|b| b.to_vec()))
            .collect::<Vec<_>>();
        let b_arr = as_fp_array(&to_array(&args[1], len)?)?
            .iter()
            .map(|v| v.map(|b| b.to_vec()))
            .collect::<Vec<_>>();

        let mut builder = FixedSizeBinaryBuilder::with_capacity(len, FP_BYTES);
        for i in 0..len {
            match (&a_arr[i], &b_arr[i]) {
                (Some(a), Some(b)) => {
                    let out = bundle_op(a, b);
                    builder.append_value(words_to_bytes(&out)).unwrap();
                }
                _ => builder.append_null(),
            }
        }
        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }
}

impl PartialEq for BundleUdf { fn eq(&self, o: &Self) -> bool { self.name() == o.name() } }
impl Eq for BundleUdf {}
impl std::hash::Hash for BundleUdf {
    fn hash<H: std::hash::Hasher>(&self, s: &mut H) { self.name().hash(s); }
}

// ─── vsa_hamming_dist ────────────────────────────────────────────────────────

#[derive(Debug)]
struct HammingDistUdf {
    signature: Signature,
}

impl datafusion::logical_expr::ScalarUDFImpl for HammingDistUdf {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn name(&self) -> &str { "vsa_hamming_dist" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> DfResult<DataType> { Ok(DataType::UInt32) }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> DfResult<ColumnarValue> {
        let args = &args.args;
        if args.len() != 2 {
            return Err(DataFusionError::Execution(
                "vsa_hamming_dist(a, b) requires 2 args".into(),
            ));
        }
        let len = match &args[0] {
            ColumnarValue::Array(a) => a.len(),
            _ => 1,
        };
        let a_fp = as_fp_array(&to_array(&args[0], len)?)?
            .iter()
            .map(|v| v.map(|b| b.to_vec()))
            .collect::<Vec<_>>();
        let b_fp = as_fp_array(&to_array(&args[1], len)?)?
            .iter()
            .map(|v| v.map(|b| b.to_vec()))
            .collect::<Vec<_>>();

        let results: Vec<Option<u32>> = (0..len)
            .map(|i| match (&a_fp[i], &b_fp[i]) {
                (Some(a), Some(b)) => Some(hamming_dist_op(a, b)),
                _ => None,
            })
            .collect();

        Ok(ColumnarValue::Array(Arc::new(UInt32Array::from(results))))
    }
}

impl PartialEq for HammingDistUdf { fn eq(&self, o: &Self) -> bool { self.name() == o.name() } }
impl Eq for HammingDistUdf {}
impl std::hash::Hash for HammingDistUdf {
    fn hash<H: std::hash::Hasher>(&self, s: &mut H) { self.name().hash(s); }
}

// ─── vsa_braid_at ────────────────────────────────────────────────────────────

#[derive(Debug)]
struct BraidAtUdf {
    signature: Signature,
}

impl datafusion::logical_expr::ScalarUDFImpl for BraidAtUdf {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn name(&self) -> &str { "vsa_braid_at" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> DfResult<DataType> {
        Ok(DataType::FixedSizeBinary(FP_BYTES))
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> DfResult<ColumnarValue> {
        let args = &args.args;
        if args.len() != 2 {
            return Err(DataFusionError::Execution(
                "vsa_braid_at(pos, fingerprint) requires 2 args".into(),
            ));
        }
        let len = match &args[0] {
            ColumnarValue::Array(a) => a.len(),
            _ => match &args[1] {
                ColumnarValue::Array(a) => a.len(),
                _ => 1,
            },
        };
        let pos_arr = to_array(&args[0], len)?;
        let fp_arr = to_array(&args[1], len)?;

        let pos_col = pos_arr.as_any().downcast_ref::<arrow::array::Int32Array>()
            .ok_or_else(|| DataFusionError::Execution("vsa_braid_at: arg 0 must be Int32".into()))?;
        let fps = as_fp_array(&fp_arr)?
            .iter()
            .map(|v| v.map(|b| b.to_vec()))
            .collect::<Vec<_>>();

        let mut builder = FixedSizeBinaryBuilder::with_capacity(len, FP_BYTES);
        for i in 0..len {
            if pos_col.is_null(i) || fps[i].is_none() {
                builder.append_null();
            } else {
                let out = braid_at_op(pos_col.value(i), fps[i].as_ref().unwrap());
                builder.append_value(words_to_bytes(&out)).unwrap();
            }
        }
        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }
}

impl PartialEq for BraidAtUdf { fn eq(&self, o: &Self) -> bool { self.name() == o.name() } }
impl Eq for BraidAtUdf {}
impl std::hash::Hash for BraidAtUdf {
    fn hash<H: std::hash::Hasher>(&self, s: &mut H) { self.name().hash(s); }
}

// ─── vsa_top_k ───────────────────────────────────────────────────────────────

#[derive(Debug)]
struct TopKUdf {
    signature: Signature,
    return_type: DataType,
}

impl datafusion::logical_expr::ScalarUDFImpl for TopKUdf {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn name(&self) -> &str { "vsa_top_k" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _: &[DataType]) -> DfResult<DataType> { Ok(self.return_type.clone()) }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> DfResult<ColumnarValue> {
        let args = &args.args;
        if args.len() != 2 {
            return Err(DataFusionError::Execution(
                "vsa_top_k(fingerprint, k) requires 2 args".into(),
            ));
        }
        let len = match &args[0] {
            ColumnarValue::Array(a) => a.len(),
            _ => 1,
        };
        let fp_arr = as_fp_array(&to_array(&args[0], len)?)?
            .iter()
            .map(|v| v.map(|b| b.to_vec()))
            .collect::<Vec<_>>();
        let k_arr = to_array(&args[1], len)?;
        let k_col = k_arr.as_any().downcast_ref::<UInt32Array>()
            .ok_or_else(|| DataFusionError::Execution("vsa_top_k: arg 1 must be UInt32".into()))?;

        // Build ListArray of UInt16
        let mut all_values: Vec<u16> = Vec::new();
        let mut offsets: Vec<i32> = vec![0];

        for i in 0..len {
            if fp_arr[i].is_none() || k_col.is_null(i) {
                offsets.push(*offsets.last().unwrap());
            } else {
                let k = k_col.value(i) as usize;
                let indices = top_k_op(fp_arr[i].as_ref().unwrap(), k);
                all_values.extend_from_slice(&indices);
                offsets.push(*offsets.last().unwrap() + indices.len() as i32);
            }
        }

        let values = Arc::new(UInt16Array::from(all_values)) as ArrayRef;
        let item_field = Arc::new(Field::new("item", DataType::UInt16, true));
        let offsets = OffsetBuffer::new(Buffer::from_vec(offsets).into());
        let list = ListArray::new(item_field, offsets, values, None);

        Ok(ColumnarValue::Array(Arc::new(list)))
    }
}

impl PartialEq for TopKUdf { fn eq(&self, o: &Self) -> bool { self.name() == o.name() } }
impl Eq for TopKUdf {}
impl std::hash::Hash for TopKUdf {
    fn hash<H: std::hash::Hasher>(&self, s: &mut H) { self.name().hash(s); }
}

// ── Public constructors ───────────────────────────────────────────────────────

pub fn vsa_unbind_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(UnbindUdf {
        signature: Signature::any(2, Volatility::Immutable),
    }))
}

pub fn vsa_bundle_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(BundleUdf {
        signature: Signature::any(2, Volatility::Immutable),
    }))
}

pub fn vsa_hamming_dist_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(HammingDistUdf {
        signature: Signature::any(2, Volatility::Immutable),
    }))
}

pub fn vsa_braid_at_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(BraidAtUdf {
        signature: Signature::any(2, Volatility::Immutable),
    }))
}

pub fn vsa_top_k_udf() -> Arc<ScalarUDF> {
    let item_field = Arc::new(Field::new("item", DataType::UInt16, true));
    Arc::new(ScalarUDF::new_from_impl(TopKUdf {
        signature: Signature::any(2, Volatility::Immutable),
        return_type: DataType::List(item_field),
    }))
}

/// Register all 5 VSA UDFs into a DataFusion `SessionContext`.
///
/// After calling this, the internal_dataset is queryable as "DuckDB over roles":
/// ```sql
/// SELECT expert_id, vsa_unbind(2, fingerprint) AS score
/// FROM internal_dataset
/// ORDER BY score DESC LIMIT 5;
/// ```
pub fn register_vsa_udfs(ctx: &SessionContext) {
    ctx.register_udf((*vsa_unbind_udf()).clone());
    ctx.register_udf((*vsa_bundle_udf()).clone());
    ctx.register_udf((*vsa_hamming_dist_udf()).clone());
    ctx.register_udf((*vsa_braid_at_udf()).clone());
    ctx.register_udf((*vsa_top_k_udf()).clone());
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_fp() -> [u8; FP_BYTES as usize] {
        [0u8; FP_BYTES as usize]
    }

    fn ones_fp() -> [u8; FP_BYTES as usize] {
        [0xFFu8; FP_BYTES as usize]
    }

    fn role_hot_fp(role: usize) -> [u8; FP_BYTES as usize] {
        // Only the role-indexed slice is set.
        let mut fp = [0u8; FP_BYTES as usize];
        let start_byte = role * ROLE_SLICE_WORDS * 8;
        let end_byte = start_byte + ROLE_SLICE_WORDS * 8;
        fp[start_byte..end_byte].fill(0xFF);
        fp
    }

    #[test]
    fn unbind_zero_is_zero() {
        let fp = zero_fp();
        assert_eq!(unbind_op(0, &fp), 0.0);
    }

    #[test]
    fn unbind_all_ones_is_one() {
        let fp = ones_fp();
        assert_eq!(unbind_op(0, &fp), 1.0);
    }

    #[test]
    fn unbind_hot_role_matches() {
        for role in 0..N_ROLES as u8 {
            let fp = role_hot_fp(role as usize);
            assert_eq!(
                unbind_op(role, &fp),
                1.0,
                "role {role} should score 1.0 in its own slice"
            );
            if role > 0 {
                assert_eq!(
                    unbind_op(role - 1, &fp),
                    0.0,
                    "adjacent role should score 0.0"
                );
            }
        }
    }

    #[test]
    fn hamming_dist_identical_is_zero() {
        let fp = ones_fp();
        assert_eq!(hamming_dist_op(&fp, &fp), 0);
    }

    #[test]
    fn hamming_dist_opposite_is_max() {
        let a = zero_fp();
        let b = ones_fp();
        assert_eq!(hamming_dist_op(&a, &b), FP_WORDS as u32 * 64);
    }

    #[test]
    fn bundle_is_xor() {
        let a = role_hot_fp(0);
        let b = role_hot_fp(1);
        let out = bundle_op(&a, &b);
        // Both role 0 and role 1 slices should be set after XOR-bundle.
        let out_bytes = words_to_bytes(&out);
        assert_eq!(unbind_op(0, &out_bytes), 1.0);
        assert_eq!(unbind_op(1, &out_bytes), 1.0);
        assert_eq!(unbind_op(2, &out_bytes), 0.0);
    }

    #[test]
    fn braid_at_zero_is_identity() {
        let fp = role_hot_fp(3);
        let out = braid_at_op(0, &fp);
        assert_eq!(out, bytes_to_words(&fp));
    }

    #[test]
    fn braid_at_full_rotation_is_identity() {
        let fp = role_hot_fp(2);
        let out = braid_at_op(FP_WORDS as i32, &fp);
        assert_eq!(out, bytes_to_words(&fp));
    }

    #[test]
    fn braid_at_negative_roundtrips() {
        let fp = role_hot_fp(1);
        let forward = braid_at_op(13, &fp);
        let forward_bytes = words_to_bytes(&forward);
        let back = braid_at_op(-13, &forward_bytes);
        assert_eq!(back, bytes_to_words(&fp));
    }

    #[test]
    fn top_k_returns_k_results() {
        let fp = ones_fp(); // all equal — top_k returns first k by sort order
        let k = 5;
        let indices = top_k_op(&fp, k);
        assert_eq!(indices.len(), k);
    }

    #[test]
    fn top_k_prefers_hot_words() {
        // Only word index 7 has bits set.
        let mut fp = zero_fp();
        fp[7 * 8..(7 + 1) * 8].fill(0xFF);
        let indices = top_k_op(&fp, 1);
        assert_eq!(indices[0], 7);
    }
}
