//! [`SumShader`] — a [`SupervisableShader`] impl that maintains a running sum
//! over Arrow `RecordBatch` payloads with a single `Int64` column named `value`.
//!
//! This is a *real* implementation, not a stub.  It uses `Arc<Mutex<i64>>` for
//! interior-mutable accumulator state so the shader is `Send + Sync + 'static`
//! as required by the trait.
//!
//! # Payload contract
//!
//! - Input: a [`RecordBatch`] with exactly one column `value: Int64`.
//! - Output: a [`RecordBatch`] with the same schema, containing **one row**
//!   whose value is the running cumulative sum after adding all rows in the
//!   input batch.
//!
//! # Example
//!
//! ```
//! use std::sync::{Arc, Mutex};
//! use four_repo_demo::cognitive::SumShader;
//! use lance_graph_contract::actor::SupervisableShader;
//! use arrow_array::{Int64Array, RecordBatch};
//! use arrow_schema::{DataType, Field, Schema};
//!
//! let shader = SumShader::new();
//! let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
//! let batch = RecordBatch::try_new(
//!     schema.clone(),
//!     vec![Arc::new(Int64Array::from(vec![5_i64]))],
//! ).unwrap();
//! let result = shader.apply(batch).unwrap();
//! let col = result.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
//! assert_eq!(col.value(0), 5);
//! ```

use std::sync::{Arc, Mutex};

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use lance_graph_contract::actor::SupervisableShader;

/// Running-sum shader.
///
/// Accepts batches with a single `value: Int64` column and emits a
/// single-row batch containing the cumulative sum after the current batch.
pub struct SumShader {
    /// Accumulator — updated atomically on every [`apply`] call.
    ///
    /// `Arc` so the shader can be wrapped in `Arc<SumShader>` as required
    /// by `CognitiveShaderActor::Arguments = Arc<S>`.
    accumulator: Arc<Mutex<i64>>,
    /// Cached output schema (`value: Int64`).
    schema: Arc<Schema>,
}

impl SumShader {
    /// Construct a fresh [`SumShader`] with accumulator at zero.
    pub fn new() -> Self {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        Self {
            accumulator: Arc::new(Mutex::new(0)),
            schema,
        }
    }

    /// Return the current running sum without advancing the accumulator.
    pub fn current_sum(&self) -> i64 {
        *self.accumulator.lock().expect("accumulator lock poisoned")
    }

    /// Build a single-row [`RecordBatch`] whose sole column contains `value`.
    fn make_batch(&self, value: i64) -> Result<RecordBatch, anyhow::Error> {
        let array = Arc::new(Int64Array::from(vec![value]));
        RecordBatch::try_new(self.schema.clone(), vec![array]).map_err(Into::into)
    }
}

impl Default for SumShader {
    fn default() -> Self {
        Self::new()
    }
}

impl SupervisableShader for SumShader {
    type Payload = RecordBatch;
    type Error = anyhow::Error;

    fn shader_name(&self) -> &'static str {
        "SumShader"
    }

    fn apply(&self, payload: RecordBatch) -> Result<RecordBatch, anyhow::Error> {
        // Extract the Int64 column named "value".
        let col_idx = payload
            .schema()
            .index_of("value")
            .map_err(|_| anyhow::anyhow!("SumShader: input batch has no 'value' column"))?;

        let col = payload
            .column(col_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow::anyhow!("SumShader: 'value' column is not Int64"))?;

        // Sum every row in the incoming batch.
        let batch_sum: i64 = (0..col.len()).map(|i| col.value(i)).sum();

        // Add to the running accumulator.
        let running = {
            let mut acc = self.accumulator.lock().expect("accumulator lock poisoned");
            *acc += batch_sum;
            *acc
        };

        self.make_batch(running)
    }

    fn drain(&self) -> Result<(), anyhow::Error> {
        // Nothing to flush for a pure in-memory accumulator.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;

    fn single_row_batch(value: i64, schema: &Arc<Schema>) -> RecordBatch {
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![value]))],
        )
        .unwrap()
    }

    fn extract_sum(batch: &RecordBatch) -> i64 {
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    }

    #[test]
    fn running_sum_accumulates() {
        let shader = SumShader::new();
        let schema = shader.schema.clone();

        let r1 = shader.apply(single_row_batch(5, &schema)).unwrap();
        assert_eq!(extract_sum(&r1), 5);

        let r2 = shader.apply(single_row_batch(7, &schema)).unwrap();
        assert_eq!(extract_sum(&r2), 12);
    }

    #[test]
    fn multi_row_batch_sums_all_rows() {
        let shader = SumShader::new();
        let schema = shader.schema.clone();
        // Batch with three rows: 1 + 2 + 3 = 6.
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![1_i64, 2, 3]))],
        )
        .unwrap();
        let result = shader.apply(batch).unwrap();
        assert_eq!(extract_sum(&result), 6);
    }

    #[test]
    fn drain_succeeds() {
        let shader = SumShader::new();
        assert!(shader.drain().is_ok());
    }
}
