//! BROADCAST: Distribute query fingerprint to all scan partitions.
//!
//! Analogous to DataFusion's Repartition but for fingerprint vectors.
//! The fingerprint is a Container (256×u64 = 16,384 bits) that gets
//! broadcast to all partitions for parallel Hamming distance computation.

use super::{Morsel, PhysicalOperator};

/// BROADCAST physical operator.
#[derive(Debug)]
pub struct BroadcastOp {
    /// The fingerprint to broadcast (Container as Vec<u64>).
    pub fingerprint: Vec<u64>,
    /// Number of target partitions.
    pub partitions: usize,
    /// Estimated cardinality (= partitions, since each gets the fingerprint).
    pub cardinality: f64,
}

impl BroadcastOp {
    pub fn new(fingerprint: Vec<u64>, partitions: usize) -> Self {
        Self {
            fingerprint,
            partitions,
            cardinality: partitions as f64,
        }
    }

    /// Execute: create one morsel per partition, each containing the fingerprint.
    pub fn execute(&self) -> Vec<Morsel> {
        (0..self.partitions)
            .map(|_| Morsel {
                num_rows: 1,
                columns: vec![super::ColumnData::Fingerprint(vec![self
                    .fingerprint
                    .clone()])],
            })
            .collect()
    }
}

impl PhysicalOperator for BroadcastOp {
    fn name(&self) -> &str {
        "Broadcast"
    }
    fn cardinality(&self) -> f64 {
        self.cardinality
    }
    fn is_pipeline_breaker(&self) -> bool {
        false
    }
    fn children(&self) -> Vec<&dyn PhysicalOperator> {
        vec![]
    }
}
