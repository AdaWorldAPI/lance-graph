//! NARS truth propagation along adjacency in one pass.
//!
//! For each edge in the batch: truth_out = semiring.multiply(truth_in, edge_truth)
//! This is WHERE the semiring algebra meets the adjacency substrate.
//! Truth values are edge properties — they live in the adjacency store.

use super::batch::AdjacencyBatch;
use super::csr::AdjacencyStore;
use crate::nars::TruthValue;
use crate::physical::accumulate::{Semiring, SemiringValue};

/// Propagate truth values along adjacency edges in one batch pass.
///
/// For each (source → target) edge:
///   truth_out[target] = semiring.multiply(truth_in[source], edge_truth)
///
/// At merge points (multiple edges → same target):
///   truth_out[target] = semiring.add(truth_out[target], new_value)
pub fn adjacent_truth_propagate(
    store: &AdjacencyStore,
    batch: &AdjacencyBatch,
    input_truths: &[TruthValue],
    semiring: &dyn Semiring,
) -> Vec<(u64, TruthValue)> {
    let mut output: std::collections::HashMap<u64, SemiringValue> =
        std::collections::HashMap::new();

    for i in 0..batch.num_sources() {
        let _source = batch.source_ids[i];
        let targets = batch.targets_for(i);
        let edge_ids = batch.edge_ids_for(i);

        // Get input truth for this source
        let input_truth = input_truths
            .get(i)
            .copied()
            .unwrap_or(TruthValue::default());
        let input_sv = SemiringValue::Truth {
            frequency: input_truth.frequency as f64,
            confidence: input_truth.confidence as f64,
        };

        for (target, edge_id) in targets.iter().zip(edge_ids.iter()) {
            // Get edge truth from adjacency store properties
            let edge_truth = store
                .edge_properties
                .truth_value(*edge_id)
                .unwrap_or((1.0, 0.9));
            let edge_sv = SemiringValue::Truth {
                frequency: edge_truth.0 as f64,
                confidence: edge_truth.1 as f64,
            };

            // Propagate: multiply input truth with edge truth
            let propagated = semiring.multiply(&input_sv, &edge_sv);

            // Merge at target: add existing output with new propagated value
            let current = output.entry(*target).or_insert_with(|| semiring.zero());
            *current = semiring.add(current, &propagated);
        }
    }

    // Convert back to TruthValue
    output
        .into_iter()
        .map(|(node_id, sv)| {
            let tv = match sv {
                SemiringValue::Truth {
                    frequency,
                    confidence,
                } => TruthValue {
                    frequency: frequency as f32,
                    confidence: confidence as f32,
                },
                _ => TruthValue::default(),
            };
            (node_id, tv)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical::accumulate::TruthPropagatingSemiring;

    #[test]
    fn test_truth_propagation_along_adjacency() {
        // Graph: 0 →[f=0.9,c=0.8]→ 1 →[f=0.7,c=0.9]→ 2
        let mut store = AdjacencyStore::from_edges("CAUSES".into(), 3, &[(0, 1), (1, 2)]);
        store.edge_properties = super::super::properties::EdgeProperties::new().with_nars_truth(
            vec![0.9, 0.7], // frequencies for edges 0, 1
            vec![0.8, 0.9], // confidences for edges 0, 1
        );

        let semiring = TruthPropagatingSemiring;

        // Hop 1: propagate from node 0 with truth (1.0, 0.9)
        let batch_1 = store.batch_adjacent(&[0]);
        let input_truths_1 = vec![TruthValue {
            frequency: 1.0,
            confidence: 0.9,
        }];
        let result_1 = adjacent_truth_propagate(&store, &batch_1, &input_truths_1, &semiring);

        // Node 1 should have propagated truth
        let node_1_truth = result_1.iter().find(|(id, _)| *id == 1).unwrap().1;
        assert!(node_1_truth.frequency > 0.0);
        assert!(node_1_truth.confidence > 0.0);

        // Hop 2: propagate from node 1 with the result from hop 1
        let batch_2 = store.batch_adjacent(&[1]);
        let input_truths_2 = vec![node_1_truth];
        let result_2 = adjacent_truth_propagate(&store, &batch_2, &input_truths_2, &semiring);

        // Node 2 should have further attenuated truth
        let node_2_truth = result_2.iter().find(|(id, _)| *id == 2).unwrap().1;
        assert!(
            node_2_truth.frequency < node_1_truth.frequency,
            "Deduction should reduce frequency at each hop"
        );
        assert!(
            node_2_truth.confidence < node_1_truth.confidence,
            "Deduction should reduce confidence at each hop"
        );
    }
}
