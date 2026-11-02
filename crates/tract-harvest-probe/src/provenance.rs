//! What each Tract stage keeps of the source graph, counted against the raw
//! protobuf ([`SourceInventory`]), never assumed.

use std::collections::{BTreeMap, HashSet};

use crate::source::SourceInventory;

/// One stage's correspondence with the source graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageProvenance {
    pub stage: &'static str,
    pub nodes: usize,
    /// Source nodes whose (non-empty) name is a node name in this stage.
    pub source_names_kept: usize,
    /// Source nodes whose name is absent here, by op type.
    pub source_names_lost: BTreeMap<String, usize>,
    /// Node names in this stage that no source node or initializer carries.
    pub synthesized_names: usize,
    /// Source node outputs that are still an outlet label of this stage.
    pub source_outputs_labelled: usize,
    pub source_outputs_total: usize,
}

/// Count a stage. `node_names` and `outlet_labels` are the stage's; the
/// source inventory is the reference.
pub fn stage<'a>(
    stage: &'static str,
    src: &SourceInventory,
    node_names: impl IntoIterator<Item = &'a str>,
    outlet_labels: impl IntoIterator<Item = &'a str>,
) -> StageProvenance {
    let names: Vec<&str> = node_names.into_iter().collect();
    let name_set: HashSet<&str> = names.iter().copied().collect();
    let labels: HashSet<&str> = outlet_labels.into_iter().collect();
    let src_names: HashSet<&str> = src
        .nodes
        .iter()
        .map(|n| n.name.as_str())
        .chain(src.initializers.iter().map(|i| i.name.as_str()))
        .collect();

    let mut kept = 0;
    let mut lost = BTreeMap::new();
    for n in src.nodes.iter().filter(|n| !n.name.is_empty()) {
        if name_set.contains(n.name.as_str()) {
            kept += 1;
        } else {
            *lost.entry(n.op_type.clone()).or_insert(0) += 1;
        }
    }
    let outputs: Vec<&str> = src
        .nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(String::as_str))
        .filter(|o| !o.is_empty())
        .collect();
    StageProvenance {
        stage,
        nodes: names.len(),
        source_names_kept: kept,
        source_names_lost: lost,
        synthesized_names: names.iter().filter(|n| !src_names.contains(**n)).count(),
        source_outputs_labelled: outputs.iter().filter(|o| labels.contains(**o)).count(),
        source_outputs_total: outputs.len(),
    }
}

/// [`stage`] over any Tract graph (inference or typed).
pub fn of_graph<F, O>(
    name: &'static str,
    src: &SourceInventory,
    g: &tract_onnx::prelude::Graph<F, O>,
) -> StageProvenance
where
    F: tract_onnx::prelude::Fact + Clone + 'static,
    O: std::fmt::Debug
        + std::fmt::Display
        + AsRef<dyn tract_onnx::prelude::Op>
        + AsMut<dyn tract_onnx::prelude::Op>
        + Clone
        + 'static,
{
    stage(
        name,
        src,
        g.nodes().iter().map(|n| n.name.as_str()),
        g.outlet_labels.values().map(String::as_str),
    )
}
