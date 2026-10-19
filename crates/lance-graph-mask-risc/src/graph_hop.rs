//! Version-scoped one-hop graph fold over borrowed SoA edge lanes.
//!
//! Origin: the AdaWorldAPI/petgraph BFS-first-layer experiment. The sole
//! evaluator remains Mask-RISC (Gather, Gather, ScatterOrU32).
//!
//! This returns an explicitly demanded one-hop mask. It is NOT permission to
//! use that mask as an intermediate in Gremlin repeat or to introduce a second
//! graph evaluator. No canonical GUIDs are replaced by these local row ids.

use crate::{
    execute_into, words_for, ExecError, Foreign, ForeignPlane, LaneRef, MaskOp, Operand, Out,
    Planes, Program, Scratch, Terminal, Value,
};

/// Edge fields share one stable row order; no copied adjacency structure.
#[derive(Clone, Copy, Debug)]
pub struct EdgeColumns<'a> {
    pub sources: &'a [u32],
    pub targets: &'a [u32],
}

/// Caller-selected witness. The visibility plane includes live-node presence.
#[derive(Clone, Copy, Debug)]
pub struct GraphSnapshot<'a> {
    pub witness_version: u64,
    pub node_rows: usize,
    pub visible_nodes: &'a [u64],
    pub live_edges: &'a [u64],
}

#[derive(Debug, PartialEq, Eq)]
pub enum HopError {
    StaleWitness,
    InvalidShape,
    InvalidSeed,
    InvisibleSeed,
    UnauthorizedFrontier,
    Executor(ExecError),
}

/// A compiled-once (in the sense of assembled-once) Mask-RISC plan.
/// The caller owns every execution buffer and may reuse them between versions.
#[derive(Debug)]
pub struct OutgoingHop {
    program: Program,
    edge_rows: usize,
    node_rows: usize,
}

impl OutgoingHop {
    pub fn new(edge_rows: usize, node_rows: usize) -> Result<Self, HopError> {
        if node_rows > u32::MAX as usize {
            return Err(HopError::InvalidShape);
        }
        let program = Program::new(
            vec![
                MaskOp::Gather {
                    lane: 0, foreign: 0,
                    under: Some(Operand::Plane(0)), dst: 0,
                },
                MaskOp::Gather {
                    lane: 1, foreign: 1,
                    under: Some(Operand::Scratch(0)), dst: 1,
                },
            ],
            Terminal::ScatterOrU32 {
                mask: Operand::Scratch(1),
                lane: 1,
                out_rows: node_rows as u32,
            },
        );
        Ok(Self { program, edge_rows, node_rows })
    }

    /// For sizing caller-owned Scratch once; the execution path allocates nothing.
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// Expand an authorized frontier to ONE demanded outgoing-neighbor mask.
    /// Rejects any invisible seed, even if its destination happens to be public.
    pub fn run(
        &self,
        edges: EdgeColumns<'_>,
        snapshot: GraphSnapshot<'_>,
        frontier: &[u64],
        expected_witness: u64,
        scratch: &mut Scratch<'_>,
        out: &mut [u64],
    ) -> Result<(), HopError> {
        if snapshot.witness_version != expected_witness {
            return Err(HopError::StaleWitness);
        }
        if snapshot.node_rows != self.node_rows
            || edges.sources.len() != self.edge_rows
            || edges.targets.len() != self.edge_rows
            || !valid_mask(snapshot.visible_nodes, self.node_rows)
            || !valid_mask(snapshot.live_edges, self.edge_rows)
            || !valid_mask(frontier, self.node_rows)
            || out.len() != words_for(self.node_rows)
            || edges.sources.iter().chain(edges.targets.iter())
                .any(|&id| id as usize >= self.node_rows)
        {
            return Err(HopError::InvalidShape);
        }
        if frontier.iter().zip(snapshot.visible_nodes.iter())
            .any(|(&seed, &visible)| seed & !visible != 0)
        {
            return Err(HopError::UnauthorizedFrontier);
        }
        if self.edge_rows == 0 {
            out.fill(0);
            return Ok(());
        }
        let masks = [snapshot.live_edges];
        let lanes = [
            LaneRef::U32(edges.sources),
            LaneRef::U32(edges.targets),
        ];
        let planes = Planes { n_rows: self.edge_rows, masks: &masks, lanes: &lanes };
        let foreign_masks = [
            ForeignPlane { words: frontier, rows: self.node_rows },
            ForeignPlane { words: snapshot.visible_nodes, rows: self.node_rows },
        ];
        let foreign = Foreign { planes: &foreign_masks, lanes: &[] };
        let result = execute_into(
            &self.program, &planes, &foreign, scratch, Out::Mask(out)
        ).map_err(HopError::Executor)?;
        debug_assert_eq!(result, Value::Scattered);
        Ok(())
    }
}

/// Masks are LSB-first with zero tail; invalid phantom rows never enter a fold.
fn valid_mask(mask: &[u64], rows: usize) -> bool {
    if mask.len() != words_for(rows) {
        return false;
    }
    rows == 0 || rows.is_multiple_of(64)
        || mask.last().is_some_and(|word| word >> (rows % 64) == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(rows: usize, ids: &[usize]) -> Vec<u64> {
        let mut words = vec![0; words_for(rows)];
        for &id in ids { words[id / 64] |= 1u64 << (id % 64); }
        words
    }

    fn execute(
        sources: &[u32], targets: &[u32], rows: usize,
        selected: &[u64], visible: &[u64], live: &[u64], witness: u64
    ) -> Result<Vec<u64>, HopError> {
        let plan = OutgoingHop::new(sources.len(), rows)?;
        let mut scratch = Scratch::for_program(plan.program(), sources.len())
            .map_err(HopError::Executor)?;
        let mut out = vec![u64::MAX; words_for(rows)];
        plan.run(
            EdgeColumns { sources, targets },
            GraphSnapshot { witness_version: witness, node_rows: rows, visible_nodes: visible, live_edges: live },
            selected, witness, &mut scratch, &mut out,
        )?;
        Ok(out)
    }

    #[test]
    fn multiple_seeds_and_duplicate_edges_cross_mask_word_boundaries() {
        let src = [0, 0, 65, 65, 129, 65, 0];
        let dst = [1, 65, 129, 70, 1, 0, 129];
        let visible = mask(130, &(0..130).collect::<Vec<_>>());
        let live = mask(src.len(), &(0..src.len()).collect::<Vec<_>>());
        assert_eq!(execute(&src, &dst, 130, &mask(130, &[0, 65]), &visible, &live, 42),
            Ok(mask(130, &[0, 1, 65, 70, 129])));
    }

    #[test]
    fn invisible_bridge_cannot_activate_visible_target() {
        let src = [0, 65, 0];
        let dst = [65, 129, 1];
        let visible = mask(130, &[0, 1, 129]); // 65 private
        let live = mask(3, &[0, 1, 2]);
        assert_eq!(execute(&src, &dst, 130, &mask(130, &[0]), &visible, &live, 4),
            Ok(mask(130, &[1])));
        assert_eq!(execute(&src, &dst, 130, &mask(130, &[0, 65]), &visible, &live, 4),
            Err(HopError::UnauthorizedFrontier));
    }

    #[test]
    fn witness_presence_changes_result_without_reordering_identities() {
        let src = [0, 0, 0];
        let dst = [1, 65, 129];
        let visible = mask(130, &[0, 1, 65, 129]);
        let seeds = mask(130, &[0]);
        assert_eq!(execute(&src, &dst, 130, &seeds, &visible, &mask(3, &[0, 1]), 7),
            Ok(mask(130, &[1, 65])));
        assert_eq!(execute(&src, &dst, 130, &seeds, &visible, &mask(3, &[0, 2]), 8),
            Ok(mask(130, &[1, 129])));
    }

    #[test]
    fn stale_witness_and_bad_masks_leave_output_unchanged() {
        let src = [0]; let dst = [1];
        let plan = OutgoingHop::new(1, 2).unwrap();
        let mut scratch = Scratch::for_program(plan.program(), 1).unwrap();
        let visible = mask(2, &[0, 1]);
        let edges = mask(1, &[0]);
        let seeds = mask(2, &[0]);
        let snap = GraphSnapshot { witness_version: 7, node_rows: 2, visible_nodes: &visible, live_edges: &edges };
        let mut out = [0xDEAD_BEEFu64];
        assert_eq!(plan.run(
            EdgeColumns { sources: &src, targets: &dst }, snap, &seeds, 6, &mut scratch, &mut out,
        ), Err(HopError::StaleWitness));
        assert_eq!(out, [0xDEAD_BEEF]);
        let bad_mask = [u64::MAX];
        let snap = GraphSnapshot { visible_nodes: &bad_mask, ..snap };
        assert_eq!(plan.run(
            EdgeColumns { sources: &src, targets: &dst }, snap, &seeds, 7, &mut scratch, &mut out,
        ), Err(HopError::InvalidShape));
        assert_eq!(out, [0xDEAD_BEEF]);
    }

    #[test]
    fn empty_edge_population_clears_explicit_result() {
        let src: [u32; 0] = [];
        let plan = OutgoingHop::new(0, 2).unwrap();
        let mut scratch = Scratch::for_program(plan.program(), 0).unwrap();
        let visible = mask(2, &[0, 1]);
        let live: [u64; 0] = [];
        let mut out = [u64::MAX];
        plan.run(
            EdgeColumns { sources: &src, targets: &src },
            GraphSnapshot { witness_version: 1, node_rows: 2, visible_nodes: &visible, live_edges: &live },
            &mask(2, &[0]), 1, &mut scratch, &mut out,
        ).unwrap();
        assert_eq!(out, [0]);
    }

    #[test]
    fn invalid_edge_ordinal_and_phantom_mask_tail_are_refused() {
        let src = [0]; let bad_dst = [130]; let good_dst = [1];
        let visible = mask(130, &[0, 1, 129]);
        let seeds = mask(130, &[0]);
        let live = mask(1, &[0]);
        assert_eq!(execute(&src, &bad_dst, 130, &seeds, &visible, &live, 2),
            Err(HopError::InvalidShape));
        let invalid_visible = [u64::MAX, u64::MAX, u64::MAX];
        assert_eq!(execute(&src, &good_dst, 130, &seeds, &invalid_visible, &live, 2),
            Err(HopError::InvalidShape));
    }
}
