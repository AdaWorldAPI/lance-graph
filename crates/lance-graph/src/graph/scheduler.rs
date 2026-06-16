// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # `scheduler` — `LanceVersionScheduler` over `VersionedGraph::versions()`.
//!
//! The **OUT-direction core impl** of the contract's `VersionScheduler`
//! (`lance_graph_contract::scheduler`, D-MBX-9-IN-impl, the CI-gated twin of
//! D-MBX-9-IN): subscribes to a [`VersionedGraph`]'s Lance dataset versions
//! and lowers each tick into the next legal Rubicon
//! [`KanbanMove`](lance_graph_contract::kanban::KanbanMove) for a given
//! `MailboxSoaView`.
//!
//! ## Why it lives here (not in the contract)
//!
//! The contract trait is sync, zero-dep, and substrate-free (it composes only
//! `MailboxSoaView` + `KanbanColumn` + `KanbanMove` + `ExecTarget`). Real Lance
//! I/O is async and brings the `lance` dep; that combination would violate the
//! contract's BBB zero-dep rule. This crate is the right home: it already
//! owns `VersionedGraph` and depends on `lance`.
//!
//! ## What it does
//!
//! `LanceVersionScheduler` wraps an inner [`VersionScheduler`] (typically
//! [`NextPhaseScheduler`], the canonical forward-arc reference) and a
//! [`VersionedGraph`]. The async surface either:
//!
//! - reads the current Lance dataset version once (`drive_once`) and feeds it
//!   to the inner scheduler, or
//! - polls `versions()` and folds the latest tick into a single proposed
//!   move (`drive_at_latest`).
//!
//! The OUT direction stays **propose, not dispose** (R1 "one SoA never
//! transformed"): the returned [`KanbanMove`] is for the caller to apply via
//! `MailboxSoaOwner::try_advance_phase` — the scheduler never mutates the
//! mailbox.
//!
//! ## Pairing with the IN-direction
//!
//! The contract `scheduler` module ships [`NextPhaseScheduler`] (reference
//! impl) and [`VersionScheduler`] (trait); this crate is the IN-direction
//! impl that closes the bidirectional kanban subscription
//! (`E-SUBSTRATE-IS-THE-SCHEDULER`). Together with `MailboxSoaOwner` (OUT
//! direction, shipped on `MailboxSoA<N>` in `cognitive-shader-driver`), the
//! loop now runs end-to-end in a Lance-backed deployment.

use lance::dataset::Dataset;
use lance_graph_contract::kanban::{ExecTarget, KanbanMove};
use lance_graph_contract::scheduler::{DatasetVersion, NextPhaseScheduler, VersionScheduler};
use lance_graph_contract::soa_view::MailboxSoaView;

use crate::error::{GraphError, Result};
use crate::graph::versioned::VersionedGraph;

/// `LanceVersionScheduler` — wraps a [`VersionedGraph`] and an inner
/// [`VersionScheduler`], lowering each Lance dataset tick into the next
/// proposed Rubicon [`KanbanMove`].
///
/// Generic over the inner policy `S` so a deployment can swap
/// [`NextPhaseScheduler`] (the canonical forward-arc reference) for a custom
/// policy (e.g. one that batches ticks, gates by delta, or routes to
/// non-`Native` exec targets).
///
/// **Lifetimes / costs.** `LanceVersionScheduler` is cheap to clone (it
/// borrows nothing) and reads a Lance dataset once per call. Each `drive_*`
/// call opens the nodes dataset; downstream optimisation would cache a
/// `Dataset` handle, but that's a real-deployment concern, not a contract
/// invariant.
#[derive(Debug, Clone)]
pub struct LanceVersionScheduler<S = NextPhaseScheduler> {
    graph: VersionedGraph,
    inner: S,
}

impl LanceVersionScheduler<NextPhaseScheduler> {
    /// Construct a `LanceVersionScheduler` with the canonical forward-arc
    /// reference [`NextPhaseScheduler`]: every tick proposes
    /// `Planning → CognitiveWork → Evaluation → Commit`, halting at the
    /// absorbing column. The `-550 µs` Libet anchor stamps the
    /// `Planning → CognitiveWork` crossing — the same convention
    /// `MailboxSoa<N>::advance_phase` uses.
    pub fn new(graph: VersionedGraph) -> Self {
        Self {
            graph,
            inner: NextPhaseScheduler,
        }
    }
}

impl<S: VersionScheduler> LanceVersionScheduler<S> {
    /// Construct with a custom inner [`VersionScheduler`] policy.
    pub fn with_policy(graph: VersionedGraph, inner: S) -> Self {
        Self { graph, inner }
    }

    /// The underlying [`VersionedGraph`] this scheduler reads from.
    pub fn graph(&self) -> &VersionedGraph {
        &self.graph
    }

    /// The inner policy that lowers `(view, version, exec)` to a move.
    pub fn policy(&self) -> &S {
        &self.inner
    }

    /// Read the current Lance dataset version (nodes), wrap it as a
    /// [`DatasetVersion`], and feed it to the inner policy on `view`.
    ///
    /// The "drive one tick" surface — the caller polls this per cycle (or per
    /// outside trigger) and applies the returned move via
    /// `MailboxSoaOwner::try_advance_phase`.
    ///
    /// Returns `Ok(None)` when the policy decides no advance is due (e.g.
    /// `view.phase().is_absorbing()`); `Ok(Some(move))` otherwise.
    /// Propagates any Lance error from opening the dataset (cold path —
    /// callers in tests use `tempfile::TempDir` + `commit_encounter_round`).
    pub async fn drive_once<V: MailboxSoaView>(
        &self,
        view: &V,
        exec: ExecTarget,
    ) -> Result<Option<KanbanMove>> {
        let v = self.current_dataset_version().await?;
        Ok(self.inner.on_version(view, v, exec))
    }

    /// Read the **latest** dataset version among `versions()` and lower it
    /// via the inner policy.
    ///
    /// For the default version-agnostic [`NextPhaseScheduler`] this is
    /// equivalent to [`drive_once`](Self::drive_once) (that policy ignores the
    /// `at` argument — the move is a pure function of `view.phase()`). For a
    /// custom version-sensitive `S` the two entry points can legally diverge
    /// under concurrent commits between the two reads; such a policy should
    /// pick one entry point and stick to it. Separated from `drive_once` to
    /// mirror the reactive shape a real `LIVE` subscription takes (the
    /// substrate fires `versions()` on every commit; the scheduler folds them
    /// into one move).
    pub async fn drive_at_latest<V: MailboxSoaView>(
        &self,
        view: &V,
        exec: ExecTarget,
    ) -> Result<Option<KanbanMove>> {
        let versions = self.graph.versions().await?;
        // `versions()` is ascending-sorted by version number in the pinned
        // lance =7.0.0 (`Dataset::versions()` ends with `sort_by_key(|v|
        // v.version)`), so `.last()` is the head = latest commit. NB the
        // upstream surface carries a `// TODO: support pagination` — if a
        // future lance bump paginates `versions()`, prefer the head via
        // `current_dataset_version()` (which reads `version().version`
        // directly) over `.last()`. An empty list is treated as v=0 (the
        // pre-commit sentinel, matching `NextPhaseScheduler`'s expectation
        // that any version triggers the forward arc).
        let latest = versions.last().map(|v| v.version).unwrap_or(0);
        Ok(self.inner.on_version(view, DatasetVersion(latest), exec))
    }

    /// Current Lance dataset version (nodes), wrapped as [`DatasetVersion`].
    ///
    /// Equivalent to `VersionedGraph::current_version()` but returns the
    /// contract carrier directly — saves a `DatasetVersion(...)` at every
    /// call site.
    pub async fn current_dataset_version(&self) -> Result<DatasetVersion> {
        // Open the nodes dataset once; we don't keep the handle across calls
        // because Lance datasets are cheap to reopen (the I/O is metadata
        // read). Real deployments cache the handle.
        let ds = Dataset::open(self.nodes_path().as_str())
            .await
            .map_err(GraphError::from)?;
        Ok(DatasetVersion(ds.version().version))
    }

    /// Path of the nodes dataset under `VersionedGraph` — `base_path/nodes.lance`,
    /// the same convention `VersionedGraph` uses internally. Reads the base path
    /// through the crate-public `VersionedGraph::base_path()` accessor (no
    /// `Debug`-scraping — that brittle path was removed per PR #507 review).
    fn nodes_path(&self) -> String {
        format!("{}/nodes.lance", self.graph.base_path())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::blasgraph::columnar::{EdgeSchema, FingerprintSchema, NodeSchema};
    use arrow_array::builder::FixedSizeBinaryBuilder;
    use arrow_array::{FixedSizeBinaryArray, RecordBatch, UInt32Array};
    use lance_graph_contract::collapse_gate::MailboxId;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};
    use std::sync::Arc;
    use tempfile::TempDir;

    /// Minimal `MailboxSoaView` with a settable phase — same pattern as the
    /// contract's `scheduler::tests::FakeView`, scoped to this module so the
    /// driving-loop test doesn't depend on a real `MailboxSoA<N>` (which
    /// lives in `cognitive-shader-driver`, not a dep of this crate).
    struct FakeView {
        id: MailboxId,
        phase: KanbanColumn,
        cycle: u32,
    }
    impl MailboxSoaView for FakeView {
        fn mailbox_id(&self) -> MailboxId {
            self.id
        }
        fn n_rows(&self) -> usize {
            0
        }
        fn w_slot(&self) -> u8 {
            (self.id & 0x3F) as u8
        }
        fn current_cycle(&self) -> u32 {
            self.cycle
        }
        fn phase(&self) -> KanbanColumn {
            self.phase
        }
        fn energy(&self) -> &[f32] {
            &[]
        }
        fn edges_raw(&self) -> &[u64] {
            &[]
        }
        fn meta_raw(&self) -> &[u32] {
            &[]
        }
        fn entity_type(&self) -> &[u16] {
            &[]
        }
    }

    const PLANE_BYTES: i32 = 2048;
    const SEAL_BYTES: i32 = 6;

    fn fixed_bin_single(width: i32) -> Arc<FixedSizeBinaryArray> {
        let mut b = FixedSizeBinaryBuilder::with_capacity(1, width);
        b.append_value(vec![0u8; width as usize]).unwrap();
        Arc::new(b.finish())
    }

    /// Build a single-row NodeSchema RecordBatch — cheapest valid input for
    /// `commit_encounter_round` to advance the Lance nodes-dataset version.
    /// Schema is `[node_id u32, plane_s/p/o FixedSizeBinary(2048),
    /// seal_s/p/o FixedSizeBinary(6), encounters u32]` (8 cols, exact match
    /// for `blasgraph::columnar::NodeSchema`).
    fn empty_node_batch() -> RecordBatch {
        RecordBatch::try_new(
            NodeSchema::arrow_schema_ref(),
            vec![
                Arc::new(UInt32Array::from(vec![0u32])),
                fixed_bin_single(PLANE_BYTES),
                fixed_bin_single(PLANE_BYTES),
                fixed_bin_single(PLANE_BYTES),
                fixed_bin_single(SEAL_BYTES),
                fixed_bin_single(SEAL_BYTES),
                fixed_bin_single(SEAL_BYTES),
                Arc::new(UInt32Array::from(vec![0u32])),
            ],
        )
        .unwrap()
    }
    /// Empty EdgeSchema batch — `[src_id u32, dst_id u32, weight Float16,
    /// label FixedSizeBinary(2048)]`. Zero rows is valid for Lance writes;
    /// the Float16 column is built empty via the builder API to avoid
    /// pulling in the `half` crate transitively just for tests.
    fn empty_edge_batch() -> RecordBatch {
        let label = FixedSizeBinaryBuilder::with_capacity(0, PLANE_BYTES).finish();
        let weight = arrow_array::builder::Float16Builder::with_capacity(0).finish();
        RecordBatch::try_new(
            EdgeSchema::arrow_schema_ref(),
            vec![
                Arc::new(UInt32Array::from(Vec::<u32>::new())),
                Arc::new(UInt32Array::from(Vec::<u32>::new())),
                Arc::new(weight),
                Arc::new(label),
            ],
        )
        .unwrap()
    }
    /// Single-row FingerprintSchema batch — `[id u32, fingerprint
    /// FixedSizeBinary(2048)]`.
    fn empty_fp_batch() -> RecordBatch {
        RecordBatch::try_new(
            FingerprintSchema::arrow_schema_ref(),
            vec![
                Arc::new(UInt32Array::from(vec![0u32])),
                fixed_bin_single(PLANE_BYTES),
            ],
        )
        .unwrap()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn current_dataset_version_reads_nodes_head() {
        let tmp = TempDir::new().unwrap();
        let g = VersionedGraph::local(tmp.path().to_str().unwrap());
        // First commit — creates the dataset; nodes head is version 1 under
        // Lance's 1-based versioning (every write ticks).
        let v1 = g
            .commit_encounter_round(empty_node_batch(), empty_edge_batch(), empty_fp_batch())
            .await
            .unwrap();
        let sched = LanceVersionScheduler::new(g.clone());
        let observed = sched.current_dataset_version().await.unwrap();
        assert_eq!(observed.0, v1, "current_dataset_version == latest commit");
        // Second commit advances the version; the scheduler sees the new head.
        let v2 = g
            .commit_encounter_round(empty_node_batch(), empty_edge_batch(), empty_fp_batch())
            .await
            .unwrap();
        assert!(v2 > v1);
        assert_eq!(sched.current_dataset_version().await.unwrap().0, v2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn drive_once_proposes_forward_arc_on_planning() {
        let tmp = TempDir::new().unwrap();
        let g = VersionedGraph::local(tmp.path().to_str().unwrap());
        g.commit_encounter_round(empty_node_batch(), empty_edge_batch(), empty_fp_batch())
            .await
            .unwrap();
        let sched = LanceVersionScheduler::new(g);
        let view = FakeView {
            id: 7,
            phase: KanbanColumn::Planning,
            cycle: 11,
        };
        let mv = sched
            .drive_once(&view, ExecTarget::Native)
            .await
            .unwrap()
            .expect("Planning is not absorbing");
        // Forward arc: Planning -> CognitiveWork carries the Libet anchor.
        assert_eq!(mv.from, KanbanColumn::Planning);
        assert_eq!(mv.to, KanbanColumn::CognitiveWork);
        assert_eq!(mv.libet_offset_us, -550_000);
        assert_eq!(mv.mailbox, 7);
        assert_eq!(mv.witness_chain_position, 11);
        assert_eq!(mv.exec, ExecTarget::Native);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn drive_at_latest_matches_drive_once() {
        let tmp = TempDir::new().unwrap();
        let g = VersionedGraph::local(tmp.path().to_str().unwrap());
        // Two commits — exercise the multi-version code path.
        g.commit_encounter_round(empty_node_batch(), empty_edge_batch(), empty_fp_batch())
            .await
            .unwrap();
        g.commit_encounter_round(empty_node_batch(), empty_edge_batch(), empty_fp_batch())
            .await
            .unwrap();
        let sched = LanceVersionScheduler::new(g);
        let view = FakeView {
            id: 13,
            phase: KanbanColumn::CognitiveWork,
            cycle: 1,
        };
        let a = sched.drive_once(&view, ExecTarget::Native).await.unwrap();
        let b = sched
            .drive_at_latest(&view, ExecTarget::Native)
            .await
            .unwrap();
        // Both lower the latest version to the same move (CognitiveWork ->
        // Evaluation, no Libet anchor); proves equivalence under current
        // single-head semantics.
        assert_eq!(a, b);
        let mv = a.unwrap();
        assert_eq!(mv.from, KanbanColumn::CognitiveWork);
        assert_eq!(mv.to, KanbanColumn::Evaluation);
        assert_eq!(mv.libet_offset_us, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn absorbing_columns_return_none_even_with_lance_head() {
        let tmp = TempDir::new().unwrap();
        let g = VersionedGraph::local(tmp.path().to_str().unwrap());
        g.commit_encounter_round(empty_node_batch(), empty_edge_batch(), empty_fp_batch())
            .await
            .unwrap();
        let sched = LanceVersionScheduler::new(g);
        // Commit and Prune are absorbing — the scheduler proposes nothing,
        // regardless of how many Lance versions are committed downstream.
        for absorbing in [KanbanColumn::Commit, KanbanColumn::Prune] {
            let view = FakeView {
                id: 1,
                phase: absorbing,
                cycle: 0,
            };
            assert!(sched
                .drive_once(&view, ExecTarget::Native)
                .await
                .unwrap()
                .is_none());
            assert!(sched
                .drive_at_latest(&view, ExecTarget::Native)
                .await
                .unwrap()
                .is_none());
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn exec_target_threads_through_lance_drive() {
        let tmp = TempDir::new().unwrap();
        let g = VersionedGraph::local(tmp.path().to_str().unwrap());
        g.commit_encounter_round(empty_node_batch(), empty_edge_batch(), empty_fp_batch())
            .await
            .unwrap();
        let sched = LanceVersionScheduler::new(g);
        let view = FakeView {
            id: 1,
            phase: KanbanColumn::Planning,
            cycle: 0,
        };
        for exec in [
            ExecTarget::Native,
            ExecTarget::Jit,
            ExecTarget::SurrealQl,
            ExecTarget::Elixir,
        ] {
            let mv = sched.drive_once(&view, exec).await.unwrap().unwrap();
            assert_eq!(mv.exec, exec, "exec target threads through");
        }
    }
}
