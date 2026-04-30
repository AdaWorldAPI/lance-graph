//! Pipeline DAG executor (LF-12 keystone).
//!
//! Takes a `Vec<UnifiedStep>` whose `depends_on` fields encode a DAG,
//! builds a topological ordering via Kahn's algorithm, and executes
//! steps in dependency-respecting order.
//!
//! # Design choices
//!
//! * **Kahn's algorithm** -- iterative, zero extra deps, O(V+E).
//! * **Cycle detection** -- if the queue drains before all nodes are
//!   visited, a cycle exists and `PipelineError::CycleDetected` is
//!   returned with the set of nodes involved.
//! * **Execution callback** -- `execute_with` takes a closure
//!   `FnMut(&mut UnifiedStep) -> Result<(), String>` so callers can
//!   plug in domain-specific dispatch (e.g. `OrchestrationBridge::route`).
//! * **No async yet** -- synchronous executor. Async fan-out for
//!   independent steps is a follow-up (PR-F4 / PR-G2 will decide
//!   the concurrency model).

use lance_graph_contract::orchestration::{OrchestrationBridge, StepId, StepStatus, UnifiedStep};
use std::collections::{HashMap, HashSet, VecDeque};

// -- Errors -------------------------------------------------------------------

/// Errors from pipeline DAG operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    /// A dependency references a `StepId` not present in the input.
    MissingDependency {
        step_id: StepId,
        missing_dep: StepId,
    },
    /// The dependency graph contains a cycle. The payload lists the
    /// `StepId`s that participate in the cycle (not necessarily all
    /// of them, but enough to diagnose).
    CycleDetected { involved: Vec<StepId> },
    /// A step's execution callback returned an error.
    StepFailed { step_id: StepId, reason: String },
    /// Duplicate `StepId` in the input vector.
    DuplicateStepId { step_id: StepId },
}

impl core::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingDependency {
                step_id,
                missing_dep,
            } => {
                write!(
                    f,
                    "step {step_id} depends on {missing_dep}, which is not in the pipeline"
                )
            }
            Self::CycleDetected { involved } => {
                write!(f, "cycle detected among steps: {involved:?}")
            }
            Self::StepFailed { step_id, reason } => {
                write!(f, "step {step_id} failed: {reason}")
            }
            Self::DuplicateStepId { step_id } => {
                write!(f, "duplicate step id: {step_id}")
            }
        }
    }
}

// -- DAG representation -------------------------------------------------------

/// A validated DAG built from a `Vec<UnifiedStep>`.
///
/// Construction validates:
/// - No duplicate `StepId`s.
/// - Every `depends_on` entry references an existing step.
/// - No cycles (Kahn's algorithm produces a full topological order).
#[derive(Debug)]
pub struct PipelineDag {
    /// Steps stored in insertion order; accessed by position index.
    steps: Vec<UnifiedStep>,
    /// Map from `StepId` to position index in `steps`.
    index: HashMap<StepId, usize>,
    /// Topological order (position indices into `steps`).
    topo_order: Vec<usize>,
}

impl PipelineDag {
    /// Build a DAG from the given steps.
    ///
    /// Returns `Err` if duplicate ids, missing dependencies, or cycles
    /// are detected.
    pub fn build(steps: Vec<UnifiedStep>) -> Result<Self, PipelineError> {
        // 1. Index by StepId, check for duplicates.
        let mut index: HashMap<StepId, usize> = HashMap::with_capacity(steps.len());
        for (pos, step) in steps.iter().enumerate() {
            let sid = step.id();
            if index.insert(sid, pos).is_some() {
                return Err(PipelineError::DuplicateStepId { step_id: sid });
            }
        }

        // 2. Validate all dependency references and compute in-degrees.
        let mut in_degree: Vec<u32> = vec![0; steps.len()];
        // adjacency: from -> [to_positions]
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); steps.len()];

        for (pos, step) in steps.iter().enumerate() {
            for &dep in &step.depends_on {
                let dep_pos = *index.get(&dep).ok_or(PipelineError::MissingDependency {
                    step_id: step.id(),
                    missing_dep: dep,
                })?;
                // dep must complete before step: edge dep_pos -> pos
                successors[dep_pos].push(pos);
                in_degree[pos] += 1;
            }
        }

        // 3. Kahn's algorithm.
        let mut queue: VecDeque<usize> = VecDeque::new();
        for (pos, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(pos);
            }
        }

        let mut topo_order: Vec<usize> = Vec::with_capacity(steps.len());
        while let Some(pos) = queue.pop_front() {
            topo_order.push(pos);
            for &succ in &successors[pos] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }

        if topo_order.len() != steps.len() {
            // Nodes with remaining in-degree > 0 form the cycle.
            let involved: Vec<StepId> = in_degree
                .iter()
                .enumerate()
                .filter(|(_, &d)| d > 0)
                .map(|(pos, _)| steps[pos].id())
                .collect();
            return Err(PipelineError::CycleDetected { involved });
        }

        Ok(Self {
            steps,
            index,
            topo_order,
        })
    }

    /// Return the topological execution order as `StepId`s.
    pub fn execution_order(&self) -> Vec<StepId> {
        self.topo_order
            .iter()
            .map(|&pos| self.steps[pos].id())
            .collect()
    }

    /// Number of steps in the DAG.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the DAG has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Return the set of root steps (no dependencies).
    pub fn roots(&self) -> Vec<StepId> {
        self.steps
            .iter()
            .filter(|s| s.depends_on.is_empty())
            .map(|s| s.id())
            .collect()
    }

    /// Return the set of leaf steps (no dependents).
    pub fn leaves(&self) -> Vec<StepId> {
        let mut has_dependent: HashSet<StepId> = HashSet::new();
        for step in &self.steps {
            for &dep in &step.depends_on {
                has_dependent.insert(dep);
            }
        }
        self.steps
            .iter()
            .filter(|s| !has_dependent.contains(&s.id()))
            .map(|s| s.id())
            .collect()
    }

    /// Execute all steps in topological order, calling `f` for each.
    ///
    /// Steps are executed synchronously. Each step's `status` is set to
    /// `Running` before the callback and to `Completed` after a
    /// successful return (or `Failed` on error).
    ///
    /// On the first step failure, execution halts and the error is
    /// returned. Downstream steps remain `Pending`.
    pub fn execute_with<F>(&mut self, mut f: F) -> Result<(), PipelineError>
    where
        F: FnMut(&mut UnifiedStep) -> Result<(), String>,
    {
        let order = self.topo_order.clone();
        for &pos in &order {
            let step = &mut self.steps[pos];
            step.status = StepStatus::Running;
            match f(step) {
                Ok(()) => {
                    step.status = StepStatus::Completed;
                }
                Err(reason) => {
                    let id = step.id();
                    step.status = StepStatus::Failed;
                    return Err(PipelineError::StepFailed {
                        step_id: id,
                        reason,
                    });
                }
            }
        }
        Ok(())
    }

    /// Execute all steps in topological order via an `OrchestrationBridge`.
    ///
    /// This is the canonical adapter for `OrchestrationBridge` consumers
    /// (PR-F4 / PR-G2 cross-domain step routing): it wires the trait's
    /// `route()` into [`PipelineDag::execute_with`] so callers do not
    /// have to fork a closure-based executor and a bridge-based one.
    ///
    /// On bridge failure, the per-step error is converted to a string
    /// (matching the `execute_with` error contract) and surfaced as
    /// [`PipelineError::StepFailed`].
    pub fn execute_via_bridge<B: OrchestrationBridge>(
        &mut self,
        bridge: &B,
    ) -> Result<(), PipelineError> {
        self.execute_with(|s| bridge.route(s).map_err(|e| format!("{e:?}")))
    }

    /// Borrow the steps slice (in insertion order, not topo order).
    pub fn steps(&self) -> &[UnifiedStep] {
        &self.steps
    }

    /// Get a step by its `StepId`.
    pub fn get(&self, id: StepId) -> Option<&UnifiedStep> {
        self.index.get(&id).map(|&pos| &self.steps[pos])
    }

    /// Consume the DAG and return the steps vector.
    pub fn into_steps(self) -> Vec<UnifiedStep> {
        self.steps
    }
}

// -- Convenience constructor --------------------------------------------------

/// Helper to create a minimal `UnifiedStep` for testing / scripting.
///
/// The `tag` argument names the step (becomes `step_id`); the numeric
/// `StepId` used for DAG edges is derived from that tag via
/// [`UnifiedStep::id`]. To depend on another step, pass the FNV id of
/// that step's tag (use [`step_id_of`] to compute it).
pub fn make_step_named(tag: &str, depends_on: Vec<StepId>) -> UnifiedStep {
    UnifiedStep {
        step_id: tag.to_string(),
        step_type: "lg.noop".to_string(),
        status: StepStatus::Pending,
        thinking: None,
        reasoning: None,
        confidence: None,
        depends_on,
    }
}

/// Compute the FNV-derived `StepId` for a step tag — useful for
/// expressing `depends_on` against a tag-named step.
pub fn step_id_of(tag: &str) -> StepId {
    // Mirror UnifiedStep::id() exactly.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in tag.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Numeric-tag convenience: `make_step(1, vec![3])` builds a step
/// tagged `"step-1"` whose dependencies are FNV ids of `"step-3"`.
/// Existing DAG tests use small numeric tags; this preserves their
/// shape while routing through the derived-id design.
pub fn make_step(id: StepId, depends_on_numeric: Vec<StepId>) -> UnifiedStep {
    let deps: Vec<StepId> = depends_on_numeric
        .into_iter()
        .map(|n| step_id_of(&format!("step-{n}")))
        .collect();
    make_step_named(&format!("step-{id}"), deps)
}

/// Numeric-tag → FNV-id mapping mirroring `make_step`.
pub fn id_of(numeric_tag: StepId) -> StepId {
    step_id_of(&format!("step-{numeric_tag}"))
}

// -- Tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- 1. Empty DAG ---------------------------------------------------------

    #[test]
    fn empty_dag_builds_and_executes() {
        let dag = PipelineDag::build(vec![]).unwrap();
        assert!(dag.is_empty());
        assert_eq!(dag.len(), 0);
        assert!(dag.execution_order().is_empty());
        assert!(dag.roots().is_empty());
        assert!(dag.leaves().is_empty());
    }

    // -- 2. Single step -------------------------------------------------------

    #[test]
    fn single_step_executes() {
        let steps = vec![make_step(1, vec![])];
        let mut dag = PipelineDag::build(steps).unwrap();
        assert_eq!(dag.len(), 1);
        let id1 = id_of(1);
        assert_eq!(dag.execution_order(), vec![id1]);
        assert_eq!(dag.roots(), vec![id1]);
        assert_eq!(dag.leaves(), vec![id1]);

        let mut executed = Vec::new();
        dag.execute_with(|step| {
            executed.push(step.id());
            Ok(())
        })
        .unwrap();
        assert_eq!(executed, vec![id1]);
        assert_eq!(dag.get(id1).unwrap().status, StepStatus::Completed);
    }

    // -- 3. Linear chain ------------------------------------------------------

    #[test]
    fn linear_chain_executes_in_order() {
        // 1 -> 2 -> 3 -> 4
        let steps = vec![
            make_step(1, vec![]),
            make_step(2, vec![1]),
            make_step(3, vec![2]),
            make_step(4, vec![3]),
        ];
        let mut dag = PipelineDag::build(steps).unwrap();
        let expected: Vec<StepId> = [1u64, 2, 3, 4].iter().map(|n| id_of(*n)).collect();
        assert_eq!(dag.execution_order(), expected);
        assert_eq!(dag.roots(), vec![id_of(1)]);
        assert_eq!(dag.leaves(), vec![id_of(4)]);

        let mut order = Vec::new();
        dag.execute_with(|step| {
            order.push(step.id());
            Ok(())
        })
        .unwrap();
        assert_eq!(order, expected);
    }

    // -- 4. Diamond DAG -------------------------------------------------------

    #[test]
    fn diamond_dag_respects_dependencies() {
        //     1
        //    / \
        //   2   3
        //    \ /
        //     4
        let steps = vec![
            make_step(1, vec![]),
            make_step(2, vec![1]),
            make_step(3, vec![1]),
            make_step(4, vec![2, 3]),
        ];
        let mut dag = PipelineDag::build(steps).unwrap();

        let order = dag.execution_order();
        assert_eq!(order.len(), 4);

        // 1 must come first
        assert_eq!(order[0], id_of(1));
        // 4 must come last
        assert_eq!(order[3], id_of(4));
        // 2 and 3 must both appear before 4 (and after 1)
        let pos_of = |id: StepId| order.iter().position(|&x| x == id).unwrap();
        assert!(pos_of(id_of(2)) > pos_of(id_of(1)));
        assert!(pos_of(id_of(3)) > pos_of(id_of(1)));
        assert!(pos_of(id_of(4)) > pos_of(id_of(2)));
        assert!(pos_of(id_of(4)) > pos_of(id_of(3)));

        let mut executed = Vec::new();
        dag.execute_with(|step| {
            executed.push(step.id());
            Ok(())
        })
        .unwrap();
        assert_eq!(executed.len(), 4);
    }

    // -- 5. Cycle detection ---------------------------------------------------

    #[test]
    fn cycle_detected() {
        // 1 -> 2 -> 3 -> 1  (cycle)
        let steps = vec![
            make_step(1, vec![3]),
            make_step(2, vec![1]),
            make_step(3, vec![2]),
        ];
        let result = PipelineDag::build(steps);
        match result {
            Err(PipelineError::CycleDetected { involved }) => {
                assert_eq!(involved.len(), 3);
                assert!(involved.contains(&id_of(1)));
                assert!(involved.contains(&id_of(2)));
                assert!(involved.contains(&id_of(3)));
            }
            other => panic!("expected CycleDetected, got {other:?}"),
        }
    }

    // -- 6. Missing dependency ------------------------------------------------

    #[test]
    fn missing_dependency_detected() {
        // Hand-build a step that references a non-existent dep id.
        let mut step = make_step_named("only", vec![]);
        let bogus: StepId = 0xDEAD_BEEF_DEAD_BEEF;
        step.depends_on = vec![bogus];
        match PipelineDag::build(vec![step]) {
            Err(PipelineError::MissingDependency {
                step_id,
                missing_dep,
            }) => {
                assert_eq!(step_id, step_id_of("only"));
                assert_eq!(missing_dep, bogus);
            }
            other => panic!("expected MissingDependency, got {other:?}"),
        }
    }

    // -- 7. Duplicate StepId --------------------------------------------------

    #[test]
    fn duplicate_step_id_detected() {
        let steps = vec![make_step(1, vec![]), make_step(1, vec![])];
        match PipelineDag::build(steps) {
            Err(PipelineError::DuplicateStepId { step_id }) => {
                assert_eq!(step_id, id_of(1));
            }
            other => panic!("expected DuplicateStepId, got {other:?}"),
        }
    }

    // -- 8. Step failure halts execution ---------------------------------------

    #[test]
    fn step_failure_halts_downstream() {
        let steps = vec![
            make_step(1, vec![]),
            make_step(2, vec![1]),
            make_step(3, vec![2]),
        ];
        let mut dag = PipelineDag::build(steps).unwrap();
        let id2 = id_of(2);

        let result = dag.execute_with(|step| {
            if step.id() == id2 {
                Err("boom".into())
            } else {
                Ok(())
            }
        });

        match result {
            Err(PipelineError::StepFailed {
                step_id,
                ref reason,
            }) if reason == "boom" && step_id == id2 => {}
            other => panic!("expected StepFailed(id_of(2), boom), got {other:?}"),
        }

        // Step 1 completed, step 2 failed, step 3 still pending.
        assert_eq!(dag.get(id_of(1)).unwrap().status, StepStatus::Completed);
        assert_eq!(dag.get(id_of(2)).unwrap().status, StepStatus::Failed);
        assert_eq!(dag.get(id_of(3)).unwrap().status, StepStatus::Pending);
    }

    // -- 9. Wide fan-out ------------------------------------------------------

    #[test]
    fn wide_fan_out_all_roots_execute_before_sink() {
        // 5 independent roots -> 1 sink
        let steps = vec![
            make_step(10, vec![]),
            make_step(20, vec![]),
            make_step(30, vec![]),
            make_step(40, vec![]),
            make_step(50, vec![]),
            make_step(99, vec![10, 20, 30, 40, 50]),
        ];
        let mut dag = PipelineDag::build(steps).unwrap();
        assert_eq!(dag.roots().len(), 5);
        assert_eq!(dag.leaves(), vec![id_of(99)]);

        let mut order = Vec::new();
        dag.execute_with(|step| {
            order.push(step.id());
            Ok(())
        })
        .unwrap();

        // Sink must be last.
        assert_eq!(*order.last().unwrap(), id_of(99));
        // All 5 roots must appear before the sink.
        let sink_pos = order.iter().position(|&x| x == id_of(99)).unwrap();
        assert_eq!(sink_pos, 5);
    }

    // -- 10. Two-node cycle ---------------------------------------------------

    #[test]
    fn two_node_cycle_detected() {
        let steps = vec![make_step(1, vec![2]), make_step(2, vec![1])];
        match PipelineDag::build(steps) {
            Err(PipelineError::CycleDetected { involved }) => {
                assert_eq!(involved.len(), 2);
            }
            other => panic!("expected CycleDetected, got {other:?}"),
        }
    }

    // -- 11. Self-loop cycle (LF-12 loose-end #3) ------------------------------

    #[test]
    fn cycle_detected_self_loop() {
        // A single step that depends on itself: 1 -> 1.
        let steps = vec![make_step(1, vec![1])];
        match PipelineDag::build(steps) {
            Err(PipelineError::CycleDetected { involved }) => {
                assert!(
                    involved.contains(&id_of(1)),
                    "self-loop must surface step 1 in `involved`, got {involved:?}",
                );
            }
            other => panic!("expected CycleDetected on self-loop, got {other:?}"),
        }
    }

    // -- 12. OrchestrationBridge adapter (LF-12 loose-end #2) ------------------

    #[test]
    fn execute_via_bridge_routes_each_step() {
        use lance_graph_contract::nars::{InferenceType, QueryStrategy, SemiringChoice};
        use lance_graph_contract::orchestration::{OrchestrationError, StepDomain};
        use lance_graph_contract::plan::ThinkingContext;
        use lance_graph_contract::thinking::{FieldModulation, ThinkingStyle};
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingBridge {
            count: AtomicUsize,
        }

        impl OrchestrationBridge for CountingBridge {
            fn route(&self, step: &mut UnifiedStep) -> Result<(), OrchestrationError> {
                self.count.fetch_add(1, Ordering::SeqCst);
                step.reasoning = Some(format!("routed:{}", step.step_id));
                Ok(())
            }
            fn resolve_thinking(
                &self,
                style: ThinkingStyle,
                inference_type: InferenceType,
            ) -> ThinkingContext {
                ThinkingContext {
                    style,
                    modulation: FieldModulation::default(),
                    inference_type,
                    strategy: QueryStrategy::CamExact,
                    semiring: SemiringChoice::Boolean,
                    free_will_modifier: 1.0,
                    exploratory: false,
                }
            }
            fn domain_available(&self, _domain: StepDomain) -> bool {
                true
            }
        }

        let bridge = CountingBridge {
            count: AtomicUsize::new(0),
        };
        let steps = vec![
            make_step(1, vec![]),
            make_step(2, vec![1]),
            make_step(3, vec![2]),
        ];
        let mut dag = PipelineDag::build(steps).unwrap();
        dag.execute_via_bridge(&bridge).unwrap();
        assert_eq!(bridge.count.load(Ordering::SeqCst), 3);
        assert_eq!(dag.get(id_of(1)).unwrap().status, StepStatus::Completed);
        assert_eq!(
            dag.get(id_of(1)).unwrap().reasoning.as_deref(),
            Some("routed:step-1"),
        );
    }
}
