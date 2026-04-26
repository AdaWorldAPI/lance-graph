//! # Execution Engine
//!
//! Morsel-driven pipeline execution (from Kuzudb/Polars pattern).
//! The logical plan is decomposed into pipelines at materialization boundaries.
//! Each pipeline is a chain of operators executed by worker threads.
//! Each thread processes a morsel (chunk) of data.

#[allow(unused_imports)] // Morsel intended for pipeline execution wiring
use crate::physical::{PhysicalPlan, Pipeline, Morsel};

/// Pipeline executor.
pub struct PipelineExecutor {
    /// Number of worker threads.
    pub num_threads: usize,
    /// Morsel size (rows per chunk).
    pub morsel_size: usize,
}

impl Default for PipelineExecutor {
    fn default() -> Self {
        Self {
            num_threads: num_cpus(),
            morsel_size: 2048,
        }
    }
}

impl PipelineExecutor {
    pub fn new(num_threads: usize, morsel_size: usize) -> Self {
        Self { num_threads, morsel_size }
    }

    /// Decompose a physical plan into pipelines.
    pub fn decompose(&self, _plan: &PhysicalPlan) -> Vec<Pipeline> {
        // Walk the operator tree. Split at pipeline breakers.
        // Each pipeline is a chain from source to sink (or breaker).
        // Dependencies track which pipelines must complete first.
        //
        // For BROADCAST → SCAN → ACCUMULATE → COLLAPSE:
        // This is a single pipeline (no breakers), parallelized across partitions.
        vec![Pipeline {
            id: 0,
            operators: vec![0], // Placeholder
            dependencies: vec![],
        }]
    }

    /// Execute all pipelines respecting dependencies.
    pub fn execute(&self, plan: &PhysicalPlan) -> ExecutionResult {
        let pipelines = self.decompose(plan);

        // Topological sort of pipelines by dependencies
        let execution_order = toposort(&pipelines);

        let mut total_rows = 0u64;

        for pipeline_id in execution_order {
            let _pipeline = &pipelines[pipeline_id];
            // In real implementation:
            // 1. Create morsel sources for leaf operators
            // 2. Spawn worker tasks (one per thread)
            // 3. Each worker pulls a morsel, processes through pipeline
            // 4. Collect results at sink
            total_rows += self.morsel_size as u64; // Placeholder
        }

        ExecutionResult { total_rows }
    }
}

/// Result of execution.
#[derive(Debug)]
pub struct ExecutionResult {
    pub total_rows: u64,
}

/// Topological sort of pipelines by dependencies.
fn toposort(pipelines: &[Pipeline]) -> Vec<usize> {
    let n = pipelines.len();
    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for p in pipelines {
        for &dep in &p.dependencies {
            adj[dep].push(p.id);
            in_degree[p.id] += 1;
        }
    }

    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);

    while let Some(node) = queue.pop() {
        order.push(node);
        for &next in &adj[node] {
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                queue.push(next);
            }
        }
    }

    order
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
