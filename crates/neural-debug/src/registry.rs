use crate::diagnosis::NeuronState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// Runtime call counter for a single function.
/// Lock-free atomic increment — near-zero overhead when neural-debug is on.
pub struct CallCounter {
    pub count: AtomicU64,
    pub total_ns: AtomicU64,
    pub nan_count: AtomicU64,
}

impl CallCounter {
    pub const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
            nan_count: AtomicU64::new(0),
        }
    }

    pub fn record(&self, elapsed_ns: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
    }

    pub fn record_nan(&self) {
        self.nan_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            call_count: self.count.load(Ordering::Relaxed),
            total_ns: self.total_ns.load(Ordering::Relaxed),
            nan_count: self.nan_count.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterSnapshot {
    pub call_count: u64,
    pub total_ns: u64,
    pub nan_count: u64,
}

impl CounterSnapshot {
    pub fn avg_ns(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.total_ns as f64 / self.call_count as f64
        }
    }
}

/// Per-row state recorded by the runtime dispatch path (e.g. cognitive-
/// shader-driver). Maps a BindSpace row to its observed `NeuronState`
/// for the most recent cycle that touched it.
pub struct RowStateMap {
    states: Mutex<HashMap<u32, NeuronState>>,
}

impl RowStateMap {
    pub fn new() -> Self {
        Self {
            states: Mutex::new(HashMap::new()),
        }
    }

    pub fn record(&self, row: u32, state: NeuronState) {
        let mut map = self.states.lock().unwrap();
        map.insert(row, state);
    }

    pub fn snapshot(&self) -> HashMap<u32, NeuronState> {
        self.states.lock().unwrap().clone()
    }

    pub fn reset(&self) {
        self.states.lock().unwrap().clear();
    }

    /// Aggregate counts the WireHealth.neural_debug overlay needs.
    pub fn diag(&self) -> RuntimeDiag {
        let map = self.states.lock().unwrap();
        let mut alive = 0usize;
        let mut stat = 0usize;
        let mut dead = 0usize;
        let mut nan = 0usize;
        let mut stub = 0usize;
        let mut wired_unused = 0usize;
        for state in map.values() {
            match state {
                NeuronState::Alive => alive += 1,
                NeuronState::Static => stat += 1,
                NeuronState::Dead => dead += 1,
                NeuronState::Nan => nan += 1,
                NeuronState::Stub => stub += 1,
                NeuronState::WiredUnused => wired_unused += 1,
            }
        }
        let total = map.len();
        let operational = alive + stat;
        let health_pct = if total == 0 {
            0.0
        } else {
            (operational as f32 / total as f32) * 100.0
        };
        RuntimeDiag {
            total_functions: total,
            total_dead: dead,
            total_stub: stub,
            total_nan: nan,
            total_alive: alive,
            total_static: stat,
            total_wired_unused: wired_unused,
            health_pct,
        }
    }
}

impl Default for RowStateMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate runtime diagnostic produced from a `RowStateMap` snapshot —
/// shaped to match the `WireHealth.neural_debug` overlay so consumers can
/// drop it straight into the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeDiag {
    pub total_functions: usize,
    pub total_dead: usize,
    pub total_stub: usize,
    pub total_nan: usize,
    pub total_alive: usize,
    pub total_static: usize,
    pub total_wired_unused: usize,
    pub health_pct: f32,
}

/// Runtime registry: collects call counters from all instrumented functions
/// AND the per-row `NeuronState` feed from the dispatch hot path.
///
/// Thread-safe via interior mutability — `&RuntimeRegistry` is enough for
/// both the shader's `dispatch()` (writer) and the health handler (reader).
/// No `&mut self` anywhere on the hot path.
pub struct RuntimeRegistry {
    counters: Mutex<HashMap<String, CounterSnapshot>>,
    rows: RowStateMap,
}

impl Default for RuntimeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeRegistry {
    pub fn new() -> Self {
        Self {
            counters: Mutex::new(HashMap::new()),
            rows: RowStateMap::new(),
        }
    }

    pub fn record(&self, function_id: &str, elapsed: Duration, is_nan: bool) {
        let mut map = self.counters.lock().unwrap();
        let entry = map
            .entry(function_id.to_string())
            .or_insert(CounterSnapshot {
                call_count: 0,
                total_ns: 0,
                nan_count: 0,
            });
        entry.call_count += 1;
        entry.total_ns += elapsed.as_nanos() as u64;
        if is_nan {
            entry.nan_count += 1;
        }
    }

    /// Record a row's most recent `NeuronState`. Constant-time per call,
    /// no allocations beyond the (one-time) HashMap entry insertion.
    #[inline]
    pub fn record_row(&self, row: u32, state: NeuronState) {
        self.rows.record(row, state);
    }

    pub fn snapshot(&self) -> HashMap<String, CounterSnapshot> {
        self.counters.lock().unwrap().clone()
    }

    /// Snapshot every row's currently recorded state.
    pub fn snapshot_rows(&self) -> HashMap<u32, NeuronState> {
        self.rows.snapshot()
    }

    /// Aggregate runtime diagnostic from the current row-state map.
    pub fn diag(&self) -> RuntimeDiag {
        self.rows.diag()
    }

    pub fn reset(&self) {
        self.counters.lock().unwrap().clear();
        self.rows.reset();
    }
}

/// Process-wide global runtime registry.
///
/// Lazy-initialized via `OnceLock`. Producers (e.g. the shader dispatch
/// hot path) call `registry()` once per cycle to record row states;
/// consumers (e.g. the WireHealth handler) call `registry().diag()` to
/// snapshot the aggregate counts. No allocation outside the first call.
static GLOBAL_REGISTRY: OnceLock<RuntimeRegistry> = OnceLock::new();

/// Access the process-wide runtime registry, initializing it on first use.
pub fn registry() -> &'static RuntimeRegistry {
    GLOBAL_REGISTRY.get_or_init(RuntimeRegistry::new)
}

/// Dependency probe result — what happened when we tried to call a dependency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub name: String,
    pub location: String,
    pub status: NeuronState,
    pub latency_us: u64,
    pub error: Option<String>,
}

/// Strategy self-diagnosis — result of probing all dependencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyDiagnosis {
    pub strategy_name: String,
    pub strategy_index: usize,
    pub deps: Vec<ProbeResult>,
    pub deps_alive: usize,
    pub deps_total: usize,
    pub self_test_passed: Option<bool>,
    pub self_test_latency_us: Option<u64>,
    pub self_test_output: Option<String>,
    pub verdict: Verdict,
    pub bottleneck: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Ready,
    Partial,
    Dead,
    Nan,
    Stub,
}

impl Verdict {
    pub fn emoji(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Partial => "partial",
            Self::Dead => "dead",
            Self::Nan => "nan",
            Self::Stub => "stub",
        }
    }
}

/// Pipeline check — test a chain of strategies in sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCheckResult {
    pub name: String,
    pub stages: Vec<PipelineStageResult>,
    pub fully_operational: bool,
    pub broke_at_stage: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStageResult {
    pub strategy_name: String,
    pub strategy_index: usize,
    pub verdict: Verdict,
    pub broke_here: bool,
    pub error: Option<String>,
}

/// Full strategy health matrix — all strategies + pipeline checks + impact analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyHealthMatrix {
    pub strategies: Vec<StrategyDiagnosis>,
    pub pipelines: Vec<PipelineCheckResult>,
    pub impact_fixes: Vec<ImpactFix>,
    pub operational_count: usize,
    pub partial_count: usize,
    pub dead_count: usize,
    pub nan_count: usize,
    pub stub_count: usize,
    pub total_strategies: usize,
    pub scan_duration_ms: u64,
}

/// A prioritized fix recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactFix {
    pub location: String,
    pub state: NeuronState,
    pub blocks_strategies: Vec<String>,
    pub blocks_pipelines: Vec<String>,
    pub blocks_features: Vec<String>,
    pub fix_complexity: String,
    pub impact_score: f32,
}

// ═════════════════════════════════════════════════════════════════════
// Tests — TD-INT-11 row-state runtime registry
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod row_state_tests {
    use super::*;

    #[test]
    fn row_state_map_records_and_diags() {
        let map = RowStateMap::new();
        map.record(0, NeuronState::Alive);
        map.record(1, NeuronState::Alive);
        map.record(2, NeuronState::Static);
        map.record(3, NeuronState::Nan);

        let diag = map.diag();
        assert_eq!(diag.total_functions, 4);
        assert_eq!(diag.total_alive, 2);
        assert_eq!(diag.total_static, 1);
        assert_eq!(diag.total_nan, 1);
        assert_eq!(diag.total_dead, 0);
        assert_eq!(diag.total_stub, 0);
        // 3 of 4 are operational (alive + static); NaN is not
        // operational. health = 75%.
        assert!((diag.health_pct - 75.0).abs() < 1e-3);
    }

    #[test]
    fn row_state_map_overwrites_per_row() {
        let map = RowStateMap::new();
        map.record(7, NeuronState::Static);
        map.record(7, NeuronState::Alive);
        let snap = map.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[&7], NeuronState::Alive);
    }

    #[test]
    fn empty_registry_diag_is_zeroed() {
        let map = RowStateMap::new();
        let diag = map.diag();
        assert_eq!(diag.total_functions, 0);
        assert_eq!(diag.health_pct, 0.0);
    }

    #[test]
    fn runtime_registry_record_row_round_trips() {
        let reg = RuntimeRegistry::new();
        reg.record_row(42, NeuronState::Alive);
        reg.record_row(43, NeuronState::Static);
        let snap = reg.snapshot_rows();
        assert_eq!(snap[&42], NeuronState::Alive);
        assert_eq!(snap[&43], NeuronState::Static);
        let diag = reg.diag();
        assert_eq!(diag.total_functions, 2);
        assert_eq!(diag.total_alive, 1);
        assert_eq!(diag.total_static, 1);
        assert!((diag.health_pct - 100.0).abs() < 1e-3);
    }

    #[test]
    fn global_registry_is_a_single_oncelock() {
        // Two calls to `registry()` must return the same address —
        // producers and consumers in different modules MUST see the
        // same map.
        let a = registry() as *const RuntimeRegistry as usize;
        let b = registry() as *const RuntimeRegistry as usize;
        assert_eq!(a, b);
    }

    #[test]
    fn runtime_registry_reset_clears_rows() {
        let reg = RuntimeRegistry::new();
        reg.record_row(1, NeuronState::Alive);
        reg.reset();
        assert_eq!(reg.diag().total_functions, 0);
    }
}
