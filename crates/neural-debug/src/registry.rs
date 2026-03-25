use crate::diagnosis::NeuronState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
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

/// Runtime registry: collects call counters from all instrumented functions.
/// Thread-safe via interior mutability.
pub struct RuntimeRegistry {
    counters: Mutex<HashMap<String, CounterSnapshot>>,
}

impl RuntimeRegistry {
    pub fn new() -> Self {
        Self {
            counters: Mutex::new(HashMap::new()),
        }
    }

    pub fn record(&self, function_id: &str, elapsed: Duration, is_nan: bool) {
        let mut map = self.counters.lock().unwrap();
        let entry = map.entry(function_id.to_string()).or_insert(CounterSnapshot {
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

    pub fn snapshot(&self) -> HashMap<String, CounterSnapshot> {
        self.counters.lock().unwrap().clone()
    }

    pub fn reset(&self) {
        self.counters.lock().unwrap().clear();
    }
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
