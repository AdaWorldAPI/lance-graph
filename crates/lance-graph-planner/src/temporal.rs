//! Temporal epistemology + deinterlacing — query-time, planner-layer policy.
//!
//! ## The deinterlace problem
//!
//! Four producers write into the same Lance dataset, each on **its own clock**:
//!
//! | Interlaced frame | Its clock |
//! |---|---|
//! | **lance** (storage) | per-writer monotonic version stream |
//! | **surrealql** (schema) | `knowable_from` — when a class/field became defined |
//! | **ractor** (awareness) | each actor's own `V_ref` reading-horizon |
//! | **thinking** (cognition) | the Markov ±5 `CognitiveEventRow` trajectory |
//!
//! A naïve "read current state" yields a **combed frame** — object state from
//! writer A torn against a schema row from a stale writer B — exactly the
//! interlacing artifact in video. The hybrid-logical-clock tick
//! `(server_id, lance_version, hlc_tick)` is the **deinterlace key**: merge-sort
//! by it and every field's row lands on one timeline. The result *is* the
//! standing wave / kanban SoA — the causally-coherent projection a reader
//! deliberates over.
//!
//! ## Why this is the planner's job, not storage's
//!
//! Epistemology is a **query-level annotation, not storage**. The Lance versions
//! already carry the temporal information; the planner picks the policy (which
//! rung opts into which mode) and runs the per-row classification. None of these
//! types belong in `ogar-vocab` — they describe *how a reader saw a row at a
//! point in time*, not what a class *is*.
//!
//! ## Two causal axes
//!
//! - **TIME-causal** (this module): the HLC tick → [`classify`] → which row is
//!   contemporary at the reader's `V_ref`.
//! - **DATA-causal** (deferred; sourced from the SPO `ModelGraph`'s `depends_on`
//!   / `enables` edges): which rows must precede which. A row is truly *ready*
//!   only when it is both time-contemporary **and** its data-dependencies are
//!   satisfied. [`deinterlace`] exposes the seam; the closure is a deferred
//!   policy, not a shape change.
//!
//! ## Cross-server is type-visible, policy-deferred
//!
//! [`QueryReference`] carries `server_id` + `hlc_tick: Option<u64>` from day one,
//! so the single-server body can ignore them and the cross-server policy (peer-
//! Raft / cluster bus) wakes them up with **no breaking signature change** —
//! avoiding the `emitted_at_millis: u64` (decision #4) non-`Option` trap.

/// A Lance dataset version — the storage frame's clock tick.
pub type LanceVersion = u64;

/// What a reader at a given rung is *allowed to know* while it deliberates.
/// Maps to the temporal-epistemology framework (`STRICT` / `AWARE` / `RETRO`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpistemicMode {
    /// Only `CONTEMPORARY` rows (`row_version ≤ ref_version`). The default.
    Strict,
    /// May also use `ANACHRONISTIC` rows — hindsight from a future frame.
    Aware,
    /// May also take a `SPOILER` — an intentional `V_now` read past the
    /// horizon (rung 9+).
    Retro,
}

impl EpistemicMode {
    /// The initial rung → mode policy (tunable). Low rungs reason strictly in
    /// the present; mid rungs admit hindsight; top rungs may spoiler-read.
    #[must_use]
    pub fn for_rung(rung: u8) -> Self {
        match rung {
            0..=4 => EpistemicMode::Strict,
            5..=8 => EpistemicMode::Aware,
            _ => EpistemicMode::Retro,
        }
    }

    /// Whether a reader in this mode may dispatch on a row with the given
    /// [`TemporalStatus`]. `Unknowable` is never admitted (the class was not yet
    /// registered at the horizon).
    #[must_use]
    pub fn admits(self, status: TemporalStatus) -> bool {
        match status {
            TemporalStatus::Contemporary => true,
            TemporalStatus::Anachronistic => {
                matches!(self, EpistemicMode::Aware | EpistemicMode::Retro)
            }
            TemporalStatus::Spoiler => matches!(self, EpistemicMode::Retro),
            TemporalStatus::Unknowable => false,
        }
    }
}

/// The per-row deinterlace decision — how a reader at `V_ref` relates to a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporalStatus {
    /// In-phase: `row_version ≤ ref_version` and the class was already knowable.
    Contemporary,
    /// A future frame's row (`row_version > ref_version`) — hindsight.
    Anachronistic,
    /// An intentional read past the horizon under `Retro` mode.
    Spoiler,
    /// The class's `knowable_from` is past the horizon — not yet knowable.
    Unknowable,
}

/// A reader's awareness reference — the lens [`classify`] reads a row through.
///
/// HLC-aware by construction: `server_id` + `hlc_tick` are carried so the
/// cross-server policy is a non-breaking addition. Single-server readers use
/// [`QueryReference::default`] (latest version, strict, rung 0, no HLC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryReference {
    /// The reader's frame of reference (which writer's version line). `0` =
    /// single-server.
    pub server_id: u16,
    /// The `KnowledgeHorizon` — the Lance version the reader is pinned at.
    /// `u64::MAX` = "latest" (the single-server default).
    pub ref_version: LanceVersion,
    /// Cross-server causal tick; `None` single-server. Wakes up under the
    /// peer-Raft / cluster-bus policy (deferred).
    pub hlc_tick: Option<u64>,
    /// What the reader is allowed to know.
    pub mode: EpistemicMode,
    /// The reader's rung (drives [`EpistemicMode::for_rung`]).
    pub rung: u8,
}

impl Default for QueryReference {
    fn default() -> Self {
        // The single-server reading: latest version, strict, rung 0, no HLC.
        Self {
            server_id: 0,
            ref_version: u64::MAX,
            hlc_tick: None,
            mode: EpistemicMode::Strict,
            rung: 0,
        }
    }
}

impl QueryReference {
    /// Build a reference pinned at `ref_version` with the mode derived from
    /// `rung` (single-server: no HLC, `server_id` 0).
    #[must_use]
    pub fn at(ref_version: LanceVersion, rung: u8) -> Self {
        Self {
            server_id: 0,
            ref_version,
            hlc_tick: None,
            mode: EpistemicMode::for_rung(rung),
            rung,
        }
    }
}

/// Classify a row against a reader's reference — the per-row deinterlace
/// decision (the TIME-causal axis).
///
/// `Unknowable` if the class was not yet registered at the horizon; otherwise
/// `Contemporary` if in-phase; otherwise a future-frame row, which is a
/// `Spoiler` under `Retro` (an intentional peek) and `Anachronistic` otherwise.
#[must_use]
pub fn classify(
    row_version: LanceVersion,
    knowable_from: LanceVersion,
    v_ref: &QueryReference,
) -> TemporalStatus {
    if knowable_from > v_ref.ref_version {
        TemporalStatus::Unknowable
    } else if row_version <= v_ref.ref_version {
        TemporalStatus::Contemporary
    } else if matches!(v_ref.mode, EpistemicMode::Retro) {
        TemporalStatus::Spoiler
    } else {
        TemporalStatus::Anachronistic
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DATA-causal axis — the depends-closure seam (type-visible, trivial body)
//
// Symmetric to the HLC split on the TIME axis: the `DependsClosure` hook is
// type-visible from day one; the single-server body is the trivial `NoDeps`
// impl. The real SPO source (ruff_*_spo `depends_on` / `reads_field`, weeks of
// producer work away) implements the trait without changing any signature here.
// Rubicon's `KausalSpec::Depends` guard talks to this trait, not to a producer —
// opaque, exactly as `CommitHook` is opaque to the membrane.
// ─────────────────────────────────────────────────────────────────────────────

/// One SPO data-dependency edge (`subject predicate object`) — e.g.
/// `("sale.order", "depends_on", "sale.order.line.price")`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepEdge {
    /// The dependent subject (a canonical OGAR identity string — *not* an
    /// `ogar-vocab` type; the planner stays decoupled from the producer crate).
    pub subject: String,
    /// One of the SPO core's closed predicates (`depends_on` / `reads_field` / …).
    pub predicate: String,
    /// The depended-upon object.
    pub object: String,
}

/// A subject's data-dependency closure at a reference, plus whether it is
/// satisfied. The `DependsClosure` impl owns the satisfaction logic; this module
/// only consumes [`satisfied`](Self::satisfied).
#[derive(Debug, Clone)]
pub struct DepClosure {
    /// The edges that must hold for the subject to be data-causally ready.
    pub edges: Vec<DepEdge>,
    /// Whether those edges are satisfied at the reference. The trivial impl
    /// reports `true` (no deps); a real SPO-backed impl checks the edges
    /// against the deinterlaced state.
    pub satisfied: bool,
}

impl DepClosure {
    /// A satisfied, dependency-free closure — the trivial result.
    #[must_use]
    pub fn ready() -> Self {
        Self {
            edges: Vec::new(),
            satisfied: true,
        }
    }
}

/// The trivial/empty closure IS the ready case — `Default` matches
/// [`DepClosure::ready`] so `..Default::default()` and the `derive(Default)`
/// reflex don't silently produce `satisfied: false` (which would make
/// [`deinterlace`] drop every otherwise-contemporary row). Codex P2 on #468.
impl Default for DepClosure {
    fn default() -> Self {
        Self::ready()
    }
}

/// The DATA-causal axis seam. Returns the `depends_on` closure for `subject` at
/// `v_ref`. Implemented by the consumer against whatever SPO source it has
/// (in-memory ndjson, the lance SPO store, an HTTP query) — Rubicon's `Depends`
/// guard is one such impl.
pub trait DependsClosure {
    /// The dependency closure for `subject` as-of `v_ref`.
    fn closure_at(&self, subject: &str, v_ref: &QueryReference) -> DepClosure;
}

/// The trivial DATA-causal impl — no dependencies, always ready. The
/// single-server default until the SPO frontends emit real edges.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoDeps;

impl DependsClosure for NoDeps {
    fn closure_at(&self, _subject: &str, _v_ref: &QueryReference) -> DepClosure {
        DepClosure::ready()
    }
}

/// The two-axis classification of a row for a reader: its TIME-causal
/// [`TemporalStatus`] and its DATA-causal readiness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Classification {
    /// The TIME-causal status (HLC axis).
    pub temporal: TemporalStatus,
    /// The DATA-causal readiness (depends-closure axis); `true` under `NoDeps`.
    pub data_ready: bool,
}

impl Classification {
    /// Whether a reader in `mode` may dispatch on this row — admitted on the
    /// time axis **and** ready on the data axis (the conjunction the standing
    /// wave requires under concurrent writers).
    #[must_use]
    pub fn dispatchable(&self, mode: EpistemicMode) -> bool {
        mode.admits(self.temporal) && self.data_ready
    }
}

/// Classify a row on **both** axes — the full deinterlace decision. Time-causal
/// via [`classify`]; data-causal via the [`DependsClosure`] hook (trivial under
/// [`NoDeps`]).
#[must_use]
pub fn classify_ready<D: DependsClosure>(
    subject: &str,
    row_version: LanceVersion,
    knowable_from: LanceVersion,
    v_ref: &QueryReference,
    deps: &D,
) -> Classification {
    Classification {
        temporal: classify(row_version, knowable_from, v_ref),
        data_ready: deps.closure_at(subject, v_ref).satisfied,
    }
}

/// A row that can be deinterlaced — exposes the clocks [`deinterlace`] merges on
/// plus its subject identity (for the data-causal closure lookup).
pub trait DeinterlaceRow {
    /// The subject's canonical identity (for the [`DependsClosure`] lookup).
    fn subject(&self) -> &str;
    /// The storage-frame clock (this row's Lance version).
    fn lance_version(&self) -> LanceVersion;
    /// The schema-frame clock (when this row's class became knowable). Sourced
    /// by `ogar-adapter-surrealql`'s `DEFINE TABLE` registration.
    fn knowable_from(&self) -> LanceVersion;
    /// The cross-server causal tick; `None` single-server.
    fn hlc_tick(&self) -> Option<u64> {
        None
    }
}

/// Deinterlace interlaced rows into the causally-coherent, *dispatchable* view
/// as-of `v_ref` — the standing-wave projection the reader deliberates over
/// (e.g. the `domain` a Rubicon `evaluate_guard` reads).
///
/// - Keeps only rows that are both mode-[`admitted`](EpistemicMode::admits) on
///   the TIME axis and data-ready on the DATA axis (via `deps`).
/// - Orders by the HLC tick when present (cross-server progressive scan),
///   falling back to `lance_version` (single-server).
///
/// Both deferred axes — cross-server HLC and the SPO `depends_on` closure — are
/// type-visible here; their bodies are trivial (`hlc_tick == None`, `NoDeps`)
/// until the cluster bus and the SPO frontends land. No signature changes when
/// they do.
#[must_use]
pub fn deinterlace<R, D>(rows: &[R], v_ref: &QueryReference, deps: &D) -> Vec<R>
where
    R: DeinterlaceRow + Clone,
    D: DependsClosure,
{
    let mut out: Vec<R> = rows
        .iter()
        .filter(|r| {
            classify_ready(
                r.subject(),
                r.lance_version(),
                r.knowable_from(),
                v_ref,
                deps,
            )
            .dispatchable(v_ref.mode)
        })
        .cloned()
        .collect();
    // Falls back to the row's own lance_version when no HLC tick is present
    // (single-server / legacy rows) — NOT to 0, which would force every
    // missing-HLC row ahead of all HLC rows regardless of its version (Codex
    // P2 on #468; honors the documented "fallback to lance_version").
    out.sort_by_key(|r| {
        (
            r.hlc_tick().unwrap_or_else(|| r.lance_version()),
            r.lance_version(),
        )
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct Row {
        subj: String,
        v: LanceVersion,
        knowable: LanceVersion,
        hlc: Option<u64>,
    }
    impl Row {
        fn new(v: LanceVersion, knowable: LanceVersion, hlc: Option<u64>) -> Self {
            Self {
                subj: "ogit-erp/sale.order/1".into(),
                v,
                knowable,
                hlc,
            }
        }
    }
    impl DeinterlaceRow for Row {
        fn subject(&self) -> &str {
            &self.subj
        }
        fn lance_version(&self) -> LanceVersion {
            self.v
        }
        fn knowable_from(&self) -> LanceVersion {
            self.knowable
        }
        fn hlc_tick(&self) -> Option<u64> {
            self.hlc
        }
    }

    /// A `DependsClosure` that reports nothing satisfied — the DATA-causal axis
    /// actively blocking otherwise time-contemporary rows.
    struct BlockDeps;
    impl DependsClosure for BlockDeps {
        fn closure_at(&self, subject: &str, _v: &QueryReference) -> DepClosure {
            DepClosure {
                edges: vec![DepEdge {
                    subject: subject.to_string(),
                    predicate: "depends_on".into(),
                    object: "unmet".into(),
                }],
                satisfied: false,
            }
        }
    }

    #[test]
    fn for_rung_policy() {
        assert_eq!(EpistemicMode::for_rung(0), EpistemicMode::Strict);
        assert_eq!(EpistemicMode::for_rung(4), EpistemicMode::Strict);
        assert_eq!(EpistemicMode::for_rung(5), EpistemicMode::Aware);
        assert_eq!(EpistemicMode::for_rung(8), EpistemicMode::Aware);
        assert_eq!(EpistemicMode::for_rung(9), EpistemicMode::Retro);
        assert_eq!(EpistemicMode::for_rung(255), EpistemicMode::Retro);
    }

    #[test]
    fn classify_time_axis() {
        let strict = QueryReference::at(100, 0);
        // in-phase
        assert_eq!(classify(50, 10, &strict), TemporalStatus::Contemporary);
        assert_eq!(classify(100, 100, &strict), TemporalStatus::Contemporary);
        // class not yet knowable at the horizon
        assert_eq!(classify(50, 101, &strict), TemporalStatus::Unknowable);
        // a future-frame row, strict reader → Anachronistic
        assert_eq!(classify(101, 10, &strict), TemporalStatus::Anachronistic);
        // same future row, retro reader → Spoiler
        let retro = QueryReference::at(100, 9);
        assert_eq!(classify(101, 10, &retro), TemporalStatus::Spoiler);
    }

    #[test]
    fn admits_per_mode() {
        assert!(EpistemicMode::Strict.admits(TemporalStatus::Contemporary));
        assert!(!EpistemicMode::Strict.admits(TemporalStatus::Anachronistic));
        assert!(EpistemicMode::Aware.admits(TemporalStatus::Anachronistic));
        assert!(!EpistemicMode::Aware.admits(TemporalStatus::Spoiler));
        assert!(EpistemicMode::Retro.admits(TemporalStatus::Spoiler));
        // Unknowable is never admitted.
        for m in [
            EpistemicMode::Strict,
            EpistemicMode::Aware,
            EpistemicMode::Retro,
        ] {
            assert!(!m.admits(TemporalStatus::Unknowable));
        }
    }

    #[test]
    fn deinterlace_filters_and_orders_single_server() {
        // Reference at version 100, strict (rung 0).
        let v_ref = QueryReference::at(100, 0);
        let rows = vec![
            Row::new(30, 10, None),  // contemporary
            Row::new(150, 10, None), // anachronistic → dropped (strict)
            Row::new(20, 200, None), // unknowable → dropped
            Row::new(10, 5, None),   // contemporary
        ];
        let out = deinterlace(&rows, &v_ref, &NoDeps);
        // Only the two contemporary rows survive, ordered by version.
        assert_eq!(out.iter().map(|r| r.v).collect::<Vec<_>>(), vec![10, 30]);
    }

    #[test]
    fn data_causal_axis_can_drop_time_contemporary_rows() {
        // A row can be in-phase on the TIME axis yet not data-ready — the
        // conjunction is what the standing wave requires under concurrent writers.
        let v_ref = QueryReference::at(100, 0);
        let rows = vec![Row::new(30, 10, None)]; // time-contemporary
        assert_eq!(deinterlace(&rows, &v_ref, &NoDeps).len(), 1); // ready under NoDeps
        assert_eq!(deinterlace(&rows, &v_ref, &BlockDeps).len(), 0); // blocked on data axis

        let c = classify_ready("s", 30, 10, &v_ref, &BlockDeps);
        assert_eq!(c.temporal, TemporalStatus::Contemporary);
        assert!(!c.data_ready);
        assert!(!c.dispatchable(v_ref.mode));
    }

    #[test]
    fn deinterlace_hlc_orders_across_frames() {
        // Two frames interlaced; HLC tick is the deinterlace key (cross-server
        // progressive scan). All contemporary, admitted under strict.
        let v_ref = QueryReference {
            ref_version: 1000,
            ..QueryReference::default()
        };
        let rows = vec![
            Row::new(900, 1, Some(3)),
            Row::new(100, 1, Some(1)),
            Row::new(500, 1, Some(2)),
        ];
        let out = deinterlace(&rows, &v_ref, &NoDeps);
        // Ordered by hlc, NOT by per-frame lance_version.
        assert_eq!(
            out.iter().map(|r| r.hlc.unwrap()).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn query_reference_default_is_single_server_latest() {
        let q = QueryReference::default();
        assert_eq!(q.server_id, 0);
        assert_eq!(q.ref_version, u64::MAX);
        assert_eq!(q.hlc_tick, None);
        assert_eq!(q.mode, EpistemicMode::Strict);
    }

    /// Codex P2 on #468: `DepClosure::default()` must be ready (matching
    /// [`DepClosure::ready`] + [`NoDeps`]), not `satisfied: false` — otherwise
    /// any consumer using `..Default::default()` silently blocks every row.
    #[test]
    fn dep_closure_default_is_ready_not_blocking() {
        let d = DepClosure::default();
        assert!(
            d.satisfied,
            "Default DepClosure must be ready (satisfied: true)"
        );
        assert!(d.edges.is_empty());
    }

    /// Codex P2 on #468: a row without an HLC tick must fall back to its own
    /// `lance_version` as the deinterlace key — NOT to 0 (which would force
    /// legacy rows ahead of all HLC rows during partial cross-server rollout).
    #[test]
    fn deinterlace_mixed_hlc_falls_back_to_lance_version() {
        let v_ref = QueryReference {
            ref_version: 1000,
            ..QueryReference::default()
        };
        // Two HLC-bearing rows + one legacy row (no HLC, lance_version 750).
        // With the old `unwrap_or(0)` the legacy row would sort first; with
        // the fallback to lance_version it sorts between HLCs 500 and 900.
        let rows = vec![
            Row::new(100, 1, Some(500)),
            Row::new(750, 1, None),
            Row::new(900, 1, Some(900)),
        ];
        let out = deinterlace(&rows, &v_ref, &NoDeps);
        let order: Vec<u64> = out.iter().map(|r| r.hlc.unwrap_or(r.v)).collect();
        assert_eq!(
            order,
            vec![500, 750, 900],
            "legacy row must sort by its lance_version, not 0"
        );
    }
}
