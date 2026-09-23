//! # lance-graph-report — a coordinate-native fold algebra that reporting falls out of
//!
//! ```text
//! population + coordinate interpretation + selection + fold algebra + terminal view
//! ```
//!
//! **Never transpose data to pivot. Transpose meaning.**
//!
//! **Pivot defines coordinates. Fold populates coordinates. Materialization
//! renders coordinates.**
//!
//! **Coordinate change is not data change.** **Strings are presentation
//! metadata, not execution coordinates.** **Filter IDs first. Hydrate raw
//! values last.**
//!
//! This crate is NOT a second engine. A [`ReportPlan`] lowers into
//! `lance-graph-quack`'s `Query` and every program it runs is executed by
//! `lance-graph-mask-risc`'s one evaluator over resident SoA lanes, so Quack,
//! z8run and native Rust callers share one semantic representation (F10, A13).
//!
//! ## Three layers, never collapsed
//!
//! | layer | owns | here |
//! |---|---|---|
//! | **KV** | raw facts, variable-sized values | [`boundary::MemKv`] over the contract's `ContentStore`/`ContentSink` |
//! | **CAM / codebook** | human-readable labels ↔ canonical ordinals | [`boundary::CamLabels`] (a codebook of `ContentId`s — no text stored) |
//! | **ABI / SoA** | fixed-width ids, ordinals, measures, masks, folds | [`AbiBatch`], [`ReportPlan`], [`exec`], [`result`] |
//!
//! The execution modules (`ids`, `batch`, `selection`, `plan`, `exec`,
//! `result`) contain no `String`; a source fence (`tests/string_fence.rs`)
//! fails the build if one appears. Text lives only in [`boundary`] (ingest,
//! catalog, CAM, KV), [`explain`] (diagnostics) and [`render`] (terminal).
//!
//! ## Module map
//!
//! * [`ids`] — `FieldId`, `MaskId`, `SourceId`.
//! * [`batch`] — resident fixed-width lanes, borrowed never copied.
//! * [`selection`] — lazy selection; a predicate is never a bitmap by construction.
//! * [`plan`] — the control-plane plan: coordinate providers with roles,
//!   mergeable fold states, the role-free [`plan::PhysicalKey`].
//! * [`exec`] — physical planning (fold key / partitions / dense-vs-sparse /
//!   selection carrier) and the fold.
//! * [`result`] — the canonical-coordinate aggregate space and zero-copy views.
//! * [`boundary`] — the only string-accepting surface: KV, CAM, catalog, ingest.
//! * [`render`] — terminal adapters (JSON / CSV data export): the materialization boundary. Paged / screen output goes through OGAR (`lance-graph-report-ogar`).
//! * [`explain`] — the human-readable physical plan.

#![forbid(unsafe_code)]

pub mod batch;
pub mod boundary;
pub mod exec;
pub mod explain;
pub mod ids;
pub mod plan;
pub mod render;
pub mod result;
pub mod selection;

pub use batch::{AbiBatch, BatchError, Column, LaneData};
pub use exec::{Accumulator, ExecStats, PhysicalPlan, PlannerPolicy, Provider, SelectionCarrier};
pub use ids::{FieldId, MaskId, SourceId};
pub use plan::{
    AxisRole, AxisSpec, CellValue, CoordSpec, FoldState, Measure, MeasureKind, PhysicalKey,
    ReportPlan, SourceRef, TopK,
};
pub use result::{CellSpace, CellState, ReportResult, ReportShape, View};
pub use selection::{CmpOp, PredicatePlan, RowRange, Scalar, Selection};

use lance_graph_mask_risc::ExecError;
use lance_graph_quack::LowerError;

/// Everything that can refuse a report. Every refusal is structural and
/// happens before or instead of population work — never a partial answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportError {
    /// A row range past the population, or inverted.
    RangeOutOfBounds {
        /// First row.
        lo: u32,
        /// One past the last.
        hi: u32,
        /// Population size.
        n_rows: usize,
    },
    /// A selection names a mask the batch does not carry.
    UnknownMask(MaskId),
    /// A plan names a field the batch does not carry.
    UnknownField(FieldId),
    /// `IN ()`.
    EmptyIn(FieldId),
    /// A literal that does not fit the field's lane kind.
    BadLiteral {
        /// The field.
        field: FieldId,
        /// The literal.
        value: Scalar,
    },
    /// An ordered compare on an ordinal field (ordinals are categorical).
    OrderedCompareOnOrdinal(FieldId),
    /// A field used as a coordinate has no declared domain.
    NotACoordinate(FieldId),
    /// A bucket coordinate over a non-`I32` field or with width <= 0.
    BadBucket(FieldId),
    /// A measure over a non-`I32` field.
    NotAMeasure(FieldId),
    /// The plan was minted against another source or generation.
    StaleSource,
    /// `reinterpret` across plans whose folds differ.
    PhysicalKeyMismatch,
    /// Top-k names a measure the plan does not have.
    BadTopK,
    /// More resident planes than a `u16` addresses.
    TooManyPlanes,
    /// The plan needs more population passes than the policy allows.
    PassBudget {
        /// Passes needed.
        passes: u64,
        /// Allowed.
        budget: u64,
    },
    /// A coordinate domain too wide for a discovery / fold-key buffer.
    DomainBudget {
        /// The domain.
        domain: u32,
        /// Allowed.
        budget: u32,
    },
    /// Quack refused the lowering.
    Lower(LowerError),
    /// The evaluator refused the program.
    Exec(ExecError),
}

impl From<LowerError> for ReportError {
    fn from(e: LowerError) -> Self {
        ReportError::Lower(e)
    }
}

impl From<ExecError> for ReportError {
    fn from(e: ExecError) -> Self {
        ReportError::Exec(e)
    }
}

impl core::fmt::Display for ReportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ReportError {}
