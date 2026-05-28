//! The [`Op<I,O>`] trait — identity + step hook + three call sites.
//!
//! Per E-OP-THREE-CALLSITES-1: one trait, three execution speeds,
//! one set of const data shared across all three. The `kind()` method
//! returns the [`OpKind`] discriminant that the cognitive shader
//! dispatches against (per `I-VSA-IDENTITIES`: identity in const data,
//! kernel logic in the shader).
//!
//! ## Call site summary
//!
//! | Method | Path | Caller |
//! |---|---|---|
//! | `step` (hook) + framework transition | Cold — single carrier, one-shot | `Interactive` context |
//! | `apply_stream` | Warm — async stream, flow-controlled (Stage 2) | `Bulk` context |
//! | `apply_soa` | Hot — SoA-swept SIMD, JIT-compiled (Stage 2) | `Periodisch` context |
//!
//! Stage 1 ships `step` (with default no-op) + the framework chain methods.
//! `apply_stream` and `apply_soa` are deferred to Stage 2; see
//! `TODO(Stage 2):` comments below.
//!
//! ## Cross-references
//! - Plan: `.claude/plans/normalized-entity-holy-grail-v1.md` §"The Op trait"
//! - Epiphanies: E-OP-THREE-CALLSITES-1, I-VSA-IDENTITIES

use super::entity::NormalizedEntity;
use super::stages::Stage;

// ── OpKind ────────────────────────────────────────────────────────────────────

/// Identity handle for an Op — the codebook entry the shader dispatches
/// against.
///
/// Per `I-VSA-IDENTITIES` + `E-CODEBOOK-INHERITS-FROM-OGIT`: the kind
/// IS the register; the kernel logic lives in the shader. Each concrete
/// Op implementation returns a unique discriminant from `kind()`.
///
/// `u32` to align with the OGIT codebook row-index width (PR #427
/// WitnessTable widening).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpKind(pub u32);

impl OpKind {
    /// Reserved sentinel for an Op whose body is not yet wired
    /// (`todo!()` body). Stage 2 replaces all uses with concrete codes
    /// from the ~50-kernel dispatch table.
    ///
    // TODO(Stage 2): Stage 2 enumerates the concrete kernel discriminants
    // (SkrAccountInRange, VatLiability, KontenerkennungSkr04, etc.)
    // and pins their u32 codes alongside the SAVANTS roster + OGIT
    // codebook. For Stage 1 we ship the trait shape only; consumers
    // register concrete OpKind values in their crates.
    pub const UNWIRED: OpKind = OpKind(0);
}

// ── Output ────────────────────────────────────────────────────────────────────

/// The final output of a chain — produced by
/// [`NormalizedEntity::<Reported>::output`](super::advance).
///
/// `output()` triggers cascade traversal per the enclosing transaction
/// context (see [`super::cascade`] and
/// [`crate::transaction::Context`]).
///
// TODO(Stage 2): the shape is TBD by Stage 2 once we have a concrete
// consumer. Likely expands to an enum over
// `(CommittedEdge, EmittedBaton, QueuedForEpoch)` — one variant per
// transaction context (Interactive / Bulk / Periodisch).
// The `success: bool` here is a Stage-1 placeholder.
#[derive(Debug, Clone, Copy)]
pub struct Output {
    /// Whether the full chain completed without escalation.
    ///
    /// `true` = committed to AriGraph + Baton emitted. `false` = chain
    /// escalated to the LLM resolver (the <25% confidence tail per
    /// CLAUDE.md "The Click").
    ///
    // TODO(Stage 2): Stage 2 replaces with a richer result type that carries
    // the committed `CausalEdge64` and the Baton target set.
    pub success: bool,
}

// ── Op trait ──────────────────────────────────────────────────────────────────

/// The chain-grammar Op. Identity + step-hook; three call sites.
///
/// Implementing an `Op<I,O>` means declaring a typed business kernel:
/// an `SkrAccountInRange`, a `VatLiability`, a `FiscalPositionResolver`.
/// The `kind()` discriminant tells the shader WHICH kernel to run.
/// The `step()` method is the validation + side-effect hook the
/// framework calls before performing the sealed stage transition.
///
/// ## Sealed stage transitions
///
/// External Op implementors override ONLY `step` + `kind`. The stage
/// transition itself (`NormalizedEntity<I>` → `NormalizedEntity<O>`) is
/// performed by the framework's chain methods (`op` / `chk_data` /
/// `review` / `abduct` / `report`) after `step` returns `Ok`. Implementors
/// cannot construct `NormalizedEntity<O>` directly because
/// `advance_stage_internal` is `pub(crate)`.
///
/// ## Why one trait, not three?
///
/// The Op holds const data (e.g. `SkrAccountInRange(8400..=8499)`) that
/// must be identical across all three call sites. Splitting into three
/// traits would force consumers to implement three times and risk
/// divergence. One trait keeps the const data in one place.
///
/// ## Stage 1 completeness
///
/// `step` has a default no-op success body — Stage 1 kernels can
/// implement only `kind()`. `apply_stream` and `apply_soa` are deferred
/// to Stage 2:
///
/// - `apply_stream` needs a `Stream` type; `futures::Stream` / std
///   `async_iter` (unstable) — decision deferred to Stage 2.
/// - `apply_soa` references `MailboxSoA<N>` from
///   `cognitive-shader-driver`, which is not yet a dep of contract.
///
/// ## Cross-references
/// - Epiphany E-OP-THREE-CALLSITES-1
/// - I-VSA-IDENTITIES (identity in const data)
/// - `crate::transaction::{Interactive, Bulk, Periodisch}`
pub trait Op<I: Stage, O: Stage>: Sized + 'static {
    /// Identity handle for this Op — the codebook entry the shader
    /// dispatches against. Per `I-VSA-IDENTITIES` +
    /// `E-CODEBOOK-INHERITS-FROM-OGIT`.
    fn kind(&self) -> OpKind;

    /// Inspect the entity at stage `I` + perform side-effects on the
    /// owning mailbox's SoA row via `entity.row`. Returns `Ok(())` to
    /// signal "advance to stage `O`"; `Err(...)` halts the chain.
    ///
    /// Default: no-op success (always advance). Override to add
    /// validation logic or to write into the SoA columns.
    ///
    /// External Op implementors override ONLY `step` + `kind`. The
    /// stage transition itself is performed by the framework after
    /// `step` returns `Ok` — implementors cannot construct
    /// `NormalizedEntity<O>` directly.
    fn step(&self, entity: &NormalizedEntity<I>) -> Result<(), OpError> {
        let _ = entity;
        Ok(())
    }

    // ── Warm path ──────────────────────────────────────────────────

    // TODO(Stage 2): `apply_stream` returns `impl Stream<Item = NormalizedEntity<O>>`
    // which requires a `Stream` abstraction in the contract crate. The
    // contract crate is currently zero-dep. Stage 2 decision: wire
    // `futures::Stream` (add futures-rs dev-dep? or stable
    // std::async_iter once stabilised?) OR define a minimal
    // `CognitionStream<T>` adapter in this crate. Until that decision
    // is made, `apply_stream` is NOT part of the trait surface.
    //
    // Warm path — async stream; one in / one out, flow-controlled.
    //
    // Used by the [`crate::transaction::Bulk`] context. The shader runs
    // the kernel per element with bounded parallelism; cascade Batons
    // batch per epoch.
    // fn apply_stream<S>(&self, s: S) -> impl Stream<Item = NormalizedEntity<O>>
    // where
    //     S: Stream<Item = NormalizedEntity<I>>;

    // ── Hot path ───────────────────────────────────────────────────

    // TODO(Stage 2): `apply_soa` references `MailboxSoA<N>` and a `BitMask`
    // type that live in `cognitive-shader-driver`, not in contract.
    // Adding that dep would break the "zero-dep contract" invariant.
    // Stage 2 either (a) defines a `SoaSweep` adapter trait here that
    // `cognitive-shader-driver` implements, or (b) moves the hot-path
    // call site to a separate `contract-hot` crate that CAN dep on
    // shader-driver. Not decided yet.
    //
    // Hot path — SoA-swept SIMD kernel over a mailbox; JIT-compiled
    // from the const-data Op + kernel handle. No allocation, no
    // virtual call.
    //
    // Used by the [`crate::transaction::Periodisch`] context.
    // fn apply_soa(&self, mb: &mut MailboxSoA<N>, mask: BitMask);
}

// ── OpError ───────────────────────────────────────────────────────────────────

/// Error returned from an Op's `step` to halt the chain.
///
/// Stage 1 carries only a static message; Stage 2 will widen to carry
/// the failing `OpKind` + a typed reason enum + the row reference for
/// audit trail purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpError {
    /// Static description of why the step was rejected.
    pub message: &'static str,
}

impl OpError {
    /// Construct an `OpError` from a static string.
    pub const fn new(message: &'static str) -> Self {
        Self { message }
    }
}
