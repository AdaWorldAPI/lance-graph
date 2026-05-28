//! The [`Op<I,O>`] trait — identity + three call sites.
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
//! | `apply` | Cold — single carrier, one-shot | `Interactive` context |
//! | `apply_stream` | Warm — async stream, flow-controlled | `Bulk` context |
//! | `apply_soa` | Hot — SoA-swept SIMD, JIT-compiled | `Periodisch` context |
//!
//! Stage 1 ships `apply` only. `apply_stream` and `apply_soa` are
//! documented as deferred to Stage 2; see `/// work` below.
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
    /// /// work: Stage 2 enumerates the concrete kernel discriminants
    /// /// (SkrAccountInRange, VatLiability, KontenerkennungSkr04, etc.)
    /// /// and pins their u32 codes alongside the SAVANTS roster + OGIT
    /// /// codebook. For Stage 1 we ship the trait shape only; consumers
    /// /// register concrete OpKind values in their crates.
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
/// /// work: the shape is TBD by Stage 2 once we have a concrete
/// /// consumer. Likely expands to an enum over
/// /// `(CommittedEdge, EmittedBaton, QueuedForEpoch)` — one variant per
/// /// transaction context (Interactive / Bulk / Periodisch).
/// /// The `success: bool` here is a Stage-1 placeholder.
#[derive(Debug, Clone, Copy)]
pub struct Output {
    /// Whether the full chain completed without escalation.
    ///
    /// `true` = committed to AriGraph + Baton emitted. `false` = chain
    /// escalated to the LLM resolver (the <25% confidence tail per
    /// CLAUDE.md "The Click").
    ///
    /// /// work: Stage 2 replaces with a richer result type that carries
    /// /// the committed `CausalEdge64` and the Baton target set.
    pub success: bool,
}

// ── Op trait ──────────────────────────────────────────────────────────────────

/// The chain-grammar Op. Same const-data identity, three call sites.
///
/// Implementing an `Op<I,O>` means declaring a typed business kernel:
/// an `SkrAccountInRange`, a `VatLiability`, a `FiscalPositionResolver`.
/// The `kind()` discriminant tells the shader WHICH kernel to run; the
/// `apply` / `apply_stream` / `apply_soa` bodies are the call sites the
/// transaction context picks between.
///
/// ## Why one trait, not three?
///
/// The Op holds const data (e.g. `SkrAccountInRange(8400..=8499)`) that
/// must be identical across all three call sites. Splitting into three
/// traits would force consumers to implement three times and risk
/// divergence. One trait with three method forms keeps the const data
/// in one place and lets the context dispatch to the right form.
///
/// ## Stage 1 completeness
///
/// Only `apply` (cold) is required to be non-`todo!()` in Stage 1.
/// `apply_stream` and `apply_soa` are left as deferred pending the
/// async + SoA dependencies in Stage 2:
///
/// - `apply_stream` needs a `Stream` type; `futures::Stream` / std
///   `async_iter` (unstable) — decision deferred to Stage 2.
/// - `apply_soa` references `MailboxSoA<N>` from
///   `cognitive-shader-driver`, which is not yet a dep of contract.
///
/// Both are documented in the `/// work` comments below.
///
/// ## Cross-references
/// - Epiphany E-OP-THREE-CALLSITES-1
/// - I-VSA-IDENTITIES (identity in const data)
/// - `crate::transaction::{Interactive, Bulk, Periodisch}`
pub trait Op<I: Stage, O: Stage>: Sized + 'static {
    /// Const-data identity of this Op — the codebook discriminant the
    /// shader dispatches against. Per `I-VSA-IDENTITIES`, this is the
    /// register; kernel logic lives in the shader, not here.
    fn kind(&self) -> OpKind;

    // ── Cold path ──────────────────────────────────────────────────

    /// Cold path — single carrier, one-shot dispatch.
    ///
    /// Used by the [`crate::transaction::Interactive`] context: eager,
    /// synchronous, one entity at a time. Dispatches the shader kernel
    /// once against the carrier's SoA row, returns the advanced entity.
    ///
    /// Stage 1: implementors MUST provide a body (even `todo!()`).
    fn apply(&self, entity: NormalizedEntity<I>) -> NormalizedEntity<O>;

    // ── Warm path ──────────────────────────────────────────────────

    // /// work: `apply_stream` returns `impl Stream<Item = NormalizedEntity<O>>`
    // /// which requires a `Stream` abstraction in the contract crate. The
    // /// contract crate is currently zero-dep. Stage 2 decision: wire
    // /// `futures::Stream` (add futures-rs dev-dep? or stable
    // /// std::async_iter once stabilised?) OR define a minimal
    // /// `CognitionStream<T>` adapter in this crate. Until that decision
    // /// is made, `apply_stream` is NOT part of the trait surface.
    // ///
    // /// Warm path — async stream; one in / one out, flow-controlled.
    // ///
    // /// Used by the [`crate::transaction::Bulk`] context. The shader runs
    // /// the kernel per element with bounded parallelism; cascade Batons
    // /// batch per epoch.
    // fn apply_stream<S>(&self, s: S) -> impl Stream<Item = NormalizedEntity<O>>
    // where
    //     S: Stream<Item = NormalizedEntity<I>>;

    // ── Hot path ───────────────────────────────────────────────────

    // /// work: `apply_soa` references `MailboxSoA<N>` and a `BitMask`
    // /// type that live in `cognitive-shader-driver`, not in contract.
    // /// Adding that dep would break the "zero-dep contract" invariant.
    // /// Stage 2 either (a) defines a `SoaSweep` adapter trait here that
    // /// `cognitive-shader-driver` implements, or (b) moves the hot-path
    // /// call site to a separate `contract-hot` crate that CAN dep on
    // /// shader-driver. Not decided yet.
    // ///
    // /// Hot path — SoA-swept SIMD kernel over a mailbox; JIT-compiled
    // /// from the const-data Op + kernel handle. No allocation, no
    // /// virtual call.
    // ///
    // /// Used by the [`crate::transaction::Periodisch`] context.
    // fn apply_soa(&self, mb: &mut MailboxSoA<N>, mask: BitMask);
}
