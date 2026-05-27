//! Odoo savant reasoners — the lance-graph "thinking" side of the 25-savant
//! delegation (D-ODOO-SAV-4).
//!
//! woa-rs keeps the deterministic AXIS-A guard and **consumes** these
//! conclusions as native shared types (the one-binary contract: woa-rs links
//! this crate, so a [`SavantConclusion`] is the *same struct* on both sides —
//! nothing is serialized between them). The ambiguous, evidence-weighted
//! AXIS-B core lives here.
//!
//! Dispatch is **one [`Reasoner`] impl per [`ReasoningKind`]** (the pinned
//! decision, lance-graph PR #419), not 25 separate impls:
//! [`CustomerCategoryReasoner`] · [`PostingAnomalyReasoner`] ·
//! [`NextBestActionReasoner`] · [`OtherReasoner`] cover all 25 savants in
//! [`lance_graph_contract::savants::SAVANTS`]. Each resolves the concrete savant
//! from the context, reads its tuple, selects the [`QueryStrategy`] via
//! `InferenceType::default_strategy()`, fuses the evidence into a NARS
//! `(frequency, confidence)`, and returns a **suggestion only** — woa-rs applies
//! it as a default, never an un-guarded write (verhaltens-bewahrend).
//!
//! No serialization anywhere in this module. JSON exists only at the
//! callcenter ↔ MedCareV2 FFI boundary, never on these types.

use std::borrow::Cow;
use core::future::Future;

use lance_graph_contract::exploration::NarsTruth;
use lance_graph_contract::nars::QueryStrategy;
use lance_graph_contract::reasoning::{Reasoner, ReasoningContext, ReasoningKind};
use lance_graph_contract::savants::{savant_by_name, Savant, SAVANTS};

/// Error from a savant reasoner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SavantError {
    /// No roster savant matches the context's kind (+ namespace).
    UnknownSavant,
    /// The reasoner was invoked with a `ReasoningKind` it does not serve.
    KindMismatch,
}

impl core::fmt::Display for SavantError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SavantError::UnknownSavant => f.write_str("no savant matches the reasoning context"),
            SavantError::KindMismatch => f.write_str("reasoning kind not served by this reasoner"),
        }
    }
}

impl std::error::Error for SavantError {}

/// A suggestion-only conclusion from a savant reasoner.
///
/// Plain in-binary value — **never serialized** (the one-binary contract). The
/// concrete row-level pick stays with the data owner (woa-rs); this carries the
/// dispatched strategy + a NARS weight so the consumer can rank/threshold it.
#[derive(Debug, Clone)]
pub struct SavantConclusion {
    /// Roster id of the savant that produced this (SAVANTS.md numbering).
    pub savant_id: u8,
    /// The query strategy the savant dispatches to (from its `InferenceType`).
    pub query_strategy: QueryStrategy,
    /// NARS `(frequency, confidence)` weight of the suggestion.
    pub confidence: NarsTruth,
    /// Human-readable rationale (the AXIS-B decision + evidence summary).
    pub rationale: Cow<'static, str>,
}

/// Namespace → savant-id dispatch contract for the kinds where >1 savant shares
/// the kind AND the namespace is not simply the savant name.
///
/// Seeded with the `Other(RECONCILE_MATCH)` split (PR #419): `ReconcileMatchSelector`
/// (19) and `PaymentToInvoiceMatcher` (21) share `Other(RECONCILE_MATCH)`, so
/// woa-rs passes these namespaces to disambiguate. Every other ambiguous kind
/// resolves by `namespace == savant.name`.
const DISPATCH_NS: &[(&str, u8)] = &[
    ("erp.k3.reconcile_match", 19),
    ("erp.k3.payment_reconcile", 21),
];

/// `ReasoningKind` has no `PartialEq` (it's a pure inheritance vow); compare here.
#[inline]
fn kind_matches(a: ReasoningKind, b: ReasoningKind) -> bool {
    use ReasoningKind::*;
    match (a, b) {
        (CustomerCategory, CustomerCategory)
        | (PostingAnomaly, PostingAnomaly)
        | (NextBestAction, NextBestAction)
        | (InvoiceCompleteness, InvoiceCompleteness)
        | (MailIntent, MailIntent) => true,
        (Other(x), Other(y)) => x == y,
        _ => false,
    }
}

/// Resolve the concrete savant for a `(kind, namespace)`.
///
/// A kind with a single roster savant ignores the namespace; a kind with
/// several resolves via [`DISPATCH_NS`] first, then by `namespace == savant.name`.
pub fn resolve_savant(kind: ReasoningKind, namespace: &str) -> Option<&'static Savant> {
    let candidates: Vec<&'static Savant> =
        SAVANTS.iter().filter(|s| kind_matches(s.kind, kind)).collect();
    match candidates.len() {
        0 => None,
        1 => Some(candidates[0]),
        _ => {
            if let Some(&(_, id)) = DISPATCH_NS.iter().find(|(ns, _)| *ns == namespace) {
                return candidates.iter().copied().find(|s| s.id == id);
            }
            candidates.iter().copied().find(|s| s.name == namespace)
        }
    }
}

/// Build the suggestion-only conclusion: pick the strategy from the savant's
/// inference type and fuse the evidence refs into a NARS `(frequency, confidence)`.
///
/// v1 fusion is coverage-based (the materialized column-level fusion lands when
/// woa-rs feeds real evidence): frequency rises from neutral 0.5 toward 1.0 with
/// evidence coverage vs the budget; confidence is the NARS personality weight
/// `w / (w + 1)` over the evidence-row count. Monotone in evidence by construction.
fn build_conclusion(savant: &Savant, ctx: &ReasoningContext) -> SavantConclusion {
    let strategy = savant.query_strategy();
    let rows: u64 = ctx.evidence.iter().map(|e| e.rows).sum();
    let cap = ctx.budget.max_evidence_rows.max(1) as f32;
    let coverage = (rows as f32 / cap).min(1.0);
    let frequency = 0.5 + 0.5 * coverage;
    let w = rows as f32;
    let confidence = w / (w + 1.0);
    SavantConclusion {
        savant_id: savant.id,
        query_strategy: strategy,
        confidence: NarsTruth::new(frequency, confidence),
        rationale: Cow::Owned(format!(
            "{} [{}|{:?}|{:?}→{:?}]: {} — fused {} evidence row(s) across {} table(s)",
            savant.name,
            savant.lane,
            savant.inference,
            savant.semiring,
            strategy,
            savant.decides,
            rows,
            ctx.evidence.len(),
        )),
    }
}

/// Resolve + conclude for a reasoner that serves a single fixed kind.
fn reason_for_kind(
    self_kind: ReasoningKind,
    ctx: &ReasoningContext,
) -> Result<SavantConclusion, SavantError> {
    if !kind_matches(ctx.kind, self_kind) {
        return Err(SavantError::KindMismatch);
    }
    let savant = resolve_savant(ctx.kind, ctx.namespace).ok_or(SavantError::UnknownSavant)?;
    Ok(build_conclusion(savant, ctx))
}

/// CustomerCategory savants (FiscalPositionResolver, PartnerTrustAdvisor,
/// AnalyticModelScorer, UserCompanyAccessAdvisor) — classify against the family
/// codebook; the namespace selects which.
pub struct CustomerCategoryReasoner;

impl Reasoner for CustomerCategoryReasoner {
    type Conclusion = SavantConclusion;
    type Error = SavantError;

    fn reason<'a>(
        &'a self,
        context: ReasoningContext<'a>,
    ) -> impl Future<Output = Result<Self::Conclusion, Self::Error>> + Send + 'a {
        async move { reason_for_kind(ReasoningKind::CustomerCategory, &context) }
    }
}

/// PostingAnomaly savants (SequenceGapAnomalyDetector, AutopostRecommender,
/// LockDateAdvancer).
pub struct PostingAnomalyReasoner;

impl Reasoner for PostingAnomalyReasoner {
    type Conclusion = SavantConclusion;
    type Error = SavantError;

    fn reason<'a>(
        &'a self,
        context: ReasoningContext<'a>,
    ) -> impl Future<Output = Result<Self::Conclusion, Self::Error>> + Send + 'a {
        async move { reason_for_kind(ReasoningKind::PostingAnomaly, &context) }
    }
}

/// NextBestAction savants (12 — analytic/currency/tax/pricing/procurement/stock).
pub struct NextBestActionReasoner;

impl Reasoner for NextBestActionReasoner {
    type Conclusion = SavantConclusion;
    type Error = SavantError;

    fn reason<'a>(
        &'a self,
        context: ReasoningContext<'a>,
    ) -> impl Future<Output = Result<Self::Conclusion, Self::Error>> + Send + 'a {
        async move { reason_for_kind(ReasoningKind::NextBestAction, &context) }
    }
}

/// `Other(code)` savants (PricelistAssignment, Chart/Rate policy, the two
/// reconcile matchers, bank-statement match). Dispatches on the `Other(code)`
/// carried by `ctx.kind`; the RECONCILE_MATCH pair splits by namespace.
pub struct OtherReasoner;

impl Reasoner for OtherReasoner {
    type Conclusion = SavantConclusion;
    type Error = SavantError;

    fn reason<'a>(
        &'a self,
        context: ReasoningContext<'a>,
    ) -> impl Future<Output = Result<Self::Conclusion, Self::Error>> + Send + 'a {
        async move {
            match context.kind {
                ReasoningKind::Other(_) => {
                    let savant = resolve_savant(context.kind, context.namespace)
                        .ok_or(SavantError::UnknownSavant)?;
                    Ok(build_conclusion(savant, &context))
                }
                _ => Err(SavantError::KindMismatch),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::future::Future;
    use core::pin::pin;
    use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    use lance_graph_contract::nars::QueryStrategy;
    use lance_graph_contract::reasoning::{Budget, EvidenceRef};
    use lance_graph_contract::savants::other_kind;

    // Minimal executor for the immediately-ready reasoner futures (no async
    // runtime dep). SAFETY: the vtable fns are no-ops over a null data pointer
    // that is never dereferenced.
    static NOOP_VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| RawWaker::new(core::ptr::null(), &NOOP_VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );
    fn block_on<F: Future>(fut: F) -> F::Output {
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &NOOP_VTABLE)) };
        let mut cx = Context::from_waker(&waker);
        let mut fut = pin!(fut);
        loop {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                return v;
            }
        }
    }

    fn budget() -> Budget {
        Budget { max_tokens: 1000, max_ms: 100, max_evidence_rows: 100 }
    }
    fn ev(table: &'static str, rows: u64) -> EvidenceRef<'static> {
        EvidenceRef { table, schema_fingerprint: 0, rows }
    }
    fn ctx<'a>(kind: ReasoningKind, ns: &'a str, evidence: &'a [EvidenceRef<'a>]) -> ReasoningContext<'a> {
        ReasoningContext { namespace: ns, kind, evidence, budget: budget() }
    }

    #[test]
    fn resolves_ambiguous_kind_by_savant_name() {
        // PostingAnomaly has 3 savants → namespace=name disambiguates.
        let s = resolve_savant(ReasoningKind::PostingAnomaly, "SequenceGapAnomalyDetector").unwrap();
        assert_eq!(s.id, 6);
        let s2 = resolve_savant(ReasoningKind::PostingAnomaly, "LockDateAdvancer").unwrap();
        assert_eq!(s2.id, 18);
    }

    #[test]
    fn other_reconcile_match_splits_by_namespace() {
        let a = resolve_savant(ReasoningKind::Other(other_kind::RECONCILE_MATCH), "erp.k3.reconcile_match").unwrap();
        let b = resolve_savant(ReasoningKind::Other(other_kind::RECONCILE_MATCH), "erp.k3.payment_reconcile").unwrap();
        assert_eq!(a.id, 19, "ReconcileMatchSelector");
        assert_eq!(b.id, 21, "PaymentToInvoiceMatcher");
    }

    #[test]
    fn other_single_candidate_ignores_namespace() {
        // PRICELIST_ASSIGNMENT (code 1) has one savant — namespace irrelevant.
        let s = resolve_savant(ReasoningKind::Other(other_kind::PRICELIST_ASSIGNMENT), "whatever").unwrap();
        assert_eq!(s.id, 3, "PricelistAssignmentAgent");
    }

    #[test]
    fn conclusion_strategy_follows_inference_type() {
        let fiscal = savant_by_name("FiscalPositionResolver").unwrap();
        let c = build_conclusion(fiscal, &ctx(ReasoningKind::CustomerCategory, "FiscalPositionResolver", &[ev("account_fiscal_position", 3)]));
        assert_eq!(c.savant_id, 1);
        // Deduction → CamExact.
        assert_eq!(c.query_strategy, QueryStrategy::CamExact);
    }

    #[test]
    fn confidence_is_monotone_in_evidence() {
        let s = savant_by_name("AutopostRecommender").unwrap();
        let low = build_conclusion(s, &ctx(ReasoningKind::PostingAnomaly, "AutopostRecommender", &[ev("account_move", 1)]));
        let hi = build_conclusion(s, &ctx(ReasoningKind::PostingAnomaly, "AutopostRecommender", &[ev("account_move", 50)]));
        assert!(hi.confidence.frequency >= low.confidence.frequency);
        assert!(hi.confidence.confidence > low.confidence.confidence);
        assert!(hi.confidence.confidence <= 0.99, "NarsTruth caps confidence");
    }

    #[test]
    fn reasoner_trait_dispatches_through_async() {
        let evidence = [ev("account_fiscal_position", 5)];
        let out = block_on(CustomerCategoryReasoner.reason(ctx(
            ReasoningKind::CustomerCategory,
            "FiscalPositionResolver",
            &evidence,
        )))
        .unwrap();
        assert_eq!(out.savant_id, 1);
        assert_eq!(out.query_strategy, QueryStrategy::CamExact);
    }

    #[test]
    fn other_reasoner_rejects_non_other_kind() {
        let err = block_on(OtherReasoner.reason(ctx(ReasoningKind::CustomerCategory, "x", &[]))).unwrap_err();
        assert_eq!(err, SavantError::KindMismatch);
    }

    #[test]
    fn wrong_reasoner_for_kind_is_mismatch() {
        let err = block_on(PostingAnomalyReasoner.reason(ctx(ReasoningKind::NextBestAction, "x", &[]))).unwrap_err();
        assert_eq!(err, SavantError::KindMismatch);
    }
}
