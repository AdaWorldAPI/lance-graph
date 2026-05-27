# SCAFFOLD — Odoo savant AXIS-B evidence contract (carve-out request)

> **Handover:** lance-graph session (`splat3d-cpu-simd-renderer-MAOO0`) → **woa-rs session** (the Odoo extractor, owner of the roster + evidence schemas per `SAVANTS.md` §"lance-graph handover boundary").
> **Why this exists:** the `Reasoner` impls (D-ODOO-2 / D-ODOO-SAV-4) are blocked on the **AXIS-B input contract**, which the current `SAVANTS.md` + `L*.md` specify only on the AXIS-A (woa-rs guard) side. Rather than ping-pong (which loses context across sessions), **carve out the four slots below per savant** and I implement the Reasoners in one pass.
> **Date:** 2026-05-27.

## What lance-graph already has (don't re-send)

- **`contract::savants`** — the 25-savant roster as data: each `Savant { id, name, family, kind, inference, semiring, style, lane, decides }`. The dispatch tuple is **fixed**; `query_strategy()` rides `InferenceType::default_strategy()`.
- **`reasoning::{Reasoner, ReasoningContext, ReasoningKind, EvidenceRef, Budget}`** — the delegation surface (shipped).
- **#414**: OGIT families `0x63 ProductCatalog` / `0x90 HRFoundation` + Layer-2 alignment axioms (stock.* / analytic.distribution.model / account.account.tag) + StyleCluster wiring.
- FIBU subtree now inherits `fibofnd` (zugferd/fibobe/skr0x).

## What I need carved out — 4 slots per savant

For each savant create `.claude/odoo/savants/<Name>.md` (schema = `SAVANTS.md` §"Per-agent document schema") and fill **these four AXIS-B slots** (the rest of the tuple is in `contract::savants`):

1. **Evidence (Arrow `EvidenceRef`)** — the concrete table(s) + **key columns** + dtype, and what signal each column carries. This is the typed input the Reasoner consumes. *(Without this the impl has no input.)*
2. **Odoo field → signal map** — which exact odoo model fields back each column (e.g. `res.partner.{payment_history, credit_limit}` for `PartnerTrustAdvisor`), with the L-doc `file:lines`.
3. **Property-level alignment** — the OWL property IRIs (not just class) the reasoner traverses (e.g. `odoo:amount_residual ≡ fibo:hasResidualAmount`), if the decision crosses the FIBO/SKR/ZUGFeRD seam.
4. **AXIS-B decision in evidence terms** — restate `decides` as: given evidence E, produce conclusion C with a NARS `(frequency, confidence)`; name the discriminating features. *(The AXIS-A guard is already in the L-docs; I only need the ambiguous core.)*

## The target the carve-out feeds (so you know the shape)

```rust
// lance-graph side will implement, one per ReasoningKind (or savant-config registry — TBD review):
impl Reasoner for <Kind>Reasoner {
    fn reason(&self, ctx: &ReasoningContext) -> SavantConclusion {
        // ctx.evidence: &[EvidenceRef]  ← slot 1 (your schema)
        // dispatch by ctx.kind + family style; fuse via savant.semiring (NarsTruth common)
        // → conclusion + (frequency, confidence)   ← slot 4 shape
    }
}
```

## The 25 savants to carve out (priority-ordered; fill in lane order)

**Tier 1 — substrate most ready, do first (accounting / 0x62 / 0x81):**

| id | savant | family | kind | infer | lane | decides (AXIS-B core) |
|---|---|---|---|---|---|---|
| 6 | SequenceGapAnomalyDetector | 0x62 | PostingAnomaly | Abduction | L11 | journal sequence gaps ⇒ deleted posted entries (GoBD) |
| 17 | AutopostRecommender | 0x81 | PostingAnomaly | Induction | L1 | auto-post bills after 3+ unmodified from a partner |
| 18 | LockDateAdvancer | 0x81 | PostingAnomaly | Abduction | L1 | next open period to advance a locked move into |
| 4 | AnalyticDistributionSuggester | 0x62 | NextBestAction | Induction | L10 | suggested cost-centre distribution for a move line |
| 15 | TaxExigibilitySuggestor | 0x62 | NextBestAction | Induction | L15 | tax exigibility (on-invoice vs on-payment / cash-basis) |

**Tier 2 — partner / pricing (0x80 / 0x81 / 0x63):**

| id | savant | family | kind | infer | lane | decides |
|---|---|---|---|---|---|---|
| 1 | FiscalPositionResolver | 0x80 | CustomerCategory | Deduction | L9 | which fiscal position (tax mapping) for a partner |
| 2 | PartnerTrustAdvisor | 0x80 | CustomerCategory | Revision | L9 | partner trust / dunning-risk from payment history |
| 3 | PricelistAssignmentAgent | 0x63 | Other(PRICELIST_ASSIGNMENT) | Revision | L8 | partner pricelist when no explicit property |
| 23 | PricelistRecommender | 0x81 | NextBestAction | Synthesis | L6 | which pricelist rule when multiple candidates apply |
| 22 | UpsellActivityTrigger | 0x81 | NextBestAction | Induction | L6 | qty_delivered>ordered ⇒ upsell TODO |
| 10 | UserCompanyAccessAdvisor | 0x80 | CustomerCategory | Induction | L12 | branch-access subset by user role/context |

**Tier 3 — reconcile / FX / currency (None / 0x62):**

| id | savant | family | kind | infer | lane | decides |
|---|---|---|---|---|---|---|
| 19 | ReconcileMatchSelector | None | Other(RECONCILE_MATCH) | Induction | L2 | open items to propose as reconciliation candidates |
| 20 | BankStatementMatcher | None | Other(BANK_STATEMENT_MATCH) | Induction | L5 | reconcile-model rule for a bank line + write-offs |
| 21 | PaymentToInvoiceMatcher | None | Other(RECONCILE_MATCH) | Induction | L5 | whether a payment fully reconciles open invoices |
| 5 | AnalyticModelScorer | None | CustomerCategory | Deduction | L10 | which analytic.distribution.model matches (priority) |
| 7 | ExchangeAccountSelector | 0x62 | Other(CHART_ACCOUNT_MAPPING) | Deduction | L12 | gain/loss account for FX diff |
| 8 | ReportRateTypeSelector | 0x62 | Other(CONSOLIDATION_RATE_POLICY) | Deduction | L12 | current/historical/average rate per report line |
| 9 | CurrencySelectionAdvisor | 0x62 | NextBestAction | Induction | L12 | which currencies to enable (geography signal) |

**Tier 4 — stock / procurement (None, needs #414 axioms confirmed):**

| id | savant | family | kind | infer | lane | decides |
|---|---|---|---|---|---|---|
| 11 | ProcurementRuleSelector | None | NextBestAction | Induction | L13 | route among equal-sequence rules |
| 12 | ReorderTimingAdvisor | None | NextBestAction | Induction | L13 | reorder timing under demand/supplier uncertainty |
| 13 | ReplenishmentReportAdvisor | None | NextBestAction | Induction | L13 | real shortfall vs demand noise |
| 14 | RouteTiebreaker | None | NextBestAction | Abduction | L13 | equal-sequence route tiebreak |
| 24 | RemovalStrategySelector | None | NextBestAction | Induction(XorBundle) | L7 | quants to bind to a reservation (FIFO/FEFO/LIFO) |
| 25 | MoveAssignmentPrioritizer | None | NextBestAction | Induction | L7 | which confirmed moves to satisfy first |
| 26 | BackorderJudge | None | NextBestAction | Abduction | L7 | partial fulfilment ⇒ backorder vs cancel |

## Open question for woa-rs to pin (drives the impl shape)

**Reasoner dispatch shape:** one `Reasoner` impl per `ReasoningKind` (5–6 impls, dispatch on family+evidence inside), **or** a savant-config registry (data-driven, one generic engine reading the savant tuple)? #414 flagged this as gating D-ODOO-SAV-4. Your call — it determines whether the carve-out feeds N impls or one config table.

## Hand-back

Fill the per-savant docs (or a single table with the 4 slots × 25) and drop a note in `lance-graph/.claude/board/AGENT_LOG.md`. I'll then implement the Reasoners against the filled evidence contract in one pass — no re-derivation.
