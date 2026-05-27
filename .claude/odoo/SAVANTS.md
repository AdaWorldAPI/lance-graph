# SAVANTS — odoo-richness reasoner roster (woa-rs ⟶ lance-graph delegation)

> Synthesis of the AXIS-B / HYBRID rules harvested across lanes L1–L15
> (`.claude/odoo/L*.md`). Each Savant is a **delegated reasoner**: the
> deterministic guard stays in woa-rs (AXIS-A Rust), the evidence-weighted /
> ambiguous core is delegated to lance-graph's thinking surface through the
> **BBB-allowed** contract crates (`lance-graph-contract`,
> `lance-graph-ontology`, `lance-graph-callcenter`). No brain-crate ever
> enters the customer binary (Iron Rule 1).
>
> **Per-agent documents** live under `.claude/odoo/savants/<name>.md`
> (schema at the end of this file). This file is the index + framework.

## The three coordinates (pinned to the real contract enums)

1. **Ontology** — `resolve_odoo_to_family(odoo_class, &OgitFamilyTable)`
   chains `odoo:<class>` → OWL pivot → `FamilyEntry` (8-bit family + 16-bit
   slot). The family carries the **default ThinkingStyle cluster**.
2. **Use case** — `lance_graph_contract::reasoning::ReasoningKind`
   { CustomerCategory, PostingAnomaly, NextBestAction, InvoiceCompleteness,
   MailIntent, Other(u32) }.
3. **Thinking** —
   `nars::InferenceType` { Deduction, Induction, Abduction, Revision, Synthesis }
   × `nars::SemiringChoice` { Boolean, HammingMin, NarsTruth, XorBundle, CamPqAdc }
   × `thinking::StyleCluster` { Analytical, Creative, Empathic, Direct,
   Exploratory, Meta } — inherited from the family.

`InferenceType::default_strategy()` already maps to a `QueryStrategy`
(Deduction→CamExact, Induction→CamWide, Abduction→DnTreeFull,
Revision→BundleInto, Synthesis→BundleAcross) — so a Savant's tuple fully
determines its runtime dispatch.

## OGIT family map (existing + proposed) and inherited style

| Family | Basin | Seeded odoo classes | Default style cluster | Rationale |
|---|---|---|---|---|
| `0x60` WorkOrderCore | work order | (woa core) | Direct | task execution |
| `0x61` BillingCore | product/billing catalogue | product.product→schema:Product | Analytical | pricing math |
| `0x62` SMBAccounting | ledger / CoA | account.account→fibo:Account; account.move.line | Analytical | ledger reasoning |
| `0x80` SmbFoundryCustomer | partner / legal entity | res.partner.Company→fibo:LegalEntity | Empathic (fiscal sub-paths Analytical) | relationship + trust |
| `0x81` SmbFoundryInvoice | document / transaction | account.move→fibo:Transaction | Direct | transaction processing |
| **`0x63` ProductCatalog** *(PROPOSED)* | product catalogue + pricelist | product.template, product.pricelist, uom.uom | Analytical | catalogue/pricing; L8 `PricelistAssignmentAgent` needs a home (currently None) |
| **`0x90` HRFoundation** *(PROPOSED)* | employee / org | hr.employee→vcard:Individual, hr.department→org:OrganizationalUnit | Empathic | people/org; all hr.* resolve None today |

**Unmapped (`None`) classes needing a Layer-2 alignment axiom**:
`stock.*` (stock.rule, stock.warehouse.orderpoint, stock.move),
`account.analytic.distribution.model`, `account.account.tag`. Savants on
these carry `family=None` until an alignment axiom lands (lance-graph side).

## Roster — gap lanes L8–L15 (16 Savants)

| # | Savant | Family | ReasoningKind | Inference | Semiring | Style | Lane | Decides |
|---|---|---|---|---|---|---|---|---|
| 1 | FiscalPositionResolver | 0x80 | CustomerCategory | Deduction | NarsTruth | Analytical | L9 | which fiscal position (tax mapping) applies to a partner |
| 2 | PartnerTrustAdvisor | 0x80 | CustomerCategory | Revision | NarsTruth | Empathic | L9 | infer partner trust/dunning-risk from payment history |
| 3 | PricelistAssignmentAgent | 0x63* | Other(PricelistAssignment) | Revision | NarsTruth | Analytical | L8 | partner pricelist when no explicit property (country-group/config fallback) |
| 4 | AnalyticDistributionSuggester | 0x62 | NextBestAction | Induction | NarsTruth | Analytical | L10 | suggested cost-centre distribution for a move line |
| 5 | AnalyticModelScorer | None | CustomerCategory | Deduction | HammingMin | Analytical | L10 | which analytic.distribution.model matches (priority-scored) |
| 6 | SequenceGapAnomalyDetector | 0x62 | PostingAnomaly | Abduction | NarsTruth | Analytical | L11 | gaps in journal sequences ⇒ deleted posted entries (GoBD) |
| 7 | ExchangeAccountSelector | 0x62 | Other(ChartAccountMapping) | Deduction | Boolean | Analytical | L12 | gain/loss account for FX diff (sign-driven; config-assist) |
| 8 | ReportRateTypeSelector | 0x62 | Other(ConsolidationRatePolicy) | Deduction | Boolean | Analytical | L12 | current/historical/average rate per report line (IFRS vs HGB) |
| 9 | CurrencySelectionAdvisor | 0x62 | NextBestAction | Induction | NarsTruth | Analytical | L12 | which currencies to enable (geography signal) |
| 10 | UserCompanyAccessAdvisor | 0x80 | CustomerCategory | Induction | NarsTruth | Empathic | L12 | branch-access subset by user role/context |
| 11 | ProcurementRuleSelector | None | NextBestAction | Induction | NarsTruth | Analytical | L13 | route among equal-sequence rules (lead/availability/reliability) |
| 12 | ReorderTimingAdvisor | None | NextBestAction | Induction | NarsTruth | Analytical | L13 | reorder timing under demand/supplier uncertainty |
| 13 | ReplenishmentReportAdvisor | None | NextBestAction | Induction | NarsTruth | Analytical | L13 | real shortfall vs demand noise in the replenishment report |
| 14 | RouteTiebreaker | None | NextBestAction | Abduction | NarsTruth | Analytical | L13 | equal-sequence route tiebreak (supplier lead/cost/capacity) |
| 15 | TaxExigibilitySuggestor | 0x62 | NextBestAction | Induction | NarsTruth | Analytical | L15 | suggest tax exigibility (on-invoice vs on-payment / cash-basis) |
\* `0x63 ProductCatalog` is the proposed family; until ratified, treat as `None`.

## Roster — original lanes L1–L7 (9 Savants; L3/L4 are fully deterministic)

| # | Savant | Family | ReasoningKind | Inference | Semiring | Style | Lane | Decides |
|---|---|---|---|---|---|---|---|---|
| 17 | AutopostRecommender | 0x81 | PostingAnomaly | Induction | NarsTruth | Analytical | L1 | recommend auto-posting bills after 3+ unmodified from a partner |
| 18 | LockDateAdvancer | 0x81 | PostingAnomaly | Abduction | NarsTruth | Analytical | L1 | which next open period to advance a move into when date is locked |
| 19 | ReconcileMatchSelector | None | Other(ReconcileMatch) | Induction | NarsTruth | Analytical | L2 | which open items to propose as reconciliation candidates (core) |
| 20 | BankStatementMatcher | None | Other(BankStatementMatch) | Induction | NarsTruth | Analytical | L5 | which reconcile-model rule matches a bank line + write-offs |
| 21 | PaymentToInvoiceMatcher | None | Other(ReconcileMatch) | Induction | NarsTruth | Analytical | L5 | whether a payment fully reconciles open invoices (Mahnwesen gate) |
| 22 | UpsellActivityTrigger | 0x81 | NextBestAction | Induction | NarsTruth | Exploratory | L6 | qty_delivered>ordered ⇒ upsell TODO for salesperson |
| 23 | PricelistRecommender | 0x81 | NextBestAction | Synthesis | NarsTruth | Exploratory | L6 | which pricelist rule when multiple candidates apply |
| 24 | RemovalStrategySelector | None | NextBestAction | Induction | XorBundle | Exploratory | L7 | which quants to bind to a reservation (FIFO/FEFO/LIFO/closest) |
| 25 | MoveAssignmentPrioritizer | None | NextBestAction | Induction | NarsTruth | Exploratory | L7 | which confirmed moves to satisfy first (priority/deadline/quants) |
| 26 | BackorderJudge | None | NextBestAction | Abduction | NarsTruth | Exploratory | L7 | partial fulfilment ⇒ backorder vs cancel remainder |

**Roster total: 25 Savants** (16 from L8–L15, 9 from L1–L7). L3 (tax compute) and L4 (reports/DATEV) are fully AXIS-A — no Savants. The `Exploratory` cluster clusters on stock/sale next-best-action (L6/L7); `Analytical` dominates accounting (0x62/0x81); `Empathic` on partner/HR (0x80/0x90).

## woa-rs ↔ lance-graph split (the delegation contract)

Per Savant, woa-rs owns the **deterministic guard** (AXIS-A) and calls the
reasoner only for the **ambiguous core** (AXIS-B):

```
woa-rs handler (AXIS-A guard: balance==0, residual, sign, prefix-match, …)
    └─ if ambiguous → lance_graph_contract::reasoning::Reasoner::reason(
            ReasoningContext { namespace: tenant, kind: <ReasoningKind>,
                               evidence: &[EvidenceRef…], budget })
         → conclusion (truth-weighted) → woa-rs applies it as a *suggestion*,
           never an un-guarded write (Iron Rule 7 verhaltens-bewahrend).
```

Evidence batches are Arrow `EvidenceRef { table, schema_fingerprint, rows }`.
The Savant's `SemiringChoice` selects how evidence fuses (NarsTruth = NARS
evidence fusion, the common case); `InferenceType` selects the query strategy.

## AXIS-A remainder (the deterministic ports)

The large majority of harvested rules are AXIS-A — deterministic Rust ports,
**not** Savants. They land in woa-rs ERP modules (K3/K7/K8/K11/K15/skr_data)
per each lane draft's "woa-rs target" + Rust sketch. The per-lane drafts
(`L*.md`) are the porter's spec; this roster covers only the delegated set.

## lance-graph handover boundary (FLAG for Jan)

The **woa-rs side** defines: the Savant roster, each delegation tuple, the
`ReasoningContext` call sites (guards), and the evidence schemas. That is all
contract-level (BBB-allowed) and lives here.

The **lance-graph side** must implement: (a) the actual `Reasoner` impls /
experts for each `ReasoningKind`, (b) the two new OGIT families
(`0x63 ProductCatalog`, `0x90 HRFoundation`) + the Layer-2 alignment axioms
for the `None` classes (stock.*, analytic.distribution.model,
account.account.tag), (c) wiring the `ThinkingStyle` clusters to those
families in `OgitFamilyTable`.

→ **When we start (b)/(c), it needs an integration plan on the lance-graph
side** (`lance-graph/.claude/board/INTEGRATION_PLANS.md` PREPEND +
`lance-graph/.claude/plans/<name>-v1.md`, per lance-graph CLAUDE.md
board-hygiene). **I will stop and notify you at that boundary** rather than
cross-commit into lance-graph from this woa-rs session.

## Per-agent document schema (`.claude/odoo/savants/<name>.md`)

```markdown
# Savant: <Name>
- **Family / ontology**: <0x..|None> (<basin>); odoo class <x> → OWL <pivot>
- **ReasoningKind**: <kind>   **Inference**: <type>   **Semiring**: <choice>
  **Style cluster**: <cluster> (inherited from family)
- **Source lane / odoo**: L<n> — <file:lines>

## What it decides (the AXIS-B core)
<the ambiguous/evidence-weighted decision, 1 paragraph>

## Deterministic guard (AXIS-A, stays in woa-rs)
<the closed-form pre/post-conditions that wrap the delegation>

## Evidence (Arrow EvidenceRef)
- table(s), key columns, what signal each carries

## Delegation call site
<which woa-rs handler calls Reasoner::reason, with what ReasoningContext>

## Parity / GoBD notes
<German-tax/GoBD specifics; suggestion-only, never un-guarded write>

## lance-graph dependency
<reasoner impl needed; family/axiom needed; integration-plan trigger Y/N>
```
