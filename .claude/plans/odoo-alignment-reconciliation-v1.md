# odoo-alignment-reconciliation-v1

**Status:** PROPOSAL — awaits Jan ratification. NO history rewrite, NO force-push
until ratified (branch `claude/odoo-owl-cache` left as-is; local-only).
**Confidence:** HIGH on "keep main's API as canonical"; HIGH on the seed-coverage
+ StyleCluster gaps; MED on the new-family byte assignments (0x63/0x64/0x90)
pending `data/family_registry.ttl` authority.
**Author:** woa-rs odoo-richness session (2026-05-26), handed over at the
woa-rs ⟶ lance-graph boundary defined in `woa-rs/.claude/odoo/SAVANTS.md`.

## 1. Context — two implementations of `odoo_alignment`

The odoo→OWL→OGIT-family cache exists **twice**:

- **`origin/main` `ecb92c0`** (`crates/lance-graph-callcenter/src/odoo_alignment.rs`,
  420 lines) — merged, canonical. `resolve_odoo(class) -> Option<OwlPivot>`,
  `resolve_odoo_to_family(class,&table) -> Option<(OgitFamily,u16)>` (validates
  against the **live** hydrated table), `resolve_odoo_entry -> &FamilyEntry`,
  `seed_family_table(&mut)`, `dolce_odoo` (suffix rules, works for unmapped
  classes), `product.*` prefix fallback, family bytes bound to
  `data/family_registry.ttl` (97/98/128/129). Doc explicitly forbids mirroring.
- **branch `claude/odoo-owl-cache` `656d335`** (522 lines) — an independent
  variant: `OwlIdentity` (not `OwlPivot`), `resolve_odoo_to_family -> &FamilyEntry`,
  `resolve_odoo_alignment`, `ODOO_ALIGNMENTS` + `SKR_ACCOUNT_CONCEPTS` slices,
  `FAM_*` consts.

A rebase of `656d335` onto main conflicts (add/add on `odoo_alignment.rs`,
content on `lib.rs`) — the two are not auto-mergeable.

### Decision (proposed)
**Keep main's `ecb92c0` API as the single source. Drop `656d335`** (rebase
`--skip`, branch → `origin/main`). Rationale: main is merged + canonical,
validates against the live family table, carries provenance/DOLCE, and
explicitly is the anti-mirror single source. Nothing unique is lost — main
already covers the worked classes **and** SKR (slot 3 `SkrAccount`). The
woa-rs `skr_data` consumer + `SAVANTS.md`/`BRIEFING.md` references must align
to main's API (`OwlPivot`, `(OgitFamily,u16)` / `resolve_odoo_entry`), NOT the
branch's `&FamilyEntry`-returning `resolve_odoo_to_family`.

→ **Blocking on Jan:** authorize `rebase --skip` of `656d335` + the (force-)push,
or keep the branch frozen. Until then this plan is documentation only.

## 2. Gaps the L1–L15 odoo harvest exposed in main's version

### Gap 1 — `ODOO_SEED` is too sparse (7 rows)
The harvest touched ~40 classes; only 7 are seeded, so ~half the Savants'
subjects resolve `None`. Add the classes that map cleanly to an **existing**
family (each needs a stable slot + label_uri + provenance + DOLCE):

| odoo class | pivot | family | DOLCE | note |
|---|---|---|---|---|
| account.journal | fibo:Journal | 0x62 | Endurant | L11 |
| account.group | fibo:AccountingGroup | 0x62 | Endurant | L11 |
| res.currency | fibo:Currency | 0x62 | Quality | L12 |
| res.currency.rate | fibo:ExchangeRate | 0x62 | Quality | L12 |
| res.company | fibo:LegalEntity | 0x80 | Endurant | L12 |
| product.pricelist | schema:PriceSpecification | 0x61 (→0x63) | Abstract | L8 |
| product.pricelist.item | schema:PriceSpecification | 0x61 (→0x63) | Abstract | L8 |
| uom.uom | qudt:Unit | 0x61 (→0x63) | Abstract | L8 — NOT caught by `product.*` prefix |
| product.category | schema:Thing | 0x61 (→0x63) | Endurant | L8 |

### Gap 2 — the "family carries the thinking style" claim is unfulfilled
`FamilyEntry` (`family_table.rs`) has `label_uri, kind, owl_characteristics,
dolce_marker, axiom_blob, provenance, verbs` — **no `StyleCluster`**. The whole
Savant design inherits its `ThinkingStyle` cluster from the family, but that
field does not exist. **Add `default_style: StyleCluster`** to `FamilyEntry`
(or a `family → StyleCluster` map in `family_registry.ttl`), so
`resolve_odoo_entry` yields the inherited style instead of it being
hand-assigned in `SAVANTS.md`. Proposed defaults: 0x61 BillingCore→Analytical,
0x62 SMBAccounting→Analytical, 0x80 SmbFoundryCustomer→Empathic,
0x81 SmbFoundryInvoice→Direct.

### Gap 3 — the Layer-2-axiom backlog has no home
Every Savant on these classes silently resolves `None`. They need new families
or an axiom table, not silent `None`:

- **`0x63 ProductCatalog`** (PROPOSED) — product.template/pricelist/uom/category
  (currently borrow 0x61 BillingCore or fall through). Style: Analytical.
- **`0x90 HRFoundation`** (PROPOSED) — hr.employee→vcard:Individual,
  hr.department→org:OrganizationalUnit, hr.job→org:Role. Style: Empathic.
- **`0x64 AccountingPolicy`** (PROPOSED) — the Abstract rule/norm classes:
  account.tax, account.tax.repartition.line, account.tax.group,
  account.fiscal.position, account.payment.term, account.reconcile.model,
  account.analytic.distribution.model. Style: Analytical.
- Still legitimately `None` (need bespoke axioms): stock.* (stock.rule,
  stock.warehouse.orderpoint, stock.move, stock.lot), sale.order(.line),
  account.analytic.account/.line.

Each new family byte must be added to `data/family_registry.ttl` first
(`family_bytes_match_registry_ttl` test enforces the binding).

### Gap 4 — SKR granularity (keep as-is)
Main collapses SKR to one `SkrAccount` slot (slot 3, account.account.template).
That is correct under Option B (no per-account slot minting). **Do NOT** port
the branch's `SKR_ACCOUNT_CONCEPTS` per-concept rows — they would violate
Option B. Confirm woa-rs `skr_data` only needs the single inherited slot.

## 3. Savant reasoner backlog (the woa-rs ⟶ lance-graph delegation)

`woa-rs/.claude/odoo/SAVANTS.md` defines **25 Savants** — delegated reasoners
keyed by `ReasoningKind`. woa-rs owns the deterministic guard (AXIS-A Rust) and
calls `lance_graph_contract::reasoning::Reasoner::reason(ReasoningContext{kind,
namespace, evidence, budget})` for the ambiguous core, fusing evidence per
`SemiringChoice` and dispatching per `InferenceType`. lance-graph must provide
the `Reasoner` impls, grouped by `ReasoningKind`:

- **CustomerCategory** — FiscalPositionResolver, PartnerTrustAdvisor,
  UserCompanyAccessAdvisor, AnalyticModelScorer.
- **PostingAnomaly** — SequenceGapAnomalyDetector, AutopostRecommender,
  LockDateAdvancer.
- **NextBestAction** — Analytic/Procurement/Reorder/Replenishment/Route/Tax
  + Upsell/Pricelist/Removal/MoveAssignment/Backorder advisors (the bulk).
- **Other(...)** — ReconcileMatchSelector, BankStatementMatcher,
  PaymentToInvoiceMatcher (reconciliation matching — the highest-value cluster),
  ExchangeAccountSelector, ReportRateTypeSelector, PricelistAssignmentAgent.

Reconciliation matching (`ReconcileMatchSelector`, L2 + L5) is the flagship
AXIS-B reasoner: open-item ↔ payment candidate proposal under NARS-truth
evidence fusion — the canonical NextBestAction/Induction case.

Only the **contract crates** (`lance-graph-contract`, `-ontology`,
`-callcenter`) are BBB-allowed into the woa-rs customer binary (woa-rs Iron
Rule 1). The Reasoner impls live behind that contract.

## 4. Phases

- **P0 (this plan):** ratify §1 decision (drop 656d335). [blocked on Jan]
- **P1:** expand `ODOO_SEED` (Gap 1) + tests. Additive, no API change.
- **P2:** add `StyleCluster` to `FamilyEntry` + `family_registry.ttl` styles
  (Gap 2). Touches family_table — coordinate with cognitive-stack consumers.
- **P3:** mint `0x63/0x64/0x90` in `family_registry.ttl` + seed rows (Gap 3).
- **P4:** `Reasoner` impls per `ReasoningKind` (§3), reconciliation first.

## 5. Acceptance
- `resolve_odoo` covers every class named in woa-rs L1–L15 with an explicit
  family OR a documented `None`+axiom-backlog entry.
- `resolve_odoo_entry(...).default_style` returns the cluster `SAVANTS.md` lists.
- `cargo test -p lance-graph-callcenter` green incl. `family_bytes_match_registry_ttl`.

## 6. Open questions
1. Byte assignments for 0x63/0x64/0x90 — confirm free in `family_registry.ttl`.
2. Should the Abstract rule-classes (tax/fiscal/reconcile) be one family
   (0x64 AccountingPolicy) or split? They share Style=Analytical but differ in
   ReasoningKind.
3. Does adding `default_style` to `FamilyEntry` ripple into existing
   `cognitive_stack`/`persona` consumers of `FamilyEntry`?
4. woa-rs `skr_data` mirror: delete it and depend on `lance-graph-callcenter`
   instead (per the anti-mirror doc), or keep a thin re-export?
