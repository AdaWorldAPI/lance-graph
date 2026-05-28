# D-ODOO-EXT-6 — Stage 1 coverage report

**Date:** 2026-05-28
**Plan:** `.claude/plans/odoo-source-extraction-v1.md`
**EXT-5 pairing table:** `extracted/pairing.rs` (`CURATED_EXTRACTED_PAIRS`)
**Gate test:** `extracted/coverage.rs`

**Summary:** 48/53 curated entities have TIER-1 extracted backing (90.6%);
5 explicit TIER-2 deferrals across 2 addons; gate PASS.

---

## Per-lane coverage

Eligible = Curated − TIER-2 exemptions.  Backed = eligible entities present in
`CURATED_EXTRACTED_PAIRS`.  Coverage = Backed / Eligible (100 % where Eligible > 0).

| Lane | Curated | Eligible | Backed | Coverage | Notes |
|------|--------:|---------:|-------:|---------:|-------|
| L1   |       3 |        3 |      3 |     100% |       |
| L2   |       3 |        3 |      3 |     100% |       |
| L3   |       6 |        6 |      6 |     100% |       |
| L4   |       6 |        6 |      6 |     100% |       |
| L5   |       5 |        5 |      5 |     100% |       |
| L6   |       4 |        4 |      4 |     100% |       |
| L7   |       6 |        6 |      6 |     100% |       |
| L8   |       6 |        6 |      6 |     100% |       |
| L9   |       6 |        6 |      6 |     100% |       |
| L10  |       5 |        5 |      5 |     100% |       |
| L11  |       4 |        4 |      4 |     100% |       |
| L12  |       5 |        5 |      5 |     100% |       |
| L13  |       5 |        4 |      4 |     100% | `stock.valuation.layer` deferred (`stock_account`, TIER-2) |
| L14  |       4 |        0 |    n/a |      n/a | All 4 entities deferred to Stage 2 (`hr` addon, TIER-2) |
| L15  |       2 |        2 |      2 |     100% |       |

**Gate result:** every lane with eligible entities is at 100 % (floor is 80 %).
L14 is wholly exempt — skip, not a failure.

---

## TIER-2 deferral catalogue

The 5 entities below are outside the 12 TIER-1 addons and are explicitly
deferred to Stage 2.  They are tracked in `coverage::COVERAGE_EXEMPTIONS`
so the gate never treats them as regressions.

| Curated `model_name`    | Lane | Addon (Stage)            | Rationale |
|-------------------------|------|--------------------------|-----------|
| `hr.contract.type`      | L14  | `hr` (Stage 2)           | `hr` addon not in TIER-1 set |
| `hr.department`         | L14  | `hr` (Stage 2)           | `hr` addon not in TIER-1 set |
| `hr.employee`           | L14  | `hr` (Stage 2)           | `hr` addon not in TIER-1 set |
| `hr.job`                | L14  | `hr` (Stage 2)           | `hr` addon not in TIER-1 set |
| `stock.valuation.layer` | L13  | `stock_account` (Stage 2)| Stock-accounting bridge not in TIER-1 `stock`; lives in the separate `stock_account` addon |

**When Stage 2 lands:** extract `hr` and `stock_account`, pair the 5 entities,
and remove their entries from `coverage::COVERAGE_EXEMPTIONS` in `coverage.rs`.

---

## TIER-1 surplus (extracted-only surface)

229 unique model names extracted across 12 TIER-1 addons.
48 are paired with curated lane consts.
**181 are extracted-only** — surface the curated set has not yet projected.
These are candidates for L-doc expansion in future plan iterations.

Per-addon breakdown (extracted / paired → surplus; 2–3 surplus examples):

- `account`: 66 extracted, 29 paired → **37 surplus**
  (e.g. `account.bank.statement`, `account.bank.statement.line`, `account.cash.rounding`)
- `base`: 114 extracted, 7 paired → **107 surplus**
  (e.g. `avatar.mixin`, `decimal.precision`, `image.mixin` — most are `ir.*` Odoo internals;
  only a handful are domain-meaningful)
- `stock`: 33 extracted, 16 paired → **17 surplus**
  (e.g. `barcode.rule`, `product.removal`, `stock.package`)
- `product`: 25 extracted, 10 paired → **15 surplus**
  (e.g. `product.attribute`, `product.attribute.value`, `product.supplierinfo`)
- `account_edi_ubl_cii`: 16 extracted, 3 paired → **13 surplus**
  (e.g. `account.edi.common`, `account.edi.ubl`, `account.edi.xml.cii`)
- `purchase`: 15 extracted, 11 paired → **4 surplus**
  (e.g. `purchase.bill.line.match`, `product.supplierinfo`)
- `account_peppol`: 10 extracted, 4 paired → **6 surplus**
  (e.g. `account_edi_proxy_client.user`)
- `sale`: 20 extracted, 11 paired → **9 surplus**
  (e.g. `crm.team`, `utm.campaign`)
- `analytic`: 9 extracted, 5 paired → **4 surplus**
  (e.g. `analytic.mixin`, `analytic.plan.fields.mixin`)
- `account_payment`: 7 extracted, 3 paired → **4 surplus**
  (e.g. `payment.provider`, `payment.transaction`)
- `l10n_de`: 8 extracted, 6 paired → **2 surplus**
  (e.g. `account.chart.template`, `ir.actions.report`)
- `uom`: 1 extracted, 1 paired → **0 surplus**

**Interpretation:** extraction surfaces what is there; the curated set decides
what matters for the German MedCare/SMB-Office accounting flow.  The 181
surplus entities are extraction artefacts — they exist in TIER-1 source but
are not (yet) projected onto a lane-doc concept.  Candidates for future L-doc
expansion should be evaluated against savant-relevance criteria, not added
wholesale.

---

## Stage 2 recommendation

The two immediate gaps the extraction revealed:

1. **`hr` addon** — closes L14 entirely (4 entities: `hr.employee`,
   `hr.department`, `hr.job`, `hr.contract.type`).  HR is a major Odoo
   domain; payroll + headcount are often relevant to SMB accounting
   integrations.
2. **`stock_account` addon** — closes the single L13 gap
   (`stock.valuation.layer`).  This is the stock-accounting bridge
   that routes inventory valuation moves into the GL; high relevance
   for manufacturing / distribution accounts.

Likely follow-ons after `hr` + `stock_account` (by downstream-savant
priority, not committed here):

- `crm` — sales pipeline, opportunity, lead; feeds `sale.order` L6
- `project` — task-based billing, timesheets; intersects analytic L10
- `mrp` / `mrp_account` — manufacturing orders, BOM costing; intersects
  stock L7 + GL L1
- `point_of_sale` — POS sessions, payments; intersects payment L5

The actual prioritization should be driven by the savant-relevance
evidence once Stage 1 reasoners are exercised.  This report only
identifies what the extraction side would need to unlock each domain.

---

## Cross-references

- Plan: `.claude/plans/odoo-source-extraction-v1.md` (Stage-1 deliverables table)
- EXT-5 pairing table: `crates/lance-graph-ontology/src/odoo_blueprint/extracted/pairing.rs`
  (`CURATED_EXTRACTED_PAIRS`, 48 entries)
- EXT-3 provenance enhancement: `OdooProvenance.regulation_iri` slot +
  `OdooEntityKind` enum; back-filled across all L1..L15 lane consts
- Epiphany `E-CODEBOOK-INHERITS-FROM-OGIT`: regulation rules are codebook
  entries inherited from OGIT — the `regulation_iri` IRIs point into that
  registry
- Gate test: `crates/lance-graph-ontology/src/odoo_blueprint/extracted/coverage.rs`
  (`COVERAGE_EXEMPTIONS`, `COVERAGE_FLOOR`, `every_lane_meets_coverage_floor`,
  `aggregate_coverage_reports_correctly`)
