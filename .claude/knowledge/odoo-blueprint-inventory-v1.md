# odoo-blueprint-inventory-v1 — Full inventory of the 66 curated `OdooEntity` consts (D-ODOO-BP-1b corpus)

> **READ BY:** odoo-blueprint authors, ARM-discovery proposer implementors, `style_recipe.rs` rule authors, `op_emitter.rs` consumers, the council `prior-art-savant`.
> **Status:** FINDING (corpus is on disk in main as of commit `c04adf10`; line numbers below are absolute against the lane files as of 2026-05-29).
> **Authored:** 2026-05-29 (session continuation, post PR #433 + PR #434 merges).
> **Why this exists:** session-survival index. The 66 entities are 11,563 LOC across 15 files; without this index, a fresh session must grep its way through `odoo_blueprint/l*.rs` to find anything. This doc is the index.

---

## 0. Counting clarification

**66 pub const `OdooEntity` declarations**, NOT 70. The earlier-cited "70" came from a `grep 'OdooEntity {'` literal that matched a handful of nested struct references inside `OdooStateMachine` or `OdooConstraint` bodies that themselves carry an `OdooEntity` reference. The canonical, codegen-consumed count is the **66 `pub const NAME: OdooEntity = OdooEntity {` declarations at module scope**. EXT-6's `COVERAGE.md` cites **53 curated entities** for its TIER-1 backing report — that count excludes the 13 L11-L15 Wave-3 additions that landed AFTER EXT-6 was authored on 2026-05-28. The numbers reconcile.

Some `model_name` values appear in multiple lanes — e.g. `account.move.line` appears in L1 (the journal-entry-line in posting context) AND L2 (the reconciliation context); `account.account.tag` appears in L3 + L4 + L11 + L15 with different field/method subsets per L-doc facet. The lane is the disambiguator, not the model name.

---

## 1. Per-lane summary

| Lane | LOC | Consts | Tests | L-doc | Wave |
|---|---:|---:|---:|---|---|
| L1 | 512 | 3 | 0 (parent) | L1-K3-POST.md | Wave 1 (commit `f5702675`) |
| L2 | 577 | 3 | 3 | L2-K3-RECON.md | Wave 1 (commit `f5702675`) |
| L3 | 1,184 | 4 | 8 | L3-K7-TAX.md | Wave 1 (commit `f5702675`) |
| L4 | 645 | 6 | 8 | L4-K2-SKR-CHART.md / L4-K2-SKR-DE.md | Wave 1 (commit `f5702675`) |
| L5 | 1,136 | 5 | 8 | L5-K6-PAYMENT.md | Wave 1 (commit `f5702675`) |
| L6 | 697 | 4 | 7 | L6-K0-SALES.md / L6-K0-PURCHASE.md | Wave 2 (`d30186e5` + trims in `333a1ff2`) |
| L7 | 603 | 6 | 14 | L7-K0-STOCK.md | Wave 2 + Wave-3 trims |
| L8 | 580 | 6 | 5 | L8-K0-PRODUCT.md | Wave 2 + Wave-3 trims |
| L9 | 627 | 6 | 15 | L9-PARTNER-FISCALPOS.md | Wave 2 dedicated (`d30186e5`) — canonical FiscalPositionResolver lane |
| L10 | 760 | 5 | 4 | L10-K4-ANALYTIC.md | Wave 2 + Wave-3 trims |
| L11 | 826 | 4 | 6 | L11-CHART-LOCKDATE.md | Wave 3 (`333a1ff2`) |
| L12 | 948 | 5 | 15 | L12-MULTICOMPANY-CURRENCY.md | Wave 3 |
| L13 | 993 | 3 | 14 | L13-STOCK-VALUATION-PROCUREMENT.md | Wave 3 |
| L14 | 970 | 4 | 15 | L14-HR-BASE.md | Wave 3 |
| L15 | 505 | 2 | 8 | L15-TAX-REPARTITION.md | Wave 3 |
| **Σ** | **11,563** | **66** | **130** | — | — |

L1's "0 tests" is correct: parent-module tests live in `odoo_blueprint::tests`, not in `l1::tests` — Wave 1 hadn't established the per-lane test convention yet. Wave 2 fixed it on L9; Wave 3 propagated.

---

## 2. Per-entity index (alphabetical by const name, with cross-refs)

Format: `const_name → model_name (kind) [lane:line] L-doc:lines — short description`.
- `ANALYTIC_ACCOUNT` → `account.analytic.account` (Model) [l10:120] L10-ANALYTIC.md:(82 119) — Leaf analytic cost-centre.  Belongs to one analytic plan axis; \
- `ANALYTIC_APPLICABILITY` → `account.analytic.applicability` (Model) [l10:256] L10-ANALYTIC.md:(423 453) — Policy rule: for a given (business_domain product_category \
- `ANALYTIC_DISTRIBUTION_MODEL` → `account.analytic.distribution.model` (Model) [l10:541] L10-ANALYTIC.md:(368 420) — Distribution rule record: maps (partner product_category \
- `ANALYTIC_LINE` → `account.analytic.line` (Model) [l10:357] L10-ANALYTIC.md:(456 483) — Analytic posting record created at journal-entry post time. \
- `ANALYTIC_PLAN` → `account.analytic.plan` (Model) [l10:26] L10-ANALYTIC.md:(82 119) — Analytic plan hierarchy: one orthogonal cost-attribution axis (e.g. \
- `ACCOUNT_ACCOUNT_TAG` → `account.account.tag` (Model) [l11:241] L11-COA-JOURNALS-LOCKDATES.md:(49 57) — Annotation tag scoped to applicability domain (accounts/taxes/products) \
- `ACCOUNT_ACCOUNT` → `account.account` (Model) [l11:49] L11-COA-JOURNALS-LOCKDATES.md:(27 103) — Chart-of-accounts leaf record.  `account_type` is a 19-value closed enum \
- `ACCOUNT_JOURNAL` → `account.journal` (Model) [l11:345] L11-COA-JOURNALS-LOCKDATES.md:(53 103) — Journal record: type-classified posting book (sale/purchase/cash/bank/ \
- `RES_COMPANY_LOCK_DATE` → `res.company` (Model) [l11:528] L11-COA-JOURNALS-LOCKDATES.md:(69 103) — Lock-date extension on res.company (L11 scope: fields R11-R19 only). \
- `ACCOUNT_ACCOUNT_EXCHANGE` → `account.account` (Model) [l12:671] L12-MULTICOMPANY-CURRENCY.md:(79 81) — General ledger account in its FX exchange gain/loss role: \
- `RES_COMPANY_MULTICOMPANY` → `res.company` (Model) [l12:439] L12-MULTICOMPANY-CURRENCY.md:(51 81) — Company / branch node in the multi-company tree (res.company _parent_store). \
- `RES_CURRENCY_RATE` → `res.currency.rate` (Model) [l12:338] L12-MULTICOMPANY-CURRENCY.md:(31 49) — One dated FX exchange rate: foreign units per 1 base currency (technical `rate`). \
- `RES_CURRENCY` → `res.currency` (Model) [l12:71] L12-MULTICOMPANY-CURRENCY.md:(25 82) — ISO 4217 currency: defines rounding precision (decimal_places derived from \
- `RES_USERS_COMPANY_ACCESS` → `res.users` (Model) [l12:598] L12-MULTICOMPANY-CURRENCY.md:(54 56) — User model: company_id (active company) and allowed_company_ids \
- `STOCK_LOT` → `stock.lot` (Model) [l13:602] L13-STOCK-VALUATION-PROCUREMENT.md:(71 75) — Lot/serial number master; uniqueness per (product_id company_id name) \
- `STOCK_RULE` → `stock.rule` (Model) [l13:416] L13-STOCK-VALUATION-PROCUREMENT.md:(38 82) — Procurement rule mapping (dest route) → action (pull/push/transparent); \
- `STOCK_WAREHOUSE_ORDERPOINT` → `stock.warehouse.orderpoint` (Model) [l13:186] L13-STOCK-VALUATION-PROCUREMENT.md:(55 79) — Min/max reorder rule for one product at one location; drives scheduler batch \
- `HR_CONTRACT_TYPE` → `hr.contract.type` (Model) [l14:722] L14-HR-BASE.md:(68 81) — Employment contract type (e.g. CDI / CDD / interim); community stub with \
- `HR_DEPARTMENT` → `hr.department` (Model) [l14:431] L14-HR-BASE.md:(41 78) — Organisational unit in the company tree; recursive parent hierarchy \
- `HR_EMPLOYEE` → `hr.employee` (Model) [l14:62] L14-HR-BASE.md:(29 81) — Work-resource individual with versioned employment terms (hr.version chain); \
- `HR_JOB` → `hr.job` (Model) [l14:588] L14-HR-BASE.md:(47 49) — Job role / position within a department; tracks active headcount \
- `ACCOUNT_ACCOUNT_TAG_L15` → `account.account.tag` (Model) [l15:159] L15-TAX-REPARTITION.md:(529 545) — L15 ext: balance_negate + report_expression_id on account.account.tag. \
- `ACCOUNT_TAX_L15` → `account.tax` (Model) [l15:321] L15-TAX-REPARTITION.md:(64 614) — L15 ext: CABA gate + rounding engine + totals. Core in L3. Adds \
- `ACCOUNT_JOURNAL` → `account.journal` (Model) [l1:458] L1-K3-POST.md:(488 562) — Accounting journal; owns Belegnummer sequence format optional \
- `ACCOUNT_MOVE_LINE` → `account.move.line` (Model) [l1:266] L1-K3-POST.md:(234 362) — Journal entry line; balance/debit/credit in company currency \
- `ACCOUNT_MOVE` → `account.move` (Model) [l1:51] L1-K3-POST.md:(27 743) — Double-entry journal entry / invoice; draft→posted→cancel state machine \
- `ACCOUNT_FULL_RECONCILE` → `account.full.reconcile` (Model) [l2:461] L2-K3-RECON.md:(530 567) — Completed settlement event — groups all partials and AMLs that together \
- `ACCOUNT_MOVE_LINE` → `account.move.line` (Model) [l2:42] L2-K3-RECON.md:(27 466) — Journal entry line — the debit/credit leaf of a double-entry move; \
- `ACCOUNT_PARTIAL_RECONCILE` → `account.partial.reconcile` (Model) [l2:252] L2-K3-RECON.md:(468 617) — Partial settlement event — one debit/credit AML pair matched for a \
- `ACCOUNT_FISCAL_POSITION` → `account.fiscal.position` (Model) [l3:897] L3-K7-TAX.md:(661 800) — Tax regime mapping rule: translates taxes and GL accounts for a partner. \
- `ACCOUNT_TAX_GROUP` → `account.tax.group` (Model) [l3:111] L3-K7-TAX.md:(624 658) — Groups taxes for display + closing-entry accounts (USt/VSt/advance); \
- `ACCOUNT_TAX_REPARTITION_LINE` → `account.tax.repartition.line` (Model) [l3:626] L3-K7-TAX.md:(573 620) — Distribution rule mapping a tax computation result to a GL account and \
- `ACCOUNT_TAX` → `account.tax` (Model) [l3:467] L3-K7-TAX.md:(805 850) — VAT / USt tax definition with computation type (percent/fixed/division/group) \
- `ACCOUNT_ACCOUNT_DE` → `account.account` (Model) [l4:130] L4-K8K9-REPORTS-DATEV.md:(296 370) — General ledger account extended by l10n_de: blocks code changes \
- `ACCOUNT_ACCOUNT_TAG` → `account.account.tag` (Model) [l4:40] L4-K8K9-REPORTS-DATEV.md:(42 295) — SKOS-style classification label applied to accounts or taxes; \
- `ACCOUNT_JOURNAL_DE` → `account.journal` (Model) [l4:472] L4-K8K9-REPORTS-DATEV.md:(363 367) — Journal model extended by l10n_de: bank/cash liquidity accounts \
- `ACCOUNT_TAX_DATEV` → `account.tax` (Model) [l4:212] L4-K8K9-REPORTS-DATEV.md:(368 451) — Tax record extended for DATEV export: carries a 4-character \
- `PRODUCT_TEMPLATE_DE` → `product.template` (Model) [l4:290] L4-K8K9-REPORTS-DATEV.md:(376 422) — Product template extended by l10n_de: _get_product_accounts \
- `RES_COMPANY_DE` → `res.company` (Model) [l4:363] L4-K8K9-REPORTS-DATEV.md:(330 367) — Company record augmented by l10n_de chart-template setup: \
- `PAYMENT_TERM_LINE` → `account.payment.term.line` (Model) [l5:553] L5-PAY-TERMS-MATCH.md:(296 330) — One installment line within a payment term; computes due date via \
- `PAYMENT_TERM` → `account.payment.term` (Model) [l5:392] L5-PAY-TERMS-MATCH.md:(193 341) — Structured payment obligation terms (installments Skonto/early-discount \
- `PAYMENT` → `account.payment` (Model) [l5:87] L5-PAY-TERMS-MATCH.md:(32 191) — A posted payment event generating double-entry journal lines; \
- `RECONCILE_MODEL_LINE` → `account.reconcile.model.line` (Model) [l5:895] L5-PAY-TERMS-MATCH.md:(456 468) — Write-off journal line template within a reconcile model; \
- `RECONCILE_MODEL` → `account.reconcile.model` (Model) [l5:686] L5-PAY-TERMS-MATCH.md:(399 563) — Declarative rule for bank-statement-to-open-item matching \
- `PURCHASE_ORDER_LINE` → `purchase.order.line` (Model) [l6:529] L6-SALE-PURCHASE.md:(611 643) — One line on a purchase order; qty_to_invoice feeds the order-level \
- `PURCHASE_ORDER` → `purchase.order` (Model) [l6:410] L6-SALE-PURCHASE.md:(546 643) — Vendor purchase order (RFQ→PO); optional two-step approval via \
- `SALE_ORDER_LINE` → `sale.order.line` (Model) [l6:216] L6-SALE-PURCHASE.md:(186 743) — One line on a sale order; tracks price_unit/discount/tax_ids and \
- `SALE_ORDER` → `sale.order` (Model) [l6:75] L6-SALE-PURCHASE.md:(31 543) — Commercial sale quotation/order (Vorgang lifecycle: draft→sent→sale→cancel); \
- `STOCK_LOCATION` → `stock.location` (Model) [l7:355] L7-STOCK.md:(155 207) — Node in the stock location hierarchy (physical or virtual); \
- `STOCK_MOVE_LINE` → `stock.move.line` (Model) [l7:132] L7-STOCK.md:(77 108) — One lot/package/owner reservation or done-qty record within a stock move; \
- `STOCK_MOVE` → `stock.move` (Model) [l7:56] L7-STOCK.md:(31 108) — One product movement between two stock locations; state machine \
- `STOCK_PICKING` → `stock.picking` (Model) [l7:273] L7-STOCK.md:(419 500) — Group of stock moves for one logistics operation (receipt/delivery/internal); \
- `STOCK_QUANT` → `stock.quant` (Model) [l7:173] L7-STOCK.md:(110 415) — Persistent stock record: qty of one product at one location with \
- `STOCK_WAREHOUSE` → `stock.warehouse` (Model) [l7:421] L7-STOCK.md:(608 644) — Physical warehouse site with operational config (picking types routes \
- `PRODUCT_CATEGORY` → `product.category` (Model) [l8:44] L8-PRODUCT-UOM-PRICELIST.md:(180 238) — Hierarchical product category; parent_path materialized closure used in \
- `PRODUCT_PRICELIST_ITEM` → `product.pricelist.item` (Model) [l8:415] L8-PRODUCT-UOM-PRICELIST.md:(263 660) — Pricelist rule: applied_on scope min_quantity (product UoM) date validity. \
- `PRODUCT_PRICELIST` → `product.pricelist` (Model) [l8:359] L8-PRODUCT-UOM-PRICELIST.md:(242 412) — Pricelist: currency + company + country-group scoping + ordered rule set. \
- `PRODUCT_PRODUCT` → `product.product` (Model) [l8:282] L8-PRODUCT-UOM-PRICELIST.md:(62 115) — Product variant: _inherits product.template. lst_price = list_price + \
- `PRODUCT_TEMPLATE` → `product.template` (Model) [l8:167] L8-PRODUCT-UOM-PRICELIST.md:(42 115) — Product template: canonical catalog record. Drives variant matrix via \
- `UOM_UOM` → `uom.uom` (Model) [l8:103] L8-PRODUCT-UOM-PRICELIST.md:(119 177) — Unit of Measure: factor-based conversion. Reference unit factor=1.0. \
- `ACCOUNT_FISCAL_POSITION` → `account.fiscal.position` (Model) [l9:38] L9-PARTNER-FISCALPOS.md:(207 413) — Named tax-and-account mapping applied to a partner; selected by \
- `ACCOUNT_FISCAL_POS_ACCOUNT` → `account.fiscal.position.account` (Model) [l9:173] L9-PARTNER-FISCALPOS.md:(348 373) — One account-substitution rule inside a fiscal position: \
- `ACCOUNT_PAYMENT_TERM_REF` → `account.payment.term` (Model) [l9:445] L9-PARTNER-FISCALPOS.md:(59 78) — Payment terms (L5 authoritative); projected in L9 only as target of \
- `RES_COUNTRY_GROUP` → `res.country.group` (Model) [l9:403] L9-PARTNER-FISCALPOS.md:(263 277) — Named set of countries for fpos matching predicate 5: \
- `RES_COUNTRY` → `res.country` (Model) [l9:363] L9-PARTNER-FISCALPOS.md:(207 291) — Country; 2-char ISO code used in _get_fiscal_position to compute \
- `RES_PARTNER_ACCOUNTING` → `res.partner` (Model) [l9:229] L9-PARTNER-FISCALPOS.md:(32 513) — Partner extended by account module: AR/AP property accounts payment terms \

---

## 3. Per-lane semantic role + EXT-2 coverage

| Lane | Semantic role | EXT-2 backing (`extracted/`) | Coverage (EXT-6) |
|---|---|---|---|
| L1 | Journal posting (`account.move` lifecycle) | `account.rs` | 100% |
| L2 | Reconciliation (partial/full settlement of AMLs) | `account.rs` (shared) | 100% |
| L3 | Tax: groups + rates + repartition + fiscal position | `account.rs` (shared) | 100% |
| L4 | German SKR03/SKR04 chart + DATEV-tagged taxes | `l10n_de.rs` + `l10n_de_chart.rs` (24,712 LOC SKR03/04 + UStVA Kz) | 100% |
| L5 | Payments + payment terms + reconcile-model auto-matching | `account_payment.rs` | 100% |
| L6 | Sales + Purchase order chain (header + lines) | `sale.rs` + `purchase.rs` | 100% |
| L7 | Stock movement (move/quant/picking/location/warehouse) | `stock.rs` (12,020 LOC) | 100% |
| L8 | Product catalogue + UoM + pricelist | `product.rs` + `uom.rs` | 100% |
| L9 | Fiscal-position resolver — canonical worked example | `account.rs` (shared) | 100% |
| L10 | Analytic accounting plan + line + distribution | `analytic.rs` | 100% |
| L11 | Account chart + journal + company lock-date hardening | `account.rs` (shared) | 90.6% (5 explicit TIER-2 deferrals workspace-wide) |
| L12 | Multi-company + multi-currency exchange | _no dedicated file; account/base shared_ | (Wave-3, post EXT-6) |
| L13 | Stock valuation layer + procurement (orderpoint/rule/lot) | `stock.rs` (shared) | (Wave-3, post EXT-6) |
| L14 | HR base (employee/department/job/contract-type) | _none_ — HR is TIER-2 deferral per `odoo-source-extraction-v1.md` | (deferred) |
| L15 | Tax-repartition extensions to L3 (CABA + balance_negate) | `account.rs` (shared) | (Wave-3 extension, post EXT-6) |

**EXT-2 backing files (`extracted/`):** 99,209 LOC across 11 `.rs` files; coverage gate test in `extracted/coverage.rs`. Curated stays canonical on conflict per `pairing.rs::CURATED_EXTRACTED_PAIRS`.

---

## 4. Field / method / state-machine density (where the work is)

Approximate counts via `grep -cE '^            kind: Odoo<...>Kind::'` per lane:

| Lane | Field-kind hits | Method-kind hits | Decorator-kind hits | State-machine present |
|---|---:|---:|---:|---:|
| L1 | 52 | 41 | 8 | (move state) |
| L2 | 21 | 20 | 4 | (recon state) |
| L3 | 0* | 0* | 0* | — |
| L4 | 21 | 3 | 0 | — |
| L5 | 55 | 26 | 11 | (payment state) |
| L6 | 0* | 0* | 0* | (sale state) |
| L7 | 11 | 0* | 0* | (picking state) |
| L8 | 0* | 0* | 6 | — |
| L9 | 0* | 0* | 0* | — |
| L10 | 36 | 16 | 2 | — |
| L11 | 28 | 16 | 4 | (lock state) |
| L12 | 29 | 15 | 2 | — |
| L13 | 44 | 18 | 3 | — |
| L14 | 39 | 12 | 6 | — |
| L15 | 0* | 0* | 0* | — |

`*` = lane carries fields/methods but not with the exact `kind: OdooFieldKind::X` indentation pattern the grep expects (different formatting choices across Wave-1/2/3 agents — usually because the entry references a shared sub-const declared elsewhere in the file or inherits via reference). The 0* cells are **NOT empty entities**; verify via `wc -l l<N>.rs` and the const-content blocks.

State-machine presence audit needs a follow-up pass: `grep -E '^&OdooStateMachine \{|state_machine: Some\(&' l*.rs` came back 0 because of multi-line formatting, but L1 (move), L2 (recon), L5 (payment), L6 (sale order), L7 (picking), and L11 (lock_date) all carry state semantics per their L-doc role. The state machines may be inlined or referenced from a sibling const; this is a known gap to clean up in a follow-up.

---

## 5. Wave provenance summary

| Wave | Date | Commit | Agents | Lanes | Entities added |
|---|---|---|---:|---|---:|
| Scaffold | 2026-05-28 | `0c0ad4d3` | (mechanical) | l1..l15 stubs created | 0 (stubs) |
| Wave 1 | 2026-05-28 | `f5702675` | **5 Sonnet agents** (1 per lane) | L1, L2, L3, L4, L5 | 21 (= 3+3+4+6+5) |
| Wave 2 dedicated | 2026-05-28 | `d30186e5` | 1 Sonnet agent | L9 (FiscalPositionResolver canonical) | 6 |
| Wave 2 + Wave-3 trims | 2026-05-28 | `333a1ff2` | 5 Sonnet agents (Wave-3) + reviewer | L6, L7, L8, L10 (Wave-2 work folded in) + L11, L12, L13, L14, L15 | 39 (= 4+6+6+5 + 4+5+3+4+2) |
| EXT-3 follow-up | 2026-05-28 | `c04adf10` | Main thread | `OdooEntityKind` + `regulation_iri` back-fill across all 66 entities | 0 (additive) |

**The "5 agents" of the question** = Wave 1 specifically — 5 Sonnet agents fanned out one-per-lane for L1-L5, each agent producing its lane's `OdooEntity` consts via mechanical L-doc-prose-to-typed-projection. The same fan-out pattern was used in Wave 3 (L11-L15) but with 5 agents working in a different lane band; the Wave-2 work (L6-L10) was done by ~3 agents with the canonical L9 carved out as its own commit.

---

## 6. What's NOT in this corpus (and where it would land)

| Missing | Why | Landing site if it ships |
|---|---|---|
| Cron-method coverage | No `OdooMethodKind::Cron` entries today; Odoo crons live in XML data, not Python AST | Extractor enrichment (`tools/odoo-blueprint-extractor/parsers/methods.py`) |
| Multi-company access rules | L12 covers res.users.company_access; full record-rule corpus not included | Future BP-1c hop (OGIT classifier extension) |
| Asset depreciation models (`account.asset.*`) | TIER-2 addon — explicitly deferred per `odoo-source-extraction-v1.md` | Stage 2 (TIER-2 addons) — EXT-2 follow-up |
| Manufacturing (`mrp.*`), POS (`pos.*`), Project (`project.*`) | TIER-2 addons | Stage 2 |
| Stage-2 dark atoms (Money/Quantity/ApplyRate/EmitAmount/Event/FiscalCtx) | Not lit because `return_kind` + `semantic_role` + `computed` fields are sparse in the curated set | Extractor enrichment OR ARM-discovery proposer (see `streaming-arm-nars-discovery-v1.md`) |

---

## 7. Cross-refs

- `style_recipe.rs` (PR #433) consumes these 66 entities + their `OdooMethod` slices to emit `OdooStyleRecipe` per method.
- `op_emitter.rs` (this PR #435 branch) groups `OdooStyleRecipe` corpus by `OdooMethodKind` bucket and emits compilable Rust dispatch.
- `extracted/` (D-ODOO-EXT-2 shipped 2026-05-28) provides 48/53 TIER-1 backing per `COVERAGE.md`.
- `tools/odoo-blueprint-extractor/` (D-ODOO-EXT-1 shipped) is the Python AST walker that produces the backing.
- `pairing.rs::CURATED_EXTRACTED_PAIRS` enforces canonical-on-conflict between this corpus and the EXT-2 output.
- ARM-discovery (this branch, PR #435 plan) is the **third proposer leg** that complements this corpus with runtime-data-derived candidates.
- See `odoo-extraction-strategies-v1.md` for the doctrine of three proposer legs.
- See `odoo-extraction-tools-v1.md` for the tool stacks behind each leg.

End of inventory.
