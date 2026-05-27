# Odoo Richness Harvest — GAP Lanes (L8–L15) + Savant-Agent Goal

> Companion to `.claude/odoo/BRIEFING.md` (read it first — the dual-axis
> classification, ontology shape, reading discipline, and output template all
> apply unchanged). This file defines the **remaining** lanes after L1–L7 and
> the end-goal those lanes feed: **Savant agents**.
>
> You are ONE read-only analysis lane. Output is a markdown spec draft only.
> **No cargo, no `src/` edits, no git.** Write exactly one file to
> `/home/user/woa-rs/.claude/odoo/<your-lane>.md`. First line MUST be
> `RICHNESS-LANE-OK`; last section MUST be the depth-proof footer
> (`Read: <file> lines=<n> depth=full` per file).

## What L1–L7 already covered (do NOT re-harvest)

| Lane | Covered |
|---|---|
| L1 | K3 double-entry posting (`account.move` state machine, `_check_balanced`, hash) |
| L2 | K3 reconciliation matching (open-item ↔ payment) |
| L3 | K7 USt/VAT compute + fiscal-position mapping (tax compute core) |
| L4 | K8 German report line-mappings + K9 DATEV export |
| L5 | Payments, payment terms, reconcile-model matching rules |
| L6 | Sale + Purchase order → invoice flow (Vorgang lifecycle) |
| L7 | Inventory: stock moves, picking, quant valuation + reservation |

## The remaining lanes (assigned per agent)

Odoo community source root: `/home/user/odoo/addons/` — present modules:
`account, hr, l10n_de, product, purchase, sale, stock`. Enterprise modules
(`account_asset`, `hr_payroll`, `account_reports`, `account_consolidation`)
are **absent** — flag, do not hallucinate; spec only base data/structure.

| Lane | Subsystem | Primary odoo files (read FULLY) | woa-rs K-target |
|---|---|---|---|
| **L8** | Product + UoM + Pricelist + costing | `product/models/product_template.py`, `product_product.py`, `product_category.py`, `product_pricelist.py`, `product_pricelist_item.py`, `uom/models/uom_uom.py` | data foundation; pricing (Vorgang line price), costing config (K3 valuation) |
| **L9** | Partner accounting properties + fiscal-position assignment | `account/models/partner.py` (res.partner extensions: `property_account_receivable_id`, `property_payment_term_id`, `property_account_position_id`, `_get_fiscal_position`, `commercial_partner_id`), base `res.partner` accounting fields | data foundation; partner→tax-mapping inference (AXIS-B) |
| **L10** | Analytic accounting (Kostenstellen) | `analytic/models/analytic_account.py`, `analytic_line.py`, `analytic_plan.py`, `analytic_distribution_model.py`; `account/models/account_move_line.py` analytic_distribution | new K-area: cost-centre allocation; distribution-model rules (AXIS-B) |
| **L11** | Chart of accounts + journals + lock dates + sequences | `account/models/account_account.py` (`account_type`, `reconcile`, `internal_group`), `account_group.py`, `account_journal.py`, `account/models/company.py` lock-date logic (`fiscalyear_lock_date`, `tax_lock_date`, `_get_violated_lock_dates`), `sequence_mixin.py` | K11 lock-date semantics + K3 sequence format families |
| **L12** | Multi-company + multi-currency | base `res.company`, `res.currency` + `res.currency.rate` (`_convert`, `_get_conversion_rate`, rounding), `account_move_line.py` `amount_currency`/`balance` compute, multi-company record rules | K15 Mehrfirma + multi-currency |
| **L13** | Stock↔Accounting valuation bridge + procurement | `stock/models/stock_valuation_layer.py`, `product/models/product.py` valuation (`_run_fifo`, `_run_average`, standard), `stock/models/stock_rule.py`, `stock_warehouse_orderpoint.py`, `stock_lot.py` | bridges stock→K3 (inventory GL postings); reordering (AXIS-A formula + AXIS-B) |
| **L14** | HR base data (employee/org/contract structure) | `hr/models/hr_employee.py`, `hr_department.py`, `hr_job.py`, `hr_contract.py` (base only) | K13 **data foundation** only — payroll ENGINE is Enterprise (absent) → built fresh; flag |
| **L15** | Tax repartition + tax groups + price_include + cash-basis | `account/models/account_tax.py` (`repartition_line_ids`, `_compute_amount`, `price_include`/`include_base_amount` ordering, `account_tax_group.py`, cash-basis transition) | deepens K7/L3 — the base/tax %-split to accounts+tags |

## End-goal these lanes feed: Savant agents

Every rule you tag **AXIS-B (heuristic → delegate)** is a candidate **Savant
agent**: a specialised reasoner defined by three coordinates. Make the
delegation tuple explicit (per `BRIEFING.md`) so the synthesis pass can mint
the Savant directly:

1. **Ontology** — the odoo class → OWL pivot → OGIT family (8-bit) via
   `resolve_odoo_to_family()`. State the expected family (e.g. `0x61
   BillingCore`, `0x62 SMBAccounting`) or `None` (→ "ontology-unmapped, needs
   a Layer-2 alignment axiom").
2. **Use case** — the K-step / business question (`ReasoningKind` ∈
   {CustomerCategory, PostingAnomaly, NextBestAction, InvoiceCompleteness,
   MailIntent, Other(label)}).
3. **Thinking** — `InferenceType` ∈ {Deduction, Induction, Abduction,
   Revision, Synthesis}; `SemiringChoice` ∈ {Boolean, HammingMin, NarsTruth,
   XorBundle, CamPqAdc}; `ThinkingStyle`-cluster ∈ {Analytical, Creative,
   Empathic, Direct, Exploratory, Meta} **inherited from the OGIT family**.

For each AXIS-B rule, end its entry with a one-line **Savant seed**:
`SAVANT: name=<x> family=<0x..|None> reasoning=<Kind> inference=<Type>
semiring=<Choice> style=<cluster> — <1-line why-delegated>`.

AXIS-A rules need no Savant — they are deterministic Rust ports; just give
the rich-AST sketch so an Opus porter can reproduce them.

## Reading discipline (Iron Rule 4 / Op-rule №3)

Read the odoo Python **fully** with the `Read` tool (whole file or
offset/limit chunks covering the entire method). `grep`/`sed`/`head` are
locators only. Quote `file:line-range` for every rule. Odoo source is
canonical; where it's odd, note it, don't "improve".

## Hard rules

- NO `cargo`, NO `src/` edits, NO git. Markdown only, to your one drafts file.
- First line `RICHNESS-LANE-OK`; depth-proof footer last.
- If a subsystem is Enterprise-only, say so and spec only base data/structure.
