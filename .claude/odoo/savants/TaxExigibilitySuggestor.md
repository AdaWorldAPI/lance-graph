# Savant: TaxExigibilitySuggestor  (id 15 · family 0x62 · lane L15)

**Tuple:** kind=NextBestAction · inference=Induction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** `NextBestActionReasoner`   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::NextBestAction` -> "induce the action with the highest expected value"
> (`examples/savant_dispatch.rs:32`). Induction -> `QueryStrategy::CamWide`. Style Analytical
> inherited from 0x62 SMBAccounting.

## What it decides (AXIS-B core)
Recommend the **tax exigibility regime** a company (or a tax configuration) should use:
`on_invoice` (Soll-Besteuerung -- VAT due at posting) vs `on_payment` (Ist-Besteuerung / cash basis
-- VAT due at payment). The hard mechanics of each regime are deterministic (transition-account
routing, CABA collection); the AXIS-B core is the *eligibility/fit recommendation*: given a company's
evidence (revenue band relative to the para 20 UStG threshold, industry/legal-form pattern, current
accounting setup), induce whether switching to (or remaining on) Ist-Besteuerung is the better
next action. Output: a recommend `on_payment` / `on_invoice` suggestion with NARS
`(frequency, confidence)`; woa-rs only proposes it, never flips the company flag.

## Deterministic guard (AXIS-A -- stays in woa-rs)
The `tax_exigibility` field space (`on_invoice`/`on_payment`), the company-level enable flag
`company.tax_exigibility` (CABA globally on/off; `hide_tax_exigibility` UI gate), the reconcilable
transition-account constraint, and the entire CABA posting/collection mechanism are deterministic:
`L15-TAX-REPARTITION.md:475-525` (R11; `account_tax.py:L164-174, L247-255, L5201-5210`,
`account_move.py:L4080-4147`). The savant runs only behind the guard `company.tax_exigibility == true`
(`L15-TAX-REPARTITION.md:617-623` RS1 AXIS-A part).

## Slot 1 -- Evidence (Arrow EvidenceRef)
Primary table the company tax-regime context, `EvidenceRef { table: "res_company.tax_exigibility_context", schema_fingerprint, rows }`
(one row = the company under evaluation):

| column | dtype | signal |
|---|---|---|
| `company_id` | `Int64` | the company the recommendation is scoped to |
| `tax_exigibility` | `Boolean` | company-level CABA enable flag (the AXIS-A precondition; if false, recommendation is hidden) |
| `current_default_exigibility` | `Utf8` (`on_invoice\|on_payment`) | the regime currently in effect (what we may switch from) |
| `cash_basis_transition_account_id` | `Int64`/nullable | presence + reconcilability gates whether on_payment is even configurable (R11 constraint) |
| **`annual_revenue`** | `Decimal128` | **NEEDS-INPUT** -- para 20 UStG <= 600,000 EUR eligibility signal; NOT defined in L15 (see Slot 2) |
| **`legal_form` / `industry_code`** | `Utf8` | **NEEDS-INPUT** -- industry/legal-form pattern signal cited by RS1; NOT in L15 |
| **`bookkeeping_obligation`** | `Boolean` | **NEEDS-INPUT** -- whether the entity is bilanzierungspflichtig (Soll forced for double-entry-obligated entities); NOT in L15 |

Only the first four columns are sourceable from L15 (the tax-mechanics lane). The revenue / industry /
legal-form discriminators that RS1 names as the inductive evidence live in the partner/company lane
(L9) and a revenue ledger, not here -- flagged below.

## Slot 2 -- Odoo field -> signal map                 (cite L-doc file:lines)
- `tax_exigibility` field space + `on_invoice`(Soll) / `on_payment`(Ist) semantics <- `L15-TAX-REPARTITION.md:483-487` (R11; `account_tax.py:L164-174`).
- company-level `company.tax_exigibility` enable flag + `hide_tax_exigibility` UI gate <- `L15-TAX-REPARTITION.md:522-525` (R11 parity notes; `account_tax.py:L163`).
- `cash_basis_transition_account_id` must be reconcilable (constraint) <- `L15-TAX-REPARTITION.md:479-480, 522-523` (R11; `account_tax.py:L247-255`).
- transition-account routing (`_get_aml_target_tax_account`) <- `L15-TAX-REPARTITION.md:488-495` (R11; `account_tax.py:L5201-5210`).
- savant seed + delegation tuple `(NextBestAction, Induction, NarsTruth, Analytical)` + para 20 UStG <= 600k threshold cited as the heuristic-not-hardcoded core <- `L15-TAX-REPARTITION.md:617-623` (RS1).
- NEEDS-INPUT: `annual_revenue`, `legal_form`/`industry_code`, `bookkeeping_obligation` -- RS1 names revenue/industry/setup as the inductive evidence (`L15-TAX-REPARTITION.md:619-623`) but L15 does not define these fields. Source candidates: L9 (partner/company facets, `res.company` / `res.partner`) and a revenue aggregate; confirm with the L9 worker / woa-rs before the impl binds these columns.

## Slot 3 -- Property-level alignment
Decision stays **within family 0x62 SMBAccounting**. `account.tax` -> `fibo:TaxTreatment`,
`account.tax.group` -> `fibo:TaxCategory` (class-level, `L15-TAX-REPARTITION.md:24-29`). The
exigibility regime is a German-tax (UStG) policy choice expressed as a scalar on company/tax config;
it does not traverse the FIBO/SKR/ZUGFeRD seam to be decided. PROPOSED only (no property axiom
exists): if VAT-regime reporting later crosses into ZUGFeRD/USt-VA, `odoo:tax_exigibility ->
zugferd:taxPointDate` / `fibofnd:hasTaxDueBasis` would be the property -- not present today. For the
AXIS-B decision: **N/A -- stays within 0x62**.

## Slot 4 -- AXIS-B decision in evidence terms
Let E = the company tax-regime context (slot 1), with the AXIS-A precondition `tax_exigibility == true`.

-> Conclusion C = `RecommendExigibility(company_id, on_payment | on_invoice)` emitted with NARS
`(frequency, confidence)` where:
- **frequency** of recommending `on_payment` (Ist) rises with: `annual_revenue` below the para 20
  UStG threshold (<= 600,000 EUR), a `legal_form`/`industry` pattern typical of cash-basis-eligible
  SMBs, and absence of a `bookkeeping_obligation` that would force Soll -- these are the
  NEEDS-INPUT signals named by RS1.
- **frequency** of `on_invoice` (Soll) rises with revenue above threshold, bilanzierungspflicht, or a
  missing/non-reconcilable `cash_basis_transition_account_id` (on_payment not configurable).
- **confidence** is the NARS weight from how much corroborating company evidence is present; with only
  the four L15-sourceable columns and the revenue/industry signals stubbed (NEEDS-INPUT), confidence
  stays low. Capped by phi-1.

Discriminating features (ranked): `annual_revenue` vs threshold (NEEDS-INPUT) >> `bookkeeping_obligation`
(NEEDS-INPUT) > `cash_basis_transition_account_id` configurability > `legal_form`/`industry`
(NEEDS-INPUT) > `current_default_exigibility` inertia. Induction: "companies like this one (revenue
band, industry) have benefited from Ist-Besteuerung."

## Parity / GoBD notes
Ist-Besteuerung (para 20 UStG) is a German small-business option; the default and majority case is
Soll-Besteuerung. The recommendation is suggestion-only (Iron Rule 7): even if adopted, every CABA
move still routes VAT through the reconcilable transition account (R11) and the periodic USt-VA
closing is unaffected in form. The savant must never auto-switch the regime (a regime change has
retroactive USt reporting consequences) and must respect `company.tax_exigibility == false` (CABA
disabled) as a hard mute.
