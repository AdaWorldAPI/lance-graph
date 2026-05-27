# Savant: PartnerTrustAdvisor  (id 2 · family 0x80 · lane L9)

**Tuple:** kind=CustomerCategory · inference=Revision · semiring=NarsTruth · style=Empathic
**Feeds Reasoner impl:** `CustomerCategoryReasoner`   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::CustomerCategory` -> "classify against the family codebook (deductive
> lookup)" (`examples/savant_dispatch.rs:29`); here the *revision* facet (belief-update on each new
> payment event) rides the same impl, selecting `QueryStrategy::BundleInto` via
> `InferenceType::Revision::default_strategy()`. Style Empathic inherited from 0x80 SmbFoundryCustomer
> (relationship + trust). Note: L9 R16's seed line names `inference=Induction` in prose but its own
> rationale ("update beliefs on each payment event via Revision inference") and the authoritative
> `contract::savants` tuple both pin **Revision** -- followed here.

## What it decides (AXIS-B core)
Maintain a partner's **trust / dunning-risk category** (`trust in {good, normal, bad}`) as an
evidence-revised belief rather than a manually-set static flag. odoo's `trust` field carries no
automatic computation (L9 R16, `partner.py:566`); the AXIS-B core is to **revise** the trust belief on
each new payment-history event -- a payment arriving on time strengthens `good`, a slipped due date or
a rising DSO weakens it toward `bad`. Output is a suggested trust level (and a dunning-escalation
hint) with NARS `(frequency, confidence)`; woa-rs surfaces it to accounting staff, who keep final say
over the stored `trust` value and any Mahnwesen action.

## Deterministic guard (AXIS-A -- stays in woa-rs)
The computed inputs are deterministic: `credit` (total receivable = sum `amount_residual` on posted
unreconciled `asset_receivable` lines) and `debit` (R6, `partner.py:365-449`), `days_sales_outstanding
= (credit / total_invoiced_tax_included) * days_since_oldest_invoice` (R7, `partner.py:472-490`),
`credit_limit` / `use_partner_credit_limit` (R15, `partner.py:515-524, 670-683`), and the
`trust in {good,normal,bad}` field space + `company_dependent` storage (R16, `partner.py:566`). The
trust *assignment* is a human judgment in odoo; the *use* of trust in escalation is the AXIS-B core
(L9 R16). woa-rs computes the balances/DSO deterministically and gates whether dunning is even active
(credit-limit enforcement lives in the sale/invoice flow, R15) -- the savant only revises the
qualitative belief.

## Slot 1 -- Evidence (Arrow EvidenceRef)
Two tables. The *partner risk snapshot*
`EvidenceRef { table: "res_partner.trust_snapshot", schema_fingerprint, rows }`
(one row = the partner under evaluation, derived over `commercial_partner_id`):

| column | dtype | signal |
|---|---|---|
| `partner_id` | `Int64` | the partner the trust belief is scoped to (`commercial_partner_id` rollup, R5) |
| `trust` | `Utf8` (`good\|normal\|bad`) | the **prior** belief being revised (current stored value) |
| `credit` | `Decimal128` | total receivable (open AR); high open balance weakens trust (R6) |
| `debit` | `Decimal128` | total payable (R6) -- context for net exposure |
| `days_sales_outstanding` | `Float64` | aggregate slowness of payment (R7); rising DSO is the main "bad" signal |
| `credit_limit` | `Decimal128` | per-partner limit; proximity/breach is a risk signal (R15) |
| `use_partner_credit_limit` | `Boolean` | whether a partner-specific limit is in force (R15) |
| `customer_rank` | `Int64` | relationship maturity / volume (R4); a long-standing high-rank customer tempers a single late event |

The *payment-event stream* `EvidenceRef { table: "account_move.partner_payment_history", ... }`
(the time-ordered events Revision folds in, one row per receivable move/payment):

| column | dtype | signal |
|---|---|---|
| `move_id` | `Int64` | invoice / payment identity |
| `invoice_date` | `Date32` | event time (recency-weights the revision) |
| `date_due` | `Date32`/nullable | promised date -- `paid_date - date_due` is the lateness signal |
| `paid_date` | `Date32`/nullable | when reconciled (NULL = still open -> overdue if past `date_due`) |
| `amount_residual` | `Decimal128` | open remainder per move (drives R6 `credit`) |
| `reconciled` | `Boolean` | whether the move is settled (an unsettled overdue move is the strongest "bad" event) |

## Slot 2 -- Odoo field -> signal map                 (cite L-doc file:lines)
- `trust in {good,normal,bad}` field, company_dependent, no auto-compute (debtor-quality signal) <- `L9-PARTNER-FISCALPOS.md:456-467` (R16; `partner.py:566`).
- `credit` / `debit` computed balances (posted, unreconciled, AR/AP account types) <- `L9-PARTNER-FISCALPOS.md:155-181` (R6; `partner.py:365-449`).
- `days_sales_outstanding` formula (`credit / total_invoiced_tax_included * days_since_oldest_invoice`) <- `L9-PARTNER-FISCALPOS.md:184-203` (R7; `partner.py:472-490`).
- `credit_limit` / `use_partner_credit_limit` / `show_credit_limit` (company default fallback) <- `L9-PARTNER-FISCALPOS.md:442-452` (R15; `partner.py:515-524, 670-683`).
- `commercial_partner_id` rollup (accounting fields incl. `credit_limit` sync parent->child) <- `L9-PARTNER-FISCALPOS.md:130-148` (R5; `partner.py:702-710`).
- `customer_rank` maturity counter <- `L9-PARTNER-FISCALPOS.md:100-126` (R4; `partner.py:600-601, 800-833`).
- delegation tuple `(CustomerCategory, Induction/Revision, NarsTruth)` + savant seed (revise trust from payment history) <- `L9-PARTNER-FISCALPOS.md:461-467` (R16).
- NEEDS-INPUT: `date_due` / `paid_date` per-move lateness columns. L9 captures `credit`/DSO as aggregates (R6/R7) but does not enumerate the per-move due/paid timestamps the Revision stream folds; these live in the move/payment-matching lanes (L2 recon, L5 payment-terms) -- confirm the exact `account.move(.line)` due/reconcile fields with the L2/L5 worker before the impl binds the event-stream columns.

## Slot 3 -- Property-level alignment
N/A -- class-level pivots only; no `owl:equivalentProperty` defined. (Confirmed: `odoo_alignment.rs`
holds only class-level `owl:equivalentClass` rows; zero property IRIs in the repo.) The evidence is
partner-internal plus its own receivable stream; `res.partner` -> `fibo:LegalEntity`
(`odoo_alignment.rs:214-219`, class-level) and `account.move` -> `fibo:Transaction` are the only
class-level pivots touched, used as identity keys, not traversed properties. The decision does not
cross the FIBO/SKR/ZUGFeRD seam at decision time. **N/A -- stays within 0x80 reading its own AR stream.**

## Slot 4 -- AXIS-B decision in evidence terms
Let E0 = the partner risk snapshot (slot 1, carrying the prior `trust`) and let
{e1, e2, ...} = the time-ordered payment events. Revision folds each event into the trust belief.

-> Conclusion C = `ReviseTrust(partner_id, good | normal | bad)` emitted with NARS
`(frequency, confidence)` where:
- the **prior** is the current stored `trust` (mapped to a frequency: good~high, normal~mid, bad~low);
  each event revises it. NarsTruth Revision merges the prior truth with the new evidence's truth.
- **frequency** rises (toward `good`) with: on-time `paid_date <= date_due` events, low/stable
  `days_sales_outstanding`, `credit` well under `credit_limit`, high `customer_rank` maturity.
- **frequency** falls (toward `bad`) with: overdue unreconciled moves (`reconciled=false` past
  `date_due`), rising DSO, `credit` near/over `credit_limit`, a cluster of recent late events.
- **confidence** rises with the number of payment events folded (a long history of consistent behaviour
  yields a confident belief; a brand-new partner with one invoice keeps confidence low). Revision's
  evidence accumulates monotonically toward the phi-1 ceiling -- never certainty.

Discriminating features (ranked): overdue/late `paid_date - date_due` (NEEDS-INPUT) >> trend in
`days_sales_outstanding` > `credit` vs `credit_limit` proximity > recency-weighted streak of clean
payments > `customer_rank`. Revision (not one-shot induction) is the right inference because trust is a
*continuously updated* belief: each payment event nudges it rather than recomputing from scratch
(L9 R16 rationale).

## Parity / GoBD notes
The `trust` value feeds Mahnwesen (dunning) escalation (L9 R16 woa-rs target; L9 open question 5 flags
that woa-rs has partial Mahnwesen from Round 9). Suggestion-only per Iron Rule 7: the savant proposes a
trust revision and a dunning hint, but a human confirms the stored `trust` and authorises any dunning
step -- automatic debtor downgrading has reputational/legal weight and must not be un-guarded. No GoBD
Festschreibung interaction (trust is master-data, not a posted ledger entry); changing it does not
touch the immutable journal. The credit-limit *enforcement* (blocking an invoice) stays in the
sale/invoice flow as a deterministic guard, separate from this advisory belief.
