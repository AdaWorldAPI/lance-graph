# Savant: PaymentToInvoiceMatcher  (id 21 · family None · lane L5)

**Tuple:** kind=Other(RECONCILE_MATCH=5) · inference=Induction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** the single `Other(5)` reasoner   (RECONCILE_MATCH; shared with id 19 ReconcileMatchSelector — see cross-savant note)

> dispatch: `ReasoningKind::Other(5)` -> "match open items / bank lines by evidence fusion (reconcile)"
> (`examples/savant_dispatch.rs:34-36`, the `5 | 6 =>` arm). Induction -> `QueryStrategy::CamWide`.
> Style Analytical (SMBAccounting basin); `family=None` — `account.payment` is ontology-unmapped
> (L5 FLAG, proposed `fibo-FBC-PAS-FPAS:Payment`, not yet in `odoo_alignment.rs`).

> **CROSS-SAVANT (RECONCILE_MATCH=5 is shared by id 19 and id 21).** The single `Other(5)` impl must
> distinguish the two by `ReasoningContext.namespace` + evidence shape, **NOT** by code:
> - **id 19 (ReconcileMatchSelector)** — `namespace = "erp.k3.reconcile_match"`; evidence = a *set* of open
>   items on ONE account; output = **propose a candidate grouping** (subset-of-ids).
> - **id 21 (this savant)** — `namespace = "erp.k3.payment_reconcile"`; evidence = one payment's counterpart
>   line + its target open invoices; output = a **boolean gate** "does this payment fully reconcile the
>   open invoices?" (the Mahnwesen / dunning gate).
> Same `Other(5)`, two ReasoningContexts; the impl branches on namespace/evidence, never on the code.

## What it decides (AXIS-B core)
Given one incoming `account.payment` (its counterpart receivable/payable line) and the partner's set of
open invoices it might clear, decide **whether the payment fully reconciles those open invoices** — the
boolean Mahnwesen gate that must be true before dunning is suppressed (or, if false, by how much it falls
short and which invoices remain open). The closed-form `is_reconciled` computation is AXIS-A (L5 R-P4); the
ambiguous core is the *attribution* when the payment amount does not cleanly equal one invoice: partial
payment across several invoices, Skonto (early-discount) tolerance, rounding/exchange residue, or an
ambiguous reference that could apply to multiple open items. Output is a boolean
"fully-reconciles?" plus the implied invoice set, with NARS `(frequency, confidence)`; woa-rs uses it
only to gate Mahnwesen escalation (Iron Rule 7), never to auto-post a reconciliation.

## Deterministic guard (AXIS-A — stays in woa-rs)
The two independent status flags are closed-form (`account_payment.py:L453-497`, L5 R-P4):
**`is_matched`** (bank side, K5) = liquidity-line residual zero OR journal uses its own
`default_account_id` directly (L490-493); **`is_reconciled`** (invoice side, K3) = all
`account.reconcile=True` counterpart/write-off lines have zero residual. The currency selector
(`amount_residual` if `pay.currency==company.currency` else `amount_residual_currency`,
`account_payment.py:L488`, L5 gotcha #4) and the payment state machine
(`in_process -> paid` when `liquidity_residual==0` or liquidity account `reconcile=False`,
`account_payment.py:L453-467`, L5 gotcha #5) stay in woa-rs. **Mahnwesen must gate on `is_reconciled`,
NOT `is_matched`** (L5 R-P4, gotcha — a payment can be bank-acknowledged yet invoice-open). The savant is
invoked only for the ambiguous attribution (which open invoices a partial/over/under payment clears).

## Slot 1 — Evidence (Arrow EvidenceRef)
Two correlated tables. The payment's counterpart line
`EvidenceRef { table: "account_payment.counterpart_line", schema_fingerprint, rows }` (one row):

| column | dtype | signal |
|---|---|---|
| `payment_id` | `Int64` | the payment whose clearing is being gated |
| `partner_id` | `Int64` | scopes which open invoices are candidates |
| `amount_residual` | `Decimal128` | company-currency residual on the counterpart line; zero ⇒ `is_reconciled` (L5 R-P4) |
| `amount_residual_currency` | `Decimal128` | foreign-currency residual; the selected axis when `pay.currency != company.currency` (L5 gotcha #4) |
| `currency_id` | `Int64` | selects which residual axis gates the boolean |
| `account_reconcile` | `Boolean` | whether the counterpart account is reconcilable (only such lines count toward `is_reconciled`, L5 R-P4) |
| `date` | `Date32` | payment date — Skonto-window check (`discount_date`) and rate snapshot |
| `memo` / `payment_reference` | `Utf8` | reference text linking the payment to a Belegnummer (ambiguity source) |

Candidate open invoices `EvidenceRef { table: "account_move.open_invoices", ... }` (one row per open invoice for the partner):

| column | dtype | signal |
|---|---|---|
| `move_id` | `Int64` | the open invoice candidate |
| `amount_residual` | `Decimal128` | still-open amount; the sum the payment must (fully or partially) match |
| `amount_total` / `amount_residual_currency` | `Decimal128` | full and foreign-currency open amounts |
| `invoice_date` / `invoice_date_due` | `Date32` | aging + Skonto-eligibility (`discount_date = date_ref + discount_days`, L5 R-P2) |
| `discount_date` | `Date32`/nullable | Skonto cutoff; a payment short by exactly the Skonto % but within the window still "fully reconciles" (L5 R-P2/P3) |
| `discount_percentage` | `Decimal128`/nullable | the early-discount % (DE `included`, NL `excluded`, BE `mixed` — L5 R-P3) |
| `name` | `Utf8` | Belegnummer for reference matching against the payment memo |
| `payment_state` | `Utf8` | `not_paid`/`partial`/`in_payment`/`paid` echo (post-hook transition target, L2 R-14) |

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- `is_reconciled` vs `is_matched` (independent flags; Mahnwesen gates on `is_reconciled`) <- `L5-PAY-TERMS-MATCH.md:373-396` (R-P4; `account_payment.py:L453-497`).
- residual currency selector (`amount_residual` vs `amount_residual_currency`) <- `L5-PAY-TERMS-MATCH.md:389, 656` (R-P4 + gotcha #4; `account_payment.py:L488`).
- payment state machine + `not any(liquidity.account_id.reconcile)` direct-bank path <- `L5-PAY-TERMS-MATCH.md:153-166, 658` (R-P1/R-P4 + gotcha #5; `account_payment.py:L453-467`).
- Skonto window + discount modes (`discount_date`, `included/excluded/mixed`) feeding "short-but-in-window still clears" <- `L5-PAY-TERMS-MATCH.md:237-263, 345-365` (R-P2/R-P3; `account_payment_term.py:L200-214, L82-90`).
- Mahnwesen reads `skonto_bis`/open-item residual before escalating <- `L5-PAY-TERMS-MATCH.md:387, 641-644` (R-P4 + K-step map).
- both-residuals-zero reconcile rule reused for the gate <- `L2-K3-RECON.md:66-89` (L2 R-1) and `L5-PAY-TERMS-MATCH.md:384-387` (R-P4).
- delegation tuple + `Other(RECONCILE_MATCH)` (id 21 shares the code with id 19) <- `SAVANTS.md:79` (roster) and `savants.rs:82` (`other_kind::RECONCILE_MATCH`).

## Slot 3 — Property-level alignment
**N/A — class-level pivots only; no `owl:equivalentProperty` defined.** `odoo_alignment.rs` holds only
class-level `owl:equivalentClass` rows and **zero** property IRIs (`odoo_alignment.rs:14, 60-68`).
`account.payment` is itself **unmapped** (L5 FLAG, proposed `fibo-FBC-PAS-FPAS:Payment`, not present —
`L5-PAY-TERMS-MATCH.md:188-190, 586-601`), which is why `family=None`. The open invoices map class-level
(`account.move -> fibo:Transaction`, `odoo_alignment.rs:222-229`) but the boolean-gate decision
traverses no property and crosses no FIBO/SKR/ZUGFeRD seam. Never invent IRIs.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the payment counterpart line + the partner's candidate open invoices (slot 1).

-> Conclusion C = `PaymentFullyReconciles(payment_id, {move_id…}) : bool` — the Mahnwesen gate, plus the
implied cleared-invoice set — emitted with NARS `(frequency, confidence)` where:
- **frequency** of the "fully reconciles" hypothesis is high when the payment residual (in the
  currency-selected axis) nets the candidate invoices' residuals to within `currency.is_zero`, or falls
  short by exactly the Skonto amount while `date <= discount_date` (L5 R-P2/P3). It is partial/low when
  the payment only covers some invoices, or over-pays (a credit remains).
- **frequency** drops with reference ambiguity (memo matches several open Belegnummern) or a currency
  mismatch where only one residual axis zeroes (L5 gotcha #4 — both must be considered).
- **confidence** is the NARS weight from agreeing features (exact amount AND reference AND single
  unambiguous invoice ⇒ high; amount-only across many candidates ⇒ low); capped by phi-1.

Discriminating features (ranked): residual-net-to-zero on the currency-selected axis >> Skonto-window
tolerance (`discount_date`/`discount_percentage`) > `memo`/`payment_reference` Belegnummer match >
single-vs-multiple candidate invoices. Induction here is "payments like this — amount, reference, timing
— have historically fully cleared this invoice set." The boolean directly gates whether Mahnwesen
escalation is suppressed.

## Parity / GoBD notes
Suggestion-only (Iron Rule 7): the savant returns a gate value; woa-rs's deterministic `is_reconciled`
(L5 R-P4) remains the authority that suppresses or fires Mahnwesen, and the actual partial/full reconcile
is posted by the AXIS-A `reconcile()` arithmetic (L2 R-2/R-8), never by the savant. The Skonto-on-VAT
modes are German-tax load-bearing: in `included` mode (DE default, L5 R-P3) the discount reduces both net
and VAT proportionally, so a "fully reconciles within Skonto" verdict implies a VAT correction the AXIS-A
guard must book — the savant flags the case but never books it. Gate on `is_reconciled`, never
`is_matched` (a bank-matched-but-invoice-open payment that wrongly suppresses dunning is the canonical
parity bug, L5 R-P4).
