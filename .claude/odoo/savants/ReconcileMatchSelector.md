# Savant: ReconcileMatchSelector  (id 19 · family None · lane L2)

**Tuple:** kind=Other(RECONCILE_MATCH=5) · inference=Induction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** the single `Other(5)` reasoner   (RECONCILE_MATCH; shared with id 21 PaymentToInvoiceMatcher — see cross-savant note)

> dispatch: `ReasoningKind::Other(5)` -> "match open items / bank lines by evidence fusion (reconcile)"
> (`examples/savant_dispatch.rs:34-36`, the `5 | 6 =>` arm). Induction -> `QueryStrategy::CamWide`
> (`InferenceType::Induction::default_strategy()`). Style Analytical — woa-rs assigns it from the
> SMBAccounting basin even though `family=None` (no alignment axiom yet; the reconcile classes are
> ontology-unmapped, L2 FLAG-1/FLAG-2).

> **CROSS-SAVANT (RECONCILE_MATCH=5 is shared by id 19 and id 21).** The single `Other(5)` impl must
> distinguish the two by `ReasoningContext.namespace` + evidence shape, **NOT** by code:
> - **id 19 (this savant)** — `namespace = "erp.k3.reconcile_match"`; evidence = a *set* of open items on
>   ONE reconcilable account; output = **propose a candidate grouping** (a subset-of-ids to reconcile).
> - **id 21 (PaymentToInvoiceMatcher)** — `namespace = "erp.k3.payment_reconcile"`; evidence = one payment's
>   counterpart line + its target open invoices; output = a **boolean gate** "does this payment fully
>   reconcile?" (Mahnwesen).
> Same `Other(5)`, two ReasoningContexts; the impl branches on namespace/evidence, never on the code.

## What it decides (AXIS-B core)
Given a set of OPEN (non-`reconciled`) journal items on a single reconcilable account (one partner or
several), propose **which subset of open items should be grouped as a reconciliation candidate** —
i.e. which debits clear which credits. The deterministic arithmetic of *executing* a chosen pairing is
entirely AXIS-A (L2 R-7/R-8); the ambiguous core is the *selection*: when amounts do not match
1:1, when several open items sum to clear one counterpart, or when reference/partner/date proximity
must break ties among many candidates. Output is a proposed grouping (subset of AML ids) with NARS
`(frequency, confidence)`; woa-rs only *proposes* it — a human or the AXIS-A `reconcile()` entry point
applies it (Iron Rule 7), never an un-guarded write.

## Deterministic guard (AXIS-A — stays in woa-rs)
Everything in L2 R-1..R-14 is closed-form and stays in woa-rs: residual computation
`_compute_amount_residual` (`account_move_line.py:L793-861`, L2 R-1), the eligibility validation
`_check_amls_exigibility_for_reconciliation` (one account, one root company, not already reconciled,
account `reconcile=True` or cash/credit-card — `account_move_line.py:L2609-2644`, L2 R-5), the
FIFO/maturity sort key `(date_maturity or date, currency_id, amount_currency, balance)`
(`account_move_line.py:L2678-2684`, L2 R-7), the debit/credit pairing loop and single-partial amount
math (`account_move_line.py:L2194-2567`, L2 R-7/R-8), and the matching-number union-find
(`account_partial_reconcile.py:L171-215`, L2 R-11). The savant is invoked only for the residual
**candidate-selection** case (L2 R-15, "HEURISTIC — Axis 2"): which open items to even offer as a
group, when the closed-form set is ambiguous.

## Slot 1 — Evidence (Arrow EvidenceRef)
Primary table `account_move_line` projected to the open-items-on-one-account schema (one row per open
item in the candidate window), `EvidenceRef { table: "account_move_line.open_items", schema_fingerprint, rows }`:

| column | dtype | signal |
|---|---|---|
| `move_line_id` | `Int64` | identity of the open item (the unit the proposal groups) |
| `account_id` | `Int64` | the single reconcilable account partition key (L2 R-5: all candidates share one account) |
| `partner_id` | `Int64`/nullable | partner-proximity feature; same-partner items are far likelier to reconcile together (L2 R-7 multi-partner sort) |
| `balance` | `Decimal128` | company-currency signed amount (`debit-credit`); sign splits the debit vs credit pools to pair |
| `amount_residual` | `Decimal128` | **the core matching axis** — the still-open company-currency amount; a candidate group's residuals must net toward zero (L2 R-1) |
| `amount_currency` | `Decimal128` | foreign-currency signed amount; multi-currency grouping needs this axis too |
| `amount_residual_currency` | `Decimal128` | foreign-currency residual; reconciled requires BOTH residuals zero (L2 R-1, gotcha #1) |
| `currency_id` | `Int64` | reconciliation-currency axis (L2 R-8 step 1); cross-currency groups split per currency (L2 R-5) |
| `date_maturity` | `Date32`/nullable | the FIFO axis (falls back to `date`); proximity in maturity raises co-grouping likelihood (L2 R-7) |
| `date` | `Date32` | posting date; date proximity is a tie-break feature |
| `name` / `ref` | `Utf8` | Belegnummer / reference text — reference-overlap is a strong same-transaction signal (fuzzy Beleg match is explicitly Axis-2, L2 R-15) |
| `matching_number` | `Utf8`/nullable | already-`P<id>` partial items can still join a group (L2 R-5); fully-matched (`reconciled`) are excluded upstream |

Discriminating window: open rows for one `account_id` (and usually one `partner_id`), capped by
`Budget.max_evidence_rows`; debit pool (`amount_residual > 0`) and credit pool (`< 0`) are the two
sides the proposal pairs across.

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- `amount_residual` / `amount_residual_currency` / `reconciled` (both-zero rule) <- `L2-K3-RECON.md:66-89` (R-1; `account_move_line.py:L793-861`).
- candidate eligibility (one account, one root company, not reconciled, `reconcile=True` or cash/credit-card) <- `L2-K3-RECON.md:172-181` (R-5; `account_move_line.py:L2609-2644`).
- FIFO/maturity sort key `(date_maturity or date, currency_id, amount_currency, balance)` + multi-partner partner-sort <- `L2-K3-RECON.md:164-168, 306-313` (R-5/R-7; `account_move_line.py:L2678-2684, L2585-2586`).
- reconciliation-currency selection + full/partial match detection (`compare_amounts`, `min_recon_amount`) <- `L2-K3-RECON.md:327-357` (R-8; `account_move_line.py:L2239-2285`).
- `matching_number` encoding (`P<partial>` partial can re-enter, plain-int = full, excluded) <- `L2-K3-RECON.md:569-610` (R-11; `account_partial_reconcile.py:L171-215`).
- the candidate-selection HEURISTIC delegation tuple + `Other("ReconcileMatch")` + namespace `erp.k3.reconcile_match` <- `L2-K3-RECON.md:753-792` (R-15 / R-Axis2; reconcile-model lives in lane L5).

## Slot 3 — Property-level alignment
**N/A — class-level pivots only; no `owl:equivalentProperty` defined.** Confirmed: `odoo_alignment.rs`
holds only class-level `owl:equivalentClass` rows and **zero** property IRIs (the file's leg-1 doc,
`odoo_alignment.rs:14, 60-68`). Moreover the reconcile classes themselves are **unmapped**:
`account.partial.reconcile` and `account.full.reconcile` resolve to `None` (L2 FLAG-1/FLAG-2,
`L2-K3-RECON.md:821-865`; `resolve_odoo("account.reconcile.model").is_none()` asserted at
`odoo_alignment.rs:502`) — which is why this savant carries `family=None`. The AML rows it groups DO
map class-level (`account.move.line -> fibo:JournalEntryLine`, `odoo_alignment.rs:232-239`) but the
*decision* traverses no property and crosses no FIBO/SKR/ZUGFeRD seam. Never invent IRIs.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the open-items window (slot 1) for one `account_id`, split into a debit pool D
(`amount_residual > 0`) and a credit pool C (`amount_residual < 0`).

-> Conclusion C = `ProposeReconcileGroup(account_id, {move_line_id…})` — a subset of ids whose residuals
net toward zero — emitted with NARS `(frequency, confidence)` where:
- **frequency** of a proposed grouping rises with: exact residual-sum cancellation (Σ debit residual ≈
  −Σ credit residual within `currency.is_zero` tolerance), same `partner_id` across the group, strong
  `ref`/`name` Belegnummer overlap, and tight `date_maturity` proximity (FIFO-adjacent).
- **frequency** falls when: residuals only partially cancel (a partial, not a full group), candidates
  span multiple partners or currencies with no reference tie, or many equal-amount candidates compete
  (ambiguous 1:N).
- **confidence** is the NARS weight from the number of independently-agreeing features (amount AND
  reference AND partner AND date all pointing to the same group ⇒ high; amount-only ⇒ low) and scales
  with window completeness; capped by the phi-1 humility ceiling.

Discriminating features (ranked): residual-sum cancellation in `recon_currency` >> `partner_id` identity
> `ref`/`name` reference overlap > `date_maturity` FIFO proximity > `currency_id` agreement. Induction
here is "open items like these — same partner, matching residual, shared Beleg — have historically
reconciled as a group." The chosen group is then handed to the AXIS-A `reconcile()` / R-8 arithmetic.

## Parity / GoBD notes
The proposal is suggestion-only (Iron Rule 7): woa-rs offers a candidate grouping; the deterministic
`reconcile()` entry point (L2 R-2) and its balanced/sync guards (L2 R-4) actually post the partials —
the savant never writes `account.partial.reconcile`/`account.full.reconcile` rows itself. Reconciliation
is not itself a GoBD Festschreibung event (the underlying moves are already posted/festgeschrieben);
but a *wrong* auto-grouping that the user blindly accepts would mis-state Offene-Posten aging, so the
human-confirm gate is the control. The both-residuals-zero rule (L2 R-1, gotcha #1) is the load-bearing
correctness check the AXIS-A guard re-verifies after any proposed group is applied.
