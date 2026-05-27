# Savant: AutopostRecommender  (id 17 · family 0x81 · lane L1)

**Tuple:** kind=PostingAnomaly · inference=Induction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** `PostingAnomalyReasoner`   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::PostingAnomaly` -> "abduce the most likely cause from the evidence trail";
> here the *inductive* facet ("things-like-X") rides the same impl, selecting `QueryStrategy::CamWide`
> via `InferenceType::Induction::default_strategy()`. Style Analytical inherited from 0x81 SmbFoundryInvoice.

## What it decides (AXIS-B core)
Given a vendor (`partner_id`) whose recent incoming bills (`in_invoice`/`in_refund`) have been posted
**without manual modification**, decide whether to **recommend enabling automatic posting** for that
vendor's future bills. Odoo's hard rule is "3+ consecutive unmodified bills" (`_show_autopost_bills_wizard`),
but the AXIS-B core generalises the count into an inductive trust signal: how strongly does this
vendor's history support unattended posting, given streak length, recency, amount stability, and
whether any bill in the window was touched. Output is a recommend/withhold suggestion with NARS
`(frequency, confidence)`; woa-rs only *offers the wizard*, never flips the flag itself.

## Deterministic guard (AXIS-A -- stays in woa-rs)
The closed-form trigger `_show_autopost_bills_wizard()` -- count consecutive unmodified bills from the
same partner, fire the wizard at >= 3 (`L1-K3-POST.md:182-188, 809-820`; `account_move.py:L5838-5875`,
called from `action_post`, `L1-K3-POST.md:159-188`). The streak counter, the `is_manually_modified`
flag (set/cleared via the `skip_is_manually_modified` context, `L1-K3-POST.md:43-47`), and the >=3
threshold stay deterministic in woa-rs.

## Slot 1 -- Evidence (Arrow EvidenceRef)
Primary table `account_move` filtered to one vendor's bill history, ordered by `(invoice_date, id)`,
`EvidenceRef { table: "account_move.vendor_bill_history", schema_fingerprint, rows }`:

| column | dtype | signal |
|---|---|---|
| `move_id` | `Int64` | bill identity |
| `partner_id` | `Int64` | the vendor the recommendation is scoped to (`commercial_partner_id` after the post-time partner sync, L1 step 9) |
| `move_type` | `Utf8` (`in_invoice\|in_refund`) | restricts the window to *bills* (purchase documents) |
| `state` | `Utf8` | only `posted` bills count toward the unmodified streak |
| `is_manually_modified` | `Boolean` | the core discriminator -- `true` resets the streak; the inductive prior is "long run of false" |
| `invoice_date` | `Date32` | recency + ordering; recent unmodified bills weigh more than stale ones |
| `amount_total` | `Decimal128` | amount stability across the streak (low variance => stronger "safe to autopost") |
| `auto_post` | `Utf8` (`no\|at_date\|monthly\|...`) | current autopost state; if already automated the recommendation is moot |
| `invoice_user_id` | `Int64` | whether the same handler processed the run (consistency signal) |
| `partner_supplier_rank` | `Int64` | vendor maturity (incremented on post, L1 step 14) -- a high rank corroborates an established relationship |

Streak feature derived in woa-rs (AXIS-A) but passed as a column for the reasoner: `unmodified_run_len`
(consecutive `is_manually_modified=false` ending at the latest posted bill).

## Slot 2 -- Odoo field -> signal map                 (cite L-doc file:lines)
- recommendation trigger (3+ consecutive unmodified) <- `L1-K3-POST.md:182-188` and `L1-K3-POST.md:809-820` (Axis-2.A; `account_move.py:L5838-5875`).
- `is_manually_modified` semantics + `skip_is_manually_modified` context patch during post <- `L1-K3-POST.md:43-47` (R K3-1 step 1).
- `auto_post` value space (`no`/`at_date`/`monthly`/`quarterly`/`yearly`) and future-date deferral flip <- `L1-K3-POST.md:84-119` (R K3-1 steps 5/8).
- `state` posting gate (only `posted` moves count) <- `L1-K3-POST.md:127-131` (R K3-1 step 11).
- `partner_id` normalised to `commercial_partner_id` at post <- `L1-K3-POST.md:121-123` (R K3-1 step 9).
- `supplier_rank` increment on post <- `L1-K3-POST.md:142-144` (R K3-1 step 14).
- contract tuple `(PostingAnomaly, Induction, NarsTruth, Analytical)` <- `L1-K3-POST.md:813-818` (Axis-2.A).

## Slot 3 -- Property-level alignment
Decision stays **within family 0x81 SmbFoundryInvoice**. The evidence is entirely
`account.move`-internal (a vendor's own bill stream); `account.move` -> `fibo:Transaction`
(`odoo_alignment.rs`, class-level). The `partner_id` reference touches 0x80
(`res.partner -> fibo:LegalEntity`) only as an identity key for grouping, not as a traversed
property. **N/A -- stays within 0x81** (no FIBO/SKR/ZUGFeRD property seam is crossed).

## Slot 4 -- AXIS-B decision in evidence terms
Let E = the vendor's ordered bill-history rows (slot 1), with derived `unmodified_run_len`.

-> Conclusion C = `RecommendAutopost(partner_id)` emitted with NARS `(frequency, confidence)` where:
- **frequency** rises with: longer `unmodified_run_len` (>= 3 is the odoo floor, but evidence beyond 3
  raises it further), low `amount_total` variance across the streak, recent `invoice_date`s, a single
  consistent `invoice_user_id`, and a healthy `supplier_rank`.
- **frequency** collapses if any bill in the window has `is_manually_modified=true` (streak break) or
  the vendor's amounts are erratic.
- **confidence** is the NARS evidence weight w = n/(n+k): more posted, unmodified bills => higher
  confidence; a short history keeps confidence low even at frequency 1.0. Capped by the phi-1 ceiling.

Discriminating features (ranked): `unmodified_run_len` >> `amount_total` stability > recency of
`invoice_date` > `invoice_user_id` consistency > `supplier_rank`. Induction here is literally
"bills like this vendor's recent bills have been safe to post unattended."

## Parity / GoBD notes
The recommendation only enables *automatic posting of future bills* -- each such post still runs the
full balanced-double-entry guard (R K3-3) and lock-date checks (R K3-1 step 6); autopost never
bypasses validation. Suggestion-only per Iron Rule 7: woa-rs may surface the wizard but the tenant
must confirm the `auto_post` flip. No GoBD Festschreibung interaction (hashing happens at/after post
regardless of how the post was triggered).
