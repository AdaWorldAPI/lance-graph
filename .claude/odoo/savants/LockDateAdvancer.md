# Savant: LockDateAdvancer  (id 18 · family 0x81 · lane L1)

**Tuple:** kind=PostingAnomaly · inference=Abduction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** `PostingAnomalyReasoner`   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::PostingAnomaly` -> "abduce the most likely cause from the evidence trail"
> (`examples/savant_dispatch.rs:31`). Abduction -> `QueryStrategy::DnTreeFull`. Style Analytical
> inherited from 0x81 SmbFoundryInvoice.

## What it decides (AXIS-B core)
A move's `date` falls in a **locked period** (one or more of the five lock dates is violated). The
fully-deterministic answer is "advance to last_violation + 1 day" (L1 step 6 / L11 R19), and woa-rs
ships that as AXIS-A. The AXIS-B core is the *ambiguous* residual: when **multiple lock types** are
in play (fiscalyear vs tax vs sale/purchase vs hard) and the journal/tax facets interact, abduce
**which open period the move most likely belongs in** -- i.e. is the right target the next day after
the binding lock, or a later period implied by the move's economic substance (tax-relevant date,
sale vs purchase journal scope, period-reset boundary)? Output: a suggested accounting period/date
with NARS `(frequency, confidence)`; woa-rs applies the deterministic +1day unless it adopts the
suggestion behind its guard.

## Deterministic guard (AXIS-A -- stays in woa-rs)
`_get_violated_lock_dates` + `_get_accounting_date` -> `move.date = last_violation + 1 day`
(`L1-K3-POST.md:98-106` R K3-1 step 6, and `L1-K3-POST.md:822-830` Axis-2.B; `account_move.py:L5637-5641`).
The lock-date taxonomy, per-journal applicability, and the +1day bump are deterministic:
`L11 R11` (5 lock types, `company.py:L59-114`), `R15` (journal-aware violation sweep, general journal
NOT subject to sale/purchase locks, `company.py:L713-729`), `R19` (`_get_accounting_date` period-reset
capping, `account_move.py:L6570-6607`), `R16/R17` (post/copy/unlink guards).

## Slot 1 -- Evidence (Arrow EvidenceRef)
Two correlated tables. Primary `EvidenceRef { table: "account_move.lock_context", schema_fingerprint, rows }`
(the move under post) joined to the company lock-date row:

| column | dtype | signal |
|---|---|---|
| `move_id` | `Int64` | the move being advanced |
| `date` | `Date32` | the requested accounting date (the one violating a lock) |
| `journal_type` | `Utf8` (`sale\|purchase\|cash\|bank\|credit\|general`) | gates which locks apply: sale-lock iff `sale`, purchase-lock iff `purchase`; `general` exempt from both (L11 R15) |
| `affects_tax_report` | `Boolean` | whether `tax_lock_date` is in scope (`_affect_tax_report`, L1 step 6) |
| `has_tax` | `Boolean` | move carries tax lines -> tax-lock relevance |
| `move_type` | `Utf8` | sale vs purchase document substance (corroborates journal_type) |
| `number_reset` | `Utf8` (`month\|year\|never`) | sequence reset family -> period-end capping target (L11 R19) |

Companion `EvidenceRef { table: "res_company.lock_dates", ... }` (one row, the move's company resolved over `parent_ids`):

| column | dtype | signal |
|---|---|---|
| `fiscalyear_lock_date` | `Date32`/nullable | global soft lock |
| `tax_lock_date` | `Date32`/nullable | entries-with-tax lock |
| `sale_lock_date` | `Date32`/nullable | sale-journal lock |
| `purchase_lock_date` | `Date32`/nullable | purchase-journal lock |
| `hard_lock_date` | `Date32`/nullable | irreversible lock (`user_hard_lock_date` = max over parents, L11 R11) |

## Slot 2 -- Odoo field -> signal map                 (cite L-doc file:lines)
- date-in-locked-period auto-advance (`move.date = _get_accounting_date(...)`) <- `L1-K3-POST.md:98-106` (R K3-1 step 6) and `L1-K3-POST.md:822-830` (Axis-2.B).
- five lock-date fields + `SOFT_LOCK_DATE_FIELDS` + `user_hard_lock_date` <- `L11-COA-JOURNALS-LOCKDATES.md:69-71` (R11; `company.py:L59-114`).
- journal-aware violation sweep (sale=journal.type==sale; purchase=...; general exempt) <- `L11-COA-JOURNALS-LOCKDATES.md:85-87` (R15; `company.py:L713-729`, `account_move.py:L6609-6616`).
- `_get_accounting_date` base = last_violation+1day, sale + number_reset capping to month/year end <- `L11-COA-JOURNALS-LOCKDATES.md:101-103` (R19; `account_move.py:L6570-6607`).
- soft-lock + per-user exception resolution (`account.lock_exception`) <- `L11-COA-JOURNALS-LOCKDATES.md:73-79` (R12/R13; `company.py:L597-663`).
- `affects_tax_report` gating tax-lock in scope <- `L1-K3-POST.md:101-102` (R K3-1 step 6) and `L11-COA-JOURNALS-LOCKDATES.md:89-91` (R16; `account_move.py:L2796-2813`).
- borderline classification note (multi-factor tax/fiscal/hard has judgment aspects) <- `L1-K3-POST.md:824-830` (Axis-2.B).

## Slot 3 -- Property-level alignment
Decision stays **within family 0x81 SmbFoundryInvoice** (the move) reading 0x62-resident company
lock-date config as plain scalar evidence. `account.move` -> `fibo:Transaction`; `res.company` ->
`fibo:LegalEntity` (both class-level in `odoo_alignment.rs`). The lock dates are date scalars on the
company, not a traversed property graph. PROPOSED only if a fiscal-period ontology is later added:
`odoo:fiscalyear_lock_date -> fibofnd:hasAccountingPeriodBoundary` (no such property axiom exists
today). For the AXIS-B decision: **N/A -- stays within 0x81 reading 0x62 scalars**.

## Slot 4 -- AXIS-B decision in evidence terms
Let E = the move lock-context row + the company lock-date row (slot 1); let
`V = { (lock_date, field) : lock_date >= move.date, field in-scope }` be the violated set.

-> Conclusion C = `AdvanceToPeriod(target_date)` emitted with NARS `(frequency, confidence)` where:
- the deterministic baseline target is `max(V.lock_date) + 1 day`. The savant abduces whether a
  **later** target is more economically faithful, driven by: `number_reset` (sale + monthly =>
  month-end cap, L11 R19), `affects_tax_report`+`has_tax` (a tax-relevant move may belong in the next
  tax period, not merely the next day), and whether the binding lock is `hard` (irreversible -- no
  exception can pull it back, L11 R12/R18) vs a soft lock that a `lock_exception` could cover.
- **frequency** of the "next-day-after-binding-lock" hypothesis is high when only one soft lock binds
  and the move is journal-scope-clear; it drops when tax-period substance or period-reset capping
  argues for a later month/year boundary.
- **confidence** rises with how cleanly `journal_type`/`move_type` agree and how few lock types
  overlap; multi-lock overlap (fiscal AND tax AND hard) lowers confidence. Capped by phi-1.

Discriminating features (ranked): which lock type is the binding `max(V)` (hard >> tax >> fiscalyear
>> sale/purchase) > `number_reset` capping > `affects_tax_report`+`has_tax` > `journal_type` scope.
Abduction selects the most parsimonious open period consistent with the violated-lock set.

## Parity / GoBD notes
`hard_lock_date` is irreversible (cannot be unset or decreased, L11 R18) -- a move dated into a hard
lock can **only** move forward; the savant must never suggest a target <= any hard lock. Advancing the
date is a silent mutation in odoo but in woa-rs it is suggestion-only (Iron Rule 7): the tenant's
guard applies the deterministic +1day by default and only adopts a later suggested period
explicitly. GoBD fiscal-year lock (Festschreibung of closed periods) is the legal backstop; the
savant never re-opens a locked period, it only chooses among **open** targets.
