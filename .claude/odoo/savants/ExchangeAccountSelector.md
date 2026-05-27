# Savant: ExchangeAccountSelector  (id 7 · family 0x62 · lane L12)

**Tuple:** kind=Other(CHART_ACCOUNT_MAPPING=3) · inference=Deduction · semiring=Boolean · style=Analytical
**Feeds Reasoner impl:** the `Other(3)` reasoner   (CHART_ACCOUNT_MAPPING)

> dispatch: `ReasoningKind::Other(3)` -> the `Other(code)` default arm "domain-specific Other(code)
> reasoner" (`examples/savant_dispatch.rs:34, 37`; only codes 5|6 take the reconcile arm). Deduction ->
> `QueryStrategy::CamExact`. Semiring `Boolean` — the gain/loss pick is a hard sign predicate, not a
> graded belief. Style Analytical inherited from 0x62 SMBAccounting.

## What it decides (AXIS-B core)
When reconciliation produces an exchange-rate difference, decide **which gain/loss account books it**.
The pick itself is sign-driven and fully deterministic (open-balance sign > 0 ⇒ loss/expense account,
< 0 ⇒ gain/income account — L12 R18); the genuinely ambiguous core is the **initial SKR chart
configuration**: which concrete SKR03/SKR04 account number plays the role of
`expense_currency_exchange_account_id` / `income_currency_exchange_account_id` for a given chart, when
the company has not yet configured them. This savant crosses the SKR seam (odoo exchange account →
`fibo:Account` + SKR chart concept) — the one place in this lane-group where Slot 3 is not a flat N/A.
Output is the suggested SKR gain/loss account pair with NARS `(frequency, confidence)`; woa-rs applies it
only as a config default a human ratifies (Iron Rule 7).

## Deterministic guard (AXIS-A — stays in woa-rs)
The runtime account pick is deterministic and stays in woa-rs: `sign = compare(open_balance, 0)`;
`amount > 0 -> company.expense_currency_exchange_account_id` (loss); else
`company.income_currency_exchange_account_id` (gain) — `account_move.py:L5218-5237` + `company.py:L135-145`
(L12 R18), mirrored by `_get_exchange_account` in the reconcile path
(`account_move_line.py:L2952-2955`, L2 R-12, L2-K3-RECON.md:627-633). The exchange-diff move itself
(two balancing lines, date `max(aml.date, journal.accounting_date)`, posted on
`company.currency_exchange_journal_id`) is closed-form double-entry (L2 R-12). The runtime config
prerequisites — exchange journal + both gain/loss accounts must be set, else `UserError` — are hard guards
(L2 R-12 "Config requirements"). The savant is invoked only for the *initial SKR account assignment* when
those config fields are empty.

## Slot 1 — Evidence (Arrow EvidenceRef)
The exchange-difference context `EvidenceRef { table: "account_move.exchange_diff_context", schema_fingerprint, rows }` (one row per diff):

| column | dtype | signal |
|---|---|---|
| `move_id` | `Int64` | the move/partial the exchange diff attaches to |
| `open_balance` | `Decimal128` | the residual whose **sign** deterministically picks loss (>0) vs gain (<0) — L12 R18 |
| `company_id` | `Int64` | the company whose SKR chart + exchange-account config applies |
| `currency_exchange_journal_id` | `Int64`/nullable | configured exchange journal (NULL ⇒ config gap the savant flags) |
| `expense_currency_exchange_account_id` | `Int64`/nullable | the **loss** account; NULL ⇒ needs SKR assignment (the AXIS-B case) |
| `income_currency_exchange_account_id` | `Int64`/nullable | the **gain** account; NULL ⇒ needs SKR assignment |
| `chart_template` | `Utf8`/nullable | which SKR chart (skr03 / skr04) — selects the candidate account-number set |

Candidate SKR account corpus `EvidenceRef { table: "skr_chart.exchange_account_candidates", ... }`
(the chart's gain/loss account numbers): NEEDS-INPUT (see Slot 2 / Slot 3) — the concrete SKR03/04
exchange gain/loss account codes are not present in the L-docs.

| column | dtype | signal |
|---|---|---|
| `skr_code` | `Utf8` | SKR account number (e.g. an "Erträge/Aufwendungen aus Währungsumrechnung" account) — **NEEDS-INPUT** |
| `role` | `Utf8` (`gain`\|`loss`) | which exchange role the SKR account fills |
| `chart` | `Utf8` (`skr03`\|`skr04`) | chart partition |

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- sign-driven gain/loss pick (`sign=compare(open_balance,0)`, expense=loss / income=gain) <- `L12-MULTICOMPANY-CURRENCY.md:79-81` (R18; `account_move.py:L5218-5237`, `company.py:L135-145`).
- `_get_exchange_account` (amount>0 ⇒ `expense_currency_exchange_account_id` else `income_...`) <- `L2-K3-RECON.md:627-633` (R-12; `account_move_line.py:L2952-2955`).
- exchange-diff move structure + date `max(aml.date, journal.accounting_date)` + posted on `currency_exchange_journal_id` <- `L2-K3-RECON.md:619-673` (R-12; `account_move_line.py:L2957-3098`).
- runtime config prerequisites (journal + both accounts required, else UserError) <- `L2-K3-RECON.md:667-671` (R-12 "Config requirements").
- delegation tuple `Other("ChartAccountMapping")` + Deduction + Boolean + Analytical (deterministic sign; heuristic only for initial SKR config) <- `L12-MULTICOMPANY-CURRENCY.md:79-81` (R18 savant seed) and `savants.rs:68` (`other_kind::CHART_ACCOUNT_MAPPING=3`).
- **NEEDS-INPUT: SKR03/04 exchange gain/loss account codes** — the concrete account numbers are absent from L12 and from the SKR seed; only the class-level pivot exists (see Slot 3).

## Slot 3 — Property-level alignment
This savant genuinely crosses the SKR seam — odoo `account.account` (the exchange gain/loss accounts) →
`fibo:Account` AND the SKR03/04 chart concept. The class-level pivots DO exist:
`account.account -> fibo:Account` (`odoo_alignment.rs:242-249`) and the SKR chart concept
`account.account.template -> fibo:Account` labelled `ogit.SMBAccounting:SkrAccount`
(`odoo_alignment.rs:252-259`, provenance "SKR03/04 chart concept => fibo:Account"). **However, no
property IRI exists** to relate an odoo company's `expense_currency_exchange_account_id`/
`income_currency_exchange_account_id` to a specific SKR gain/loss account, and the concrete SKR account
candidates are not enumerated in the L-docs:

**NEEDS-INPUT: SKR03/04 exchange gain/loss account codes + (property IRI not yet defined in repo).**

`odoo_alignment.rs` holds only class-level `owl:equivalentClass` rows and **zero** property IRIs
(`odoo_alignment.rs:14, 60-68`); the gain/loss-role-to-SKR-account relation would be a property axiom that
must be authored upstream — do not invent it. Until then the savant can name the *role* (gain/loss) from
the deterministic sign but cannot resolve the concrete SKR number.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the exchange-diff context row (slot 1); let `s = compare(open_balance, 0)`.

-> Conclusion C, two facets:
1. **Runtime role (deterministic, Boolean):** `ExchangeAccountRole = if s > 0 { Loss } else { Gain }` —
   frequency 1.0, confidence at ceiling; this is the AXIS-A pick restated, the savant adds nothing.
2. **Initial SKR assignment (the AXIS-B core):** `SuggestSkrExchangeAccount(chart, role) -> skr_code`
   emitted with NARS `(frequency, confidence)` where **frequency** rises with how canonically a candidate
   SKR account is the standard Währungsumrechnung gain/loss account for that chart, and **confidence**
   reflects corroboration across the chart template / prior tenant config. Because the candidate set is
   **NEEDS-INPUT** (no SKR codes in the L-docs, no property IRI), the reasoner can currently emit only the
   role with high confidence and the concrete `skr_code` with confidence floored to "insufficient
   evidence" until the SKR account corpus is supplied.

Discriminating features (ranked): `open_balance` sign (role, deterministic) >> `chart_template` (skr03 vs
skr04 candidate set) > prior tenant config of the exchange accounts. The Boolean semiring fits facet 1
exactly; facet 2 is the residual config-assist the tuple flags as the only heuristic part.

## Parity / GoBD notes
The exchange-diff posting is an accounting identity (the residual that must balance the books once an FX
rate moves between invoice and payment) — it is a real posted move subject to K11 Festschreibung
(L12 open-question #6). Suggestion-only (Iron Rule 7): woa-rs books the diff via the deterministic
sign-driven pick on the *configured* accounts; the savant never invents an account at posting time — if
the SKR exchange accounts are unset it raises the config gap (mirroring odoo's `UserError`, L2 R-12) for a
human to resolve, rather than guessing a `skr_code`. A wrong gain/loss account mis-states the P&L
(Aufwand vs Ertrag), so the human-ratify gate on the initial SKR assignment is the control.
