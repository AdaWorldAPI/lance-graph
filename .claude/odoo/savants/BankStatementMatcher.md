# Savant: BankStatementMatcher  (id 20 · family None · lane L5)

**Tuple:** kind=Other(BANK_STATEMENT_MATCH=6) · inference=Induction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** the `Other(6)` reasoner   (BANK_STATEMENT_MATCH; distinct code from the RECONCILE_MATCH=5 savants)

> dispatch: `ReasoningKind::Other(6)` -> "match open items / bank lines by evidence fusion (reconcile)"
> (`examples/savant_dispatch.rs:34-36`, the `5 | 6 =>` arm). Induction -> `QueryStrategy::CamWide`.
> Style Analytical (SMBAccounting basin); `family=None` — `account.reconcile.model` is ontology-unmapped
> (L5 FLAG, proposed `sh:NodeShape`/`sh:rule`; `resolve_odoo("account.reconcile.model").is_none()`
> asserted at `odoo_alignment.rs:502`).

> **Code note:** id 20 is the ONLY savant on `Other(BANK_STATEMENT_MATCH=6)` — its impl is selected by
> the code itself (unlike the shared `Other(5)`), though it shares the `5 | 6 =>` dispatch arm's
> "reconcile by evidence fusion" approach. It is the reconcile-MODEL-rule matcher (bank line + write-offs),
> whereas the `Other(5)` pair work on already-posted AML/payment residuals.

## What it decides (AXIS-B core)
Given an incoming bank statement line (amount, label/narration, partner IBAN, date, journal) and the set
of configured `account.reconcile.model` rules plus candidate open items, decide **which reconcile-model
rule matches the bank line and what write-off lines it generates** — i.e. categorise the bank line and
propose the contra-postings. Odoo community applies the model dimensions as hard AND-filters in greedy
`sequence, id` order and takes the first match (`account_reconcile_model.py:L94`); the AXIS-B core
replaces that brittle first-match with multi-dimensional evidence fusion: a label-regex hit raises
frequency, a partner match raises confidence, an amount-in-range raises both, and ambiguous lines (no
single high-confidence model) fall back to ranking candidate open items. Output is the best-fit model id
(or none) + its write-off line spec, with NARS `(frequency, confidence)`; woa-rs applies it via the
existing `bank_match.rs` engine as a *proposal* (`trigger='manual'`) unless `auto_reconcile` and the
tenant confirms (Iron Rule 7).

## Deterministic guard (AXIS-A — stays in woa-rs)
The hard-match core stays in woa-rs `src/erp/engine/bank_match.rs` (Sprint-3b, confidence levels
100/80/60/40/0, L5 R-P5): exact Belegnummer extraction, IBAN match, exact-amount equality — the
deterministic ~40% of the reconcile-model surface. The four model dimensions, when set, are themselves
deterministic hard filters (`account_reconcile_model.py:L123-145`, L5 R-P5): journal-in-set
(`match_journal_ids`), amount `lower|greater|between` (`match_amount*`), label `contains|not_contains|
match_regex` (`match_label*`, regex validated at save L149-156), partner-in-set (`match_partner_ids`).
The `can_be_proposed` / `mapped_partner_id` computes (`account_reconcile_model.py:L158-167`, L5 gotcha
#11) and the write-off `amount_type` arithmetic (`fixed|percentage|percentage_st_line|regex`,
`account_reconcile_model.py:L25-51`) are deterministic. The savant is invoked for the *fusion across
dimensions* and the *greedy-first-match-is-ambiguous* residual (L5 R-P5 Axis-2).

## Slot 1 — Evidence (Arrow EvidenceRef)
Three correlated tables. The bank line `EvidenceRef { table: "bank_statement_line.match_context", schema_fingerprint, rows }` (one row):

| column | dtype | signal |
|---|---|---|
| `st_line_id` | `Int64` | the bank statement line to categorise |
| `amount` | `Decimal128` | matched against `match_amount` `lower/greater/between` window (L5 R-P5 dim 2) |
| `currency_id` | `Int64` | line currency |
| `label` | `Utf8` | narration / SVWZ text — the primary textual axis for `contains`/`regex` (L5 R-P5 dim 3) |
| `partner_iban` | `Utf8`/nullable | partner identification; backs the `match_partner_ids` filter (L5 R-P5 dim 4) |
| `partner_id` | `Int64`/nullable | resolved partner (hard partner filter) |
| `date` | `Date32` | value date — aging + candidate-window |
| `journal_id` | `Int64` | gates `match_journal_ids` (empty set = all journals, L5 R-P5 dim 1) |

Candidate reconcile-model rules `EvidenceRef { table: "account_reconcile_model.rules", ... }` (one row per configured model):

| column | dtype | signal |
|---|---|---|
| `model_id` | `Int64` | rule identity (the conclusion to pick) |
| `sequence` | `Int32` | odoo greedy priority (`_order='sequence, id'`, L5 R-P5) — a prior, not a hard winner under fusion |
| `match_journal_ids` | `List<Int64>` | journal hard-filter set (empty = all) |
| `match_amount` / `match_amount_min` / `match_amount_max` | `Utf8`/`Decimal128` | amount-window dimension |
| `match_label` / `match_label_param` | `Utf8` | `contains`/`not_contains`/`match_regex` + its parameter (L5 R-P5 dim 3) |
| `match_partner_ids` | `List<Int64>` | partner hard-filter set |
| `trigger` | `Utf8` (`manual`\|`auto_reconcile`) | enforcement level (auto-apply vs propose), L5 R-P5 |
| `mapped_partner_id` | `Int64`/nullable | if set, this is a partner-MAPPING lookup, NOT a reconcile candidate (exclude, L5 gotcha #11) |

Write-off line sub-model `EvidenceRef { table: "account_reconcile_model_line.writeoff", ... }` (one row per model line):

| column | dtype | signal |
|---|---|---|
| `model_id` | `Int64` | parent rule |
| `account_id` | `Int64` | contra/categorisation account for the write-off |
| `amount_type` | `Utf8` (`fixed`\|`percentage`\|`percentage_st_line`\|`regex`) | how the write-off amount is computed (L5 R-P5 sub-model) |
| `amount_string` | `Utf8` | authoritative amount/regex source (`amount` float is a cached compute, L5 gotcha #12) |
| `tax_ids` | `List<Int64>` | taxes on the write-off amount |
| `label` | `Utf8` | generated line label |

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- four match dimensions (journal / amount / label / partner) as configured filters <- `L5-PAY-TERMS-MATCH.md:407-451` (R-P5; `account_reconcile_model.py:L123-145`).
- label match modes `contains|not_contains|match_regex` + save-time regex validation <- `L5-PAY-TERMS-MATCH.md:417-422` (R-P5; `account_reconcile_model.py:L135-143, L149-156`).
- greedy `_order='sequence, id'` first-match (the brittleness the savant replaces) <- `L5-PAY-TERMS-MATCH.md:454, 668` (R-P5 + gotcha #10; `account_reconcile_model.py:L94`).
- `can_be_proposed` / `mapped_partner_id` (partner-mapping models are NOT reconcile candidates) <- `L5-PAY-TERMS-MATCH.md:431-452, 670` (R-P5 + gotcha #11; `account_reconcile_model.py:L158-167`).
- write-off `amount_type` (`fixed`/`percentage`/`percentage_st_line`/`regex`) + `amount_string` authoritative <- `L5-PAY-TERMS-MATCH.md:456-468, 672` (R-P5 + gotcha #12; `account_reconcile_model.py:L8-89`).
- delegation tuple `Other("BankStatementMatch")` + Induction(+Abduction fallback) + NarsTruth + Analytical, namespace `erp.bank_statement.match` <- `L5-PAY-TERMS-MATCH.md:476-541` (R-P5 Axis-2).
- existing woa-rs `bank_match.rs` covers ~40% (hard match); savant adds configurable-pattern + write-off + auto-trigger fusion <- `L5-PAY-TERMS-MATCH.md:543-556` (R-P5 mapping).

## Slot 3 — Property-level alignment
**N/A — class-level pivots only; no `owl:equivalentProperty` defined.** `odoo_alignment.rs` holds only
class-level `owl:equivalentClass` rows and **zero** property IRIs (`odoo_alignment.rs:14, 60-68`).
`account.reconcile.model` is **unmapped** (proposed `sh:NodeShape`/`sh:rule`, not present —
`L5-PAY-TERMS-MATCH.md:558-563`; explicit `is_none()` assertion at `odoo_alignment.rs:502`), so
`family=None`. The bank line and its open-item candidates touch `account.move.line ->
fibo:JournalEntryLine` (`odoo_alignment.rs:232-239`) class-level only; the rule-matching decision
traverses no property IRI and crosses no FIBO/SKR/ZUGFeRD seam. Never invent IRIs.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the bank line + the configured reconcile-model rules (minus `mapped_partner_id` partner-mapping
rows) + write-off sub-lines (slot 1).

-> Conclusion C = `MatchReconcileModel(st_line_id, Option<model_id>, {writeoff_line…})` emitted with NARS
`(frequency, confidence)` where:
- **frequency** of a model match rises with the number of its set dimensions that the bank line
  satisfies — label `contains`/`regex` hit, amount in `between` window, partner in set, journal in set —
  fused (NOT hard-ANDed): each agreeing dimension lifts frequency rather than gating it.
- **confidence** rises with the count of *independent* dimensions agreeing (label AND partner AND amount
  ⇒ high; label-only ⇒ low) and with the model's `sequence` prior; when no model clears the confidence
  floor the reasoner switches to the **Abduction fallback** (L5 R-P5) — "what open item best explains this
  line?" — ranking candidate open items instead of picking a model. Capped by phi-1.
- the write-off line spec rides along once a model is chosen: its `amount_type` arithmetic
  (`percentage_st_line` of the line amount, `regex` extraction from `label`) stays AXIS-A; the savant only
  decides *which* model (hence which write-off template) applies.

Discriminating features (ranked): label `match_regex`/`contains` hit >> partner (IBAN/`match_partner_ids`)
match > amount-in-window > journal-in-set > `sequence` prior. The intentional improvement over odoo is
that the highest-truth match may differ from the greedy first-in-`sequence` match (L5 gotcha #10).

## Parity / GoBD notes
Suggestion-only (Iron Rule 7): woa-rs's `bank_match.rs` applies the chosen model as a *proposal*
(`trigger='manual'`); `auto_reconcile` exists as a stored flag but its auto-application is Enterprise and
requires explicit tenant opt-in (L5 E3). The deterministic hard match (Belegnummer / IBAN / exact amount)
stays in Rust and the write-off arithmetic re-runs there; the savant never posts the contra-entry itself.
The bank-vs-invoice distinction matters downstream: matching a statement line (K5) is *not* the same as
clearing the open item (K3) — Mahnwesen gates on the K3 `is_reconciled` (see sibling id 21
PaymentToInvoiceMatcher), not on a successful bank-line categorisation here.
