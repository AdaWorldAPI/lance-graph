# Savant: AnalyticModelScorer  (id 5 · family None · lane L10)

**Tuple:** kind=CustomerCategory · inference=Deduction · semiring=HammingMin · style=Analytical
**Feeds Reasoner impl:** `CustomerCategoryReasoner`   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::CustomerCategory` -> "classify against the family codebook (deductive lookup)"
> (`examples/savant_dispatch.rs:29`). Deduction -> `QueryStrategy::CamExact`. Semiring `HammingMin`
> (each matching criterion adds a bit; the minimum-sufficient match wins) — note this is the only
> reconcile/FX-family-adjacent savant NOT on NarsTruth. `family=None` —
> `account.analytic.distribution.model` is ontology-unmapped (`resolve_odoo(...).is_none()` asserted at
> `odoo_alignment.rs:503, 648`).

> **Sibling note (id 4 AnalyticDistributionSuggester).** id 4 (NextBestAction · Induction · NarsTruth)
> does the *inductive multi-rule FUSION* that proposes a distribution before scoring; **this savant (id 5)**
> does the *deductive priority-scored single-WINNER* selection once the scoring weights are fixed. They
> feed different Reasoner impls (CustomerCategory vs NextBestAction) and must not be merged — see
> `AnalyticDistributionSuggester.md` Slot 4 sibling paragraph.

## What it decides (AXIS-B core)
Given a move line's match arguments (product, product category, partner, partner categories, account-code
prefix, company) and the corpus of `account.analytic.distribution.model` rules, decide **which
distribution model wins** — i.e. classify the line into the highest-scoring applicable rule bucket. The
scoring criteria themselves (+1 per matching dimension; −1 to exclude on a hard mismatch) and the greedy
"first model wins per plan, later models skip already-filled plans" fill are deterministic algorithms; the
ambiguous core is **how to weight the criteria when scores tie or only partially match** (product vs
company precedence, prefix-vs-category). Output is the selected `model_id` (per plan) with NARS
`(frequency, confidence)`; woa-rs applies it only as the default the per-plan 100% guard then validates
(Iron Rule 7).

## Deterministic guard (AXIS-A — stays in woa-rs)
The deterministic algorithm is L10 R8: `_get_applicable_models` prefix filter (model `account_prefix`
split by `;`/`,`, incoming `account.code` must `startswith` any, `account_analytic_distribution_model.py:
L34-48`), `_get_score` (+1 per matching criterion, −1 on mismatch — `account_analytic_plan.py:L59-76`,
L10 R9), the `sequence`-ordered greedy fill, and the "skip a model whose plans are already covered"
rule (`test_model_sequence:L456-468`, L10 R8). The per-plan 100%-sum validation
(`_validate_analytic_distribution`, `account_move_line.py:L3146-3177`, L10 R4) and the archived-account
block (L10 R12) are hard guards wrapping the selection. The savant supplies only the weighting judgment
when the deterministic scores are tied/partial.

## Slot 1 — Evidence (Arrow EvidenceRef)
Two tables. The line's match arguments
`EvidenceRef { table: "account_move_line.distribution_args", schema_fingerprint, rows }` (one row = the line being classified):

| column | dtype | signal |
|---|---|---|
| `move_line_id` | `Int64` | the line being classified into a rule bucket |
| `product_id` | `Int64`/nullable | product criterion (the highest-precedence match axis, L10 R9 "product takes precedence over company") |
| `product_categ_id` | `Int64`/nullable | category criterion (`_get_score` +1, L10 R8/R9) |
| `partner_id` | `Int64`/nullable | partner criterion |
| `partner_category_ids` | `List<Int64>` | partner-tag criterion |
| `account_code` | `Utf8` | `account_id.code` — prefix-startswith axis (L10 R8 `_get_applicable_models`) |
| `company_id` | `Int64` | company criterion (only scored when the model HAS a company, L10 R9) |
| `business_domain` | `Utf8` (`invoice`\|`bill`\|`general`) | gates which applicabilities apply (L10 R4/R9) |

Candidate model corpus `EvidenceRef { table: "account_analytic_distribution_model", ... }` (the rules to score):

| column | dtype | signal |
|---|---|---|
| `model_id` | `Int64` | rule identity (the conclusion) |
| `analytic_distribution` | `Utf8` (JSON) | the distribution this rule contributes (`{csv-of-ids: pct}`, L10 R1) |
| `sequence` | `Int32` | priority within a score tier (lower wins; greedy fill order, L10 R8) |
| `partner_id` | `Int64`/nullable | rule partner constraint (NULL = unconstrained) |
| `product_id` | `Int64`/nullable | rule product constraint |
| `product_categ_id` | `Int64`/nullable | rule category constraint |
| `account_prefix` | `Utf8`/nullable | rule prefix constraint (`;`/`,` split, startswith) |
| `company_id` | `Int64`/nullable | rule company constraint |

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- `_get_applicable_models` prefix filter + `_get_default_search_domain_vals` (False = no constraint) <- `L10-ANALYTIC.md:368-383` (R8; `account_analytic_distribution_model.py:L34-48`).
- `_get_score` criteria (+1 per match, −1 exclude; product>company precedence) <- `L10-ANALYTIC.md:423-452` (R9; `account_analytic_plan.py:L59-76`, `test_applicability_score:L405-431`).
- greedy `sequence`-ordered "first model wins per plan, skip already-filled plans" <- `L10-ANALYTIC.md:368-419` (R8; `test_model_sequence:L433-468`, `test_model_score:L232-258`).
- `analytic_distribution` JSON shape (`{csv-of-account-ids: pct}`, cross-plan keys) <- `L10-ANALYTIC.md:48-77` (R1; `account_move_line.py:L418-420`).
- per-plan 100% validation (NOT global) + skipped account types + archived-account block <- `L10-ANALYTIC.md:185-225, 486-504` (R4/R12; `account_move_line.py:L3146-3177, L2011-2034`).
- delegation tuple `name=AnalyticModelScorer family=None reasoning=CustomerCategory inference=Deduction semiring=HammingMin style=Analytical` <- `L10-ANALYTIC.md:413-419` (R8 AXIS-B seed line).

## Slot 3 — Property-level alignment
**N/A — class-level pivots only; no `owl:equivalentProperty` defined.** `odoo_alignment.rs` holds only
class-level `owl:equivalentClass` rows and **zero** property IRIs (`odoo_alignment.rs:14, 60-68`).
`account.analytic.distribution.model` is **ontology-unmapped** (family `None`, `L10-ANALYTIC.md:41`;
`resolve_odoo("account.analytic.distribution.model").is_none()` asserted at `odoo_alignment.rs:503, 648`),
which is why this savant carries `family=None` and needs a Layer-2 alignment axiom (lance-graph follow-on,
the same axiom the sibling id 4 needs). The analytic accounts it routes to map class-level
(`account.analytic.account -> fibo:Account` cost sub-type, `L10-ANALYTIC.md:37`) but the scoring decision
traverses no property IRI and crosses no FIBO/SKR/ZUGFeRD seam. Never invent IRIs.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the line's match-argument row + the applicable model corpus (slot 1), pre-filtered by the AXIS-A
prefix/score guard to the candidates with `_get_score >= 0`.

-> Conclusion C = `SelectDistributionModel(move_line_id, plan_id -> model_id)` — the winning model per
root plan — emitted with NARS `(frequency, confidence)` where:
- because the semiring is **HammingMin**, each matching criterion is a set bit and the
  minimum-sufficient (most-specific clearing-the-threshold) model wins; **frequency** is highest for the
  model whose set criteria are a superset of the more general candidates (product+categ+prefix beats
  company-only).
- **confidence** rises when the top model's score strictly dominates the runner-up (a clear winner) and
  falls when scores tie within a `sequence` tier (the genuinely ambiguous case the savant exists for —
  it then leans on `sequence` and criterion-specificity to break the tie).
- the greedy per-plan fill stays AXIS-A: the savant ranks/picks the winner per plan; the guard applies it
  plan-by-plan, never overwriting an already-filled plan (L10 R8).

Discriminating features (ranked): `product_id` match >> `product_categ_id` match > `account_code` prefix
match > `partner_id`/partner-category match > `company_id` match (the L10 R9 precedence). Deduction here is
"given fixed scoring weights, the highest scorer IS the bucket" — the weight definition is the only
heuristic facet, which is why this is the deductive single-winner counterpart to id 4's inductive fusion.

## Parity / GoBD notes
Analytic distribution is Kostenrechnung (cost accounting), a parallel non-financial dimension — **not**
part of the GoBD double-entry ledger (L10 R3: analytic amount is `-balance` in company currency, a derived
view). Suggestion-only (Iron Rule 7): the per-plan 100%-mandatory validation (L10 R4) and the
archived-account hard block (L10 R12) remain woa-rs guards the selected model must satisfy before any post;
draft lines never persist analytic lines (L10 R5), so the selection materialises only at/after posting.
