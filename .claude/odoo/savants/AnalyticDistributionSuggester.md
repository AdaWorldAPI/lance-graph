# Savant: AnalyticDistributionSuggester  (id 4 · family 0x62 · lane L10)

**Tuple:** kind=NextBestAction · inference=Induction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** `NextBestActionReasoner`   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::NextBestAction` -> "induce the action with the highest expected value"
> (`examples/savant_dispatch.rs:32`). Induction -> `QueryStrategy::CamWide`. Style Analytical
> inherited from 0x62 SMBAccounting.

## What it decides (AXIS-B core)
For a `display_type == 'product'` move line (or any line on a non-invoice move), suggest the
**cost-centre (analytic) distribution** -- the `{ "<analytic_account_id_csv>": percentage }` JSON --
that this line should carry, given its context (product, product category, partner, partner
categories, account code prefix, company), and *which root plans are still unallocated*. This is the
inductive "lines like this combination have historically used distribution X" decision. Output is a
suggested `analytic_distribution` map with NARS `(frequency, confidence)`; woa-rs writes it only as a
default the user can override (the odoo compute uses `... or line.analytic_distribution`, never
forcing).

## Deterministic guard (AXIS-A -- stays in woa-rs)
The argument assembly, the `frozendict` per-arguments cache, the `display_type=='product'` /
non-invoice guard, and the `related | model` dict-merge with `or existing` fallback are deterministic
(`L10-ANALYTIC.md:298-357` R7 AXIS-A part; `account_move_line.py:L1217-1248`). Per-plan
100%-sum validation at post (`_validate_analytic_distribution`, `L10-ANALYTIC.md:185-225` R4) and the
archived-account block (R12) are deterministic guards wrapping the suggestion.

## Slot 1 -- Evidence (Arrow EvidenceRef)
Two tables. The *query line context* `EvidenceRef { table: "account_move_line.analytic_context", schema_fingerprint, rows }`
(one row = the line needing a suggestion):

| column | dtype | signal |
|---|---|---|
| `move_line_id` | `Int64` | the line identity |
| `product_id` | `Int64` | primary match axis (distribution-model criterion) |
| `product_categ_id` | `Int64` | category fallback axis (model `_get_score` +1) |
| `partner_id` | `Int64` | partner match axis |
| `partner_category_ids` | `List<Int64>` | partner-tag match axis (`category_id.ids`) |
| `account_code` | `Utf8` | `account_id.code` -- prefix-match axis (model `account_prefix` startswith) |
| `company_id` | `Int64` | company scoping |
| `display_type` | `Utf8` | guard echo (`product` vs structural) |
| `related_root_plan_ids` | `List<Int64>` | plans already allocated by `_related_analytic_distribution` (SO/PO carry-over) -> the reasoner must NOT re-suggest these |

The *candidate corpus* `EvidenceRef { table: "account_analytic_distribution_model", ... }` (the rules to induce over):

| column | dtype | signal |
|---|---|---|
| `model_id` | `Int64` | rule identity |
| `analytic_distribution` | `Utf8` (JSON) | the candidate distribution this rule would apply |
| `sequence` | `Int32` | priority (lower wins; greedy plan-fill order) |
| `partner_id` | `Int64`/nullable | rule's partner constraint (NULL = unconstrained) |
| `product_id` | `Int64`/nullable | rule's product constraint |
| `product_categ_id` | `Int64`/nullable | rule's category constraint |
| `account_prefix` | `Utf8`/nullable | rule's account-prefix constraint (`;`/`,` split, startswith) |
| `company_id` | `Int64`/nullable | rule's company constraint |

## Slot 2 -- Odoo field -> signal map                 (cite L-doc file:lines)
- `_compute_analytic_distribution` reactive compute + merge + `or existing` fallback <- `L10-ANALYTIC.md:298-357` (R7; `account_move_line.py:L1217-1248`).
- `_get_analytic_distribution_arguments` dict (product_id, product_categ_id, partner_id, partner_category_id, account_prefix, company_id, related_root_plan_ids) <- `L10-ANALYTIC.md:321-331` (R7; `account_move_line.py` arg-assembly region).
- candidate model fields + `_get_applicable_models` prefix filter + `_get_score` (+1 per matching criterion) + greedy "first model wins per plan" <- `L10-ANALYTIC.md:368-419` (R8; `account_analytic_distribution_model.py:L34-48`, tests test_model_score/test_model_sequence).
- `analytic_distribution` JSON shape (`{ "<csv-of-account-ids>": pct }`, cross-plan keys) <- `L10-ANALYTIC.md:48-77` (R1; `account_move_line.py:L418-420`).
- per-plan 100% rule (NOT global sum) + skipped account types + `validate_analytic` gating <- `L10-ANALYTIC.md:185-225` (R4; `account_move_line.py:L3146-3177, L2011-2034`).
- delegation tuple `(NextBestAction, Induction, NarsTruth, Analytical)` + savant seed line <- `L10-ANALYTIC.md:358-364` (R7 AXIS-B).

## Slot 3 -- Property-level alignment
Decision stays **within family 0x62 SMBAccounting** on the line side, but reaches the
**ontology-unmapped** `account.analytic.distribution.model` (family `None`,
`L10-ANALYTIC.md:41`) and `account.analytic.account` (mapped to `fibo:Account` cost sub-type, 0x62,
`L10-ANALYTIC.md:37`). No FIBO/SKR/ZUGFeRD seam is crossed for the suggestion itself -- the
distribution-model class needs a **Layer-2 alignment axiom** (lance-graph follow-on, flagged in
SAVANTS.md "Unmapped (None) classes"). Property-level alignment is **N/A today** (no axiom exists);
when the distribution-model class is aligned, the traversed relation would be
`odoo:analytic_distribution -> <cost-allocation property>` -- PROPOSED, not present.
NEEDS-INPUT: the Layer-2 family + alignment axiom for `account.analytic.distribution.model` (and the
shared sibling `AnalyticModelScorer` id 5) -- this is lance-graph-side work, not sourceable from L10.

## Slot 4 -- AXIS-B decision in evidence terms
Let E = the line-context row + the candidate distribution-model corpus (slot 1), minus models whose
plans are in `related_root_plan_ids`.

-> Conclusion C = `SuggestDistribution(move_line_id, { csv_key: pct, ... })` emitted with NARS
`(frequency, confidence)` where:
- candidate models are ranked by match strength (partner / product / product_categ / account_prefix /
  company agreement -- the `_get_score` evidence). Multiple matching rules **fuse** under NarsTruth
  rather than a single hard winner: agreement across several rules on the same plan raises frequency.
- **frequency** of a proposed `(plan -> account, pct)` assignment rises with the number and
  specificity of matching criteria (product+categ+prefix beats company-only) and with historical
  co-occurrence of that distribution for similar lines.
- **confidence** is the NARS weight from the count of corroborating rules/observations; a single
  weakly-matching model yields low confidence even at frequency 1.0. Capped by phi-1.
- the greedy "first-filled plan wins" stays AXIS-A; the savant only ranks/weights the candidates the
  guard then applies plan-by-plan, and never re-suggests an already-`related` plan.

Discriminating features (ranked): product_id match >> product_categ_id match > account_code prefix
match > partner_id / partner_category match > company match. Sibling note: id 5 `AnalyticModelScorer`
(CustomerCategory/Deduction/HammingMin) does the *deductive priority-scored single-winner* selection
once weights are fixed; this savant does the *inductive multi-rule fusion* that proposes the
distribution before scoring -- they feed different Reasoner impls (NextBestAction vs CustomerCategory).

## Parity / GoBD notes
Analytic distribution is Kostenrechnung (cost accounting), **not** part of the GoBD double-entry
ledger -- analytic lines are a parallel, non-financial dimension (L10 R3: sign is `-balance` in
company currency, a derived view). The suggestion is suggestion-only (Iron Rule 7): the per-plan
100%-mandatory validation (R4) and archived-account block (R12) remain hard woa-rs guards that the
suggestion must satisfy before any post. Draft lines never persist analytic lines (R5), so the
suggestion materialises only at/after posting.
