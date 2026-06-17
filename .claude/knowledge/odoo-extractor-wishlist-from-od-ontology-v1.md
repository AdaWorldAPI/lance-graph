# odoo-extractor-wishlist-from-od-ontology-v1 — what the consumer needs from the SPO emitter

> **READ BY:** anyone touching `tools/odoo-blueprint-extractor/` or the
> `crates/lance-graph/src/graph/spo/odoo_ontology.{rs,spo.ndjson}` emit
> path; the ClassView design session driving `classid → ClassView` /
> composition / inheritance / recompute-ordering; `prior-art-savant` on
> Odoo enrichments.
>
> **Status:** FINDING (per `od-ontology::RecomputeDag` probe shipped at
> [`AdaWorldAPI/odoo-rs@d8a270d`][odoo-rs-probe]).
> **Authored:** 2026-06-17.
> **Companion docs:** `odoo-extraction-strategies-v1.md` (the three
> proposer legs), `odoo-extraction-tools-v1.md` (what backs each leg),
> `odoo-blueprint-inventory-v1.md` (the 66-entity corpus).

[odoo-rs-probe]: https://github.com/AdaWorldAPI/odoo-rs/blob/main/crates/od-ontology/src/recompute_dag.rs

---

## 0. The doctrine in one paragraph

`AdaWorldAPI/odoo-rs` is the downstream consumer of the SPO corpus this
repo's Odoo extractor produces. It lowers the corpus into native
SurrealDB DDL (`DEFINE TABLE` / `FIELD` / `FUNCTION` / `EVENT`) — the
ontology becomes the database, not an ORM. Its wishlist (the
canonical source-of-truth for what the consumer needs) lives at:

> **[`UPSTREAM_WISHLIST.md`](https://github.com/AdaWorldAPI/odoo-rs/blob/main/crates/od-ontology/specs/UPSTREAM_WISHLIST.md)**

This knowledge doc names the *highest-certainty, lowest-cost*
enrichment the consumer needs from the extractor — surfaced by a
shipped corpus-side probe — so the next session touching the emitter
sees it without re-deriving the analysis.

---

## 1. The ask, narrowed to one sentence

**When `@api.depends('<relation>.<leaf>')` is declared, emit
`(<method>, reads_field, <target_model>.<leaf>)` as a sibling to the
existing `(<method>, reads_field, <model>.<relation>)` triple.**

Same wire shape as the existing predicate. Same provenance source
(`@api.depends` AST node). Same emitter call site. One new triple per
declared relation-path read.

---

## 2. Why now — the probe finding

`od-ontology` shipped a topological-sort + cycle-detection module
([`RecomputeDag`][odoo-rs-probe]) on 2026-06-17 — 540 LOC, 8 tests,
clippy-clean, with a `MethodKind` filter (`Compute` / `Check` /
`Onchange` / `Inverse` / `Search` / `Other` by underscore-prefix) so
the projection-relevant subgraph can be queried in isolation.

Two findings against the slice 1 / slice 2 corpora:

1. **`MethodKind::Compute` subsets ARE acyclic in the current
   corpus.** The audit-found P0 cycle in `_compute_amount.md`
   (`move._compute_amount` → `line.reconciled` →
   `line._compute_amount_residual`) is **STRUCTURALLY INVISIBLE** to
   the consumer-side DAG. The corpus emits a single triple —
   `(_compute_amount, reads_field, account_move.line_ids)` — naming
   the relation, NOT the transitive `account_move_line.reconciled`
   read declared by `@api.depends('line_ids.reconciled')`. The
   extractor surfaces the relation hop; the leaf read is implicit.

2. **The corpus DOES carry a real cycle** — between two `_onchange_*`
   methods in slice 1 (`_onchange_invoice_vendor_bill` ↔
   `_onchange_quick_edit_total_amount`, both emit + read
   `invoice_line_ids`). This is a *legitimate* Odoo UI cooperative
   loop (the form view manages re-fire), filtered out by
   `MethodKind::Compute`. Demonstrates `MethodKind` filtering is the
   correct invariant, and the consumer's machinery is working.

The machinery is shipped; the corpus needs one more triple.

---

## 3. Concrete shape

```text
# Today — the extractor emits:
{"s":"odoo:account_move._compute_amount","p":"reads_field",
 "o":"odoo:account_move.line_ids","f":0.85,"c":0.75}

# Proposed — also emit a sibling per `@api.depends` leaf:
{"s":"odoo:account_move._compute_amount","p":"reads_field",
 "o":"odoo:account_move_line.reconciled","f":0.85,"c":0.75}
{"s":"odoo:account_move._compute_amount","p":"reads_field",
 "o":"odoo:account_move_line.amount_residual","f":0.85,"c":0.75}
```

Source (the AST node carrying the truth) is the same node the existing
`reads_field` triple already reads: `@api.depends('line_ids.reconciled',
'line_ids.amount_residual', …)`. The current emitter takes the FIRST
hop (`line_ids` → `account_move.line_ids`); the proposed emitter ALSO
takes the LEAF (`reconciled` → `account_move_line.reconciled`,
resolving the target model via the One2many's `comodel_name`).

**Confidence + frequency**: same as the current `reads_field`
(`f=0.85, c=0.75`) — it's the same `@api.depends` declaration with one
more level of AST walk, not a new evidence source.

---

## 4. What this is NOT asking for

- **Not a new predicate.** Same `reads_field` wire shape; just more
  rows.
- **Not a ClassView design.** The recompute-DAG sort is consumer-side,
  shipped, working. The ClassView session can ratify the ordering
  later without affecting this enrichment.
- **Not a Frankenstein-flatten of `traverses_relation`.**
  `traverses_relation` is `(method) → (model.field)` — names which
  methods walk which relations. The new `reads_field` sibling is
  `(method) → (target_model.leaf)` — names what was *read* through
  that walk. Distinct shapes, distinct uses.
- **Not retroactive on the existing 22 245 triples.** Forward-only;
  if + when the extractor re-runs, the new triples land alongside.
  Existing consumers are forward-compatible (`reads_field` set grows,
  never shrinks).

---

## 5. Companion ratifications from the broader corpus

Two `AdaWorldAPI/ruff` PRs merged 2026-06-17 firm up cross-language
predicate conventions worth knowing about when this enrichment lands:

- **ruff#19** routes Rails STI parent to `inherits_from` — the same
  predicate the C++ harvester emits — making `inherits_from` the
  **cross-language canonical inheritance predicate**. The Odoo
  extractor should adopt the same shape for `_inherit`. See the
  wishlist's P1 `_inherit` item; not blocking this ask, but the
  natural pair landing.
- **ruff#21** introduces `Predicate::ValidationKind`
  (`(attribute_iri, validation_kind, "presence"|"uniqueness"|…)`).
  The Odoo analog is per-`@api.constrains` method. See the
  wishlist's P2 ask.

The deep-`reads_field` ask in §3 is P0; the other two are P1/P2 from
the same wishlist.

---

## 6. Source citations the consumer relies on

- **Wishlist**: <https://github.com/AdaWorldAPI/odoo-rs/blob/main/crates/od-ontology/specs/UPSTREAM_WISHLIST.md>
  (the P0 section names this exact ask + the PROBE FINDING block under it)
- **Probe** (cycle detector that surfaces the gap):
  <https://github.com/AdaWorldAPI/odoo-rs/blob/main/crates/od-ontology/src/recompute_dag.rs>
- **Spec that originally found the cycle by hand**:
  <https://github.com/AdaWorldAPI/odoo-rs/blob/main/crates/od-ontology/specs/_compute_amount.md>
  (see § "MISSED-1 (P0)")

---

## 7. Why this file lives here, not in odoo-rs

odoo-rs's `UPSTREAM_WISHLIST.md` is the canonical wishlist. This file
is a **session breadcrumb on the producer side**: it names the ask
+ the supporting probe + the source where the consumer's machinery
is built, so the next session touching the Odoo extractor sees the
context without round-tripping to the consumer repo. Append-only;
no edits to the wishlist's content here. If the wishlist's P0
framing changes (it might — the consumer notes it as a moving
status), this doc gets a dated update entry below, not an in-place
rewrite.
