# Evaluation: Foundry Workshop low-code on Elixir abstraction, compiled to Rust

> **READ BY:** agents working the Elixir frontend, Foundry-shape codegen,
> Workshop/A2UI surface, or the SPO ontology consumer path.
>
> **Status:** EVALUATION (2026-05-28, opus-4-8 as SavantPattern). Grounded in
> read files (cited file:line); honest about gaps. Companion to
> `.claude/knowledge/semantic-operational-handbook-v0.1.md` (the primitive
> catalogue) and the now-verified SPO ontology loader
> (`crates/lance-graph/src/graph/spo/odoo_ontology.rs`, 4 tests green).
>
> **Deterministic only.** No LLM, no similarity, no inference. The Workshop
> low-code layer resolves names against a typed ontology + dispatches
> compile-time recipes; the runtime is Rust + ractor, not a model.

## The question

Can Foundry Workshop's "apps over ontology" low-code model sit on top of an
**Elixir-syntax abstraction layer** whose **holy grail is compile-to-Rust
underneath**? Where does the mapping hold, where does it break?

## The three layers, concretely

```
  Workshop low-code surface   (what a domain expert authors)
        │  view/action/widget bound to ontology
        ▼
  Elixir-syntax abstraction   (readable DSL — NOT BEAM at runtime)
        │  tree-sitter-elixir parse → typed resolve against ontology
        ▼
  Rust compiled underneath    (the holy grail — executable semantics)
        │  recipe → KernelHandle → ractor mailbox → SPO store
        ▼
  SPO ontology substrate       (what names resolve against)
        odoo_ontology.spo.ndjson → SpoStore (22 245 triples, VERIFIED)
```

The bottom layer is **built and tested as of this session**. The top two are
partially specced (`lance-graph-elixir-frontend-v1.md`) and partially gapped
(below).

## What HOLDS (grounded in read files)

### 1. The ontology substrate exists and is queryable — deterministically

`crates/lance-graph/src/graph/spo/odoo_ontology.rs` (this session, 4 tests
green) loads 22 245 triples into the existing `SpoStore`. The Foundry
primitives map 1:1 onto triple classes:

| Foundry (Workshop)     | SPO triple class                                  | count |
| ---                    | ---                                               | ---:  |
| Object Type            | `(odoo:<family>, rdf:type, ogit:ObjectType)`      | 388   |
| Property               | `(odoo:<fam>.<field>, rdf:type, ogit:Property)`   | 3 107 |
| Function               | `(odoo:<fam>.<fn>, rdf:type, ogit:Function)`      | 3 328 |
| Function dependency    | `(field, depends_on, dep)`                        | 6 309 |
| Function output        | `(field, emitted_by, fn)`                         | 3 228 |
| Link / traversal       | `(fn, traverses_relation, rel)`                   | 11    |
| Guard signal           | `(fn, raises, exc:<Type>)`                        | 451   |

The "**a + b → c through d?**" query — Foundry's core Function-resolution
semantics — is a deterministic graph deduction over `depends_on` +
`emitted_by`, NOT a similarity search. Verified working:
`account_move.amount_total emitted_by _compute_amount, depends_on
{line_ids.balance, line_ids.amount_residual, currency_id, …}`.

### 2. The Elixir-as-surface decision is already doctrine, not invention

`.claude/plans/lance-graph-elixir-frontend-v1.md` (read this session) states
**"The BEAM is not a runtime target — Elixir is the SURFACE only; the
substrate stays typed Rust + SIMD + JIT."** This is exactly the
"Elixir abstraction, compile to Rust" question — and the workspace already
committed to that shape. tree-sitter-elixir parses; HM-style inference over
`|>` pipelines resolves types; the target is `lance-graph-contract::cognition`
calls. 9 stages, 5 verbs, ~50 OpKind discriminants per domain.

### 3. The recipe hot-load protocol matches Workshop's add-without-redeploy

`crates/lance-graph-contract/src/recipe.rs:28-35` (read this session) defines
the Elixir open/closed split:
- `add-atom` = **data** change (new row in the LOCKED 33-dim basis, no recompile)
- `add-style`/`add-persona` = **template** change (register a `RecipeTemplate`
  → JIT `KernelHandle` at next activation)

This is precisely Workshop's promise: a power-user adds an Action or Function
without a platform redeploy. The recipe layer is the compile-time analogue of
Workshop's runtime action registry.

### 4. The mailbox IS Foundry Automate, actor-native

The 400 ms ractor mailbox + SurrealQL kanban ("request-lose Goalstate
Maschine", `EPIPHANIES.md E-FOUNDRY-LAYER-1`) is the actor-native form of
Foundry Automate: object-change → goal-card → mailbox resolves → action
staged/applied → audit. Foundry does this as a runtime service; this stack
does it as supervised actors. Equivalent operational loop, different
substrate.

## What BREAKS / is GAPPED (honest)

### Gap 1 — Workshop's widget/view layer has NO ontology-native primitive

The existing `ruff_python_dto_check::codegen` (read this session:
`codegen/jinja.rs:1-12`, `codegen/columns.rs:1-20`) is **Flask→askama HTML
translation** — `<table>{% for %}` extraction, `_translate_cell_expr`,
`elif→else-if`. That is route-handler HTML view generation, NOT
ontology-bound widgets. Workshop widgets (`object table`, `object view`,
`action button`, `kanban`, `scenario`) bind to Object Types + Actions, not to
jinja templates.

**Consequence:** the `view` primitive in the handbook (§5) has no existing
emitter. Building it is NOT a reuse of the jinja codegen — it is a new
ontology→widget emitter. Do not conflate them (this was a category error
earlier in the session).

### Gap 2 — Actions exist as openings, but `requires{}` / `effects{}` are unsplit

The 16 openings (`.claude/odoo/openings_hops.py`, this session) classify
3 555 methods into named verbs (`iter_records_compute_from_related`,
`iter_records_raise_on_violation`, …). But a Foundry Action needs the
guard/effect split: `requires { precondition }` + `effects { mutation }`.
The openings give the *shape* (e.g. `iter_records_raise_on_violation` = a
guard that raises); they do NOT yet emit the structured
`requires{}`/`effects{}` blocks. The `raises` triples (451) are the guard
signal; the `emitted_by`/`writes` are the effect signal — both are in the SPO
graph now, but the Action-DSL emitter that composes them is unbuilt.

### Gap 3 — Submission criteria / governance has no Odoo-extracted primitive

Foundry Actions carry submission criteria (user/role/object/relation
permissions). The Odoo extraction has `@api.constrains` validators (data
integrity) but NOT the user-permission layer — Odoo `ir.model.access` /
record rules live in XML/CSV, not in the method bodies the harvest read.
**The governance spine is absent from the current extraction.** It would need
a second extraction pass over Odoo's security CSVs.

### Gap 4 — depends_on hop chains are stored as flat strings, not resolved links

`account_move.amount_total depends_on
line_ids.matched_debit_ids.debit_move_id.move_id.line_ids.amount_residual`
— that 5-hop dotted path is stored as ONE object string. It is a real link
chain (Foundry would resolve each segment to a Link traversal across Object
Types), but the current emitter does not split it into per-hop
`(ObjectType, link, ObjectType)` triples. The chain is queryable as a literal
but not yet as a traversable link path. Splitting it requires the
relation-target table (which model `line_ids` points at) — partially in the
`OdooEntity` consts (`odoo_blueprint/extracted/`), not yet joined.

## Verdict

**The Elixir-on-Rust-with-Foundry-Workshop shape is sound and the bottom two
layers are real**, not vapor:

- The SPO ontology substrate is built + tested (this session).
- The Elixir-surface/compile-to-Rust decision is committed doctrine.
- The recipe hot-load + ractor mailbox are the Workshop-action + Automate
  analogues, actor-native and compile-first.

**The top layer (Workshop widgets) and the Action guard/effect split are the
real unbuilt work** — and crucially, the existing jinja codegen is NOT a
shortcut to them (it is HTML-route translation, a different domain).

The honest one-line assessment:

> Foundry operationalizes ontology at **runtime** via a governed platform.
> This stack compiles ontology into **executable Rust semantics** at build
> time, dispatched by actors. The ontology spine + dependency graph are
> done; the widget surface + the governance/permission spine are the gaps.

## Concrete next deliverables (deterministic, no model)

1. **Action guard/effect emitter** — compose `raises` (guard) + `emitted_by`
   (effect) per opening into `requires{}`/`effects{}` blocks. Input: the SPO
   graph (built). Output: one Action spec per (family, opening).
2. **Link-chain splitter** — join `depends_on` dotted paths against the
   `OdooEntity` relation-target table (`odoo_blueprint/extracted/`) to emit
   per-hop `(ObjectType, link, ObjectType)` triples. Makes the dependency
   graph traversable as Foundry Links.
3. **Governance extraction (second pass)** — read Odoo `ir.model.access.csv` +
   record rules → `(role, may_execute, action)` triples. Closes Gap 3.
4. **Workshop widget emitter** — ontology Object Type → object-table /
   object-view / action-button widget spec. New emitter, NOT the jinja path.
   This is the largest unbuilt piece.

Each consumes the SPO ontology that is now in the store, and each is
deterministic graph→spec emission.
