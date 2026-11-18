# D-OMR-0..7 — An ontology ↔ method registry through the `lance-graph-ontology` cache

> **Draft / documentation only, 2026-11-18.** This file authorizes no Rust
> code, no cache schema change, no OGAR `ActionDef` or classid mutation and no
> change to DT64. Each step below needs its own PR with a parity test and a
> disable run.
>
> **Origin:** operator, 2026-11-18: "when you look how odoo and OGIT register
> predicates, that part is also valuable to implement as an ontology <> method
> registry through lance-graph-ontology cache". Reviewed by the HIRO / bardioc
> session the same day; its two corrections are folded in and marked
> **[HIRO]**. Its first constraint ("the row is the stored form of an
> `ActionDef`") was then reframed by the operator's one-stop ruling, relayed by
> that session and agreed on EXO-291: **the registry is the single intake;
> `ActionDef` is an input form, not a store.**

## 0. Problem

A class's predicates have a stored home. Its methods do not.

- **Predicates (OGIT).** A producer emits a `MappingProposal` of kind `Entity`,
  `Edge` or `Attribute { predicate, semantic_type }`
  (`lance-graph-ontology/src/proposal.rs:18-65`). `OntologyRegistry::append_mapping`
  (`registry.rs:214`) appends it, idempotent by checksum, and the rows persist
  append-only in the Lance dictionary cache (`lance_cache.rs`, "never UPDATE /
  DELETE").
- **Behaviour (odoo).** `odoo_blueprint` holds each model as a compiled
  `OdooEntity` const with `OdooMethod { name, kind, return_kind, triggers }`,
  decorators with their targets, a state machine and constraints
  (`odoo_blueprint/mod.rs:170-200, 399-437`).
- **Bindings (DT64).** `BindStore` holds `MethodBinding`s in memory, immutable
  per classid, and states "no persistence path" (`lance-graph-dt64/src/lib.rs:24-30`).

So ON / WHEN / DO are never stored beside the predicates they read:
`MappingProposalKind` stops at entity, edge and attribute.

## 1. Decision proposed

Add a fourth row kind, **Method**, to the same append path and the same cache.

**The registry is the one stop.** Predicates and the methods that read them
live in the same append-only store, keyed by IRI. DT64's bindings are hydrated
from it; nothing registers into DT64 directly.

**`ActionDef` is the input form.** OGAR's `ActionDef`
(`ogar-vocab/src/lib.rs:402`, read at OGAR `cae37ed`) already carries, in
names: `identity`, `predicate`, `object_class`, `kausal`
(`KausalSpec::{StateGuard, LifecycleTrigger, …}`), `decorators`, `reads`,
`writes`, `calls`, raises. odoo-rs, zammad-rs, ruff and a future Knowledge Item
reader keep emitting it. It is lowered into a Method proposal and appended. No
second method vocabulary is created: a Method row holds what an `ActionDef`
states plus its effect class and guard logic.

**OGAR #385 changes role [HIRO].** Today `register_domain_table` goes from
`ActionDef` straight to the in-memory `BindStore` (stated by the HIRO session;
not read by this session). It becomes a producer that emits Method proposals.
Its conversion and its by-name refusals stay; the registration target moves to
the registry.

A Method row carries:

| Field | Content |
| --- | --- |
| identity | the `ActionDef` identity and its owner class, as IRIs |
| signature revision | so a changed method is a new row |
| ON | the trigger, as predicate IRIs |
| WHEN | the guard over predicate IRIs, **with its logic** (two-valued or Kleene) |
| DO | reads / writes / calls as predicate IRIs |
| effect class | OGAR's `EffectClass` (`ogar-vocab/src/effect_class.rs:74`) |
| provenance | source harvested from; for OGIT the tree commit |
| confidence | 1.0 declared; below 1.0 mined or harvested |

**Everything is stated in predicate IRIs, never in field positions.** OGIT ids
are row positions of the files as read; a name added in the middle shifts the
ids after it, and the tree witness refuses a stale code space **[HIRO]**. Names
do not move. Positions are resolved through the class view when the class is
bound, at the bound schema revision.

## 2. Mapping odoo's vocabulary (corrected)

| odoo | Row |
| --- | --- |
| `@api.depends(targets)`, `@api.onchange` | ON: which predicates trigger |
| state-machine transition guard | WHEN |
| `@api.constrains`, constraint | effect class `Validation`; **not** a WHEN. It adds errors to a record; it does not decide whether a method fires. Refused for `F000` **[HIRO]** |
| `Compute` / `Inverse` | effect class `FieldTransform`, through the gate. A field write is never an `F000` effect. Only a pure compute lowered as a query can be an `F000` fold, and that lowering does not exist yet **[HIRO]** |
| `Action`, `Cron` | through the gate as an invocation (`LifecycleWrite`, `TransactionEffect`, `ExternalHost` as classified) |

## 3. Rules

- **R1 — the registry mints nothing.** OGAR mints classes and actions. The
  registry caches what OGAR and consumers propose.
- **R2 — append-only.** A changed method is a new row and a derived classid
  plus a re-route (`BindStore::register_derived`, `route`); DT64 already
  refuses anything else as `BehaviourImmutable`.
- **R3 — the reading is declared at intake.** A row's guard logic and effect
  class are fixed when it is appended. Nothing is chosen at query time.
- **R4 — DT64 stays the in-memory binding and evaluates nothing new.** It is
  hydrated from the registry, which yields what `BindStore::register` takes.
  No producer registers into DT64 directly.
- **R5 — stored is not executable.** A row with confidence below 1.0 (mined by
  `lance-graph-arm-discovery`, harvested by ruff) never becomes a binding by
  being stored. It needs an explicit promotion step **[HIRO]**.
- **R6 — no second vocabulary.** A Method row round-trips to the `ActionDef`
  it came from; a field OGAR does not have is not added here first.

## 4. Deliverables

| D-id | What | Gate |
| --- | --- | --- |
| D-OMR-0 | This plan, the decisions below ruled | owner |
| D-OMR-1 | `MappingProposalKind::Method` and its `MappingRow` fields; cache `SCHEMA_VERSION` bump in the same commit (`lance_cache.rs:56` rule) | round-trip through the Lance cache equals the proposal; an old-version dataset is rebuilt, not misread |
| D-OMR-2 | `ActionDef` → Method proposal: OGAR #385's conversion re-targeted from `BindStore` to the registry, in a crate allowed to depend on OGAR (not `lance-graph-contract`) | every field of `ActionDef` survives the round trip; a lossy field fails the test; the by-name refusals are unchanged |
| D-OMR-3 | Method rows → `MethodBinding`s for one class at one schema revision, IRIs resolved through the class view | a predicate the view does not have is refused by name; reordering the view's fields changes no binding |
| D-OMR-4 | odoo: emit proposals beside the compiled consts | a test that the two agree for every entity; consts stay the source until it has held |
| D-OMR-5 | HIRO: a Knowledge Item arrives as a Method proposal with a Kleene WHEN | KI-1 run from a stored row equals the oracle |
| D-OMR-6 | Promotion step for rows below confidence 1.0 | an unpromoted row yields no binding (can-fire and can-stay-silent) |
| D-OMR-7 | zammad-rs: Rails callbacks and scopes registered the same way | the bindings equal those zammad-f000 builds today |

## 5. Decisions needed

1. **Who mints.** Proposed: the registry caches and mints nothing; OGAR mints.
2. **Cache schema.** Proposed: accept the version bump; a row carries the
   predicate IRIs, the guard logic, the effect class, a signature revision and
   its source.
3. **Where odoo's constants go.** Proposed: keep the consts and also emit
   proposals, with an agreement test; make proposals the source only after the
   test has held.
4. **Promotion.** Who may promote a mined or harvested row, and what evidence
   the step records.
5. **Call parameters.** Still open on EXO-291: no parameter channel exists in
   `ActionInvocation`. A Method row can name its parameters, but nothing can
   carry them yet.

## 6. What this settles if adopted

- EXO-291 question 3: the effect class gets a recorded home.
- EXO-326: a Kleene WHEN is declared with the class, not at the call.
- OGIT's shifting ids: ON and WHEN in names sidestep them.
- zammad-rs phase 2 and odoo share one registration path.

## 7. Not verified by this session

- OGAR #385 (the `ActionDef` → DT64 conversion) and the tree witness: stated
  by the HIRO session; OGAR is outside this session's repositories. `ActionDef`
  and `EffectClass` themselves were read from the OGAR checkout cargo resolved
  (`cae37ed`).
- How bardioc loads the HIRO ontology through `lance-graph-ontology`.
- No HIRO lane source exists yet (stated by the HIRO session).
