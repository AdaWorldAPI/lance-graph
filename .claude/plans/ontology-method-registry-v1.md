# D-OMR-0..7 — An ontology ↔ method registry through the `lance-graph-ontology` cache

> **⊘ AMENDED 2026-11-18, after merge (#2133). §1 "a fourth row kind in the
> ontology cache" and rule R4's hydration step are SUPERSEDED by §8.** Operator,
> same day, on the question "shouldn't the BindStore be that registry?":
> *"theres a missed opportunity for bindstore and DT64 / DR to integrate —
> bindstore = DT classview / DR widefieldmask"*. The two-store split below was
> never weighed against the alternative: it followed the wording of the request
> and an available append path, and it left the authority question open (which
> of row and binding says a method exists; which wins when they differ; who may
> call `BindStore::register`). The text below is kept as written; read it
> through §8.

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
(`ogar-vocab/src/lib.rs:402`, read at OGAR `75bb38b`) has 16 fields, all
stated in names: `identity`, `predicate`, `object_class`, `default_subject`,
`default_temporal`, `default_modal`, `kausal`, `body_source`, `decorators`,
`reads`, `writes`, `calls`, `raises`, `on_enter`, `guard_failure_policy`,
`state_timeout_millis`. odoo-rs, zammad-rs, ruff and a future Knowledge Item
reader keep emitting it. It is lowered into a Method proposal and appended. No
second method vocabulary is created: a Method row holds what an `ActionDef`
states plus its effect class and guard logic.

**OGAR #385 changes role [HIRO].** Today `register_domain_table` goes from
`ActionDef` straight to the in-memory `BindStore` (stated by the HIRO session;
not read by this session). It becomes a producer that emits Method proposals.
Its conversion and its by-name refusals stay; the registration target moves to
the registry.

A Method row carries every field of the `ActionDef` it came from, plus four of
its own. None may be dropped: D-OMR-2's gate is that all 16 survive the round
trip.

| Field | Content |
| --- | --- |
| `identity`, `predicate`, `object_class` | the method and its owner class, as IRIs |
| `default_subject`, `default_temporal`, `default_modal` | as stated |
| `kausal` | the trigger or guard; see ON and WHEN below |
| `reads`, `writes`, `calls`, `raises` | DO, as predicate IRIs and names |
| `on_enter` | the enter effect. OGAR's classifier treats it as a commit-stage effect, so it decides the effect class **[HIRO]** |
| `guard_failure_policy`, `state_timeout_millis` | as stated |
| `body_source`, `decorators` | as stated (provenance of the harvest) |
| + signature revision | so a changed method is a new row |
| + guard logic | two-valued or Kleene, for the WHEN |
| + effect class | OGAR's `EffectClass` (`ogar-vocab/src/effect_class.rs:74`) |
| + provenance and confidence | source harvested from (for OGIT the tree commit); 1.0 declared, below 1.0 mined or harvested |

**ON and WHEN come from `KausalSpec`, which has seven variants [HIRO]:**
`StateGuard`, `LifecycleTrigger { event }`, `Depends`, `Constrains`,
`Onchange`, `ContextDepends`, `External`.

- Five are stated over predicates and are stored as predicate IRIs.
- **Two are not predicates:** `LifecycleTrigger { event }` and `External`. The
  row needs an event form for ON beside the predicate form. Today DT64's `on`
  is a field mask only, and OGAR's converter refuses `LifecycleTrigger` by
  name. Hydration (D-OMR-3) must keep that refusal until DT64 has an event
  trigger; an event row is stored, and yields no binding.

**Predicates are stated as IRIs, never as field positions.** OGIT ids
are row positions of the files as read; a name added in the middle shifts the
ids after it, and the tree witness refuses a stale code space **[HIRO]**. Names
do not move. Positions are resolved through the class view when the class is
bound, at the bound schema revision.

## 2. Mapping odoo's vocabulary (corrected)

| odoo | Row |
| --- | --- |
| `@api.depends(targets)`, `@api.onchange` | ON: which predicates the method is declared over. (This row said "which predicates trigger" until 2026-11-19. Nothing is triggered: a method is called, and the classid's left side already names what it is on. Owner statements on EXO-329 and EXO-333; the wording of the dt64 comments is EXO-336's to rule.) |
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
| D-OMR-3 | Method rows → `MethodBinding`s for one class at one schema revision, IRIs resolved through the class view | a predicate the view does not have is refused by name; reordering the view's fields changes no binding; an event-triggered row yields no binding and is refused by name |
| D-OMR-4 | odoo: emit proposals beside the compiled consts | a test that the two agree for every entity; consts stay the source until it has held |
| D-OMR-5 | HIRO: a Knowledge Item arrives as a Method proposal with a Kleene WHEN. **Blocked:** no Knowledge Item reader and no samples exist yet, so nothing can produce the proposal | KI-1 run from a stored row equals the oracle |
| D-OMR-6 | Promotion step for rows below confidence 1.0 | an unpromoted row yields no binding (can-fire and can-stay-silent) |
| D-OMR-7 | zammad-rs: Rails callbacks and scopes registered the same way | the bindings equal those `rails-f000` builds today |

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
  by the HIRO session; OGAR is outside this session's repositories. `ActionDef`,
  `KausalSpec` and `EffectClass` themselves were read from the OGAR checkout
  cargo resolved (`75bb38b`).
- How bardioc loads the HIRO ontology through `lance-graph-ontology`.
- No HIRO lane source exists yet (stated by the HIRO session).

## 8. Amendment: the BindStore is the method registry

**The reading.** The `BindStore` already is the place where the two sides
meet:

- **DT side, the class view.** A class's fields are read through a
  `ClassView`, never copied (`BindStore::resolve`, `lance-graph-dt64/src/lib.rs:1099`).
  The ontology registry already implements it:
  `lance_graph_ontology::class_resolver::RegistryClassView` (`class_resolver.rs:104`).
  So the predicates' intake is unchanged, and it is the name-level form.
- **DR side, the field mask.** A method's ON and reads are `WideFieldMask`s
  over that basis (`MethodBinding`, `lib.rs:222-241`); routes and their version
  log are DR's (`BindStore.routes`).

A method declaration is therefore a `MethodBinding` bound against a class view
at a schema revision. A second store that repeats it in names is a copy of
what the class view already supplies.

**What changes against §1 to §4.**

| Before | Now |
| --- | --- |
| A `Method` row in the ontology cache | No new row kind. `lance-graph-ontology` stays the intake for predicates and the provider of the class view |
| DT64 hydrated from the registry (R4) | No hydration step. Producers register with the `BindStore`; it is the one authority that a method exists |
| D-OMR-1: `MappingProposalKind::Method` and a cache schema bump | Withdrawn |
| D-OMR-3: rows → bindings | Withdrawn |

**What the `BindStore` lacks for this role, each its own deliverable and none
authorized by this file:**

| D-id | Gap | Note |
| --- | --- | --- |
| D-OMR-8 | **Registration by name.** A producer states ON / reads / writes as predicates; the store resolves them to a mask through the class view it validated, and refuses an unknown predicate by name | Today each producer resolves names to positions itself (OGAR #385, `rails-f000`). Positions shift when the ontology changes; the mask must be derived at the bound revision, never carried |
| D-OMR-9 | **A state for known but not executable.** A declaration that is stored and admitted for no run: confidence below 1.0 awaiting promotion, an event trigger (`KausalSpec::LifecycleTrigger`, `External`), an action with no parameter channel | Today the store refuses what it cannot bind, so it has no place for these |
| D-OMR-10 | **The effect class on the binding.** So `register` can refuse a field effect on a method whose class needs the gate | Answers EXO-291 question 3. A new field changes every binding digest: a versioned change |
| D-OMR-11 | **Durability.** The declarations and the route log persisted | `lance-graph-dt64-lance` already holds the durable map table; whether it fits was not read |

**What stands unchanged:** rules R1 (nothing here mints), R2 (append-only, a
change is a derived classid and a re-route), R3 (the reading is fixed at
intake), R5 (stored is not executable), R6 (no second vocabulary: `ActionDef`
is the input form); the corrected odoo mapping of §2; D-OMR-2 re-targeted to
"`ActionDef` → `BindStore` registration by name"; D-OMR-4, -5, -6, -7 with
"proposal" read as "declaration in the store".

**Decisions now needed** (replacing decisions 1 to 3 of §5):

1. Whether DT64's "no persistence path" is lifted (D-OMR-11), and where the
   durable form lives.
2. Whether a not-executable declaration belongs in the `BindStore` (D-OMR-9),
   or beside it.
3. Whether the effect class goes on `MethodBinding` (D-OMR-10).
4. Promotion (unchanged) and 5. call parameters (unchanged).

**Not verified:** how a class whose basis changed is re-bound today (the
header says another basis for a bound class is refused); whether
`RegistryClassView` keeps a history per schema revision; `dt64-lance`'s fit.

## 9. Correction: this plan is subordinate to D-DT64-7, which had already decided most of it

**What was missed.** `.claude/plans/d-dt64-7-behaviour-addressing-registration-v1.md`
was not read before this plan or its §8 were written. It carries operator
decisions of 2026-10-31 and later that answer the questions this plan treated
as open, and its execution order ends with "Not introduced: a new runtime, **a
second authoritative registry**". §1 of this plan proposed exactly that.

**Operator, 2026-11-18, on this plan (verbatim):**

> the classid is already the answer to ON

> F###-F### for methods includes both ON and DO and when is a register call of the method

> ODOO[timesheet]-F### = on Odoo/timesheet > DO F###

So the two sides read directly: **the left side is ON** (the class the
behaviour applies on, here Odoo's timesheet) and **the right `0xF###` is DO**
(the behaviour).

> 0xFFFF :: 0xF### = on the FFFF global algebra primitive DO F###

So the hardcoded form reads the same way: the left side `0xFFFF` is the global
algebra primitive, the thing the behaviour applies on, and the right `0xF###`
is what is done. It is not "a behaviour concept with no class": the algebra is
the class.

**Reading (this session's, checked against D-DT64-7 §2):** a classid has two
sides, `left (concept) :: right (ClassView / WideFieldMask selection within
it)`, and `0xF###` on either side names behaviour. A method's classid in the
behaviour ranges therefore already states what it applies on and what it
does; nothing stores an ON beside it. WHEN is not addressed at all: it is
what the `register` call supplies for that classid.

**What D-DT64-7 already decides, against this plan's rows:**

| This plan | D-DT64-7 |
| --- | --- |
| §8 "the BindStore is the method registry" | D2: Learned = registered in the `BindStore`. D13'', D15'': DT64 holds the registration authority |
| D-OMR-9, a known-but-not-executable state inside the store | **Contradicts D2 and §3.** Explore = not yet registered, and it lives OUTSIDE the store: "There is no lifecycle state besides versioned, replayable immutability." **Withdrawn** |
| D-OMR-6, a promotion step for rows below confidence 1.0 | Explore → Learned is a `register` (§3). Promotion is that call, under the registration constraints C1 to C8 |
| "ON needs an event form" (§1, HIRO fix 2) | No ON is stored; the classid answers it. **Withdrawn** |
| D-OMR-8, registration by name | O10: the consumer supplies its own `ClassView`; registration ties the owner and activation to a versioned basis witness and refuses a mismatch. Labels are resolved late through the view. **Withdrawn as a separate deliverable** |
| D-OMR-11, durability | O5 / D10': the global → local selection is a versioned lance-graph table read through `ClassView × WideFieldMask`. **Withdrawn here; it is O5's** |
| R1 "the registry mints nothing" | D13'': DT64 registers behaviour classids for its range without asking `ogar-vocab`; consumers register under their own `Domain::AppId :: 0xF###` (D14'') |

**What is left of D-OMR that D-DT64-7 does not cover:**

- **D-OMR-10:** the effect class on the binding (EXO-291 question 3). Still an
  owner decision.
- **D-OMR-2, -4, -5, -7:** producers (`ActionDef` from OGAR #385, odoo's
  consts, a Knowledge Item reader, `rails-f000`) register Learned behaviour
  under their own `Domain::AppId :: 0xF###`. These are consumers of D-DT64-7,
  not a registry design.
- The corrected odoo mapping of §2, with one more correction: `@api.depends`
  and `@api.onchange` are reads, not an ON.
- The call-parameter channel (EXO-291), unchanged.

**Status of the D-OMR ids:** D-OMR-1, -3, -6, -8, -9, -11 withdrawn. D-OMR-0
closes with this section. D-OMR-2, -4, -5, -7, -10 remain, read as work under
D-DT64-7.
