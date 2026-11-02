# D-DT64-6: DR64 routing, DT64 bind store, per-cell WHEN×ON×DO behaviour and OGIT reference predicates

**Date:** 2026-10-28. **Status:** PROPOSED / DOCS ONLY. **Parent:** [D-DT64-0](d-dt64-holographic-predicate-capstone-v1.md), [D-DT64-Q0](d-dt64-q0-borrowed-quack-binding-seam-v1.md). **Not:** code, ABI, test, runtime, or implementation authorization.

This plan adds no code, no ABI, no test, no runtime and no merge authorization. It records decisions the user stated in session on 2026-10-28, maps them onto what exists, and stages the follow-up work. #1812 is quarantined; nothing here derives from it.

Evidence states used below: VERIFIED-IN-CODE (file:line re-read), CURRENT-CONTRACT (merged plan text), WORKING-MODEL, PROPOSED, OPEN. A user decision is a decision, never technical truth.

## 1. Decisions recorded

**DECISION (2026-10-28):** DT64 has three levels, all 64×64×64, reached by focused global→local. (1) DR64³ is Identity(+value) × Property(+value); combinations are additional planes; no value is duplicated. (2) Identity × Property × Predicate. (3) Predicate³ = WHEN × ON × DO.
**SCOPE:** the logical shape of the three levels and how a reader reaches them. No physical layout is chosen.
**BASIS:** the user's statement in session, 2026-10-28 (decision, not measurement)
**REVISIT WHEN:** NFR-OP-2 (the physical DR³ third axis) is adjudicated, or a level is shown not to fit 64×64×64.

**DECISION (2026-10-28):** The primary relational view is DR#3, Identity × Properties. Identity × Identity (causes / caused-by, a direction-flip bit, a typed inverse only when declared) and Properties × Properties (co-occurrence, covariance, derived on demand) are OPTIONAL derived views. Global selection is `DR256::WHERE` over 65,536 addresses with an 8 KiB mask. Not goals: no 65K×65K dense matrix, no eager 64-property cube, no new writer or persistence path. Persistence stays separate: resident → squash/release → coalesced dirty rows → async writer (EXO-55).
**SCOPE:** which views are primary versus derived, and what is excluded.
**BASIS:** the user's statement in session, 2026-10-28 (decision, not measurement)
**REVISIT WHEN:** a derived view is needed on a hot path, or EXO-55 changes the writer path.

**DECISION (2026-10-28):** WHEN × ON × DO attaches per DR#3 cell, as the registry over methods.
**SCOPE:** where the method registry attaches. Not how it is stored (see §3).
**BASIS:** the user's statement in session, 2026-10-28 (decision, not measurement)
**REVISIT WHEN:** an attested WHEN source exists and shows the per-cell attachment is wrong.

**DECISION (2026-10-28):** DR64 is direct routing: a mapping table from a position to the target bound there. DT64 is the DTO bind store for all holographic addresses, including the derived views. Behaviour is registered as a DTO: classid → polyfill over mask-risc / ndarray. DT64 gives versioned, replayable immutability for addressing behaviour.
**SCOPE:** the roles of DR64 and DT64, and how behaviour is registered.
**BASIS:** the user's statement in session, 2026-10-28 (decision, not measurement)
**REVISIT WHEN:** the Q0 amendment in §4 is refused, or a bind store proves to need copied business data.

**DECISION (2026-10-28):** A classid's behaviour is permanently immutable. It is versioned under a fixed concept through a derived DTO classid. Registration goes through the lance-graph-contract `hotplug.rs` pattern.
**SCOPE:** the unit of immutability and the registration route.
**BASIS:** the user's statement in session, 2026-10-28 (decision, not measurement)
**REVISIT WHEN:** the open question on the immutable unit (§6, first item) is settled.

**DECISION (2026-10-28):** Global behaviour needs no classid and lives in lance-graph. Consumers need a classid, enabled by the plug-and-play (hotplug) pattern.
**SCOPE:** who needs a classid.
**BASIS:** the user's statement in session, 2026-10-28 (decision, not measurement)
**REVISIT WHEN:** a global behaviour needs consumer-visible identity.

**DECISION (2026-10-28):** OGIT is immutable and is read unpinned. The AdaWorldAPI OGIT fork's `master` moves when upstream PRs are merged, and readers follow it. No pins.
**SCOPE:** how OGIT is read by every reader named in §5.
**BASIS:** the user's statement in session, 2026-10-28 (decision, not measurement)
**REVISIT WHEN:** an OGIT change breaks a reader and a provenance receipt is not enough.

**DECISION (2026-10-28):** DR64 direct routing references existing identities; it mints none. For the global→local register, the prefetch amortizes its cost by holding a copy: the copy is taken once per frame and reused by every read under that frame.
**SCOPE:** what the held copy contains and how long it lives. It contains identity references and local positions only (the pointer table), never business values, so it is not a data replica. It is valid only while its frame is current: a frame check that fails (source, relation map or coordinate revision moved, or a foreign store) discards it and prefetches again, never reads through it.
**BASIS:** the user's statement in session, 2026-10-28 (decision, not measurement). Consistent with the zero-copy law's single carve-out, global→local inversion materializing only the pointer table (lance-graph `CLAUDE.md`, "No millisecond results" section, bullet "There are no boundaries"), and with Q0's demand that any copy be explicit and accounted (Q0 Phase Q0 and Phase Q3, which count source read/prefetch in the cost).
**REVISIT WHEN:** a measurement shows re-prefetching per read is cheaper than holding the copy, or the copy is found to carry values.

## 2. Grounding table

What exists today. "odoo-rs" (main @558f12a), "OGAR" (main @6b43a48) and #1814 and #1819, both merged into lance-graph `main` and read in full here on 2026-10-28, are cited from this repo; the other two label other checkouts, whose lines were read in a session crosswalk on 2026-10-28 and are not re-read in this file. In-repo sources are cited by symbol or section heading, not by line number, so a later edit to the target does not silently re-point the citation. Every in-repo source below was read in full on 2026-10-28.

| Item | Evidence state | Citation |
|---|---|---|
| `AxisKind` has `Identity`, `Property`, `Community`, `Capability` | VERIFIED-IN-CODE | `crates/dr-math-binding-probe/src/lib.rs`, enum `AxisKind` |
| The 64×64 bitplane is 512 B and carries no global ids, bag multiplicity or provenance | VERIFIED-IN-CODE | `crates/dr-math-binding-probe/src/lib.rs`, module doc comment |
| `Focus` (one `revision` field): which 64×64 window is read; a new focus over the same frame is not a new frame | VERIFIED-IN-CODE | `crates/dr-math-binding-probe/src/lib.rs`, struct `Focus` |
| `FrameRevisions` holds source, relation map and coordinate revisions; `FrameStore` compares frames by identity of issue (in-memory lab stand-in) | VERIFIED-IN-CODE | `crates/dr-math-binding-probe/src/lib.rs`, structs `FrameRevisions`, `FrameStore` and `FrameStore::check` |
| `Operand` carries validity bits with the relation bits | VERIFIED-IN-CODE | `crates/dr-math-binding-probe/src/lib.rs`, struct `Operand` |
| D-DRMC compares a guarded Identity×Property DR64 against Identity² | CURRENT-CONTRACT | `.claude/plans/dr-mathematical-compiler-v1.md`, sections "Staged falsifiers and review gates" (stage 1) and "Experimental success criterion" |
| Direct routing maps a field position to the executable bound to it | VERIFIED-IN-CODE (lab test) | `crates/dir-rmap-probe/tests/classview_dr.rs`, doc comment on `type Route` |
| Direct routing maps a method position to a fold, and a position on a page to an executable | VERIFIED-IN-CODE (lab tests) | `crates/dir-rmap-probe/tests/qr_focus.rs` and `crates/dir-rmap-probe/tests/qr_register.rs`, doc comments on `type Route` |
| `hotplug.rs` is the plug-and-play pattern every consumer migrates to | VERIFIED-IN-CODE | `crates/lance-graph-contract/src/hotplug.rs`, module doc comment |
| OGAR is the authority; it verifies expected consumer, coverage in both directions, and ids minted exactly once | VERIFIED-IN-CODE | `crates/lance-graph-contract/src/hotplug.rs`, module doc comment, role 2 |
| The consumer declares one `HotPlug` const naming its classids and covered capabilities, and calls `activate` in its own binary or tests | VERIFIED-IN-CODE | `crates/lance-graph-contract/src/hotplug.rs`, module doc comment, role 3 |
| `HotPlug` has fields `consumer`, `classids` (a `u16` slice) and `covered` | VERIFIED-IN-CODE | `crates/lance-graph-contract/src/hotplug.rs`, struct `HotPlug` |
| A consumer composes its own full `u32` as concept shifted left 16, OR its app prefix | VERIFIED-IN-CODE | `crates/lance-graph-contract/src/hotplug.rs`, doc comment on `Activation::read_modes` |
| `CapabilityAuthority::activate` takes a `HotPlug` and returns an `Activation` or an `ActivationDrift` | VERIFIED-IN-CODE | `crates/lance-graph-contract/src/hotplug.rs`, trait `CapabilityAuthority` |
| The `HotPlug` declaration is a compile-time const; verification is the `activate` call, run once in the consumer | WORKING-MODEL | reading of the `hotplug.rs` module doc comment, role 3 |
| `ttl_parse` builds `LinkSpec` with placeholder subject/object types; `ogit:from-to` and `ogit:allowed` are not yet expanded | VERIFIED-IN-CODE | `crates/lance-graph-ontology/src/ttl_parse.rs`, module doc section "What this parser handles today" |
| `ttl_parse` `classify` makes a subject a verb only for `a rdfs:Class` plus `rdfs:subClassOf ogit:Verb`. A verb written `a owl:ObjectProperty; rdfs:subPropertyOf ogit:Verb` falls to `SubjectKind::Other` and yields no proposal | VERIFIED-IN-CODE | `crates/lance-graph-ontology/src/ttl_parse.rs`, fn `classify` |
| OGAR's verb reader is the exact complement: `parse_verb` returns `None` unless the file contains `a owl:ObjectProperty`, so the `rdfs:Class` verbs are rejected there. `extract_allowed_block` reads one `[ verb target ]` pair per line from the first `ogit:allowed (`…`)`; `OgitTables::load` keeps a tuple only when both verb and target resolve, and reports the rest as `Unresolved` | VERIFIED-IN-CODE (other repo) | `OGAR:crates/ogar-from-schema/src/sgo.rs`, fn `parse_verb`; `OGAR:crates/ogar-from-schema/src/ttl.rs`, fn `extract_allowed_block`; `OGAR:crates/ogar-action-handler/src/ogit.rs`, fn `OgitTables::load` |
| OGIT verb files: 523. Of these, 12 are `a rdfs:Class; rdfs:subClassOf ogit:Verb`, all under `NTO/WorkOrder/verbs/` and all carrying `ogit:from-to`; 511 are `a owl:ObjectProperty; rdfs:subPropertyOf ogit:Verb`. `ogit:allowed` occurs in 960 files, none under a `verbs/` directory. So neither reader sees every verb: lance-graph sees 12, OGAR 511 | MEASURED (`find -path '*/verbs/*.ttl'` plus per-pattern file counts over an unpinned OGIT clone, commit `b5731a8`, receipt only; three sample files read in full) | AdaWorldAPI/OGIT, `NTO/WorkOrder/verbs/Assigned.ttl`, `NTO/Compliance/verbs/financiallySupports.ttl`, `NTO/MARS/entities/Machine.ttl` |
| The three DT64 plans (capstone, DT64-1, Q0) do not name ON, WHEN or DO; their third axis is predicates. ON×WHEN×DO is established in merged lance-graph research docs instead: the D-CFA series (logical ON×WHEN×DO incidence S1, a method-analysis view kept separate from the DT64-0 catalogue, with K3 WHEN guards and ablation over it) and the EAW series (Identity × Property/Edge × Predicate with ON/WHEN/DO as typed mapping roles, the logical third axis kept open under NFR-OP-2), plus #1819's factored ON×DO plane | CURRENT-CONTRACT (plans) and CURRENT-CONTRACT (research docs, not implementations) | the three plans; `docs/research/counterfactual-behavior/` (README, `cfa-1`, `cfa-2`, `cfa-4`, `thought-dependency-scratchpad`); `docs/research/epistemic-awareness/` (`eaw-1`, `eaw-2`, `eaw-5`, `eaw-6`, `eaw-10`, `dr64c-epistemic-awareness-scratchpad-20261027`); all read in full on 2026-10-28 |
| Capstone: the Odoo method fold is listed as missing | CURRENT-CONTRACT | `.claude/plans/d-dt64-holographic-predicate-capstone-v1.md`, section "Evidence and non-goals" |
| Capstone: 64×64×64 = 262,144 cells = 32 KiB dense; never eagerly allocate a cube for an 8- or 23-field question | CURRENT-CONTRACT | `.claude/plans/d-dt64-holographic-predicate-capstone-v1.md`, section "Logical capacity and materialization" |
| odoo-rs registry: lab design, "Nothing here is production code"; §1 axes of at most 64 positions per frame; §3 dispatch fold; §5 collisions; §8 OGIT counts; §10 measured results; §11 open points | CURRENT-CONTRACT (other repo, lab design doc) | `odoo-rs:docs/DT64-ON-WHEN-DO-REGISTRY.md:3,13-24,55-56,98-99,139-169,200-210,246-258` |
| #1814 cfa-1 (merged, docs only, falsifiers not run): S1 is a proposed method-analysis view, not equivalent to the DT64-0 logical 64-predicate catalogue; WHEN is a reference to a typed guard with a three-valued result and cannot be folded as a simple OR of satisfied atoms; one cell maps to zero, one or several method identities | CURRENT-CONTRACT (research doc, not an implementation) | `docs/research/counterfactual-behavior/cfa-1-typed-incidence.md`, sections "Question", "Logical shape, no Rust ABI commitment" and "Kill tests" (I01, I03) |
| #1819 (merged, docs only, not implemented or measured): ON×DO 256² presence bitmap plus borrowed Quack WHERE populations; the 256² bitmap and the 65,536-row WHERE mask are equal-sized but different coordinate domains; several mappings onto one polyfill, and convergence of rules into one state, are normal and must all be kept | CURRENT-CONTRACT (research doc, not an implementation) | `docs/research/on-do-watched-where-factored-dispatch-plan-20261027.md`, sections "Research question", "Operator correction" and "Normal convergence of rules into a shared state" |
| OGAR `KausalSpec` variants `StateGuard`, `LifecycleTrigger`, `Depends`, `Constrains`, `Onchange`, `ContextDepends`, `External` | VERIFIED-IN-CODE (other repo) | `OGAR:crates/ogar-vocab/src/lib.rs:610-652` |
| OGAR mints `action_handler` 0x0C07, `action_applicability` 0x0C08, `automation_trigger` 0x0C09; `action_capability` is RESERVED | VERIFIED-IN-CODE (other repo) | `OGAR:crates/ogar-vocab/src/lib.rs:1415-1440` |
| OGAR DO-arm lift: schema half coded; behavioural half is CONJECTURE | CURRENT-CONTRACT (other repo) | `OGAR:docs/HIRO-DO-ARM-LIFT.md:86-90,161` |
| OGAR `OgitTables` has five tables incl. `Allowed` with `src`, `verb`, `dst` as `u32` lanes; refuses a table past 65,536 rows | VERIFIED-IN-CODE (other repo) | `OGAR:crates/ogar-action-handler/src/ogit.rs:1-14,17` |
| OGAR depends on `lance-graph-ontology` (optional, feature `ontology-cache`), so the reverse edge would be a cycle | VERIFIED-IN-CODE (other repo) | `OGAR:crates/ogar-action-handler/Cargo.toml:27` |
| OGIT counts: 523 verb files, 378 used verbs, 2,894 allowed edges over 941 entities; per-class p95 10, max 123, 2 classes above 64 | CURRENT-CONTRACT (doc report, not recomputed) | `odoo-rs:docs/DT64-ON-WHEN-DO-REGISTRY.md:139-169` |
| OGAR #363's reader: 939 types, 518 verbs, 2,756 allowed edges, 4,861 chainable verb pairs; 102 rejected files and 508 unresolved references, which are UNKNOWN coverage, never negative edges. A different reader output, not an OGIT change: never merged with 523 / 2,894 | CURRENT-CONTRACT (doc report, not recomputed) | `docs/research/epistemic-awareness/eaw-5-odoo-ogit-law-and-learning-reality-check.md`, the OGAR #363 paragraph |
| Odoo slice 1: ON 194 × WHEN 5 × DO 219; 628 live registrations; 8 multi-method cells | CURRENT-CONTRACT (doc report of a first run; the test file pins no literals) | `odoo-rs:docs/DT64-ON-WHEN-DO-REGISTRY.md:200-210` |
| D-SAG-6 (#1760, unchanged since merge): Codegraph → Mathematics → Logic → Tracert over typed read-only views; distinct node identities (SourceDefId, MIRBodyId, Method/ClassId, RecipeId among them); evidence grades kept apart; changed source SHA with retained recipe or callsite ids must invalidate or rebind (F11); the same DR ordinal in another frame is refused (F07); unexamined regions are UNKNOWN (F06) | CURRENT-CONTRACT (plan, docs only) | `.claude/plans/d-sag-6-rust-codegraph-math-logic-tracert-v1.md`, sections "Minimal graph contract" and "Falsification matrix, preregistered" |
| odoo-rs #46 (merged, docs plus an optional JSON Schema and example; nothing run): an Odoo Codegraph harvest view whose nodes and edges carry one of four graph partitions (`structural`, `odoo-declared`, `odoo-effective`, `documentation`) and an evidence record (basis, status, knownness). Beside them, a top-level `dt64_candidates` array maps `source_node_ids` to an `algebra_ref`, with an optional existing `predicate_ref` and status `proposed`, `needs_attestation` or `refused`. It "does not create DT64 seats or grant algebra/DO authority". (The doc's table lists `dt64_candidates` as a fifth layer; the schema's `graph` enum does not, so the two disagree.) Every `emitted_by` MethodId must be kept: `RecomputeDag::from_triples`'s single-emitter map loses 26 edges in slice 1 and 37 in slice 2 (a #44 measurement as reported). An `effective_registry_observation` export must carry a `registry_snapshot` (`observation_id`, `addon_set_digest`, `registry_fingerprint`). Missing triggers, calls or ACLs are UNKNOWN | PROPOSED (other repo, read in full here) | `odoo-rs:docs/ODOO-CODEGRAPH-DT64-HARVEST-V1.md`, sections "Proposed mapping", "Identity and inheritance are not simple name edges" and "Multiple emitters and declared compute ordering"; `odoo-rs:schemas/odoo-codegraph-harvest-v1.schema.json` |
| crai-rs CR-H0 / CR-H1 (crai-rs #1 and #2, private repo, `main` @378a24a) are a consumer of #1760, not its successor: a sidecar retrofit around the D-SAG-6 motor, which stays owned by D-SAG-6 and its plans. CR-H0 lists #1813–#1830 as open "as checked 2026-10-27"; #1814 and #1819 have merged since, so that list is dated. It maps #1760 and #1813–#1830 onto one design: D-SAG-6 is the motor; Ruff harvest plus DT64 versioned bindings are the transmission; ON×WHEN×DO from #1814 is "source intake and guard ledger"; the deduplicated SPO view is lossy, so behaviour needs a separate ordered event lane | PROPOSED (research docs, other repo, read in full here) | `crai-rs:docs/CR-H0-CHASSIS-RETROFIT-MAP.md`, sections 1–4; `crai-rs:docs/CR-H0-RUFF-REVERSE-MOLD.md`, sections 2 and 4; `crai-rs:docs/CR-H1-SPOG-GROK-EXPORT-CROSSWALK.md`, section "The actual minimal export seam" |

## 3. Design consequences

Everything in this section is WORKING-MODEL or PROPOSED. None of it is decided beyond §1.

**Address per cell, store per (class / ClassView, property).** A 64³ frame is 32 KiB (odoo-rs §1). One frame per cell of a 64×64 DR#3 tile would cost 4,096 × 32 KiB = 128 MiB. That is the eager cube the "not goals" box excludes. The registry is therefore stored once per `(class, property)` and shared by every identity in that class. odoo-rs keys it by `(class, cell)`. A DR#3 cell reaches it through its property column and its identity's class. No per-identity copy exists. (PROPOSED)

**ON and DO live on the property axis.** odoo-rs §3: ON includes "field f changed" and DO includes "writes field f"; chaining works because DO and ON share a domain (`odoo-rs:docs/DT64-ON-WHEN-DO-REGISTRY.md:61-65`). Under D3 those field positions are DR#2 property ordinals, not a second vocabulary. Non-field events (record created) and non-field capabilities (raise, call) still need their own positions. (PROPOSED)

**WHEN is a Quack WHERE into a guard mask G.** A guard is a condition on Identity × Property values. Over the selected population it is a Quack WHERE that yields `G`. Identity-specific behaviour comes only from `G`, never from per-identity registry rows. WHEN has no attested source data yet (odoo-rs §11), so this is a shape, not a measurement. (PROPOSED)

**Dispatch order for a population.**

1. `DR256::WHERE` selects the population (8 KiB mask).
2. Quack evaluates each needed guard into `G`.
3. DR64 routes each address to its classid.
4. odoo-rs §3 gives the DO mask per event as the fold `OR over w in G of C[e][w]`. That fold is two-valued and treats each WHEN as one atom, which #1814 rules out (kill test I03: with `WHEN = A AND B`, A true and B false or unknown must not fire). The fold therefore stands only for candidate DO presence, the `ON_DO[e]` mask of #1819. Selection evaluates each registration's typed guard over the population in K3, and UNKNOWN never counts as FALSE.
5. DO bits that are property writes land on DR#3 cells as resident changes, then follow the persistence path (EXO-55).

(PROPOSED)

**The held register copy is keyed to its frame (D8).** The prefetch builds the local register once: local position → reference to an existing global identity. Reads in steps 1–4 of the dispatch order above go through that copy without re-resolving identities. The copy records the frame it was built under, and every read checks the frame first, as `FrameStore::check` does in `crates/dr-math-binding-probe/src/lib.rs`. Any refusal (`ForeignFrame`, `SourceChanged`, `MapChanged`, `CoordinatesChanged`, `FrameResealed`) drops the copy and re-prefetches. A new focus over the same frame reuses the copy, because a focus is not a new frame (same file, struct `Focus`). The measured cost must include building the copy, amortized over the reads it serves (Q0 Phase Q3). (PROPOSED)

**DT64-6 is the binding layer of the D-SAG-6 transmission.** In the crai-rs CR-H0 sidecar design around #1760, Ruff harvests source structure and DT64 binds it to versioned, source-witnessed coordinates that the D-SAG-6 motor reads. This plan's per-cell ON×WHEN×DO registry is that binding for behaviour. Three D-SAG-6 gates apply to it directly: F07 (a DR ordinal from another frame is refused, which D8's frame-keyed register copy enforces), F11 (a changed source SHA under a retained recipe id must invalidate or rebind, which D5 meets by re-routing to a new derived classid rather than mutating behaviour), and F06 (an unregistered cell is UNKNOWN, never "no behaviour"). Ruff has no Rust frontend: its SPO crates on `main` are C++, C#, MeTTa, Python, Ruby and SQLAlchemy (`ruff:crates/`, `ruff_*_spo`), so for Rust sources this transmission has no harvester yet. (PROPOSED)

**The Odoo intake to the registry is odoo-rs #46's `dt64_candidates` array.** A DR#3 cell's registrations come from those entries, each pointing at an existing `algebra_ref` (the polyfill of D4): one entry per source MethodId, every `emitted_by` emitter kept, never collapsed by the single-emitter map. They stay candidates until a binding is witnessed. Effective behaviour, after inheritance and module order, is admitted only under a pinned registry observation (#46's `registry_snapshot`: `observation_id`, `addon_set_digest`, `registry_fingerprint`), which belongs to the frame the D8 register copy is keyed to. (PROPOSED)

**A behaviour registration references ordered behaviour, never only a DO set.** CR-H0's negative mold shows two methods with the same unordered reads/writes (`*x = 1; *x += 1` versus the reverse) that give different results. A DO bitmap, or a deduplicated SPO bag, cannot tell them apart. So classid → polyfill (D4) must point at a recipe or ordered event record that carries effect order. The DO bit is only candidate presence, as #1819 already says. Under a lossy projection the honest answer is UNKNOWN. (PROPOSED)

**Only routing is versioned.** DR64 maps a position to a classid and changes between versions. The classid → behaviour binding in DT64 does not change. Replay "as of version v" means which classid each address routed to at v; that classid's behaviour is the same as it is today. (PROPOSED reading of D5)

**Override = a new derived classid plus a re-route.** odoo-rs models an override as `Retract` plus `Register` at one cell. Under D5 an override registers a new classid and re-routes the cell to it. The old binding stays readable forever. A `Retract` may remove a route; it must never alter a classid's behaviour. (PROPOSED)

**Consumer composed classid as the derived DTO classid.** For a consumer, the derived DTO classid can be its own composed `u32` (`concept << 16 | app prefix`, `hotplug.rs`, doc comment on `Activation::read_modes`). OGAR mints the concept once, the app prefix is the consumer's, and hotplug activates the pair. No new bits, no extra mint, no reuse of the render prefix. Global revisions are versioned in DT64 directly and need no classid (D6). OGAR doctrine says the low half chooses the render skin and neither half carries behaviour; DT64 stores the resolution and the classid stays an address. (PROPOSED; the doctrine is from `OGAR CLAUDE.md`, not re-read here)

## 4. Amendment requested to D-DT64-Q0, section "One-sentence contract"

The Q0 contract sentence reads, verbatim:

> The binding is an *interface seam*, not a second copy of business data and not the persistent Binding Store generalized in old plans. Do not invent a FrameStore, tensor physical allocation, predicate evaluator, Odoo executor, ONNX interpreter or new RelAlg AST.

**The conflict.** D4 makes DT64 a bind store with versioned, replayable immutability. Read literally, the Q0 contract forbids a persistent binding store.

**Proposed reading.** "Store" means bindings and their version log. It never means copied business data. The user's rule of "no value duplicates" (D1) supports that reading. The rest of the contract is unchanged: no FrameStore as a data holder, no tensor allocation, no predicate evaluator, no Odoo executor, no ONNX interpreter, no new RelAlg AST.

**Requested change.** A one-line addition to the Q0 contract that states this reading. Until the owner accepts it, the contract stands as written and this plan does not override it. (PROPOSED; this plan edits no other file)

## 5. Staged work

Each stage is docs or lab work needing its own approval. None is authorized by this plan.

| D-id | Deliverable | Falsifier | Owner | Gate |
|---|---|---|---|---|
| D-DT64-6-0 | This plan and its board entry (docs). | A reader finds a claim without an evidence state, or a decision labelled as technical truth. | lance-graph plan author | Docs review; no code. |
| D-DT64-6-1a | Classify every OGIT verb file in BOTH readers before comparing them. Today lance-graph `ttl_parse` accepts the 12 `rdfs:Class` verbs and OGAR `parse_verb` the 511 `owl:ObjectProperty` verbs, so the two verb sets are disjoint. Each reader either accepts both forms or reports the other form as rejected, by file. | A verb file is in neither reader's accepted set nor its rejected report; or the two readers' verb sets differ after this step without a reported reason. | lance-graph-ontology and OGAR owners | Owner approval before code. |
| D-DT64-6-1 | OGIT reader reconciliation. Run `lance-graph-ontology` `ttl_parse` and OGAR `OgitTables` over the SAME checkout in ONE run. Both print the same partition: accepted / rejected / duplicate / unresolved. Explain 523 files / 2,894 edges (odoo-rs §8) versus 518 verbs / 2,756 edges (OGAR #363 reader, eaw-5) as two reader outputs over one tree; neither is an OGIT change. Record the observed OGIT commit as a provenance receipt only, never as a gate. | Totals fail to reconcile exactly after the partition is printed; or the receipt is used as a pin. | lance-graph-ontology and OGAR owners | D-DT64-6-1a green; owner approval before code; read-only run. |
| D-DT64-6-2 | Build the global, classid-free `(src type, verb, dst type)` table from the ENTITY files' `ogit:allowed` blocks (the 960 files that carry them; no verb file does), plus the 12 WorkOrder verbs' `ogit:from-to` pairs, with every unresolved verb or target reported. One reader owns the table. OGAR `OgitTables::Allowed` already is that table, so the default is to read it, not to grow a second expansion in `lance-graph-ontology`; the placeholder `LinkSpec` on verb proposals is replaced from it. A standing parity test belongs next to the existing cache/tables comparison in `OGAR:crates/ogar-action-handler/tests/ontology_cache.rs`. | Two parsers produce the table; or pair-for-pair equality across the two readers fails without a reported reason; or a rejected or unresolved file is dropped instead of reported. | lance-graph-ontology and OGAR owners | D-DT64-6-1 green; owner approval. |
| D-DT64-6-3 | Register OGIT verbs as DT64 predicates in 64-wide blocks. 378 used verbs; p95 10, max 123 per class, 2 classes above 64 (odoo-rs §8). Ordinals are per frame, never global ids. | A global verb id appears in a descriptor; or a class above 64 verbs is truncated. | DT64 catalogue owner | D-DT64-6-2 green. |
| D-DT64-6-4 | Consumer read-only access to the OGIT predicates through HotPlug activation. No consumer copy. | A consumer holds its own copy of the table. | consumer owner with OGAR authority | D-DT64-6-3 green; the `activate` call as in `hotplug.rs`, trait `CapabilityAuthority`. |
| D-DT64-6-5 | Exit from reference-only. Once a real domain (Odoo or IAM) has source-attested predicates, OGIT becomes a cross-check oracle, not the authority. | OGIT disagrees with attested predicates and OGIT wins by default. | DT64 catalogue owner | One domain with source-attested predicates. |

## 6. Open questions

All OPEN.

1. Which unit is immutable: a `(classid, DT64 binding version)` entry, or one classid per revision? The second reopens the bit-placement question. This conflicts with D5's wording "classid behaviour is permanently immutable".
2. Does per-class inheritance resolution (`_inherit`, C3 order, odoo-rs §9 P6) run before the cell lookup? odoo-rs #46 narrows it without answering it: `_inherit` without `_name` extends a model in place, `_name` plus `_inherit` is classical inheritance, and `_inherits` delegates fields but not Python methods. Each extending declaration keeps its own identity, and the effective model is known only from a pinned registry observation. So a cell's registrations would be per declaration, and resolution would need that observation as part of the frame.
3. Coexisting methods in one cell: odoo-rs reports a "collision" (`odoo-rs:docs/DT64-ON-WHEN-DO-REGISTRY.md:98-99`). Merged #1819 settles the direction: several mappings and convergence are normal and must all be kept, and a real conflict needs incompatible source-qualified results under the same fully typed coordinate and snapshot plus an applicable domain law. Still open: whether odoo-rs's "collision" cells meet that bar or are ordinary convergence.
4. WHEN has no attested source data. The Odoo corpus carries only check-method kinds.
5. The Q0 contract amendment in §4 is not accepted yet.
6. Laws beyond Boolean associativity (inverse, symmetric, transitive) need instance evidence before admission (odoo-rs §11).
7. EXO-46 (global↔local admission), EXO-148 (durable identities for replay) and NFR-OP-2 (physical DR³ third axis) remain open. The levels here are logical.
8. The capstone's second axis is "targets" (Identity × Identity × Predicate), while D1 level 2 says Identity × Property × Predicate. D2 reads the capstone cube as the optional derived view. The owner of DT64-1..3 must confirm.
9. Odoo slice 1 needs 194 ON and 219 DO positions, so 4 blocks of 64 each. Merged #1819 proposes a padded 256×256 plane instead. The two shapes are not reconciled.

## 7. Non-goals and guards

- No code, ABI, test, runtime or merge authorization. This is a docs-only plan.
- No 65K×65K dense matrix. No eager 64-property cube. No per-cell 32 KiB frame.
- No new writer or persistence path. Persistence stays resident → squash/release → coalesced dirty rows → async writer (EXO-55).
- No second registry and no second `ogit:allowed` parser. D-DT64-6-2 replaces the placeholder `LinkSpec` from the one table.
- No pins on OGIT. A provenance receipt records what was read; it never gates.
- OGIT verbs are not Ruff `PredicateRole` bits and not the merged DT64 slot ids.
- Laws (inverse, symmetric) stay unadmitted until instance evidence exists. Schema mirroring is not evidence (`OGAR:crates/ogar-action-handler/tests/ogit_schema_composition.rs:19-21`).
- The 518 / 2,756 figures are OGAR #363's reader output (eaw-5) and are never merged with 523 / 2,894.
- The ruff role names cited by #1814 (READS, DERIVES, EFFECTS, REQUIRES, AUTHORIZES, TRIGGERS) are `PredicateRole` bits in `ruff:crates/ruff_spo_triplet/src/role.rs` on ruff `main`, read there; `DO_ARM` is EFFECTS | TRIGGERS and `PRECONDITION` is REQUIRES | AUTHORIZES.
- ON, WHEN and DO are not DT64-0 slot terms; the three DT64 plans do not name them. They are established in the merged D-CFA and EAW research docs and #1819 (§2), which keep S1 separate from the DT64-0 catalogue and the third axis logical.
- `ClassView::FieldMask` stays field presence, never predicate truth (capstone, section "Thesis and precise terminology").
- Do not cite the quarantined item. Nothing here derives from it.

## 8. Sources

- lance-graph [#1760](https://github.com/AdaWorldAPI/lance-graph/pull/1760), [#1792](https://github.com/AdaWorldAPI/lance-graph/pull/1792), [#1794](https://github.com/AdaWorldAPI/lance-graph/pull/1794), [#1795](https://github.com/AdaWorldAPI/lance-graph/pull/1795), [#1796](https://github.com/AdaWorldAPI/lance-graph/pull/1796), [#1797](https://github.com/AdaWorldAPI/lance-graph/pull/1797), [#1798](https://github.com/AdaWorldAPI/lance-graph/pull/1798), [#1799](https://github.com/AdaWorldAPI/lance-graph/pull/1799), [#1814](https://github.com/AdaWorldAPI/lance-graph/pull/1814) (merged, docs), [#1819](https://github.com/AdaWorldAPI/lance-graph/pull/1819) (merged, docs).
- odoo-rs [#44](https://github.com/AdaWorldAPI/odoo-rs/pull/44), [#45](https://github.com/AdaWorldAPI/odoo-rs/pull/45), [#46](https://github.com/AdaWorldAPI/odoo-rs/pull/46) (test-only lab and harvest schema).
- OGAR [#363](https://github.com/AdaWorldAPI/OGAR/pull/363) (`ogit_schema_composition.rs`).
- crai-rs #1 and #2, CR-H0 / CR-H1 (private repo; a sidecar consumer of #1760; links resolve only with access).
- Board coordination: Linear EXO-50 (append-only; coordination, not approval).

---

# Amendment A (2026-10-30): what the bind store applies from DR, the ONNX reference path, and measured gates

**Status:** PROPOSED / DOCS + two lab tests. Folded in from the retired D-LITHO note (#1859, #1860, #1862), whose "photolithography" wording was the operator's metaphor, not a production name. Nothing here changes §1–§7; it records what a DT binding must know and where the bind store already runs.

## A.1 DR facts a DT binding must know and apply

DR owns every fact below. DT decides none of them, but a binding cannot be stated without them. Every source was read in full on 2026-10-30.

| DR fact | Source | What a DT binding does with it |
|---|---|---|
| DR64 focus: 64 × 64 = 4,096 positions; Boolean tile 512 B; u16 position → SoA-row map 8 KiB; at most 2 MiB of distinct 512 B rows. DR256: 65,536 positions, 8 KiB, 128 KiB, 32 MiB. | D-DRF-1 §1; falsifier G1 (PASS, lab) | Names DR64 or DR256 and states its cost in these units. |
| Dense 64³ volume 32 KiB, 256³ 2 MiB; map plus cube 40 KiB. | D-DRF-1 §1 | Never assumes a cube exists (§7). |
| Axis semantics are declared by the caller; the physical third axis is NFR-OP-2 (DR). | D-DRF-1 §1; capstone DT64-OP-2 | **The DT binding is that caller**: it declares typed axes (`AxisKind`) and never decides the physical third axis. |
| HHTL16 address = `Morton8x8`; an aligned run of 4^j codes is an aligned 2^j × 2^j square (a 64-row word is 8 × 8, a 256-row block 16 × 16). | `morton8x8.rs`; D-DRDT §2; test in A.3 | A view is an address map over the canonical words; rotation is never a mask op (`mask-risc/tests/rotation.rs`). |
| Fixed width 64 / 256; underflow carried by the presence bit; absent is never FALSE or zero. | user ruling 2026-10-30 (EXO-50 17–18); D-DRF-1 G7; D-DRAV-1 AV11 | Out-of-focus contributors are expanded through a witnessed binding or refused. |
| The sealed map is immutable and versioned; the same ordinal under another frame is refused. | D-DRF-1 §0, §2; G3; D-SAG-6 F07 | Consistent with D8: the register copy is keyed to its frame. |
| A Boolean cell is support under a declared semiring; no Bag, NULL or evidence. `FieldMask` is presence, never predicate truth. | D-DRF-1 §1; D-QSP; capstone | A binding declares its population; Bag is refused today (`bind_dr64`: `BagNeedsMultiplicity`). |
| `Keep` → `Plane` reusable under one frame; `ScatterOrU32` is demand-only; a detached `Keep` is recomputed or refused (1,032 vs 31). | `mask-risc ir.rs:329-337`; `dr256-where-capstone-v1.md` B.6, E.5 | No escaped reuse without a complete dependency witness. |
| Map and value lifetimes are separate; a map-only operation needs no value fetch. | D-DRAV-1 §2, §6 | A method binding declares value-free or value-demanding. |
| The 64-thought plane is separate from the relation selector and the map index. | D-DRF-1 §1; R2 | Thoughts are reserved, not claimed (allocate on Thought-Git `Commit`), never a cube axis. |

## A.2 The bind-store shape runs in a lab for ONNX: `tract-harvest-probe`

**Status:** the bind store is a plan-level decision (D4, D-DRDT §1); no bind store type exists in the codebase. What exists: the contract pieces DT is built from (`ClassView::fields`, `methods_for`, `actions_for`, `hotplug`) and this lab crate. `crates/tract-harvest-probe` runs the shape of D1–D8 for ONNX op signatures, checked bit for bit against ORT (`tests/tiny.rs`, `tests/jina.rs`). Read in full 2026-10-30.

| This plan | `tract-harvest-probe` | Test |
|---|---|---|
| DR#3 Identity × Property | `dr3::OnnxGraph`: nodes = Identity, values = Property, `TILE = 64`, else `TileOverflow` | `dep_composition_is_the_source_dataflow` |
| ON / DO share the property domain (§3) | `on`, `do_` relations; `dep = do_ ∘ read_by` via `relcomp::compose`; schedule from `dep` | `schedule_comes_from_dep_not_source_order` |
| D4 classid → polyfill; D5 immutable classid | `registry::Registry`: one ONNX lens classid (`ONNX_LENS`, lab `0x0F00`); each op signature is a field under it (`MethodKey { lens, ordinal }`) labelling a classid-free algebra address; `exposed` is the lens's `WideFieldMask`; DO in `lance-graph-onnx` | `base_through_the_graph_is_bit_exact_against_ort` |
| WHEN is K3; UNKNOWN never TRUE (§3 step 4) | `Registration::when` returns `K3`; resolution refuses non-TRUE | `unknown_shapes_refuse`, `incompatible_broadcast_is_when_false` |
| Unregistered is UNKNOWN (F06) | `Refusal::Unregistered` | `unregistered_op_is_refused` |
| D8 frame-keyed binding | `Frame` revisions = (source digest, registry version, coordinate digest); `FrameStore::check` per run | `stale_generation_is_refused`, `registry_change_is_a_frame_change` |
| Numeric annotation | `Numeric::{Exact, Bounded}`; ULP measured vs ORT and f64 | `erf_sweep_against_f64_and_ort`, `trig_sweep_against_f64_and_ort` |

Production step (registry module doc): the one ONNX lens classid becomes an OGAR-minted concept activated through `hotplug` (D6), with its fields as capabilities. Op signatures never get classids of their own. (Corrected 2026-10-30; this row first read "op signature → one classid", which was the tract session's mistake.)

## A.3 Measured gates (lab)

| Gate | Test | Result |
|---|---|---|
| Morton tile | `lance-graph-contract` `morton8x8::tests::every_aligned_code_run_is_an_aligned_square` (j = 0..=8) | PASS; disable run (runs start one code late) red |
| Count as Boolean planes | `lance-graph-mask-risc/tests/dose.rs`: K = 6 planes over a 4,096-row focus, `And` + `Count` per plane, weighted 2^k, equals the per-position sum at 0/3/50/97/100 % selection; every plane load-bearing | PASS; disable run (swap adjacent weights) red. A measured fact, not an adopted Bag carrier |
| First non-ONNX consumer (stated in a test; no store built) | XLOG #1864: `NodeGuid` stand-in, R/S as `ClassView` × `WideFieldMask`, count method by reference, `Population::Bag`, expected {2:4, 3:1, 7:1} | failing fixture by design: refused with `BagNeedsMultiplicity` |

## A.4 Open (unchanged owners)

The Bag carrier (who admits a counted population into `bind_dr64`); the D-DRDT §4 DT gaps (MethodId, K3 guards, effect order, one lookup); §6 questions 1–9.

## A.5 (2026-10-30) The bind store as production code: `crates/lance-graph-dt64`

**Status:** VERIFIED-IN-CODE. This supersedes A.2's status line ("no bind store type exists in the codebase") and closes the four D-DRDT §4 gaps listed in A.4. A workspace member; its only dependency is `lance-graph-contract` (zero-dep). It stores bindings and their version log, never business data, and evaluates nothing.

| Decision / gap | In `lance-graph-dt64` | Test (`tests/bind_store.rs`) |
|---|---|---|
| D-DRDT §4 gap 1: method identity independent of name | `MethodId` + `SigRev`; the source name rides in `MethodBinding::name` | `same_read_write_sets_…` (stale `SigRev` refused) |
| gap 2: K3, conjunctive WHEN | `Guard { expr: SetExpr, logic: Logic }` from `set_algebra`, the description Quack's WHERE evaluates (§3); `Guard::witness` is the normal form under its own logic (`normalize_kleene` for K3) | `a_kleene_guard_never_treats_unknown_as_true` (`x<5 ∨ x≥5` is `All` two-valued, not under Kleene) |
| gap 3: effect order | `MethodBinding::effects: Vec<Effect>` (field position + `AlgebraRef`, the D4 polyfill reference); `writes()` is the unordered set | `same_read_write_sets_…` (equal reads/writes, unequal bindings, both kept) |
| gap 4: one lookup | `BindStore::resolve(view, classid, schema_rev)`: fields borrowed from the `ClassView`, methods from the store; `resolve_at` by DR position and route version | `one_lookup_joins_…`, `override_is_…` |
| D5 immutable classid; override = derived classid + re-route | `register` idempotent for an identical binding, else `BehaviourImmutable`; `register_derived` records the base | `class_behaviour_is_immutable`, `override_is_…` |
| Only routing is versioned (§3) | append-only route log, `route` / `retract` / `routed_at(version, position)`; future versions refused | `override_is_…`, `routes_and_derivations_…` |
| F06: unregistered is UNKNOWN | `Refusal::{UnknownClass, Unrouted, UnknownMethod}`, never an empty answer | `an_unregistered_class_is_unknown_not_empty` |
| D-DRDT §5: refuse another schema revision; preserve guard witnesses; the first DR #4 falsifier | `resolve` refuses `SchemaRevisionMismatch` and `FieldBasisChanged`; `admit(MethodRef)` refuses `GuardDropped` / `GuardAdded` / `GuardMismatch` / `SignatureRevisionMismatch` | `schema_revision_and_field_basis_are_checked`, `same_read_write_sets_…`, `a_guard_added_…` |
| A.1: population, truth, value demand | `Population` / `Truth` from `set_rules` and `ValueDemand` ride the binding; a Bag is recorded, never refused here (admitting it is the binder's call) | `population_truth_and_demand_ride_the_binding` |
| Registration hygiene | `DuplicateMethodId`, `FieldOutOfBasis` (ON, reads, effect targets), `BasisTooWide` (> 256) | `registration_refuses_…` |

Disable runs (on commit `7936665e2`; each patch asserted to apply exactly once; restored with `git checkout`): each of the eight guards removed in turn leaves exactly one test red (10 passed, 1 failed): the `GuardDropped` arm, the `BehaviourImmutable` arm, the route-log version cut, the Kleene witness, the field-basis check, the schema-revision check, the duplicate-MethodId check, the out-of-basis check.

Not done: wiring OGAR's `CapabilityAuthority` (D6, the production route for consumer classids); persistence of the store (EXO-55 owns the writer path); a router (DR #4) that carries `MethodRef`s; §6 questions 1–9, including whether the immutable unit is the classid or `(classid, binding version)` — this crate implements the D5 wording (the classid).

**A.5 addendum (2026-10-30, after the XLOG session's critique and the operator's correction).** A classid is a codebook lens, never one per codebook item. Two changes follow:
- `ClassSchema` no longer declares a view class. `view_class(classid)` is the codebook's own split (`ogar_codebook::classid_concept`).
- `Effect` carries an `EffectOp`: `Algebra(AlgebraRef)`, or `Call(MethodRef)` for reusing another lens's entry by reference. A `Call` is admitted when the caller registers (`CallRefused` wraps the reason), and `callers_of(lens)` lists who reuses it.

In the test, XLOG's per-Y count is three Calls into one algebra lens: group-count twice, then checked i64 Mul. That binds two classes in total; no algebra item gets a classid. Route targets stay bare classids (XLOG: a per-route mask would break D5). Open, with the owners: whether tract-harvest-probe's per-op lab classids (`0x0F01…`) become entries under one lens (EXO-50 message 27); the `MethodSignature` vs `MethodRef` identity (dr-math owner); where the ONNX/XLOG algebra entries are registered.

**A.5 addendum 2 (2026-10-30, XLOG correction).** A Call had no operands, so the stored binding could not tell XLOG's count from one that counts R twice. Fix: `Effect { target, op, args: Vec<Arg> }`, where `Arg` is `Field(u8)` (a position checked against the caller's basis) or `Const(i64)` (a key column, a lane). `MethodBinding::inputs` is part of the entry's signature, and a Call passing another arity is refused at registration (`ArityMismatch` inside `CallRefused`). XLOG's count: `cnt_r = Call(group-count, [r, key 1])`, `cnt_s = Call(group-count, [s, key 0])`, `count = Call(Mul i64, [cnt_r, cnt_s])`. `demand` stays `Values`, because the key column is read as a value; that is stated here, not settled.

**A.5 addendum 3 (2026-10-30, operator).** "The classid switches the behaviour, labels and polyfill over otherwise agnostic addresses or algebra methods." Addendum 1 gave the algebra its own lens classid and had consumers reach it by a class-to-class `Call`, which was wrong. Now:
- The algebra is a classid-free catalogue in the store: `AlgebraEntry { id: AlgebraId, sig_rev, polyfill, inputs }`, declared once with `declare_algebra`, immutable, and versioned in DT directly. This is D6: "global behaviour needs no classid".
- A consumer classid switches labels (its `ClassView`) and polyfill (`Effect { target, algebra: AlgebraRef, args }`) over those entries. `register` refuses `UnknownAlgebra`, `AlgebraRevisionMismatch` and `ArityMismatch`.
- `users_of(AlgebraId)` lists the classids that use an entry.
- `EffectOp::Call` and `MethodBinding::inputs` are removed.

In the test, XLOG's count and an ONNX `Mul(int64)` node reuse the same entry under their own classids; only the two consumers carry classids.

**A.5 addendum 4 (2026-10-30, operator).** "Classid widefieldmask ergonomics is the ERB Redmine fieldview pattern." This matches OGAR `ogar-render-askama::field_view` ("the ERB pattern fieldview ClassView rendering from memory without serialization") and openproject-nexgen-rs ("fieldview/erb → classview/askama"). `Resolved::field_view(&WideFieldMask)` yields `FieldRow { position, label, predicate }`, ascending, with the position as the layout address and labels borrowed late from the ClassView. It is OGAR's `FieldView` minus the value, since the store holds none. `Resolved::action_ordinals()` gives the methods as action addresses (T2). Not wired into OGAR's renderer: OGAR builds its `FieldView`s itself, and that call site is the OGAR owner's.
Scope of that projection (XLOG session; quote verified at `.claude/knowledge/ogar-consumer-preflight.md:277` and `cross-glove-business-parity-v1.md:881`): operator, 2026-10-10, *"Classview can be used for any compute masking, we explicitly expanded the ERB redmine fieldview pattern for risk mask of everything."* `ClassView × WideFieldMask` is one masking pattern; rendering, RBAC, SPOG and risk masks are uses of it, and so are a binding's ON and `reads` sets. `field_view` is that pattern's read, not a render-only API.

**A.5 addendum 5 (2026-10-30): field meaning and meaning-masks through the lens.** DT owns field semantics (D-DRDT §1). The contract already carries them per predicate: `property::PropertySpec { kind, codec_route, semantic_type, marking }`, keyed by the same predicate as `FieldRef::predicate_iri`.
- `Resolved::field_meaning(mask, schema)` joins each fieldview row to its spec.
- `Resolved::mask_where(schema, pred)` derives `WideFieldMask`s from meaning: a risk mask from `marking >= Pii`, an obligation mask from `kind == Required`. This is the "risk mask of everything" over the same pattern.
- `Resolved::unspecified(schema)` surfaces the fields with no spec. A field with no spec is UNKNOWN and never takes the spec default (`Internal` / `PlainText`); a test pins that.

The physical lane kind (u32 code vs i32, which picks XLOG's comparator) is the SoA column's (`soa_envelope::ColumnKind`), not a meaning. The store does not hold it.

**A.5 addendum 6 (2026-10-30): dataflow and behaviour footprint.**
- `register` refuses `UndeclaredRead` when a step reads a field that is neither in the method's `reads` nor written by an earlier step. A recipe's order must be what it says: XLOG's count with the multiply moved first is refused.
- `MethodBinding::footprint()` is ON ∪ reads ∪ writes, and `Resolved::methods_touching(mask)` lists the methods a meaning mask reaches. A risk or RBAC mask thereby applies to behaviour, not only to fields.

Open, for the frame owner (D8): registering a class or declaring an algebra entry moves no revision a frame can check; only routes are versioned. Classes and entries are immutable and append-only, so nothing a frame read can change. Whether a frame should still record the store's catalogue extent (for example a count of bindings) is undecided.

**A.5 addendum 7 (2026-10-30, after #1869 and #1870 merged).** #1870 settles addendum 1's open question on tract-harvest-probe's per-op classids: there is one ONNX lens classid (`ONNX_LENS`), and each op signature is a field under it labelling a classid-free algebra address. That is the shape this crate implements: the lens is a registered class, its fields come from its `ClassView`, and each method's effects reference `AlgebraEntry`s by `AlgebraRef`. Addendum 1's `EffectOp::Call` text is superseded by addendum 3. Still open: whether tract's `MethodKey { lens, ordinal }` and this crate's `MethodRef { class, method }` are one identity (EXO-50); the frame extent question above.

**A.5 addendum 8 (2026-10-30, operator: "ever heard of polyfill?").** The question "who declares the catalogue entries and assigns their ids" had an answer all along: the polyfill. In `ndarray::simd` a consumer names a facade word and nothing else. Nobody declares the word to the consumer and nobody assigns it a number, and which backend realizes it is decided per build. So:
- `AlgebraId` is removed. `AlgebraRef` is the word's facade path; there is no second, numeric spelling.
- The per-entry `SigRev` and `AlgebraRevisionMismatch` are removed. A changed realization keeps the word, and a different operation is a different word.
- `AlgebraEntry { word, inputs }` only mirrors the word's arity so `ArityMismatch` can be checked. The polyfill decides what the word is; the store mirrors it, and a word mirrored with two arities is refused (`AlgebraImmutable`).

Which polyfill (operator: "check lance-graph-relalg and lance-graph-mask-risc, that's the polyfill"): `lance-graph-relalg` describes a selection and lowers it, `lance-graph-mask-risc`'s op vocabulary (`Pred`, `MaskOp`, `Terminal`) is the one polyfill above `ndarray::simd` that every fold lowers to, and `ndarray::simd` computes. So a keyed count names mask-risc's `Terminal::GroupReduce { key: GroupKey::Lane, fold: GroupFold::Count }`, not the `ndarray::simd` word it delegates to. mask-risc has no elementwise i64 arithmetic, so XLOG's multiply names an `ndarray::simd` word. That word was reachable only as `ndarray::hpc::checked_i64::checked_binary_i64`; a consumer never names `hpc`, so ndarray #407 re-exports it through `ndarray::simd`. Disable runs (on commit `d140ab4c8`): removing the `AlgebraImmutable` arm or the `UnknownAlgebra` check each leaves exactly one test red (18 passed, 1 failed).

**A.5 addendum 9 (2026-10-30, operator: "your DTO mapping is one classid one meaning surface, same as ClassView WideFieldMask; did you use the pattern").** No: addendum 8 kept the polyfill as a string-keyed catalogue beside `ClassView`, a second meaning surface. Now the polyfill vocabulary is one lens classid, read the way every class is read:
- Its words are the fields of that lens, and their labels (the mask-risc op or `ndarray::simd` word) come from its `ClassView`.
- A word is a field position, `AlgebraRef { lens, word }`, and a set of words is a `WideFieldMask`.
- `declare_algebra(lens, schema_rev, arity)` binds the lens with one operand count per word. It is immutable like a class (`AlgebraImmutable`).
- `resolve_algebra` reads the lens through its `ClassView`, with the same schema-revision and field-basis checks as `resolve` (one shared `check_basis`). `field_view` gives each word's position and label.
- `words_used(class, lens)` gives a consumer's words as a mask over the lens.

No word gets a classid. This is the ONNX shape of #1870, applied to the polyfill. Disable runs on `9d2fba457`, each turning tests red:

| guard removed | failing tests |
|---|---|
| the `words_used` lens filter | 1 |
| the `AlgebraImmutable` arm | 1 |
| the arity position | 5 |
| the shared basis check | 4 |

Which classid the polyfill lens carries in production is OGAR's mint (D6), not this crate's; the test uses a lab value.

**A.5 addendum 10 (2026-10-30, operator: "make sure that consumers like ONNX are not limited to 64 but instead can choose the other modes like DR256² × 64").** A.1 already gives the binding this job: it names DR64 or DR256 and states its cost in those units. Until now the store only capped a class at 256 fields. `ClassSchema` now carries an `Extent` the consumer chooses:
- The extent is Identity × Property × Predicate, each axis `Width::W64` or `Width::W256`. `Extent::DR64_CUBE` and `Extent::DR256_SQ_X64` are named, and any other combination is allowed.
- Registration refuses fields past the property axis (`FieldsExceedExtent`) and methods past the predicate axis (`MethodsExceedExtent`).
- The extent is part of the immutable binding (D5).
- `dense_bytes` states what a dense Boolean volume of the extent would cost if anyone materialized one: DR64³ is 32 KiB and DR256² × 64 is 512 KiB. It is a cost statement and allocates nothing (§7).
- The polyfill lens declares its extent too.

The test registers an ONNX graph with 200 values: DR64³ refuses it, naming the axis, and DR256² × 64 binds it. A graph with 70 nodes needs a 256-wide predicate axis.

Disable runs on the extent commit:

| guard removed | failing tests |
|---|---|
| the property-axis check | 1 |
| the predicate-axis check | 1 |
| W256 read as 64 | 2 |

Still capped at 64: `tract-harvest-probe`'s `dr3::TILE = 64`, a single DR64 tile built from `relcomp::Rows = [u64; 64]`. That is the tract session's lab crate. Its full model (4,146 of 7,620 nodes WHEN-true) already reports ">64 positions untiled". Lifting it is that session's change; the binding side no longer forces 64.

**A.5 addendum 11 (2026-10-30, operator: "reserve but don't claim, thin provisioning; Jina as a result is not able to use its 7000 nodes").** Addendum 10 gave each axis one width, W64 or W256, so the most any class could address was 256 identities, and Jina's 7,620 nodes could not be bound. Each axis is now a reservation, `Axis { tile: W64 | W256, tiles }`:
- `Extent::DR256_GLOBAL_X64` reserves the whole DR256 identity space: 65,536 positions in 256 tiles of 256.
- Reserving claims nothing. `BindStore::claimed(class, version)` counts the identity tiles that hold a position routed to the class at that version.
- `tile_bytes` states the cost of one claimed tile: 256 × 256 × 64 / 8 = 512 KiB. Routing Jina's 7,620 nodes claims 30 tiles (15 MiB stated, nothing allocated), out of 256 reserved.
- `route` refuses a position past the class's identity reservation (`OutsideReservation`) and moves no version.
- An axis wider than it can address is refused (`ReservationTooWide`), never truncated.

Disable runs, each failing exactly one test:

| guard removed | failing tests |
|---|---|
| the reservation check in `route` | 1 |
| claiming by tile (counting positions instead) | 1 |
| the too-wide check | 1 |
| costing the reservation instead of the claim | 1 |

Still capped, and stated: one class's property axis is at most 256 fields, because a `WideFieldMask` position is a `u8` (contract). Wider would mean tiled field addressing in the contract, which this crate does not own. In the bind store, Jina's nodes are identity positions, and an ONNX lens's fields are its op signatures, so this does not block Jina here. The tract lab's own DR#3 layout puts Jina's 7,620 values on its property axis, and `dr3::TILE = 64` is that session's to lift.

**A.5 addendum 12 (2026-10-31, operator: "wouldn't it make sense to move DR into DT64 — at least from the lab probe crate").** DR's addressing is now the module `lance_graph_dt64::dr`, lifted from `dr-math-binding-probe`. Your diagram's "DR routes, DT interprets" becomes a module boundary inside the bind store, not a separate crate. D4 already makes DT64 the bind store for all holographic addresses, and §3 puts the route versioning there.

`dr` holds:
- reservations: `Width`, `Axis`, `Extent`, `Claim`, `MAX_RESERVED`;
- the route log: `RouteLog` with `route`, `retract`, `routed_at`, `claimed`;
- the frame witness: `FrameStore`, `Frame`, `FrameRevisions`, `FrameRefusal`;
- the guards: `Focus`, `Scope`, `Guards::check`;
- the typed axis: `AxisKind`, `ClassAxis`;
- the global → local pointer table: `LocalMap`, `address`, `coordinates`, `rotate_yz`.

It names nothing from the binding side, and a test pins that. `BindStore` keeps its API and delegates to `RouteLog`.

Two changes against the probe:
- The probe's `Axis { class, kind }` is `ClassAxis` here, because `Axis` is the reservation axis.
- `LocalMap` and `address` take a tile `Width` instead of a fixed 64, so a DR256 tile is addressed the same way.

Not lifted:
- `relcomp`, `where_dr3`, `population`, `recipe` and the folds in `hhtl16` are computations, proofs and oracles, so they belong to Quack. `relcomp` also carries egglog and sha2, which a contract-only crate must not.
- `dr4::route` produces values, so it is execution.
- `bind_dr64` / `BoundDr64` wait on the `MethodSignature` vs `MethodRef` identity question (EXO-50).

The probe itself is unchanged; its owner switches to `dt64::dr`.

Also restored here: #1871 merged before its last two commits were pushed. `9014fa07b` (thin-provisioned extent) and addendum 11 were on the branch only and are cherry-picked into this PR.

Disable runs on the `dr` commit, each turning a test red:

| guard removed | failing tests |
|---|---|
| the foreign-store check | 2 |
| the source-revision refusal | 1 |
| the focus guard | 1 |
| `LocalMap` width read as 64 | 1 |
| address width read as 64 | 1 |
| a `use crate::` in `dr.rs` (the boundary test) | 1 |
