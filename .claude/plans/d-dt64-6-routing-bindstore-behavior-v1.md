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
| The merged DT64 plans do not use ON, WHEN or DO as terms; the third axis is predicates | CURRENT-CONTRACT | `.claude/plans/d-dt64-holographic-predicate-capstone-v1.md`, `.claude/plans/d-dt64-1-predicate-catalogue-source-crosswalk-v1.md`, `.claude/plans/d-dt64-q0-borrowed-quack-binding-seam-v1.md`, each read in full on 2026-10-28 |
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
| Odoo slice 1: ON 194 × WHEN 5 × DO 219; 628 live registrations; 8 multi-method cells | CURRENT-CONTRACT (doc report of a first run; the test file pins no literals) | `odoo-rs:docs/DT64-ON-WHEN-DO-REGISTRY.md:200-210` |

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

**Only routing is versioned.** DR64 maps a position to a classid and changes between versions. The classid → behaviour binding in DT64 does not change. Replay "as of version v" means which classid each address routed to at v; that classid's behaviour is the same as it is today. (PROPOSED reading of D5)

**Override = a new derived classid plus a re-route.** odoo-rs models an override as `Retract` plus `Register` at one cell. Under D5 an override registers a new classid and re-routes the cell to it. The old binding stays readable forever. A `Retract` may remove a route; it must never alter a classid's behaviour. (PROPOSED)

**Consumer composed classid as the derived DTO classid.** For a consumer, the derived DTO classid can be its own composed `u32` (`concept << 16 | app prefix`, `hotplug.rs`, doc comment on `Activation::read_modes`). OGAR mints the concept once, the app prefix is the consumer's, and hotplug activates the pair. No new bits, no extra mint, no reuse of the render prefix. Global revisions are versioned in DT64 directly and need no classid (D6). OGAR doctrine says the low half chooses the render skin and neither half carries behaviour; DT64 stores the resolution and the classid stays an address. (PROPOSED; the doctrine is from `OGAR CLAUDE.md`, not re-read here)

## 4. Amendment requested to D-DT64-Q0 line 9

Q0 line 9 reads, verbatim:

> The binding is an *interface seam*, not a second copy of business data and not the persistent Binding Store generalized in old plans. Do not invent a FrameStore, tensor physical allocation, predicate evaluator, Odoo executor, ONNX interpreter or new RelAlg AST.

**The conflict.** D4 makes DT64 a bind store with versioned, replayable immutability. Read literally, Q0 line 9 forbids a persistent binding store.

**Proposed reading.** "Store" means bindings and their version log. It never means copied business data. The user's rule of "no value duplicates" (D1) supports that reading. The rest of line 9 is unchanged: no FrameStore as a data holder, no tensor allocation, no predicate evaluator, no Odoo executor, no ONNX interpreter, no new RelAlg AST.

**Requested change.** A one-line addition to Q0 line 9 that states this reading. Until the owner accepts it, Q0 line 9 stands as written and this plan does not override it. (PROPOSED; this plan edits no other file)

## 5. Staged work

Each stage is docs or lab work needing its own approval. None is authorized by this plan.

| D-id | Deliverable | Falsifier | Owner | Gate |
|---|---|---|---|---|
| D-DT64-6-0 | This plan and its board entry (docs). | A reader finds a claim without an evidence state, or a decision labelled as technical truth. | lance-graph plan author | Docs review; no code. |
| D-DT64-6-1 | OGIT reader reconciliation. Run `lance-graph-ontology` `ttl_parse` and OGAR `OgitTables` over the SAME checkout in ONE run. Both print the same partition: accepted / rejected / duplicate / unresolved. Explain 523 files / 2,894 edges (odoo-rs §8) versus 518 / 2,756 (board claim for OGAR #363, NOT VERIFIED here). Record the observed OGIT commit as a provenance receipt only, never as a gate. | Totals fail to reconcile exactly after the partition is printed; or the receipt is used as a pin. | lance-graph-ontology and OGAR owners | Owner approval before code; read-only run. |
| D-DT64-6-2 | Expand `ogit:allowed` in `lance-graph-ontology` into a global, classid-free `(src type, verb, dst type)` table, replacing the placeholder `LinkSpec`. No second registry. | Pair-for-pair equality with OGAR `OgitTables::Allowed` fails; or a rejected / unresolved file is dropped instead of reported. | lance-graph-ontology owner | D-DT64-6-1 green; owner approval. |
| D-DT64-6-3 | Register OGIT verbs as DT64 predicates in 64-wide blocks. 378 used verbs; p95 10, max 123 per class, 2 classes above 64 (odoo-rs §8). Ordinals are per frame, never global ids. | A global verb id appears in a descriptor; or a class above 64 verbs is truncated. | DT64 catalogue owner | D-DT64-6-2 green. |
| D-DT64-6-4 | Consumer read-only access to the OGIT predicates through HotPlug activation. No consumer copy. | A consumer holds its own copy of the table. | consumer owner with OGAR authority | D-DT64-6-3 green; the `activate` call as in `hotplug.rs`, trait `CapabilityAuthority`. |
| D-DT64-6-5 | Exit from reference-only. Once a real domain (Odoo or IAM) has source-attested predicates, OGIT becomes a cross-check oracle, not the authority. | OGIT disagrees with attested predicates and OGIT wins by default. | DT64 catalogue owner | One domain with source-attested predicates. |

## 6. Open questions

All OPEN.

1. Which unit is immutable: a `(classid, DT64 binding version)` entry, or one classid per revision? The second reopens the bit-placement question. This conflicts with D5's wording "classid behaviour is permanently immutable".
2. Does per-class inheritance resolution (`_inherit`, C3 order, odoo-rs §9 P6) run before the cell lookup?
3. Coexisting methods in one cell: odoo-rs reports a "collision" (`odoo-rs:docs/DT64-ON-WHEN-DO-REGISTRY.md:98-99`). Merged #1819 settles the direction: several mappings and convergence are normal and must all be kept, and a real conflict needs incompatible source-qualified results under the same fully typed coordinate and snapshot plus an applicable domain law. Still open: whether odoo-rs's "collision" cells meet that bar or are ordinary convergence.
4. WHEN has no attested source data. The Odoo corpus carries only check-method kinds.
5. The Q0 line 9 amendment in §4 is not accepted yet.
6. Laws beyond Boolean associativity (inverse, symmetric, transitive) need instance evidence before admission (odoo-rs §11).
7. EXO-46 (global↔local admission), EXO-148 (durable identities for replay) and NFR-OP-2 (physical DR³ third axis) remain open. The levels here are logical.
8. The capstone's second axis is "targets" (Identity × Identity × Predicate), while D1 level 2 says Identity × Property × Predicate. D2 reads the capstone cube as the optional derived view. The owner of DT64-1..3 must confirm.
9. Odoo slice 1 needs 194 ON and 219 DO positions, so 4 blocks of 64 each. Merged #1819 proposes a padded 256×256 plane instead. The two shapes are not reconciled.

## 7. Non-goals and guards

- No code, ABI, test, runtime or merge authorization. This is a docs-only plan.
- No 65K×65K dense matrix. No eager 64-property cube. No per-cell 32 KiB frame.
- No new writer or persistence path. Persistence stays resident → squash/release → coalesced dirty rows → async writer (EXO-55).
- No second registry. D-DT64-6-2 replaces the placeholder `LinkSpec`.
- No pins on OGIT. A provenance receipt records what was read; it never gates.
- OGIT verbs are not Ruff `PredicateRole` bits and not the merged DT64 slot ids.
- Laws (inverse, symmetric) stay unadmitted until instance evidence exists. Schema mirroring is not evidence (`OGAR:crates/ogar-action-handler/tests/ogit_schema_composition.rs:19-21`).
- The 518 / 2,756 figures are a board claim, not verified here, and are never merged with 523 / 2,894.
- The ruff role names are quoted, not checked against ruff: #1814 lists READS, DERIVES, EFFECTS, REQUIRES, AUTHORIZES and TRIGGERS; the odoo-rs list was read in another checkout.
- ON, WHEN and DO are not DT64-0 slot terms. The merged plans use none of them.
- `ClassView::FieldMask` stays field presence, never predicate truth (capstone, section "Thesis and precise terminology").
- Do not cite the quarantined item. Nothing here derives from it.

## 8. Sources

- lance-graph [#1760](https://github.com/AdaWorldAPI/lance-graph/pull/1760), [#1792](https://github.com/AdaWorldAPI/lance-graph/pull/1792), [#1794](https://github.com/AdaWorldAPI/lance-graph/pull/1794), [#1795](https://github.com/AdaWorldAPI/lance-graph/pull/1795), [#1796](https://github.com/AdaWorldAPI/lance-graph/pull/1796), [#1797](https://github.com/AdaWorldAPI/lance-graph/pull/1797), [#1798](https://github.com/AdaWorldAPI/lance-graph/pull/1798), [#1799](https://github.com/AdaWorldAPI/lance-graph/pull/1799), [#1814](https://github.com/AdaWorldAPI/lance-graph/pull/1814) (merged, docs), [#1819](https://github.com/AdaWorldAPI/lance-graph/pull/1819) (merged, docs).
- odoo-rs [#44](https://github.com/AdaWorldAPI/odoo-rs/pull/44), [#45](https://github.com/AdaWorldAPI/odoo-rs/pull/45), [#46](https://github.com/AdaWorldAPI/odoo-rs/pull/46) (test-only lab and harvest schema).
- OGAR [#363](https://github.com/AdaWorldAPI/OGAR/pull/363) (`ogit_schema_composition.rs`).
- Board coordination: Linear EXO-50 (append-only; coordination, not approval).
