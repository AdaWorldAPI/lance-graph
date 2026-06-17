# Transcode EXTEND-CORE + in-env probe slice — v1 (UNDER COUNCIL REVIEW)

> **Status:** PROPOSAL — under 5-consolidate + 3-brutal council review (2026-06-17).
> NOT ratified. Touches operator-locked OGAR canon (additively); do not implement
> until the council lands a LAND verdict.
> **Decision owner:** operator delegated autonomous decision-making backed by the
> council (this session). The council IS the review the doctrine's EXTEND-CORE rule
> requires ("filed + reviewed").

## Context (what's shipped, what the design agents found)

- **Harvester shipped** — `ruff_cpp_spo` (AdaWorldAPI/ruff PR #17, merged): 17 C++
  machine-plane SPO predicates, codegen-ready, deterministic. `tesseract::UNICHARSET`
  harvested with full method signatures (`unichar_to_id`/`id_to_unichar` as
  `has_function` + `returns_type` + ordered `has_param_type` + `is_const`, per-overload
  `(params)` IRI suffix).
- **Codegen plan committed** — `tesseract-rs/.claude/plans/tesseract-rs-ast-dll-codegen-v1.md`
  (the adapter-body half). Generator is standalone-buildable; byte-parity is gated on a
  leptonica build env this checkout lacks.
- **Probe ruling (core-gap-auditor):** `EXTEND-CORE`. The unicharset id↔utf8 bijection is
  a variable-length **shared registry**, NOT fixed-width per-node state — so it rides a
  **classid-keyed content-store tier** (shaped exactly like the shipped
  `deepnsm::Vocabulary`), resolved `classid → &UniCharSet` via a `LazyLock` registry
  mirroring `classid_read_mode`/`BUILTIN_READ_MODES`. An adapter owning its own table =
  the rejected Adapter-State-Leak. Byte-parity is BLOCKED (no leptonica; zero `.unicharset`
  on disk).

## The decision under review

Implement the **EXTEND-CORE content-store tier + the in-env dispatch-validation probe
slice** in lance-graph now (Option A below).

## Proposed change — ADDITIVE ONLY (the load-bearing safety claim)

It does **NOT** modify the locked 512-byte node: no new `ValueTenant`, no `ValueSchema`
variant, no stride change, no `ENVELOPE_LAYOUT_VERSION` bump. It adds a **sibling content-
store tier** — the doctrine's `I-VSA-IDENTITIES` Layer-3 content store, the shape the Core
"already has."

1. **`UniCharSet` content store** — `reverse: Vec<String>` (id→repr, byte-exact mirror of
   C++ `unichars[id].representation`) + `lookup: HashMap<String,u32>` (the `UNICHARMAP`
   inverse). `id_to_unichar(id)->&str`, `unichar_to_id(repr)->u32` (sentinel on miss).
   Structurally one-for-one with `deepnsm::Vocabulary`.
2. **`classid → &UniCharSet` resolver** — `LazyLock<HashMap<u32, UniCharSet>>` + free fn,
   structurally identical to `classid_read_mode`.
3. **Mint one non-default `UNICHARSET` classid.**
4. **Two thin adapters** — `unichar_to_id`/`id_to_unichar`, each a lookup, zero owned state.
   (For the probe scope use the `old_style_included_ == true` semantics so the leaf is a
   pure table lookup; the `CleanupString` normalization is a separate, later leaf.)
5. **ClassView composing them** from the harvested `has_function` manifest; invoke via
   `UnifiedStep`.
6. **Dispatch-validation test** — round-trip bijection (`id_to_unichar(unichar_to_id(s)) == s`,
   and `unichar_to_id(<absent>) == INVALID`) against a **hand-authored `.unicharset`** in
   the documented text format — no libtesseract.

## What it proves / does NOT prove

- **Proves:** the Core CAN hold the adapters end-to-end — content store + classid +
  resolver + ClassView-composed-from-manifest + `UnifiedStep` dispatch. A leak here would
  be the cheapest possible Core-gap discovery (the auditor expects none — the registry tier
  fits).
- **Does NOT prove:** byte-parity with libtesseract on a real trained table. So it CANNOT
  promote the doctrine CONJECTURE→FINDING, and **scaling stays BLOCKED** per the iron guard.

## Alternatives the council must weigh

- **A — EXTEND-CORE probe slice now (this plan).** Validates the mechanism in-env;
  additive; sets up the codegen. Does not reach byte-parity.
- **B — Golden-fixture first.** Defer EXTEND-CORE until real byte-parity is possible.
  Requires the operator to run a leptonica host to capture `eng.unicharset` + expected-ids.
  Only path to FINDING, but blocks all in-env progress on the operator's host.
- **C — Codegen-generator standalone first.** Build the ndjson→Rust generator + its tests
  (reassembly/signature/golden/Frankenstein-refusal), no Core change, defer the probe.
- **D — Harvester-polish first.** GAP-3 `has_visibility` (genuinely-dropped access
  specifiers). Orthogonal; small; verifiable in ruff. (Note: codegen's GAP-1 `member_of`
  is REDUNDANT — member→class is recoverable by inverting `has_function`/`has_field`; it is
  a codegen-side fix, not a harvester predicate.)

## Council questions

1. **core-first-architect:** does A TARGETS-CORE (vs residue / parallel-model)? Is the
   content-store tier the right home, or is it smuggling a second object model?
2. **truth-architect:** is building A justified when byte-parity (the real validation) is
   BLOCKED? Is the dispatch-validation a real measurement or synthesis-without-a-falsifier?
3. **container-architect:** is A genuinely ADDITIVE — zero impact on the locked
   `NodeRow`/`ValueTenant`/`ValueSchema`/stride/layout-version? Any hidden lock break?
4. **integration-lead:** sequencing — A vs B vs C vs D. What does each unblock; what's the
   critical path to FINDING + scaling?
5. **adapter-shaper:** are `unichar_to_id`/`id_to_unichar` genuinely thin mechanical leaves
   under this shape, or does the real C++ (`CleanupString`, fragments) leak state into them?

## Consolidation — 5-agent council (2026-06-17)

| Agent | Verdict | Load-bearing finding |
|---|---|---|
| core-first-architect | **TARGETS-CORE** | The content-store tier targets the Core, doesn't treat it as residue. Found the `ocr.rs::LayoutBlock::to_node_row` precedent — a shipped classid-keyed adapter that already rides a variable-length registry beside the SoA node. Option A is the *same shape*, not a new object model. |
| container-architect | **ADDITIVE-CONFIRMED** | Zero impact on the locked `NodeRow`/`ValueTenant`/`ValueSchema`/stride/`ENVELOPE_LAYOUT_VERSION`. The content-store tier is a sibling `LazyLock` registry; no lock break, no hidden stride change. |
| adapter-shaper | **THIN-CONFIRMED (scoped)** | `unichar_to_id`/`id_to_unichar` are pure table lookups **iff** scoped to `old_style_included_ == true`. Precondition: fixture AND oracle must both be old-style; do NOT target `id_to_unichar_ext` (fragment-expansion leaks state). `CleanupString` normalization is a separate later leaf, correctly excluded. |
| truth-architect | **PREMATURE** | The Option-A dispatch-validation is a hand-authored bijection round-trip — a *tautology*. It re-derives the already-specced `ocr-probes-v1.md` OCR-SCHEMA without adding a falsifier that CAN fail. Building it now is synthesis-without-a-measurement. Byte-parity (the real validation) stays operator-BLOCKED. |
| integration-lead | **SEQUENCE C→A→D→B** | libclang-18 + the Tesseract corpus are present in-env → **Option C (the ndjson→Rust generator) is the only critical-path move fully runnable here, today.** A needs the council gate (touches canon); B needs the operator's leptonica host; D is orthogonal polish. C unblocks A and B both. |

### Consolidated leading decision: **C-FIRST**

Revises the plan's original "Option A now". The council splits cleanly: A is
*architecturally sound* (3 of 5 confirm shape/additivity/thinness) but the *wrong
next move* (2 of 5: it's a tautology gated behind canon-review, while the
critical-path in-env work is unblocked and waiting).

**The C-FIRST move, verbatim (to be brutally critiqued before execution):**

1. **C — build + test the codegen generator (ndjson→Rust)** against REAL harvested
   `tesseract::UNICHARSET` ndjson, fully in-env: reassembly (anchor-first group-by),
   signature reconstruction from `returns_type`/`has_param_type`/`is_const`,
   overload-split on the `(params)` IRI suffix, golden + run-twice determinism,
   Frankenstein-refusal (intrusive/stateful method → route to hand-port, never force
   the adapter mold). **No Core change → no council gate.**
2. **OCR-SCHEMA** — run the already-specced, free, in-env falsifier that CAN fail
   (`ruff/.claude/plans/ocr-probes-v1.md`), satisfying truth-architect's
   measurement-before-synthesis demand.
3. **A — EXTEND-CORE probe slice** — AFTER C lands and the brutal council returns
   LAND. Still additive, still council-gated.
4. **B — golden fixture vs libtesseract** — handed to the operator (leptonica host);
   the only path to CONJECTURE→FINDING.

## Brutal panel (3 critics, 2026-06-17) — C-FIRST is RE-SCOPED, not rejected

| Critic | Verdict | Load-bearing finding |
|---|---|---|
| brutally-honest-tester | **HOLD → re-scope** | Ran the real harvest. Of C-FIRST's 5 sub-tests, **2 are tautologies** (run-twice determinism is a pure-fn-over-sorted-input no-op already covered by `cpp_ast_rt_determinism`; the self-authored golden matches itself), **1 is impossible on current data** (Frankenstein-refusal: `id_to_unichar` and the state-leaking `id_to_unichar_ext` are triple-identical except for name — no closed-vocab predicate distinguishes table-lookup from fragment-expansion), and **2 have teeth** (reassembly, signature reconstruction). Real falsifier: the harvest contains `UNICHARMAP::unichar_to_id` AND `UNICHARSET::unichar_to_id` + two `UNICHARSET::unichar_to_id` overloads → correct reassembly must yield 4 distinct adapters. Also: **freshness landmine** (live harvest 2032 triples vs committed 880 — vocab still moving), and `ruff_cpp_spo::extract` is still `todo!()` (only the example path runs). |
| adk-behavior-monitor | **AP-flagged, borderline** | **OCR-SCHEMA is mis-cited** as the C-FIRST step-2 falsifier — it tests `ValueSchema` enum-fit on the Core side, zero connection to the harvest→generator chain, so it cannot satisfy the measurement demand. The self-authored golden is AP1 confirmation bias. Redirect (identical to the tester's): gate on the **round-trip against the harvester's own IR** (`CppClass → expand → ndjson → reassemble → assert ≈ original`); never let the generator author write the expected output. Honesty guardrail: every generated adapter carries `PARITY: UNRUN (operator-blocked: leptonica)` so a green generator suite is never misread as the byte-parity FINDING. |
| baton-handoff-auditor | **CATCH-LATENT** (no CATCH-CRITICAL) | Input contract is sound to build against: ndjson roundtrip is lossless (proven on real ccutil), `CppClass→Model` unpack is an exhaustive compile-checked match, reassembly anchor-first group-by is CLEAN (GAP-1 `member_of` confirmed a misdiagnosis), generated-Rust→Core names real Core types (no parallel model), content-store-vs-value-tenant correctly drawn. **Two P1 latent drops to queue:** (a) the `(params)` IRI suffix has **no clean inverse** for comma-bearing templated types (`std::map<int, int>` → `f(std::map<int, int>,int)` → naive `,`-split gives wrong arity) — **fix: derive overload identity from index-prefixed `has_param_type` triples, never parse the suffix**; (b) `virtually_overrides` target is string-reconstructed independently of the member IRI — **fix: harvester-side intra-graph referential-integrity test** (every override object must byte-equal an existing member-subject IRI). |

### FINAL DECISION (8/8 consolidated): execute **re-scoped C-FIRST**

Build the generator now (the in-env critical path), but with the gate replaced and
the tautologies/mis-cites dropped:

1. **Gate = round-trip structural-equivalence falsifier, NOT a self-authored golden.**
   The first executable deliverable is the **reassembler** (generator stage 1) +
   a property test in `ruff_spo_triplet`: `ModelGraph → expand → Vec<Triple> →
   to_ndjson → from_ndjson → reassemble → ModelGraph'`, assert `ModelGraph' ≈
   ModelGraph`. This is the inverse of `expand()`, lives in the crate that owns both
   directions, is **immune to the freshness landmine** (compares against the live
   object), and **CAN fail** (the `UNICHARMAP`/`UNICHARSET` collision + overload
   split are the adversarial cases). truth-architect's PREMATURE flag is resolved:
   the round-trip is a real falsifier, not a tautology.
2. **Baton P1(a) baked in from line one:** reassembly derives per-overload identity
   from the index-prefixed `has_param_type` triples, never by parsing the `(params)`
   IRI suffix. (Avoids the comma-split latent drop before it can be written.)
3. **Drop run-twice as a gate** → replace with a `HashMap`-in-emit-path lint.
   **Drop the OCR-SCHEMA mis-cite** from the C-FIRST step list.
4. **Frankenstein-refusal → honest hand-curated deny-list test** (assert
   `id_to_unichar_ext` is hand-ported *by listing*, document the missing signal),
   and **queue a harvester GAP** for an intrusiveness predicate (GAP-3
   `has_visibility` is the nearest candidate).
5. **Queue baton P1(b):** harvester intra-graph referential-integrity test for
   `virtually_overrides` (before scaling past unicharset).
6. **Honesty markers:** when real `.rs` emission lands, every generated file +
   `mod.rs` records `PARITY: UNRUN (operator-blocked: leptonica)`. A green generator
   suite is NOT a green `PROBE-OGAR-ADAPTER-UNICHARSET`; byte-parity promotion lives
   only at the operator's leptonica host (Option B).

Sequence unchanged from integration-lead: **C (re-scoped) → A (EXTEND-CORE probe,
council-gated) → D (harvester polish) → B (byte-parity, operator)**.

## C-FIRST step 1 — LANDED (2026-06-17)

The re-scoped gate is built and green (AdaWorldAPI/ruff, branch
`claude/happy-hamilton-0azlw4`):

- **`ruff_spo_triplet::reassemble`** — generator stage 1, the inverse of
  `expand`'s C++ projection. Method identity recovered from index-prefixed
  `has_param_type` triples (baton P1(a) honored — never a `,`-split of the
  `(params)` suffix); class attribution anchor-first. **`cpp_projection`**
  exposed as the formal round-trip target.
- **Round-trip falsifier (NOT a self-golden):** in-crate, 6 tests incl. the
  three adversarial cases (same-name-method cross-attribution guard, overload
  split, comma-bearing templated param). 54 `ruff_spo_triplet` tests green.
- **`CPP-REASSEMBLE-RT` against the REAL corpus** (gated on `TESSERACT_SRC`):
  Tesseract 5.5.0 ccutil, **67 classes — class-set preservation + idempotence
  hold**. truth-architect's PREMATURE flag resolved: this is a real measurement
  on real data, not a tautology.
- **Falsifier earned its keep — it found a real bug class:** 48/67 round-trip
  byte-exact, **19 differ** = const/non-const overloads colliding on one method
  IRI (const-ness absent from the `(params)` suffix; `expand`'s `(s,p,o)` dedup
  merges them). Generalizes baton P1(a) to a cv-qualification axis, quantified
  19/67 on real data. Queued as **GAP-CONST-OVERLOAD** (harvester IRI-scheme
  change, operator-gated) in `ruff/.claude/plans/cpp-spo-probes-v1.md` — a
  documented known-gap, not a silent drop.
- **adk-behavior-monitor honesty guardrail** still owed when real `.rs`
  emission lands (step ≥2): `PARITY: UNRUN (operator-blocked: leptonica)`
  markers so a green generator suite is never misread as the byte-parity
  FINDING (Option B, operator's host).

**Next:** C step 2 (the Rust *emitter* consuming the reassembled `ModelGraph`)
remains; it is where the `PARITY: UNRUN` markers and the GAP-CONST-OVERLOAD
single-merged-adapter handling apply. Options A / D / B unchanged.

---

## C step 2 — the emitter: plan + council (UNDER REVIEW 2026-06-17)

> **Status:** PROPOSAL — under 5-consolidate + 3-brutal council review. The
> emitter is the first codegen that produces actual Rust *source*; the
> Core-First doctrine treats this as the central "parallel-model" danger, so it
> is gated by the doctrine ensemble before a line is written.

### What step 2 is

Stage 1 (`reassemble`, landed) turns the harvested triples back into a
`ModelGraph`. Stage 2 — the **emitter** — turns that `ModelGraph` into Rust
that targets the OGAR Core (`lance-graph-contract`: `canonical_node.rs`
NodeRow/classid, `class_view.rs` ClassView, `UnifiedStep`). The doctrine names
the harvest "the ClassView method-resolution manifest", so the emitter's job is
to project that manifest onto the Core — NOT to grow a second object model.

### The fork the council must resolve — what does the emitter emit FIRST?

The content-store tier for variable-length state (the unicharset id↔utf8
registry) is **Option A, council-gated and NOT built**. So the emitter cannot
yet produce full unicharset adapter bodies. Three candidate first cuts:

- **(1) Manifest-first (thinnest).** Emit ONLY the `classid → ClassView`
  method-resolution manifest as Rust data: per class, the ordered method set
  (name, params, return, const/static, override target) from `has_function` +
  the signature predicates. No adapter bodies, no state mapping. Targets the
  existing `ClassView` trait. Needs no content-store tier. Verifiable in-env
  (compiles; manifest matches the harvest). Risk: too thin to be "codegen"?
- **(2) Signature-stub adapters.** Emit adapter fn signatures targeting Core
  types, bodies = `// HAND-PORT` / `todo!()`. Forces the per-method
  value-tenant vs content-store mapping decision now → hits the Option-A gate
  for any unicharset leaf. Risk: blocked on Option A; premature.
- **(3) Full leaf-method bodies.** Deepest; needs the content-store tier + any
  Core gaps resolved + byte-parity (operator-blocked). Out of scope for step 2.

### Cross-cutting questions

- **Placement:** does the emitter live in `ruff` (a `ruff_cpp_codegen` module
  consuming `ruff_spo_triplet::ModelGraph`), in `tesseract-rs` (per
  `tesseract-rs-ast-dll-codegen-v1.md`), or as a standalone? What does it
  depend on, and does that respect the BBB / no-parallel-model rules?
- **Core-targeting:** does the emitted Rust reference REAL Core types
  (`NodeRow`/`classid`/`ClassView`/`UnifiedStep`), or does it drift into a
  parallel model? (doctrine's central litmus.)
- **GAP-CONST-OVERLOAD:** the 19/67 const-overload collisions — the emitter
  must treat a colliding method IRI as a single merged adapter (documented),
  not silently pick one. How is that surfaced in the generated output?
- **Honesty markers:** `PARITY: UNRUN (operator-blocked: leptonica)` on every
  generated file + `mod.rs`.

### Council questions

1. **core-first-architect:** of the three first-cuts, which TARGETS-CORE vs
   risks a PARALLEL-MODEL? Is manifest-first the right thinnest projection of
   the "ClassView method-resolution manifest", or does even that smuggle a
   second model?
2. **core-gap-auditor:** does the OGAR Core (today's `class_view.rs` ClassView
   trait + `canonical_node.rs`) already accept the manifest the emitter would
   produce, or is there a Core gap (EXTEND-CORE) before the emitter can target
   it? Is manifest-first reachable without Option A?
3. **adapter-shaper:** for the chosen cut, what is the concrete emitted shape —
   a `const`/`static` manifest table, a generated `impl ClassView`, or fn
   stubs? How does it carry the GAP-CONST-OVERLOAD merged-method case honestly?
4. **container-architect:** does the emitted output respect the locked
   NodeRow/ValueTenant/ValueSchema/ClassView layout — zero new variant, zero
   stride change? Any hidden lock break in a generated `impl`?
5. **integration-lead:** placement (ruff vs tesseract-rs vs standalone) +
   sequencing. What does manifest-first unblock; what stays gated on Option A /
   byte-parity? Is step 2 the right next move, or does D (harvester polish:
   GAP-CONST-OVERLOAD fix) come first so the manifest is collision-free?

### Consolidation — 5-agent council (2026-06-17, step 2)

**MAJOR REFRAME (3 of 5 independently): the plan's premise is false against the
code.** `ClassView` (`class_view.rs:152-249`) is a **field/render** vocabulary
(`fields() -> &[FieldRef]`, `template()`, `value_schema()`, `edge_codec_flavor()`)
— it has **no method-resolution surface**; the string `has_function` does not
appear anywhere in `lance-graph-contract`. So "emit a `classid → ClassView`
method manifest targeting the existing trait" targets a home that does not exist.

| Agent | Verdict | Load-bearing finding |
|---|---|---|
| core-first-architect | **re-scope; none of the 3 cuts as written** | The 3 cuts are PARALLEL-MODEL (1, as framed) / RESIDUE-CORE (2) / out-of-scope (3). **Discovered `codegen_spine.rs` — a SHIPPED codegen contract the plan never cites:** `TripletProjection` + `roundtrip_eq` (build-time loss falsifier), `Genericity` (codegen-vs-runtime marker), `manifest.rs` (build.rs-generated `&'static` const data is already a Core-sanctioned pattern). Proposes "projection-first": emit via `TripletProjection` validated by the shipped `roundtrip_eq`, sourced from `ruff_spo_triplet::CppMethod` directly (no new struct), no `impl ClassView`, no classid mint. GAP-CONST-OVERLOAD then becomes an **observable round-trip failure**, not a documented-and-ignored merge. Rule: emitter may only produce `impl TripletProjection` const (validated by `roundtrip_eq`) OR `ocr.rs::to_node_row`-shape bodies; never a new Rust type for a C++ class/method; never an `impl ClassView` method member (that's unsanctioned EXTEND-CORE on locked canon). |
| core-gap-auditor | **EXTEND-CORE (minimal)** | A *method*-resolution manifest has no home on today's ClassView. Minimal additive delta: a `MethodRef` POD + a free `classid_methods(classid) -> &[MethodRef]` `LazyLock` registry — the **method-axis sibling of `classid_read_mode`/`BUILTIN_READ_MODES`** — never a string-mangled `FieldRef`, never an adapter that carries its own table. No node bytes / ValueTenant / stride / version bump. **Honest split:** a *value/field* manifest IS Core-ready today (the `ocr.rs` precedent); only the *method* manifest needs the delta. Manifest-first sidesteps Option A (content-store). GAP-CONST-OVERLOAD at manifest level = benign "one entry not two" iff identity key = `(name, param_types)` (which `reassemble` already uses). |
| adapter-shaper | **manifest-first, but NOT `impl ClassView`** | `FieldRef` is `String`-backed (`ontology.rs:467`) → a `const` ClassView field table is impossible. Thinnest TARGETS-CORE shape: a `classid → &'static [MethodSig]` registry mirroring `BUILTIN_READ_MODES`, with a NEW `&'static str`-backed `MethodSig`/`Receiver` type in a new `lance-graph-contract::codegen_manifest` module. Mapped `CppMethod`→`MethodSig` (name, ordered params, ret, receiver=Const/Static, overrides); dropped the body-shaping flags (pure_virtual/constexpr/noexcept/operator/requires). GAP-CONST-OVERLOAD → `merged_overload: true` flag + comment (machine-checkable). Frankenstein guard = hand-curated deny-list (`id_to_unichar_ext`, `CleanupString`, LSTM/ELIST) in `ruff_cpp_codegen/src/hand_port_denylist.rs`. PARITY markers in `//!` headers. |
| container-architect | **ADDITIVE-CONFIRMED (all 3 cuts)** | Generated code is ClassView-side; never touches NodeRow/ValueTenant/ValueSchema/stride/`ENVELOPE_LAYOUT_VERSION`. Single invariant: generated impls **SELECT** existing `ValueSchema`/`EdgeCodecFlavor` presets, never **CONSTRUCT** a layout — unbreakable because the enums have no byte-offset constructor. Adding a sibling `codegen_manifest` module is additive (same posture as `ocr.rs` adding `LayoutBlock` beside `NodeRow`). |
| integration-lead | **placement (a) + D-first** | Emitter = new `ruff_cpp_codegen` crate in ruff next to `ruff_spo_triplet`, **emit-text-only — compile NOTHING against lance-graph** (consumes `ModelGraph`, only NAMES contract types as text; a ruff→lance-graph-contract compile edge is FORBIDDEN — inverts the contract's consumers-depend-on-it arrow; verified zero `lance` in ruff manifests). **Run D (GAP-CONST-OVERLOAD harvester fix) BEFORE the emitter** so the first manifest is collision-free (0/67 not 19/67); D is small, in-ruff, operator-gated. Manifest-first = maximal in-env slice; bodies gated on Option A, parity on Option B. |

### The unresolved fork (for the 3-brutal panel)

All 5 agree: **manifest-first** (not stubs/bodies), **additive**, **placement =
`ruff_cpp_codegen` emit-text-only**, **D before the emitter**, **PARITY markers +
Frankenstein deny-list**. The genuine disagreement the brutal panel must settle:

- **A — `MethodSig` registry (adapter-shaper + core-gap-auditor):** mint a minimal
  `MethodSig` POD + `classid_methods` `LazyLock` registry (method-axis sibling of
  `classid_read_mode`). Runtime-dispatchable; EXTEND-CORE but additive.
- **B — `TripletProjection` projection-first (core-first-architect):** ride the
  SHIPPED `codegen_spine::TripletProjection`/`roundtrip_eq`; emit NO new type;
  GAP-CONST-OVERLOAD becomes an observable `roundtrip_eq` failure. Warns that a
  new `MethodSig` re-implements `ruff_spo_triplet::CppMethod` = the parallel-model
  trap, and that re-emitting the harvest as a Rust table is a **tautology**
  ("emit the manifest, test it matches the harvest it came from").

The brutal panel adjudicates: (1) is `MethodSig` a justified minimal EXTEND-CORE
or the parallel-model/`CppMethod`-duplication trap? (2) does the shipped
`codegen_spine::TripletProjection` actually fit a method manifest, or is that
stretching a triple-projection contract? (3) is manifest-first real codegen or a
tautology — and does D go first?

### 3-brutal panel (2026-06-17, step 2)

| Critic | Verdict | Load-bearing finding |
|---|---|---|
| adk-behavior-monitor | A=tautology, B=falsifier, **emitter-before-D** | Option A's "manifest matches harvest" test is a tautology (AP2) + duplicates `CppMethod` (AP3). `roundtrip_eq` checks `decompile(project(input)) == input` *internally* (verified `codegen_spine.rs:138-183`), so it's a real falsifier. Claimed GAP-CONST-OVERLOAD becomes an observable round-trip failure and proposed building against the dirty corpus first as the fixture. **[Refuted on the const-overload point — see below.]** |
| baton-handoff-auditor | **A (re-scoped), D-first, no classid mint** | **Decisive correction:** the const-overload merge happens UPSTREAM in `expand`'s `(s,p,o)` dedup (`expand.rs:123-128`, `:581-585`) — both overloads collapse to one IRI *before* any projection/reassembly. So `roundtrip_eq` sees the collapsed set and passes; it CANNOT observe the gap. `MethodSig`≠`CppMethod`-dup (harvest-IR vs `&'static` Core-registry, the `ReadMode`/`BUILTIN_READ_MODES` posture; verified neither exists yet). Emitter must emit **name/IRI-keyed text and NEVER mint a classid** (the `ocr.rs::to_node_row(classid,…)` precedent takes classid as a caller param). Forbidden ruff→lance edge confirmed absent. |
| brutally-honest-tester | **HOLD → build with 3 fixes; A-vs-B is a FALSE BINARY** | `codegen_spine::TripletProjection` genuinely fits (generic over `type Const`; a method manifest IS a valid `Const`) — but core-first-architect's "emit NO new type" is self-contradictory (the trait REQUIRES a `Const` type). The synthesis: **B's `roundtrip_eq` machinery = the build-time GATE; A's `MethodSig` shape = the emitted text target; A's runtime registry = DEFERRED canon-review.** Confirmed const-overload is a round-trip fixed point (invisible) → carry `merged_overload` explicitly AND **D-first** (more firmly than integration-lead). The one discipline that separates codegen from tautology: gate the manifest `Const` with a round-trip against the **LIVE** harvested triples, never a hand-authored table. |

### FINAL DECISION (8/8 consolidated)

**Build the emitter — but D first, with B's round-trip as the gate and A's shape
as the output.** Execution order:

1. **D — cv-aware method IRI (NOW, autonomous correctness fix).** The current
   IRI scheme silently merges 19/67 const/non-const overloads — a **lossy-harvest
   defect** the falsifier quantified, not a feature. Append ` const` to the method
   IRI when `is_const` (in `expand`), reconstruct it in `reassemble`, and make the
   override target cv-aware in `clang_walker`. **Reclassified from "operator-gated"
   to autonomous correctness fix:** it adds NO predicate (vocab stays 54), it is
   ruff-internal, and it has a hard falsifier — `CPP-REASSEMBLE-RT` must go
   **48/67 → 67/67** byte-exact. (Reversible: a string format in three functions.)
2. **Emitter scaffold (`ruff_cpp_codegen`, emit-text-only).** Walks the
   reassembled `ModelGraph`, emits `MethodSig`-shaped Rust **text** naming
   `lance_graph_contract` types; `ruff_spo_triplet`-only dep (no lance edge).
   **Gate = a ruff-side round-trip** (the `TripletProjection` PATTERN over
   `ruff_spo_triplet::Triple`: emit manifest `Const` → decompile → assert
   `(s,p,o)`-set-equal to the live harvested triples). NEVER mints a classid
   (name/IRI-keyed). PARITY markers + hand-curated Frankenstein deny-list.
3. **`MethodSig` EXTEND-CORE in lance-graph (additive, council-reviewed).** A
   `&'static str`-backed POD + `classid_methods` `LazyLock` registry — the
   method-axis sibling of `classid_read_mode`. container-architect:
   ADDITIVE-CONFIRMED (generated impls SELECT presets, never CONSTRUCT layout).
   Lands when the emitter's output is wired; classid binding stays OGAR-side.
4. **Wire + byte-parity (Option B, operator's leptonica host)** — the only path
   to CONJECTURE→FINDING; unchanged.

**Corrections folded in:** the gate is a live-triple round-trip (not a
self-golden); GAP-CONST-OVERLOAD is fixed at the source by D (not relied on as a
round-trip failure); A and B are lifecycle stages, not a choice. Sequence:
**D → emitter scaffold → MethodSig EXTEND-CORE → wire/parity.**

### D — LANDED (2026-06-17, AdaWorldAPI/ruff)

cv-aware method IRI shipped: `expand` appends ` const` when `is_const`,
`clang_walker` does the same for the override target, `reassemble` reconstructs
it. **`CPP-REASSEMBLE-RT` went 48/67 → 67/67 byte-exact, now a hard gate.**

**The falsifier corrected the council's own assumption.** "19/67 = const
overloads" was an inference; the cv-aware IRI fixed only **3**. Tracing the rest
(per-class methods/templates/fields delta) showed the council had conflated three
distinct causes: **13** were benign duplicate `template_instantiates` that
`expand` dedups but `cpp_projection` did not; **2** were duplicate-harvested
methods (same cause); **1** (UnicityTable) was an equal-`(name,params)`-sort-key
ordering artifact for a const/non-const pair. Two round-trip-metric fixes
followed — `cpp_projection` now de-duplicates every collection (mirroring
`expand`'s `(s,p,o)` dedup), and the methods sort key includes `is_const`. Net:
the round-trip is now a faithful, total-order, deduped projection; real
collisions (same key, different content) still surface, exact duplicates collapse.

This is the C-FIRST doctrine working as intended: build the cheap in-env
falsifier, run it on real data, and let it overturn the synthesis — the council
*assumed* a single const-overload cause; the measurement found three, and only
one needed D. `GAP-CONST-OVERLOAD` is RESOLVED; no merged-adapter known-gap
remains. **Next: the emitter scaffold (`ruff_cpp_codegen`, autonomous,
emit-text + ruff-side round-trip gate); then the `MethodSig` EXTEND-CORE in
lance-graph (additive, the emitted text's compile target).**

### Emitter scaffold — LANDED (2026-06-17, AdaWorldAPI/ruff)

`ruff_cpp_codegen` shipped (depends on `ruff_spo_triplet` only — the forbidden
`ruff → lance-graph` edge confirmed absent):

- **`project(&ModelGraph) → Vec<ClassManifest>`** extracts the method plane
  (`MethodSig` = name, ordered params, return, `is_const`, `is_static`, override;
  body-shaping flags dropped). **`render(&[ClassManifest]) → String`** emits
  Rust **text** naming `lance_graph_contract::codegen_manifest::MethodSig` —
  emit-text-only, deterministic, escaped (comma-bearing templated params stay one
  quoted element), with `PARITY: UNRUN` headers and the hand-curated Frankenstein
  deny-list (`id_to_unichar_ext`, `CleanupString` → `// HAND-PORT`).
- **The gate has teeth (not a self-golden):** `decompile(project(g))` must equal
  `expand(g)` on the signature plane — the `codegen_spine::roundtrip_eq` pattern
  over the *live* triples, implemented over `ruff_spo_triplet::Triple` to stay
  lance-graph-free. `round_trip_detects_a_dropped_method` proves it can fail.
- **`CPP-CODEGEN-RT` (gated e2e on the real corpus):** ccutil harvest → 67
  classes, **857 methods → a 124 KB `MethodSig` manifest**; the signature-plane
  round-trip holds and the render carries PARITY + one literal per method.
  classid minting stays OGAR-side (name-keyed text). 10 + 16 tests green.

**Next (operator-gated): the `MethodSig` EXTEND-CORE in `lance-graph-contract`**
— a `&'static str`-backed `MethodSig` POD + `codegen_manifest` module (additive,
container-architect ADDITIVE-CONFIRMED), the compile target the emitted text
names. It touches operator-locked canon (additively), so unlike D and the
scaffold (ruff-internal correctness/codegen) it is the deliberate-Core-growth
step the doctrine says to file + review before landing. Then: wire the generated
crate in tesseract-rs and run byte-parity (Option B, operator's leptonica host).

### MethodSig EXTEND-CORE — LANDED (2026-06-17, AdaWorldAPI/lance-graph)

`lance_graph_contract::codegen_manifest` shipped (the council pre-review =
file + review; container-architect ADDITIVE-CONFIRMED):

- **`MethodSig`** — the dispatch signature in a `const`-constructible shape (all
  fields `&'static`: `name`, `params: &'static [&'static str]`, `ret`,
  `is_const`, `is_static`, `overrides`). This is the exact literal
  `ruff_cpp_codegen::render` emits, so the generated text now has a real compile
  target. The `&'static` shape is the whole point — `class_view::FieldRef` is
  `String`-backed and cannot appear in a `const`; `MethodSig` is the method-axis
  sibling that can.
- **`ClassMethods{classid, methods}`** + **`methods_for(registry, classid)`** —
  the registry-entry type + pure zero-fallback lookup. classid is bound
  OGAR-side (never minted here); the runtime `classid→methods` registry DATA is
  generated downstream (consumer repo), NOT stored in lance-graph — the
  honest-tester's "defer the runtime registry" honored.
- **Additive**: a sibling module, zero `NodeRow`/`ValueTenant`/`ValueSchema`/
  stride/`ENVELOPE_LAYOUT_VERSION` impact. Board hygiene: LATEST_STATE Contract
  Inventory updated in the same commit (D-CPP-CODEGEN-1). +2 tests
  (const-constructibility proof — the load-bearing property — + zero-fallback
  lookup); 640 contract lib green; clippy `-D warnings` clean.

**C-FIRST status:** D ✓ → emitter scaffold ✓ → MethodSig EXTEND-CORE ✓. The
in-env, lance-graph-internal arc of the pipeline is complete: a C++ harvest now
flows harvest → reassemble → project → render → a `MethodSig` manifest whose
type exists in the Core. **Remaining (operator-gated, out of this env):** wire
the generated crate into tesseract-rs (needs the leptonica build env) and run
`PROBE-OGAR-ADAPTER-UNICHARSET` byte-parity (Option B) — the only path to
CONJECTURE→FINDING. Everything to here is CONJECTURE per the doctrine; the
`PARITY: UNRUN` markers on every generated file say so.
