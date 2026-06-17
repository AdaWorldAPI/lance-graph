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
