# Persona modeling vs the rung-content ladder — the demarcation

> READ BY: v3-template-smith, truth-architect, integration-lead,
> perspective-weaver (persona side), any session touching StyleFamily,
> ThinkingStyle, runbooks, recipes, the 144 verbs, RungLevel semantics,
> or persona modeling. Operator-ruled 2026-07-14 (this session, chat
> transcript); board anchor: `E-RUNG-CONTENT-LADDER-1` in
> `.claude/board/EPIPHANIES.md`.

## Why this file exists

The workspace carried FOUR similar-looking spaces that sessions kept
conflating — three 12-spaces and a 36-space — and D-TSC-1 (`1a11038`)
canonized one conflation into the board ("36 = literal NARS runbooks").
The operator separated them. This file is the demarcation, so no future
session re-merges them.

## The rung-content ladder (operator ruling, 2026-07-14)

The 0–9 rung ladder came from ladybug-rs thinking and migrated to
`crates/thinking-engine`. Its CONTENT reading — what kind of thinking
object occupies each rung — is:

| Rung | Occupant | Where it lives (today) |
|---|---|---|
| 0–1 | **Observation** | Pearl L1 side of `contract::cognitive_shader::RungLevel` |
| 2 | **Thinking atoms** — the 12×12 = 144 universal-grammar verbs (12 verb families × 12 tenses) | `cognitive-shader-driver/src/sigma_rosetta.rs` (`VERB_ROOTS`/`VERB_TENSES`, 64 glyphs × 144 verbs = 9,216 atoms) + `contract/src/grammar/verb_table.rs` (`VerbRoleTable [[SlotPrior;12];12]`, TEKAMOLO priors) |
| 3 | **The 34 NARS reasoning tactics** (induction, abduction, deduction, extrapolation, counterfactual, …) — THE runbooks, in the literal sense: executable inference recipes | Catalogue spine: `lance_graph_contract::recipes` (`RECIPES: [Recipe; 34]`, Stakelum/ladybug numbering, Sun et al. 2025 tiers, SPO-2³ coverage). Substrate primitives: `ndarray::hpc::styles` (29/34 modules shipped, `fn(Base17, NarsTruth) → result`) — most of it migrated toward lance-graph via the recipes spine |
| 4 | **Thinking macros / styles** = `StyleFamily` (12) — Autopoiesis through the triangle **frozen × learned × exploration** (frozen = compiled templates / StepMask; learned = NARS revision / awareness; exploration = MUL / exploratory machinery) | `lance_graph_contract::style_family` |

The code's shipped `RungLevel` carries the **Pearl causal-depth**
reading (0–2 observe / 3–5 intervene / 6–9 counterfactual) plus the
homeostatic elevation policy. That coexists with — but does NOT
implement — the content ladder above. The rung↔content wiring is the
half-hearted part of the 0–9 rewiring and is OPEN (see below).

## Persona modeling — a SEPARATE storyline (the demarcation)

The **36 adjective styles** in `lance_graph_contract::thinking`
(Empathetic, Warm, Blunt, Frank, Poetic, … — 36 variants, 6 clusters,
τ addresses, `FieldModulation`) are the **persona-modeling storyline**:

- They are **NOT rung 3.** They are not reasoning moves.
- They are **NOT the NARS runbooks.** The runbooks are the 34 recipes
  (rung 3, above).
- They are **not wired** into the rung ladder, the shader dispatch, or
  the planner execution paths today ("carried and displayed, not acted
  on" — the audited finding). That is expected, not a defect: persona
  modeling is future work with its own storyline (perspective-weaver /
  persona.rs territory), and the adjective vocabulary is its asset.
- They happen to be 12-groupable (`ThinkingStyle::family()` → 36→12,
  distribution 1..6 per family — NOT 3-per-family, so they are also
  not the rung-4 triangle decomposition).

**What made the conflation easy:** persona-36 is 12-groupable, the
word "style" appears in all four spaces, and D-TSC-1's board ruling
`E-STYLE-FAMILY-VS-RUNBOOK-1` labeled the persona-36 "literal NARS
runbooks seeding the rung ladder." That label is CORRECTED by this
demarcation: `StyleFamily::default_runbook()` bridges rung-4 macros to
the *persona vocabulary*, while wearing the name of the *rung-3
tactics*. The mapping pair (`family()`/`default_runbook()`) is
mechanically sound as a 12↔36 bridge WITHIN the persona storyline; only
its "runbook" naming and the board label cross storylines.

## The four spaces, one line each (anti-conflation card)

1. `StyleFamily` (12) — rung-4 thinking macros, orchestration dispatch. Canonical, deduped, frozen ordinals (driver order, Deliberate=0).
2. `recipes::RECIPES` (34) — rung-3 NARS reasoning tactics. THE runbooks. Catalogue in contract; primitives in ndarray `hpc/styles`.
3. 144 verbs (12×12) — rung-2 thinking atoms, the unified predicate surface (`sigma_rosetta` + `verb_table`). Computing, test-pinned.
4. persona-36 adjectives — persona-modeling storyline. Off-ladder, unwired, future work. Keep out of rung vocabulary.

## OPEN / CURRENT (the live items this demarcation leaves)

- **O1 — rung↔content wiring absent.** `RungLevel` knows Pearl depth,
  not content occupants. No rung references `sigma_rosetta`,
  `verb_table`, `recipes`, or `StyleFamily`. The content ladder exists
  only as this ruling.
- **O2 — the rung-4→rung-3 edge (`StyleFamily` → recipe selection)
  does not exist.** A macro choosing which of the 34 tactics fire is
  the true "default_runbook" semantics; today's `default_runbook()`
  points at the persona vocabulary instead. Eventual fix: retarget (or
  twin) the mapping at `recipes::Recipe`, and rename the persona bridge
  honestly (e.g. `default_persona()`).
- **O3 — persona storyline unwired by design.** Future persona work
  picks up the adjective-36 + `thinking-engine/persona.rs`; until then
  no rung/dispatch code should consume it as if it were reasoning.
- **O4 — D-TSC-1 residue (from the cross-session audit):**
  `PlannerStyleExt` re-exported at `thinking::` but still NOT at
  `api::` (the q2 E0599 path); the `#[deprecated]` note on the planner
  alias points at the wrong space ("36-runbook space") and needs the
  corrected wording; the two lab-only name→ordinal tables (`wire.rs`,
  `auto_style.rs`) remain untethered.
- **O5 — probe results to ledger (run 2026-07-14, this session):**
  p64-bridge `STYLES[ord % 12]` is DORMANT (zero external callers;
  dependents import only `CognitiveShader`) — latent-API hazard only;
  free hardening = `#[deprecated]` on `style_by_ordinal` steering to
  by-name. `UNIFIED_STYLES` (driver `engine_bridge.rs`) was
  TETHERED-not-collapsed by D-TSC-1 (names/ordinals/len parity test) —
  answers the MODULE-TABLE census's open question.
- **O6 — triangle structure unbuilt.** Rung 4's autopoiesis triangle
  (frozen × learned × exploration) has its three poles in separate
  subsystems (StepMask/templates, NARS awareness, MUL) but no composed
  macro object. `FUTURE-DESIGN.md` E-THINKING-STYLES-ARE-CLASSES-1
  ("style = class: StepMask × WideFieldMask + rung set + KausalSpec")
  is the landing zone.
- **O7 — the two 144s of rung 2 are divergent (added 2026-07-15,
  cross-repo audit; gates O1).** The rung-2 row above cites
  `sigma_rosetta` + `verb_table` jointly, but their vocabularies
  disagree: ~2/12 root/family overlap (become, transform), different
  tense membership (subjunctive/gerund vs 3×continuous +
  Potential-as-Subjunctive), and skewed ordinals on shared tense names
  (habitual 7 vs 9, imperative 9 vs 11). Both ordinal-indexed — a naive
  bridge silently mis-maps. Rung-2 wiring needs either a parity-tested
  mapping or an explicit "atom vocabulary vs parse projection"
  demarcation first. Receipts: `E-RUNG2-TWO-144S-1`; debt row:
  `TD-RUNG2-144-VOCAB-SPLIT`.
- **O8 — a third 0–9 ladder, off-canon (added 2026-07-15).**
  `learning::cognitive_frameworks::Rung` (Noise=0 … Transcendent=9) is a
  meaning-granularity ladder whose interior ordinals all diverge from
  `RungLevel` while the endpoints deceptively agree. Needs the persona-36
  treatment: demarcate as its own storyline or rename. Debt row:
  `TD-THIRD-RUNG-LADDER-LEARNING`.

## Cross-refs

- `.claude/board/EPIPHANIES.md` — `E-RUNG-CONTENT-LADDER-1` (this
  ruling), `E-STYLE-FAMILY-VS-RUNBOOK-1` (carries the corrected label
  via this file), `E-FIVE-STYLE-TABLES-1` (the dedup finding).
- `.claude/v3/MODULE-TABLE.md` ADDENDUM 2026-07-10 (2) — the
  ancestry-pipeline census rows for `verb_table.rs`, `sigma_rosetta.rs`,
  p64-bridge `StyleParams`, `UNIFIED_STYLES`, `auto_style.rs`.
- `.claude/v3/FUTURE-DESIGN.md` — E-THINKING-STYLES-ARE-CLASSES-1.
- `lance_graph_contract::recipes` module doc — the 34-tactic spec
  provenance (ladybug `34_TACTICS_x_REASONING_LADDER`; ladybug is spec
  source, never dependency — ada-rewrite-charter D0).
