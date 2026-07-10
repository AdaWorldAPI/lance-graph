# D-TSC-1 SPEC v1 — M9 ThinkingStyle dedup (council-grade, 5+3 protocol)

> **Protocol:** `.claude/agents/5plus3-council.md`. This is the Phase-0 spec.
> The council verifies and hardens; it does not redesign. Version ledger at
> the bottom tracks v1 → v2 (post-savant) → v3 (ratified).
> **Scope:** lance-graph repo. Parent plan: `temporal-markov-and-style-classes-v1`
> Track B row D-TSC-1. Milestone: ENTROPY-MILESTONES **M9**.

## 1. FROZEN DECISIONS (cite-or-VIOLATES; never re-opened on taste)

- **F1** — The canonical survivor lives in `lance-graph-contract` (zero-dep).
  Source: ENTROPY-MILESTONES M9 ("→ contract `thinking.rs` 36-style taxonomy";
  gate "grep non-contract ThinkingStyle defs = re-exports only").
- **F2** — Dispatch stays MetaWord bits; NO new `StyleClass`-like trait; this
  dedup mints no classids (that is D-TSC-2, gated on the batched OGAR mint).
  Source: `E-THINKING-STYLES-ARE-CLASSES-1` (EPIPHANIES 2026-07-10).
- **F3** — The 36-space and the 12-space are BOTH kept: contract-36 keeps the
  name `ThinkingStyle`; the 12-space gets ONE canonical type with a DISTINCT
  name (`CascadeStyle`), related to the 36 by explicit total mappings. Basis:
  `THINKING_RECONCILIATION.md` (36 = persona clusters, 12 = cascade-parameter
  clusters — different granularities, not the same taxonomy); deleting either
  leg would be collapse, merging them into one enum would be dilution.
- **F4** — `I-LEGACY-API-FEATURE-GATED`: every retired definition leaves a
  deprecated re-export/alias at its old path; no call site silently changes
  semantics; behavior-pinning tests accompany every routing change.
- **F5** — `lance-graph-contract` gains ZERO dependencies.
- **F6** — Gates: `cargo test` green on the four touched crates; `clippy -D
  warnings` adds no NEW warnings vs pre-change baseline (stash-compare method;
  pre-existing planner `cache/nars_engine.rs` deprecation warnings are known
  tech debt, out of scope); `cargo fmt` clean; board hygiene same-commit.
- **F7** — n8n-rs is EVICTED (2026-06-21): ledger copies #3/#4 in
  `docs/TYPE_DUPLICATION_MAP.md` §6 are moot; no compatibility obligation.
- **F8** — `FieldModulation` divergence (contract ×1000 vs planner ×2000+100
  threshold formulas) is OUT OF SCOPE — a separate ledger row with a real
  semantic decision; conflating it into this PR would hide it.

## 2. INPUT INVENTORY (verified by Explore sweep 2026-07-10; savant #3 re-verifies)

| # | Definition | file:line | Shape |
|---|---|---|---|
| I1 | `ThinkingStyle` CANONICAL 36/6-cluster, τ 0x20–0xC5, `ALL:[;36]` | `crates/lance-graph-contract/src/thinking.rs:23` | `#[repr(u8)]` enum |
| I2 | `ThinkingStyle` planner-local 12/4-cluster, τ 0x20–0xC0, `.to_scan_params()` ×2000+100 | `crates/lance-graph-planner/src/thinking/style.rs:16` | plain enum |
| I3 | `ThinkingStyle` thinking-engine 12 (same 12 names as I2), + `StyleParams` (5×f32) | `crates/thinking-engine/src/cognitive_stack.rs:60` | enum + params struct |
| I4 | `ThinkingStyle` superposition-DETECTED 5 (`Analytical, Creative, Emotional, Intuitive, Diffuse`) | `crates/thinking-engine/src/superposition.rs:30` | enum (runtime inference result, not a card) |
| I5 | `UnifiedStyle` + `UNIFIED_STYLES:[;12]` ("THE canonical mapping. Three type systems, one ordinal") | `crates/cognitive-shader-driver/src/engine_bridge.rs:524,542` | struct + const table; ordinals 0=deliberate … 11=metacognitive |
| I6 | `planner_style_to_contract()` 12→36 (from THINKING_RECONCILIATION table) | `crates/lance-graph-planner/src/orchestration_impl.rs:188` | fn |
| I7 | `contract_style_to_engine(u8)` 36→12 by ordinal; superposition 5→12 | `crates/thinking-engine/src/contract_bridge.rs:16,103` | fns |
| I8 | `style_ord_to_inference()` 12→5 NARS; `ord_to_thinking_style()` 12→36 | `crates/cognitive-shader-driver/src/driver.rs:949,967` | fns |
| I9 | `StyleVector` `{name:&str, weights:[f32;8]}`, 5 string-keyed vectors (incl. `empathetic` — a 36-space name) | `crates/lance-graph-planner/src/cache/nars_engine.rs:254` | struct + fns |
| I10 | `parse_style_name()` — 12 YAML card names → contract-36 | `crates/lance-graph-contract/src/grammar/thinking_styles.rs:629` | fn |
| I11 | 12 YAML style cards (the data-level 12-name source) | `crates/deepnsm/assets/grammar_styles/*.yaml` | data |
| I12 | `StyleRegistry` trait, ORPHANED (n8n-era) | `crates/lance-graph-contract/src/jit.rs:63` | trait, zero impls — **M10's row, untouched here** |
| I13 | `PaletteStyle`(6) ↔ `p64::ThinkingStyle`(6 consts) — identical names/order, zero conversion code | ndarray `src/hpc/causal_diff.rs:974` / `crates/p64/src/lib.rs:682` | **OTHER REPO — non-goal, filed** |

The 12-name set (identical across I2/I3/I5/I10/I11): deliberate, analytical,
convergent, systematic, creative, divergent, exploratory, focused, diffuse,
peripheral, intuitive, metacognitive.

## 3. THE COMMITTED RESOLUTION

**One new zero-dep contract type; every 12-space definition becomes a
re-export or a keyed consumer of it; the 36 stays untouched as ThinkingStyle.**

### S1 — contract: `crates/lance-graph-contract/src/cascade_style.rs` (NEW)

```rust
/// The canonical 12-style CASCADE taxonomy (coarse, parameter-cluster space).
/// Distinct from `thinking::ThinkingStyle` (36, persona space) — related by
/// the explicit total mappings below, never merged (spec F3).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CascadeStyle {
    #[default]
    Deliberate = 0, Analytical = 1, Convergent = 2, Systematic = 3,
    Creative = 4, Divergent = 5, Exploratory = 6, Focused = 7,
    Diffuse = 8, Peripheral = 9, Intuitive = 10, Metacognitive = 11,
}
```

Ordinal order is FROZEN to I5's `UNIFIED_STYLES` order (the driver already
ships it; changing it would silently re-key the driver's const table).
Surface: `ALL: [CascadeStyle; 12]`; `name() -> &'static str` (== the YAML
card names, I11); `from_name(&str) -> Option<Self>`;
`from_ordinal(u8) -> Option<Self>`; `to_contract_style() -> ThinkingStyle`
(the 12→36 mapping MOVED from I6 — one authoritative table); and
`ThinkingStyle::to_cascade() -> CascadeStyle` (36→12, total — formalizing
I7's ordinal arithmetic as an explicit match). Registered in `lib.rs`.
Tests in-module: 12→36→12 identity; 36→12 totality; ordinal/name round-trips;
the 12-name set pinned literally.

### S2 — planner: I2 becomes re-export + extension trait

Delete the enum in `thinking/style.rs`; add
`pub use lance_graph_contract::cascade_style::CascadeStyle as ThinkingStyle;`
plus `pub trait PlannerStyleExt` (impl for `CascadeStyle`) carrying the
planner-local semantics: `.cluster() -> ThinkingCluster` (the 4-cluster
grouping stays planner-owned), `.tau_address()`, `.to_scan_params()` (the
×2000+100 formula UNCHANGED — F8). Existing call sites keep compiling via
the re-export + trait import. `ThinkingCluster` and planner `FieldModulation`
stay where they are.

### S3 — planner: I6 delegates

`planner_style_to_contract(s)` body becomes `s.to_contract_style()`.
(Internal fn; callers unchanged; the mapping table lives in contract now.)

### S4 — thinking-engine: I3 becomes re-export; StyleParams stays engine-owned

Delete the enum in `cognitive_stack.rs`; re-export `CascadeStyle as
ThinkingStyle`. `StyleParams` + its calibrated per-style values remain
engine-local (calibration data, not taxonomy), keyed via a
`StyleParams::for_style(CascadeStyle)` match. `contract_bridge.rs`:
`contract_style_to_engine` routes through
`ThinkingStyle36::to_cascade()`; the superposition 5→12 mapping (I7 :103)
retargets `CascadeStyle` with identical arm choices.

### S5 — thinking-engine: I4 renamed to what it is

`superposition.rs` `ThinkingStyle` → **`DetectedStyle`** (it is a runtime
detection RESULT, not a taxonomy card; keeping the name violates M9's gate).
`#[deprecated] pub type ThinkingStyle = DetectedStyle;` at the old path
(variant paths through type aliases compile since Rust 1.37).

### S6 — driver: I5/I8 keyed by the canonical ordinal

`UNIFIED_STYLES` struct/table unchanged (hot path, minimal diff); ADD a
parity test: `UNIFIED_STYLES[i].name == CascadeStyle::from_ordinal(i).name()`
and `.ordinal == i` for all 12. `driver.rs` `ord_to_thinking_style()` body
becomes `CascadeStyle::from_ordinal(ord).map(|c| c.to_contract_style())`
(delete the local 12→36 match). `style_ord_to_inference()` (12→5 NARS)
stays driver-local (a driver policy, not taxonomy).

### S7 — contract: I10 routes through the one mapping

`parse_style_name()` body becomes
`CascadeStyle::from_name(s).map(|c| c.to_contract_style())` — output must be
IDENTICAL to today (existing tests pass unmodified = the parity pin).

### S8 — planner: I9 gains an enum-keyed accessor

`nars_engine.rs`: add `style_vector_for(style: ThinkingStyle/*36*/) ->
Option<StyleVector>` matching the 5 existing vectors (`Empathetic →
empathetic_style()` etc. — 36-space because `empathetic` is not a 12-name).
String fns stay (deprecation optional, savant #4 advises).

### S9 — ledger + board (same commit)

M9 row → RESOLVED-for-lance-graph (gate output pasted);
`docs/TYPE_DUPLICATION_MAP.md` §6 dated addendum (append, don't rewrite);
I13 (ndarray PaletteStyle/p64 pair) filed in `.claude/board/TECH_DEBT.md`;
STATUS_BOARD D-TSC-1 flip; AGENT_LOG council entry; EPIPHANIES only if the
council surfaces a genuine finding.

## 4. NON-GOALS (each with why)

- **N1** FieldModulation dedup — F8; separate semantic decision.
- **N2** M10 StyleRegistry retire/fold — its own milestone; I12 untouched.
- **N3** ndarray I13 pair — other repo, build currently red there; filed.
- **N4** classid mints / style-as-class wiring — D-TSC-2/3, gated on the
  batched OGAR mint; this PR is pure type dedup (F2).
- **N5** StepMask catalogue work — UNBLOCKED by this PR, not part of it.
- **N6** Out-of-workspace taxonomies in THINKING_RECONCILIATION.md (bighorn,
  agi-chat) — repos absent; noted for cross-repo follow-up.

## 5. PRE-REGISTERED GATES

- **G1 (the M9 gate):** `grep -rn "enum ThinkingStyle" crates/` → contract
  `thinking.rs` ONLY; all other hits are re-exports/aliases.
- **G2:** 12→36→12 identity test green; 36→12 total (every variant mapped).
- **G3:** UNIFIED_STYLES↔CascadeStyle parity test green (names + ordinals).
- **G4:** `parse_style_name` existing tests pass UNMODIFIED.
- **G5:** `cargo test -p lance-graph-contract -p lance-graph-planner` and
  `cargo test -p thinking-engine -p cognitive-shader-driver` green; no NEW
  clippy warnings (stash-compare); fmt clean.
- **G6:** contract `Cargo.toml` dependency section diff = empty.

## 6. PER-SAVANT QUESTION SETS (Phase 1; output contract per council card)

**Savant 1 — prior art** (`prior-art-savant` charter):
1. Does any type named `CascadeStyle` / `StyleOrdinal` / `Style12` already
   exist in lance-graph (code or board docs)? 2. Does
   `crates/cognitive-shader-driver/THINKING_RECONCILIATION.md` contradict the
   I6 12→36 mapping this spec canonizes? 3. Does any board/knowledge doc pin
   the 12-name ordinal order DIFFERENTLY than UNIFIED_STYLES? 4. Is there an
   existing E-id or TECH_DEBT row covering this dedup that S9 must cite?

**Savant 2 — iron rules** (`iron-rule-savant`):
1. Does the S2/S4/S5 deprecated-alias plan satisfy I-LEGACY-API-FEATURE-GATED
   (per instance)? 2. Does any step introduce dispatch machinery violating F2?
3. F5 preserved by S1 exactly? 4. Does S6 touch the driver hot path in a way
   that implicates ADR-022 (no hot-path serialization)? 5. AP1–AP9 scan of
   S1–S8.

**Savant 3 — code truth** (runtime-archaeologist charter via general-purpose):
For EACH of I1–I12: CODED/CLAIMED/ABSENT at the stated file:line (drifted
lines: report the real one). Additionally: (a) does thinking-engine already
depend on lance-graph-contract (Cargo.toml evidence)? (b) does
cognitive-shader-driver depend on it? (c) confirm planner `ThinkingStyle`
consumer list (thinking/mod.rs, semiring_selection.rs, orchestration_impl.rs,
api.rs, prediction/{mod,scenario}.rs) is COMPLETE via grep. (d) confirm
enum-variant paths through `type` aliases compile on the pinned toolchain
(cite any existing usage in-tree or a one-line check).

**Savant 4 — cascade impact** (`cascade-impact-savant`):
1. Full file list S1–S9 must touch (mandatory-same-commit vs follow-up).
2. Any consumer OUTSIDE the four named crates importing I2/I3/I4 types
   (ontology, callcenter, deepnsm, examples, benches, tests)? 3. Which v3
   docs (MODULE-TABLE, COMPONENT-MAP) carry rows for the touched files that
   must be updated? 4. Does any doc cite `orchestration_impl.rs:188` or
   `cognitive_stack.rs` ThinkingStyle by line such that S3/S4 stales it?

**Savant 5 — different views** (`creative-explorer-savant`):
1. Strongest alternative to `CascadeStyle` as the name — one candidate, one
   line of case (spec freezes naming unless the case reveals a REAL collision
   or canon conflict). 2. Strongest alternative RESOLUTION (e.g. "12-space
   should be a classid-backed catalogue now") and why the frozen decisions
   already exclude it — or a RISK if they don't. 3. Second-order consequence
   of canonizing the UNIFIED_STYLES ordinal as THE ordinal (what else keys
   off 0..12 implicitly?). 4. What does D-TSC-2 need from this type that
   would be painful to retrofit (one RISK max)?

## 7. VERSION LEDGER

- **v1** (2026-07-10) — this document; authored pre-council.
- v2 — post-Phase-2 consolidation (change ledger appended below).
- v3 — ratified post-Phase-4.
