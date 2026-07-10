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
- **F3** *(amended v2 per operator ruling E-STYLE-FAMILY-VS-RUNBOOK-1)* —
  The 36-space and the 12-space are BOTH kept and are DIFFERENT KINDS:
  **12 = abstract FAMILIES for orchestration** (canonical type
  **`StyleFamily`** — the operator's own vocabulary; grep-verified
  unclaimed), **36 = literal NARS RUNBOOKS** (contract keeps the name
  `ThinkingStyle`). Related by `StyleFamily::default_runbook() ->
  ThinkingStyle` and `ThinkingStyle::family() -> StyleFamily`. Runbook
  content (rung sets, KausalSpec, templates) attaches at 36-level in
  D-TSC-3 — families carry NO runbook logic. Deleting either leg is
  collapse; merging them is dilution.
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
| I14 *(v2)* | Dangling `crate::cognitive::ThinkingStyle` imports — DORMANT: behind `#[cfg(feature = "wip")]`, no `mod cognitive` exists | `crates/learning/src/{session.rs:6, moment.rs:3}`, `crates/lance-graph-cognitive/src/{spectroscopy/detector.rs:16, fabric/mrna.rs:10, fabric/butterfly.rs:9}` | wip-gated, never compiled — **non-goal, TECH_DEBT line in S9** |
| I15 *(v2)* | p64-bridge `STYLES: [StyleParams; 12]` in the OLD planner order (Analytical=0…Metacognitive=11) + doc comment claiming "matches lance-graph-planner ThinkingStyle enum order"; indexed `STYLES[ord % 12]` | `crates/p64-bridge/src/lib.rs:162,263` | S2 stales the doc claim; potential live ordinal mismatch vs UNIFIED ords → S2b + TECH_DEBT probe |

**Consumer-list completeness note (v3):** `ThinkingStyle` also greps in
planner `cache/lane_eval.rs`, `pipeline.rs`, `lib.rs`,
`strategy/style_strategy.rs` — verified NON-consumers of I2 (doc-comments
or direct contract-36 imports); stated here so the I2 consumer list's
completeness claim is explicit, not assumed.

**The ordering fact (v2, grounded):** two 12-orderings coexist today —
{planner I2, p64-bridge I15} start at **Analytical=0**; {cognitive_stack I3,
UNIFIED_STYLES I5, auto_style consts} start at **Deliberate=0**.
`StyleFamily` freezes the Deliberate=0 order (I5), so S2's re-export CHANGES
the planner declaration order — safe in planner itself (zero `as u8` casts,
name-matched mapping only; savant-verified) but it invalidates I15's doc claim.

The 12-name set (identical across I2/I3/I5/I10/I11): deliberate, analytical,
convergent, systematic, creative, divergent, exploratory, focused, diffuse,
peripheral, intuitive, metacognitive.

## 3. THE COMMITTED RESOLUTION

**One new zero-dep contract type; every 12-space definition becomes a
re-export or a keyed consumer of it; the 36 stays untouched as ThinkingStyle.**

### S1 *(amended v2)* — contract: `crates/lance-graph-contract/src/style_family.rs` (NEW)

```rust
/// The 12 abstract style FAMILIES for orchestration (coarse dispatch:
/// which KIND of thinking a cycle runs). Distinct in KIND from
/// `thinking::ThinkingStyle` (36 — literal NARS runbooks): families
/// orchestrate, runbooks execute (E-STYLE-FAMILY-VS-RUNBOOK-1).
/// Related by default_runbook()/family(), never merged (spec F3).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StyleFamily {
    #[default]
    Deliberate = 0, Analytical = 1, Convergent = 2, Systematic = 3,
    Creative = 4, Divergent = 5, Exploratory = 6, Focused = 7,
    Diffuse = 8, Peripheral = 9, Intuitive = 10, Metacognitive = 11,
}
```

Ordinal order FROZEN to I5's `UNIFIED_STYLES` order (Deliberate=0; the
driver ships it; re-keying would corrupt the const table). Surface:
`ALL: [StyleFamily; 12]`; `name()` (== YAML card names, I11);
`from_name()`; `from_ordinal(u8) -> Option<Self>`;
`default_runbook() -> ThinkingStyle`; `ThinkingStyle::family() ->
StyleFamily` (total). Registered in `lib.rs`.

**THE CANONICAL 12-ARM TABLE (v2; upgraded v3 — the FOUR-tables finding).**
Grounded reads found FOUR mutually divergent 12→36 tables in the tree:
`THINKING_RECONCILIATION.md:27-44` (36→12 collapse EXEMPLARS, one per
cell), planner I6 (shifted from the doc at cells 9/10/11), driver
`ord_to_thinking_style` (shifted at 8/9/10/11), and — caught by
reviewer-1 in Phase 3 — `parse_style_name` I10 (shifted at 8/9/10:
Reflective/Curious/Empathetic; see S7). **No two of the four fully
agree.** Duplication had already produced live semantic drift — the M9
payoff made concrete. *Adjudication of the driver's own doc claim
("picks the closest semantic match per cluster", driver.rs:963-966,
reviewer-2 C1): if the driver arms were deliberate policy rather than
drift, the OTHER tables would not each have shifted DIFFERENTLY from the
doc and from each other; four independent shifts with no two agreeing is
the signature of independent hand-rolling, not of one policy.* The
canonical `default_runbook()` arms, decided by
**exact-runbook-name-first, doc collapse-exemplar otherwise**:

| ord | family | default_runbook | rationale | planner I6 today | driver today |
|---|---|---|---|---|---|
| 0 | Deliberate | Methodical | doc exemplar | same | same |
| 1 | Analytical | Analytical | exact name | same | same |
| 2 | Convergent | Logical | doc exemplar | same | same |
| 3 | Systematic | Systematic | exact name | same | same |
| 4 | Creative | Creative | exact name | same | same |
| 5 | Divergent | Imaginative | doc exemplar | same | same |
| 6 | Exploratory | Exploratory | exact name | same | same |
| 7 | Focused | Precise | doc exemplar | same | same |
| 8 | Diffuse | Gentle | doc exemplar | same | **Speculative → changes** |
| 9 | Peripheral | Speculative | doc exemplar | **Poetic → changes** | **Curious → changes** |
| 10 | Intuitive | Poetic | doc exemplar | **Curious → changes** | **Reflective → changes** |
| 11 | Metacognitive | Metacognitive | exact name (explicitly OVERRIDES the doc's Reflective exemplar — reviewer-2 D6; the driver and parse_style_name already use Metacognitive, corroborating) | **Reflective → changes** | same |

`ThinkingStyle::family()` (36→12) must satisfy
`f.default_runbook().family() == f` for all 12 (G2); the full 36→12
assignment follows the reconciliation doc's cell semantics with each
default runbook in its own family's cell. The planner/driver arm changes
are DOCUMENTED behavior changes (not silent — G7 pins all 12 arms; the
commit message and the EPIPHANIES finding entry name them).
Tests in-module: G2 identity; 36→12 totality; ordinal/name round-trips;
the 12-name set pinned literally; discriminant pins
(`StyleFamily::Deliberate as u8 == 0` … `Metacognitive as u8 == 11` —
protects I3's `as u8` cast site, savant-2 RISK).

### S2 *(amended v2, re-amended v3 per reviewer-2 D1+D2)* — planner: I2 dies; call sites migrate to `StyleFamily`

Delete the enum in `thinking/style.rs`; add
`pub use lance_graph_contract::style_family::StyleFamily;` and
`#[deprecated(note = "the 12-space is StyleFamily (orchestration families); \
ThinkingStyle is the 36-runbook space in contract::thinking")] pub type
ThinkingStyle = StyleFamily;` — a deprecated TYPE ALIAS, not a bare
rename-re-export: under `-D warnings` every in-crate use of the old name
becomes a build error, so **all planner call sites migrate to
`StyleFamily` in the same commit** (mechanical rename across
thinking/mod.rs, semiring_selection.rs, orchestration_impl.rs, api.rs,
prediction/{mod,scenario}.rs — the savant-verified complete list). The
alias survives only for out-of-crate compat. *Trade-off named (D1): a
bare `as ThinkingStyle` re-export would have recreated the 12-vs-36 name
collision at the two biggest consumer surfaces; the deprecated alias +
same-commit migration kills the collision instead of deferring it.*
Also: `pub trait PlannerStyleExt` (impl for `StyleFamily`) carrying the
planner-local semantics: `.cluster() -> ThinkingCluster` (stays
planner-owned), `.tau_address()`, `.to_scan_params()` (×2000+100 formula
UNCHANGED — F8). Safe re declaration-order change: planner has zero
`as u8`/discriminant casts and I6 matches by name (savant-2 verified).
**S2b (v2):** fix I15's doc comment in `crates/p64-bridge/src/lib.rs:162`
— it claims the OLD planner order; after S2 it must say "p64-bridge
internal historical order (Analytical=0) — NOT StyleFamily order; convert
by name". The potential live mismatch between UNIFIED ords and p64-bridge
`STYLES[ord % 12]` indexing is filed as its own TECH_DEBT probe (S9), not
fixed here.

### S3 — planner: I6 delegates

`planner_style_to_contract(s)` body becomes `s.default_runbook()`.
Arm changes at cells 9/10/11 per the S1 canonical table — DOCUMENTED
behavior change, pinned by G7.

### S4 *(amended v2, re-amended v3)* — thinking-engine: I3 dies; call sites migrate to `StyleFamily`

**Pre-step (savant-3 GAP): add `lance-graph-contract = { path =
"../lance-graph-contract" }` to `crates/thinking-engine/Cargo.toml`** —
the dep does NOT exist today (contract_bridge takes raw `u8` precisely
because of this). Contract is zero-dep, so no tree bloat; this is the
intended dependency direction. Then: delete the enum in
`cognitive_stack.rs`; same D1+D2 treatment as S2 (v3, reviewer-2):
`pub use lance_graph_contract::style_family::StyleFamily;` +
`#[deprecated] pub type ThinkingStyle = StyleFamily;`, with **all
in-crate call sites (ghosts.rs, qualia.rs, contract_bridge.rs) migrated
to `StyleFamily` in the same commit**.
`StyleParams` + calibrated values stay engine-local, keyed via
`StyleParams::for_style(StyleFamily)`. `contract_bridge.rs`:
`contract_style_to_engine` routes through 36-`ThinkingStyle::family()`;
the 5→12 mapping (I7 :103) retargets `StyleFamily`, identical arm choices.
The `config.style as u8` cast at **`contract_bridge.rs:216`** (v3 —
reviewer-1 FIX-P1: v2 text implied the wrong file) is protected by S1's
discriminant pins (I3's current declaration order == the frozen order,
verified).

### S5 *(amended v2)* — thinking-engine: I4 renamed to what it is

`superposition.rs` `ThinkingStyle` → **`DetectedStyle`** (a runtime
detection RESULT, not a taxonomy card). `#[deprecated] pub type
ThinkingStyle = DetectedStyle;` at the old path (alias variant-paths OK
since 1.37; toolchain is 1.95). **ALL in-crate call sites migrate to
`DetectedStyle` in the SAME commit** — savant-2 VIOLATES: under `-D
warnings` a surviving in-crate use of the deprecated alias
(`contract_bridge.rs:105-123 from_superposition`) is a hard build error,
not a soft warning. The alias exists only for out-of-crate compat (none
known).

### S6 *(amended v2)* — driver: I5/I8 keyed by the canonical ordinal

`UNIFIED_STYLES` struct/table values unchanged (hot path, minimal diff);
its "THE canonical mapping" doc comment updated to defer to `StyleFamily`
as the canonical ordinal authority. ADD parity test:
`UNIFIED_STYLES[i].name == StyleFamily::from_ordinal(i).name()`,
`.ordinal == i`, and `UNIFIED_STYLES.len() == StyleFamily::ALL.len()`
(pins the `% 12` at engine_bridge.rs:726 and driver-side modulo sites —
savant-5). `driver.rs` `ord_to_thinking_style()` body becomes
`StyleFamily::from_ordinal(ord)`-then-`default_runbook()` (delete the
local match). **Behavior change at ords 8/9/10** (Speculative→Gentle,
Curious→Speculative, Reflective→Poetic; ord 11 unchanged) — documented,
G7-pinned; this is a RUNTIME path (`ShaderDriver::new` → awareness
bootstrap, savant-2), so the commit message names it explicitly.
`style_ord_to_inference()` stays driver-local; add one doc-comment line
noting its ordinal-range grouping is safe ONLY because StyleFamily
discriminants are frozen (savant-5 RISK, accepted).

### S7 *(REWRITTEN v3 — reviewer-1 BLOCK-P0)* — contract: I10 is the FOURTH divergent table

The v2 claim "output IDENTICAL to today; verified compatible" was FALSE —
an overclaim caught in review. Grounded at
`grammar/thinking_styles.rs:642-644`: `parse_style_name` diverges from the
canonical table at THREE arms (`diffuse→Reflective`, `peripheral→Curious`,
`intuitive→Empathetic` vs canonical Gentle/Speculative/Poetic). It is the
**fourth divergent 12→36 table** (its `metacognitive→Metacognitive` arm
independently supports the exact-name-first rule). New body: try
`StyleFamily::from_name(&lower).map(|f| f.default_runbook())` FIRST, then
fall back to the existing canonical-36-name PASSTHROUGH arms (the
passthrough must be preserved — routing only the 12 family names through
the canonical mapping). This is a **documented behavior change at 3
parsed names**, listed in the commit message alongside the S3/S6 arm
changes and pinned by G7 (all 12 parsed family names asserted literally).
G4 is REWRITTEN accordingly (see §5) — the old G4 would have reported
green through silent corruption since no existing test pins those 3 arms.

### S8 — planner: I9 gains an enum-keyed accessor

`nars_engine.rs`: add `style_vector_for(style: ThinkingStyle/*36*/) ->
Option<StyleVector>` matching the 5 existing vectors (`Empathetic →
empathetic_style()` etc. — 36-space because `empathetic` is not a 12-name).
String fns stay (deprecation optional, savant #4 advises).

### S8b *(v2 note)* — S8 unchanged, keyed by the 36 (runbook space) since
`empathetic` is a runbook name, not a family name — consistent with
E-STYLE-FAMILY-VS-RUNBOOK-1 (StyleVector is runbook content).

### S9 *(amended v2)* — ledger + board (same commit)

M9 row → RESOLVED-for-lance-graph (gate output pasted), citing
**E-THINKING-STYLES-ARE-CLASSES-1 + E-STYLE-FAMILY-VS-RUNBOOK-1** (savant-1
GAP); `docs/TYPE_DUPLICATION_MAP.md` §6 dated addendum;
**`.claude/v3/COMPONENT-MAP.md` (rows :66, :144) + `MODULE-TABLE.md`
(rows :182, :231, :263, :332) dated addenda** (savant-4 — duplication
counts and "reconcile before M9" notes resolve); superseded banner on
`.grok/03_cognitive_layers/thinking_styles.md` (savant-1 — stale
pre-eviction doc); TECH_DEBT rows for I13 (ndarray pair), I14 (wip-gated
dangling imports), I15 (p64-bridge ordinal-mismatch probe: do UNIFIED ords
ever flow into `STYLES[ord % 12]`?); STATUS_BOARD D-TSC-1 flip; AGENT_LOG
council entry (which 5, which 3, verdict counts, v1→v2→v3);
**`.claude/board/LATEST_STATE.md` Contract Inventory row for the new
`style_family` module** (v3 — reviewer-3 W4: mandatory per the
board-hygiene table) and a `PR_ARC_INVENTORY.md` PREPEND when a PR
merges; **EPIPHANIES finding entry: the FOUR-divergent-tables discovery**
(reconciliation doc vs planner I6 vs driver vs parse_style_name — no two
fully agree; duplication had already produced live semantic drift; the
canonical table + the full arm-change list incl. S7's three parse arms).

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
  `thinking.rs` ONLY; all other hits are re-exports/aliases (I14's wip-gated
  dangling imports are `use` statements, not defs — they don't trip G1).
- **G2 (v2):** `f.default_runbook().family() == f` for all 12 families;
  `ThinkingStyle::family()` total (every 36-runbook mapped).
- **G3 (v2):** UNIFIED_STYLES↔StyleFamily parity test green (names +
  ordinals + lengths).
- **G4 (REWRITTEN v3 — reviewer-1):** `parse_style_name` output pinned
  LITERALLY for all 12 family names (new test — the old "existing tests
  pass unmodified" gate was too weak: no existing test pins the 3
  changed arms, so it would report green through silent corruption).
  The canonical-36-name passthrough behavior also pinned (≥3 sample
  passthrough names asserted).
- **G5:** `cargo test -p lance-graph-contract -p lance-graph-planner` and
  `cargo test -p thinking-engine -p cognitive-shader-driver` green; no NEW
  clippy warnings (stash-compare); fmt clean.
- **G6:** contract `Cargo.toml` dependency section diff = empty
  (thinking-engine's Cargo.toml GAINS the contract dep per S4 — that diff
  is expected and G6 does not cover it).
- **G7 (NEW v2, extended v3):** the canonical 12-arm `default_runbook`
  table pinned by an explicit contract test (all 12 arms asserted
  literally); discriminant pins `StyleFamily::X as u8` for all 12; the
  three S7 parse arm changes and the S3/S6 arm changes are all covered by
  the same literal pins.

## 6. PER-SAVANT QUESTION SETS (Phase 1 — ANSWERED 2026-07-10; kept verbatim as the historical record; findings consolidated into the v2 change ledger §7. References to `CascadeStyle` below predate the v2 rename to `StyleFamily`.)

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

## 6b. FORWARD NOTES (v2 — recorded for D-TSC-2/3, no action this PR)

- `StyleOrdinal` is doc-reserved in `.claude/contracts/ripple-dto-contracts.md:95,127`
  (uncoded proposal) — D-TSC-2/3 naming must not collide (savant-1).
- D-TSC-3 may need `.cluster()` (12→4) promoted from `PlannerStyleExt`
  into contract — additive, non-breaking retrofit; accepted (savant-5).
- Out-of-workspace taxonomies (bighorn 36-op, agi-chat) remain unreconciled
  (repos absent) — N6 stands.

## 7. VERSION LEDGER

- **v1** (2026-07-10) — authored pre-council; committed `da5c68c`.
- **v2** (2026-07-10) — Phase-2 consolidation of the 5-savant fan-out
  (fan-out: prior-art / iron-rules / code-truth / cascade-impact /
  different-views, all worker-tier per the council token-economy table,
  parallel; raw outputs banked in AGENT_LOG summary form) + two
  mid-session operator rulings. **Change ledger:**
  1. `CascadeStyle` → **`StyleFamily`** + `default_runbook()`/`family()`
     (operator ruling E-STYLE-FAMILY-VS-RUNBOOK-1; savant-5's naming
     concern independently pointed the same direction).
  2. **The three-tables finding** (consolidator grounding): reconciliation
     doc vs planner I6 vs driver diverge; canonical 12-arm table added to
     S1 with per-arm rationale + the explicit arm-change lists; new G7.
  3. S4 pre-step: thinking-engine LACKS the contract dep — add it
     (savant-3 GAP 2a).
  4. S5: in-crate call-site migration mandatory same-commit — deprecated
     alias would hard-fail `-D warnings` (savant-2 VIOLATES).
  5. S6: driver mapping change acknowledged as RUNTIME behavior change
     (awareness bootstrap path), named in commit + G7-pinned (savant-2
     VIOLATES resolved by documentation + gate, not avoidance).
  6. S1: discriminant pins added (savant-2 RISK re `as u8` cast).
  7. S2b: p64-bridge doc-comment fix + I15 TECH_DEBT probe (savant-4 RISK
     + consolidator grounding of the two-orderings fact).
  8. I14 added (wip-gated dangling imports — dormant, out of scope).
  9. S9 extended: E-id citations, COMPONENT-MAP/MODULE-TABLE addenda,
     .grok superseded banner, three TECH_DEBT rows, EPIPHANIES
     three-tables finding entry (savants 1+4).
  10. G3 extended with length assert (savant-5's `% 12` catch).
  Conflicts resolved: savant-1 "reconciliation matches spec" vs
  consolidator's divergence finding — savant-1 verified ordinal ORDER
  (correct), not the 12→36 arms (spec v1 hadn't enumerated them); both
  observations stand (anti-collapse: neither deleted).
- **v3** (2026-07-10) — RATIFIED post-Phase-4. Reviewer verdicts:
  overclaim-auditor (1×BLOCK-P0, 1×FIX-P1, 2×FIX-P2, 4×PASS),
  dilution-collapse-sentinel (1×FIX-P1, 3×FIX-P2 incl. one merged with
  D1, 3×PASS), firewall-warden (1×BLOCK-P0, 1×FIX-P1, 3×PASS + one
  self-corrected false-block line). **Fix ledger:**
  1. **S7 REWRITTEN** (overclaim BLOCK-P0): `parse_style_name` is the
     FOURTH divergent table (3 arms change, documented; passthrough
     preserved); G4 rewritten from "existing tests pass" (would have
     greenlit silent corruption) to literal 12-arm output pins; the
     finding upgraded three→FOUR tables everywhere.
  2. S2/S4 re-exports → `#[deprecated] pub type` aliases + mandatory
     same-commit in-crate migration to `StyleFamily` (dilution D1+D2 —
     kills the 12-vs-36 name collision instead of re-creating it).
  3. Driver doc-claim adjudicated in S1 (dilution C1: four independent
     shifts, no two agreeing = hand-rolling, not policy); ord-11
     exact-name override of the doc exemplar named in the table (D6).
  4. S4's cast citation corrected to `contract_bridge.rs:216`
     (overclaim FIX-P1).
  5. S9 + LATEST_STATE Contract-Inventory row + PR_ARC-on-merge
     (firewall W4).
  6. §7 v2 ledger tier-wording neutralized (firewall W2; workspace
     precedent: CLAUDE.md Model Policy commits tier vocabulary by
     design, but the reword costs nothing and clears the strict
     reading).
  7. §2 note: the 4 extra `ThinkingStyle` grep hits in planner
     (cache/lane_eval.rs, pipeline.rs, lib.rs, strategy/style_strategy.rs)
     are doc-comments or contract-36 usage, NOT I2 consumers
     (overclaim §2 FIX-P2 — the unstated assumption stated).
  Execution may begin: gates G1-G7 are the acceptance surface.
- **v3-impl note** (2026-07-10, post-execution) — SHIPPED, all gates
  green (G1 = 1 enum + 3 deprecated aliases; 1549 tests across the four
  crates; no new clippy warnings; fmt clean). Implementation surfaced a
  **FIFTH divergent table**: thinking-engine `contract_style_to_engine`'s
  36→12 ordinal-RANGE mapping (S4 always prescribed routing it through
  `family()`, so the change was in-spec, but the divergence magnitude —
  e.g. id 0 Logical→Analytical vs canonical Convergent, Empathic→Intuitive
  vs Diffuse — was unquantified until the code was opened; its 3 pinned
  test arms flipped to canonical as documented witnesses). Finding banked
  as `E-FIVE-STYLE-TABLES-1`. Minor deviations from the letter of the
  spec, all in its spirit: engine `Display for StyleFamily` moved into
  contract (orphan rule; prints the Debug form to preserve engine call
  sites); `EngineStyleExt` extension trait (params/butterfly_sensitivity/
  all) mirrors `PlannerStyleExt` for the engine's calibrated values —
  no new dispatch machinery (F2 intact).
