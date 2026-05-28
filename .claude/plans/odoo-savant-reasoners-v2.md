# odoo-savant-reasoners-v2 — reshape: `Reasoner` trait → typed composition over `CausalEdge64` + `Tactic` + `callcenter/role_keys`

> **Status:** PROPOSAL. v1 SHIPPED in PR #420 (`D-ODOO-SAV-4`: `Reasoner`
> trait + 4 `*Reasoner` impls + `SavantConclusion` + `SavantSuggestion` +
> `build_conclusion`). v1's "MED on dispatch shape" caveat resolved on review:
> **the Reasoner-trait surface is the wrong shape per the AGI-as-glove + The
> Click litmus tests in CLAUDE.md.** v2 is the architectural reshape — v1
> stays feature-gated + `#[deprecated]` until consumers migrate; the canonical
> path becomes typed composition over the agnostic substrate that already
> exists.
>
> **Confidence:** HIGH on the diagnosis (CLAUDE.md "P-1 The Click" and "P0
> The Stance / AGI-as-glove" verbatim name v1's anti-patterns). HIGH on the
> right-shape vocabulary (`CausalEdge64` + `Tactic` + 33-TSV atoms +
> role-key catalogues are already shipped agnostic primitives — PR #411
> ratified the 34-tactic `Tactic` trait as "the Elixir-like recipe layer
> that later fronts the real fingerprint substrate via cognitive-shader-driver
> with no change to the 34 call sites"). MED on the per-savant typed
> composition declarations (substantial translation pass from
> `.claude/odoo/savants/<N>.md` slot 1/4 + `.claude/odoo/L*.md` business
> semantics into typed `SavantPattern` consts).
>
> **Predecessors:** PR #420 (v1 ship, the surface to be deprecated),
> PR #418 (BindSpace → mailbox-owned SoA; `E-BATON-1`), PR #419 (25
> AXIS-B evidence contracts, `.claude/odoo/savants/<N>.md`), PR #416
> (Odoo savant roster `contract::savants`), PR #414 (`D-ODOO-SAV-1/2/3`
> families + Layer-2 axioms + style wiring), PR #411 (33-TSV atom layer +
> 34-tactic `Tactic` trait + `recipe_kernels`). v1 plan file:
> `odoo-savant-reasoners-v1.md`.
>
> **Anchored iron rules:** AGI-as-glove (P0 — "new capability lands as a
> new column, not a new layer"), The Click P-1 ("free function on a carrier's
> state = reject"), `I-VSA-IDENTITIES` (`callcenter/role_keys.rs` is the
> named future home of the Layer-2 catalogue), `E-BATON-1` (mailbox-owned
> SoA; cross-boundary state is `(u16, CausalEdge64)`), `I-LEGACY-API-FEATURE-GATED`
> (v1 deprecation under feature gate + migration pointer; removal only after
> consumers migrate).

## The diagnosis — why v1 is wrong-shape

v1 added to `lance-graph-callcenter` (and the `Reasoner` trait to
`lance-graph-contract::reasoning`):

- `Reasoner` trait + 4 `*Reasoner` impls (`CustomerCategoryReasoner`,
  `NextBestActionReasoner`, `PostingAnomalyReasoner`, `OtherReasoner`)
- `SavantConclusion` struct + `SavantSuggestion` enum
- `build_conclusion(savant, ctx)` free function

Three doctrine litmus tests in CLAUDE.md name this verbatim:

1. **"New capability lands as a new column, not a new layer"** (P0
   AGI-as-glove). The `Reasoner` trait IS a new layer above the existing
   `FingerprintColumns` / `QualiaColumn` / `MetaColumn` / `EdgeColumn`
   surface.
2. **"Does this add a free function on a carrier's state, or a method on
   the carrier? Free function = reject"** (P-1 litmus). `build_conclusion(savant, ctx)`
   is the named anti-pattern verbatim.
3. **"Wrap the axes in a new struct = breaks the SIMD sweep"** (P0
   AGI-as-glove). `SavantConclusion` + `SavantSuggestion` duplicate
   `CausalEdge64` — the AGI's edge emission IS the conclusion.

The substrate that v1 should have composed instead — all already shipped:

- **`CausalEdge64`** (`crates/causal-edge` v0.2.0, zero-dep): 64-bit
  causal neuron — SPO palette (S/P/O u8 each) + NARS truth (frequency
  u8, confidence u8) + Pearl 2³ mask (3b) + inference mantissa
  (v2 signed i4) + plasticity + W/lens/spare. **The savant's decision
  IS a CausalEdge64 emission**, not a separate type.
- **`Tactic` trait + 34 kernels** (PR #411,
  `lance-graph-contract::recipe_kernels`): "the **Elixir-like** recipe
  layer — common behaviour + 34 hot-dispatchable units, registry-routed
  by tactic id ... later fronts the real fingerprint substrate (atom
  pack/unpack via cognitive-shader-driver) with no change to the 34
  call sites." Each `Tactic::apply(&self, ctx: &mut ThoughtCtx) -> Outcome`
  performs its characteristic operation. Composing a savant = composing
  Tactic dispatches over `ThoughtCtx`.
- **33-TSV atoms** (PR #411, `contract::atoms::CANONICAL_ATOMS`):
  3 Pearl + 9 Rung + 5 Σ + 8 Operations + 4 Presence + 4 Meta = 33,
  bare-metal `I4x32`. Atoms → cognitive-shader-driver → SIMD.
- **Role-key catalogues** (`I-VSA-IDENTITIES`): `grammar/role_keys.rs`
  exists; **`callcenter/role_keys.rs` is explicitly named in the iron
  rule as the future Layer-2 home** but does not yet exist.
- **`Baton`** (`E-BATON-1`, 2026-05-26): cross-boundary state IS
  `(u16 target, CausalEdge64)` discrete handoffs (`CollapseGateEmission`
  in contract). Mailbox-owned SoA is the substrate; `Vsa16kF32` is
  deprecated as a carrier.

## Scope of v2 — the reshape (three deliverable groups)

### Group D — Agnostic composition primitives in `lance-graph-contract`

A typed-composition vocabulary that wires (role-key id + Tactic sequence
over `ThoughtCtx` + `CausalEdge64` emission spec). Zero-dep, sits next to
`atoms` / `recipe_kernels` / `nars` / `causal_edge` references:

- **`SavantPattern`** — the typed declaration: role-key id ref + ordered
  `TacticInvocation` slice + `EdgeEmissionSpec` + `AtomTouchMask` +
  `confidence_floor: f32` + `max_hops: u8` (≤ 128).
- **`TacticInvocation`** — one tactic id (1..=34) + per-tactic parameter
  binding drawn from `ThoughtCtx`. No new trait surface; the existing
  `Tactic::apply` is the dispatch.
- **`EdgeEmissionSpec`** — SPO palette indices (u8 each) + Pearl 2³
  mask template + inference mantissa template (v2 signed i4 per
  `I-LEGACY-API-FEATURE-GATED`). At runtime: combine with `ThoughtCtx`-derived
  NARS truth → `CausalEdge64::pack_v2(...)` row committed to `EdgeColumn`.
- **`AtomTouchMask`** — bitmask of which 33-TSV atoms the savant reads
  (drives shader fan-out planning).

The savant IS its `SavantPattern` const. The shader executes the pattern
over the SoA columns; the existing `Tactic::apply` interface handles
per-tactic dispatch; the emission spec lands as a row in `EdgeColumn`.

### Group E — `crates/lance-graph-callcenter/src/role_keys.rs`

Per `I-VSA-IDENTITIES` Layer-2 catalogue:

- 25 disjoint `Vsa16kF32` slices (one identity per `OdooSavant` from
  `contract::savants`), bipolar ±1 in slice, zero elsewhere.
- Lookup by enum: `pub fn savant_role_key(s: OdooSavant) -> &'static Vsa16kF32`.
- Slice allocation map MUST NOT overlap with `grammar/role_keys.rs` (or any
  future catalogue) — coordinate via a workspace-level slice manifest
  (probably `.claude/knowledge/role-key-slice-allocation.md` — verify or
  create as Group E sub-deliverable).

### Group F — 25 typed per-savant `SavantPattern` consts

One `const FISCAL_POSITION_RESOLVER: SavantPattern = ...` per savant in
`lance-graph-callcenter`, drawn from:

- `.claude/odoo/savants/<Name>.md` slot 4 → tactic composition + emission spec
- `.claude/odoo/savants/<Name>.md` slot 1 → Arrow `EvidenceRef` table mapping
- `.claude/odoo/L<n>-*.md` → lane-level business semantics + atom-touch mask

The 14 NEEDS-INPUT savants ship with declared pattern + a `NEEDS-INPUT`
flag on the EvidenceRef slot. **Composition is correct + immutable**; live
inference is gated on consumer materialization, same boundary as today.

### Group G — Deprecate v1 with migration pointers

Per `I-LEGACY-API-FEATURE-GATED`: feature-gate (do NOT delete) under a
`legacy-reasoner` opt-in feature. Apply `#[deprecated(note = ...)]` with
migration pointers on:

- `lance_graph_contract::reasoning::Reasoner` trait
- `lance_graph_callcenter::{CustomerCategoryReasoner, NextBestActionReasoner,
  PostingAnomalyReasoner, OtherReasoner}`
- `lance_graph_callcenter::{SavantConclusion, SavantSuggestion}`
- `lance_graph_callcenter::savant_reasoners::build_conclusion`

Canonical path: `SavantPattern` resolution via the existing shader-driver
dispatch. Removal in a follow-up PR after woa-rs migrates its
`Reasoner::reason()` call sites.

## Deliverables

| D-id | Scope | Crate | Status |
|---|---|---|---|
| **D-ODOO-SAV-5a** (Group D) | `SavantPattern` + `TacticInvocation` + `EdgeEmissionSpec` + `AtomTouchMask` primitives (zero-dep, in contract) | `lance-graph-contract` | Queued |
| **D-ODOO-SAV-5b** (Group E) | `callcenter/role_keys.rs` with 25 disjoint Vsa16kF32 slices + lookup-by-enum + slice manifest | `lance-graph-callcenter` | Queued |
| **D-ODOO-SAV-5c** (Group F) | 25 `SavantPattern` consts drawn from savant-doc + L-doc curation | `lance-graph-callcenter` | Queued |
| **D-ODOO-SAV-5d** (Group G) | `#[deprecated]` + `legacy-reasoner` feature gate + migration pointers on v1 surface | `lance-graph-contract` + `lance-graph-callcenter` | Queued |
| **D-ODOO-SAV-5e** | End-to-end test: FiscalPositionResolver `SavantPattern` over a synthetic ontology fixture → expected CausalEdge64 row (SPO + NARS + mantissa) | `lance-graph-callcenter` tests | Queued |

## Execution

1. **D-ODOO-SAV-5a first** — composition primitives in contract (additive, zero churn). Ships with this plan + INTEGRATION_PLANS prepend + STATUS_BOARD rows + EPIPHANIES entry (board hygiene).
2. **D-ODOO-SAV-5b in parallel** — `role_keys.rs` is independent of 5a; can land in the same or adjacent PR.
3. **D-ODOO-SAV-5c after 5a + 5b** — per-savant declarations depend on both. Likely one D-id per savant in a Wave (5c-1..5c-25) if the translation is large.
4. **D-ODOO-SAV-5d after 5c** — deprecation marks land once the new path is the canonical path (so the migration pointer can name a real target).
5. **D-ODOO-SAV-5e throughout** — per-D-id tests; the end-to-end test ships with 5c completion as the proof the reshape works.

woa-rs consumer migration is **OUT OF SCOPE** for this plan but UNBLOCKED
by 5d (the deprecation is the migration signal).

## Open questions / risks

- **Tactic kernel coverage.** Are the 34 existing tactics sufficient for
  every savant's AXIS-B decision? Some savants likely need *compositions*
  of multiple tactics; a few may surface needs the 34 don't cover. Scope-creep
  risk: new tactics land as a 35th, 36th kernel in `recipe_kernels` — NOT as
  a new layer. Mitigation: drive from the savant docs, propose new tactics
  only when the existing 34 demonstrably can't compose to the AXIS-B
  decision.
- **Role-key slice allocation coordination.** `grammar/role_keys.rs`
  occupies its slice (need to confirm range); `callcenter/role_keys.rs`
  must claim a disjoint range. Open: is there an existing slice manifest
  I haven't found, or does Group E include creating one?
- **NEEDS-INPUT savants.** 14 of 25 savants have NEEDS-INPUT slot 1.
  Their `SavantPattern` declarations are correct (the composition IS the
  pattern); execution needs consumer-materialized evidence. Pattern +
  emission spec ship in 5c; consumer-feed wiring is out of scope.
- **CausalEdge64 v1 vs v2 mantissa.** `EdgeEmissionSpec` must use v2
  signed-mantissa semantics per `I-LEGACY-API-FEATURE-GATED`. Pearl 2³
  mask + inference mantissa template follow the v2 layout (§6 Option F,
  locked 2026-05-16). `pack_v2` not `pack`.

## Invariants

- **AGI-as-glove**: new capability is a column population (SoA rows /
  `CausalEdge64` emissions), not a new trait surface.
- **`I-VSA-IDENTITIES`**: savant identity in `role_keys` catalogue, content
  (composition spec) in atoms/tactic/emission/AriGraph. Never bundled.
- **The Click P-1 litmus**: every operation lives as a method on its
  carrier; no free function on a carrier's state.
- **`I-LEGACY-API-FEATURE-GATED`**: v1 deprecation under feature gate +
  migration pointer; removal only after consumers migrate.
- **"Suggestion-only, never un-guarded write"** stays in force: the savant's
  `CausalEdge64` emission commits to `EdgeColumn` / AriGraph SPO-G; woa-rs
  reads it as a candidate, applies its AXIS-A guard before write.
- **Board hygiene**: this plan + INTEGRATION_PLANS PREPEND + STATUS_BOARD
  rows + EPIPHANIES entry land in the same commit as D-ODOO-SAV-5a.
