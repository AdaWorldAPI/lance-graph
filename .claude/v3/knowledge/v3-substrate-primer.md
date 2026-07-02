# V3 Substrate Primer — the ruled model in one page

> READ BY: every agent doing V3 work (mailbox-warden, envelope-auditor,
> kanban-executor-engineer, template-smith, integration-lead, dto-soa-savant,
> baton-handoff-auditor) and every fresh session entering via `.claude/v3/README.md`.

## Status: FINDING (operator-ruled 2026-07-02; board entries are the canonical text)

This is the orientation doc. Each section names its canonical board entry —
cite THOSE in PRs, not this primer.

---

## 1. The mailbox-kanban model (E-MAILBOX-KANBAN-NO-COLLAPSEGATE)

There is **no CollapseGate** in the singleton-BindSpace sink-in bundle/bind
sense. The unit of cognition ownership is the **Mailbox**:

- **One Mailbox = one Kanban board**, carried as a TENANT (per-mailbox,
  sibling of the per-row `KanbanTenant`). Six phases:
  `Planning → CognitiveWork → Evaluation → Commit → Plan → Prune`.
- Kanban updates execute in **lance-graph-planner** (arm #1, seam
  D-MBX-A6: `Outcome → KanbanMove` in `strategy/style_strategy.rs`) or in
  **SurrealDB-on-kv-lance symbiont mode** (arm #2, `crates/symbiont` is the
  POC shape). `lance-graph-supervisor kanban_actor.rs` is the ractor
  structural owner.
- **ractor is a compile-time ownership dummy** — spawn-only, never a
  hot-path bus. Rust move semantics through the actor model prove
  single-ownership at compile time; that proof is the point, not the runtime.
- The **batch writer fires an AHEAD kanban update on write CAST** — it does
  not wait for the write to land. Ownership delegation is checked via cache
  logic at cast time (cast id vs envelope stamp).
- **Thinking cycles follow a standing async plan** regardless of updates
  (they can't stop thinking; they never wait to be called). The 64k–256k SoA
  **load-balances within a 550 ms net budget** (minus load delays).

## 2. The envelope ownership contract (SHIPPED — `SoaEnvelope::mailbox_owner()`)

Every SoA envelope is **zero-copy from creation to Lance tombstone**
(`docs/architecture/soa-three-tier-model.md` is canonical). The LE contract
inherits mailbox ownership **up and down**:

- **Structural** (compile-time): envelope views are `&self` owner-borrows.
- **Nominal** (runtime): `SoaEnvelope::mailbox_owner() -> MailboxId`
  (default `0` = bootstrap per the zero-fallback ladder) stamps the owner
  for Lance-down provenance and consumer-up write-on-behalf.
- **Iron rule (fleet-wide): every consuming crate writes ON BEHALF OF the
  ractor dummy-owner mailbox.** See `write-on-behalf.md`.

## 3. The DTO ladder (E-DTO-LADDER-OWNERSHIP-SPLIT + E-TWO-RESONANCES-SPLIT)

`thinking-engine → p64 → cognitive-shader-driver`, four rungs:

| Rung | DTO | Role | V3 note |
|---|---|---|---|
| Φ | `StreamDto` | perturbation ingress | ancestor of the standing-async-plan ruling |
| Ψ | `dto.rs::ResonanceDto` | MECHANICAL Morton-tile inverse-pyramid perturbation field | renames → `PerturbationDto` (D-PERT-1, deprecated alias) |
| B | `BusDto` | cognitive commit (what was thought, how settled) | **never grows ownership fields**; batch writer pairs `cast(on_behalf = envelope.mailbox_owner(), payload = BusDto)` |
| Γ | `ThoughtStruct` | persistence | persists via the owner-stamped envelope |

`awareness_dto.rs::ResonanceDto` **keeps its name** — it is the PERSPECTIVAL
resonance (Piaget Three-Mountains: 3D subject/predicate/object HdrResonance +
inferred user model) against the object's self-Gestalt (`classid → ClassView`
= "the object speaks for itself"). "Cascade" (HEEL/HIP/TWIG key tiers) is
canon vocabulary and is NOT renamed.

**L4 learning loop:** converged perturbation residue persists into tenant
lanes via the owner-stamped envelope; the next cycle's standing template
reads the row — the persisted perturbation reshapes the next F landscape.

## 4. Compiled thinking templates (E-COMPILED-THINKING-TEMPLATES)

Orchestration compiles like `askama ↔ ClassView × FieldMask`:

- DSL triple ships in-tree: `elixir-template` (representation + parser) +
  `template-runtime` (deterministic OGAR-action dispatch) +
  `template-equivalence` (replay grading). `ogar-from-elixir` = future
  richer frontend.
- A successful LLM run (**Rig = optional LLM template oracle**) compiles
  DOWN to a deterministic replayable template. `rs-graph-llm` (graph-flow)
  executes instances as replayable sessions **inheriting SoA ownership**.
- A **StepMask** bitmask (sibling of `FieldMask`) selects live steps per
  style/dispatch. Post-P4 the classid custom half indexes the template
  catalogue — 36 thinking styles as 36 lenses over the same canon concept.

## 5. Classid canon-high + the V3 monitor (E-CLASSID-* + E-V3-MARKER-IS-A-MONITOR)

- Composed classid = `[hi u16 CANON concept/domain:appid][lo u16 CUSTOM]`.
  Compose only via `contract::render_classid` / `compose_classid`;
  discriminate only via `classid_canon` / `classid_canon_compat`.
  FORBIDDEN on the composed u32: `as u16`, `& 0xFFFF`, `>> 8`, `>> 16`.
- `0x1000` in the custom half = the **V3-adoption MONITOR**, temporary by
  declaration. P4's trigger is defined: adoption reads 100%; then the marker
  retires and the custom half opens for the 64k render/template catalogue.
- Canon-high is a **clustered index**: domain scans = key-range predicates;
  the adoption monitor + corpus-proof scanner are ONE two-metric
  range-count tool.

## 6. What must NOT be reinvented

| Superseded | Successor |
|---|---|
| Singleton BindSpace + CollapseGate sink-in | per-mailbox `MailboxSoA` + Kanban tenant |
| Baton / `CollapseGateEmission` / `emit()` | zero-copy envelope; Lance columnar I/O is the only writer |
| `Vsa16kF32` as cross-mailbox carrier | intra-compartment bundle math only (The Click, local) |
| Ownership fields on `BusDto` | `SoaEnvelope::mailbox_owner()` + cast pairing |
| Local classid bit math in consumers | contract/ogar-vocab composers |

Cross-ref: `.claude/board/EPIPHANIES.md` (entries named above),
`.claude/handovers/2026-07-02-classid-canon-high-flip-to-v3-thinking-sessions.md`,
`docs/architecture/soa-three-tier-model.md`, `.claude/v3/README.md`.
