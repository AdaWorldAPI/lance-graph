# Compiled Thinking Templates — the DSL triple, StepMask, oracle compile-down

> READ BY: template-smith, kanban-executor-engineer, any session touching
> elixir-template / template-runtime / template-equivalence /
> cognitive-compiler, rs-graph-llm (graph-flow), or rig.

## Status: FINDING (operator-ruled 2026-07-02 — E-COMPILED-THINKING-TEMPLATES)

---

## The analogy that IS the design

```
askama template  ↔  ClassView × FieldMask     (rendering: masked selection over a class)
elixir DSL       ↔  Template  × StepMask      (thinking:  masked selection over a plan)
```

Orchestration **compiles**. The elixir-like low-code syntax (NOT Elixir) is
the source; the compiled orchestration graph is a **deterministic replayable
template**; a **StepMask** bitmask (sibling of `FieldMask`) selects which
steps are live for a given style/dispatch. Knowledge transfers as compiled
templates, not as prompts.

## The in-tree DSL triple (already shipped, scaffolded)

| Crate | Role | State |
|---|---|---|
| `crates/elixir-template` | representation + parser (`pipeline do step :x end`, `OgarAction` enum) | scaffolded, tests green |
| `crates/template-runtime` | deterministic OGAR-action dispatch (reflex executor) | scaffolded |
| `crates/template-equivalence` | replay grading (Exact / RankOrder / confidence-drift) | scaffolded |
| `crates/cognitive-compiler` | trace → template synthesis surface | scaffolded (synthesis = first probe) |

`ogar-from-elixir` (OGAR side) is the future richer frontend; the triple
above is canonical today. The crate doc of `elixir-template` names the
compile-down direction: *a successful LLM run compiles down to the
deterministic template.*

## The execution + oracle split

- **rs-graph-llm (graph-flow)** executes template INSTANCES as replayable
  sessions. `NextAction` (Continue / ContinueAndExecute / WaitForInput /
  End / GoTo) maps 1:1 onto `OgarAction` transitions and gen_statem
  semantics — the template is a typed state machine.
- **Rig** is the optional **LLM API template oracle**: consulted when the
  deterministic template hits a FailureTicket (the <25% tail); a successful
  oracle run is graded by `template-equivalence` and, when it passes,
  **compiled down** into the template catalogue by `cognitive-compiler`.
- **Ownership inheritance is mandatory:** a graph-flow session executing a
  template inherits the SoA ownership of the mailbox it serves — it writes
  on behalf of `envelope.mailbox_owner()`, never as itself
  (see `write-on-behalf.md`).

## The standing async plan connection

The template IS the standing async plan from the mailbox-kanban ruling:
thinking cycles follow their compiled template without waiting to be called
(the `StreamDto` can't-stop-thinking lineage). A kanban update reprioritizes
which template steps are live (StepMask), it does not wake the thinking.

## Post-P4: the catalogue in the classid

Once the `0x1000` V3 monitor retires (P4), the classid **custom half**
indexes the template catalogue — 36 thinking styles dispatch as 36 lenses
over the same canon concept, exactly like app render skins
(handover finding F2). Style dispatch becomes catalogue addressing:
`(canon concept, custom template-id) → compiled template`.

## Gap list (see INTEGRATION-PLAN W3)

- `StepMask` type does not exist yet (mint in contract, sibling of FieldMask).
- ElixirTemplate → graph-flow `GraphBuilder` adapter (rs-graph-llm side).
- Rig oracle node + equivalence-gated compile-down loop (D-VCW-7 lineage).
- Catalogue dispatch keyed by classid custom half (P4-gated).

Cross-ref: `v3-substrate-primer.md` §4, board `E-COMPILED-THINKING-TEMPLATES`,
STATUS_BOARD `cognitive-compilation-v1` rows (D-CC-*).
