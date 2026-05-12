# Archetype Transcode Crate Scaffold — v1

> **Status:** In progress (2026-04-24)
> **Owner:** @archetype-specialist, @truth-architect
> **Scope:** NEW crate `crates/lance-graph-archetype/`; deps only on `lance-graph-contract`, `arrow`, `lance` (peer-dep, optional).
> **Depends on:** ADR-0001 Decision 1 (transcode-not-bridge). No runtime dependency on upstream Python.

## Goal

Flip `lance-graph-archetype` from "does-not-exist" to "scaffolded-and-locked." Ship the 6 foundational trait/struct files per ADR-0001 Decision 1. No runtime behaviour yet — this is the LOCKED-MAPPING-INCOMPLETE → LOCKED-AND-SCAFFOLDED pivot.

## Deliverables

- **DU-2.1** — `crates/lance-graph-archetype/Cargo.toml` + `src/lib.rs` + workspace `members` entry in root `Cargo.toml`.
- **DU-2.2** — `src/component.rs: pub trait Component { fn arrow_field() -> arrow::datatypes::Field; fn type_id() -> &'static str; }` plus a test-only `MockComponent` impl asserting trait-object construction.
- **DU-2.3** — `src/processor.rs: pub trait Processor { fn matches(schema: &arrow::datatypes::Schema) -> bool; fn process(batch: arrow::record_batch::RecordBatch) -> Result<arrow::record_batch::RecordBatch, ArchetypeError>; }`.
- **DU-2.4** — `src/world.rs: pub struct World { tick: u64, dataset_uri: String }` with `new() / tick() / current_tick() / fork(&self, branch: &str) / at_tick(&self, tick: u64)` methods. `fork()` and `at_tick()` return `Err(ArchetypeError::Unimplemented { method: "..." })` stubs — docstrings tie to ADR-0001:61-72 / 95.
- **DU-2.5** — `src/command_broker.rs: pub struct CommandBroker { queue: Vec<Command>, ... }` + `pub enum Command { Spawn, Despawn, Update }` — channel-based drain interface with `submit() / drain()` method stubs.
- **DU-2.6** — `src/error.rs: pub enum ArchetypeError { Unimplemented { method: &'static str }, SchemaMismatch { ... }, LanceIo(...) }` with `thiserror::Error` impl.

## Non-goals (explicit)

- Runtime World tick behaviour — stubs only.
- `AsyncProcessor` (Python async equivalent) — future follow-up.
- Entity=`PersonaCard` wiring — DU-2.7, later PR.
- Lance dataset integration beyond the `dataset_uri: String` placeholder — the `fork()` → `lance::checkout(branch)` wiring is DU-2.8.

## Acceptance criteria

- `cargo check -p lance-graph-archetype` compiles cleanly.
- `cargo test -p lance-graph-archetype` — minimum 4 tests pass (one per core trait + one per stub-returns-Unimplemented).
- `cargo test --workspace` — no regressions in other crates.
- Root `Cargo.toml` workspace.members updated.
- `STATUS_BOARD.md` DU-2 row status: Queued → In progress.
- Verdict flip in `.claude/plans/unified-integration-v1.md §6`: Archetype row `LOCKED-MAPPING-INCOMPLETE` → `LOCKED-AND-SCAFFOLDED`.
- `.claude/board/INTEGRATION_PLANS.md` — prepend entry pointing to this plan file.
- `.claude/board/LATEST_STATE.md § Contract Inventory` — add a new block for `lance-graph-archetype` naming the shipped types.
- `.claude/board/EPIPHANIES.md` — prepend short FINDING entry noting scaffold landed.

## Architecture notes

Per ADR-0001 Decision 1 (`.claude/adr/0001-archetype-transcode-stack.md:14-102`): this crate defines its OWN Rust interface. It does NOT mirror the Python `VangelisTech/archetype` API. The Python repo is a DESIGN SPEC, not a runtime dependency. "Upstream Python API unstable" is NOT a blocker.

Per ADR-0001 Decision 3 (`adr/0001-archetype-transcode-stack.md:320-334`): BBB invariant bans `Vsa16kF32` / `RoleKey` / `NarsTruth` / `BlackboardEntry` from crossing the membrane. Archetype types defined in this crate are INSIDE-BBB; they do NOT appear on `CognitiveEventRow`. The scalar projection for "archetype tick happened" is already covered by `CognitiveEventRow.cycle_fp_hi/lo` + `MetaWord`.

Mapping (locked, do not re-litigate):

| ECS concept | lance-graph-contract type | This crate |
|---|---|---|
| Entity | `contract::persona::PersonaCard` | imported, not redefined |
| World | `contract::a2a_blackboard::Blackboard` (runtime) + `World { dataset_uri, tick }` (archetype meta) | the latter is new here |
| Tick | `contract::collapse_gate::GateDecision` fire | imported, not redefined |
| Component | trait in this crate | **DU-2.2** |
| Processor | trait in this crate | **DU-2.3** |
| CommandBroker | struct in this crate | **DU-2.5** |

## File layout

```
crates/lance-graph-archetype/
  Cargo.toml
  src/
    lib.rs              # pub use component::*; etc.
    component.rs        # trait Component
    processor.rs        # trait Processor
    world.rs            # struct World
    command_broker.rs   # struct CommandBroker, enum Command
    error.rs            # enum ArchetypeError (thiserror)
```

## Test layout

Each module gets a `#[cfg(test)] mod tests` with at minimum one test. Minimum 4 tests total:

1. `component::tests::mock_component_has_arrow_field`
2. `processor::tests::trait_object_is_constructable`
3. `world::tests::fork_returns_unimplemented`
4. `world::tests::tick_increments`

## Dependencies

```toml
[dependencies]
lance-graph-contract = { path = "../lance-graph-contract" }
arrow = { workspace = true }
thiserror = { workspace = true }

[dev-dependencies]
# nothing initially
```
