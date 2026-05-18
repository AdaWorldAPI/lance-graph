# cognitive-shader-actor

**Glue #4** from the lance-graph ↔ surrealdb ↔ sea-orm ↔ ndarray integration plan (§6).

Wraps any [`SupervisableShader`](lance-graph-contract) implementor as a first-class
[`ractor::Actor`](https://crates.io/crates/ractor) sitting inside a supervision tree.

---

## Why this crate exists (plan §6)

Today the cognitive crates (`cognitive-shader-driver`, `thinking-engine`, `causal-edge`,
`deepnsm`, `holograph`) run as in-process invocations. For an AGI-style architecture with
let-it-crash recovery and back-pressure they need to live in ractor's supervision tree.

This crate is the **adapter layer** — it is purely additive:

- The cognitive crates themselves are **unchanged**.
- Existing in-process call sites **keep working**.
- Consumers that want supervised behaviour add `cognitive-shader-actor` as a dep;
  consumers that want direct invocation do not.

---

## Module layout

| Module | Contents |
|---|---|
| `messages` | `ShaderMessage<P>` — the mailbox vocabulary (`Apply`, `ApplyDelta`, `Drain`) |
| `actor` | `CognitiveShaderActor<S>` — the `ractor::Actor` adapter; state = `Arc<S>` + `inflight` |
| `supervisor` | `ShaderSupervisor` — one-for-one restart policy backed by `RestartBackoff` + `SupervisionPolicy` |

---

## Supervisor topology (plan §6)

```
ShaderSupervisor (one-for-one, max 5 restarts in 60 s)
├── ThinkingEngineActor
├── CausalEdgeActor
├── DeepNSMActor
├── HolographActor
└── CognitiveShaderDriverActor   ← orchestrates the others
```

A misbehaving shader (panic, OOM, timeout) is restarted by its supervisor without
affecting peer shaders.

---

## Quick start (Sprint 2 — not yet wired)

```rust
use ractor::Actor;
use cognitive_shader_actor::{actor::CognitiveShaderActor, messages::ShaderMessage};
use lance_graph_cognitive::ThinkingEngineShader;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Sprint 2: uncomment once ractor wiring lands.
    // let (actor, _handle) = Actor::spawn(
    //     Some("thinking-1".into()),
    //     CognitiveShaderActor::<ThinkingEngineShader>::new(),
    //     Arc::new(ThinkingEngineShader::new()),
    // ).await?;
    //
    // actor.send_message(ShaderMessage::ApplyDelta {
    //     delta: some_record_batch,
    // })?;
    Ok(())
}
```

---

## Sprint status

This crate is **scaffolded in Sprint 1** (fan-out, worker LG-2).

All method bodies use `unimplemented!("LG-2 stub — Sprint 2")`. The `ractor::Actor` trait
impl block is kept as a doc-comment scaffold in `src/actor.rs` ready for Sprint 2 to
uncomment and wire. Real ractor spawn + supervisor registration lands in Sprint 2 (plan §7).

**Known gaps for Sprint 2:**

- The commented-out `ractor::Actor` impl in `src/actor.rs` must be confirmed against the
  exact ractor version resolved by `"*"` (currently resolves to 0.15.x). In particular:
  `ActorProcessingErr` re-export path, `async fn` trait form vs `async-trait` macro.
- `ShaderSupervisor::spawn_child` return type will become
  `ractor::ActorRef<ShaderMessage<S::Payload>>` once wired.
- The workspace root `Cargo.toml` must add `crates/cognitive-shader-actor` to `members`
  in the orchestrator consolidation pass (iron rule: LG-2 does not modify that file).

---

## Dependencies

| Dep | Purpose |
|---|---|
| `lance-graph-contract = "0.2"` | `SupervisableShader`, `RestartBackoff`, `SupervisionPolicy` |
| `ractor = "*"` | Actor framework: `Actor`, `ActorRef`, `RpcReplyPort` |
| `tokio` (rt-multi-thread, macros) | Async runtime for ractor's tokio backend |
| `anyhow` | Ergonomic error handling |
| `arrow-array` | `RecordBatch` — the default payload type in the shipped topology |

---

## License

Apache-2.0 — same as the rest of lance-graph.
