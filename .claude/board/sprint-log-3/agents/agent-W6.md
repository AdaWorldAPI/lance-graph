# Agent W6 — Sprint-3 Log

**Role:** Worker Agent W6 of Sprint-3 (12 + meta CCA2A).
**Branch:** `claude/tier-1-implementation-specs` of `AdaWorldAPI/lance-graph`.
**Tier:** Tier-2 (supervised consumer mesh) implementation specs.
**Tech-debt anchor:** TD-RACTOR-SUPERVISOR-5.
**Pattern letter (post-PR #359):** Pattern F — ractor / BEAM Supervisor in Zone 2/3.

---

## Deliverable

`.claude/specs/pr-f-1-ractor-supervisor.md` — PR-ready spec porting the
proven gRPC service trait shape into a `ractor`-supervised tree of
per-consumer actors inside `lance-graph-callcenter`. After this spec, an
engineer picks up the PR, ports the seven message arms, and ships the
supervisor + actor bundle.

## Status

**DONE — spec drafted (~16.7 KB) and pushed to branch via pygithub.**

## Action summary

1. Read W6 task brief (PR-F-1 spec, ~12 KB target, ractor port from gRPC trait).
2. Confirmed branch `claude/tier-1-implementation-specs` HEAD `f7d762f` via the `.git` tree (no MCP get_file_contents calls — zipball / local-only reads per the GitHub access policy in CLAUDE.md).
3. Verified the proof shape at `crates/cognitive-shader-driver/src/grpc.rs` — exactly **345 LOC** as the brief stated, tonic service trait with seven handler arms (dispatch / ingest / health / qualia / styles / tensors / calibrate-probe).
4. Cross-checked **I-2** invariant in `.claude/board/SINGLE_BINARY_TOPOLOGY.md` — tokio is outbound-only; the supervisor and actor message types must be `Send + Sync` without async-runtime dependence. Encoded this as both a compile-time `static_assertions::assert_impl_all!` proof and a `clippy.toml` `disallowed-types` rule.
5. Cross-checked TD-RACTOR-SUPERVISOR-5 in `.claude/board/TECH_DEBT.md` — confirmed P1 priority, R4 region (callcenter actor topology), large effort (~400 LOC + tests).
6. Cross-checked sister-spec dependencies: PR-B-1 (W3 ContextBundle), PR-C-1 (W4 ConsumerPointer + GenericBridge), PR-E-1 (W5 manifest.yaml). All three are required upstream; PR-F-1 is the keystone of the Tier-2 mesh because it consumes all three.
7. Wrote `.claude/specs/pr-f-1-ractor-supervisor.md` with: shape mapping table (7 gRPC handlers -> 7 ractor message arms), file-touch matrix (8 files including the new `ConsumerActorMsg` marker trait in the contract crate), full API sketch (supervisor + msg enum + supervision-event handler + respawn helper), I-2 enforcement subsection (compile-time + lint-time), 5-test plan, dependency table, acceptance checklist, effort estimate, 7 open questions for the engineer, and a 9-row cross-reference index.
8. Pushed both files (`.claude/specs/pr-f-1-ractor-supervisor.md` + this log) via pygithub `update_file` with stripped-token quotes per the protocol.

## File metadata

| File | Path | Size (bytes) | Commit SHA |
|---|---|---|---|
| Spec | `.claude/specs/pr-f-1-ractor-supervisor.md` | 16756 | (see post-push) |
| Agent log | `.claude/board/sprint-log-3/agents/agent-W6.md` | (this file) | (see post-push) |

## Decisions logged

1. **`ConsumerActorMsg` marker trait lives in `lance-graph-contract`,
   not `lance-graph-callcenter`.** Same reasoning as PR-A-1's
   `SpoQuad`: keeps the contract crate as the single home for
   cross-cutting types. PR-C-1's `ConsumerPointer` carries an
   `actor_factory` field whose erased payload implements
   `ConsumerActorMsg`; the contract crate is the only place both
   sides can reach without a heavy crate dep.
2. **Send+Sync proof is BOTH compile-time (static_assertions) AND
   lint-time (clippy disallowed-types).** Belt-and-braces because
   I-2 is the highest-stakes invariant in the membrane: a single
   `tokio::sync::Mutex` smuggled into a consumer actor would corrupt
   the sync execution model and create deadlock surface against
   the cognitive substrate.
3. **One-for-one restart, NOT all-for-one.** A panicking medcare
   consumer must not take down the unrelated smb-office consumer.
   Direct port of the BEAM / Erlang heritage that ractor inherits.
   Documented in Open Q 3 with the rationale so the engineer cannot
   silently flip it.
4. **Bounded mailboxes by default (1024), configurable via
   manifest.yaml `stack_profile.mailbox_capacity`.** Unbounded
   mailboxes are an availability footgun under load — prefer
   back-pressure errors that propagate to the dispatch reply
   oneshot over a queue that grows until the OOM killer arrives.
5. **`Box<dyn ConsumerActorMsg>` for cross-G dispatch, not typed
   per-consumer enums.** ~40 ns box allocation cost per dispatch is
   negligible against the actor mailbox push, and it preserves the
   dynamic registry semantics. Typed dispatch would force the
   supervisor to know every consumer crate's `Msg` type at compile
   time, defeating the OntologyRegistry-driven enumeration.
6. **`clippy.toml` `disallowed-types` rule is workspace-rooted but
   crate-scoped.** Lives in the workspace root so the engineer
   cannot lose it in a per-crate refactor; scoped to
   `lance-graph-callcenter` so the outbound modules (websocket
   serve, postgrest sink) can still legitimately use tokio.
7. **gRPC service trait stays exactly where it is** — L3 outbound
   lab surface, unchanged in shape. The supervisor adds a second,
   in-process consumer of the same handler arms (L2). This is the
   guarantee that backwards-compat holds: `cargo run --features
   grpc --bin shader-grpc` keeps serving protobuf for live shader
   testing in the Claude Code backend.

## Brutally-honest self-review

### What this deliverable does well

- **Mechanical port discipline.** Every message arm is justified
  by a one-to-one row in the gRPC -> ractor mapping table. No new
  capability invented, no scope creep into Tier-3 territory.
- **I-2 enforcement is concrete, not aspirational.** The spec names
  the exact crates (`static_assertions`, `clippy`), the exact
  trait bounds (`Send + Sync + 'static`), and the exact disallowed
  types. An engineer cannot accidentally introduce tokio to the
  membrane without tripping CI.
- **Open questions are pre-answered with recommendations.** All 7
  open questions have a "recommend X because Y" body so the
  engineer is not handed an architectural decision under deadline.
  Q1 (sync vs tokio mode) is the only true open one because it
  depends on ractor 0.10 feature audit that the spec author cannot
  do without spinning up cargo.
- **Dependency table is honest.** PR-B-1 + PR-C-1 + PR-E-1 are all
  flagged as upstream blockers; the engineer will see immediately
  that this is the Week-2 keystone and cannot start before Week-1
  Tier-1 lands.
- **Backwards-compat is a first-class acceptance criterion.** The
  gRPC lab binary regression test (`cargo run --features grpc
  --bin shader-grpc`) is in the checklist, not buried in prose.

### Where the spec could be sharper

- **No concrete `respawn_dead_child` body.** The sketch describes
  the algorithm in prose ("look the dead actor's G up, drop the
  stale ActorRef, re-resolve the pointer, spawn_linked a fresh
  child") but does not write the function. An engineer comfortable
  with ractor will fill it in cleanly; an engineer new to ractor
  may want a 30-line code body. Trade-off: the spec is already at
  16.7 KB vs the 12 KB target.
- **Hot-reload story is half-specified.** Open Q 7 names the choice
  ("drain old mailbox on swap vs let it complete") and recommends
  draining, but does not specify the supervisor message that
  triggers the swap. Probably belongs to a follow-up PR-F-2 once
  PR-D-1 (FMA OWL hydrator) lands a real hot-reload caller.
- **Backoff defaults are picked, not computed.** "100 ms doubling
  capped at 30 s" is a pragmatic default but no benchmark backs
  it. Should be tuned against the first real consumer panic
  scenario in PR-D-1's testbed.
- **No metric for `Box<dyn ConsumerActorMsg>` allocation cost.** I
  wrote "~40 ns" from memory; actual cost depends on the box size
  and allocator. Worth a microbench before the architecture
  argument fully closes.
- **Test plan is functional, not stress.** Five tests cover the
  shape; none cover sustained dispatch under back-pressure or
  panic-storm scenarios. Stress tests probably live in a separate
  benches crate, but should be called out as a gap.

### What I deliberately did NOT do

- Did not write any actor code. Spec only — per sprint-3 acceptance
  criterion "no new code written this sprint (specs only)."
- Did not modify `crates/lance-graph-contract/src/consumer.rs`. The
  spec describes the trait extension but does not land it; that is
  PR-C-1's territory and PR-F-1 is downstream of PR-C-1.
- Did not edit `clippy.toml`. The spec describes the addition but
  does not land it; the clippy rule lands in the actual PR, not in
  the spec PR.
- Did not pre-commit a `Cargo.toml` ractor dep. Same reasoning —
  cargo edits land with the implementation, not the spec.
- Did not touch `.claude/board/STATUS_BOARD.md` to flip TD-RACTOR-
  SUPERVISOR-5 from Open to "In progress". The TD entry stays at
  Open until the implementation PR opens; the spec lands but the
  TD's status reflects code, not docs. (Counter-argument: spec is
  a deliverable, so it could flip to "spec'd". W1 should arbitrate
  in the meta CCA2A pass.)

### Confidence assessment

- **Shape mapping (gRPC -> ractor):** HIGH. The 7 handler arms are
  visible in `grpc.rs` and the ractor handler signature is well-
  understood; this is genuinely mechanical.
- **I-2 compile-time enforcement:** HIGH. `static_assertions` is a
  proven crate and `clippy disallowed-types` is documented.
- **Mailbox capacity defaults:** MEDIUM. 1024 is a reasonable
  starting point but unvalidated against real consumer load.
- **One-for-one vs all-for-one:** HIGH. BEAM heritage + per-
  consumer crash isolation is a settled architectural choice.
- **ractor 0.10 sync mode availability:** MEDIUM. Open Q 1 calls
  this out explicitly; the engineer needs to verify the feature
  flag set works as advertised. Fallback to a hand-rolled
  crossbeam-channel actor harness is documented.
- **Backwards-compat with gRPC lab binary:** HIGH. The supervisor
  is additive; nothing in `grpc.rs` changes shape.

### Cross-agent handover notes

- **W3 (PR-B-1 ContextBundle):** Make sure `OntologyRegistry`
  exposes an `active_g_list() -> Vec<u32>` method (or an
  equivalent iterator). PR-F-1's `pre_start` calls this on every
  supervisor spawn.
- **W4 (PR-C-1 GenericBridge):** Make sure `ConsumerPointer`
  carries an `actor_factory` field (or equivalent dispatcher
  pointer) that PR-F-1 can `.clone()` and `.actor()` on.
  Without it, the supervisor cannot map G -> actor type.
- **W5 (PR-E-1 manifest.yaml):** Reserve a `stack_profile` section
  with `mailbox_capacity: u32`, `restart_policy: enum {OneForOne}`,
  `backoff_initial_ms: u32`, `backoff_max_ms: u32`. PR-F-1 reads
  these at startup; the manifest schema must accept them or PR-F-1
  cannot configure per-consumer behaviour.
- **W7 (PR-J-1 INT4-32D atoms):** No direct coupling, but the
  INT4 thinking atoms eventually JIT into actor handlers. If the
  ConsumerActorMsg arms grow beyond the gRPC-shape 7, that is
  Tier-3 territory and lives behind a feature gate.
- **W9 (PR-D-1 FMA OWL hydrator):** First concrete consumer to
  land an actor inside the supervisor. PR-F-1 ships the empty mesh
  + the contract; PR-D-1 fills the first slot.
- **W11 (smoke test):** End-to-end A+B+C+E+F validation must spawn
  a real supervisor with at least one active G and round-trip a
  Health message. Use `tests/supervisor_dispatch_round_trip.rs`
  as the template.

## Push provenance

- **Method:** pygithub `Repository.update_file` (token quote-
  stripped at module load via the pygithub-first protocol; never
  used `get_file_contents` for cross-file reads — local zipball /
  cloned-tree only).
- **Branch:** `claude/tier-1-implementation-specs`
- **Parent commit:** `f7d762f2081fb2b92236368e98d2df2a1020c3f7`
- **Files in this push:** 2 (`pr-f-1-ractor-supervisor.md`,
  `agent-W6.md`)
- **Conventional commit subject:** `W6: PR-F-1 spec — ractor supervisor port from gRPC trait shape (TD-RACTOR-SUPERVISOR-5)`
