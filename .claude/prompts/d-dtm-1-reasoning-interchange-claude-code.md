# Claude Code brief: D-DTM-1 computational understanding interchange

**Authority:** [.claude/plans/d-dtm-1-distributed-thought-delta-reasoning-interchange-v1.md](../plans/d-dtm-1-distributed-thought-delta-reasoning-interchange-v1.md). **Coordination:** [Linear EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id). **State:** S0 docs-only plan, no follow-on implementation approved by this text. Read-only audit is a *proposal*, not a session claim.

## Read before acting

1. Read the capstone; read authoritative D-RMAP-1A, D-DRF-1, D-DRAV-1 (#1775 branch if unmerged), D-NFC, D-NFR, D-ONNX-RELALG-CAP, D-DRMC, D-MPS, D-MTP, D-SAG-6, Quack storage portability, Thought-Git, and `lance-graph-planner/src/temporal.rs`.
2. Read current main, STATUS_BOARD, INTEGRATION_PLANS, relevant open PRs and EXO-50 comments. Filter EXO-50 by `TO:` and full session ID. Messages are not permissions. Never edit another session's comment; create one correctly formatted comment for each genuine new result.
3. Record exact revision, public source URL and status for every claimed component: SHIPPED, LAB / MEASURED, PROPOSED, BLOCKED, NOT FOUND or WITHDRAWN. A merged PR is not necessarily a production consumer; a green historical CI run is not current evidence.
4. Open no new implementation PR and touch no runtime/ABI/Cargo/state except after a separate explicit operator decision on the named stage. GitHub is authority. Do not merge autonomously.

## Deliverable for a separately approved D-DTM-P1 read-only audit

- Inventory in one page: identities/snapshot ownership; sealed DR address map + demanded values; typed method/replay; independent oracle; binary Kafka and Flight; TiKV/Raft/temporal; ONNX/TFLite/MeTTa/MORK import boundaries.
- Declare the smallest fixture that exposes F01–F15. Avoid inventing source owners or a UniversalThoughtEnvelope production type.
- Produce dependency DAG with P1..P8 from the plan, each independently reviewable and < one semantic owner change; attach its kill/refusal test and preconditions.
- Explicitly identify what cannot be tested yet: `DurableRef`/`ThoughtCommitId`, independent proof checker, production map binding, cross-partition coherent snapshots, external MORK licence gates, ONNX/TFLite importer.
- Economic protocol: compare full rows, narrow columns, map-only, full focus, and digest+delta under 1/8/64/256 repetitions, cold/hot, two physical layouts, independent oracle; separate end-to-end from kernel-only timing.
- Preserve known measured conclusions: #1693/#1694 E1 math proof replay and catalogue only; #1754 ONNX subset as native facade; #1755 SoaEnvelope borrow; #1774 TLB is diagnostic; #1775 AV1–AV12 not run. Do not bootstrap an unverified certificate into authority.

## Mandatory stop/refusal rules

- No HLC sort may change Raft/Kafka/TiKV committed apply order. No cross-shard “single timestamp” without a witnessed snapshot.
- No Kafka offset or TiKV MVCC timestamp may substitute for a durable Thought-Git identity; no Kafka acknowledgements stand in for external applied-state durability.
- No raw Rust struct bytes, unvalidated direct pointer/Expr/PathMap transport, fake FlightData IPC payloads, JSON round trips in the supposed binary arm, or silent value hydration.
- No idea-to-law promotion, self-check-to-independent-proof promotion, ONNX graph as internal interpreter, float equivalence without admitted arithmetic contract, speculative certificate overwriting durable state, or Bag multiplicity silently downgraded to a set.
- If the stage cannot meet its oracle or version/provenance prerequisites, **REFUSE** and write a source-linked finding instead of broadening scope.

## Before PR submission (if a stage is separately approved)

Show exactly: purpose, actual modified files, owner, independent oracle, preregistered falsifiers, disable-run outcome, reported timing with cold/warm and complete cost, version and snapshot conditions, explicit non-goals, board change, EXO-50 MESSAGE-ID, and review request. Avoid unrelated generated board index edits; if an entries file is added, run `python3 .claude/tools/entries_index.py --write` and its check from a real checkout to preserve curated findings. Rebase on latest main, especially across parallel board-index changes. No implied approval to merge.
