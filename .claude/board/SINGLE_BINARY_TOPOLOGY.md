# Single-Binary Topology — canonical architecture reference

> **Architectural invariant doc.** Three nested layers, all in one
> binary. Tokio is outbound-only. The CognitiveShader → callcenter
> DTO transition is a compile-time-enforced contract handshake, not
> serialization. Consumers depend on full `lance-graph` (no headless
> mode). Per-row and per-cadence gates are *different primitives*.
>
> **Governance — APPEND-ONLY.** Invariants are immutable once landed.
> Corrections append a `**Correction (YYYY-MM-DD):**` line; do not
> edit prior text. Names introduced here become canonical and
> propagate to plans, ledger rows, PR descriptions, and code.
>
> **READ BY:** every session touching the cognitive substrate, the
> callcenter ecosystem, consumer crates (`medcare-rs`,
> `smb-office-rs`), or any boundary work. Read this BEFORE proposing
> a new "membrane" / "transcode" / "subscriber" plan — the conflation
> this doc settles has cost three different framings already.

---

## TL;DR

```
╔══════════════════════════════════════════════════════════════════╗
║  ONE BINARY                                                      ║
║  (lance-graph + medcare-rs + smb-office-rs all linked together;  ║
║   consumers depend on FULL lance-graph — no headless mode)       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌────────────────────────────────────────────────────────┐      ║
║  │ LAYER 1 — BindSpace 8-column zero-copy SoA             │      ║
║  │                                                        │      ║
║  │   Driver DTO:  CognitiveShader                         │      ║
║  │   Storage:     BindSpace Arrow SoA, zero-copy          │      ║
║  │   Ops:         VSA-1 (Markov-exclusive Vsa16kF32)      │      ║
║  │                BUNDLE-1 (vsa16k_bundle ±5)             │      ║
║  │                NARS-1 / THINK-1                        │      ║
║  │   Timescale:   20–200 ns / op                          │      ║
║  │   Concurrency: sync, single-thread or std::thread      │      ║
║  └────────────────────────────────────────────────────────┘      ║
║                          │                                       ║
║                          │  Arrow column slices.                 ║
║                          │  CognitiveShader DTO ⇄ callcenter DTO ║
║                          │  is the CONTRACT HANDSHAKE — type-    ║
║                          │  checked at link time, no copy.       ║
║                          ▼                                       ║
║  ┌────────────────────────────────────────────────────────┐      ║
║  │ LAYER 2 — Callcenter Palantir-Foundry-equivalent       │      ║
║  │           ecosystem ontology (in-process, sync)        │      ║
║  │                                                        │      ║
║  │   Driver DTO:  callcenter (Wire types, CommitFilter)   │      ║
║  │                                                        │      ║
║  │   Per-row gate (existing, R2 of SoA-DTO FMA map):      │      ║
║  │     CollapseGate / GateDecision { gate, merge }        │      ║
║  │     2-byte microcopy; MergeMode::{Xor,Bundle,          │      ║
║  │     Superposition,AlphaFrontToBack}                    │      ║
║  │     — decides HOW one delta lands per cycle.           │      ║
║  │                                                        │      ║
║  │   Per-cadence accumulator (new, missing primitive):    │      ║
║  │     CycleAccumulator                                   │      ║
║  │     — decides WHEN a batch flushes outbound.           │      ║
║  │     — absorbs the 10,000× speed ratio between          │      ║
║  │       Layer 1 (20–200 ns) and Layer 3 (2–200 ms).      │      ║
║  │                                                        │      ║
║  │   Membrane (transcode + RBAC enforcement):             │      ║
║  │     • DM-2 LanceMembrane (zero-copy projection)        │      ║
║  │     • DM-3 CommitFilter → DataFusion Expr              │      ║
║  │     • POLICY-1 / MEMBRANE-GATE-1                       │      ║
║  │       SMB side SHIPPED (PR #29 SmbMembraneGate)        │      ║
║  │       medcare side PENDING                             │      ║
║  │     • WATCHER-1 / DM-4 / DM-6 — in-process dispatch    │      ║
║  │                                                        │      ║
║  │   Consumers live HERE (in-process, sync):              │      ║
║  │     • medcare-rs    — speaks callcenter DTO contract   │      ║
║  │     • smb-office-rs — speaks callcenter DTO contract   │      ║
║  │     Both depend on FULL lance-graph; both read         │      ║
║  │     BindSpace zero-copy through that dependency.       │      ║
║  └────────────────────────────────────────────────────────┘      ║
║                          │                                       ║
║                          │  CycleAccumulator flush               ║
║                          │  (threshold-driven or pull-driven)    ║
║                          │                                       ║
╠══════════════════════════╪═══════════════════════════════════════╣
║                          │                                       ║
║   ━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ║
║   ║  TOKIO BOUNDARY — OUTBOUND ONLY                           ║  ║
║   ║  (anything past this line LEAVES the process)             ║  ║
║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ║
║                          │                                       ║
║                          ▼                                       ║
║  ┌────────────────────────────────────────────────────────┐      ║
║  │ LAYER 3 — Outbound sinks (past process boundary)       │      ║
║  │                                                        │      ║
║  │   • MySQL sink-in (legacy oracle receiving writes      │      ║
║  │     via tokio + blocking driver)                       │      ║
║  │   • Network egress (HTTP / WS / gRPC responses to      │      ║
║  │     remote clients — DM-5 PhoenixServer + DM-8         │      ║
║  │     PostgRestHandler are SERVING endpoints HERE)       │      ║
║  │   • Probes from external processes (e.g. C# MedCareV2  │      ║
║  │     LanceProbe ring — separate Windows .NET 4.8        │      ║
║  │     desktop calling /api/__parity/csharp)              │      ║
║  │                                                        │      ║
║  │   Timescale:    2–200 ms (10,000× slower than L1)      │      ║
║  │   Concurrency:  tokio runtime drives the slow side     │      ║
║  └────────────────────────────────────────────────────────┘      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## The four invariants

| # | Invariant | Enforcement | Consequence |
|---|---|---|---|
| **I-1** | **Single binary; consumers depend on full `lance-graph`** | Cargo workspace; `medcare-rs` and `smb-office-rs` import `lance-graph-callcenter` and (transitively) `lance-graph-contract`. No headless mode. | `lance-graph-contract` (zero-deps) and `lance-graph-callcenter` (DataFusion / auth-rls-lite) ship together. Their API surfaces cannot diverge — they always link as one binary. |
| **I-2** | **Tokio outbound only** | No `async fn` in cognitive substrate or callcenter membrane. Tokio appears only past the `CycleAccumulator` flush boundary, driving Layer 3. | Inner cycles, the membrane, and consumer crates are all sync, in-process, function-call-driven. `#[tokio::test]` does not appear in those crates. |
| **I-3** | **BBB compile-time enforced** | `external_membrane.rs:7-13`: `Self::Commit` MUST NOT contain `Vsa10k`, `RoleKey`, `SemiringChoice`, `NarsTruth`, `HammingMin`. Those types do not implement Arrow's `Array` trait, so they physically cannot appear in a `RecordBatch` column. The compiler rejects violations — no runtime check needed. | Inner-ontology types are unleakable to Layer 2. The DTO handshake is a relabel + compile-time check, not a runtime serialization. |
| **I-4** | **Per-row and per-cadence gates are distinct primitives** | `collapse_gate::GateDecision` (2-byte microcopy, R2) is per-delta. `CycleAccumulator` (new, missing primitive) is per-batch. Both are gates; they govern different boundaries. Naming them with one term creates a `GATE-2` namespace clash on top of the existing `GATE-1` between `mul::GateDecision` and `collapse_gate::GateDecision`. | Code, plans, and entropy-ledger rows must distinguish *which gate* they mean. The doc pins this so the conflation doesn't recur. |

---

## Layer 1 — BindSpace zero-copy SoA (CognitiveShader DTO)

The AGI substrate. Eight columns of Arrow-typed SoA backing the entire
cognitive cycle. CognitiveShader is the driver DTO: every cycle emits a
`MetaWord` plus a write into the columnar BindSpace.

**Op timescale: 20–200 ns.** All compute is sync, in-thread or via
`std::thread` workers. No serialization across the layer; the next
layer reads the same Arrow buffers via column slicing.

**Anchored entropy-ledger rows** (Section A of `ARCHITECTURE_ENTROPY_LEDGER.md`):
- VSA-1 — `Vsa16kF32` newtype (Markov-exclusive substrate)
- BUNDLE-1 — `vsa16k_bundle` (Markov ±5 superposition; SHIPPED PR #243)
- NARS-1 — six-copy collapse to single `nars` crate (entropy-cluster 17)
- THINK-1 — four-copy collapse to contract-36 (entropy-cluster 24)

**Anchored plans:**
- `bindspace-columns-v1` — Columns E/F/G/H (Phase 1 H shipped #272)
- `elegant-herding-rocket-v1` — Phase 1 shipped #210; Phase 2 D5/D7
  shipped #243; D2/D3/D8/D10 queued
- `unified-integration-v1` — DU-0..DU-5 mapping to existing types
- `thought-cycle-soa-awareness-integration-v1` — PRs 1-10 (plan #335 active)

---

## Layer 2 — Callcenter Foundry-equivalent ecosystem ontology

The same data, viewed through the callcenter contract DTO. This is
where the membrane lives, where consumers attach, and where both
gates fire. Sync, in-process, zero-copy view over Layer 1.

### Per-row gate: `collapse_gate::GateDecision`

**Existing primitive.** R2 of the SoA-DTO FMA map. 2-byte microcopy.

```rust
pub struct GateDecision {
    pub gate:  u8,         // 0=Flow, 1=Block, 2=Hold
    pub merge: MergeMode,  // Xor / Bundle / Superposition / AlphaFrontToBack
}
```

Decides **how a single delta commits** to BindSpace. Fires per-cycle.
`external_membrane.rs::ExternalMembrane::project()` is documented as
"called on every `CollapseGate` fire with `EmitMode::Persist`" — that's
the per-row commit path.

### Per-cadence gate: `CycleAccumulator` (canonical name; missing primitive)

**New, missing primitive.** Decides **when a batch flushes outbound**.
Absorbs the 10,000× speed ratio between Layer 1 (20–200 ns/op) and
Layer 3 (2–200 ms/external-write). Without this, either the cognitive
cycle stalls waiting on slow MySQL/network writes, or the outbound
side drops data under burst.

Conceptual shape (subject to refinement on first implementation):

```rust
pub struct CycleAccumulator<C> {
    pending:        Vec<C>,           // accumulated commits since last flush
    threshold_rows: usize,            // flush at N rows
    threshold_ms:   u32,              // OR flush at T ms
    on_flush:       Box<dyn Fn(&[C])>, // outbound sink driver
}
```

Flush trigger is threshold-driven (rows-since-last-flush >= N OR
ms-since-last-flush >= T) or pull-driven (downstream tokio runtime
calls `flush_now()`). Either way, the flush itself crosses into Layer 3.

**Naming alternatives considered:** `BatchEpoch`, `OutboundEpoch`,
`FlushEpoch`. `CycleAccumulator` chosen for symmetry with
`CollapseGate` (both per-cycle/per-row primitives) plus explicit
"accumulator" semantics. May be refined when the type lands; the
pinning here is *that it must be distinct from `collapse_gate`*, not
the exact final identifier.

### Membrane (transcode + RBAC at the column boundary)

The membrane is the typed boundary between CognitiveShader DTO and
callcenter DTO. The transcode is a compile-time relabel (per I-3),
not a copy. RBAC fires here as a sync gate per row.

**Components:**

- **DM-0 / DM-1** — `ExternalMembrane` trait + `lance-graph-callcenter` skeleton (SHIPPED 2026-04-22)
- **DM-2** — `LanceMembrane::project()` (in progress; Phase A `9a8d6a0` — full Lance append pending DM-4)
- **DM-3** — `CommitFilter → DataFusion Expr` translator (queued)
- **POLICY-1 / MEMBRANE-GATE-1**:
  - SMB side: **SHIPPED PR #29** (`SmbMembraneGate` over `Arc<lance_graph_rbac::Policy>` — newtype-bridges the orphan rule; 13 tests)
  - medcare side: **PENDING** (mirror as `MedCareMembraneGate` over `Arc<medcare_rbac::Policy>`; ~30 LOC)
- **WATCHER-1** — `Dataset::checkout_latest().version()` polled on a `std::thread`; bumps `ArcSwap<u64>` and notifies an `event_listener::Event` (NOT `tokio::sync::watch`, per I-2). Replaces the stub at `lance_membrane.rs:24`.
- **SEAL-1** — `MembraneRegistry::seal()` topo sort (queued upstream)
- **PROJECT-LANCE-1** — `CognitiveEventLanceSink` mirror of `LanceAuditSink` (queued upstream)

### Consumer crates (live in Layer 2)

`medcare-rs` and `smb-office-rs` are **part of the callcenter
ecosystem ontology**. They depend on full `lance-graph` and read
BindSpace zero-copy through that dependency. They speak callcenter
DTO as the contract handshake. They are sync. They are in-process.

The Foundry-equivalent surface that consumers see *is* Layer 2 —
not a separate process, not a wire format. PR #29's `SmbMembraneGate`
gates in-process zero-copy crossings, not network requests.

---

## Layer 3 — Outbound sinks (past tokio boundary)

Everything that **leaves the process**. Tokio is the I/O runtime
that drives this layer; it does not appear inside Layer 1 or Layer 2.

**Sinks:**

- **MySQL sink-in** — legacy oracle receiving writes from the Rust
  binary (medcare-rs / smb-office-rs MySQL reconcilers). Tokio +
  blocking driver. Subject to the parity-clean window discussion in
  `foundry-consumer-parity-v1`.
- **Network egress (serving)** — DM-5 `PhoenixServer` (WS) and DM-8
  `PostgRestHandler` (HTTP) ARE serving endpoints in this layer. The
  callcenter ontology in Layer 2 produces the rows; Layer 3 drives
  them out the wire on tokio's runtime.
- **External probes** — separate processes calling our serving
  endpoints. Example: `MedCareV2 LanceProbe` ring is a Windows
  .NET Framework 4.8 desktop calling `/api/__parity/csharp` over
  HTTP. From the Rust binary's perspective, the parity ingest
  endpoint at `routes/parity.rs:46` is an OUTBOUND serving point in
  Layer 3 (M5-class). From the C# probe's perspective it's an
  outbound calling client. Both sides are tokio-bound on their
  respective runtimes; nothing inside the Rust binary's Layer 1/2 is
  async on the probe's behalf.

**Timescale: 2–200 ms.** 10,000× slower than Layer 1. The
`CycleAccumulator` in Layer 2 is what makes this work — it absorbs
the speed differential by batching many fast inner cycles into one
slow outer flush.

---

## Where each in-flight integration plan lives on the diagram

(See full doc in branch — content abbreviated due to push_files size constraints; full version on commit 384cbe03 of branch claude/lance-datafusion-integration-gv0BF.)

---

## Cross-references

- **`ARCHITECTURE_ENTROPY_LEDGER.md`** Section A row anchors
- **`external_membrane.rs:7-13`** — the BBB invariant text that I-3 enforces
- **`collapse_gate.rs`** — per-row `GateDecision`
- **`INTEGRATION_PLANS.md`** / **`LATEST_STATE.md`** / **`PR_ARC_INVENTORY.md`** / **`CROSS_REPO_PRS.md`**

---

## Maintenance

When a new design proposal arrives that names a "membrane",
"transcode", "subscriber", "external surface", or "boundary":

1. Locate it on the layer diagram. If it doesn't fit a layer, that's
   the first review question.
2. Check it against the four invariants. Violations need explicit
   `**Correction (YYYY-MM-DD):**` justification or rework.
3. Cross-reference it from `INTEGRATION_PLANS.md` to this doc.
4. If it introduces a new gate / accumulator / boundary primitive,
   add it to the I-4 distinct-naming rule before it lands.
