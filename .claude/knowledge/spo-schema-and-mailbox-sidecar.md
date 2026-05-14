# SPO Schema Adjustment + Mailbox Sidecar Design

> **READ BY:** integration-lead, truth-architect, anyone touching `lance-graph-supervisor::MailboxSoA` / `AttentionMaskActor` / `CallcenterSupervisor` / `cognitive-shader-driver::BindSpace`
>
> **PREREQUISITES:** `causal-edge-64-spo-variant.md`, `causal-edge-64-thinking-engine-variant.md`, `causal-edge-64-synergies-and-pr-trajectory.md`
>
> **Status:** CONJECTURE + design discussion (pending sprint-10 ratification on CSI-1 resolution)

---

## 1. The Three Schema Options

### 1.1 SPO-G (named-graph quad)

Standard RDF quad pattern:

```text
Triplet { Subject, Predicate, Object, Graph: u32 }
```

`Graph` is the **partition key** — every triple lives in exactly one named graph; cross-graph queries are explicit joins. AriGraph's W5 spec ships this:

```rust
pub struct Triplet {
    pub subject:   String,
    pub predicate: String,
    pub object:    String,
    pub g:         u32,         // ← named-graph context
    pub truth:     TruthValue,
    pub timestamp: u64,
    pub pearl_rung: u8,
    pub witness_ref: u64,
}
```

### 1.2 SPO-W (witness tetrahedron)

Per `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §8 — extend SPO with a fourth vertex W:

```text
W = witness / why / worldline / evidence / provenance / weight

Tetrahedron:  S / P / O / W

Hole forms (the 7 questions the system can ask):
  SP_ asks O    (classic SPO completion)
  S_O asks P
  _PO asks S
  SPO asks W   ← "what witness/angle makes this consistent?"
  SPW asks O
  SOW asks P
  POW asks S
```

W is not a partition — it's a **fourth content vertex**. Every fact has an associated witness; the witness IS evidence-tracking + provenance + reasoning-context. The 7 hole-forms convert classical SPO completion into **metacognitive completion**.

### 1.3 Both: SPO-G + SPO-W concurrently

```rust
pub struct Triplet {
    pub subject:   String,
    pub predicate: String,
    pub object:    String,
    pub g:         u32,         // SPO-G: named-graph partition
    pub w:         WitnessRef,  // SPO-W: tetrahedron 4th vertex
    pub truth:     TruthValue,
    pub timestamp: u64,
    pub pearl_rung: u8,
}
```

G and W answer different questions (G = which partition do I belong to? W = what evidence supports me?). They are orthogonal axes; both can coexist.

---

## 2. Pros and Cons

### 2.1 SPO-G alone — pros

| Pro | Detail |
|---|---|
| **Standard RDF pattern** | Direct mapping to Oxigraph; SPARQL queries work out of the box |
| **Hard partitioning** | Tenant isolation, belief-context separation, multi-domain reasoning all use G as the boundary key |
| **Efficient partition filter** | `WHERE g = G_med` is a bitmask AND on the partition index — O(1) lookup |
| **Backward-compatible** | Existing Triplet ships with G; only need to extend other fields |

### 2.2 SPO-G alone — cons

| Con | Detail |
|---|---|
| **No provenance** | "Why is this true?" requires walking back to source documents externally; not queryable within the schema |
| **No reasoning trace** | Counterfactual / abductive paths can't be addressed by SPO-G alone — they need W |
| **Metacognitive completion impossible** | "What angle makes this consistent?" is not expressible as a hole — SPO-G has only `SP_`, `S_O`, `_PO` |

### 2.3 SPO-W alone — pros

| Pro | Detail |
|---|---|
| **First-class provenance** | Every triple references the witness chain that supports it; reasoning over evidence becomes structural |
| **7 hole-forms** | Metacognitive completion — system can ask "what witness justifies this triple?" as a regular query |
| **Pearl-3 native** | Counterfactual edges naturally bind to their observed-counterpart via W; no separate counterfactual table |
| **Anaphora-friendly** | Text-order causality (Relativpronomen binding) maps to W = position-in-witness-chain |

### 2.4 SPO-W alone — cons

| Con | Detail |
|---|---|
| **No partition** | Cross-tenant security isolation has no native layer; must be enforced at ractor-supervisor level |
| **W cardinality unbounded** | A witness corpus grows linearly with experience; needs CAM-PQ-indexed retrieval (see `spo-ontology-format-stack.md`) |
| **Non-standard for RDF tools** | Oxigraph SPARQL doesn't natively understand SPOW; need a translation layer |
| **Risk of W swallowing G** | If W carries the corpus root and the corpus is per-tenant, G becomes implicit-via-W — but this is fragile if the partition rule changes |

### 2.5 SPO-G + SPO-W together — pros

| Pro | Detail |
|---|---|
| **Orthogonal axes** | G is partition (security/multi-tenant); W is content provenance. Each does its job |
| **No information loss** | Both questions answerable: "which partition?" and "what evidence?" |
| **Backward-compatible incremental migration** | G stays; W added as new field; existing queries continue working |

### 2.5 SPO-G + SPO-W together — cons

| Con | Detail |
|---|---|
| **Schema width grows** | Each triple gains 4-8 bytes (W pointer); AriGraph row size increases proportionally |
| **G/W coordination** | The corpus root that W points to must agree with the G it lives in (a witness rooted in G_med can't justify a fact in G_legal without explicit cross-G bridging) |
| **Two indices required** | AriGraph needs both `g_index` and `w_index` for fast filtering on either axis |

---

## 3. Time as Sidecar — Two Interpretations

The user's question: **time as sidecar vs CausalEdge64 as sidecar of cognitive-shader-driver in cycle.**

### 3.1 Time as sidecar (the OLDER framing)

```text
CausalEdge64 carries:
  S/P/O addressing + truth + Pearl rung + direction + plasticity + inference

TIME lives in a separate parallel column in the SoA:
  ┌─────────────────────────────────┐
  │ BindSpace::EdgeColumn:          │
  │   CausalEdge64 row[i]           │
  ├─────────────────────────────────┤
  │ BindSpace::TimeColumn:          │  ← time as sidecar column
  │   u64 timestamp[i]              │
  └─────────────────────────────────┘
```

The 12-bit temporal field in CausalEdge64 was a **truncated cache** of the parallel `TimeColumn`. Edge is the cycle-speed read; TimeColumn is the cycle-speed wider time anchor.

**Pros:**
- 12-bit edge field saves a column lookup for coarse-time decisions
- Cache-line locality (both columns striped together in SoA)
- Drop temporal from edge → recover 12 bits — but lose coarse-time-without-lookup

**Cons:**
- TimeColumn duplicates AriGraph `Triplet.timestamp: u64` at commit time
- Two homes for time = drift risk
- Doesn't generalize to other "cached-in-edge" concepts cleanly

### 3.2 CausalEdge64 as sidecar of cognitive-shader-driver (the CORRECTED framing)

Per CLAUDE.md "AGI-as-glove" doctrine: **CausalEdge64 IS the EdgeColumn of BindSpace.** It IS the sidecar — not the carrier of one.

```text
cognitive-shader-driver::BindSpace = SoA with 4 columns:
  FingerprintColumns (Topic axis)
  QualiaColumn       (Angle axis)
  MetaColumn         (Thinking axis)
  EdgeColumn         (Planner axis) ← CausalEdge64

Each row = one slot of AGI working memory.
CausalEdge64 = the per-row causal proposition that THIS slot is currently reasoning about.
```

The "sidecar" framing inverts: CausalEdge64 doesn't HAVE a time sidecar; **it IS the awareness sidecar of the SoA row itself.** The SoA row's other columns (fingerprint, qualia, meta) carry topic/angle/style; the edge column carries the causal proposition.

Time, then, has three homes:
1. **In CausalEdge64 itself** (temporal field, 12 bits) — coarse cycle bucket for cycle-speed dispatch
2. **In SpoWitnessChain position** (W slot points to a chain; chain position = relative order)
3. **In AriGraph Triplet.timestamp: u64** — canonical wall-clock at commit

**This is the correct framing.** The "time-as-sidecar" was a misreading where time was treated as a peer of CausalEdge64 in the SoA, but the AGI-as-glove doctrine puts CausalEdge64 itself in the sidecar role with time as one of its internal fields (or referenced via W).

---

## 4. Functions CausalEdge64 Needs to Fulfill in Ractor Mailbox Communication

Per sprint-10 W6/W7 specs + PR #366's `CallcenterSupervisor`:

### 4.1 Mailbox payload

`MailboxSoA<N>` (W6 spec) carries per-row CausalEdge64 in the EdgeColumn. The mailbox is the cycle-speed buffer between cognitive cycles. Functions:

| Function | Purpose | Cycle budget |
|---|---|---|
| `push_row(role, sigma, view, intent)` | Enqueue a new compartment with default CausalEdge64 | ~50 ns (SoA write) |
| `dispatch_cycle(cycle: u32)` | SIMD sweep over CausalEdge64 column: `forward()` per row, update truth/direction/plasticity | ~200 ns × N rows |
| `drop_row(id) -> CompartmentReport` | Emit Hebbian rollup (plasticity counter + role + g_slot_at_drop) | ~100 ns |
| `emit_one(id) -> CausalEdge64` | Read one row's edge for outbound message | ~10 ns |

### 4.2 Inter-actor message format

ractor messages between `CallcenterSupervisor`, `SigmaTierRouter`, and downstream consumers carry CausalEdge64 in different ways depending on the tier:

| Tier (per W7 Σ-band) | Edge variant carried | Message wrapper |
|---|---|---|
| **Σ1-Σ5 (Static Reflex)** | SPO-variant | `ReflexEdgeMsg { edge: CausalEdge64, role: RoleId, deadline: Instant }` |
| **Σ6 (Emergent)** | SPO-variant + Hebbian counters | `EmergentEdgeMsg { edge: CausalEdge64, plasticity_delta: u32 }` |
| **Σ7-Σ8 (Twig Branching)** | SPO-variant (cycle-speed dispatch) | `TwigEdgeMsg { edge: CausalEdge64, branch_arm: u8 }` |
| **Σ9-Σ10 (Epiphany/Rubicon)** | SPO-variant + witness chain handle | `EpiphanyWitness { edge: CausalEdge64, from_mailbox: MailboxId, witness_root: WitnessRef }` |

The **8-channel cascade variant** does NOT travel between actors — it's interior to the thinking-engine cycle. Once the cycle commits (L3 → AriGraph), only the SPO-variant edge crosses actor boundaries.

### 4.3 Wire format invariants

When CausalEdge64 traverses an `AuditSink` boundary (per PR #366 `AuditSink` trait):
- The full u64 is captured (no field-level redaction)
- Tenant boundary check fires via `CognitiveBridgeGate` BEFORE wire emission
- Merkle-anchored chain ensures tamper-evidence (per `UnifiedAuditEvent::canonical_bytes` 26-byte invariant)

### 4.4 Cycle-speed dispatch payload

Per the corrected hot-path mental model (`cognitive-shader-driver-thinking-engine-reunification.md`):

```text
thinking-engine MatVec (200-500 ns) → top-k atoms
   ↓
emit_causal_edges(k) → Vec<(u16 target, CausalEdge64_8ch)>   ← interior cascade
   ↓
TIER L3 commit point:
   transcode 8ch edges → SPO-variant edges
   (mapping per `causal-edge-64-synergies-and-pr-trajectory.md` §6.3)
   ↓
mailbox.push_row(spo_edge)                                    ← exterior commit
   ↓
SigmaTierRouter dispatches to AttentionMask + AriGraph
   ↓
AriGraph orchestrator: HashMap.get(entity_index[target]) — O(1), 20-200 ns
   ↓
update Triplet (revise/add/delete)
```

CausalEdge64 (SPO-variant) is the **wire format between cognitive cycles**. The 8-channel variant is the **wire format within one cognitive cycle**, never persisted beyond commit.

---

## 5. Recommended Resolution

For sprint-10 + sprint-11:

| Decision | Recommendation |
|---|---|
| **SPO-G or SPO-W or both?** | **Both, with G as edge-level partition and W as edge-level content vertex.** G stays in AriGraph SPO-G quad; W is added as a 4th vertex with WitnessRef pointer to corpus root. CausalEdge64 (SPO-variant) gains a W-slot field (6-8 bits) pointing into the witness corpus. G stays at AriGraph level (queryable filter) but is implicit in CausalEdge64 (per per-tenant SoA architecture). |
| **Time placement?** | Three homes: (1) coarse 12-bit cycle bucket in CausalEdge64 — but this is the v2 reclaim target; (2) `SpoWitnessChain` chain position; (3) AriGraph `Triplet.timestamp: u64` canonical. **Drop temporal from edge in v2** if v2 ships W-slot + lens, freeing 12 bits. |
| **Sidecar framing?** | CausalEdge64 IS the sidecar (per AGI-as-glove doctrine), not the bearer of one. Other "sidecar" framings (time as sidecar of edge) are misreadings. |
| **Mailbox payload?** | Always SPO-variant edges in ractor messages. 8-channel variant is thinking-engine internal only, transcoded at L3 commit. |
| **Σ-tier routing of CausalEdge64?** | Per W7 spec — different message wrappers per Σ-tier; same SPO-variant edge inside; Σ9-Σ10 additionally carry witness handle. |

---

## 6. Cross-references

- `causal-edge-64-spo-variant.md`, `causal-edge-64-thinking-engine-variant.md`, `causal-edge-64-synergies-and-pr-trajectory.md` — the dual-variant analysis
- `spo-ontology-format-stack.md` — how SPO storage formats relate to G/W/edge
- `ogit-owl-dolce-ontology-compartments.md` — G as compartment vs global addressing
- `cognitive-shader-driver-thinking-engine-reunification.md` — the SoA sidecar role + drift
- `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §8 — SPOW tetrahedron
- `.claude/specs/pr-ce64-mb-4-arigraph-spo-g.md` — W5 AriGraph SPO-G spec
- `.claude/specs/pr-ce64-mb-5-mailbox-soa-attentionmask.md` — W6 MailboxSoA spec
- `.claude/specs/pr-ce64-mb-6-sigma-tier-router.md` — W7 SigmaTierRouter spec
- `.claude/board/sprint-log-10/meta-review.md` — CSI-1 resolution path

---

*Authored 2026-05-14. Pending user ratification of sprint-10 CSI-1 resolution.*
