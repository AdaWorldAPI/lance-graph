# Spec — EpisodicWitness64 ↔ CausalEdge64 prefetch seam (the white-matter connectome)

**READ BY:** integration-lead, truth-architect; anyone wiring the EW64↔CE64
"fire together → wire together" prefetch loop.
**Status:** SPEC (2026-06-01). **Phase A SHIPPED** (#446/#447/#448 merged);
Phases B–E are GATED / needs-design. This doc consolidates the shipped hot tier,
the remaining phases with their gates, and the **three decisions** that unblock
the next code.
**Grounds:** `E-PLANNING-IS-WHITE-MATTER`, `E-EW64-STRENGTH-IS-CE64-PLASTICITY`
(+ its 2026-06-01 correction), `E-EW64-IS-PREDICTIVE-PREFETCH`,
`E-SUBSTRATE-IS-THE-SCHEDULER`, `E-ARIGRAPH-IS-AN-ISLAND`. Authored from the
2026-06-01 Plan-agent roadmap (all file:lines verified against source there).

---

## 0. Purpose

The episodic connectome is the system's **white matter**: the CE64 (causal) +
EW64 (episodic) edges between the 64k mailboxes (the **grey matter** = compute).
**Planning lives here** — a plan is the Hebbian-strengthened path through the
connectome (recency + plasticity), retrieved by prefetch — **not** an OTP/BEAM
scheduler over isolated processes (`KanbanMove`/`VersionScheduler` are grey-matter
*coordination*, not the planner). This spec covers the hot tier (shipped) and the
loop that grows + prunes the wiring.

```text
GREY (compute)                WHITE (connectome / planning)              COLD (persist + re-prefetch)
64k mailbox SoA      ──fire──►  EpisodicEdges64 hot 4-slot MRU   ──demote──►  DemotionSink impl
(Fingerprint/Qualia/Meta)      (slot order = strength; promote)            (surreal/LanceDB-LIVE "wingman")
                                        ▲                                          │
                                        └───────────── prefetch (re-promote) ──────┘
```

---

## 1. Phase A — SHIPPED (the hot tier), `contract::episodic_edges`

Zero-dep, offline-tested, firewall-clean. The hot 4-slot MRU word + its read/write
surface + the cold-tier exit seam.

| D-id | symbol | what | PR |
|---|---|---|---|
| D-EW64-1 | `EpisodicEdges64(u64)` = 4×`EdgeRef{family:u8, local:u16}` | AriGraph episodic edges; `family==0` intra-basin (~98.6%, #444), `1..=15` cross-family nibble | #446 |
| D-EW64-2 | `promote(e) -> (Self, Option<EdgeRef>)`, `strongest()` | MRU: fire→slot 0, survivors shift down, full+fresh evicts coldest (returned); slot order = strength | #447 |
| D-EW64-3 | `coldest()`, `contains(e)` | the eviction victim (= the next demotion); family-discriminating membership | #448 |
| D-EW64-4 | `trait DemotionSink { fn demote(&mut self, EdgeRef); }`, `promote_into(e, sink)` | the hot→cold exit seam; impls deferred (dependency-inversion, like `MailboxSoaOwner`) | #448 |

**Invariant established (review-verified, exhaustive):** `coldest()` == the edge
`promote` evicts on full+fresh; `promote` never creates a hole; `promote_into`
word == `promote().0` with the sink receiving exactly the eviction.

**What "strength" is here:** *recency only* — the slot index. No per-edge weight
is stored in the 64-bit word. The Hebbian *weight* (Phase B) is the co-addressed
CE64's plasticity; this spec keeps them separate (register-laziness).

---

## 2. The phased plan (B–E) — each GATED or needs-design

### Phase B — Hebbian co-fire (the "wire together" weight-bump) — **GATED**
Goal: when an edge fires, bump the co-addressed `CausalEdge64`'s plasticity toward
hot, so repeated co-firing consolidates a path (procedural memory).
**Gate (3 counts):** (1) `causal-edge` does NOT build offline (anstream uncached);
(2) **plasticity-model mismatch** (Decision 1 below); (3) `I-LEGACY-API-FEATURE-GATED`
— the v1 `PLAST_SHIFT=49` vs v2 `=50` boundary, codex-caught 5× in sprint-11.
A correctly-v2-gated getter/setter exists (`causal-edge::edge.rs:471/483`); driving
it from a co-fire op is the gated work.

### Phase C — cold connectome `DemotionSink` impl (the "wingman") — **GATED on OQ-11.6**
Goal: persist demoted edges + re-prefetch them into the hot tier when their basin
re-activates (`E-SUBSTRATE-IS-THE-SCHEDULER`: surreal-LIVE over the version arc is
the inbound scheduler). The contract seam (`DemotionSink`) is shipped; the impl
(surreal-LIVE or the LanceDB-LIVE fallback) is gated on the `surreal_container`
fork + Lance-6 pin (OQ-11.6) and lancedb-offline. A zero-dep **in-memory
ring** `DemotionSink` reference impl is buildable now if a testable cold tier is
wanted before the substrate lands (optional, low-priority).

### Phase D — `EpisodicWitness64` SoA column — **GATED (offline)**
Goal: make the hot word a real SoA column the cognitive shader borrows
(`soa_view.rs:77` `episodic_witness()` is a deferred accessor). Gate: needs
`cognitive-shader-driver`'s `MailboxSoA<N>` (workspace crate, doesn't build
offline).

### Phase E — comprehension ↔ arcuate ±5 ambiguity wire — **NEEDS-DESIGN**
Goal: the Wernicke faculty (`deepnsm::comprehension`) resolves coreference via the
arcuate ±5 chain before routing fact/story. Blocker: `SentenceStructure`
(`parser.rs:57-66`) carries no ambiguity/candidate signal — only
triples/modifiers/negations/temporals. The sense-candidate source is net-new and
firewall-sensitive (Decision 3).

---

## 3. THREE DECISIONS for @jan (each unblocks a phase)

1. **Plasticity model (unblocks Phase B).** "Co-addressed CE64 plasticity" is
   ambiguous — two models exist: `high_heel::Heel` (`high_heel.rs:168`, W15 byte-3
   **scalar 0=frozen..3=hot**, a 128-byte container field) vs
   `causal-edge::PlasticityState` (`plasticity.rs`, **3 bits, one per S/P/O plane**).
   The hot-tier MRU (Phase A) is unaffected either way. **Which model is the Hebbian
   weight the co-fire bumps?** (Recommendation: the per-plane `PlasticityState` is
   the 64-bit-edge-native one; `Heel` is the container roll-up.)

2. **`RawEdge` / `EpisodicEdge` impl-location, mantissa-only? (unblocks D-EW64-5).**
   `counterfactual.rs:175 EpisodicEdge` is a trait BLOCKED on where its impl lands.
   A zero-dep `RawEdge(u64)` newtype could resolve it **scoped to the inference
   mantissa nibble only — NO plasticity write** (which would re-enter Phase B's
   minefield). **Confirm: build `RawEdge` mantissa-only?**

3. **Sense-candidate source for the comprehension wire (unblocks Phase E).** Where
   do the ±5 disambiguation candidates originate — `vocabulary.rs` neighbors,
   `similarity.rs` top-k, or net-new? They are language-side and must sign-binarize
   to `Binary16K` before crossing (as `arcuate.rs` already does). **Which source?**

---

## 4. Firewall invariants (hold across all phases)

- EW64/CE64 store **opaque handles** (`EdgeRef{family,local}`, `spo_fact_ref:u64`,
  mantissa) — never COCA `rank:u16`. Language stays upstream in DeepNSM.
- `DemotionSink` and any cold-tier impl carry opaque `EdgeRef` only.
- Float (the splat/VSA) stays offline/upstream; the connectome is integer.
- The ~4096 story basins (`local` 12-bit) ≠ COCA-4096 (`OQ-BASIN-COUNT`).

---

## 5. Test plan per phase

- **A (done):** MRU promote/evict/dedup/idempotence; `coldest`==eviction-victim
  invariant; `promote_into` sink routing; family discrimination. 26 tests, green.
- **B:** field-isolation matrix tests across the v1/v2 PLAST_SHIFT boundary
  (MANDATORY per I-LEGACY); co-fire idempotence; plasticity monotonic toward hot.
- **C:** fake-substrate `DemotionSink` round-trip (demote→persist→re-prefetch);
  the in-memory ring as the offline reference.
- **D:** SoA borrow round-trip via `MailboxSoaView` (gated).
- **E:** coreference resolution over a ±5 fixture; firewall (no COCA crosses; only
  `Binary16K` + the resolved referent).

---

*Cross-ref:* `episodic_edges.rs` (Phase A), `causal-edge::{edge.rs:471/483,
plasticity.rs}` + `high_heel.rs:168` (Phase B), `soa_view.rs:77` (Phase D),
`deepnsm::{comprehension.rs, arcuate.rs}` + `parser.rs:57-66` (Phase E),
`I-LEGACY-API-FEATURE-GATED`, OQ-11.6.

---

## 6. Decision resolutions — grounded recommendations (other-session feedback #1 + verified `causal-edge/src/layout.rs`, 2026-06-01; **PENDING @jan ratification** — the decisions remain @jan's)

**Verified locked layout** (corrects §2's imprecise "PLAST_SHIFT 49 vs 50"): per-plane **plasticity = 3 bits (S/P/O) at bits 50–52**; **mantissa = signed i4 at bits 46–49**; **`Heel` = the 128-byte `high_heel` container** (a roll-up, not the 64-bit edge).

**① Plasticity model → per-plane, and DON'T store a graded field.** Reject the `Heel` scalar `0..3` — a CISC collapse: a single scalar throws away SPO directionality (S×P co-fire ≠ P×O co-fire, which the signed mantissa already encodes per-edge). Pick the **per-plane 3-bit (50–52)** — but **binary hot/cold, not graded**: gradedness already lives in three *shipped* signals — **MRU slot-order (#447) × signed mantissa (direction/magnitude) × per-plane hot/cold** — so **compose** "strength" from those dumb signals rather than storing a weight (a stored graded scalar duplicates it and can drift out of sync with slot-order — the RISC answer). Only a *proven* need for a graded-per-plane weight flips this → then 3 graded values in `Heel`, never one collapsed scalar. **Reshapes Phase B:** the co-fire sets the per-plane binary bit; "strength" = a **zero-dep `compose(slot_order, mantissa, per_plane)`** function — NOT a stored weight.

**② `RawEdge` mantissa-only → make it STRUCTURAL, not conventional.** Yes mantissa-only; the addition: `RawEdge` is a newtype that **structurally cannot read/write plasticity/W/truth/temporal** (exposes only the i4 at 46–49) — a *type guarantee*, the way the `MailboxSoaView`/`Owner` split made "read-only" structural. One-writer-per-field, enforced by the type, so "mantissa-only" can't rot into "mantissa-mostly."

**③ Sense-candidate source → reuse the proposer layer; lowest priority.** `vocabulary` neighbors / `similarity` top-k is the right source AND already legal: sense-disambiguation is a **proposal, not an addressing act** (CAM-vs-ANN firewall — similarity lives in the proposer layer). So **don't build net-new** — reuse the proposer machinery (VSA16k role-candidates / aerial `TopKDistance`), emit sense-candidates as proposals carrying ⟨f,c⟩. Firewall: top-k runs upstream in comprehension; the substrate only ever sees the resolved opaque `(family, local)` edge, never the COCA/sense vectors. **Rank last** — least load-bearing for closing the loop.

**Net for the build queue (pending @jan's pick):** ② `RawEdge` mantissa-only **type** and the ①-**compose** `strength` fn are both **buildable now** (contract, zero-dep, offline). The plasticity **WRITE** stays gated (causal-edge offline + I-LEGACY field-isolation tests). ③ is proposer-layer *reuse*, lowest priority. Decisions remain @jan's — these are grounded recommendations, not a resolution.
