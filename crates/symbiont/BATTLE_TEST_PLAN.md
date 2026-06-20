# Battle-test plan — substrate → SoA migration → cognition → scale

> Companion to `INTEGRATION_PLAN.md`. Every item is a **falsifiable probe**
> (pass/fail), per the workspace probe-first discipline: "compiles" is never a
> pass; a green assertion on real data is. Tracked on STATUS_BOARD
> `symbiont-golden-image-harness`.

## Premise — SoA-first, but sharpened

The operator's intuition ("only makes sense after switching everything from
singleton BindSpace to SoA") is **right for cognition, wrong for the substrate**,
and that distinction IS the sequencing:

- **Cognition probes** (thinking, awareness, free-energy, kanban loop, coref) run
  *on* the four SoA columns (`FingerprintColumns/QualiaColumn/MetaColumn/EdgeColumn`).
  Testing them on a singleton `BindSpace` tests the wrong object → **gated on the
  SoA switch (Phase B).**
- **Substrate probes** (finite/NaN, SIMD parity, determinism, Markov semigroup,
  Jirak noise-floor) are properties of the math/kernels, independent of
  SoA-vs-singleton → **run NOW** as the regression baseline that de-risks the
  migration.

Order: **A (now) → B (migrate, with B2 parity as the gate) → C/D/E on the SoA.**

---

## Phase A — Substrate (run now, pre-SoA)

- **A1 — Finite-closure.** Property test: any graph + perturbation input → every
  cascade round + stats reduction finite. Pass: 10⁶ random topologies, zero
  NaN/Inf escape. *(Partially green: `perturbation_shape_is_always_finite`;
  D1's `grid_to_noderows_is_always_finite_and_roundtrips` extends it onto the SoA.)*
- **A2 — SIMD parity.** Each `ndarray::simd` primitive: AVX-512 vs AVX2 vs scalar
  bit-identical. Pass: parity test green on all three backends.
- **A3 — Determinism.** Same input → byte-identical output across runs and
  `target-cpu` levels. Pass: hash-equal over 1k seeds.
- **A4 — Markov semigroup (I-SUBSTRATE-MARKOV).** Chapman-Kolmogorov:
  `bundle(bundle(a,b),c) ≈ bundle(a,bundle(b,c))` within the JL bound. Pass:
  deviation ≤ e^(−d) envelope at d=16384.
- **A5 — Jirak noise-floor.** Calibrate σ-thresholds against weak-dependence
  (Jirak 2016), not classical Berry-Esseen. Pass: every "N σ above floor" cites a
  Jirak rate.

## Phase B — The SoA migration (the gate) + its parity proof

- **B0 — Inventory** every singleton-`BindSpace` read/write site. Output: a
  migration checklist.
- **B1 — One representation.** Resolve D0: canonical `NodeRow` (key16|edges16|value480)
  is the SoA; `VersionedGraph` SPO-plane and perturbation-sim's f64 `Grid` become
  *views/encoders* over it, not rivals. *(D1 made `Grid` an encoder INTO `NodeRow` —
  the first concrete step.)*
- **B2 — The parity gate (the crux).** For every migrated site: singleton-path
  output == SoA-path output, bit-for-bit, same workload. The equivalence proof
  that makes the migration safe.
- **B3 — Zero-copy-to-Lance roundtrip.** A `NodeRow` SoA envelope writes to Lance
  and reads back identical, no serialization in the hot path (three-tier model,
  `last_active_cycle`). Pass: roundtrip hash-equal. *(D1 proved the 512-B stride
  via `NodeRowPacket::as_le_bytes`; B3 adds the Lance write/read.)*
- **B4 — Key correctness.** Zero-fallback ladder: `classid==0`/`family==0` route
  to defaults; a non-zero mint wakes routing with zero layout-version change.
  Pass: field-isolation matrix test.

## Phase C — Cognition (post-SoA)

- **C1 — VSA round-trip + capacity.** bind→bundle→unbind→cosine recovers
  identities up to N ≤ √d/4 ≈ 32 (I-VSA-IDENTITIES Test 1). Pass: recovery curve
  matches the knee.
- **C2 — Free-energy descent.** The "shader can't-not-think" loop: F descends
  while surprise exists, rests when not. Pass: F(t) strictly decreasing to the
  homeostasis floor.
- **C3 — NARS revision.** Repeated evidence revises ⟨f,c⟩ toward the φ-1 ceiling,
  never past. Pass: confidence asymptotes below φ-1.
- **C4 — Memory-as-tissue.** Coref resolves via `graph.nodes_matching` +
  `episodic.retrieve_similar`; ablating the graph flips the outcome. Pass:
  ablation test.

## Phase D — Integration (the five-crate runtime edges)

- **D1 — Grid→NodeRow bridge.** ✅ **SHIPPED** (`bridge.rs`): cascade result
  encodes onto SoA `NodeRow`, NaN-free, 512-B stride. First runtime edge.
- **D2 — Kanban loop end-to-end.** `LanceVersionScheduler` (a real ractor actor)
  → `KanbanMove` → jitson formula → `MailboxSoaView` write → Lance commit. The 2
  missing arrows of 5. Pass: a perturbation step dispatched *through* the scheduler.
- **D3 — surrealdb-on-Lance.** kv-lance persists + queries a `NodeRow` batch.
  Pass: write-then-query roundtrip.
- **D4 — OGAR ontology → ClassView.** A `classid` resolves to its `ClassView`.
  Pass: one DDL class dispatches.

## Phase E — Scale (the 16K-cores vision, honestly)

- **E1 — Spain-grid acceptance gate.** Every Spanish electricity node as a
  `NodeRow`; cascade sweeps them in parallel, NaN-free; clippy+machete clean.
  First instance of "N real nodes updating in parallel on the actual substrate"
  — the degenerate first case of the 16,384-core picture. (D1 is the 64-bus toy
  version; E1 needs an in-tree Spain fixture — currently only network-fetched
  `examples/iberian.rs` exists.)
- **E2 — Parallel SoA sweep.** N nodes, one SIMD sweep, no aliasing (mailbox-as-
  owner compile-time proof). Pass: throughput ~linear to core count; `cargo`
  proves no data race.
- **E3 — Honest scale claim.** Measure nodes/sec; cite Jirak for any
  "above-noise" claim. Pass: a *measured* number (truth-architect gates this).

## Cross-cutting — the honesty gates (already built)

Run on every phase: `brutally-honest-tester` (P0/P1/P2), `overclaim-auditor`
(said-vs-proven), `truth-architect` (no perf claim without a bench),
`firewall-warden` (no PII / no serialization in the hot path). These keep
"battle-tested" from becoming "asserted."
