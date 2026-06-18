# mailbox-belief-update-and-substrate-test-v1 — "what did I learn" + two-axis substrate test

> **Status:** CONJECTURE / design. 5+3 council COMPLETE. Rides AFTER the cycle-aware write
> contract (S2.5) → slots **S2.5b**.
> **Date:** 2026-06-18.
> **Parent:** `bindspace-singleton-to-mailbox-soa-v1` §11 (belief-state arc) + `E-SOA-CYCLE-OWNERSHIP`.

---

## Epiphany (less is more)

**"This thought made me smarter — what did I learn" = the NARS-revision delta, exactly.** It is a
**DERIVED read** of `CausalEdge64` (which already stamps `frequency()` / `confidence()`), not a new
stored structure. The witness arc IS the revision log (parent §11.2 iron rule) — a per-consume quad
would be the parallel-struct anti-pattern the rule forbids.

---

## The belief-update (council-ratified)

- **Single-step delta = 0-friction win** (convergence OPPORTUNITY): a `belief_delta(row) -> BeliefDelta`
  read method on `MailboxSoaView`, **zero new column** — `(f,c)` reads off `edges[row]` + a `qualia`
  subtraction. Pure `Copy` microcopy arithmetic, no `&mut self` (data-flow rule).
- **Multi-cycle arc = the already-queued `D-MBX-A3` witness column** (`[CausalEdge64; W]`); NOT free,
  but already planned. Do not pretend the arc is free (convergence DROP of the over-claim).
- **It MUST NOT be the lossy net Δ⟨f,c⟩** (contradiction-cartographer P0). NARS revision *averages*,
  so confidence rises whether evidence **agreed** or **collided** — net Δ cannot tell
  learning-by-revision from learning-by-contradiction-commit. Carry:
  1. the **signed residual `(Δw⁺, Δw⁻)`** (reuse `a2a_blackboard::{support:[u16;4], dissonance:f32}`),
  2. the **regime tag** (Revision vs contradiction-commit — `detect_contradictions` same-S-same-O-diff-relation),
  3. the **qualia delta** (Staunen→Wisdom — `QualiaI4_16D` diff),
  4. net Δ⟨f,c⟩ as the *summary*, never the *record*.
- **Emitted at Commit** (Rubicon col 4), not every consume (trajectory) — the AriGraph SPO-G quad is
  the durable witness; AriGraph is thinking tissue, not a new service (dto-soa: FITS-COLUMN).
- **Second-order (creative-explorer):** the per-item delta accumulates into a per-mailbox
  **competence self-model** — *which kinds of items reliably grow this compartment* — the measurable
  φ-1 Dunning-Kruger surface. The per-item delta is a special case of this self-model.

---

## Two-axis substrate test

Two DIFFERENT claims; want BOTH; they compose (Sudoku = workload, learning-curve = belief-update
measured over it).

### Axis 1 — THROUGHPUT + CORRECTNESS floor: "solved 16M sudoku in 3.4 min"

- **What it is:** a raw speed benchmark on an **exact-oracle constraint workload** (Sudoku has unique
  solutions → correctness is free; 16M instances → hard throughput number; ties to the 16M-envelope /
  1024-prefix / 8–40 min budget).
- **What it certifies:** the field-propagation engine is **fast + correct at scale** (a *speed* claim).
- **What it does NOT certify (critic guard):** it is **NOT** a Weyl/spectral or concentration
  certification. Both cross-domain-synthesizer and theorem-checker graded "Sudoku ↔ edge-Weyl" as
  **[S] RHYME** — the Sudoku graph is 20-regular/vertex-transitive (maximally *degenerate* spectrum),
  the opposite of the degeneracy-*breaking* aperiodic spreading φ-Weyl models. **Drop the "edge-Weyl"
  label.** theorem-checker's salvage: a fine *exact-oracle regression test for monotone-fixpoint
  propagation in the deterministic limit* — state it as throughput, nothing spectral.

### Axis 2 — LEARNING: "improved thinking-style at exponential rate, ceiling at x"

- **What it is:** the belief-update (Axis-1 delta) accumulated into the competence self-model, plotted
  as a learning curve. **"ceiling at x" = the φ-1 humility ceiling** (canon: "φ-1 ceiling = permanent
  humility").
- **What it certifies:** the substrate gets **smarter, not just faster** — the cognition differentiator.
- **Metric is NATIVE** (Δ⟨f,c⟩ + signed residual + qualia — the substrate's own integer metrics).
  **deepeval = DROP** (Python LLM-judge in a Rust-native substrate = firewall breach + Python-inference
  anti-pattern; cross-domain [S]). Borrow at most its trivial `(case, threshold)→pass/fail` *shape* =
  already `cargo test`.

### Secondary harness — goban (not Sudoku, not core dep)

`Sagebati/goban` as a **belief-state** harness: liberties≈support, capture≈belief-death,
ko≈contradiction-cycle are **[H] MECHANISM** (support-counter zero-crossing; history-keyed cycle ban).
**Drop the influence≈resonance leg ([S] rhyme).** Test-harness only; substrate stays game-agnostic.

---

## Sequencing

S2.5 (cycle-aware write contract) → **S2.5b** (this: `belief_delta()` read + Axis-1/Axis-2 harness).
Axis-1 (Sudoku throughput) can land independently as a bench; Axis-2 needs `belief_delta()` +
(for history) `D-MBX-A3`. Cold persistence stays Lance-native (lancedb 0.30 / lance 7).

---

## Council grades (8-agent 5+3, 2026-06-18)

| Agent | Verdict |
|---|---|
| trajectory-cartographer | DERIVED read; arc IS the revision log; emit at Commit not consume |
| dto-soa-savant | FITS-COLUMN (derived read + AriGraph quad); no new layer |
| creative-explorer | RICH — per-item delta is a special case of the competence self-model; 2 axes (ΔF + ΔStaunen) |
| contradiction-cartographer | net Δ⟨f,c⟩ is LOSSY — carry signed residual + regime tag + qualia delta |
| convergence-architect | single-step = OPPORTUNITY (no new column); multi-cycle arc = D-MBX-A3 (not free) |
| cross-domain-synthesizer | Sudoku↔Weyl [S] DROP; Sudoku field-prop TEST-HARNESS-ONLY; goban [H] (drop influence); deepeval DROP |
| theorem-checker | Sudoku↔Weyl [S]; constraint-prop↔VSA [H]-skeleton/[S]-semantics; throughput certifies speed+correctness only |
