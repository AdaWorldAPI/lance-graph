# 2026-10-11 — Novelty → Wisdom: eight-stage 3-bit maturation ladder (D-MATURATION3-0)

**Status:** EXPERIMENTAL probe, test-local. No production change.
`crates/lance-graph-planner/tests/maturation_ladder.rs`

```text
cargo test -p lance-graph-planner --test maturation_ladder
cargo test --release -p lance-graph-planner --test maturation_ladder measure_ -- --nocapture
```

## Codebook (bits 50..52 under an explicitly declared experimental reading)

| code | stage | earned by (executed result) |
|---|---|---|
| 7 | Staunen | violation behind one observation root |
| 6 | Anomaly | violation behind ≥ 2 independent roots |
| 5 | DogEar | `hydrate` → ProposedOnly (gap localized) |
| 4 | Candidates | `hydrate` → Partial |
| 3 | Mechanism | `hydrate` → Hydrated |
| 2 | Falsification | discriminating test: survivor passed AND rival ruled out |
| 1 | Integration | applied as a premise; ≥ 1 new inference resolved through it |
| 0 | Wisdom | from Integration, an ablation in an isolated copy has impact > 0 |

Forward results never move a more mature state backward; only a violation
reopens it. Repeating a completed result never advances it. Unsupported
results abstain.

## Findings

- **VERIFIED-IN-CODE — production hydration has no feedback seam.**
  `pearl::hydrate` reads only the sealed chain; a resolved relation is never a
  premise. Measured on the A → B → C chain: baseline residue 2, test-local
  feedback (production `hydrate` unchanged, over a premise chain) residue 0.
- **MEASURED (release, this container):** `mature` median 1.99–2.83 ns,
  p95 3.0–4.6 ns, 0 allocations over 1e6 transitions. Fixed point, 3 relations
  / 4 observations: baseline ~165 ns, 10 allocs, residue 2; feedback ~157 ns,
  8 allocs, residue 0. The probe loop allocates (premise pool `Vec`); that is
  test scaffolding, not a production cost claim.
- **EXPERIMENTAL — anchor ablation:** cutting derived A in an isolated copy
  raises residue by 2 (B and C lose their bridge); the original population is
  untouched.
- **EXPERIMENTAL — lineage:** a derived premise carries the union of its
  parents' observation roots and never adds one (C: 4 roots from 4
  observations). Two paths from the same roots count 2 supports, not 4.
- **EXPERIMENTAL — self-support:** only resolved relations are premises; a
  two-relation cycle stays open. Disabled, it grounds itself.
- **EXPERIMENTAL — deeper Staunen:** "1 → 4 does not hold" contradicts
  nothing without the anchor chain and is a violation with it.

Disable runs, all red: forward result moving backward; derived dropping a
parent's roots; write leaking into W; write not clearing the old code;
integration without new inferences; Wisdom without ablation impact; existing
provenance accepted; discrimination without ruling out; feedback off; open
relations as premises. (The first T1 fixture hid the leak/clear disables; it
was fixed before recording.)

## CONTRACT DECISION — migration (not taken)

Bits 50..52 are plasticity: `CausalEdge64::learn` gates on them,
`MooreNars16` packs them at bit 50, `CausalEdgeV3` byte 3 carries them, and
D-CE64-STAUNEN-0 measured that another meaning there changes `learn`. The
probe therefore refuses `V2Stamped`, `V3Register`, `V1Legacy` and `Unknown`.
Smallest viable path: a new declared reading (a provenance/generation such
that historic rows keep plasticity), a decision where plasticity lives under
that reading, and a `learn` that does not gate on maturation bits. Not done.

## OPEN

- Feedback in production requires D2 provenance ids (roots here are a
  test-local u64 bitset).
- `Executed` results are constructed by the test from real `hydrate`
  outcomes; mapping recipe outputs onto them is undesigned.
- T5 uses hydration outcomes as the discriminating test, a weak stand-in for
  a Pearl trial; certification is untouched by design.
