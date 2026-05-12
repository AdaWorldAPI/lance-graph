# KNOWLEDGE: Composition Discipline

## READ BY: ALL AGENTS

---

## Three concerns, three responses

### 1/3 — Precision cliffs and zombies: KILLED in ndarray certification

The v2.4/v2.5 lane certification already hardened this. Every encoding
has a measured ρ vs f32. γ+φ post-rank is a proven zombie (Lane 3 = Lane 1).
Naive u8 floor (ρ=0.999749) is the endgame gate. Any encoding below it
is worse than doing nothing. These are FINDINGS, not risks.

```
Done: lane verdicts, Fisher 3σ CIs, BCa bootstrap, CHAODA filter,
      naive u8 floor, reality anchors, threshold discipline.
Not a checklist item anymore — it's measured infrastructure.
```

### 2/3 — Basin mismatch and boundary typing: bgz-hhtl-d's job

This is what the 2×BF16 branching proposal (Slot D = CLAM tree path,
Slot V = BF16 value) is designed to solve. The composition question:
does bucket identity (distribution basin) map cleanly to centroid
identity (semantic basin) at the CLAM tree boundary?

```
The answer is Probe M1: does CLAM build a 3-level tree on 256 Jina
centroids where the bucket→centroid mapping is 1:1 at depth 2?
If yes: basin boundary is clean. If no: rethink bucketing.

Every boundary label (rank / value / bucket / address) is implicit
in the encoding-ecosystem.md pipeline map. The rule is simple:
  rank → rank: OK
  value → value: OK
  bucket → address: ONLY if Probe M1 passes
  anything else: wrong, fix it
```

### 3/3 — End-to-end verification: just measure it

```
Boundary                        ρ vs f32      Status
──────────────────────────────  ────────────  ──────────────
f32 → BF16 RNE                 0.999978      DONE (v2.4)
f32 → u8 CDF                   0.999992      DONE (v2.4)
f32 → naive u8                 0.999749      DONE (v2.5)
f32 → scent byte               0.937         DONE
f32 → i16[17] Base17           ?             MEASURE
f32 → ZeckF64 full (8 bytes)   ?             MEASURE
f32 → palette index            ?             MEASURE
f32 → NeuronPrint 6D           ?             MEASURE
f32 → Slot D only              ?             MEASURE
```

No checklist needed. Run the "?" rows. Report the number. Done.

---

## Ref

Xu et al. 2026 "VibeTensor" arXiv:2601.16238 §7 — Frankenstein
composition effect. Their failure modes (mutexes, buffers, concurrency)
are Python/C++ problems Rust prevents. Ours are mathematical:
precision cliffs (killed), basin mismatches (bgz-hhtl-d), and
unmeasured boundaries (just measure them).
