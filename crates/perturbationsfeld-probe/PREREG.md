# D-PFP-0 — pre-registration for the D-PFP-1 Perturbationsfeld probe

Committed BEFORE any run output (F12). Nothing below changes after the first
run. The ratified spec is `.claude/plans/perturbationsfeld-probe-v1.md` §9
(5+3 council, 2026-09-29). A test (`tests/prereg_constants.rs`) asserts every
code constant appears here verbatim.

## Constants

```
THETA = 0.01
DELTA_IDS = 2
TOP = 8
Q = 32
M = 8
MAX_CYCLES = 10
STIMULUS_SEED = 0x9E3779B97F4A7C15
PERMUTATION_SEED = 0x5EED000000000001
DEGENERATE_CEILING_PCT = 25
SANITY_MIN_ELIGIBLE = 8
EMPTY_WINDOW = 64
```

THETA reuses `SCAN_WORTHY_ENERGY` (`cognitive-shader-driver/src/engine_bridge.rs:107`).
THETA, DELTA_IDS, DEGENERATE_CEILING_PCT and SANITY_MIN_ELIGIBLE are
hand-set; single seed; descriptive only — no significance claim is made, and
no result will be described as "significant" (I-NOISE-FLOOR-JIRAK).

## Inputs

- PRIMARY lens: Jina v5, `crates/thinking-engine/data/jina-v5-codebook/distance_table_256x256.u8`
  + `codebook_index.u16` (151,936 LE u16, max 255).
- REPLICATION lens: BGE-M3, `crates/thinking-engine/data/bge-m3-hdr/distance_table_256x256.u8`
  + `codebook_index.u16` (250,002 LE u16, max 255).
- Engine: `thinking_engine::engine::ThinkingEngine::new` (u8 table, p75
  floor), `perturb`, `think(MAX_CYCLES)`, `reset` — RESET between cycles.
- N = 256 (table size; ruled in §9.1).
- Prior art cited, not duplicated: `crates/thinking-engine/examples/chunker_falsifier.rs`.

## Arms

- **C**: window over `{id ∈ P_n.top_k : e > THETA}` (empty ⇒ `[0, EMPTY_WINDOW)`)
  via mask-risc `Pred::Range` + `Keep`.
- **C'**: the same window as a scalar id list (TRIPWIRE).
- **E**: key lane `k(e) = b<0 ? b ^ 0x7FFF_FFFF : b` (`b = e.to_bits() as i32`)
  via `Pred::GtI32{t: k(THETA)}` + `Keep`. The key lane is probe-only, not a
  lane pattern.
- **E_m** (reported only): top-|ids_C| rows by energy among `e > 0`, ties by
  lower id.
- **S**: `π(ids_E)`, π = Fisher–Yates permutation of 0..N from PERMUTATION_SEED.

## Metrics

`act(P)` = `top_k` ids with `e > 0`. `I(a,b) = |act(a) ∩ act(b)|`.
`R_X = Σ_s I(P_n, P_{n+1}^X)`. `ΔR = Σ_s over comparison-valid stimuli of
(I(P_n,P_{n+1}^E) − I(P_n,P_{n+1}^C))`. `D(E,S) = Σ (TOP − I(P_{n+1}^E, P_{n+1}^S))`.
N0 = mean cross-stimulus `I(P_n^s,P_n^t)/TOP`. All decisions in integers.

## Outcomes (evaluated in this order; exactly one per lens)

1. **INVALID**: NaN; lowering oracle mismatch (mask-risc vs scalar vs
   `reference_scratch`); θ inertness fails for E (must shrink at 2θ AND grow
   at θ/2 on ≥ 1 stimulus); tripwire fails (determinism; C vs C');
   > DEGENERATE_CEILING_PCT % of stimuli excluded.
2. **INPUT-INSENSITIVE**: positive-control overlap `> TOP − DELTA_IDS`, OR
   `(1 − N0)·TOP < DELTA_IDS`.
3. **RELABEL-INSENSITIVE**: over ≥ SANITY_MIN_ELIGIBLE stimuli with
   `|ids_E| ≤ N/2`, `D(E,S) < DELTA_IDS · eligible`.
4. On ΔR with margin `DELTA_IDS · Q_valid`: **HIGHER-RETENTION** /
   **LOWER-RETENTION** / **NO-RETENTION-DIFFERENCE**.

Verdict: PRIMARY decides; a differing non-INVALID REPLICATION adds
", LENS-SPECIFIC"; an INVALID REPLICATION adds ", REPLICATION INVALID".

Harness-bug clause: a run INVALID because of a probe defect (not data) is
fixed with constants unchanged, and BOTH runs are quoted in the result entry.

## Scope (pre-registered wording)

- INPUT-INSENSITIVE is scoped to these two 256² tables, p75 floor, 10
  cycles, RESET; it casts doubt, not a verdict, on thinking-engine "unwired
  gems" that assume input-dependent energy.
- Retention is self-consistency, not fidelity or information gain; E ≥ C is
  the expected direction; no outcome is attributed to "address" alone.
- No outcome decides D-WFL-W5, CONTINUE, or the 2-D address identity
  (ISS-PERTURBATION-P64-ADDRESS-IDENTITY-UNPROVEN, D-PFP-2).
