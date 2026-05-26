# rung-ladder-grounding-v1 — ground agichat's RungShift ladder + CollapseGate as LE-contract

**Status:** PROPOSAL (the most-obvious first grounding per `E-AGICHAT-DIMENSION-CONTRACT`)
**Date:** 2026-05-26
**Confidence:** HIGH — the ladder is deterministic integer logic, no VSA in the loop; cleanest possible first restore.
**Predecessors:** `E-AGICHAT-DIMENSION-CONTRACT` (afabefd), `E-I4-META-1`, `E-BATON-1`; shipped floor: ndarray `SoaColumns` (42cb7123) + i4-32 unpack (8de1dcf8).
**Design refs (allowlist-external, read-only):** agichat `src/thinking/{rung-shift,collapse-gate}.ts`; ladybug-rs `src/cognitive/rung.rs`.

---

## 1. Why this one first

The rung ladder is the single most groundable piece of the gestell: it is **already deterministic integer/threshold logic** (RungLevel 0-9, counters, fixed thresholds) with **zero VSA resonance in the decision path**. agichat had it grounded; ladybug-rs kept it (rung.rs is a faithful port). There is nothing to "de-inflate" — only to express as a bit-exact Pod on the SoA floor and wire the triggers to grounded signals. It also has a clean hook: the **9 Rung dims (R1-R9)** are a dimension-group of the 33-TSV, so this grounds that group + the escalation logic that walks it.

## 2. The mined ladder spec (the contract to preserve, verbatim semantics)

**CollapseGate** (`collapse-gate.ts`): SD = std-dev of candidate resonance scores ∈ [0,1]; `SD_MAX = 0.5`; `FLOW < 0.30·SD_MAX (=0.15) ≤ HOLD ≤ 0.70·SD_MAX (=0.35) < BLOCK`. **SD is dispersion (compute allocation), NOT confidence** (Canonical Invariant #2).

**RungShift** (`rung-shift.ts`): `RungLevel 0-9`, `RungBand {0-2, 3-5, 6-9}`. Thresholds `{sustainedBlockTurns:3, pMetricThreshold:0.3, pMetricWindow:5, cooldown:10s}`. Per cycle, `update(gate, p_metric, has_legal_parse)`: BLOCK → `consecutive_blocks++` else reset; push `p_metric` into a 5-window; `structural_mismatch = !has_legal_parse`. Then `evaluate_shift`:
1. cooldown active → no shift
2. at MAX_RUNG (9) → no shift
3. **sustained_block** (`consecutive_blocks ≥ 3`) → +1
4. **predictive_failure** (`avg(window) < 0.3`, window full) → +1
5. **structural_mismatch** → +1
6. else no shift

(`RungShift` is **separate from SD** — Canonical Invariant #3.)

## 3. The grounded contract (LE-contract form)

Bit-exact, Pod, fixed-size, no `Vec`, no wall-clock, no float carrier:

```rust
// lance-graph-contract (zero-dep)
#[repr(u8)] enum GateState { Flow=0, Hold=1, Block=2 }          // 2 bits
#[repr(u8)] enum RungBand  { Low=0, Mid=1, High=2 }             // 2 bits
#[repr(u8)] enum RungTrigger { None, SustainedBlock, PredictiveFailure, StructuralMismatch, Manual }

// RungLevel = u8 (0..=9), 4-bit field.

#[repr(C)] #[derive(Clone, Copy)]                                // Pod, lives in a SoA column
struct RungState {
    rung: u8,               // 0..=9  (4-bit used)
    consecutive_blocks: u8,
    flags: u8,              // bit0 = structural_mismatch
    p_head: u8,             // ring index 0..5
    p_window: [u8; 5],      // P-metric quantized to u8 (0..255 = 0.0..1.0); fixed ring, NO Vec
    last_shift_tick: u32,   // tick-based cooldown (NOT wall-clock ms)
    _pad: [u8; 2],
}                            // 16 bytes = 2 atoms, or folds into a MailboxSoA meta column

#[derive(Clone, Copy)]
struct RungShiftDecision { should_shift: bool, current: u8, target: u8, trigger: RungTrigger }

const SUSTAINED_BLOCK_TURNS: u8 = 3;
const P_METRIC_THRESHOLD_U8: u8 = 77;   // 0.30 × 255
const P_METRIC_WINDOW: usize = 5;
const SHIFT_COOLDOWN_TICKS: u32 = /* tuned; replaces 10s wall-clock */ ;
const SD_FLOW_U8: u8 = 38;   // 0.15 × 255   (SD over u8-quantized candidate scores)
const SD_BLOCK_U8: u8 = 89;  // 0.35 × 255
```

**Translations from the agichat carrier → grounded:**
- `recentPMetrics: number[]` (heap Vec) → `p_window: [u8;5]` fixed ring (Pod, no alloc).
- `shiftCooldownMs` wall-clock → `SHIFT_COOLDOWN_TICKS` (the SoA cycle is tick-driven).
- P-metric / candidate scores: f32 [0,1] → **u8-quantized** (or i4 via the shipped i4-32 unpack). SD computed over the quantized lane (ndarray SIMD), so the gate runs on bit-exact distances, never 10K-D resonance.
- `shiftHistory: Vec<Event>` → out of the hot Pod; if needed, an append-only ring in a cold column (not in `RungState`).

## 4. Layer placement (one canonical home per piece)

| piece | home | note |
|---|---|---|
| `GateState`, `RungBand`, `RungLevel`, `RungState` (Pod), `RungShiftDecision`, `RungTrigger`, threshold consts | **lance-graph-contract** | zero-dep contract types |
| `calculate_sd`, `gate_state_from_sd`, `evaluate_rung_shift` (PURE fns), `apply_rung_shift` (builder) | **lance-graph-planner/src/elevation/** | reconcile with existing `homeostasis.rs` / `operator.rs` |
| `RungState` carried cycle-to-cycle | **ndarray `SoaColumns`** | one column; O(1) `Arc`-clone carry-over (shipped) |
| SD over candidate scores | **ndarray SIMD** (i4/u8 lanes) | dispersion on bit-exact distances |

## 5. Closed loop (grounded)

```
candidate scores (u8/i4, e.g. SignificanceLevel distances)
   → calculate_sd → gate_state_from_sd  → GateState (2-bit)
   → update RungState (tick): BLOCK→consec++, p_window ring-push, mismatch flag
   → evaluate_rung_shift (PURE): cooldown? max? sustained? pred-fail? mismatch? → +1
   → rung(4b) + band(2b) → bucket addressing  (= the 9 Rung dims R1-R9 of the 33-TSV)
```

No `&mut self` during compute (data-flow iron rule): `evaluate_rung_shift` takes `RungState` by value, returns `RungShiftDecision` by value; `apply_rung_shift` is the gated write-back (a builder step).

## 6. Reconciliation (don't fork)

- **`lance-graph-planner/src/elevation/`** already has `homeostasis.rs` + `learning.rs` + `budget.rs` + `operator.rs` (the "cost model that smells resistance"). The rung ladder is the **meaning-depth escalation** elevation should drive — fold `evaluate_rung_shift` in beside `homeostasis` (which is MUL-L6); do not add a parallel module.
- **`lance-graph-contract/collapse_gate.rs`** already exists (the Baton `CollapseGateEmission`). Add `GateState`/SD there — the gate already lives in the contract; extend, don't duplicate.
- **The 33-TSV (`E-AGICHAT-DIMENSION-CONTRACT`)**: RungLevel is derived from / indexes the **R1-R9 dim-group** of `ThinkingStyleI4_32D`. The ladder *writes* the rung profile; this grounds that group. Sequence after the TSV type lands, or co-land the rung group first as the pilot.

## 7. Deliverables

| D-id | title | crate | ~LOC | risk |
|---|---|---|---|---|
| D-RUNG-1 | contract types (RungLevel/Band/GateState/RungState Pod/Decision/Trigger + consts) | lance-graph-contract | 150 | LOW |
| D-RUNG-2 | pure ladder logic (`calculate_sd`, `gate_state_from_sd`, `evaluate_rung_shift`, `apply_rung_shift`) folded into `elevation/` | lance-graph-planner | 200 | LOW |
| D-RUNG-3 | `RungState` as a `SoaColumns` column + tick-driven `update` | ndarray bind + planner | 100 | LOW |
| D-RUNG-4 | wire SD→GateState into `collapse_gate.rs`; map rung→R1-R9 TSV group | contract + planner | 120 | MED |

Tests: verbatim-semantics parity vs the agichat spec (sustained-block at exactly 3, pred-fail window-full + avg<0.3, mismatch, cooldown, max-rung cap, band boundaries 2/3 and 5/6), all on integer/u8 inputs. Gate green via central `cargo fmt`/`clippy -D warnings`/`test`.

## 8. Invariants honored

- **No `Vec`/alloc in the hot Pod** (fixed `[u8;5]` ring). - **No `&mut` during compute** (pure `evaluate`, builder `apply`). - **Tick-based, not wall-clock**. - **Integer rung, no float-resonance carrier** (the de-grounding ladybug-rs did). - **SD = dispersion, not confidence** (Invariant #2). - **RungShift separate from SD** (Invariant #3). - Bit-packed (rung 4b + band 2b + gate 2b in a meta byte).

## 9. Cross-refs

`E-AGICHAT-DIMENSION-CONTRACT` (grounding doctrine + the 33-TSV / 9-Rung group); shipped floor `SoaColumns` (42cb7123) + i4-32 (8de1dcf8); `lance-graph-planner/src/elevation/{homeostasis,operator}.rs`; `lance-graph-contract/src/collapse_gate.rs`; design refs agichat `src/thinking/{rung-shift,collapse-gate}.ts`, ladybug-rs `src/cognitive/rung.rs`. Iron rules: data-flow (no `&mut` compute), `I-NOISE-FLOOR-JIRAK` (SD on bit-exact distances, not bundles).
