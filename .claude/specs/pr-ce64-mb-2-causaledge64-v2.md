# PR-CE64-MB-2 — CausalEdge64 v2 Layout Extension Spec

> **Status:** Draft — sprint-log-10 W2 deliverable; **PATCHED 2026-05-16 per cognitive-substrate-convergence-v1.md** (plan §6 locks final v2 layout; OQ-LAYOUT-1 resolved)
> **PR scope:** `crates/causal-edge/` in-place layout extension; ~400 LOC
> **Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §3 (primary) + §2 (truth-band lens collapse)
> **Authoritative layout anchor:** `.claude/plans/cognitive-substrate-convergence-v1.md` §6 (SUPERSEDES parent plan §3 for bit layout)
> **Worker:** W2 (causaledge64-v2), Sonnet, sprint-log-10
> **Date:** 2026-05-14
> **Patched:** 2026-05-16 (sprint-10 specs patch for cognitive-substrate-convergence-v1; OQ-LAYOUT-1 resolved; §"Signed Mantissa Rationale" and §"Counterfactual via causal_mask" added)
> **Depends on:** PR-CE64-MB-1 (par-tile crate apex) — must land first so TrustTexture reference type is available
> **Blocks:** PR-CE64-MB-5 (MailboxSoA + AttentionMaskActor wiring, which reads W/truth from CausalEdge64)

---

## §0 Critical Finding — Actual vs. Plan §3 "Current" Layout Discrepancy

> **RESOLUTION (2026-05-16):** OQ-LAYOUT-1 is **RESOLVED**. The cognitive-substrate-convergence-v1.md plan (§6) locks the final v2 bit layout. The recommended path is the **i4-signed-mantissa + drop-temporal + drop-G-slot** option — NOT Option C as originally recommended here. See §2 (updated) for the ratified layout and §"Signed Mantissa Rationale" for the encoding rationale. Implementation may proceed on the locked layout.

**Before any implementation begins, this discrepancy must be resolved with the plan author.**

Parent plan §3 describes the "current CausalEdge64 layout" as:

```
bits  field           count
0-2   Pearl rung       3 bits
3-10  S palette        8 bits
11-18 P palette        8 bits
19-26 O palette        8 bits
27-42 temporal        16 bits
43-50 style ord        8 bits
51-63 reserved        13 bits   <- reclaim target
```

The **actual shipped code** in `crates/causal-edge/src/edge.rs` has a completely different layout
with NO style ordinal and NO reserved bits:

```
bits  field           count   shift constant
0-7   S palette        8 b    S_SHIFT = 0
8-15  P palette        8 b    P_SHIFT = 8
16-23 O palette        8 b    O_SHIFT = 16
24-31 NARS frequency   8 b    FREQ_SHIFT = 24
32-39 NARS confidence  8 b    CONF_SHIFT = 32
40-42 Causal mask      3 b    CAUSAL_SHIFT = 40
43-45 Direction triad  3 b    DIR_SHIFT = 43
46-48 Inference type   3 b    INFER_SHIFT = 46
49-51 Plasticity flags 3 b    PLAST_SHIFT = 49
52-63 Temporal index  12 b    TEMPORAL_SHIFT = 52
```

**All 64 bits are in use. There are zero reserved bits today.**

This is the single highest-risk finding in this spec. The parent plan §3 table describes what appears
to be a design-phase layout that was never implemented as written. The actual shipped type has:
- NARS frequency AND confidence as separate 8-bit fields (plan §3 omits these entirely)
- Direction triad (3 bits) — not in plan §3
- Inference type (3 bits) — not in plan §3
- Plasticity flags (3 bits) — not in plan §3
- 12-bit temporal (not 16-bit as in plan §3)
- NO style ordinal field
- NO reserved bits

**Resolution options (requires plan-author sign-off before implementation):**

| Option | Strategy | Net bit delta | Risk | Status |
|---|---|---|---|---|
| A | Compress temporal 12 to 4 bits; move absolute cycle to AriGraph SPO-G quad | Frees 8 bits; still need 5 more | HIGH | Rejected |
| B | Replace direction(3) + inference(3) with G(5)+W(6) | Frees 6 bits; still need 7 more | HIGH | Rejected |
| C | Drop temporal(12) to AriGraph; repurpose plasticity(3) for truth(2)+spare(1); G(5)+W(6) in freed 15 bits | Frees 15 bits, uses 13+2 spare | MEDIUM | Rejected (G-slot dropped per L-3) |
| D | Extend to CausalEdge128 (two u64) | No reclaim needed | HIGH for EdgeColumn | Rejected |
| E | Complex split across dir+infer+plast | Complex mapping | HIGH | Rejected |
| **F (RATIFIED)** | **Drop temporal(12); expand inference 3→4b SIGNED (i4); add W(6)+truth-band-lens(2)+spare(3); DROP G-slot entirely** | **Frees 12b; spends 10b; 3b spare** | **LOW** | **LOCKED by cognitive-substrate-convergence-v1.md §6** |

**Option F is ratified** per cognitive-substrate-convergence-v1.md §6 (locked decisions L-2, L-3, L-4, L-6, L-7). Key changes from original Option C: (1) G-slot DROPPED (L-3: redundant via palette family-prefix + SoA partition + witness corpus root); (2) Inference mantissa EXPANDED to 4-bit SIGNED i4 (L-4: direction × NARS rule composition); (3) Truth-band lens 2b and W-slot 6b retained; (4) 3-bit spare for sprint-12+ headroom.

**OQ-LAYOUT-1: RESOLVED.** See cognitive-substrate-convergence-v1.md §6 for the authoritative layout table. Implementation proceeds on Option F (§2 below reflects the locked layout).

---

## §1 Statement of In-Place Reclaim (No Type Bump)

Per parent plan §3 doctrine: CausalEdge64 stays exactly 8 bytes (one u64, one register). No type
rename, no version suffix. The struct name CausalEdge64 is canonical; "v2" refers to the layout
extension, not the type.

**Binary contract:** A CausalEdge64 written by a v1 binary reads correctly by a v2 binary when the
new fields (W, truth-band-lens, mantissa) were zero in the v1-written value. For CausalEdge64::ZERO
edges: w_slot()=0, truth_lens()=0 (Crystalline), mantissa=0 — the correct "no witness, fully
trusted, neutral inference" defaults. For non-zero v1 edges, the reclaimed bits will read garbage
unless a version gate is applied (see §6.1).

**Net bit allocation (Option F — RATIFIED per cognitive-substrate-convergence-v1.md §6):**

```
Total bits: 64
v1 fields kept:        S(8)+P(8)+O(8)+freq(8)+conf(8)+causal(3)+dir(3)+plasticity(3) = 49 bits
v1 fields dropped:     temporal(12) = 12 bits freed
v1 fields EXPANDED:    inference 3b unsigned → 4b SIGNED (net +1 bit consumed)
v2 new fields:         W(6)+truth-band-lens(2)+spare(3) = 11 bits consumed
Reclaim arithmetic:    12 freed − 1 (mantissa expand) − 6 (W) − 2 (lens) = 3 spare
Net change:            0 (still 64 bits, zero unused)
```

**Architectural rationale for each changed field:**
- InferenceType 3b → 4b signed i4 (bits 46-49): EXPANDED, not dropped. Direction × NARS rule
  composition: sign bit = forward (+) vs backward (−) chain; magnitude = base rule index (0..7).
  Per cognitive-substrate-convergence-v1.md L-4. See §"Signed Mantissa Rationale".
- PlasticityState (3 bits, bits 50-52): RETAINED. Per L-8: plasticity is load-bearing dispatch
  payload; relocation costs extra Zone-1 lookup per cycle. (Bit positions shift by 1 due to mantissa
  expansion.)
- temporal (12 bits, bits 52-63 in v1): DROPPED. Covered by chain-position in SpoWitnessChain +
  AriGraph Triplet.timestamp. "Temporal causality is structural" doctrine (L-2).
- G-slot (5 bits, was proposed in Option C): NOT ADDED. Per L-3: three-way redundant via SoA
  partition (tenant), witness corpus root (belief), palette family-prefix (ontology).
- W-slot (6 bits, bits 53-58): NEW. Discourse corpus root handle; 64 active corpora. Per L-6.
- Truth-band lens (2 bits, bits 59-60): NEW. 4 lens states incl. "13% ambiguous direction". Per L-7.
- Spare (3 bits, bits 61-63): NEW. Reserved for sprint-12+ probe headroom (Rubicon-commit marker,
  Markov-decay-rate quantum, or I-NOISE-FLOOR-JIRAK threshold storage).

---

## §2 Bit-Layout Diagram — v2 Final (Option F — LOCKED per cognitive-substrate-convergence-v1.md §6)

> **OQ-LAYOUT-1: RESOLVED.** The layout below is the authoritative v2 bit assignment. Source of truth: cognitive-substrate-convergence-v1.md §6. No further adjudication needed.

```
v2 CausalEdge64 bit layout (LSB = bit 0, MSB = bit 63):

[ 0:  7]  S palette index           u8       (256 subject archetypes)
[ 8: 15]  P palette index           u8       (256 predicate archetypes)
[16: 23]  O palette index           u8       (256 object archetypes)
[24: 31]  NARS frequency            u8       (f = val/255)
[32: 39]  NARS confidence           u8       (c = val/255)
[40: 42]  Causal mask               3b       (Pearl 2³ — IS the rung axis; 0b111=SPO=Counterfactual)
[43: 45]  Direction triad           3b       (sign(palette[idx].dim0) per S/P/O)
[46: 49]  Inference mantissa        4b s     (−8..+7 signed — direction × NARS rule; see §"Signed Mantissa Rationale")
[50: 52]  Plasticity flags          3b       (hot/cold per S/P/O plane)
[53: 58]  W slot                    6b       ← NEW: corpus root handle (64 active corpora)
[59: 60]  Truth-band lens           2b       ← NEW: 4 lens states (Crystalline/Solid/Fuzzy/Murky)
[61: 63]  Spare                     3b       ← reserved headroom for sprint-12+
                                    ───
                                    64b      zero unused
```

**Exact field positions (v2 FINAL — Option F):**

| bits  | field              | count | shift | change from v1         |
|-------|--------------------|-------|-------|------------------------|
| 0-7   | S palette idx      | 8 b   | 0     | unchanged              |
| 8-15  | P palette idx      | 8 b   | 8     | unchanged              |
| 16-23 | O palette idx      | 8 b   | 16    | unchanged              |
| 24-31 | NARS frequency     | 8 b   | 24    | unchanged              |
| 32-39 | NARS confidence    | 8 b   | 32    | unchanged              |
| 40-42 | Causal mask        | 3 b   | 40    | unchanged              |
| 43-45 | Direction triad    | 3 b   | 43    | unchanged              |
| 46-49 | Inference mantissa | 4 b s | 46    | EXPANDED: 3b unsigned → 4b signed i4 (−8..+7) |
| 50-52 | Plasticity flags   | 3 b   | 50    | SHIFTED by 1 (was 49-51 in v1) |
| 53-58 | W slot             | 6 b   | 53    | NEW: corpus root handle |
| 59-60 | Truth-band lens    | 2 b   | 59    | NEW: TrustTexture 4-state |
| 61-63 | Spare              | 3 b   | 61    | NEW: reserved           |

**Sum check:** 8+8+8+8+8+3+3+4+3+6+2+3 = 64 bits. Verified.

**Reclaim arithmetic (L-2, L-3, L-4, L-6, L-7):** Drop temporal(12 bits freed) → Inference mantissa expand +1, Plasticity shift 0, W-slot +6, truth-band-lens +2 = 9 bits spent + 3 spare. G-slot NOT added (L-3: redundant via SoA partition + palette family-prefix + witness corpus root).


---

## §"Signed Mantissa Rationale"

> **Added 2026-05-16 per cognitive-substrate-convergence-v1.md §6 + locked decisions L-4 and L-9.**

### Why sign = direction

The inference mantissa (bits 46-49, 4-bit signed i4, range −8..+7) encodes both the *direction* of
inference and the *identity of the base NARS rule* in a single field:

- **`signum(mantissa)` = chain direction:**
  - `+` (values 0..+7): forward-chain / compose / commit direction — Deduction, Synthesis,
    Revision-positive, Induction (forward generalization), Exemplification.
  - `−` (values −8..−1): backward-chain / decompose / refute direction — Abduction, Contraposition,
    Revision-negative, Analogy-negative, Counterfactual refutation.
  - Zero (value 0): neutral / identity / no-op inference; used for bare SPO assertions with no
    active NARS rule applied.

- **`abs(mantissa)` = base NARS rule index (0..7):**

  | |mantissa| | Rule name (+ direction) | Rule name (− direction) |
  |---|---|---|
  | 0 | Identity / neutral | Identity / neutral |
  | 1 | Deduction | Abduction |
  | 2 | Induction | Contraposition |
  | 3 | Exemplification | Analogy-negative |
  | 4 | Revision-positive | Revision-negative |
  | 5 | Synthesis | Decomposition |
  | 6 | Reserved5 (absorbs PR-LL-1 Intervention) | Reserved6 (absorbs PR-LL-1 Counterfactual) |
  | 7 | Extension (future) | Intension-negative (future) |

  Slots 6/7 absorb PR-LL-1's `Intervention` and `Counterfactual` variants from
  `nars_dispatch.rs` per locked decision L-9: those variants land in `Reserved5`/`Reserved6` of the
  canonical `causal_edge::InferenceType` enum when v2 ships. They do NOT need a separate bit; the
  signed mantissa already encodes them at the correct magnitude slot.

### Three SIMD wins from signed i4 mantissa

**Win 1 — signum/abs are free in SIMD.** Computing chain direction from a signed i4 lane is a
single arithmetic-right-shift: `direction = lane >> 3` (extracts sign bit). Computing magnitude is a
single `abs` intrinsic (`vpabsb` on x86/AVX2, `abs` on ARM NEON). No branch, no table lookup, no
enum discriminant comparison. Entire EdgeColumn can be swept for direction distribution in one AVX2
pass over `[i8; N]` (pairs of 4-bit lanes packed to i8 for SIMD efficiency).

**Win 2 — palette × mantissa propagation stays in register family.** The three SPO palette indices
are u8 (bits 0-23); the mantissa is i4 (bits 46-49). Products of palette-distance lookups
(`u8 × i4 → i8` via sign-extend + multiply) stay within the i4/i8/i16 family — no f32 conversion
needed. Qualia dimension products (`QualiaI4_16D × InferenceMantissa`) are similarly `i4 × i4 → i8`,
one SIMD multiply per 16-dim row. This enables the MUL evaluation (DK / TrustTexture / FlowState /
GateDecision) to run entirely in integer SIMD per locked decision L-18.

**Win 3 — 8-channel thinking-engine transcoder is a near-bitcast at L3 commit.** The
`thinking_engine::CausalEdge64::net_strength()` already returns a signed value whose `signum()`
maps directly to the inference mantissa sign bit, and whose magnitude maps directly to the base rule
index (per the 8-channel → SPO-palette reunion, Option R-3 in synergies doc). The L3 commit
transcoder (`transcode_to_spo()`) is therefore:

```
mantissa_sign  = net_strength.signum()           // forward vs backward
mantissa_mag   = channel_to_rule_index(channel)  // which of 8 base rules
mantissa_i4    = sign_extend(mantissa_sign * mantissa_mag, 4)
```

This is 3 arithmetic operations per edge, no table lookup, no float. The "near-bitcast" claim from
cognitive-substrate-convergence-v1.md §3.1 is literal: both sides share the same signed integer
algebra; the transcoder is an algebra homomorphism, not a format conversion.

### PR-LL-1 absorption at Reserved5/6

PR #375 (PR-LL-1) added `Intervention` and `Counterfactual` variants to `nars_dispatch.rs` only.
Per L-9, these absorb into `Reserved5`/`Reserved6` of the canonical `causal_edge::InferenceType`
when v2 ships. The signed mantissa already has structural slots for them:

- `+6` = Intervention (forward — apply a do-calculus intervention, fix a variable)
- `−6` = Counterfactual (backward — evaluate the Pearl-3 antecedent counterfactually)

This makes PR-LL-1's dispatch path a zero-cost addition: the variants already have homes at mantissa
magnitude 6, positive and negative. No new bits needed, no new enum discriminant collision.

---

## §"Counterfactual via causal_mask, NOT via separate bit"

> **Added 2026-05-16 per cognitive-substrate-convergence-v1.md locked decision L-5.**

### causal_mask 0b111 SPO IS Pearl-3 already

The 3-bit `causal_mask` field (bits 40-42) encodes the Pearl causal hierarchy as a bitmask over the
SPO triple:

| causal_mask | Binary | Pearl rung | Semantics |
|---|---|---|---|
| 0b000 | S=0 P=0 O=0 | Rung 0 — Association | Observational correlation only; no causal claim |
| 0b001 | S=0 P=0 O=1 | Rung 1 — Intervention (partial) | Object intervened upon |
| 0b010 | S=0 P=1 O=0 | Rung 1 — Intervention (partial) | Predicate intervened upon |
| 0b011 | S=0 P=1 O=1 | Rung 2 — Intervention (full PO) | do(P,O) applied |
| 0b100 | S=1 P=0 O=0 | Rung 1 — Intervention (partial) | Subject intervened upon |
| 0b101 | S=1 P=0 O=1 | Rung 2 — Intervention (full SO) | do(S,O) applied |
| 0b110 | S=1 P=1 O=0 | Rung 2 — Intervention (full SP) | do(S,P) applied |
| 0b111 | S=1 P=1 O=1 | **Rung 3 — Counterfactual** | Full SPO mask = "what would have happened if S had done P to O" |

`causal_mask = 0b111` (all three SPO components masked) IS the Pearl-3 counterfactual operator by
construction. The mask means: "hold S, P, and O jointly fixed in a counterfactual world." No
separate Pearl-3 modifier bit is needed or correct — adding one would duplicate information already
present in the mask.

### Why no separate Pearl-3 modifier bit

A dedicated Pearl-3 modifier bit would be:
1. **Redundant.** `causal_mask == 0b111` is already the necessary and sufficient condition for
   Pearl-3 semantics. Any consumer checking a hypothetical `pearl3_flag` bit should check
   `causal_mask()` instead.
2. **Inconsistent.** If `pearl3_flag=1` but `causal_mask != 0b111`, the edge carries contradictory
   causal claims — an illegal state requiring extra validation logic in every consumer.
3. **Wasteful.** The 3 spare bits (bits 61-63) are reserved for principled sprint-12+ uses
   (Rubicon-commit marker, Markov-decay quantum, Jirak threshold). Spending one on a redundant flag
   is the wrong trade.

### 4-bit mantissa carries WHICH base rule at THAT rung

The `causal_mask` axis (Pearl rung 0-3) and the `inference_mantissa` axis (NARS base rule index
0-7) are orthogonal:

- `causal_mask` answers: *at which level of Pearl's causal hierarchy does this edge operate?*
- `mantissa` answers: *which NARS inference rule produced this edge, and in which direction?*

A single edge can be simultaneously:
- `causal_mask = 0b111` (Pearl-3 counterfactual reasoning)
- `mantissa = −6` (PR-LL-1 Counterfactual NARS rule, backward direction)

This is the full composition: the mask locates the edge in Pearl's hierarchy; the mantissa records
the NARS algebra that generated the strength value. No additional bit is needed to express
"counterfactual Pearl-3 via NARS Counterfactual rule" — the existing two fields together encode it
exactly and without ambiguity.

## §3 Bit-Shift Constants Module (crates/causal-edge/src/layout.rs)

New file. All shift constants, masks, InferenceMantissa type, TrustTexture enum, and the
const_assert guard live here. Updated to reflect the FINAL v2 layout (Option F, locked by
cognitive-substrate-convergence-v1.md §6). G-slot is NOT present (L-3).

```rust
//! CausalEdge64 v2 layout constants — FINAL (Option F, locked 2026-05-16).
//!
//! Cite: cognitive-substrate-convergence-v1.md §6 (authoritative bit layout)
//!       + pr-ce64-mb-2-causaledge64-v2.md §2 (implementation contract).
//! OQ-LAYOUT-1: RESOLVED. G-slot dropped (L-3). Mantissa = 4b signed i4 (L-4).

// v1 fields preserved (shifts unchanged from v1)
pub const S_SHIFT:      u32 = 0;
pub const P_SHIFT:      u32 = 8;
pub const O_SHIFT:      u32 = 16;
pub const FREQ_SHIFT:   u32 = 24;
pub const CONF_SHIFT:   u32 = 32;
pub const CAUSAL_SHIFT: u32 = 40;
pub const DIR_SHIFT:    u32 = 43;

// v1→v2 EXPANDED field (3b unsigned → 4b signed i4; shift unchanged at 46)
/// Inference mantissa: 4-bit signed (−8..+7). sign = direction; |val| = NARS base rule index.
/// Encodes direction × NARS rule in one field. See §"Signed Mantissa Rationale".
/// sign = forward/backward chain direction; magnitude = base rule (Deduction=1..Reserved=7).
pub const INFER_SHIFT:  u32 = 46;
pub const BITS4_MASK:   u64 = 0xF;   // 4-bit unsigned mask for pack/unpack of signed i4
pub const INFER_MASK:   u64 = BITS4_MASK << INFER_SHIFT;

// v1 fields SHIFTED (plasticity: was bits 49-51 in v1, now bits 50-52 due to mantissa +1)
pub const PLAST_SHIFT:  u32 = 50;

// v1 field DEPRECATED
#[deprecated(since = "0.2.0", note = "bits 52-63 reclaimed for W/truth/spare + mantissa expansion; time is structural (chain-position + AriGraph Triplet.timestamp)")]
pub const V1_TEMPORAL_SHIFT: u32 = 52;

// v2 NEW fields (reclaimed from dropped temporal 12 bits)
/// 6-bit witness corpus root handle (0..=63). 0 = no corpus anchor. Per L-6.
pub const W_SHIFT:      u32 = 53;
/// 2-bit truth-band lens (TrustTexture ordinal). 0 = Crystalline. Per L-7.
pub const TRUTH_SHIFT:  u32 = 59;
/// 3-bit spare — reserved for sprint-12+ (Rubicon-commit marker, Markov-decay quantum,
/// or I-NOISE-FLOOR-JIRAK threshold). Per §6 reclaim arithmetic.
pub const SPARE_SHIFT:  u32 = 61;

// Common masks
pub const BYTE_MASK:  u64 = 0xFF;
pub const BITS3_MASK: u64 = 0x7;
pub const BITS6_MASK: u64 = 0x3F;
pub const BITS2_MASK: u64 = 0x3;

pub const PLAST_MASK:  u64 = BITS3_MASK << PLAST_SHIFT;
pub const W_MASK:      u64 = BITS6_MASK << W_SHIFT;
pub const TRUTH_MASK:  u64 = BITS2_MASK << TRUTH_SHIFT;
pub const SPARE_MASK:  u64 = BITS3_MASK << SPARE_SHIFT;

// Const-assert: all 64 bits covered exactly once.
// If this fails at compile time, the layout is inconsistent.
const _LAYOUT_COVERAGE: () = {
    let all: u64 = (BYTE_MASK  << S_SHIFT)
        | (BYTE_MASK  << P_SHIFT)
        | (BYTE_MASK  << O_SHIFT)
        | (BYTE_MASK  << FREQ_SHIFT)
        | (BYTE_MASK  << CONF_SHIFT)
        | (BITS3_MASK << CAUSAL_SHIFT)
        | (BITS3_MASK << DIR_SHIFT)
        | (BITS4_MASK << INFER_SHIFT)  // 4b signed mantissa
        | (BITS3_MASK << PLAST_SHIFT)  // plasticity shifted to 50-52
        | (BITS6_MASK << W_SHIFT)      // W slot 53-58
        | (BITS2_MASK << TRUTH_SHIFT)  // truth-band lens 59-60
        | (BITS3_MASK << SPARE_SHIFT); // spare 61-63
    // 8+8+8+8+8+3+3+4+3+6+2+3 = 64
    assert!(all == u64::MAX, "CausalEdge64 v2 bit layout must cover all 64 bits exactly once");
};

/// Two-bit truth band — 4 levels, 4 consumer-lens projections.
///
/// Lens table (per causaledge64-mailbox-rename-soa-v1.md §2):
///   0b00 = Crystalline | Mastered     | Quiet  | Proceed
///   0b01 = Solid       | Calibrated   | Mild   | Proceed
///   0b10 = Fuzzy       | Uncertain    | Active | Sandbox
///   0b11 = Murky       | Contradiction| Loud   | Compass (veto)
///
/// NOTE: Local definition in causal-edge (zero-dep crate). The canonical
/// contract type is lance_graph_contract::mul::TrustTexture.
/// The 2-bit encoding is byte-compatible by construction.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
#[repr(u8)]
pub enum TrustTexture {
    #[default]
    Crystalline = 0,
    Solid = 1,
    Fuzzy = 2,
    Murky = 3,
}

impl TrustTexture {
    #[inline]
    pub fn from_bits_2(v: u8) -> Self {
        match v & 0b11 {
            0 => Self::Crystalline,
            1 => Self::Solid,
            2 => Self::Fuzzy,
            _ => Self::Murky,
        }
    }

    #[inline]
    pub fn to_bits_2(self) -> u8 { self as u8 }
}
```

---

## §4 Accessor Method Sketches (crates/causal-edge/src/edge.rs additions)

Naming follows the method-on-carrier pattern per CLAUDE.md doctrine.
Read accessors: `&self` Copy methods returning Copy values.
Setters: builder-shape `with_*` returning `Self`. Mutating `set_*` for hot-path `&mut` callers.
All gated by `#[cfg(feature = "causal-edge-v2-layout")]`.

Note: G-slot is NOT present in the v2 layout (dropped per L-3 in cognitive-substrate-convergence-v1.md).
The `with_routing` method signature is `(w: u8, t: TrustTexture)` — no `g` parameter.
Any prior references to `g_slot`, `with_g_slot`, `set_g_slot`, or `G_SHIFT` are stale and removed.

```rust
#[cfg(feature = "causal-edge-v2-layout")]
impl CausalEdge64 {
    // ── Read Accessors ──────────────────────────────────────────────────────

    /// Witness corpus root handle (6-bit, 0..=63). 0 = no corpus anchor.
    /// WARNING: GARBAGE for non-zero v1 edges (bits 53-58 were temporal MSBs in v1).
    /// Use version gate before calling on edges of unknown provenance.
    /// CausalEdge64::ZERO -> 0 (correct default: no corpus).
    #[inline(always)]
    pub fn w_slot(self) -> u8 {
        use crate::layout::{W_SHIFT, BITS6_MASK};
        ((self.0 >> W_SHIFT) & BITS6_MASK) as u8
    }

    /// Inference mantissa: 4-bit signed i4, range −8..+7.
    /// sign = chain direction (+ = forward, − = backward); abs = base rule index.
    /// See §"Signed Mantissa Rationale" for the full encoding table.
    #[inline(always)]
    pub fn inference_mantissa(self) -> i8 {
        use crate::layout::{INFER_SHIFT, BITS4_MASK};
        let raw = ((self.0 >> INFER_SHIFT) & BITS4_MASK) as u8;
        // sign-extend 4-bit unsigned to i8
        if raw & 0x8 != 0 { (raw | 0xF0) as i8 } else { raw as i8 }
    }

    /// Chain direction extracted from mantissa sign: 1 (forward), -1 (backward), 0 (neutral).
    #[inline(always)]
    pub fn inference_direction(self) -> i8 {
        let m = self.inference_mantissa();
        if m > 0 { 1 } else if m < 0 { -1 } else { 0 }
    }

    /// Base rule index (0..7) extracted from mantissa magnitude.
    #[inline(always)]
    pub fn inference_rule_index(self) -> u8 {
        self.inference_mantissa().unsigned_abs() & 0x7
    }

    /// Truth band as TrustTexture (2-bit). Returns Crystalline for ZERO edges.
    /// WARNING: v1 edges with temporal >= 128 may read as Solid/Fuzzy/Murky.
    /// Bits 59-60 were temporal bits 7-8 in v1. Version gate required.
    #[inline(always)]
    pub fn truth(self) -> crate::layout::TrustTexture {
        use crate::layout::{TRUTH_SHIFT, BITS2_MASK, TrustTexture};
        TrustTexture::from_bits_2(((self.0 >> TRUTH_SHIFT) & BITS2_MASK) as u8)
    }

    /// Raw truth band value (0..=3) without TrustTexture import.
    #[inline(always)]
    pub fn truth_raw(self) -> u8 {
        use crate::layout::{TRUTH_SHIFT, BITS2_MASK};
        ((self.0 >> TRUTH_SHIFT) & BITS2_MASK) as u8
    }

    // ── Builder-Shape Setters (functional update, returns Self) ─────────────

    /// Return new edge with W slot set. debug_assert!(w <= 63).
    #[inline]
    pub fn with_w_slot(self, w: u8) -> Self {
        use crate::layout::{W_SHIFT, BITS6_MASK, W_MASK};
        debug_assert!(w <= 63, "w_slot must fit 6 bits (0..=63), got {w}");
        Self((self.0 & !W_MASK) | (((w as u64) & BITS6_MASK) << W_SHIFT))
    }

    /// Return new edge with truth band set.
    #[inline]
    pub fn with_truth(self, t: crate::layout::TrustTexture) -> Self {
        use crate::layout::{TRUTH_SHIFT, BITS2_MASK, TRUTH_MASK};
        Self((self.0 & !TRUTH_MASK) | ((t.to_bits_2() as u64 & BITS2_MASK) << TRUTH_SHIFT))
    }

    /// Return new edge with signed inference mantissa set. Range −8..+7.
    #[inline]
    pub fn with_inference_mantissa(self, m: i8) -> Self {
        use crate::layout::{INFER_SHIFT, BITS4_MASK, INFER_MASK};
        debug_assert!((-8..=7).contains(&m), "mantissa must be −8..+7, got {m}");
        let raw = (m as u8) & 0xF;
        Self((self.0 & !INFER_MASK) | ((raw as u64 & BITS4_MASK) << INFER_SHIFT))
    }

    /// Set W + truth in one mask-and-or (hot-path emit operation).
    /// Used by MailboxSoA::dispatch_cycle() when stamping routing onto emissions.
    /// NOTE: No `g` parameter — G-slot is absent in v2 layout (L-3).
    #[inline]
    pub fn with_routing(self, w: u8, t: crate::layout::TrustTexture) -> Self {
        use crate::layout::{W_SHIFT, TRUTH_SHIFT, BITS6_MASK, BITS2_MASK, W_MASK, TRUTH_MASK};
        debug_assert!(w <= 63, "w ({w}) out of range");
        let routing = ((w as u64 & BITS6_MASK) << W_SHIFT)
            | ((t.to_bits_2() as u64 & BITS2_MASK) << TRUTH_SHIFT);
        Self((self.0 & !(W_MASK | TRUTH_MASK)) | routing)
    }

    // ── Mutating Setters (hot-path, &mut self) ──────────────────────────────

    #[inline]
    pub fn set_w_slot(&mut self, w: u8) {
        use crate::layout::{W_SHIFT, BITS6_MASK, W_MASK};
        debug_assert!(w <= 63);
        self.0 = (self.0 & !W_MASK) | (((w as u64) & BITS6_MASK) << W_SHIFT);
    }

    #[inline]
    pub fn set_truth(&mut self, t: crate::layout::TrustTexture) {
        use crate::layout::{TRUTH_SHIFT, BITS2_MASK, TRUTH_MASK};
        self.0 = (self.0 & !TRUTH_MASK) | ((t.to_bits_2() as u64 & BITS2_MASK) << TRUTH_SHIFT);
    }

    #[inline]
    pub fn set_inference_mantissa(&mut self, m: i8) {
        use crate::layout::{INFER_SHIFT, BITS4_MASK, INFER_MASK};
        debug_assert!((-8..=7).contains(&m));
        let raw = (m as u8) & 0xF;
        self.0 = (self.0 & !INFER_MASK) | ((raw as u64 & BITS4_MASK) << INFER_SHIFT);
    }
}

// v1 stub accessors (feature off) — return safe zero/Crystalline defaults
#[cfg(not(feature = "causal-edge-v2-layout"))]
impl CausalEdge64 {
    #[inline(always)] pub fn w_slot(self) -> u8 { 0 }
    #[inline(always)] pub fn inference_mantissa(self) -> i8 { 0 }
    #[inline(always)] pub fn inference_direction(self) -> i8 { 0 }
    #[inline(always)] pub fn inference_rule_index(self) -> u8 { 0 }
    #[inline(always)] pub fn truth(self) -> TrustTexture { TrustTexture::Crystalline }
    #[inline(always)] pub fn truth_raw(self) -> u8 { 0 }
    #[inline] pub fn with_w_slot(self, _w: u8) -> Self { self }
    #[inline] pub fn with_truth(self, _t: TrustTexture) -> Self { self }
    #[inline] pub fn with_inference_mantissa(self, _m: i8) -> Self { self }
    #[inline] pub fn with_routing(self, _w: u8, _t: TrustTexture) -> Self { self }
}
```

---

## §5 Feature Flag (crates/causal-edge/Cargo.toml)

```toml
[package]
name = "causal-edge"
version = "0.2.0"
edition = "2021"
description = "CausalEdge64: 64-bit causal neuron with SPO palette, NARS truth, Pearl hierarchy, G/W/truth v2"
license = "MIT"
repository = "https://github.com/AdaWorldAPI/lance-graph"

[features]
default = ["causal-edge-v2-layout"]

# v2 layout (Option F, locked 2026-05-16 by cognitive-substrate-convergence-v1.md §6):
#   - Drop temporal(12b); expand inference 3b unsigned -> 4b signed i4; add W(6b)+truth(2b)+spare(3b)
#   - G-slot NOT present (L-3: redundant via palette family-prefix + SoA partition + witness corpus)
#   - mantissa sign = chain direction; abs(mantissa) = NARS base rule index (0..7)
# Off -> v1 accessor stubs return zeros (no corpus, Crystalline, neutral mantissa).
# Default = ON for all new builds. Opt-out for downstream compat:
#   causal-edge = { path = "...", default-features = false }
causal-edge-v2-layout = []

[dependencies]
# No dependencies — this crate is self-contained.
# TrustTexture is defined locally (not imported from lance-graph-contract)
# to preserve the zero-dep invariant.

[dev-dependencies]
```

---

## §6 Compatibility Invariants (Merge Gates)

All four must hold before PR-CE64-MB-2 can land.

### 6.1 PAL8 4101-byte Serialization

The PAL8 form is referenced in plan §3 and session knowledge doc at
`crates/lance-graph-planner/.claude/knowledge/session_autocomplete_cache.md`. However, the
serialization implementation was NOT found in `crates/causal-edge/` during code survey. This is
OQ-PAL8-FORMAT (see §11).

**Byte-level analysis (assuming PAL8 bytes 0-7 = CausalEdge64 raw u64, little-endian):**

The reclaimed bits land at these byte positions (Option F layout):
- Mantissa bits 46-49: byte 5 bits 6-7 + byte 6 bits 0-1
- Plasticity bits 50-52: byte 6 bits 2-4
- W slot (bits 53-58): byte 6 bits 5-7 + byte 7 bits 0-2
- truth (bits 59-60): byte 7 bits 3-4
- spare (bits 61-63): byte 7 bits 5-7

**v1 compat guarantee for CausalEdge64::ZERO edges:**
If a v1 PAL8 file stores a zero-default edge, then w_slot()=0, truth()=Crystalline, mantissa=0.
Correct defaults. Round-trip safe.

**v1 compat HAZARD for non-zero v1 edges:**
v1 temporal in bits 52-63: temporal=128 (0x80) has bit 59=1 -> truth()=Solid/Fuzzy in v2. WRONG.
v1 temporal=4095 reads truth()=Murky, w_slot()=garbage. This is the most dangerous hazard.

**Required mitigation:** Version byte in PAL8 header to distinguish v1 from v2. W3 must implement.
Gate merge of PR-CE64-MB-2 on W3's regression spec landing first.

### 6.2 NarsTables LUT Layout

NarsTables is keyed on freq_u8 (bits 24-31) and conf_u8 (bits 32-39) only.
NarsTables::revise(f1, c1, f2, c2) and ::deduce(f1, f2) index arrays exclusively in the bits 0-39
range — entirely below the reclaim zone (bits 46-63). LUT shape unchanged.

In nars_engine.rs, SpoHead mirrors S/P/O palette + NARS truth + causal mask — bits 0-42 only.
SpoDistances tables are keyed on palette indices (bits 0-23). Style vectors operate on Pearl
projection scores from SpoDistances. None access bits 46-63.

**NarsTables and SpoDistances LUT invariant: unchanged. Zero regression risk.**

### 6.3 EdgeColumn (BindSpace Column D) Binary Layout

8 x CausalEdge64 = 64 B/row. The 8 edges per row are 8 u64 entries. The new G/W/truth fields live
inside the existing u64 — not as new columns. EdgeColumn binary layout: unchanged.
Cache-line alignment (64 bytes = 8 edges) preserved.

### 6.4 p64-bridge::STYLES Codebook

The 8-bit style ordinal referenced in plan §3 does NOT exist in the shipped CausalEdge64. Under
Option C, direction triad (bits 43-45) is preserved. Style slot semantics live in
AttentionMask::style_slots per plan §4 — not in CausalEdge64 itself.
p64-bridge::STYLES binary codebook format: unchanged.

---

## §7 Per-Method Semantics and Edge Cases

### w_slot(self) -> u8

- Returns 6-bit corpus root handle, 0..=63. 0 = "no corpus anchor."
- CausalEdge64::ZERO: returns 0 (correct: no corpus).
- v2-written edge: returns the value stamped by with_w_slot() or set_w_slot().
- v1-written non-zero edge: GARBAGE. Bits 53-58 were temporal MSBs in v1.
  Callers MUST apply a version gate before interpreting w_slot() on edges of unknown provenance.
- Atomicity: 64-bit load of self.0 is atomic on x86_64 and ARM64. No locking for single-edge reads.

### inference_mantissa(self) -> i8

- Returns signed i4 in range −8..+7. Zero = neutral/identity (no NARS rule applied).
- `signum()` = chain direction: + = forward-chain, − = backward-chain.
- `unsigned_abs() & 0x7` = base rule index (see §"Signed Mantissa Rationale" table).
- CausalEdge64::ZERO: returns 0 (neutral, correct default for bare SPO assertions).
- v1 compat: bits 46-48 held 3-bit unsigned InferenceType; sign-extends cleanly to +0..+7.
  v1 edges read as forward-direction with the original rule index. Zero regression for v1 readers
  that only consumed positive mantissa values.

### truth(self) -> TrustTexture

- Returns TrustTexture::Crystalline for CausalEdge64::ZERO (bits 59-60 = 00).
- Most dangerous v1 hazard: bits 59-60 were temporal bits 7-8. v1 edges with temporal >= 128
  (0x80) have bit 59=1 -> reads Solid or higher. A v1 edge with high temporal (= well-established
  history) appears contradicted in v2. This is the worst false-positive.

### with_routing(self, w, t) -> Self

- Single mask-and-or for W slot and truth band. Hot-path emit for MailboxSoA::dispatch_cycle().
- NOTE: No `g` parameter — G-slot is absent in v2 (L-3: redundant via palette family-prefix).
- Used when a compartment knows its W slot (corpus handle) and truth band (MUL gate state).
- v1 fields (S, P, O, freq, conf, causal, dir) are preserved unchanged.
- Composable: `edge.with_routing(12, TrustTexture::Solid).with_inference_mantissa(-1)`.

---

## §8 Files to Touch

| File | Change type | LOC estimate | Notes |
|---|---|---|---|
| crates/causal-edge/src/lib.rs | Modify | ~10 LOC | Add pub mod layout; + cfg pub use |
| crates/causal-edge/src/layout.rs | NEW | ~120 LOC | Shift constants, masks, TrustTexture, const_assert |
| crates/causal-edge/src/edge.rs | Modify | ~110 LOC | Add v2 accessors/builders under cfg; deprecate v1 |
| crates/causal-edge/Cargo.toml | Modify | ~8 LOC | Add features; bump to 0.2.0 |
| crates/causal-edge/src/tables.rs | No change | 0 LOC | LUT keyed on freq/conf only; unaffected |
| crates/causal-edge/src/network.rs | Minor | ~15 LOC | Update Debug; deprecate inference_type/plasticity |
| crates/causal-edge/src/pearl.rs | No change | 0 LOC | CausalMask unaffected |
| crates/causal-edge/src/plasticity.rs | Deprecate | ~8 LOC | PlasticityState deprecated with migration note |

**Total: ~240 LOC new + ~40 LOC modifications + deprecation annotations. Within ~400 LOC estimate.**

---

## §9 Test Plan

### Category 1 — Accessor Round-Trip (unit tests, src/v2_layout_tests.rs)

```
test_g_slot_roundtrip:
  For g in [0, 1, 15, 31]: assert CausalEdge64::ZERO.with_g_slot(g).g_slot() == g

test_w_slot_roundtrip:
  For w in [0, 1, 31, 63]: assert CausalEdge64::ZERO.with_w_slot(w).w_slot() == w

test_truth_roundtrip:
  For t in [Crystalline, Solid, Fuzzy, Murky]:
    assert CausalEdge64::ZERO.with_truth(t).truth() == t

test_with_routing_roundtrip:
  edge = CausalEdge64::ZERO.with_routing(17, 42, TrustTexture::Fuzzy)
  assert edge.g_slot() == 17
  assert edge.w_slot() == 42
  assert edge.truth() == TrustTexture::Fuzzy

test_v2_fields_do_not_disturb_v1_fields:
  base = CausalEdge64::pack(143, 7, 201, 209, 181, CausalMask::PO, 0b101, ...)
  v2 = base.with_routing(5, 10, TrustTexture::Solid)
  assert v2.s_idx() == 143
  assert v2.p_idx() == 7
  assert v2.o_idx() == 201
  assert v2.frequency_u8() == 209
  assert v2.confidence_u8() == 181
  assert v2.causal_mask() == CausalMask::PO
  assert v2.direction() == 0b101

test_zero_edge_v2_defaults:
  e = CausalEdge64::ZERO
  assert e.g_slot() == 0
  assert e.w_slot() == 0
  assert e.truth() == TrustTexture::Crystalline

test_g_slot_max_no_w_contamination:
  e = CausalEdge64::ZERO.with_g_slot(31)
  assert e.g_slot() == 31
  assert e.w_slot() == 0  // no cross-contamination

test_w_slot_max_no_g_contamination:
  e = CausalEdge64::ZERO.with_w_slot(63)
  assert e.w_slot() == 63
  assert e.g_slot() == 0  // no cross-contamination

test_truth_max_no_spare_contamination:
  e = CausalEdge64::ZERO.with_truth(TrustTexture::Murky)
  assert e.truth_raw() == 3
  assert e.w_slot() == 0  // bits 57-58 do not leak into W (bits 51-56)

test_size_unchanged:
  assert std::mem::size_of::<CausalEdge64>() == 8
  assert 8 * std::mem::size_of::<CausalEdge64>() == 64

test_const_assert_mask_coverage: [compile-time — passes if layout.rs compiles]
```

### Category 2 — PAL8 Round-Trip Regression (DEFER to W3)

Per plan §3 D-CE64-MB-2. Gate merge of PR-CE64-MB-2 on W3's regression landing first.
W3 must: (1) find/document PAL8 serialization impl, (2) add version byte, (3) verify zero-edge
round-trip, (4) verify non-zero v2 round-trip.

### Category 3 — NarsTables LUT Invariant (DEFER to W3)

Per plan §3 D-CE64-MB-3. NarsTables keys are freq_u8 and conf_u8 (bits 24-39). Trivially
unaffected by reclaim zone (bits 46-63) but must have regression test.

### Category 4 — EdgeColumn Binary Regression

Covered by test_size_unchanged. Confirm [CausalEdge64; 8] = 64 bytes = one cache line.

---

## §10 Risk Matrix

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| PAL8 version gate missing: v2 reads v1 PAL8 and returns garbage G/W/truth | HIGH | HIGH (v1 files have non-zero infer/plast/temporal by design) | Gate merge on W3's regression. Add PAL8 version byte before any v2 files are written. |
| OQ-LAYOUT-1 resolved as Option F: G-slot dropped; mantissa 4b signed; W+truth+spare added | LOW | RESOLVED | cognitive-substrate-convergence-v1.md §6 locks the layout. Implementation may proceed. |
| forward() reads InferenceType from bits 46-48 which are now G slot bits | HIGH | HIGH (forward() uses self.inference_type() that reads bits 46-48) | forward() must accept explicit InferenceType param. Scoped into PR-CE64-MB-2 or follow-on. |
| NarsTables LUT key shift: if bits 46-51 used as LUT keys anywhere not found in survey | MEDIUM | LOW (survey found no such usage; keys are freq/conf only) | W3 regression + grep for V1_INFER_SHIFT/V1_PLAST_SHIFT usage across all crates |
| inference_type() callers break when bits reclaimed | MEDIUM | MEDIUM (used in forward() and network.rs) | Deprecation markers + migration guide |
| TrustTexture local copy diverges from contract canonical | LOW | LOW (byte-compatible; 2-bit enum is trivial) | Long-term: From<local::TrustTexture> for contract::TrustTexture bridge |

---

## §11 Open Questions for Meta-Review

**OQ-LAYOUT-1 (RESOLVED 2026-05-16):** Ratified as Option F — drop temporal(12), expand inference
3b unsigned → 4b signed i4, add W(6)+truth-band-lens(2)+spare(3), DROP G-slot entirely. Locked by
cognitive-substrate-convergence-v1.md §6 (decisions L-2, L-3, L-4, L-6, L-7). Implementation may
proceed. No further adjudication needed.

**OQ-PAL8-FORMAT (BLOCKER for W3):** The 4101-byte PAL8 serialization implementation was NOT found
in crates/causal-edge/ during code survey. Is PAL8 serialization already shipped elsewhere, or is
it a planned deliverable? If not shipped, the compat analysis in §6.1 is analyzing a planned future
format. W3's spec is blocked on this answer.

**OQ-FORWARD-REFACTOR:** forward() uses self.inference_type() which reads bits 46-48 — these become
G slot bits under Option C. Is refactoring forward() in-scope for PR-CE64-MB-2 (accepting explicit
InferenceType from caller), or is it a follow-on PR? Recommendation: include in PR-CE64-MB-2.

**OQ-ZERODEP:** Should TrustTexture be defined locally in causal-edge (preserving zero-dep
invariant) or imported from lance-graph-contract (single canonical source, adds a dep)?
Recommendation: local definition with From conversion at the planner boundary.

**OQ-TEMPORAL-MIGRATION:** With local temporal dropped, CausalNetwork::causal_query() (sorts by
temporal) needs updating. In-scope for PR-CE64-MB-2 or follow-on in PR-CE64-MB-4?
Recommendation: follow-on in PR-CE64-MB-4 (AriGraph upgrade owns temporal context).

---

## §12 DELTA Summary vs. Parent Plan §3

Parent plan §3 settles: G(5) + W(6) + truth(2) = 13 bits from "reserved 13 bits."
This spec's implementation-side delta:

1. **Critical finding:** There are NO reserved bits in the shipped code. Reclaim requires dropping
   temporal(12 bits). Option F drops temporal, expands inference mantissa to 4b signed i4, adds W(6)
   + truth-band-lens(2) + spare(3). G-slot NOT added (L-3).

2. **Exact bit positions (Option F RATIFIED):** Mantissa=bits 46-49 (INFER_SHIFT=46, 4b signed),
   Plasticity=bits 50-52 (PLAST_SHIFT=50, shifted by 1), W=bits 53-58 (W_SHIFT=53),
   truth=bits 59-60 (TRUTH_SHIFT=59), spare=bits 61-63 (SPARE_SHIFT=61).

3. **Backward compat is more fragile than plan §3 suggests:** v1 PAL8 files with non-zero temporal
   will produce wrong truth band and W-slot values in v2 reads. A version gate is mandatory.

4. **Three v1 fields changed** (not "reserved"): temporal DROPPED; InferenceType EXPANDED to 4b
   signed; PlasticityState SHIFTED to bits 50-52. Each formally deprecated/noted with migration
   notes to AriGraph, signed-mantissa dispatch, and MailboxSoA.

5. **forward() requires refactor:** Core hot-path composition method uses InferenceType from the
   edge; must accept it as an explicit parameter after the bit reclaim.

6. **3 spare bits:** Bits 61-63 reserved for future use (Rubicon-commit marker, Markov-decay-rate
   quantum, I-NOISE-FLOOR-JIRAK threshold). NOT pre-allocated.

7. **causal-edge version bump:** 0.1.0 to 0.2.0 (minor version; layout-breaking for non-zero v1
   edges; SEMVER-compatible for ZERO-edge callers).

8. **Counterfactual is causal_mask=0b111, NOT a separate bit:** Per L-5 and §"Counterfactual via
   causal_mask". PR-LL-1 Counterfactual variant lands at mantissa −6 (magnitude slot 6, backward
   direction). No Pearl-3 modifier bit needed or correct.

---

## §13 Coordination Notes

- **W3** (pal8-nars-regression): Merge of PR-CE64-MB-2 is gated on W3's regression landing first.
  W3 must audit PAL8 format for version-gate (OQ-PAL8-FORMAT). Both specs must be reviewed together.
- **W1** (par-tile-crate): TrustTexture in par-tile AttentionMask should import from
  causal_edge::layout::TrustTexture (not contract) to keep the chain unambiguous.
- **W4** (bindspace-efgh): BindSpaceView reads W/truth from EdgeColumn CausalEdge64 entries.
  Must wait for PR-CE64-MB-2 to land first. Note: no G-slot in v2 layout.
- **W6** (mailbox-soa-attentionmask): MailboxSoA::dispatch_cycle() uses with_routing(w, t) to stamp
  W/truth onto emissions. Must wait for PR-CE64-MB-2. Note: with_routing signature changed (no `g`
  parameter in v2; G-slot dropped per L-3).
- **Meta-reviewer (Opus):** OQ-LAYOUT-1 is RESOLVED. OQ-FORWARD-REFACTOR is the primary remaining
  risk that may expand scope. All other OQs are implementation-detail decisions resolvable during
  PR review.
