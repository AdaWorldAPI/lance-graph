# PR-CE64-MB-2 — CausalEdge64 v2 Layout Extension Spec

> **Status:** Draft — sprint-log-10 W2 deliverable
> **PR scope:** `crates/causal-edge/` in-place layout extension; ~400 LOC
> **Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §3 (primary) + §2 (truth-band lens collapse)
> **Worker:** W2 (causaledge64-v2), Sonnet, sprint-log-10
> **Date:** 2026-05-14
> **Depends on:** PR-CE64-MB-1 (par-tile crate apex) — must land first so TrustTexture reference type is available
> **Blocks:** PR-CE64-MB-5 (MailboxSoA + AttentionMaskActor wiring, which reads G/W/truth from CausalEdge64)

---

## §0 Critical Finding — Actual vs. Plan §3 "Current" Layout Discrepancy

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

| Option | Strategy | Net bit delta | Risk |
|---|---|---|---|
| A | Compress temporal 12 to 4 bits; move absolute cycle to AriGraph SPO-G quad | Frees 8 bits; still need 5 more | HIGH |
| B | Replace direction(3) + inference(3) with G(5)+W(6) | Frees 6 bits; still need 7 more | HIGH |
| C (RECOMMENDED) | Drop temporal(12) to AriGraph; repurpose plasticity(3) for truth(2)+spare(1); G(5)+W(6) in freed 15 bits | Frees 15 bits, uses 13+2 spare | MEDIUM |
| D | Extend to CausalEdge128 (two u64) | No reclaim needed | HIGH for EdgeColumn |
| E | Complex split across dir+infer+plast | Complex mapping | HIGH |

**Option C is recommended.** Drop the 12-bit local temporal entirely (move to AriGraph SPO-G quad per
plan §5 E-CE64-MB-3). Repurpose plasticity(3) as truth(2) + 1 spare; add G(5) + W(6) in the freed
12+3=15 bits. This is backward-compatible for CausalEdge64::ZERO edges.

**BLOCKER:** This discrepancy must be resolved before implementation. Sections below are drafted
assuming Option C. If plan author selects a different option, the bit positions in §2 must be updated.

---

## §1 Statement of In-Place Reclaim (No Type Bump)

Per parent plan §3 doctrine: CausalEdge64 stays exactly 8 bytes (one u64, one register). No type
rename, no version suffix. The struct name CausalEdge64 is canonical; "v2" refers to the layout
extension, not the type.

**Binary contract:** A CausalEdge64 written by a v1 binary reads correctly by a v2 binary when the
new fields (G, W, truth) were zero in the v1-written value. For CausalEdge64::ZERO edges:
g_slot()=0, w_slot()=0, truth()=Crystalline — the correct "unrouted, no witness, fully trusted"
defaults. For non-zero v1 edges, the reclaimed bits will read garbage unless a version gate is
applied (see §6.1).

**Net bit allocation (assuming Option C is ratified):**

```
Total bits: 64
v1 fields kept:     S(8)+P(8)+O(8)+freq(8)+conf(8)+causal(3)+dir(3) = 46 bits
v1 fields dropped:  infer(3)+plasticity(3)+temporal(12) = 18 bits freed
v2 new fields:      G(5)+W(6)+truth(2)+spare(5) = 18 bits consumed
Net change:         0 (still 64 bits, no overflow)
```

**Architectural rationale for each dropped field:**
- InferenceType (3 bits, bits 46-48): subsumed by style-slot routing via AttentionMask. Inference
  type belongs in the session-ephemeral rename table, not in every edge.
- PlasticityState (3 bits, bits 49-51): subsumed by MailboxSoA::plasticity_counters column —
  plasticity lives at compartment granularity, not per-edge.
- temporal (12 bits, bits 52-63): subsumed by AriGraph SPO-G quad temporal annotation. "AriGraph IS
  the long-term memory" (plan §3 E-CE64-MB-3). Local temporal moves to AriGraph.

---

## §2 Bit-Layout Diagram — v2 Proposed (Option C)

Cite: parent plan §3 proposed v2 column + implementation-side delta from code survey.

```
v2 CausalEdge64 bit layout (LSB = bit 0, MSB = bit 63):

bit  63..59   58..57   56..51   50..46   45..43   42..40   39..32   31..24   23..16  15..8    7..0
     +-------+--------+--------+--------+--------+--------+--------+--------+--------+-------+--------+
     |spare  |truth   |W slot  |G slot  |dir     |causal  |conf    |freq    |O idx   |P idx  |S idx   |
     |5 bits |2 bits  |6 bits  |5 bits  |3 bits  |3 bits  |8 bits  |8 bits  |8 bits  |8 bits |8 bits  |
     +-------+--------+--------+--------+--------+--------+--------+--------+--------+-------+--------+
```

**Exact field positions (v2, Option C):**

| bits  | field          | count | shift | reclaimed? |
|-------|----------------|-------|-------|------------|
| 0-7   | S palette idx  | 8 b   | 0     | no         |
| 8-15  | P palette idx  | 8 b   | 8     | no         |
| 16-23 | O palette idx  | 8 b   | 16    | no         |
| 24-31 | NARS frequency | 8 b   | 24    | no         |
| 32-39 | NARS confidence| 8 b   | 32    | no         |
| 40-42 | Causal mask    | 3 b   | 40    | no         |
| 43-45 | Direction triad| 3 b   | 43    | no         |
| 46-50 | G slot         | 5 b   | 46    | YES (from infer+plast low) |
| 51-56 | W slot         | 6 b   | 51    | YES (from temporal low)    |
| 57-58 | truth band     | 2 b   | 57    | YES (from temporal mid)    |
| 59-63 | spare          | 5 b   | 59    | YES (from temporal MSBs)   |

**Sum check:** 8+8+8+8+8+3+3+5+6+2+5 = 64 bits. Verified.

---

## §3 Bit-Shift Constants Module (crates/causal-edge/src/layout.rs)

New file. All shift constants, masks, TrustTexture enum, and the const_assert guard live here.

```rust
//! CausalEdge64 v2 layout constants.
//!
//! Cite: causaledge64-mailbox-rename-soa-v1.md §3 + pr-ce64-mb-2-causaledge64-v2.md §2.

// v1 fields preserved (shifts unchanged from v1)
pub const S_SHIFT:      u32 = 0;
pub const P_SHIFT:      u32 = 8;
pub const O_SHIFT:      u32 = 16;
pub const FREQ_SHIFT:   u32 = 24;
pub const CONF_SHIFT:   u32 = 32;
pub const CAUSAL_SHIFT: u32 = 40;
pub const DIR_SHIFT:    u32 = 43;

// v1 fields DEPRECATED (renamed V1_ for PAL8 decode cross-reference)
#[deprecated(since = "0.2.0", note = "bits 46-48 reclaimed for G slot; use AttentionMask style routing")]
pub const V1_INFER_SHIFT: u32 = 46;
#[deprecated(since = "0.2.0", note = "bits 49-51 reclaimed for G slot high bits; use MailboxSoA::plasticity_counters")]
pub const V1_PLAST_SHIFT: u32 = 49;
#[deprecated(since = "0.2.0", note = "bits 52-63 reclaimed for W/truth/spare; use AriGraph SPO-G temporal")]
pub const V1_TEMPORAL_SHIFT: u32 = 52;

// v2 new fields (reclaimed from infer + plast + temporal)
/// 5-bit OGIT domain slot (0..=31). 0 = unrouted.
pub const G_SHIFT:     u32 = 46;
/// 6-bit witness palette slot (0..=63). 0 = no witness.
pub const W_SHIFT:     u32 = 51;
/// 2-bit truth band (TrustTexture ordinal). 0 = Crystalline (fully trusted).
pub const TRUTH_SHIFT: u32 = 57;
/// 5-bit spare — reserved for future use (e.g., compartment generation, ghost-edge flag).
pub const SPARE_SHIFT: u32 = 59;

// Masks
pub const BYTE_MASK:  u64 = 0xFF;
pub const BITS3_MASK: u64 = 0x7;
pub const BITS5_MASK: u64 = 0x1F;
pub const BITS6_MASK: u64 = 0x3F;
pub const BITS2_MASK: u64 = 0x3;
pub const BITS12_MASK:u64 = 0xFFF;

pub const G_MASK:     u64 = BITS5_MASK << G_SHIFT;
pub const W_MASK:     u64 = BITS6_MASK << W_SHIFT;
pub const TRUTH_MASK: u64 = BITS2_MASK << TRUTH_SHIFT;
pub const SPARE_MASK: u64 = BITS5_MASK << SPARE_SHIFT;

// Const-assert: all 64 bits covered exactly once.
// If this fails at compile time, the layout is inconsistent.
const _LAYOUT_COVERAGE: () = {
    let all: u64 = (BYTE_MASK << S_SHIFT)
        | (BYTE_MASK << P_SHIFT)
        | (BYTE_MASK << O_SHIFT)
        | (BYTE_MASK << FREQ_SHIFT)
        | (BYTE_MASK << CONF_SHIFT)
        | (BITS3_MASK << CAUSAL_SHIFT)
        | (BITS3_MASK << DIR_SHIFT)
        | (BITS5_MASK << G_SHIFT)
        | (BITS6_MASK << W_SHIFT)
        | (BITS2_MASK << TRUTH_SHIFT)
        | (BITS5_MASK << SPARE_SHIFT);
    // 8+8+8+8+8+3+3+5+6+2+5 = 64
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

```rust
#[cfg(feature = "causal-edge-v2-layout")]
impl CausalEdge64 {
    // ── Read Accessors ──────────────────────────────────────────────────────

    /// OGIT domain slot (5-bit, 0..=31). 0 = unrouted.
    /// WARNING: returns garbage for v1-written non-zero edges (bits 46-50
    /// were InferenceType+Plasticity in v1). Use version gate before calling
    /// on edges of unknown provenance. CausalEdge64::ZERO -> 0 (correct default).
    #[inline(always)]
    pub fn g_slot(self) -> u8 {
        use crate::layout::{G_SHIFT, BITS5_MASK};
        ((self.0 >> G_SHIFT) & BITS5_MASK) as u8
    }

    /// Witness palette slot (6-bit, 0..=63). 0 = no witness.
    /// Same v1-provenance caveat as g_slot().
    #[inline(always)]
    pub fn w_slot(self) -> u8 {
        use crate::layout::{W_SHIFT, BITS6_MASK};
        ((self.0 >> W_SHIFT) & BITS6_MASK) as u8
    }

    /// Truth band as TrustTexture (2-bit). Returns Crystalline for ZERO edges.
    /// WARNING: v1 edges with temporal >= 512 may read as Solid/Fuzzy/Murky.
    /// Bits 57-58 were temporal MSBs in v1 — high-temporal v1 edges appear
    /// contradicted in v2. Version gate required.
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

    /// Return new edge with G slot set. debug_assert!(g <= 31).
    #[inline]
    pub fn with_g_slot(self, g: u8) -> Self {
        use crate::layout::{G_SHIFT, BITS5_MASK, G_MASK};
        debug_assert!(g <= 31, "g_slot must fit 5 bits (0..=31), got {g}");
        Self((self.0 & !G_MASK) | (((g as u64) & BITS5_MASK) << G_SHIFT))
    }

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

    /// Set G + W + truth in one mask-and-or (hot-path emit operation).
    /// Used by MailboxSoA::dispatch_cycle() when stamping routing onto emissions.
    #[inline]
    pub fn with_routing(self, g: u8, w: u8, t: crate::layout::TrustTexture) -> Self {
        use crate::layout::{G_SHIFT, W_SHIFT, TRUTH_SHIFT, BITS5_MASK, BITS6_MASK, BITS2_MASK,
                            G_MASK, W_MASK, TRUTH_MASK};
        debug_assert!(g <= 31 && w <= 63, "g ({g}) or w ({w}) out of range");
        let routing = ((g as u64 & BITS5_MASK) << G_SHIFT)
            | ((w as u64 & BITS6_MASK) << W_SHIFT)
            | ((t.to_bits_2() as u64 & BITS2_MASK) << TRUTH_SHIFT);
        Self((self.0 & !(G_MASK | W_MASK | TRUTH_MASK)) | routing)
    }

    // ── Mutating Setters (hot-path, &mut self) ──────────────────────────────

    #[inline]
    pub fn set_g_slot(&mut self, g: u8) {
        use crate::layout::{G_SHIFT, BITS5_MASK, G_MASK};
        debug_assert!(g <= 31);
        self.0 = (self.0 & !G_MASK) | (((g as u64) & BITS5_MASK) << G_SHIFT);
    }

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
}

// v1 stub accessors (feature off) — return safe zero/Crystalline defaults
#[cfg(not(feature = "causal-edge-v2-layout"))]
impl CausalEdge64 {
    #[inline(always)] pub fn g_slot(self) -> u8 { 0 }
    #[inline(always)] pub fn w_slot(self) -> u8 { 0 }
    #[inline(always)] pub fn truth(self) -> TrustTexture { TrustTexture::Crystalline }
    #[inline(always)] pub fn truth_raw(self) -> u8 { 0 }
    #[inline] pub fn with_g_slot(self, _g: u8) -> Self { self }
    #[inline] pub fn with_w_slot(self, _w: u8) -> Self { self }
    #[inline] pub fn with_truth(self, _t: TrustTexture) -> Self { self }
    #[inline] pub fn with_routing(self, _g: u8, _w: u8, _t: TrustTexture) -> Self { self }
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

# v2 layout: G(5-bit OGIT domain slot) + W(6-bit witness slot) + truth(2-bit TrustTexture)
# carved from reclaimed bits per causaledge64-mailbox-rename-soa-v1.md §3 (Option C ratified).
# Off -> v1 accessor stubs return zeros (unrouted, no witness, Crystalline default).
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

The reclaimed bits land at these byte positions:
- G slot (bits 46-50): bit 46-47 in byte 5 (bits 6-7), bits 48-50 in byte 6 (bits 0-2)
- W slot (bits 51-56): bits 51-55 in byte 6 (bits 3-7), bit 56 in byte 7 bit 0
- truth (bits 57-58): byte 7 bits 1-2
- spare (bits 59-63): byte 7 bits 3-7

**v1 compat guarantee for CausalEdge64::ZERO edges:**
If a v1 PAL8 file stores a zero-default edge, then g_slot()=0, w_slot()=0, truth()=Crystalline.
Correct defaults. Round-trip safe.

**v1 compat HAZARD for non-zero v1 edges:**
v1 InferenceType in bits 46-48 is NOT zero in general (Deduction=0, Induction=1, Abduction=2, ...).
A v1 edge with InferenceType=Abduction(2) reads g_slot()=2 in v2 — NOT unrouted. GARBAGE.

v1 temporal in bits 52-63: temporal=1024 (0x400) has bit 58=1 -> truth()=Fuzzy in v2. WRONG.
v1 temporal=4095 reads truth()=Murky. This is the most dangerous hazard.

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

### g_slot(self) -> u8

- Returns 5-bit OGIT domain slot, 0..=31. 0 = "unrouted."
- CausalEdge64::ZERO: returns 0 (correct: unrouted default).
- v2-written edge: returns the value stamped by with_g_slot() or set_g_slot().
- v1-written non-zero edge: GARBAGE. Bits 46-50 were InferenceType(3) + Plasticity_low(2). May be
  any value 0..=31. Callers MUST apply a version gate before interpreting g_slot() on edges of
  unknown provenance.
- Atomicity: 64-bit load of self.0 is atomic on x86_64 and ARM64. No locking for single-edge reads.

### w_slot(self) -> u8

- Same edge-case semantics as g_slot(). Returns 0 for ZERO edges. GARBAGE for non-zero v1 edges.
- 6-bit range (0..=63). 0 = "no witness."

### truth(self) -> TrustTexture

- Returns TrustTexture::Crystalline for CausalEdge64::ZERO (bits 57-58 = 00).
- Most dangerous v1 hazard: bits 57-58 were temporal MSBs. v1 edges with temporal >= 512 (0x200)
  have bit 57=1 -> reads Solid or higher. v1 edges with temporal >= 1024 (0x400) have bit 58=1
  -> reads Fuzzy or Murky. A v1 edge with high temporal (= well-established history) appears
  contradicted in v2. This is the worst false-positive.

### with_g_slot(self, g: u8) -> Self

- Builder pattern: returns new CausalEdge64 with G slot set, all other fields preserved.
- debug_assert!(g <= 31) fires in debug on overflow; release silently masks to 5 bits.
- Composable: edge.with_g_slot(5).with_w_slot(12).with_truth(TrustTexture::Solid).
- Prefer with_routing() in hot-path code (single mask-and-or vs three sequential operations).

### with_routing(self, g, w, t) -> Self

- Single mask-and-or for all three routing fields. Hot-path emit for MailboxSoA::dispatch_cycle().
- Used when a compartment knows its G slot (AttentionMask::lookup_g()), W slot
  (AttentionMask::lookup_w()), and truth band (current MUL gate state).
- v1 fields (S, P, O, freq, conf, causal, dir) are preserved unchanged.

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
| Option C not ratified: plan author selects different strategy, bit positions change | HIGH | MEDIUM (plan §3 and actual code do not match; adjudication required) | Implementation blocked on OQ-LAYOUT-1. Do not start until resolved. |
| forward() reads InferenceType from bits 46-48 which are now G slot bits | HIGH | HIGH (forward() uses self.inference_type() that reads bits 46-48) | forward() must accept explicit InferenceType param. Scoped into PR-CE64-MB-2 or follow-on. |
| NarsTables LUT key shift: if bits 46-51 used as LUT keys anywhere not found in survey | MEDIUM | LOW (survey found no such usage; keys are freq/conf only) | W3 regression + grep for V1_INFER_SHIFT/V1_PLAST_SHIFT usage across all crates |
| inference_type() callers break when bits reclaimed | MEDIUM | MEDIUM (used in forward() and network.rs) | Deprecation markers + migration guide |
| TrustTexture local copy diverges from contract canonical | LOW | LOW (byte-compatible; 2-bit enum is trivial) | Long-term: From<local::TrustTexture> for contract::TrustTexture bridge |

---

## §11 Open Questions for Meta-Review

**OQ-LAYOUT-1 (BLOCKER):** Parent plan §3 "current CausalEdge64 layout" does not match edge.rs.
Which reclaim strategy (Options A-E in §0) does the plan author ratify? Spec recommends Option C.
If another option is selected, bit positions in §2 change. Implementation cannot start until resolved.

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
   InferenceType + PlasticityState + temporal. Option C recommended.

2. **Exact bit positions (Option C):** G=bits 46-50 (G_SHIFT=46), W=bits 51-56 (W_SHIFT=51),
   truth=bits 57-58 (TRUTH_SHIFT=57), spare=bits 59-63 (SPARE_SHIFT=59).

3. **Backward compat is more fragile than plan §3 suggests:** v1 PAL8 files with non-zero temporal
   will produce wrong truth band values in v2 reads. A version gate is mandatory.

4. **Three v1 fields deprecated** (not "reserved"): InferenceType, PlasticityState, temporal —
   each formally deprecated with migration notes to AttentionMask, MailboxSoA, AriGraph.

5. **forward() requires refactor:** Core hot-path composition method uses InferenceType from the
   edge; must accept it as an explicit parameter after the bit reclaim.

6. **5 spare bits:** Plan §3 does not account for these. Bits 59-63 reserved for future use
   (compartment generation counter, ghost-edge flag, style-cluster high bits).

7. **causal-edge version bump:** 0.1.0 to 0.2.0 (minor version; layout-breaking for non-zero v1
   edges; SEMVER-compatible for ZERO-edge callers).

---

## §13 Coordination Notes

- **W3** (pal8-nars-regression): Merge of PR-CE64-MB-2 is gated on W3's regression landing first.
  W3 must audit PAL8 format for version-gate (OQ-PAL8-FORMAT). Both specs must be reviewed together.
- **W1** (par-tile-crate): TrustTexture in par-tile AttentionMask should import from
  causal_edge::layout::TrustTexture (not contract) to keep the chain unambiguous.
- **W4** (bindspace-efgh): BindSpaceView reads G/W/truth from EdgeColumn CausalEdge64 entries.
  Must wait for PR-CE64-MB-2 to land first.
- **W6** (mailbox-soa-attentionmask): MailboxSoA::dispatch_cycle() uses with_routing() to stamp
  G/W/truth onto emissions. Must wait for PR-CE64-MB-2.
- **Meta-reviewer (Opus):** Flag OQ-LAYOUT-1 as the sole critical implementation blocker.
  OQ-FORWARD-REFACTOR as a secondary risk that may expand scope. All other OQs are
  implementation-detail decisions resolvable during PR review.
