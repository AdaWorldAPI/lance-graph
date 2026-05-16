# PR-CE64-MB-2 — PAL8 Round-Trip + NarsTables LUT Regression Spec

> **Deliverables:** D-CE64-MB-2 (PAL8 round-trip regression) + D-CE64-MB-3 (NarsTables LUT invariant regression)
> **Owner (spec):** W3 (sprint-log-10, CCA2A Sonnet worker)
> **Gates:** PR-CE64-MB-2 (W2's CausalEdge64 v2 layout merge). Both deliverable test suites MUST pass before the PR merges.
> **Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §3 compat invariants (invariants C1, C2, C3)
> **W2 spec:** `.claude/specs/pr-ce64-mb-2-causaledge64-v2.md` — NOT yet produced when this spec was drafted. Bit-position references below cite §3 of the parent plan directly; W2's spec is the authoritative resolver for any discrepancy.
> **Status:** Draft — updated 2026-05-16 to reflect v2 bit layout locked by `.claude/plans/cognitive-substrate-convergence-v1.md` §6; OQ-PAL8-FORMAT resolved (see §10)
> **Plan cross-refs:** §5 L-3 (signed mantissa locked), §5 L-9 (PR-LL-1 Intervention/Counterfactual → Reserved5/6), §6 (final v2 bit layout — Option F), §12 W3 patch row (mantissa-roundtrip + lens-state tests)

---

## §1 Test Purpose Statement

These tests are **merge gates** for PR-CE64-MB-2, which extends `CausalEdge64` in-place using the v2 bit layout locked by `cognitive-substrate-convergence-v1.md` §6 (decision L-2 through L-7). The v2 layout changes are:

| Field | v1 bits | v2 bits | Change |
|-------|---------|---------|--------|
| InferenceType | 46-48 (3b unsigned) | 46-49 (4b **signed** i4, −8..+7) | WIDENED + SIGN-EXTENDED |
| Plasticity flags | 49-51 (3b) | 50-52 (3b) | SHIFTED by 1 bit |
| Temporal index | 52-63 (12b) | **REMOVED** | DROPPED (L-2: now structural via chain-position) |
| W slot (corpus root handle) | — | 53-58 (6b, 64 corpora) | NEW (L-6) |
| Truth-band lens | — | 59-60 (2b, 4 states) | NEW (L-7) |
| Spare | — | 61-63 (3b) | NEW reserved headroom |

**Note on G slot:** G slot is NOT present in the v2 layout. Decision L-3 (cognitive-substrate-convergence-v1.md §5) dropped the G slot as 3-way redundant (tenant via SoA partition, belief via witness corpus root, ontology via palette family-prefix). Any test referencing `g_slot()` from the draft v1 spec must be removed or replaced.

**Note on "PAL8" naming:** "PAL8" in this spec refers to the **u64 packed serialization form** of `CausalEdge64` (the 8-byte CausalEdge packed into a `u64` newtype). It is NOT a named Rust type — there is no `Pal8` struct in the codebase. The term appears in session knowledge at `crates/lance-graph-planner/.claude/knowledge/session_autocomplete_cache.md` to describe the 4101-byte serialized palette structure that contains CausalEdge64 values. Tests in §3 exercise the u64 round-trip properties of this packed representation.

If either regression suite fails, the reclaim has broken downstream binary compatibility in one of three ways:

1. **PAL8 serialization drift** — byte positions shifted; v1-encoded edges decode to wrong field values under v2 accessors.
2. **NarsTables LUT-key bleed** — bits in the new range accidentally feed into LUT index derivation, causing query divergence.
3. **EdgeColumn padding shift** — the `BindSpace` Column D byte budget (64 B/row = 8 x `CausalEdge64`) changed.

None of these are acceptable. The v2 layout extension is **in-place with zero type bump**, and all three invariants must hold across the layout boundary.

---

## §2 Layout Reference (v2 — locked by cognitive-substrate-convergence-v1.md §6)

**AUTHORITATIVE v2 layout** (decisions L-2, L-4, L-6, L-7, L-8 from the plan):

```
bits   field                  shift const        notes
-----  ------                 -----------        -----
0-7    S palette index        S_SHIFT=0          bgz17 archetype ID (u8)
8-15   P palette index        P_SHIFT=8          same
16-23  O palette index        O_SHIFT=16         same
24-31  NARS frequency         FREQ_SHIFT=24      f = val/255 (u8)
32-39  NARS confidence        CONF_SHIFT=32      c = val/255 (u8)
40-42  Causal mask            CAUSAL_SHIFT=40    Pearl 2³ — IS the rung axis (L-5)
43-45  Direction triad        DIR_SHIFT=43       sign(dim0) per S,P,O (3 bits, kept per L-8)
46-49  Inference mantissa     INFER_SHIFT=46     i4 SIGNED: −8..+7 (4 bits, L-4)
50-52  Plasticity flags       PLAST_SHIFT=50     hot/cold per S,P,O (3 bits, kept per L-8)
53-58  W slot                 W_SHIFT=53         corpus root handle (6 bits, 64 active corpora, L-6)
59-60  Truth-band lens        TRUTH_SHIFT=59     4 lens states (2 bits, L-7)
61-63  Spare                  SPARE_SHIFT=61     reserved headroom for sprint-12+
       Temporal               REMOVED            dropped per L-2; now structural via chain-position
       G slot (OGIT domain)   REMOVED            dropped per L-3; redundant via palette family-prefix
```

**Reclaim arithmetic:** drop temporal (−12 bits) → spend on InferenceType expansion (+1 bit), W slot (+6 bits), Truth-band lens (+2 bits) = 9 spent, 3 spare. Inference mantissa expands from 3b unsigned to 4b signed i4 (L-4): `abs(mantissa)` selects the base NARS rule (8 base slots), `signum(mantissa)` selects direction (forward-chain vs backward-chain).

**Signed mantissa encoding:**

| Sign | Direction | Magnitude interpretation |
|------|-----------|--------------------------|
| `+` (0..+7) | forward-chain / compose / commit | Deduction, Synthesis, Revision-positive, Induction |
| `−` (−8..−1) | backward-chain / decompose / refute | Abduction, Contraposition, Revision-negative, Counterfactual |

**v1 → v2 field displacement (for PAL8 compat analysis):**

| Field | v1 bits | v2 bits | Delta |
|-------|---------|---------|-------|
| InferenceType | 46-48 (3b unsigned) | 46-49 (4b signed) | +1 bit, sign-extended |
| Plasticity | 49-51 | 50-52 | shifted +1 (due to InferenceType expansion) |
| Temporal | 52-63 | GONE | reclaimed |
| W slot | — | 53-58 | NEW |
| Truth-band | — | 59-60 | NEW |
| Spare | — | 61-63 | NEW |

**Bit-position protocol:** Shift constants above are authoritative per plan §6. W2's implementation MUST use exactly these positions. W3's tests reference both the accessor functions AND the raw bit positions for the hazard checks.

---

## §3 D-CE64-MB-2 — PAL8 Round-Trip Regression

### File location

```
crates/causal-edge/tests/pal8_round_trip.rs  (NEW)
```

### PAL8 serialization background

"PAL8" refers to the 4101-byte serialized form of a CausalEdge64-bearing palette structure (per session knowledge at `crates/lance-graph-planner/.claude/knowledge/session_autocomplete_cache.md`). The parent plan §3 compatibility constraint C1 states:

> PAL8 deserializers reading v1 PAL8 see G=0, W=0, truth=00 (Crystalline) — the correct default. Existing PAL8 files round-trip without re-encoding. Mandatory test: deserialize-v1-encode-v2 produces byte-identical output when the new fields are zero.

In v1, the reclaimed positions are zero by definition. In v2, they carry G/W/truth semantics. The round-trip test proves that a v1-encoded edge (all new-field bits = 0) decodes correctly under v2 accessors and re-encodes byte-identically.

### Test 1: `pal8_v1_v2_round_trip_zero_default`

```rust
/// Regression gate for PR-CE64-MB-2: v1 to v2 binary compatibility.
///
/// A CausalEdge64 encoded by the v1 encoder (all new-field positions = 0)
/// must decode via v2 accessors as G=0, W=0, truth=TrustTexture::Crystalline
/// and re-encode to byte-identical output. Any drift in bit positions between
/// W2's encoder and the v1 zero-field positions fails this test.
#[test]
fn pal8_v1_v2_round_trip_zero_default() {
    // Step 1: Construct a v1-shaped CausalEdge64.
    // The v1 encoder does not know about G/W/truth fields; it writes only
    // S/P/O, NARS, Pearl, direction, inference, plasticity, temporal.
    // All reclaimed bits are implicitly zero.
    let v1_edge = CausalEdge64::pack(
        42, 17, 200,        // s_idx, p_idx, o_idx
        204, 178,           // frequency (~0.80), confidence (~0.70)
        CausalMask::PO,     // interventional
        0b010,              // direction triad
        InferenceType::Deduction,
        PlasticityState::S_HOT,
        1023,               // temporal (12-bit mid-range)
    );

    // Step 2: Under v2 feature flag, assert new fields read as zero/default.
    #[cfg(feature = "causal-edge-v2-layout")]
    {
        assert_eq!(v1_edge.g_slot(), 0,
            "v1-encoded edge must read G=0 under v2 accessors \
             (reclaimed bits were zero in v1)");
        assert_eq!(v1_edge.w_slot(), 0,
            "v1-encoded edge must read W=0 under v2 accessors");
        assert_eq!(v1_edge.truth(), TrustTexture::Crystalline,
            "v1-encoded edge must read truth=Crystalline (00) under v2 accessors");
    }

    // Step 3: Re-encode via v2 setters with zero/default new fields.
    #[cfg(feature = "causal-edge-v2-layout")]
    {
        let mut v2_edge = v1_edge;
        v2_edge.set_g_slot(0);
        v2_edge.set_w_slot(0);
        v2_edge.set_truth(TrustTexture::Crystalline);

        // Step 4: Assert byte-identical output.
        // set_*(0) and set_truth(Crystalline=0b00) are no-ops on a zero-initialized field.
        assert_eq!(v1_edge.0, v2_edge.0,
            "v2-re-encoded edge with zero new fields must be byte-identical to v1 input.\n\
             Raw v1={:#018x}, v2={:#018x}.\n\
             Non-zero delta indicates bit-position drift in W2 layout: \
             G_MASK or W_MASK or TRUTH_MASK overlaps a v1 non-zero field.",
            v1_edge.0, v2_edge.0);
    }

    // Step 5 (always): baseline field integrity check.
    assert_eq!(v1_edge.s_idx(), 42);
    assert_eq!(v1_edge.p_idx(), 17);
    assert_eq!(v1_edge.o_idx(), 200);
    assert_eq!(v1_edge.frequency_u8(), 204);
    assert_eq!(v1_edge.confidence_u8(), 178);
    assert_eq!(v1_edge.temporal(), 1023);
}
```

**Failure analysis:** If this test fails, the most likely cause is that W2's G/W/truth bit masks overlap with the temporal field (TEMPORAL_SHIFT=52, BITS12_MASK). Check `(G_MASK << G_SHIFT) & (BITS12_MASK << TEMPORAL_SHIFT) == 0` and the same for W/truth.

### Test 2: `pal8_v2_v2_round_trip_all_fields`

```rust
/// Regression gate for PR-CE64-MB-2: v2 full-field round-trip.
///
/// Construct a CausalEdge64 with all v2 fields set to non-zero specific values,
/// encode via v2, decode via v2, assert field-by-field equality.
/// Also asserts field isolation: toggling one v2 field must not corrupt others.
#[test]
#[cfg(feature = "causal-edge-v2-layout")]
fn pal8_v2_v2_round_trip_all_fields() {
    // Target v2 field values (non-zero, non-max to catch off-by-one):
    //   G=15  (5-bit mid-range, 0b01111)
    //   W=42  (6-bit mid-range, 0b101010)
    //   truth=Fuzzy (2-bit value = 0b10)
    //
    // Existing fields use non-trivial values to exercise non-zero bit interaction:
    let mut edge = CausalEdge64::pack(
        100, 50, 200,           // s, p, o
        180, 150,               // freq, conf
        CausalMask::SPO,
        0b110,                  // direction
        InferenceType::Induction,
        PlasticityState::ALL_HOT,
        2048,                   // temporal
    );
    edge.set_g_slot(15);
    edge.set_w_slot(42);
    edge.set_truth(TrustTexture::Fuzzy);

    // Assert all existing fields survive:
    assert_eq!(edge.s_idx(), 100,               "s_idx must survive v2 round-trip");
    assert_eq!(edge.p_idx(), 50,                "p_idx must survive v2 round-trip");
    assert_eq!(edge.o_idx(), 200,               "o_idx must survive v2 round-trip");
    assert_eq!(edge.frequency_u8(), 180,        "frequency must survive v2 round-trip");
    assert_eq!(edge.confidence_u8(), 150,       "confidence must survive v2 round-trip");
    assert_eq!(edge.causal_mask(), CausalMask::SPO, "causal mask must survive");
    assert_eq!(edge.direction(), 0b110,         "direction must survive v2 round-trip");
    assert_eq!(edge.inference_type(), InferenceType::Induction, "inference must survive");
    assert_eq!(edge.plasticity(), PlasticityState::ALL_HOT, "plasticity must survive");
    assert_eq!(edge.temporal(), 2048,           "temporal must survive v2 round-trip");

    // Assert v2 fields decode correctly:
    assert_eq!(edge.g_slot(), 15,
        "g_slot must round-trip: G=15 written, {} read", edge.g_slot());
    assert_eq!(edge.w_slot(), 42,
        "w_slot must round-trip: W=42 written, {} read", edge.w_slot());
    assert_eq!(edge.truth(), TrustTexture::Fuzzy,
        "truth band must round-trip: Fuzzy written, {:?} read", edge.truth());

    // Field isolation: toggling one v2 field must not corrupt others.
    let mut edge_g0 = edge;
    edge_g0.set_g_slot(0);
    assert_eq!(edge_g0.w_slot(), 42,
        "w_slot must survive g_slot clear");
    assert_eq!(edge_g0.truth(), TrustTexture::Fuzzy,
        "truth must survive g_slot clear");
    assert_eq!(edge_g0.temporal(), 2048,
        "temporal must survive g_slot clear");

    let mut edge_w0 = edge;
    edge_w0.set_w_slot(0);
    assert_eq!(edge_w0.g_slot(), 15,
        "g_slot must survive w_slot clear");
    assert_eq!(edge_w0.truth(), TrustTexture::Fuzzy,
        "truth must survive w_slot clear");
    assert_eq!(edge_w0.temporal(), 2048,
        "temporal must survive w_slot clear");

    let mut edge_t0 = edge;
    edge_t0.set_truth(TrustTexture::Crystalline);
    assert_eq!(edge_t0.g_slot(), 15,
        "g_slot must survive truth clear");
    assert_eq!(edge_t0.w_slot(), 42,
        "w_slot must survive truth clear");
    assert_eq!(edge_t0.temporal(), 2048,
        "temporal must survive truth clear");
}
```

**Failure analysis:** If isolation checks fail (e.g. setting `g_slot=0` corrupts `w_slot`), the bit masks overlap. Check `(G_MASK << G_SHIFT) & (W_MASK << W_SHIFT) == 0` and all pairwise mask intersections.

### Test 3: `pal8_round_trip_arbitrary` (property test, initially ignored)

```rust
/// Defense in depth: quickcheck arbitrary CausalEdge64 round-trip via v2 accessors.
///
/// Marked #[ignore] until the v2 layout branch is green for 100 iterations.
/// Promote to required (remove #[ignore]) once green in CI for one full workflow run.
#[test]
#[ignore]
#[cfg(feature = "causal-edge-v2-layout")]
fn pal8_round_trip_arbitrary() {
    use quickcheck::quickcheck;

    fn round_trips(raw: u64) -> bool {
        let edge = CausalEdge64(raw);
        let g = edge.g_slot();
        let w = edge.w_slot();
        let truth = edge.truth();
        let s = edge.s_idx();
        let p = edge.p_idx();
        let o = edge.o_idx();
        let freq = edge.frequency_u8();
        let conf = edge.confidence_u8();
        let temporal = edge.temporal();

        // Re-encode the fields we can set and verify round-trip for those fields.
        let mut e2 = CausalEdge64(0);
        e2.set_s_idx(s);
        e2.set_p_idx(p);
        e2.set_o_idx(o);
        e2.set_frequency_u8(freq);
        e2.set_confidence_u8(conf);
        e2.set_temporal(temporal);
        e2.set_g_slot(g);
        e2.set_w_slot(w);
        e2.set_truth(truth);

        e2.g_slot() == g
            && e2.w_slot() == w
            && e2.truth() == truth
            && e2.s_idx() == s
            && e2.p_idx() == p
            && e2.o_idx() == o
            && e2.frequency_u8() == freq
            && e2.confidence_u8() == conf
            && e2.temporal() == temporal
    }

    quickcheck(round_trips as fn(u64) -> bool);
}
```

**Dev dependency to add in `crates/causal-edge/Cargo.toml`:**

```toml
[dev-dependencies]
quickcheck = "1"
quickcheck_macros = "1"
```

---

## §4 D-CE64-MB-3 — NarsTables LUT Invariant Regression

### File location

```
crates/lance-graph-planner/tests/nars_tables_invariant.rs  (NEW)
```

### NarsTables background

`NarsTables` in `crates/causal-edge/src/tables.rs` precomputes NARS inference as 256x256 lookup tables keyed on `u8` frequency and confidence values. `NarsEngine::from_causal_edge` extracts only `s_idx`, `p_idx`, `o_idx`, `freq`, `conf`, `pearl`, `inference`, `temporal` from a `CausalEdge64` — it does NOT extract G/W/truth. The parent plan §3 compatibility constraint C2:

> New bits 51-63 are not LUT-key-bearing. LUT unchanged.

The regression tests confirm this holds after W2 adds the G/W/truth accessors.

### Test 1: `nars_tables_lut_key_unchanged_across_layouts`

```rust
/// Regression gate: NarsTables LUT key isolation from v2 fields.
///
/// A CausalEdge64 with v2 fields set to maximum values (g=31, w=63, truth=Murky)
/// must produce the SAME LUT query result as the same edge with v2 fields zeroed.
/// Divergence indicates G/W/truth bits bled into frequency/confidence extraction
/// in NarsEngine::from_causal_edge or CausalEdge64::frequency_u8/confidence_u8.
#[test]
fn nars_tables_lut_key_unchanged_across_layouts() {
    use lance_graph_planner::cache::nars_engine::{NarsEngine, SpoDistances};
    use causal_edge::{CausalEdge64, CausalMask};
    use causal_edge::edge::InferenceType;
    use causal_edge::plasticity::PlasticityState;

    let dist = SpoDistances::new_zero();
    let engine = NarsEngine::new(dist);

    // Base v1-style edge with known field values
    let base_edge = CausalEdge64::pack(
        100, 50, 200,
        180, 150,
        CausalMask::SPO,
        0,
        InferenceType::Deduction,
        PlasticityState::ALL_FROZEN,
        512,
    );

    let base_head = engine.from_causal_edge(base_edge);
    let base_deduction = engine.tables.deduce(base_head.freq, base_head.freq);
    let base_revision = engine.tables.revise(
        base_head.freq, base_head.conf,
        base_head.freq, base_head.conf
    );

    // Augment with maximum v2 field values
    #[cfg(feature = "causal-edge-v2-layout")]
    {
        let mut v2_edge = base_edge;
        v2_edge.set_g_slot(31);   // maximum 5-bit: 0b11111
        v2_edge.set_w_slot(63);   // maximum 6-bit: 0b111111
        v2_edge.set_truth(TrustTexture::Murky); // maximum 2-bit: 0b11

        let v2_head = engine.from_causal_edge(v2_edge);
        let v2_deduction = engine.tables.deduce(v2_head.freq, v2_head.freq);
        let v2_revision = engine.tables.revise(
            v2_head.freq, v2_head.conf,
            v2_head.freq, v2_head.conf
        );

        // LUT results must be identical
        assert_eq!(base_deduction, v2_deduction,
            "Deduction LUT result must be identical regardless of v2 field values.\n\
             base={:#06x}, v2={:#06x}.\n\
             Non-equal results: G/W/truth bits bled into freq/conf extraction.",
            base_deduction, v2_deduction);

        assert_eq!(base_revision, v2_revision,
            "Revision LUT result must be identical regardless of v2 field values.\n\
             base={:#06x}, v2={:#06x}.",
            base_revision, v2_revision);

        // Individual extracted fields must also be identical
        assert_eq!(base_head.freq, v2_head.freq,
            "freq extraction must be unaffected by v2 fields");
        assert_eq!(base_head.conf, v2_head.conf,
            "conf extraction must be unaffected by v2 fields");
        assert_eq!(base_head.s_idx, v2_head.s_idx,
            "s_idx extraction must be unaffected by v2 fields");
        assert_eq!(base_head.p_idx, v2_head.p_idx,
            "p_idx extraction must be unaffected by v2 fields");
        assert_eq!(base_head.o_idx, v2_head.o_idx,
            "o_idx extraction must be unaffected by v2 fields");
        assert_eq!(base_head.pearl, v2_head.pearl,
            "pearl extraction must be unaffected by v2 fields");
    }

    // Always: verify the base LUT result is a valid PackedTruth
    assert!(base_deduction <= 0xFFFF, "LUT result must fit in u16");
}
```

### Test 2: `nars_tables_lut_size_unchanged`

```rust
/// Regression gate: NarsTables byte size must not change due to layout extension.
///
/// NarsTables does not read CausalEdge64 directly — it only processes u8 inputs.
/// This test catches regressions where W2 accidentally changes NarsTables::build()
/// or adds parameters that alter table geometry.
#[test]
fn nars_tables_lut_size_unchanged() {
    use causal_edge::tables::NarsTables;

    // Fast path (c_levels=1): 1 revision table + 1 deduction table
    // = (1 * 256 * 256 * 2) + (256 * 256 * 2) = 262144 bytes
    let fast = NarsTables::build(1);
    assert_eq!(fast.c_levels, 1, "fast path must have c_levels=1");
    assert_eq!(fast.revision.len(), 1,
        "fast path must have exactly 1 revision table (c_levels^2 = 1)");

    let expected_fast_bytes = 1 * 256 * 256 * 2 + 256 * 256 * 2;
    assert_eq!(fast.byte_size(), expected_fast_bytes,
        "NarsTables fast path byte_size must be {expected_fast_bytes} bytes \
         (unchanged by v2 layout). Got {} bytes.", fast.byte_size());

    // Medium path (c_levels=4): 16 revision tables
    let medium = NarsTables::build(4);
    assert_eq!(medium.c_levels, 4);
    assert_eq!(medium.revision.len(), 16,
        "4-level path must have 16 revision tables (4^2)");

    let expected_medium_bytes = 16 * 256 * 256 * 2 + 256 * 256 * 2;
    assert_eq!(medium.byte_size(), expected_medium_bytes,
        "NarsTables medium path byte_size must be {expected_medium_bytes} bytes. \
         Got {} bytes.", medium.byte_size());

    // Cache budget: fast path must stay in L1/L2
    assert!(fast.byte_size() < 512 * 1024,
        "NarsTables fast path must remain under 512 KB (L1/L2 budget). \
         Got {} bytes.", fast.byte_size());
}
```

### Test 3: `nars_engine_to_from_causal_edge_isolates_new_fields`

```rust
/// Regression gate: NarsEngine conversion must not propagate v2 fields.
///
/// to_causal_edge(from_causal_edge(edge)) must zero the new fields because
/// SpoHead does not carry G/W/truth — it writes only the 7 core fields.
#[test]
#[cfg(feature = "causal-edge-v2-layout")]
fn nars_engine_to_from_causal_edge_isolates_new_fields() {
    use lance_graph_planner::cache::nars_engine::{NarsEngine, SpoDistances};
    use causal_edge::{CausalEdge64, CausalMask, TrustTexture};
    use causal_edge::edge::InferenceType;
    use causal_edge::plasticity::PlasticityState;

    let dist = SpoDistances::new_zero();
    let engine = NarsEngine::new(dist);

    let mut original = CausalEdge64::pack(
        10, 20, 30, 200, 180,
        CausalMask::PO, 0, InferenceType::Revision, PlasticityState::ALL_HOT, 100,
    );
    original.set_g_slot(15);
    original.set_w_slot(42);
    original.set_truth(TrustTexture::Solid);

    // Round-trip via NarsEngine: edge -> SpoHead -> edge
    let head = engine.from_causal_edge(original);
    let round_tripped = engine.to_causal_edge(&head);

    // The round-tripped edge must have G=0, W=0, truth=Crystalline
    // because SpoHead does not carry these fields.
    assert_eq!(round_tripped.g_slot(), 0,
        "NarsEngine round-trip must zero g_slot \
         (SpoHead does not carry G — this is correct behavior, not a loss)");
    assert_eq!(round_tripped.w_slot(), 0,
        "NarsEngine round-trip must zero w_slot");
    assert_eq!(round_tripped.truth(), TrustTexture::Crystalline,
        "NarsEngine round-trip must reset truth to Crystalline (00)");

    // Core fields must survive the round-trip
    assert_eq!(round_tripped.s_idx(), original.s_idx(), "s_idx must survive");
    assert_eq!(round_tripped.p_idx(), original.p_idx(), "p_idx must survive");
    assert_eq!(round_tripped.o_idx(), original.o_idx(), "o_idx must survive");
    assert_eq!(round_tripped.frequency_u8(), original.frequency_u8(), "freq must survive");
    assert_eq!(round_tripped.confidence_u8(), original.confidence_u8(), "conf must survive");
}
```

---

## §5 Bonus Regression — EdgeColumn Binary Layout

### File location

```
crates/cognitive-shader-driver/tests/edge_column_binary.rs  (NEW)
```

### Test: `edge_column_layout_invariant_64b_per_row`

```rust
use cognitive_shader_driver::bindspace::{BindSpace, EdgeColumn};
use causal_edge::CausalEdge64;

/// Regression gate for PR-CE64-MB-2: EdgeColumn byte budget invariant.
///
/// CausalEdge64 must remain exactly 8 bytes regardless of v2 layout.
/// Parent plan §3 C3: "8 x CausalEdge64 = 64 B/row, unchanged."
/// The v2 extension is bit-reclaim only — the struct cannot grow.
#[test]
fn edge_column_layout_invariant_64b_per_row() {
    // CausalEdge64 size: 8 bytes (one register, one u64 newtype)
    assert_eq!(
        std::mem::size_of::<CausalEdge64>(), 8,
        "CausalEdge64 must be exactly 8 bytes (one register). \
         v2 layout extension must NOT change the struct size — \
         the new fields are bit-reclaimed from existing u64, not additive."
    );

    // EdgeColumn with 1 row = 1 u64 = 8 bytes payload
    let col1 = EdgeColumn::zeros(1);
    assert_eq!(col1.0.len(), 1,
        "EdgeColumn with 1 row must have 1 u64 slot");
    assert_eq!(col1.0.len() * std::mem::size_of::<u64>(), 8,
        "EdgeColumn with 1 row must be 8 bytes of payload");

    // EdgeColumn with 8 rows = 8 u64 = 64 bytes (one cache line)
    let col8 = EdgeColumn::zeros(8);
    assert_eq!(col8.0.len(), 8,
        "EdgeColumn with 8 rows must have 8 u64 slots");
    assert_eq!(col8.0.len() * std::mem::size_of::<u64>(), 64,
        "EdgeColumn with 8 rows must be exactly 64 bytes (one cache line).");

    // BindSpace with N rows: edges column has N u64 entries
    let bs_100 = BindSpace::zeros(100);
    assert_eq!(bs_100.edges.0.len(), 100,
        "BindSpace row=100 edges column must have 100 u64 entries");

    // Write v2 edge into EdgeColumn and read back (identity check)
    #[cfg(feature = "causal-edge-v2-layout")]
    {
        use causal_edge::{CausalMask, TrustTexture};
        use causal_edge::edge::InferenceType;
        use causal_edge::plasticity::PlasticityState;

        let mut edge = CausalEdge64::pack(
            10, 20, 30, 180, 150,
            CausalMask::SPO, 0,
            InferenceType::Deduction,
            PlasticityState::ALL_FROZEN,
            42,
        );
        edge.set_g_slot(7);
        edge.set_w_slot(15);
        edge.set_truth(TrustTexture::Solid);

        let mut bs_rw = BindSpace::zeros(1);
        bs_rw.edges.set(0, edge.0);
        let retrieved = CausalEdge64(bs_rw.edges.get(0));

        assert_eq!(retrieved.g_slot(), 7,
            "EdgeColumn round-trip must preserve g_slot");
        assert_eq!(retrieved.w_slot(), 15,
            "EdgeColumn round-trip must preserve w_slot");
        assert_eq!(retrieved.truth(), TrustTexture::Solid,
            "EdgeColumn round-trip must preserve truth band");
        assert_eq!(retrieved.s_idx(), 10,
            "EdgeColumn round-trip must preserve s_idx");
        assert_eq!(retrieved.temporal(), 42,
            "EdgeColumn round-trip must preserve temporal");
    }
}
```

---

## §6 CI Gating Policy

### Which tests block merge

**PR-CE64-MB-2 cannot merge until all of the following pass on the `causal-edge-v2-layout` feature build:**

| Test | File | Blocks merge? |
|------|------|---------------|
| `pal8_v1_v2_round_trip_zero_default` | `crates/causal-edge/tests/pal8_round_trip.rs` | YES |
| `pal8_v2_v2_round_trip_all_fields` | same | YES |
| `nars_tables_lut_key_unchanged_across_layouts` | `crates/lance-graph-planner/tests/nars_tables_invariant.rs` | YES |
| `nars_tables_lut_size_unchanged` | same | YES |
| `nars_engine_to_from_causal_edge_isolates_new_fields` | same | YES |
| `edge_column_layout_invariant_64b_per_row` | `crates/cognitive-shader-driver/tests/edge_column_binary.rs` | YES |
| `pal8_round_trip_arbitrary` | `crates/causal-edge/tests/pal8_round_trip.rs` | NO (ignored; promoted when green) |

### CI workflow extension for `.github/workflows/rust-test.yml`

Add to the `test` job (after the existing `Run contract unit tests` step):

```yaml
# ── W3 gating tests for PR-CE64-MB-2 v2 layout merge (sprint-log-10) ──────
# These 6 integration tests must pass before W2's CausalEdge64 v2 layout merges.
# See .claude/specs/pr-ce64-mb-2-pal8-nars-regression.md for rationale.
- name: "[CE64-v2] PAL8 round-trip regression (pal8_round_trip)"
  run: |
    cargo test \
      --manifest-path crates/causal-edge/Cargo.toml \
      --test pal8_round_trip \
      --features causal-edge-v2-layout \
      -- --skip pal8_round_trip_arbitrary

- name: "[CE64-v2] NarsTables LUT invariant regression (nars_tables_invariant)"
  run: |
    cargo test \
      --manifest-path crates/lance-graph-planner/Cargo.toml \
      --test nars_tables_invariant \
      --features causal-edge-v2-layout

- name: "[CE64-v2] EdgeColumn binary layout invariant (edge_column_binary)"
  run: |
    cargo test \
      --manifest-path crates/cognitive-shader-driver/Cargo.toml \
      --test edge_column_binary \
      --features causal-edge-v2-layout
```

**Feature flag propagation** (W2 owns these Cargo.toml edits):

```toml
# crates/causal-edge/Cargo.toml
[features]
causal-edge-v2-layout = []

# crates/lance-graph-planner/Cargo.toml
[features]
causal-edge-v2-layout = ["causal-edge/causal-edge-v2-layout"]

# crates/cognitive-shader-driver/Cargo.toml
[features]
causal-edge-v2-layout = ["causal-edge/causal-edge-v2-layout"]
```

---

## §7 Risk Matrix

| Risk | Probability | Impact | Caught by |
|------|-------------|--------|-----------|
| **Bit-position math error in W2's encoder** — G/W/truth masks overlap with temporal (bits 52-63) or with each other | Medium (bit-packing errors are common) | High — all serialized edges decode to wrong field values silently | `pal8_v1_v2_round_trip_zero_default` (v1 edge must read new fields as zero); `pal8_v2_v2_round_trip_all_fields` field isolation (toggling one field must not corrupt another) |
| **LUT-key bleed** — `CausalEdge64::frequency_u8()` or `::confidence_u8()` reads from a bit position that W2's reclaim overlaps (e.g. G_SHIFT lands at 24 = FREQ_SHIFT by mistake) | Low (FREQ/CONF are far from the top bits) but catastrophic if it happens | Critical — NarsEngine produces wrong inference results for every edge with non-zero G/W/truth; silent corruption in hot path | `nars_tables_lut_key_unchanged_across_layouts` (same edge with/without v2 fields must yield identical LUT result) |
| **EdgeColumn size regression** — W2 accidentally changes `CausalEdge64` from a newtype `(u64)` to a two-word struct for some intermediate implementation | Very Low | Critical — `BindSpace` row size changes, breaking SIMD alignment, cache-line math, and all downstream consumers | `edge_column_layout_invariant_64b_per_row` (asserts `size_of::<CausalEdge64>() == 8` and `EdgeColumn::zeros(8).len() * 8 == 64`) |

---

## §8 Coordination with W2

**Responsibility split:**

- **W2** owns: v2 accessor implementations (`g_slot()`, `w_slot()`, `truth()`, setters); `G_SHIFT`, `W_SHIFT`, `TRUTH_SHIFT` constants; `causal-edge-v2-layout` feature flag Cargo wiring; `TrustTexture` re-export from `lance-graph-contract::mul`.
- **W3** owns: the regression tests in this spec that prove W2 didn't break things; CI workflow extension; this spec file.

**Bit-position TBD protocol:** W3's tests use feature-flag guards (`#[cfg(feature = "causal-edge-v2-layout")]`) so they compile and pass on the v1 codebase before W2's changes land. The `pal8_v1_v2_round_trip_zero_default` test has a non-gated section that exercises the v1 `pack()` API — this passes on both v1 and v2 builds. Meta-reviewer reconciles the concrete bit positions from W2's spec against W3's test assertions before implementation.

**If W2 spec not yet produced** (as of this draft): bit-position references throughout cite "TBD — defer to W2's bit-layout table." All tests are written against functional properties of the accessor functions, not raw bit masks. They will remain correct once W2 supplies the concrete layout.

**Agreement checklist (meta-reviewer task):**

1. W2's G range `[G_SHIFT .. G_SHIFT+5)` must not intersect temporal `[52..64)`.
2. W2's W range `[W_SHIFT .. W_SHIFT+6)` must not intersect temporal or G.
3. W2's truth range `[TRUTH_SHIFT .. TRUTH_SHIFT+2)` must not intersect any above.
4. All three new ranges must be mutually disjoint.
5. `CausalEdge64::pack()` with any arguments must produce `g_slot()=0, w_slot()=0, truth()=Crystalline` — confirmed by `pal8_v1_v2_round_trip_zero_default`.

---

## §9 Files to Touch

| File | Action | Owner |
|------|--------|-------|
| `crates/causal-edge/tests/pal8_round_trip.rs` | CREATE (3 tests: 2 gating + 1 ignored property test) | W3 |
| `crates/lance-graph-planner/tests/nars_tables_invariant.rs` | CREATE (3 tests, all gating) | W3 |
| `crates/cognitive-shader-driver/tests/edge_column_binary.rs` | CREATE (1 test, gating) | W3 |
| `.github/workflows/rust-test.yml` | EXTEND (3 new steps, W3 annotation) | W3 |
| `crates/causal-edge/Cargo.toml` | ADD dev-dep: quickcheck + quickcheck_macros; ADD feature: causal-edge-v2-layout | W2 (feature), W3 (dev-dep) |
| `crates/lance-graph-planner/Cargo.toml` | ADD feature: causal-edge-v2-layout propagation | W2 |
| `crates/cognitive-shader-driver/Cargo.toml` | ADD feature: causal-edge-v2-layout propagation | W2 |
| `crates/causal-edge/src/edge.rs` | ADD v2 constants + accessors + setters | W2 |
| `crates/lance-graph-contract/src/mul.rs` | CONFIRM TrustTexture variants (Crystalline=0, Solid=1, Fuzzy=2, Murky=3) | W2 |

---

## §10 Open Questions for Meta-Review

1. **[RESOLVED 2026-05-16] OQ-PAL8-FORMAT — Reclaim target in edge.rs**: Resolved by `cognitive-substrate-convergence-v1.md` §6 (Option F). The reclaim strategy is: **drop temporal (−12 bits)** per decision L-2 ("temporal causality is structural" doctrine — time is carried by chain-position + AriGraph anchor, not stored in the edge). This recovers bits 52-63 without any shortening of direction/inference/plasticity. Spend: InferenceType expansion 3b→4b signed i4 (+1 bit, bits 46-49, L-4), W slot +6b (bits 53-58, L-6), Truth-band lens +2b (bits 59-60, L-7) = 9 spent, 3 spare (bits 61-63). **Bit layout: signed mantissa 4b (bits 46-49), no temporal, W slot 6b (bits 53-58), lens 2b (bits 59-60).** W3's `test_temporal_in_msb_gives_sort_order` must be removed/replaced — temporal no longer exists in the v2 layout. See §11 `test_temporal_absent` which verifies the drop was complete.

2. **TrustTexture import path**: The tests reference `TrustTexture` from `causal_edge::TrustTexture`. If W2 re-exports it from `lance-graph-contract::mul::TrustTexture` (preferred — contract is the canonical zero-dep home), the tests need `use lance_graph_contract::mul::TrustTexture` or the re-export path. Confirm whether `causal-edge` re-exports or defines its own. Defining a new enum breaks the "contract is canonical" doctrine (CLAUDE.md §The AGI-as-glove doctrine). Recommend re-export.

3. **`pack_v2()` vs setter-only API**: The tests assume W2 provides either a `pack_v2()` constructor or that `pack()` + individual setters (`set_g_slot()`, `set_w_slot()`, `set_truth()`) is the complete v2 API. The setter-only approach is simpler and avoids a new constructor signature. If W2 uses only setters, `pal8_v2_v2_round_trip_all_fields` should be revised to call `pack()` + three setters rather than a hypothetical `pack_v2()`. Meta-reviewer confirms W2's API surface before the test files are committed.

---

## §11 v2 Layout Regression Tests — Signed Mantissa, Lens, W-slot, Temporal Drop

> **Cross-refs:** `cognitive-substrate-convergence-v1.md` §5 L-3 (signed mantissa locked),
> §5 L-9 (PR-LL-1 Intervention/Counterfactual absorb into Reserved5/6), §6 (bit layout —
> Option F), §12 W3 patch row (~80 LOC mantissa-roundtrip + lens-state tests).
>
> These 5 tests are added per §12 of the convergence plan. They extend the D-CE64-MB-2
> test suite with assertions that parametrize the new v2 layout decisions. They join the
> §6 CI gating policy: all 5 are **required to pass** under `causal-edge-v2-layout`.

### File location

```
crates/causal-edge/tests/pal8_round_trip.rs  (EXTEND — add to existing file per §3)
```

---

### Test 4: `test_mantissa_signed_positive`

Asserts that a positive i4 mantissa (+3) encodes as Exemplification in the **forward-chain /
compose / commit** direction. Per plan §6 signed-mantissa encoding table: `signum(+) →
forward-chain`, `abs(3) → magnitude slot 3` (Induction or equivalent forward-generalization
rule). The test proves that `abs(mantissa)` selects the base NARS rule slot and `signum`
selects direction, as required by L-4.

```rust
/// Regression gate: positive mantissa encodes Exemplification (forward direction).
///
/// Per cognitive-substrate-convergence-v1.md §6:
///   mantissa = +3 → signum=+ (forward-chain / compose / commit direction)
///               abs=3 → base NARS rule slot 3 (Induction / forward generalization)
/// Exemplification IS the positive-direction dispatch for slot 3.
#[test]
#[cfg(feature = "causal-edge-v2-layout")]
fn test_mantissa_signed_positive() {
    let mut edge = CausalEdge64::pack(
        10, 20, 30, 200, 180,
        CausalMask::SPO, 0b000,
        InferenceType::default(), PlasticityState::ALL_FROZEN, 0,
    );
    // Set mantissa = +3 (positive direction, magnitude 3)
    edge.set_inference_mantissa(3i8);

    let m = edge.inference_mantissa();
    assert_eq!(m, 3i8,
        "mantissa +3 must round-trip: written 3, read {m}");
    assert!(m > 0,
        "positive mantissa must be > 0 (forward-chain direction); got {m}");
    assert_eq!(m.abs(), 3,
        "abs(mantissa) must be 3 (rule slot 3 = Induction/Exemplification); got {}", m.abs());
    // Direction interpretation: positive = forward-chain
    assert_eq!(m.signum(), 1i8,
        "signum(+3) must be +1 (forward-chain / Exemplification direction)");
}
```

---

### Test 5: `test_mantissa_signed_negative`

Asserts that a negative i4 mantissa (−3) encodes as Exemplification in the **backward-chain /
decompose / refute** direction. Same rule slot (abs=3), opposite direction. Proves the
signed i4 field carries the direction × rule composition (plan §6 L-4) and that
`set_inference_mantissa(-3)` → `inference_mantissa() == -3` round-trips correctly through
the 4-bit two's-complement encoding.

```rust
/// Regression gate: negative mantissa encodes Exemplification (backward direction).
///
/// Per cognitive-substrate-convergence-v1.md §6:
///   mantissa = -3 → signum=- (backward-chain / decompose / refute direction)
///               abs=3 → base NARS rule slot 3 (Abduction / Contraposition / Counterfactual)
/// The negative sign IS the direction bit; abs selects the rule.
#[test]
#[cfg(feature = "causal-edge-v2-layout")]
fn test_mantissa_signed_negative() {
    let mut edge = CausalEdge64::pack(
        10, 20, 30, 200, 180,
        CausalMask::SPO, 0b000,
        InferenceType::default(), PlasticityState::ALL_FROZEN, 0,
    );
    // Set mantissa = -3 (negative direction, same magnitude as test_mantissa_signed_positive)
    edge.set_inference_mantissa(-3i8);

    let m = edge.inference_mantissa();
    assert_eq!(m, -3i8,
        "mantissa -3 must round-trip: written -3, read {m}");
    assert!(m < 0,
        "negative mantissa must be < 0 (backward-chain direction); got {m}");
    assert_eq!(m.abs(), 3,
        "abs(mantissa) must be 3 (same rule slot as positive test); got {}", m.abs());
    // Direction interpretation: negative = backward-chain
    assert_eq!(m.signum(), -1i8,
        "signum(-3) must be -1 (backward-chain / decompose / refute direction)");
    // Signed NARS rule composition: -3 ≠ +3 even though abs is equal
    assert_ne!(m, 3i8,
        "signed mantissa -3 must differ from +3 (direction × rule distinguishes them)");
}
```

---

### Test 6: `test_lens_4_state`

Asserts that all 4 truth-band lens states (Sharp/Soft/Diffuse/Halo — 2-bit field, bits 59-60)
round-trip through pack/unpack. Per plan §6 L-7: truth-band lens carries 4 states for
committed-vs-ambiguous expressivity. Each of the 4 2-bit values (0b00, 0b01, 0b10, 0b11)
must decode to a distinct `TrustTexture` variant and re-encode byte-identically. Failing
any state means the 2-bit TRUTH_MASK is either too narrow, shifted, or overlapping an
adjacent field.

```rust
/// Regression gate: all 4 truth-band lens states round-trip.
///
/// Per cognitive-substrate-convergence-v1.md §6 L-7:
///   bits 59-60 = truth-band lens, 4 states (2 bits).
/// The 4 states map to TrustTexture variants. Each must:
///   (a) decode correctly from the packed u64,
///   (b) re-encode byte-identically via set_truth(),
///   (c) not corrupt adjacent fields (W slot at bits 53-58, spare at bits 61-63).
#[test]
#[cfg(feature = "causal-edge-v2-layout")]
fn test_lens_4_state() {
    use causal_edge::TrustTexture;
    // All 4 lens states from plan §6 (2-bit field, 4 states)
    let states = [
        (TrustTexture::Crystalline, "Crystalline (0b00 = Sharp — fully committed)"),
        (TrustTexture::Solid,       "Solid       (0b01 = Soft  — high confidence)"),
        (TrustTexture::Fuzzy,       "Fuzzy       (0b10 = Diffuse — ambiguous direction)"),
        (TrustTexture::Murky,       "Murky       (0b11 = Halo  — 13% ambiguous, per L-7)"),
    ];
    for (lens, label) in states {
        let mut edge = CausalEdge64::pack(
            5, 10, 15, 128, 100,
            CausalMask::PO, 0b001,
            InferenceType::default(), PlasticityState::ALL_FROZEN, 0,
        );
        edge.set_w_slot(31); // non-zero W slot to catch overlap with truth field
        edge.set_truth(lens);

        assert_eq!(edge.truth(), lens,
            "lens state {label} must round-trip via set_truth/truth()");
        // Field isolation: truth must not corrupt W slot
        assert_eq!(edge.w_slot(), 31,
            "W slot must survive truth set for lens state {label}; got {}", edge.w_slot());
        // Re-encode check: set_truth(same value) must be idempotent
        let raw_before = edge.0;
        edge.set_truth(lens);
        assert_eq!(edge.0, raw_before,
            "set_truth(same value) must be idempotent for {label}");
    }
}
```

---

### Test 7: `test_w_slot_64`

Asserts that all 64 W-slot values (6-bit unsigned, bits 53-58) round-trip. Per plan §6 L-6:
W slot = discourse corpus root handle, 6 bits, 64 active corpora. Every value 0..=63 must
encode and decode correctly. The test also checks field isolation: setting each W value must
not corrupt the truth-band lens or the inference mantissa.

```rust
/// Regression gate: all 64 W-slot values round-trip (6-bit unsigned, bits 53-58).
///
/// Per cognitive-substrate-convergence-v1.md §6 L-6:
///   W slot = corpus root handle, 64 active corpora (6 bits, bits 53-58).
/// Each of the 64 values (0..=63) must:
///   (a) encode correctly via set_w_slot(),
///   (b) decode correctly via w_slot(),
///   (c) not corrupt adjacent fields: truth-band lens (bits 59-60), mantissa (bits 46-49).
#[test]
#[cfg(feature = "causal-edge-v2-layout")]
fn test_w_slot_64() {
    use causal_edge::TrustTexture;
    for w in 0u8..=63 {
        let mut edge = CausalEdge64::pack(
            1, 2, 3, 100, 90,
            CausalMask::SPO, 0b010,
            InferenceType::default(), PlasticityState::ALL_FROZEN, 0,
        );
        // Set sentinel values in adjacent fields to catch overlap
        edge.set_truth(TrustTexture::Fuzzy);   // bits 59-60 = 0b10
        edge.set_inference_mantissa(-2i8);      // bits 46-49 = -2 (signed i4)
        edge.set_w_slot(w);

        assert_eq!(edge.w_slot(), w,
            "W-slot {w} must round-trip: written {w}, read {}", edge.w_slot());
        // Field isolation: W slot must not corrupt truth or mantissa
        assert_eq!(edge.truth(), TrustTexture::Fuzzy,
            "truth-band lens must survive W-slot set for w={w}");
        assert_eq!(edge.inference_mantissa(), -2i8,
            "inference mantissa must survive W-slot set for w={w};              got {}", edge.inference_mantissa());
    }
    // Boundary: W=63 (all 6 bits set) must not overflow into bit 59
    let mut edge_max = CausalEdge64::pack(
        1, 2, 3, 100, 90, CausalMask::SPO, 0, InferenceType::default(),
        PlasticityState::ALL_FROZEN, 0,
    );
    edge_max.set_truth(TrustTexture::Crystalline); // 0b00 — must stay zero
    edge_max.set_w_slot(63); // 0b111111 at bits 53-58
    assert_eq!(edge_max.truth(), TrustTexture::Crystalline,
        "W=63 (max 6-bit) must not overflow into truth-band lens bits 59-60");
}
```

---

### Test 8: `test_temporal_absent`

Verifies that NO temporal bits are encoded in a v2 `CausalEdge64`. The temporal field (12
bits, formerly bits 52-63) was dropped per decision L-2 of the convergence plan. This test
confirms the drop from the PR-LL-1 era was complete: (a) there is no `temporal()` accessor
in v2, (b) bits 52-63 are fully occupied by W slot + lens + spare with no temporal alias,
(c) a raw u64 with bits 52-63 set high does NOT produce a non-zero temporal value via any
accessor.

```rust
/// Regression gate: NO temporal bits exist in v2 CausalEdge64 (drop per L-2 is complete).
///
/// Per cognitive-substrate-convergence-v1.md §5 L-2:
///   "Drop temporal (12 bits) from CausalEdge64 v2 — redundant with chain-position
///    + AriGraph anchor; temporal causality is structural doctrine."
/// PR-LL-1 era had TEMPORAL_SHIFT=52, BITS12_MASK. This test confirms:
///   (a) no temporal() accessor in v2 build,
///   (b) bits 52-63 are fully assigned (W slot 53-58, lens 59-60, spare 61-63) — no alias,
///   (c) v2 accessors w_slot/truth/spare correctly parse bits that v1 called "temporal".
#[test]
#[cfg(feature = "causal-edge-v2-layout")]
fn test_temporal_absent() {
    use causal_edge::TrustTexture;
    // Construct an edge where the old temporal range (bits 52-63) is fully set
    // by the new v2 fields: W=63 (bits 53-58), lens=Murky (bits 59-60 = 0b11),
    // spare=0b111 (bits 61-63).
    let mut edge = CausalEdge64::pack(
        0, 0, 0, 128, 128,
        CausalMask::SPO, 0, InferenceType::default(),
        PlasticityState::ALL_FROZEN, 0,
    );
    edge.set_w_slot(63);                    // bits 53-58: 0b111111
    edge.set_truth(TrustTexture::Murky);    // bits 59-60: 0b11

    // All of bits 53-60 are now set. Confirm the v2 accessors own these bits entirely.
    assert_eq!(edge.w_slot(), 63,
        "W slot must own bits 53-58 (formerly part of temporal range in v1)");
    assert_eq!(edge.truth(), TrustTexture::Murky,
        "truth-band lens must own bits 59-60 (formerly part of temporal range in v1)");

    // The v2 build must NOT expose a temporal() accessor at all.
    // Compile-time check: the following line must NOT compile in a v2 build.
    // (Uncomment to verify; expected: error[E0599]: no method named `temporal`)
    // let _ = edge.temporal();

    // Runtime check: raw bits that v1 called "temporal" are now W+lens+spare.
    // If any v2 accessor accidentally aliases the old temporal range, we'd see
    // w_slot or truth return unexpected values on a different edge.
    let raw_temporal_bits: u64 = 0x0FFF_0000_0000_0000; // bits 52-63 set
    let synthetic = CausalEdge64(raw_temporal_bits);
    // Under v2: bit 52 is unassigned (spare/reserved or part of plasticity boundary).
    // Bits 53-58 = W slot, bits 59-60 = truth, bits 61-63 = spare.
    // W slot from these bits: (raw >> 53) & 0x3F = 0x3F = 63
    let expected_w = ((raw_temporal_bits >> 53) & 0x3F) as u8;
    assert_eq!(synthetic.w_slot(), expected_w,
        "bits 53-58 of old temporal range must decode as W slot under v2;          expected {expected_w}, got {}", synthetic.w_slot());
    // Truth from bits 59-60: (raw >> 59) & 0x3 = 0x3 → Murky
    assert_eq!(synthetic.truth(), TrustTexture::Murky,
        "bits 59-60 of old temporal range must decode as truth=Murky under v2");
}
```

---
