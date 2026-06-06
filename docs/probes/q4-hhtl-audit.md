# Q4 Probe — HHTL Audit: Address Algebra vs Accretion

> **Branch:** `claude/stoic-turing-M0Eiq`
> **Date:** 2026-06-06
> **Files audited:** `crates/lance-graph-contract/src/hhtl.rs` (19 KB, ~470 lines),
> `crates/lance-graph-contract/src/high_heel.rs` (42 KB, ~900 lines)
> **Method:** Read both files completely. Classify every struct/enum/function
> into exactly one of five categories. No narrative — code citations.

---

## Headline finding

**The address algebra lives almost entirely in `hhtl.rs`. `high_heel.rs` contains
essentially zero address algebra.**

The two files share the "HHTL" brand but are different concepts.
`hhtl.rs` is a clean nibble-addressed prefix tree. `high_heel.rs` is a binary container
format + an online metric clustering accumulator + a lens-calibration subsystem.
They communicate in module docs, not in code.

---

## Formal definition of an HHTL address (what the code actually says)

**`hhtl.rs` — the real address: `NiblePath { path: u64, depth: u8 }`**

- Fan-out: 16 (`FAN_OUT = 16`). One nibble (4 bits) per level, root-first.
- Max depth: 16 (`MAX_DEPTH = 16`). Max 16 nibbles = 64 bits.
- **Truncation is explicit and exact:**
  - `parent()` → `path >> 4, depth - 1`
  - `child(n)` → `(path << 4) | n, depth + 1`
  - These are literal inverses. No ambiguity.
- **Containment is a single prefix kernel** — `is_ancestor_of` shifts `other.path`
  right by `4 * (other.depth - self.depth)` nibbles and compares. One function,
  called by all relation operations. No ad-hoc re-implementation anywhere.

**`high_heel.rs` — NOT an address.**

`Heel::dn_address: u64` is an opaque identity key. It is never truncated,
never prefix-compared, never tested for containment. The field name contains
"address" but the semantics are identity, not addressing. The module doc
claims "HHTL cascade mapping (HEEL=scent / HIP=palette / TWIG=SpoBase17 / LEAF=full
planes)" — this is a layout legend in a comment. **No code routes by prefix.**

---

## Classification table

### `hhtl.rs` (~470 lines incl. tests)

| Symbol | Category |
|--------|----------|
| `NiblePath { path, depth }` | **1 — Address algebra** |
| `FAN_OUT = 16`, `MAX_DEPTH = 16`, `EMPTY` | **1 — Address algebra** (the axioms) |
| `root()`, `child(n)`, `try_child(n)`, `parent()` | **1 — Address algebra** (extension / truncation) |
| `basin()`, `leaf()`, `depth()`, `packed()` | **1 — Address algebra** (prefix readout) |
| `is_ancestor_of` | **1 — Address algebra** (the containment kernel — see §6 below) |
| `is_descendant_of` | **3 — Topology consequence** (`other.is_ancestor_of(self)` delegation) |
| `is_sibling_of` | **3 — Topology consequence** (same parent, different child) |
| `common_ancestor` | **3 — Topology consequence** (LCA via depth alignment) |
| `is_full` | **2 — LOD consequence** (depth == MAX\_DEPTH = u64 capacity limit) |
| `FieldMask` usage in test impls | **3 — Topology consequence** |

### `high_heel.rs` (~900 lines incl. tests)

| Symbol | Category |
|--------|----------|
| `CONTAINER_WORDS/BYTES`, `HEEL_WORDS`, `MAX_EDGES` | **5 — Unrelated utility** (container layout constants) |
| `SpoBase17` + `l1_distance`, `l1_subject`, `scent()` | **5 — Unrelated utility** (distance metrics on i16 planes) |
| `Heel { dn_address, frequency, confidence, scent, plasticity, temporal }` | **5 — Unrelated utility** (NARS bitfield packing) |
| `pack_truth_meta`, `unpack_truth_meta` | **5 — Unrelated utility** (NARS bit serialization) |
| `HighHeelBGZ { buf: [u64; CONTAINER_WORDS] }` | **5 — Unrelated utility** (wire container) |
| `HighHeelBGZ::new`, `add_edge`, `edge_count`, `wire_size` | **5 — Unrelated utility** |
| `pack`/`unpack`, `spo_to_bytes`/`bytes_to_spo` | **5 — Unrelated utility** (serialization) |
| `is_crystallized` | **4 — Special-case accretion** (NARS confidence threshold state machine) |
| `revise_truth` | **4 — Special-case accretion** (4-branch plasticity threshold magic numbers) |
| `BasinAccumulator`, `ingest`, `calibrate`, `stats` | **4 — Special-case accretion** (online metric clustering + EMA centroid) |
| `BasinStats` | **5 — Unrelated utility** (monitoring DTO) |
| `EncodingPath`, `LensProfile`, `LensProfile::build` | **5 — Unrelated utility / should move out** (encoding calibration — zero addressing) |
| `LensConfig`, `LensFamily`, `TokenizerFamily`, `LENS_REGISTRY` | **5 — Unrelated utility / should move out** (hardcoded model registry) |
| experiment / test harness | test infrastructure |

---

## LOC estimate per category (both files combined, ~1370 lines)

| Category | LOC | Where |
|----------|-----|-------|
| **1 — Address algebra** | ~110 | `hhtl.rs` only |
| **2 — LOD consequence** | ~8 | `hhtl.rs` only |
| **3 — Topology consequence** | ~70 | `hhtl.rs` only |
| **4 — Special-case accretion** | ~170 | `high_heel.rs` only |
| **5 — Unrelated utility / move out** | ~620 | `high_heel.rs` only |
| tests / experiments | ~390 | both |

**Address algebra + direct consequences: ~190 LOC, ~14% of total.
All 14% lives in `hhtl.rs`. `high_heel.rs` is 0% address algebra.**

---

## Verdict: Is HHTL fundamentally addressing?

**Partial — and split along file lines.**

`hhtl.rs` **is** fundamentally addressing. Every interesting behavior derives from two
axioms: `child = (path << 4) | n` and prefix-compare via right-shift.
`is_ancestor_of`, `is_descendant_of`, `is_sibling_of`, `common_ancestor`, `is_full`,
`parent` — all follow. No special cases. Claim vindicated *here*.

`high_heel.rs` is **not** addressing. Belonging is decided by `l1_distance < threshold`
(metric clustering), not prefix containment. "HHTL" in this file is a branding label,
not a shared abstraction. The file has become a bucket: three independent subsystems
(wire container, basin clustering, lens calibration) filed under one name.

---

## The kernel that proves the concept

`NiblePath::is_ancestor_of` in `hhtl.rs`:

```rust
pub const fn is_ancestor_of(self, other: Self) -> bool {
    if self.depth == 0 || self.depth > other.depth {
        false
    } else {
        (other.path >> (4 * (other.depth as u32 - self.depth as u32))) == self.path
    }
}
```

Four lines. Const, O(1). Every other relation reduces to it. This single function proves
"HHTL is addressing" — truncate `other` to `self`'s depth by shifting off trailing
nibbles, compare prefix. This is the load-bearing stone.

---

## Three most egregious accretions in `high_heel.rs`

### A1 — `LensProfile` / `LensConfig` / `LENS_REGISTRY` (~290 LOC)

Encoding-distortion profiling with hardcoded `cos_range`, `gamma_offset`,
`TokenizerFamily` (jina-v3/bge-m3/reranker-v3), model-specific lens configs.
**Zero addressing, zero clustering, zero container relationship** to the rest of the file.
It is a transplanted calibration module.

**Action:** Extract to `lens_profile.rs`. Long-term: move out of `lance-graph-contract`
entirely (the zero-deps invariant is not violated by its presence, but it has no business
in the contract surface — it is an encoding implementation detail).

### A2 — `BasinAccumulator::calibrate` + EMA centroid drift (~120 LOC combined)

Online metric clustering with exponential-moving-average centroid updates and
pairwise-distance percentile auto-threshold. Sophisticated but entirely metric,
not derivable from any prefix axiom.

**Additional smell:** The test prints its own clustering failure:
`eprintln!("Low merge ratio … threshold may be too tight")`. The file ships
embedded evidence that the algorithm does not work well.

**Action:** Extract to `basin_accumulator.rs` or a dedicated clustering module.

### A3 — `revise_truth` plasticity state machine

```rust
let new_plasticity = if merged_c > 0.8 { 0 } else if merged_c > 0.6 { 1 }
                     else if merged_c > 0.3 { 2 } else { 3 };
```

Four branches, three magic-number thresholds, zero citation. Not derivable from
address structure. Per `I-LEGACY-API-FEATURE-GATED`: magic thresholds must say
"hand-tuned" and cite the calibration experiment. These don't.

**Action:** Extract to a `nars_truth` helper or document thresholds with rationale.

---

## The dead cascade

**`scent()` — the advertised 95%-rejection HEEL pre-filter — is dead code in the merge path.**

`scent()` is computed on `SpoBase17` and tested in isolation. `BasinAccumulator::ingest`
does a flat linear scan over all basins computing full `l1_distance` and never calls
`scent()` as a pre-filter. The headline "cascade" performance claim has no implementation.

If the pre-filter were wired: `ingest` would call `candidate.scent().is_compatible(new.scent())`
first, reject mismatches without computing `l1_distance`, and achieve O(k) instead of O(n)
where k << n is the scent-compatible subset. Currently it's O(n) flat scan. To fix:

```rust
// In BasinAccumulator::ingest — proposed, not yet implemented:
for basin in &mut self.basins {
    if !basin.scent.is_compatible(candidate.scent) { continue; }  // HEEL gate
    if basin.centroid.l1_distance(&candidate) < self.threshold {   // HIP merge
        // merge
    }
}
```

---

## Refactor prescription

1. **Keep `hhtl.rs` as-is.** It is clean, minimal, provably correct. Do not merge it into anything.
2. **Split `high_heel.rs` into three files:**
   - `high_heel_container.rs` — `HighHeelBGZ`, `Heel`, `SpoBase17`, pack/unpack (~200 LOC)
   - `basin_accumulator.rs` — `BasinAccumulator`, `BasinStats`, calibrate (~170 LOC)
   - `lens_profile.rs` — `LensProfile`, `LensConfig`, `LensFamily`, `TokenizerFamily`, `LENS_REGISTRY` (~290 LOC). **Move out of contract surface eventually.**
3. **Wire `scent()` into `BasinAccumulator::ingest`** or delete the pre-filter claim from all docs.
4. **Cite or remove the plasticity thresholds** in `revise_truth`.

---

## Proposed test that proves the kernel is correct

```rust
#[test]
fn prefix_kernel_completeness() {
    // Every relation must reduce to is_ancestor_of or the shift idiom.
    let root = NiblePath::root();
    let a = root.child(3).child(7);
    let b = root.child(3).child(7).child(2);
    let c = root.child(3).child(9);

    // Containment
    assert!(a.is_ancestor_of(b));
    assert!(!b.is_ancestor_of(a));
    assert!(!a.is_ancestor_of(c)); // different last nibble at depth 2

    // Topology consequences follow
    assert!(b.is_descendant_of(a));
    assert!(a.is_sibling_of(c));   // same parent (root.child(3)), different child
    assert_eq!(a.common_ancestor(c), root.child(3));

    // LOD consequence
    assert!(!a.is_full()); // depth 2 < MAX_DEPTH
    let full = (0..NiblePath::MAX_DEPTH).fold(NiblePath::root(), |p, i| p.child(i as u8 % 16));
    assert!(full.is_full());

    // Cascade routing: truncating to any depth gives a valid ancestor
    for depth in 0..=b.depth() {
        let ancestor = NiblePath { path: b.path >> (4 * (b.depth() - depth) as u32), depth };
        assert!(ancestor.is_ancestor_of(b) || ancestor == b);
    }
}
```

This test would fail if `is_ancestor_of` had an off-by-one, if `parent()` didn't invert
`child()`, or if the nibble-shift arithmetic were wrong. It does not test the claimed cascade
performance (O(k) scent pre-filter) because that code does not exist yet.
