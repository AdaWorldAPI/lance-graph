# PR-SPRINT-13-WITNESS-CAM-PQ — WitnessIndexCamPq (real ndarray CAM-PQ wiring)

> **Status:** Spec (2026-05-16; CCA2A sprint-13 preflight PP-5 — Opus planner) — D-CSV-16
> **Scope deliverable:** D-CSV-16 — `WitnessIndexCamPq` replaces the sprint-12 HashMap placeholder (`WitnessIndexHashMap`) for distance-ranked SPO witness lookups.
> **Parent plan:** `.claude/plans/cognitive-substrate-convergence-v1.md` §11 D-CSV-6b (was: "real CAM-PQ wiring; follow-up PR after D-CSV-6a")
> **Architectural anchor:** `.claude/specs/pr-ce64-mb-4-arigraph-spo-g.md` §3.3 — original `WitnessCorpus` design + iron rule `W5-INV-CAM-PQ-INDEX`
> **Primary references:**
> - `crates/lance-graph/src/graph/arigraph/witness_corpus.rs` — current state, sprint-12 `WitnessIndexHashMap` (formerly `CamPqIndexPlaceholder` until CSI-15 rename in PR #390) — the placeholder this PR replaces.
> - `/home/user/ndarray/src/hpc/cam_pq.rs` — the canonical CAM-PQ codec (`CamCodebook`, `DistanceTables`, `PackedDatabase`, 6×256 PQ on a `total_dim`-length f32 vector; 256D recommended per W-G2 report).
> - `/home/user/ndarray/src/hpc/cam_index.rs` — `GraphHV` (3 × `Fingerprint<256>` = 49 152 bits) reference layout for the SPO triple addressed by LSH; this PR uses CAM-PQ instead of LSH for the same SPO carrier.
> - `.claude/knowledge/encoding-ecosystem.md` (P0 mandatory pre-codec read).
> - `.claude/knowledge/cam-pq-unified-pipeline.md` — CAM-PQ unified pipeline (encode/decode/distance) doctrine.
> **Depends on:** sprint-12 PR #388 (Wave G2 — `WitnessIndexHashMap` placeholder), PR #390 (CSI-15 rename).
> **Iron rules:** I-VSA-IDENTITIES (Layer-1 adapter SHALL operate on identity fingerprints, not on quantized content) · I-NOISE-FLOOR-JIRAK (CAM-PQ-induced weak dependence is the canonical source — Jirak 2016 bounds apply to recall@k claims).
> **LOC estimate:** ~400 LOC source + ~250 LOC tests = ~650 LOC total (target ceiling: 700).

---

## §0 Status, Cross-Refs, and Sprint-13 Position

| Field | Value |
|---|---|
| Sprint | 13 |
| Deliverable | D-CSV-16 — `WitnessIndexCamPq` real wiring |
| Predecessor | D-CSV-6a (`WitnessCorpus` + `WitnessIndexHashMap`) — shipped in sprint-12 Wave G PR #388 + CSI-15 rename PR #390 |
| Status | Spec draft — awaiting OQ-CSV-N ratification (see §8) |
| Iron rules | W5-INV-CAM-PQ-INDEX (newly TESTABLE at API + behaviour level, not just shape) · I-VSA-IDENTITIES · I-NOISE-FLOOR-JIRAK |
| Feature gate | `with-cam-pq` (lance-graph crate; default OFF until ndarray HPC features are universally available in consumer builds) |
| New iron rules | None — this PR makes an existing rule (W5-INV-CAM-PQ-INDEX) genuinely enforceable for the first time. |

The W5-INV-CAM-PQ-INDEX iron rule (originally from `pr-ce64-mb-4-arigraph-spo-g.md` §3.3) reads:

> **W5-INV-CAM-PQ-INDEX:** `cam_pq_index` is the canonical search structure for `WitnessCorpus`. After every `insert(entry)`, `cam_pq_index.insert(spo, WitnessId)` MUST be called and SHALL reflect the entry in the index. Eviction (`evict_stale_before`) MUST also rebuild the index for all evicted entries. Distance-ranked queries (`cam_pq_search(spo, k)`) MUST return the top-k entries by CAM-PQ ADC distance order.

Sprint-12 (D-CSV-6a) tested this **only at the API-shape level** because the index was an empty unit-struct placeholder (`CamPqIndexPlaceholder` → renamed `WitnessIndexHashMap` in CSI-15 PR #390). Sprint-13 promotes the test from shape-level to behaviour-level by wiring the real codec.

Cross-refs:
- `.claude/plans/cognitive-substrate-convergence-v1.md` §11 (D-CSV-6b)
- `.claude/specs/pr-ce64-mb-4-arigraph-spo-g.md` §3.3 (original design)
- `.claude/board/PR_ARC_INVENTORY.md` (PR #388, PR #390 entries)
- `.claude/knowledge/encoding-ecosystem.md` (P0 codec map)
- `.claude/knowledge/cam-pq-unified-pipeline.md` (codec doctrine)

---

## §1 Statement of Scope

This PR ships the real CAM-PQ-backed witness index — `WitnessIndexCamPq` — alongside the
sprint-12 `WitnessIndexHashMap`. The CAM-PQ index does **NOT** displace the HashMap: both
coexist, each serving a different query class.

| Query class | Structure used | Cost | Sprint |
|---|---|---|---|
| Exact-match by SPO u64 | `WitnessIndexHashMap` (HashMap<u64, Vec<usize>>) | O(1) amortized | sprint-12 |
| First-k chain-order by SPO u64 | `WitnessIndexHashMap` + chain scan | O(k) post-lookup | sprint-12 |
| **Distance-ranked top-k by SPO** | **`WitnessIndexCamPq`** (real CAM-PQ ADC) | **O(log N) candidate pool + O(k · 6) distance** | **sprint-13** |
| Fuzzy / nearest-neighbour by partial SPO | `WitnessIndexCamPq` | Same as above | sprint-13 |

**What this PR adds:**

1. **Layer-1 adapter — SPO u64 → 256D identity fingerprint.** Encodes the (s, p, o)
   triple of u8 palette indices into a 256D f32 vector via a VSA-bind composition of
   three role-keyed identity fingerprints. See §2.
2. **Layer-2 wrapper — `WitnessIndexCamPq`.** Wraps `ndarray::hpc::cam_pq::CamCodebook`
   + `DistanceTables` + `PackedDatabase`. Maintains parallel HashMap and CAM-PQ structures
   so all sprint-12 exact-match query paths continue to work. See §3.
3. **Layer-3 integration — `WitnessCorpus::cam_pq_search(spo, k)`** method, gated by a
   new `cam_pq_index: Option<WitnessIndexCamPq>` field. Lazy construction via
   `WitnessCorpus::enable_cam_pq(codebook)` and on first `cam_pq_search` call. See §5.
4. **Feature gate `with-cam-pq`** — keeps the placeholder-only build (no ndarray
   HPC dep) compiling cleanly. See §4.

**Out of scope for this PR:**
- Codebook training (offline; the codebook is passed in by the caller as a `CamCodebook` argument).
- Stroke-cascade query path (`PackedDatabase::cascade_query`) — this PR exposes only the
  precompute-distance / distance / top_k path. Stroke cascade can land in a follow-up.
- AVX-512 dispatch tuning — inherited from `ndarray::hpc::cam_pq` and not re-tuned here.
- `WitnessCorpusStore` 64-slot array (still belongs to MailboxSoA W-slot integration — D-CSV-7).
- Persistence of the index across process boundaries (Lance MVCC, etc.) — separate PR.

---

## §2 Layer-1 Adapter — SPO u64 → 256D Identity Fingerprint

### 2.1 The shape mismatch

`ndarray::hpc::cam_pq::CamCodebook::encode(&self, vector: &[f32]) -> CamFingerprint`
expects a `&[f32]` of length `total_dim`. The W-G2 report and the cam_pq module docstring
("170× compression for 256D vectors") recommend `total_dim = 256`. Each subspace is
`subspace_dim = 256 / 6 ≈ 42` floats (the codec accepts any `total_dim` divisible by 6;
the spec settles on **256D with 42.66 → 43-padded subspace floats**, identical to the
W-G2 W-slot test fixture).

Per `cam_index::GraphHV`, the canonical lance-graph SPO carrier is 3 × `Fingerprint<256>` =
49 152 bits = 1 536 u64s = far wider than CAM-PQ wants. Two adapter strategies:

### 2.2 Option A — VSA bind of role-keyed identity fingerprints (RECOMMENDED)

Per `I-VSA-IDENTITIES`: each of `s`, `p`, `o` palette indices selects an identity
fingerprint from a Layer-2 catalogue, and the three identities are VSA-bound by role.
Concretely:

```rust
// 1. Per-position role identity vectors (deterministic, generated once at index init):
//    R_S, R_P, R_O ∈ ℝ^256, bipolar ±1 vectors. Mutually orthogonal by construction
//    (disjoint 85-component slices: R_S in [0..85), R_P in [85..170), R_O in [170..255);
//    one slot at index 255 reserved for a NULL-or-OUT-OF-PALETTE marker).
//
// 2. Per-palette-entry content identities: identity_for(palette_idx: u8) -> Vec<f32; 256>
//    derived as a row-permutation of a hashed-bipolar pattern keyed by `palette_idx`
//    (NOT a one-hot — that defeats CAM-PQ's geometric centroid clustering by collapsing
//    256 distinct entries onto a single non-zero axis).
//
// 3. Bind = element-wise multiply:
//    v_spo = (R_S ⊙ id(s_idx)) + (R_P ⊙ id(p_idx)) + (R_O ⊙ id(o_idx))
//    The bundle (+) lives well below the JL/concentration capacity bound (N=3 ≪ √256/4 ≈ 4).
```

Why this is the right choice:

- **Lossless unbind.** `v_spo ⊙ R_S` recovers an approximation of `id(s_idx)` (up to bundle cross-talk
  ≈ √(2/d) for d=256, ≈ 8.8% expected magnitude); cosine match against the Layer-2 catalogue
  recovers `s_idx` with > 99% accuracy at N=3 superposition (Plate 1995 capacity bound; Kleyko et al. 2022 confirms for d ≥ 128).
- **Role-orthogonality preserved.** Disjoint slices ⇒ `R_S · R_P = 0` exactly, so role
  interference is zero (not just small).
- **CAM-PQ-friendly geometry.** Identity vectors are dense bipolar; subspace centroids cluster
  meaningfully (rather than collapsing onto axis-aligned one-hots, which would make 5 of the
  6 subspaces degenerate).
- **VSA doctrine compliance.** Per `I-VSA-IDENTITIES` Test-2 (role orthogonality) and Test-3
  (cleanup codebook): Layer-2 identity catalogue is the codebook; orthogonal role bipolars
  satisfy Test-2; Plate-Kleyko bound is the cleanup-codebook guarantee.

### 2.3 Option B — One-hot SPO embedding (REJECTED)

```rust
// v[s_idx] = 1.0          // 0..85
// v[85 + p_idx] = 1.0     // 85..170
// v[170 + o_idx] = 1.0    // 170..255
// v[255] = 0.0  // reserved
```

Why this is wrong:

1. **Register-loss problem (per `I-VSA-IDENTITIES` Layer-3 / "Lazy VSA check" rule).** A
   one-hot is the content register itself, not an identity pointing to it. Bundling /
   distance computation operates on the register — which means the geometric structure
   CAM-PQ codebook training would extract is empty (only 256 non-zero locations in the
   entire input distribution; 254 of every input vector are exactly zero). Subspace
   k-means degenerates: 5 of the 6 subspaces (Branch / TwigA / TwigB / Leaf / Gamma)
   have zero variance across the dataset for any single SPO except where a palette
   index falls in their slice. Codebook quality is near-zero.
2. **No nearest-neighbour structure for partial queries.** With a one-hot, `cos(v_A, v_B)`
   for two different SPO triples is either 0 (no overlap), 1/3 (one of s/p/o shared),
   2/3, or 1. Only 4 distinct distance values exist; top-k ranking is degenerate.
3. **Violates I-VSA-IDENTITIES Test-0 (register-laziness check).** "Does this thing
   have a natural name/ID/enum? Yes ⇒ use the register" — a one-hot reaches for a VSA-
   shaped tool for a HashMap-shaped problem. The HashMap already exists
   (`WitnessIndexHashMap`); the CAM-PQ index exists for *geometric distance ranking*,
   which one-hot destroys.

### 2.4 Decision

**Recommendation: Option A.** This is what `OQ-CSV-N` (§8) asks the user to ratify
before sprint-13 spawn.

The adapter API:

```rust
/// SPO u64 -> 256D f32 identity fingerprint (Option A).
///
/// `R_S`, `R_P`, `R_O` are generated deterministically from a seed (default seed = 0xCAFE_BABE)
/// once per `WitnessIndexCamPq` lifetime; `id_for(palette_idx)` is a per-entry catalogue
/// also generated deterministically. Both live in the index struct, not as global statics
/// (one index per `WitnessCorpus`, no cross-corpus role-key sharing required).
pub fn spo_to_fingerprint(
    spo: u64,
    role_s: &[f32; 256],
    role_p: &[f32; 256],
    role_o: &[f32; 256],
    palette_id: &[[f32; 256]; 256],   // 256 entries × 256 floats = 256 KB lookup table per index
) -> [f32; 256] {
    let s_idx = ((spo >> 0)  & 0xFF) as usize;
    let p_idx = ((spo >> 8)  & 0xFF) as usize;
    let o_idx = ((spo >> 16) & 0xFF) as usize;
    let mut v = [0f32; 256];
    for i in 0..256 {
        v[i] = role_s[i] * palette_id[s_idx][i]
             + role_p[i] * palette_id[p_idx][i]
             + role_o[i] * palette_id[o_idx][i];
    }
    v
}
```

Memory footprint per index: 3 × 256 × 4 = 3 KB roles + 256 × 256 × 4 = 256 KB palette
catalogue = ~259 KB constant overhead per `WitnessIndexCamPq`. Acceptable.

---

## §3 `WitnessIndexCamPq` Struct and Lifecycle

### 3.1 Struct definition

```rust
// crates/lance-graph/src/graph/arigraph/witness_corpus.rs (additions)

#[cfg(feature = "with-cam-pq")]
use ndarray::hpc::cam_pq::{CamCodebook, CamFingerprint, DistanceTables};

/// CAM-PQ-backed witness index.
///
/// Wraps `ndarray::hpc::cam_pq::CamCodebook` for distance-ranked top-k queries.
/// Maintains parallel HashMap structures (`spo_to_position`, `position_to_spo`)
/// so all sprint-12 query paths continue to function unchanged.
///
/// Iron rule: W5-INV-CAM-PQ-INDEX — every insert/evict on `WitnessCorpus`
/// MUST be mirrored here when `cam_pq_index.is_some()`.
#[cfg(feature = "with-cam-pq")]
#[derive(Clone, Debug)]
pub struct WitnessIndexCamPq {
    /// The trained codebook (passed in by caller; trained offline against the corpus).
    codec: CamCodebook,

    /// 256D role-key identity fingerprints (R_S, R_P, R_O). Generated deterministically
    /// once at construction time. Stored unboxed for cache-locality during encode.
    role_s: [f32; 256],
    role_p: [f32; 256],
    role_o: [f32; 256],

    /// 256-entry palette identity catalogue. Generated deterministically at construction.
    /// `palette_id[i]` is the identity fingerprint for palette index `i`.
    palette_id: Box<[[f32; 256]; 256]>,

    /// Pre-encoded CAM-PQ fingerprints, parallel to `position_to_spo`.
    cam_fps: Vec<CamFingerprint>,

    /// SPO u64 -> entry positions (Vec because multiple witnesses can share an SPO).
    /// Mirrors the sprint-12 `WitnessIndexHashMap` semantics for back-compat.
    spo_to_position: std::collections::HashMap<u64, Vec<usize>>,

    /// Position -> SPO u64. Needed during eviction to look up which CAM-PQ entry to remove.
    /// Indices match `WitnessCorpus::entries` slice order at encode time.
    position_to_spo: Vec<u64>,
}
```

### 3.2 Lifecycle / API

```rust
#[cfg(feature = "with-cam-pq")]
impl WitnessIndexCamPq {
    /// Construct an empty index with the given pre-trained codebook.
    /// Role-keys and palette catalogue are generated deterministically from `seed`.
    pub fn new(codec: CamCodebook, seed: u64) -> Self { /* ... */ }

    /// Insert a single (spo, position) pair. Mirrors `WitnessIndexHashMap::insert`.
    /// Encodes the SPO via the §2 adapter, stores the CAM fingerprint, updates the maps.
    pub fn insert(&mut self, spo: u64, position: usize) { /* ... */ }

    /// Exact-match lookup (back-compat with the HashMap path).
    /// Returns positions in insertion order; identical semantics to the HashMap version.
    pub fn lookup_exact(&self, spo: u64) -> &[usize] { /* ... */ }

    /// CAM-PQ distance-ranked top-k lookup. NEW capability vs the HashMap.
    /// 1. Encode query SPO -> 256D fp.
    /// 2. `codec.precompute_distances(&fp)` -> DistanceTables.
    /// 3. `DistanceTables::top_k(&self.cam_fps, k)` -> Vec<(idx, dist)>.
    /// 4. Map idx -> position via `position_to_spo`-aligned indices.
    pub fn cam_pq_search(&self, query_spo: u64, k: usize) -> Vec<(usize, f32)> { /* ... */ }

    /// Rebuild after a `WitnessCorpus::evict_stale_before` removes entries.
    /// `keep` is the surviving slice; reusing the existing fingerprints by gather is
    /// cheaper than re-encoding (avoids the 6×256 k-means lookup per surviving entry).
    pub fn rebuild_after_evict(&mut self, keep: &[(u64, usize)]) { /* ... */ }

    pub fn len(&self) -> usize { self.cam_fps.len() }
    pub fn is_empty(&self) -> bool { self.cam_fps.is_empty() }
}
```

### 3.3 Encoding determinism

`spo_to_fingerprint` is a pure function of the SPO bits and the three role-keys + palette
catalogue. Same SPO ⇒ same fingerprint ⇒ same CAM-PQ encoding (modulo codebook). This
matters for the eviction-rebuild test (T9) and for reproducible benchmarks.

---

## §4 Feature Gate `with-cam-pq`

### 4.1 Cargo.toml changes

```toml
# crates/lance-graph/Cargo.toml — additions

[features]
default = ["unity-catalog", "delta", "ndarray-hpc"]
# ... existing features ...
with-cam-pq = ["ndarray-hpc"]   # implies ndarray-hpc; cannot be enabled without it.

[dependencies]
# ndarray is already optional via ndarray-hpc; no new dep entry needed.
```

### 4.2 Source-level gating

- `WitnessIndexCamPq` struct and impl block: `#[cfg(feature = "with-cam-pq")]`.
- `WitnessCorpus::cam_pq_index: Option<WitnessIndexCamPq>` field: also gated. When
  the feature is OFF, the field is omitted entirely (not just `None`-typed) so the
  struct memory layout shrinks for the minimal build.
- `WitnessCorpus::enable_cam_pq` / `cam_pq_search` methods: gated.

### 4.3 Two-build invariant

Both `cargo build -p lance-graph` (default, `with-cam-pq` ON via `ndarray-hpc`) and
`cargo build -p lance-graph --no-default-features` (placeholder-only, `with-cam-pq` OFF)
MUST compile cleanly. This is enforced by tests T11 (ON build) and T12 (OFF build) in §6.

---

## §5 `WitnessCorpus` Integration

### 5.1 Field addition

```rust
#[derive(Clone, Debug)]
pub struct WitnessCorpus {
    entries: Arc<Vec<WitnessEntry>>,

    /// Sprint-12 placeholder, still present as the exact-match backbone.
    pub witness_index: WitnessIndexHashMap,

    /// Sprint-13 NEW: real CAM-PQ index for distance-ranked queries.
    /// Lazy: None until `enable_cam_pq(codec)` is called or first `cam_pq_search`.
    #[cfg(feature = "with-cam-pq")]
    pub cam_pq_index: Option<WitnessIndexCamPq>,
}
```

### 5.2 Lazy construction policy

```rust
#[cfg(feature = "with-cam-pq")]
impl WitnessCorpus {
    /// Enable CAM-PQ indexing with a caller-supplied trained codebook.
    /// Backfills all existing entries through the encoder. O(N · 256 · 6) one-time cost.
    pub fn enable_cam_pq(&mut self, codec: CamCodebook, seed: u64) {
        let mut idx = WitnessIndexCamPq::new(codec, seed);
        for (pos, e) in self.entries.iter().enumerate() {
            idx.insert(e.spo, pos);
        }
        self.cam_pq_index = Some(idx);
    }

    /// Distance-ranked top-k search. Requires CAM-PQ enabled; returns empty if not.
    pub fn cam_pq_search(&self, query_spo: u64, k: usize) -> Vec<(WitnessId, f32)> {
        let Some(idx) = self.cam_pq_index.as_ref() else { return vec![]; };
        idx.cam_pq_search(query_spo, k)
            .into_iter()
            .map(|(pos, d)| (self.entries[pos].witness_id(), d))
            .collect()
    }
}
```

### 5.3 Insert/Evict mirroring (the iron-rule machinery)

```rust
// pseudo-diff against current witness_corpus.rs
pub fn insert(&mut self, entry: WitnessEntry) -> WitnessId {
    // ... existing sorted-insert logic ...
    let pos = /* ... binary search position ... */;
    entries.insert(pos, entry);

    // W5-INV-CAM-PQ-INDEX: mirror to both indices.
    self.witness_index.insert(entry.spo, pos);   // existing HashMap path
    #[cfg(feature = "with-cam-pq")]
    if let Some(idx) = self.cam_pq_index.as_mut() {
        idx.insert(entry.spo, pos);              // NEW CAM-PQ path
    }
    id
}

pub fn evict_stale_before(&mut self, cutoff_ns: u64) -> usize {
    let entries = Arc::make_mut(&mut self.entries);
    let n_before = entries.len();
    entries.retain(|e| e.timestamp_ns >= cutoff_ns);

    // Rebuild both indices against the surviving slice.
    self.witness_index.rebuild_from(entries);
    #[cfg(feature = "with-cam-pq")]
    if let Some(idx) = self.cam_pq_index.as_mut() {
        let keep: Vec<_> = entries.iter().enumerate().map(|(p, e)| (e.spo, p)).collect();
        idx.rebuild_after_evict(&keep);
    }
    n_before - entries.len()
}
```

### 5.4 Position shifts and the rebuild contract

Sprint-12's `WitnessIndexHashMap` had to deal with `binary_search_by` shifting positions on
each insert. Sprint-13 inherits the same constraint for `WitnessIndexCamPq`: positions in
`cam_fps` and `position_to_spo` MUST match `WitnessCorpus::entries` indices at all times.

Two acceptable implementations:

1. **Cheap shift-on-insert.** On `insert(spo, pos)`, shift every entry in `position_to_spo`
   at `>= pos` by +1, then write `pos`. O(N) per insert but constant-factor cheap.
2. **Append-and-relabel.** Always append to `cam_fps` (no shifts), but maintain a
   `cam_idx -> entries_pos` mapping that gets resorted lazily.

This spec recommends option 1 for simplicity; T6 verifies that out-of-order insertions
keep both indices coherent.

---

## §6 Test Plan — 12 Tests

All tests live in `crates/lance-graph/src/graph/arigraph/witness_corpus.rs#[cfg(test)] mod tests`.
Tests T1-T6 + T11 require `--features with-cam-pq`; T12 specifically requires `--no-default-features`.

| # | Test | What it asserts | Iron rule |
|---|---|---|---|
| T1 | `spo_to_fingerprint_option_a_roundtrip` | Encode 3 distinct SPO triples; unbind via `⊙ R_S` and cosine-match the catalogue. Top-1 catalogue hit recovers `s_idx` with cosine > 0.30 (well above bundle cross-talk floor √(2/256) ≈ 0.088). | I-VSA-IDENTITIES |
| T2 | `cam_pq_codec_insert_roundtrip` | Insert 100 distinct SPO entries; `cam_fps.len() == 100`; `position_to_spo[i]` matches insertion order. | W5-INV-CAM-PQ-INDEX |
| T3 | `cam_pq_search_top_k_matches_adc_distance_order` | Insert 50 entries; query with one of them; `cam_pq_search(query_spo, 5)` returns the exact-match entry as `top[0]` (distance 0.0 modulo quantization). | W5-INV-CAM-PQ-INDEX |
| T4 | `cam_pq_search_partial_query_returns_neighbours` | Insert entries with SPO `(s, p, *)` for fixed `s, p` and varying `o`; query with a never-inserted `o`; top-k contains near-`o` entries. | W5-INV-CAM-PQ-INDEX |
| T5 | `back_compat_with_witness_index_hashmap` | After CAM-PQ enable, `witness_index.lookup(spo)` continues to return positions in insertion order (sprint-12 path unchanged). | (regression) |
| T6 | `out_of_order_insert_keeps_both_indices_coherent` | Insert (ts=200, ts=100, ts=150); for each entry assert `entries[position_to_spo_idx[i]].spo == cam_fps[i].decoded_spo`. | W5-INV-CAM-PQ-INDEX |
| T7 | `lazy_construction_starts_empty` | `WitnessCorpus::new(); corpus.cam_pq_index.is_none() == true`; calling `cam_pq_search` returns empty Vec without panic. | (lifecycle) |
| T8 | `enable_cam_pq_backfills_existing_entries` | Insert 10 entries; `enable_cam_pq(codec, seed=0)`; assert `cam_pq_index.unwrap().len() == 10`. | (lifecycle) |
| T9 | `eviction_rebuilds_cam_pq_index` | Insert 5 entries (ts=100..500); `evict_stale_before(300)` removes 2; assert `cam_pq_index.len() == 3` and `cam_pq_search` matches new positions. | W5-INV-CAM-PQ-INDEX |
| T10 | `option_a_role_orthogonality` | `dot(R_S, R_P) == 0.0`, `dot(R_S, R_O) == 0.0`, `dot(R_P, R_O) == 0.0` (disjoint slices guarantee exact zero). | I-VSA-IDENTITIES |
| T11 | `build_with_with_cam_pq_feature_on` | Compile-time guard: file must compile with `--features with-cam-pq`; one runtime smoke-test (`WitnessIndexCamPq::new` runs). | (feature gate) |
| T12 | `build_without_with_cam_pq_feature_off` | Compile-time guard: file must compile with `--no-default-features` (no `cam_pq_index` field, no `WitnessIndexCamPq` symbol referenced). One smoke-test: `WitnessCorpus::new()` and `query(spo)` still work. | (feature gate) |

### 6.1 Test fixtures

A tiny pre-trained `CamCodebook` is needed for T2-T9. The spec recommends a deterministic
random-init codebook (one k-means iteration on 256 synthetic vectors) generated in a
`test_helpers::tiny_codebook()` function — under 1 KB on the stack, no fixture files.

### 6.2 Bench (optional, deferred)

A bench mirroring T-WP-2 of `pr-ce64-mb-4-arigraph-spo-g.md` §T8 — 10K entries, p99
< 50 µs for `cam_pq_search` — is **scheduled for D-CSV-16-BENCH** in sprint-14, not
included in this PR's `~650 LOC` budget.

---

## §7 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **R1: ndarray dep size in default build** | Medium | Low | `with-cam-pq` defaults to ON only via the `ndarray-hpc` chain; downstream consumers that don't want ndarray can already disable `ndarray-hpc`. No new dep added. |
| **R2: CAM-PQ codec shape incompatibility** | Low | High | Spec settles on 256D in §2. T1 asserts the §2 adapter produces a usable input shape. If W-G2's report is misremembered and the codec actually wants different `total_dim`, the spec change is one constant in `spo_to_fingerprint`. |
| **R3: Lazy-vs-eager construction tradeoff** | Medium | Medium | Spec chooses lazy (Option A in §5.2). The cost is one explicit `enable_cam_pq(codec)` call from callers. Eager would force every corpus to pay the encode cost even when only HashMap queries are used. T7+T8 lock the lazy contract. |
| **R4: Position-shift bug between HashMap and CAM-PQ indices** | High | High | T6 explicitly tests out-of-order insertion coherence between both indices. The shift-on-insert path (§5.4 option 1) is the safer choice. |
| **R5: Option A bundle cross-talk breaks recall@k** | Low | Medium | N=3 bundle at d=256 is well below √d/4 ≈ 4 capacity (Plate-Kleyko bound). T1 measures the actual cosine recovery margin; if recall@1 ever drops below 0.95 in T3-T4, escalate to Plate's contextual VSA dimension d=1024 (4× memory; acceptable per cam_pq doc: 256D is the lower bound, not a ceiling). |
| **R6: Codebook training is out-of-scope but assumed** | Low | Low | Spec explicitly says callers supply a trained codebook. T-fixture uses a degenerate random-init codebook good enough for shape/order assertions (T1-T9 don't depend on reconstruction quality, only on encode-determinism + distance-ordering monotonicity). Production training is a follow-up PR (D-CSV-16-TRAIN). |
| **R7: I-NOISE-FLOOR-JIRAK applies to recall@k claims** | Medium | Low | This PR makes no statistical-significance claims; the bench (D-CSV-16-BENCH) will cite Jirak 2016 if/when a recall threshold needs justification. |

---

## §8 OQ-CSV-N — Open Question (ratify before sprint-13 spawn)

**OQ-CSV-N (NEW):** Adapter strategy — Option A (VSA bind, recommended) vs Option B
(one-hot, rejected) for the SPO u64 → 256D fingerprint layer.

**Opus recommendation: Option A.** Per §2.2 and §2.3 above. The rejection of Option B
is grounded in `I-VSA-IDENTITIES` doctrine — specifically Test-0 (register-laziness)
and the Layer-3 register-loss problem.

**Ratification ask:** the user is asked to confirm Option A before sprint-13 implementation
spawn. If Option B is preferred (e.g., for ndarray code-simplicity reasons), then this spec's
§2.2 / §3 / §6.T1 / §6.T10 require revision and `I-VSA-IDENTITIES` requires an
amendment-or-exception entry on the `pr-ce64-mb-4-arigraph-spo-g.md` invariant list.

Secondary ratification points (lower-stakes, can be deferred to sprint-13 spawn turn):

- **OQ-CSV-N.1:** seed for role-key / palette catalogue generation. Default `0xCAFE_BABE`
  unless persistence-across-restart matters (no consumers identified today).
- **OQ-CSV-N.2:** whether `enable_cam_pq` should be implicit on first `cam_pq_search` call
  (current spec: explicit only, returns empty Vec if not enabled). Risk: silent surprise
  if the caller forgot to enable.
- **OQ-CSV-N.3:** subspace `total_dim`. Spec says 256; W-G2 confirms; user can override.

---

## §9 LOC Estimate

| Area | LOC (source) | LOC (tests) |
|---|---|---|
| `WitnessIndexCamPq` struct + impl | ~180 | — |
| `spo_to_fingerprint` adapter (§2) | ~50 | — |
| Role-key + palette catalogue generators (deterministic seeded) | ~40 | — |
| `WitnessCorpus::enable_cam_pq` + `cam_pq_search` (§5) | ~40 | — |
| Insert/evict mirroring (§5.3) | ~30 | — |
| Cargo.toml feature gate + `#[cfg]` annotations | ~10 | — |
| `tiny_codebook` test fixture (§6.1) | ~25 | — |
| T1 (option-A roundtrip) | — | ~25 |
| T2 (codec insert) | — | ~20 |
| T3 (top-k matches ADC) | — | ~25 |
| T4 (partial-query neighbours) | — | ~25 |
| T5 (HashMap back-compat) | — | ~20 |
| T6 (out-of-order insert coherence) | — | ~25 |
| T7 (lazy construction) | — | ~15 |
| T8 (enable backfills) | — | ~20 |
| T9 (eviction rebuilds) | — | ~25 |
| T10 (role orthogonality) | — | ~10 |
| T11 (feature ON build smoke) | — | ~15 |
| T12 (feature OFF build smoke) | — | ~15 |
| **Total** | **~375 LOC** | **~240 LOC** |
| **Grand total** | | **~615 LOC** |

Comfortably under the 700-LOC ceiling. The next sprint (sprint-14) can add the bench
(T-WP-2 mirror), stroke-cascade fast-path, and codebook training without crowding this PR.

---

## §10 Acceptance Checklist

- [ ] §2 adapter (Option A) ratified by user (OQ-CSV-N).
- [ ] `WitnessIndexCamPq` compiles under `--features with-cam-pq`.
- [ ] `WitnessCorpus` compiles under `--no-default-features` (no CAM-PQ field).
- [ ] All 12 tests pass under their respective feature flags.
- [ ] `LATEST_STATE.md` Contract Inventory row added for `WitnessIndexCamPq`.
- [ ] `PR_ARC_INVENTORY.md` entry prepended (Added: WitnessIndexCamPq + spo_to_fingerprint adapter; Locked: HashMap path unchanged; Deferred: bench, stroke-cascade, training).
- [ ] `STATUS_BOARD.md` D-CSV-16 row updated to "In PR" then "Shipped".
- [ ] Spec patch trail in `pr-ce64-mb-4-arigraph-spo-g.md`: §3.3 W5-INV-CAM-PQ-INDEX is now genuinely testable (cross-ref this PR).
- [ ] `EPIPHANIES.md` entry (if any new finding emerges during impl — e.g., Option A bundle cross-talk margins in T1).

---

## §11 Sprint-13 Spawn Notes

- **Agent assignment.** This PR is a `family-codec-smith` + `truth-architect` co-spawn.
  `family-codec-smith` owns the §2 adapter, the role-key construction, and the codec
  wiring. `truth-architect` owns the W5-INV-CAM-PQ-INDEX iron-rule enforcement and the
  feature-gate guard rails (T11/T12).
- **Knowledge bootload (Tier-1).** `encoding-ecosystem.md` (P0), `cam-pq-unified-pipeline.md`,
  `vsa-switchboard-architecture.md` (for I-VSA-IDENTITIES doctrine reference).
- **Mandatory pre-impl reads.**
  - `crates/lance-graph/src/graph/arigraph/witness_corpus.rs` (sprint-12 state).
  - `/home/user/ndarray/src/hpc/cam_pq.rs` lines 109-209 (CamCodebook + DistanceTables API).
  - This spec, sections §2-§5.
- **Parallel-work cap.** This PR can land in parallel with any sprint-13 PR that does
  NOT touch `witness_corpus.rs`. Conflicts likely with: future `WitnessCorpusStore`
  (MailboxSoA W-slot) PR — coordinate via STATUS_BOARD.
