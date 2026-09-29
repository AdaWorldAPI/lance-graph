# deepnsm-v2 Cam96 → 6 × pairwise distribution — spec (v1)

> **Status:** SPEC v1, PROPOSAL. No code authorized. The change alters the
> meaning of 12 bytes that a contract layout already persists (§6.3), retrains
> an artifact, and moves a codec into the contract, so it is council-grade:
> run `/5plus3` on this spec before any implementation wave.
> **D-ids:** `D-C96P-1..8` (§9).
> **Written against:** lance-graph `main` `5282dfa3`.

## 0. The one-sentence change

Cam96 today is **12 independent axis codebooks**; each rail's two bytes are
two unrelated halves of a 16-d subspace, quantized apart and summed apart.
It becomes **6 subspaces × one 256-centroid codebook each, with every rail a
`(c₁:c₂)` pair of centroids of that one codebook, and the relation between
them read from a per-subspace Fisher-z table.** One byte finds a point; the
pair carries the spread.

## 1. Sources, read in full before this spec

| source | what it decides |
|---|---|
| `.claude/plans/deepnsm-morton-comma-facet-v1.md` §0, §1, §2, §5 | a word = `classid(4B) = norm(prefix, frequency, PoS)` · `payload(12B) = 6 × (FisherZ:FisherZ)`; frequency is a HEADER, never a distance axis; distance is analytic Fisher-z, palette256 is the cache |
| `.claude/board/entries/2026-08-26-e-palette256-is-a-needle-the-colon-is-the-distribution-1.md` | "palette256 can find a point like a needle in a haystack. For distribution it needs pairwise." · "`6×(u8:u8)` is six *relations*, not twelve indices" · the pair's carrier is the Fisher-z k×k LUT · self-similarity is 1.0 **by address**, never a lookup · gamma is fitted on off-diagonals only |
| `.claude/v3/soa_layout/le-contract.md` §3 L4 (`:59`), §"canonical cosine/centroid replacement is ANALYTIC" (`:169-186`) | L4 = `6 × (8:8)` palette256², "each byte pair indexes the 256×256 palette distance/compose tables"; the Fisher-z codec is canon and a k×k table is a cache of the formula |
| `EPIPHANIES-ARCHIVE-2026-09-20.md` `E-CAM96-DISTRIBUTION-MEASURED-1` (`:21058`) and `E-CAM96-REVIEW-CORRECTIONS-1` (`:21036`) | frequency ⟂ meaning (ρ ≈ −0.07 vs Jina); held-out 96-bit 0.766 vs 48-bit 0.624; the `axis_dim` defect and its strong test |
| `crates/deepnsm-v2/probes/README.md` §4 | KJV held-out: 48-bit POINT 0.617 · **96-bit POINT (RQ) 0.786** · 96-bit 12-axis 0.774 — "the 96-vs-48 gain is therefore mostly BUDGET"; the distribution's justification is the facet algebra, which residual coding breaks |
| `crates/deepnsm-v2/probes/fidelity_48_vs_96.py:7`, `train_codebook.py:22-29, 57, 77-95` | how the shipped code was made: "each 16-d subspace split into a 256:256 pair" = 12 × 8-d PQ |
| `crates/deepnsm-v2/src/space.rs`, `codebook.rs`, `basin.rs`, `lib.rs`, `vocab.rs`; `crates/deepnsm-v2/data/README.md` | the shipped code, its loader, its consumers, and where the artifacts live |
| `crates/bgz-tensor/src/fisher_z.rs` | `FamilyGamma` (8 B) + `FisherZTable` (k×k i8), the certified codec |
| `crates/lance-graph-contract/src/distance.rs`, `recipe_substrate.rs:76-156` | the contract's Fisher-z helpers (`similarity_z`, `fisher_z_inverse`, `mean_similarity_fisher`) and `PairPalette` |

## 2. What ships today (verified in code, not from the board)

- `pub type Cam96 = [u8; 12]` (`space.rs:163`); `Cam96Space` holds 12 axis
  codebooks and panics unless `axes.len() == 12` (`space.rs:182-196`).
- `encode` quantizes each axis's own consecutive chunk against that axis's own
  centroids (`space.rs:230-251`). Rail `r`'s two bytes are nearest centroids of
  **two different 8-d halves**; neither byte knows the other exists.
- `distance` = `Σ_k ‖axis_k[a_k] − axis_k[b_k]‖²` over 12 axes (`:257-272`). No
  pairing, no Fisher-z, no cosine. `rails()` (`:287-289`) is a view that no
  distance ever reads.
- The artifact format enforces the same shape: `CAM96CB1`, `n_axes == 12`
  (`codebook.rs:12-14, 58`); the trained file is 12 × 256 × 8-d.
- Consequence, in the needle entry's own terms: this is **twelve indices**, a
  finer point, not six relations. Its 0.774 sits below the equal-budget RQ
  point (0.786), which is what one expects of a finer point that is not a
  better one.
- **Frequency also reaches a distance path.** `SemanticSpace::similarity`
  scores words at their `(basin, identity)` address from `PaletteVocab::pair`
  (`space.rs:74-80`), and `vocab.rs:14-21` makes that basin byte the
  frequency-rank band. That is the facet plan's `PF=Payload` defection. It is
  exported (`lib.rs:79`) and `Spo::pairs()` documents it as its consumer
  (`spo.rs:51`).
- **The header does not exist.** No `classid` is minted for a word; routing is
  the frequency-ranked `WordId` (`vocab.rs:23-26`), PoS lives in `fsm::Pos` and
  `lexical::PosCode`, counts in `LexicalEvidence` (`data/README.md:19-26`).

## 3. The target shape — the committed reading

### 3.1 The rail

For each of the 6 subspaces `s` (16-d each, the 48-bit code's split), train
**one** codebook `C_s` of 256 centroids. A word's subvector `v_s` encodes to

```
rail_s = (c₁ : c₂)   c₁ = argmin_c ‖v_s − C_s[c]‖,   c₂ = second-nearest in the SAME C_s
```

- **Each byte alone is a valid needle** into the same codebook: `c₁` is exactly
  the 48-bit point code. This keeps the independent addressability that
  `probes/README.md` §4 names as the distribution's reason to exist, and that
  residual coding breaks.
- **The colon is the relation.** `(c₁, c₂)` is a cell of `C_s`'s own k×k
  Fisher-z table; its value, `cos(C_s[c₁], C_s[c₂])`, is the spread of the
  word's neighbourhood in that subspace. That is what "L4: each byte pair
  indexes the 256×256 palette distance table" reads as for one word.
- **The diagonal is reserved for a point.** `c₁ == c₂` never arises from
  2-NN, so it is free to mean "no second pole": a word whose second-nearest is
  far is written `(c₁ : c₁)` and reads as a needle with zero spread. The
  far-threshold is a pre-registered variant, not a default (§7, G3b).

### 3.2 Distance between two words

Per subspace, the relation between two rails comes only from `T_s`, the
Fisher-z table of `C_s`. Three candidates, chosen by measurement (G1), not by
taste:

| id | per-rail similarity | reads | what it is |
|---|---|---|---|
| R1 | `z̄ = mean_z(T(a₁,b₁), T(a₁,b₂), T(a₂,b₁), T(a₂,b₂))` | 4 | both poles against both poles, averaged in z-space |
| R2 | `z̄ = ½·(z(a₁,b₁) + mean_z(T(a₁,b₂), T(a₂,b₁)))` | 3 | the needles weighted as heavily as the cross terms |
| R3 | `T(a₁,b₁)` only | 1 | the 48-bit point, as a control |

Word similarity = `tanh(mean over 6 rails of z̄_s)`. Averaging in z-space is
the contract's `mean_similarity_fisher` (`distance.rs:62-69`); no averaging
of raw cosines anywhere.

A cell `(a, a)` is answered as `1.0` by address and never read from `T_s`
(needle entry, "self-similarity is 1.0 by definition of the address").

**Stated conflict:** le-contract L4 says "similarity = ONE table read".
R1/R2 read 3–4 cells per rail, and R3, the one-read form, is the needle. Only
R3 matches the one-read wording. If R1 or R2 wins G1, the L4 row needs an
amendment. This spec does not make it.

### 3.3 Where the Fisher-z codec lives

`deepnsm-v2` depends on `lance-graph-contract` only, and `FamilyGamma` /
`FisherZTable` live in `bgz-tensor`. **Decision proposed:** move the analytic
codec into the contract. It is about 40 zero-dep lines, and le-contract already
names it as canon for L4. `bgz-tensor` would then consume it rather than own it.
The rejected alternatives:

- a deepnsm-v2 copy (a reimplementation);
- a `bgz-tensor` dependency (ndarray + holograph weight on a crate that is
  zero-dep by design).

This is a contract change and needs `v3-envelope-auditor` review.

### 3.4 Readings rejected, and why

- **B — `(z_location : z_spread)` as two quantized scalars.** Literal to the
  plan's `FisherZ:FisherZ`, but a cosine to an anchor loses *which* direction:
  two opposite words can share a location byte. It keeps no needle, so it fails
  the independent-addressability requirement.
- **C — RQ `(coarse : residual)`**, which is also `deepnsm-morton-comma-facet-v1`
  §3b's `(verb-atom : residual)`. It is the best raw point (0.786), but its
  second byte is meaningless without the first, which `probes/README.md` §4
  names as the thing the facet algebra forbids. §3b is a v2 hook gated on the
  144-verb basis. It is not in scope and not adopted here.
- **D — keep 12 axes, re-document them as pairs.** The rails would still be
  unrelated halves, so this is renaming, not the change.

## 4. The header (frequency, PoS)

- **Invariant kept:** frequency and PoS never enter the 12-byte payload or any
  word-to-word distance. The trained codes already satisfy this: they come
  from embeddings, and frequency only indexes them. `SemanticSpace` breaks it,
  so D-C96P-6 retires it from the meaning path.
- **OPEN — the classid conflict.** The plan puts `norm(prefix, frequency, PoS)`
  in `classid`. The classid canon (hi u16 = concept minted in `ogar-vocab`,
  lo u16 = app prefix, never a shape ordinal) makes classid a class, not
  per-word data. This spec does not resolve that. **Interim:**
  - frequency stays in the routing `WordId`;
  - PoS and counts stay in `LexicalEvidence`;
  - both remain outside the payload, which is the invariant the plan's
    measurement (`E-DEEPNSM-FACET-BLIND-CONVERGENCE-1`) actually confirmed.
  Resolving where the header lives (a minted word concept plus the key
  facet's routing tiers, or something else) is its own decision.

## 5. The producer

- **The codebook needs the 96-d embeddings, and they are not in the release.**
  `v0.1.0-cam96-data` carries the 12 × 8-d codebook, the codes and
  `bible_vocab.txt` (`data/README.md:9-13`). `bible_vocab_emb96.npy`, which
  `train_codebook.py:3` reads, was never published.
- **Options:**
  - (a) Re-embed through `probes/embed_bible_vocab.py` with a key the
    operator supplies through the environment. No credential may enter the
    repo, a commit or a brief.
  - (b) Reconstruct each word from its current 12-axis code and fit on that.
    This quantizes a quantization and is **allowed only as a probe arm**,
    never as the shipped artifact.
- **The trainer moves to Rust** (`examples/train_cam96p.rs`). It reads a
  little-endian `f32` embeddings blob and runs:
  - k-means-256 per 16-d subspace;
  - the 2-NN encode;
  - the per-subspace `FamilyGamma` fit on off-diagonals.
  It keeps the seed `0x9E3779B9` and the train/eval split of
  `train_codebook.py` so G1 compares against 0.774 / 0.786 / 0.617 on the
  same words. Embedding stays a lab script, because it needs the network; the
  file it produces is data, not code.

## 6. Format, types, and the version gate

### 6.1 Artifacts
- `CAM96P01` codebook, laid out in this order:
  1. `u32` n_sub (6);
  2. `u32` dim (16);
  3. `u32` k (≤256);
  4. the `6 × k × dim` f32 centroids, subspace-major;
  5. `6 × FamilyGamma` (48 B — the "analytic 48 B γ" of the facet plan §5).

  An optional materialized `6 × k × k` i8 table section is a cache and is
  never required.
- `CAM96PW1` codes: the same 12 bytes per word, with different meaning.
- **New magics are mandatory.** A `CAM96CB1` blob must be refused by the new
  loader, and the old loader must refuse the new blob.

### 6.2 Types
- The new code is a newtype, `Cam96Pair([u8; 12])`, not the `Cam96` alias, so
  a CB1 code cannot be scored in a pair space by accident. Mixing them is a
  compile error, not a runtime surprise. Old rows stay readable; new ones
  cannot be minted in the old shape (I-LEGACY-API-FEATURE-GATED).
- `Cam96Space` becomes `Cam96AxisSpace`, deprecated. It still loads the
  published v0.1.0 artifacts for reading and comparison, and nothing new is
  built on it.

### 6.3 The persisted 12 bytes
`lance-graph-contract::episodic_basin::BasinRow.self_code` (`episodic_basin.rs:78-97`)
and `canonical_node.rs:1062` store "the basin's own Cam96 centroid" as 12 raw
bytes. Once codes change meaning, a stored self-code is ambiguous without a
marker. **Required before any pair-coded self-code is written:**
- a read-mode / version marker on the `EpisodicBasin` lane;
- `v3-envelope-auditor` sign-off.

Until then no pair-coded self-code is persisted.

## 7. Pre-registered gates (fill with measured numbers only)

All on the §5 held-out protocol (train 10,000 / eval 2,543 KJV words). Every
guard below gets a disable run: red with the guard removed, green with it back.

| gate | criterion | what fails it |
|---|---|---|
| **G1 fidelity** | best of R1/R2 has held-out ρ ≥ **0.774** (the 12-axis) | < 0.774 and ≥ 0.617 means PARTIAL: report it; the operator decides. < 0.617 (the 48-bit point) means KILL |
| G1 report | also report R3, the RQ point (0.786), and recon MSE | — |
| **G2 needle** | with the same seeds and split, every word's `c₁` bytes equal its 48-bit PQ code byte for byte; R3's ρ is reported beside the 48-bit ρ | `c₁` is not the point code (R3 scores through `T_s` cosines, not full-vector L2, so its ρ is reported, not gated) |
| **G3 spread is load-bearing** | shuffling every word's `c₂` across words (keeping `c₁`) **lowers** R1/R2 ρ | if ρ does not drop, the second byte is decoration and the change is only a relabel |
| G3b diagonal variant | report ρ with the far-second-pole → `(c₁:c₁)` rule at 3 pre-set thresholds | report only |
| **G4 header orthogonality** | word similarity is bit-identical under a permutation of `WordId` routing and of `LexicalEvidence` counts | frequency or PoS leaking into meaning |
| **G5 diagonal by address** | `sim(a, a) == 1.0` for every code, and no `T_s` diagonal read occurs | a lookup answering a needle question |
| **G6 version gate** | the new loader refuses `CAM96CB1`, the old loader refuses `CAM96P01`, and a `compile_fail` doctest shows the two code types do not mix | silent cross-reading |
| G7 basin | re-run the D-SRS-3 held-out gates (`basin.rs`) on pair codes | report only; the July negative is not re-litigated |
| G8 downstream | re-measure `tesseract-paperless::consistency`'s four recorded pair similarities and `ABSOLUTE_ENDORSE_THRESHOLD` | report, then re-pin; never defend the old number |

## 8. Consumers that migrate (named in-tree)

- `deepnsm-v2`:
  - `space.rs`, `codebook.rs`, `lib.rs` (`Nsm` fields and `word_similarity`);
  - `basin.rs`, whose `reconstruct → mean → encode` needs a definition of
    `reconstruct` for a pair code: `C_s[c₁]` (the needle) or the segment
    midpoint. Decided by G7; both reported;
  - `examples/bible_wave.rs`, `examples/pop_readout.rs`, and the
    `lexical.rs` test at `:566-571`.
- `lance-graph-contract`: `episodic_basin.rs` and `canonical_node.rs` (§6.3),
  plus the codec move (§3.3).
- **Out of tree:** `tesseract-paperless/src/consistency.rs` and
  `examples/graph_recovery_demo.rs` (tesseract-rs). They migrate after the
  artifact lands.

## 9. Deliverables

| D-id | deliverable | gate |
|---|---|---|
| D-C96P-1 | analytic Fisher-z codec (`FamilyGamma` encode/decode, off-diagonal fit, table-as-cache) in `lance-graph-contract`; `bgz-tensor` consumes it | parity with `bgz-tensor`'s current output, byte for byte |
| D-C96P-2 | `Cam96Pair` + `Cam96PairSpace` (2-NN encode, R1/R2/R3 similarity, diagonal by address) | G2, G5 |
| D-C96P-3 | `CAM96P01`/`CAM96PW1` loaders + the version gate; `Cam96AxisSpace` rename | G6 |
| D-C96P-4 | Rust trainer `examples/train_cam96p.rs` + the held-out harness | G1, G3, G3b |
| D-C96P-5 | the artifact: produce it (§5 option a) and publish it as a new release | G1 PASS |
| D-C96P-6 | retire `SemanticSpace` from the meaning path (`Spo::pairs` stays an address, never a distance) | G4 |
| D-C96P-7 | `EpisodicBasin` read-mode marker | `v3-envelope-auditor` LAYOUT-GATED verdict |
| D-C96P-8 | consumer migration (§8), in-tree then tesseract-rs | G7, G8 |

## 10. Non-goals

- `lance-graph-contract::recipe_substrate::PairPalette` has the same
  two-independent-axes shape (`recipe_substrate.rs:125-147`, "sum of the
  per-axis squared-L2s"). That is a finding, not in scope; it is filed in
  `ISSUES.md`.
- Minting the header classid (§4, OPEN).
- The 144-verb coarse basis (facet plan §3b).
- Vocabulary coverage: the KJV's 12,543 words, about 80% on modern English
  and nearly none on German. A per-work lemma codebook is the separate
  question the AUTO matcher raises.
- AUTO content cues in `tesseract-paperless`. They wait on this spec.

## 11. Open points

- Is the rail ordered (`c₁` nearest first, as specified) or canonicalized
  (`min:max`)? Ordered keeps the needle in byte 0; canonical halves the key
  space. Not measured.
- Which of R1/R2 wins, and whether the L4 wording changes (§3.2).
- Where the header lives (§4).
- The embedding source for the artifact (§5).
- Whether `basin.rs` should become z-native (average rails in z-space) rather
  than point-native. G7 reports both reconstruct choices; neither is assumed.
