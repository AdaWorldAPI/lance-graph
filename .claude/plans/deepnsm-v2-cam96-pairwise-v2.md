# deepnsm-v2 Cam96 → 6 × pairwise distribution — spec (v2, council draft)

> **Status:** SUPERSEDED by `deepnsm-v2-cam96-pairwise-v3.md` (ratified 2026-09-30); kept as the council's draft record. Was: DRAFT v2. This is the 5+3 council's consolidation of `-v1.md` and the
> five savants' findings (prior art, iron rules, code truth, cascade impact,
> different views). It is **not ratified**, and no code is authorized.
> **Supersedes** `deepnsm-v2-cam96-pairwise-v1.md` for everything below. v1 stays
> as the record. The change ledger (§12) names every delta and its source.
> **D-ids:** `D-C96P-0..8` (§9). D-C96P-0 is new.
> **Written against:** lance-graph `claude/brave-mayer-65y3cy` @ `81aa42c7`
> (on `main` `0d31c54f`).

## 0. The one-sentence change

Cam96 today is **12 independent axis codebooks**. Each rail's two bytes are
nearest centroids of two unrelated 8-d halves, and they are scored as 12
separate squared-L2 terms. The target is **six rails, each a genuine pair**
whose value is a Fisher-z relation, not two unrelated indices. **Which pair**
is decided by a pre-registered measurement between three readings the
governing documents support (§3). Only one of them survives into code.

## 1. Frozen decisions (cited; the council checked compliance and re-opened none)

- **F1.** A word is `classid(4B) = norm(prefix, frequency, PoS)` · `payload(12B) = 6 × (FisherZ:FisherZ)`. Frequency is a header, never a distance axis.
  Source: `deepnsm-morton-comma-facet-v1.md` §0, §2, §5. Measured: `E-DEEPNSM-FACET-BLIND-CONVERGENCE-1`.
- **F2.** One palette256 index is a needle. The pair carries the distribution, and its value is read from the Fisher-z k×k LUT. Self-similarity is 1.0 **by address**, never a lookup. Gamma is fitted on off-diagonals only.
  Source: `entries/2026-08-26-e-palette256-is-a-needle-the-colon-is-the-distribution-1.md`.
- **F3.** L4 = `6 × (8:8)` palette256². The analytic Fisher-z codec is canon, and a materialized table is a cache of the formula.
  Source: `le-contract.md:59`, `:169-186`.
- **F4.** Classid canon: no bit math on a composed classid.
  Source: `sonnet-worker-guardrails.md` §1 rule 4.
- **F5.** I-LEGACY-API-FEATURE-GATED: no path silently reads the other version's bytes.
- **F6.** deepnsm-v2 is the inbound leg.
  Source: `E-DEEPNSM-V2-IS-INBOUND-LEG-REASONING-LIVES-IN-LANCE-GRAPH-1`.
- **F7.** A rail's byte 0 is its needle (nearest centroid). Byte order is never canonicalized away from that. Settled in PR review; it applies to every reading below that has a needle byte.
- **F8.** Zero-copy law.
  Source: `.claude/agents/zero-copy-warden.md`.
- **F9.** Python is lab-only; committed producers are Rust. No `cargo --all`. No model identifiers.
- **F10 (added).** A pair reading is **always selected by the ClassView, never by inspecting payload bytes**.
  Source: `le-contract.md:122-140` (S1-8).
- **F11 (added).** Fisher-z wins a RANK/TAIL read and loses an INTERPOLATE/LEVEL read, measured 5× worse there. Which codebook axis a class uses is decided per class, by measurement.
  Source: `entries/2026-08-11-e-the-byte-was-only-the-selector-the-pair-is-the-carrier-1.md:45-52` (S1-4).

## 2. What ships today (every citation graded CODED by S3)

- `pub type Cam96 = [u8; 12]` (`space.rs:163`). `Cam96Space` has 12 axes and asserts `len == 12` (`:181-196`).
- `encode` works per axis on its own 8-d chunk (`:230-251`). `distance` is `Σ` over 12 squared-L2 terms (`:257-272`).
- `rails()` (`:287-289`) is referenced only by its definition and one test (S3-1).
- The artifact format enforces 12 axes (`codebook.rs:12-14`, `:58`). The producer "split each 16-d subspace into a 256:256 pair" (`fidelity_48_vs_96.py:7`), i.e. 12×8-d PQ (`train_codebook.py:57`, `:77`).
- **Frequency reaches a distance path.** `SemanticSpace::similarity` scores words at their frequency-rank `(basin, identity)` address (`space.rs:74-80`, `vocab.rs:14-21`). That is the facet plan's `PF=Payload` defection. It is exported (`lib.rs:79`).
- **KJV held-out baselines** (`probes/README.md` §4, `codebook.rs:6-8`):

  | code | ρ |
  |---|---|
  | 48-bit point | 0.617 |
  | 96-bit RQ point | 0.786 |
  | 12-axis | 0.774 |

  The *general-vocab* probe's numbers (0.624 / 0.766, `space.rs:173-178`) are a different vocabulary and split. They are **not** the bar. This resolves S2-8's "two baselines".
- **The metric behind every baseline** is Spearman of `1 − cos` between **L2-normalized reconstructions**, against Jina cosine distance (`train_codebook.py:61-67`). Rust's `Cam96Space::distance` is un-normalized summed squared-L2, which is a different metric (S3-3).

## 3. The contested reading, and how it is decided

The governing documents support **three** readings of `6 × (FisherZ:FisherZ)`.
v1 committed to A without naming B's literal support or D's. The council found
both (S1-5, S1-6, S5-1, S5-2), so the choice is made by measurement (§7, G0).

| | a rail is | pair value | source for the reading | byte 0 needle? | risk found |
|---|---|---|---|---|---|
| **A** | `(c₁ : c₂)`, the nearest and second-nearest centroid of ONE per-subspace codebook `C_s` | Fisher-z cell `T_s(c₁,c₂)`: the spread of the word's neighbourhood | needle entry: "the pair is `(a,b) → value`"; 08-11 entry: "the pair is a cell in the centroid tile" | yes: `c₁` is the 48-bit PQ code | c₂ may be near-deterministic given c₁, carrying ~2–3 bits (S5-3). Voronoi swaps (S5-6) |
| **B** | `(z_loc : z_spread)`, two quantized Fisher-z *values* per subspace | the two values themselves; L2 over them ≈ cosine | facet plan `:22-28`: "each subspace becomes a 2-D distributional coordinate (FisherZ:FisherZ) … the pair keeps spread, not just location" | **no** — no centroid index | loses direction: two opposite words can share `z_loc` (v1 §3.4) |
| **D** | `(a : b)`, a cell of a jointly trained 256×256 tile over the subspace (two coupled axes of one 16-d subspace) | the tile's Fisher-z cell distance between two words' cells | 08-11 entry "cell in the centroid tile"; `deepnsm-v3-convergence-v1.md:55-56` "the code and its distance-table dual" | **no** (byte 0 alone is one axis, not a point) | the shipped 12-axis code is the uncoupled version of D; coupling must beat 0.774 to justify itself |

**Committed resolution:**
- **A is the working hypothesis.** It is the only reading that keeps a needle in byte 0 (F7) and makes the colon a relation (F2). B and D are measured, not dismissed.
- G0 (§7) runs all three against the same held-out harness **before any artifact or type is built**.
- If A fails G0's information pre-gate, the spec returns to Phase 0 with G0's numbers. It does not fall back silently.
- The interpretation of F1 is also **escalated to the operator** (§11 Q1): the facet plan's literal text leans to B.

The rest of this spec is written for A. Everything in §5–§9 that depends on A is marked `[A]`.

## 4. Shape A in detail `[A]`

- **Codebook:** one `C_s` per subspace (6 × 16-d), k = 256.
- **Encode:** `c₁ = argmin`, `c₂ = second argmin` in the same `C_s`, written nearest-first (F7).
- **Relation table:** `T_s` = Fisher-z of centroid cosines, with gamma fit on off-diagonals (F2). A diagonal read `(a,a)` returns 1.0 **by address** and never touches `T_s`. That guard is new code, because `FisherZTable::lookup_*` has none (S3-6).
- **Candidate per-rail similarities** (the pick is made on a **validation** split, never on the eval set; S2-8):

  | id | per-rail similarity |
  |---|---|
  | R1 | `mean_z` of the 4 cross cells, symmetric under an `a₁↔a₂` swap |
  | R2 | the needle cell weighted as much as the cross cells |
  | R3 | `T(a₁,b₁)` only |

- **Word similarity:** `tanh(mean_s z̄_s)`, via `contract::distance::mean_similarity_fisher` (`distance.rs:62-69`).
- **The (c₁:c₁) "no second pole" variant (v1 G3b):** report-only, at 3 pre-set thresholds. S5-7 flagged the discontinuity, so it is **never** the default.
- **Stated conflict:** le-contract L4 says "similarity = ONE table read"; R1/R2 read 3–4 cells. That amendment is **escalated** (§11 Q2), not made here.
- **Compose:** undefined for a pair code (z-addition does not compose cosines; `le-contract.md:178-180`). A non-goal (S5-9).

## 5. The producer `[A, B, D arms]`

- **The 96-d embeddings are not in the release.** `data/README.md:9-13`; `bible_vocab_emb96.npy` is absent (S3-9).
  - **Option (a):** re-embed with an operator-supplied key passed through the environment. It never enters a file, commit or brief.
  - **Option (b):** reconstruct from the old codes. Allowed only as a probe arm, never as the artifact.
- **Split parity (S3-9).** A Rust trainer cannot reproduce numpy's PCG64 `permutation`. The lab embedding step therefore writes the train/validation/eval **index file** beside the embeddings (data, not code). The Rust harness reads that file. This is the only way "same words as 0.774" holds.
- **k-means reuse (S1-10).**
  - The harness is an example with an **ndarray dev-dependency**, reusing `ndarray::hpc::edge_codec::Codebook::train` (seeded Lloyd's, `edge_codec.rs:69`).
  - It does not use `cam_pq::kmeans` (farthest-first, no seed; `cam_pq.rs:541`) and it is not another copy.
  - deepnsm-v2 has no dev-dependencies today (`Cargo.toml:23-29`), so this adds the first one.
- **Metric parity (S3-3).** The harness computes the baselines' exact metric (normalized-reconstruction `1 − cos` Spearman) **and** the arm's own similarity. Both are reported; the gate uses the first.
- **The 48-bit reference (S3-10).** The harness trains an **independent** 6×16-d k-means-256 with a different seed, as the G2 comparand. Without it, "c₁ equals the 48-bit code" is true by construction.

## 6. Format, types and the version gate `[A]`

### 6.1 Artifacts
- **`CAM96P01` codebook**, laid out as:
  - `u32` n_sub (6), `u32` dim (16), `u32` k;
  - the centroids;
  - 6 × `FamilyGamma` (48 B).
- **`CAM96PW1` codes** carry a **`u64` digest of the codebook blob** in the header (S2-5). The loader refuses a mismatch.
  - OPEN: the digest function is chosen at D-C96P-3 from what the contract already exposes; that is not verified here.
- **Magics:** all four rejections — pair codebook ⟂ `CAM96CB1`, axis codebook ⟂ `CAM96P01`, pair codes ⟂ `CAM96WD1`, axis codes ⟂ `CAM96PW1`.
  - Two of these already exist (`codebook.rs:43-45`, `:91-93`; S2-6). All four get disable runs.
- **The PW1 loader borrows:** it returns `&[[u8;12]]` over the blob, not a `Vec` (S2-9; the existing loader copies at `codebook.rs:99-105`).

### 6.2 Types
- The new code is `Cam96Pair`. It has **no public field and no `From<[u8;12]>`**; it is constructible only via the pair encoder or loader (S2-7).
- **The old names stay un-deprecated.** v1 proposed renaming `Cam96Space` → `Cam96AxisSpace`. S4-7 showed that tesseract-rs builds against lance-graph's default branch with no pin (`rust.yml:35-38`, `Dockerfile:98`), so a rename or deprecation breaks it the day it merges. New types are added **beside** the old ones. Retiring the old names is its own proof-gated PR.

### 6.3 The persisted 12 bytes (the one gate v1 left open)
- **The problem.** `BasinRow.self_code` is a raw `[u8;12]` (`episodic_basin.rs:92`). Its `from_le_bytes` is total (`:130-144`). The row is fully packed at 2+2+12+8+8 = 32 bytes (`:76`, `:120-127`). The newtype never reaches it (S2-3/4, S4-5).
- **Resolution.** The shape of `self_code` becomes a **ClassView-selected reading** (F10), modelled on `EdgeCodecFlavor` (`canonical_node.rs:805-819`): same bytes, per-class reading, no layout change.
  - It is a new `ReadMode` axis, which trips the structural fuse at `canonical_node.rs:1629-1633` and touches 16 `ReadMode` literals plus `hotplug.rs` and `lance-graph-ogar` (S1-9, S4).
  - **Zero value = the legacy axis reading.**
  - It requires a `v3-envelope-auditor` LAYOUT-GATED verdict **before** D-C96P-7 is written.
- **EMPTY is unaffected.** `BasinRow::EMPTY` is judged on the whole row (`member_count = 0`), so an all-zero self-code in a real basin stays distinguishable (S3-7, S4-6).
- **Today's only producer.** `arigraph/episodic.rs:227` writes `[0;12]` (S3-7). No writer of a non-zero code exists, so "nothing pair-coded is persisted" currently holds by absence, not by a guard (S4-6).
- **Ordering rule** (§9): D-C96P-7 lands before D-C96P-8 may emit pair self-codes.

## 7. Pre-registered gates

Every guard gets a disable run: red with the guard removed, green with it back. The eval set is the held-out split of §5. The R1/R2 choice is made on validation.

| gate | criterion | fails when |
|---|---|---|
| **G0 information pre-gate (new, runs first)** | For A: over the real code population, report per subspace (i) the entropy H(c₂ \| c₁) and (ii) the variance of `T_s(c₁,c₂)`. For B and D: their own held-out ρ on the same harness and metric. | A is **KILLED** (return to Phase 0 with the numbers) if H(c₂\|c₁) < 1 bit in 4 of 6 subspaces. The threshold is a **policy pin, labelled as such**. |
| **G1 fidelity** | the best of R1/R2, chosen on validation, has eval ρ ≥ **0.774** on the baselines' own metric | < 0.774 and ≥ 0.617 = PARTIAL (operator decides); < 0.617 = KILL |
| G1 report | separate rows: R1, R2, R3 beside the 48-bit 0.617; RQ 0.786; the 12-axis 0.774; B; D; recon MSE for each | — |
| **G2 needle** | c₁ from `C_s` agrees with the **independently trained** 48-bit reference on at least a reported fraction of words; R3's ρ is reported beside 0.617 | report-only. The byte-for-byte check is dropped because it was vacuous by construction (S3-10) |
| **G3 spread is load-bearing** | shuffling c₂ **only among words that share c₁** lowers R1/R2 ρ by more than a **word-blocked bootstrap noise floor** (hand-tuned block size, labelled) | no drop beyond the floor. The v1 full-shuffle null is dropped: it creates pairs 2-NN never produces, so it could not stay silent (S5-4, S2-8) |
| G3v Voronoi (new) | under a pre-set small embedding perturbation, report the c₁/c₂ swap rate and the c₂-only change rate | report only (S5-6) |
| **G4 header orthogonality** | word similarity is bit-identical under a permutation of `WordId` and of `LexicalEvidence` counts | frequency or PoS leaks in |
| **G5 diagonal by address** | `sim(a,a) == 1.0` with no `T_s` read on the diagonal | a lookup answers a needle question |
| **G6 version gate** | the four magic rejections, the PW1 digest mismatch refused, and a `compile_fail` doctest that the code types do not mix | silent cross-reading |
| G7 basin | D-SRS-3 gates on pair codes, **both** reconstruct choices (`C_s[c₁]`, segment midpoint). Averaging is over decoded points or z, **never over index bytes** (S2-2) | report only. F11 predicts that a LEVEL read may not favour Fisher-z |
| G8 downstream | re-measure `tesseract-paperless::consistency`'s `ABSOLUTE_ENDORSE_THRESHOLD` (0.5, `consistency.rs:289`) on the new scale | report, then re-pin. Its tests track the constant (`:783`, `:791`), so nothing fails on a scale change (S4-8). `LOW_CONFIDENCE_THRESHOLD` is LSTM-side and unaffected |

**Scope of every verdict:** the KJV held-out vocabulary (2,543 words). No gate here says anything about other vocabularies or languages (S5-10).

## 8. Consumers (S3-7 + S4 census)

**In tree, mandatory.** In deepnsm-v2:
- `space.rs`, `codebook.rs`, `lib.rs:14`, `:29-35`, `:61`, `:79`, `:102-140`
- `basin.rs` (33 refs; gates take `&Cam96Space` at `:185`, `:224`, `:262`)
- `lexical.rs:492`, `:566-571`, `spo.rs:51`
- `examples/bible_wave.rs:30`, `:118-119`, `:461-494`, `:951-952`
- `examples/pop_readout.rs:84`, `:168-169`, `:279-280`, `:322`, `:404-417`, `:743` (these include a direct `Cam96` × `self_code` distance)
- `data/README.md`, `probes/README.md` §4

In `lance-graph-contract`: `episodic_basin.rs`, `canonical_node.rs:1062`, `:1078`, `:1235`, `:1351`, `:2653-2698`, `band_reading.rs:151`, `lib.rs:98-99`. Also `lance-graph/src/graph/arigraph/episodic.rs:215-233`, `:959`, and `.claude/v3/soa_layout/tenants.md` (the tenant-table row `EpisodicBasin`, D-ACR-6 rail).

**Codec move.**
- `bgz-tensor/Cargo.toml:27` (the contract dependency goes from optional to required);
- `bgz-tensor/src/fisher_z.rs`, `src/lib.rs:69`, `:114` (it re-exports, so no call-site churn in `shared_palette.rs`, `hhtl_d.rs`, `morton_cascade/*`).

**Out of tree, follow-up** (after D-C96P-5):
- tesseract-rs `consistency.rs:63`, `:245-289`, `:315-319`, `:826-830`; `examples/graph_recovery_demo.rs:61-62` (hard-coded asset names);
- `README.md:55`, `TOKEN-SEAM-ARCHITECTURE.md:383`, `:465`.

paperless-rs pins deepnsm-v2 at git rev `abac911b` and is a documented dead copy, so it is **not** migrated.

**Plans to annotate:**
- `self-reasoning-substrate-v1.md:141`, `:425`, `:456-460`
- `post-teardown-buildup-survey-v1.md:87`, `:141`, `:185`, `:219`
- `deepnsm-v2-lexical-evidence-consumer-v1.md`: the new release must republish an **identical** `bible_vocab.txt`.

## 9. Deliverables and ordering

| D-id | deliverable | gate | after |
|---|---|---|---|
| **D-C96P-0** | the G0 harness: all three arms, the metric-parity scorer, the split-index reader, the independent 48-bit reference | G0, G1 report | embeddings (§5) |
| D-C96P-1 | the analytic Fisher-z codec into `lance-graph-contract` (details below) | a **new golden-bytes parity test** vs bgz-tensor's current output; no arithmetic change, so the ρ≥0.999 example certification need not re-run | — |
| D-C96P-2 `[A]` | `Cam96Pair` + `Cam96PairSpace`: 2-NN encode, R1/R2/R3, diagonal by address | G2, G3, G5 | 0 (A survives), 1 |
| D-C96P-3 `[A]` | `CAM96P01`/`CAM96PW1` loaders, digest, four rejections, borrowing codes loader | G6 | 2 |
| D-C96P-4 | the trainer example (same harness as D-0) | G1 | 3 |
| D-C96P-5 | the artifact + release, with an identical `bible_vocab.txt` | G1 PASS | 4 |
| D-C96P-6 | retire `SemanticSpace` from the meaning path | G4 | — |
| D-C96P-7 | the `self_code` shape ReadMode axis | v3-envelope-auditor verdict | — |
| D-C96P-8 | consumer migration, in tree then tesseract-rs | G7, G8 | 5, **7** |

D-C96P-1 in detail:
- **Moved:** `FamilyGamma` encode/decode/fit, table build, `cosine_f32`; about 100 lines, not v1's "~40" (S3-5).
- **Reused:** the contract's existing `atanh` clamp (`distance.rs:42-52`), so the formula is spelled once (S1-2).
- **Canon clamp:** ±0.9999, as bgz-tensor and the contract already agree. helix's ε = 1e-9 (`helix/src/fisher_z.rs:55-58`) is a separate spatial codec and out of scope (S1-3).
- **Stays in bgz-tensor:** `build_from_palette`, which has no callers (S4-3).
- **bgz-tensor** re-exports the moved items.

## 10. Non-goals

- `PairPalette` (`ISS-PAIRPALETTE-IS-TWO-AXES-NOT-A-PAIR`).
- The classid header mint (§11 Q3).
- The 144-verb basis.
- Vocabulary coverage and per-work lemma codebooks.
- AUTO content cues.
- A compose table for pair codes.
- Retiring the old Cam96 names.
- Migrating paperless-rs.

## 11. Escalated to the operator (not decided by the council)

- **Q1.** Which reading of `6 × (FisherZ:FisherZ)` did the facet plan intend: A, B or D (§3)? G0 measures all three, but the intent is yours.
- **Q2.** If R1/R2 win, may le-contract L4's "similarity = ONE table read" be amended?
- **Q3.** Where does the frequency/PoS header live, given the classid canon?
- **Q4.** The embedding source (§5 option a) needs a key you supply through the environment.

## 12. Change ledger (v1 → v2)

| # | change | source |
|---|---|---|
| 1 | §3 rewritten: the three readings, a measurement-decided choice, and escalation of F1's intent | S1-5, S1-6, S5-1, S5-2 |
| 2 | G0 information pre-gate added, runs first, can KILL A | S5-3, S5-5 |
| 3 | G3 null is within-c₁ with a word-blocked noise floor; the full-shuffle null is dropped | S5-4, S2-8 |
| 4 | G1 uses the baselines' own normalized `1−cos` metric; R1/R2 picked on validation | S3-3, S2-8 |
| 5 | G2 byte-for-byte check dropped as vacuous; an independent 48-bit reference was added | S3-10 |
| 6 | G3v Voronoi instability report added | S5-6 |
| 7 | §6.3 resolved as a ClassView-selected ReadMode axis (zero = legacy); envelope-auditor gate; ordering D-7 before D-8 | S2-3, S2-4, S1-8, S1-9, S4-5, S4-6 |
| 8 | PW1 carries a codebook digest | S2-5 |
| 9 | `Cam96Pair` has no public field or `From`; the PW1 loader borrows | S2-7, S2-9 |
| 10 | Old names kept un-deprecated (v1's rename dropped) | S4-7 |
| 11 | Codec move: ~100 lines, reuse the contract's atanh, canon clamp, `build_from_palette` stays, re-export, golden-bytes test | S1-2, S1-3, S3-5, S3-6, S4-2, S4-3 |
| 12 | Trainer: split-index file, reuse ndarray `edge_codec::Codebook::train` via dev-dependency | S3-9, S1-10 |
| 13 | F10 and F11 added as frozen decisions | S1-8, S1-4 |
| 14 | §8 consumer list completed (arigraph producer, pop_readout, contract refs, tenants.md, asset names, plans) | S3-7, S4-1, S4-8, S4-9 |
| 15 | Baseline disagreement resolved: KJV §4 numbers are the bar; general-vocab numbers labelled | S2-8 |
| 16 | G7 forbids averaging index bytes and reports both reconstruct choices; F11 cited | S2-2, S1-4 |
| 17 | Verdict scope pinned to the KJV vocabulary | S5-10 |
| 18 | Compose is a non-goal | S5-9 |
| — | **not adopted:** S4-10's worry that the codec move duplicates helix/jc Fisher-z. S1-2 showed FamilyGamma exists only in bgz-tensor, and helix is a different (spatial, f64, ε 1e-9) codec. Recorded, not deleted | S4-10 vs S1-2 |
