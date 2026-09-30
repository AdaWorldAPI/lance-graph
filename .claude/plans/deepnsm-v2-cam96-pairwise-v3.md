# deepnsm-v2 Cam96 → 6 × pairwise distribution — spec (v3, RATIFIED)

> **Status:** RATIFIED v3 of the 5+3 council.
> - v1 was the proposal and v2 the council draft; both are kept as the record. Ledger in §13.
> - **Implementation is authorized only for D-C96P-0 (the measurement harness) and D-C96P-1 (the codec move).** Both run on data that can be produced in-tree.
> - Every other deliverable waits on G0/G1, which wait on the embeddings (§5, Q4).
> - The reading of F1 is resolved as the merged reading **M** (§3). The operator proposed it on 2026-09-30; confirmation is Q1.
>
> **D-ids:** `D-C96P-0..8`.
> **Written against:** lance-graph `claude/brave-mayer-65y3cy` @ `81aa42c7` (base `main` `0d31c54f`).

## 0. The change in one paragraph

Today, rail `r`'s two bytes quantize the two contiguous 8-d halves of one 16-d
slice independently (`space.rs:182`, `:287-289`; `train_codebook.py:22-24`),
and distance is 12 separate squared-L2 terms (`space.rs:257-272`). Neither byte
refers to the other.

The target makes each rail one object: a **needle** plus **where the word sits
between that needle and one of its fixed neighbours**. Similarity is read
exactly from the per-subspace relation table.

The competing readings A, B and D (v2 §3) are projections of this one object,
measured as its ablations. Losing arms are recorded as findings, never deleted.

## 1. Frozen decisions

- **F1.** A word is `classid(4B) = norm(prefix, frequency, PoS)` · `payload(12B) = 6 × (FisherZ:FisherZ)`, and frequency is a header, never a distance axis.
  - Source: `deepnsm-morton-comma-facet-v1.md` §0, §2, §5.
  - **The literal text is frozen. Its reading was contested (v2 §3) and is resolved here as M, pending Q1.**
- **F2.** One palette256 index is a needle. The pair carries the distribution through the Fisher-z k×k LUT. Self-similarity is 1.0 by address. Gamma is fitted on off-diagonals only.
  - Source: `entries/2026-08-26-e-palette256-is-a-needle-…-1.md`.
- **F3.** L4 = `6 × (8:8)` palette256². The analytic Fisher-z codec is canon, and a materialized table is a cache.
  - Source: `le-contract.md:59`, `:169-186`.
- **F4.** No bit math on a composed classid.
- **F5.** I-LEGACY-API-FEATURE-GATED: no path silently reads the other version's bytes.
- **F6.** deepnsm-v2 is the inbound leg.
- **F7 — DECISION.** Byte 0 of a rail is its needle, the nearest centroid.
  - Basis: PR AdaWorldAPI/lance-graph#1303 review thread on v1 §11. G2 depends on it.
  - Scope: every arm that has a needle byte.
- **F8.** Zero-copy law.
- **F9.** Python is lab-only; committed producers are Rust. No `cargo --all`. No model identifiers.
- **F10.** A pair reading is selected by the ClassView, never by inspecting payload bytes.
  - Source: `le-contract.md:122-140`.
- **F11.** Fisher-z wins RANK/TAIL reads and loses INTERPOLATE/LEVEL reads (measured 5× worse). The axis is chosen per read, by measurement.
  - Source: `entries/2026-08-11-e-the-byte-was-only-the-selector-…-1.md:45-52`, a storm-pressure probe; its transfer to word codes is itself tested by G5b.
- **F12 (added).** "Without materialization" holds only on transmitted-index content. A static position must be **stored**; it cannot be regenerated.
  - Source: `E-PERTURBATION-CONVERGENCE-1` invariant 1 (`EPIPHANIES-ARCHIVE-2026-09-20.md:21855`); `E-3DGS-MU-HYDRATION-2`.

## 2. What ships today

The citations below were verified by the code-truth savant and the overclaim
reviewer. The README ρ values are recorded Python outputs, not code facts.

- `pub type Cam96 = [u8;12]` (`space.rs:163`). `Cam96Space` has 12 axes (`:181-196`).
- Encoding is per-axis (`:230-251`). Distance is a sum of 12 squared-L2 terms (`:257-272`).
- `rails()` is a view used only at `:287` and a test (`:368`).
- The 12-axis artifact format: `codebook.rs:12-14`, `:58`. KJV producer: `train_codebook.py:57`, `:77`.
- The quote "each 16-d subspace split into a 256:256 pair" is the **general-vocab** probe (`fidelity_48_vs_96.py:7`).
- `SemanticSpace` scores at the frequency-rank address (`space.rs:74-80`, `vocab.rs:14-21`).
  - It is an **exported API with no caller outside its own tests** (searched lance-graph/crates and tesseract-rs).
  - It is a latent `PF=Payload` defection, not a live path.
- **KJV held-out reference figures** (`probes/README.md` §4, recorded outputs):

  | code | ρ |
  |---|---|
  | 48-bit | 0.617 |
  | RQ | 0.786 |
  | 12-axis | 0.774 |

  - They were measured on `bible_vocab_emb96.npy`, which is **absent**.
  - The general-vocab 0.624 / 0.766 (`space.rs:173-178`) are a different vocabulary and split.
- **The baseline metric** is Spearman of `1 − cos` of L2-normalized reconstructions against the reference embedding's (Jina-v3, data provenance) cosine distance (`train_codebook.py:61-67`).
  - Rust's `Cam96Space::distance` is a different metric.

## 3. The reading: M, and A/B/D as its ablations

**The object.** In subspace `s` there is one codebook `C_s` (256 centroids, 16-d) with norms `n_c = ‖C_s[c]‖` and a centroid-cosine table `T_s(c,c')`. A word's subvector `x` becomes:

```
byte 0 = a                 = argmin_c ‖x − C_s[c]‖          (needle; = the 48-bit PQ code by F7)
byte 1 = (j : 4 bits | t : 4 bits)
         b = N_a[j]        N_a = a's 16 nearest centroids by T_s, fixed once C_s is trained
         t ∈ {0..15}/15    position along the arc a→b, uniform in ANGLE (F11: a level read)
         (j, t) chosen to minimize the angle between x and x̂(a, b, t)
         byte 1 = 0        means "needle only, no second pole" (t = 0 ⇒ x̂ = u_a)
x̂ = slerp(u_a, u_b, t) · n̄   where u_c = C_s[c]/n_c, φ = arccos T_s(a,b),
    w_a = sin((1−t)φ)/sin φ, w_b = sin(tφ)/sin φ,
    and n̄ = (1−t)·n_a + t·n_b interpolates the norm
```

**Why this is the merge of the two operator readings.**
- The **resonance** reading gives `r_a = cos(x,u_a)` and `r_b = cos(x,u_b)`.
- The **segment** reading gives the fixed arc length `φ`.
- Then `t = θ_a/φ` exactly when `x` lies on the arc (`θ_a + θ_b = φ`).
- The triangle excess `ε = θ_a + θ_b − φ ≥ 0` is what the two poles leave unexplained.
- The "properties in between" are rendered deterministically from `(a, j, t)` and the fixed codebook. Which point the word is (`t`) is stored, per F12.

**Similarity (one formula, every arm).**
- `x̂` is a combination of two unit centroids, so the exact reconstruction cosine needs no float vector. It reads the Gram blocks from `T_s` and the norms:

  ```
  dot_s(A,B) = Σ_{p∈{a₁,a₂}, q∈{b₁,b₂}} w_p w_q n̄_A n̄_B T_s(p,q)
  cos(x̂_A, x̂_B) = Σ_s dot_s(A,B) / sqrt(Σ_s dot_s(A,A) · Σ_s dot_s(B,B))
  ```

- This is **exactly the baseline metric** (normalized-reconstruction cosine; §2), so no metric-parity gap exists.
- A shared centroid (`p = q`) contributes `T(p,p) = 1` by address (F2). That is the correct Gram entry, not a needle averaged into a z-mean, so R2's diagonal-mixing concern does not arise here.
- Fisher-z enters only as the storage of `T_s` (i8 + gamma). Its quantization loss is measured by G5b.

**A, B and D as ablations of M** (each is a G0 arm):

| arm | what is removed | what it is | source of the reading |
|---|---|---|---|
| M | — | needle + neighbour + position | the operator's 2026-09-30 merge |
| M−t | position (t = ½ fixed) | pole pair, no position | — |
| M−j | neighbour choice (j = 0, the nearest neighbour) | needle + position toward a fixed neighbour | — |
| A | position; the second pole stored as a full byte (2-NN of the word) | v1/v2's `(c₁:c₂)` | needle entry `(a,b)→value`; 08-11 entry `:13` ("a cell in the centroid tile"). The latter is a storm-probe source, a domain transfer |
| B | addresses | per subspace, quantized Fisher-z of `cos(x, anchor₁)`, `cos(x, anchor₂)`, anchors = the subspace's top-2 principal directions | facet plan `:25-27` ("2-D distributional coordinate (FisherZ:FisherZ)") |
| D | per-word position and neighbour | the 48-bit code scored through `T_s` (the code's distance-table dual) | `deepnsm-v3-convergence-v1.md:55-56` |

- The v2 reading "D = a jointly trained 256×256 tile per word" is **not supported by its cited source** (R1-§3). It would also be a 65,536-cell address the canon never writes (R2-§3). It is dropped as a reading and recorded here.

**Decision table (pre-registered).** Each rule is measured on eval, with the noise floor from G3:
- M beats every arm → M ships.
- An ablation ties M (within the floor) → the removed component carries nothing. The **simpler** arm ships and M's extra field is dropped.
- A, B or D beats M (outside the floor) → return to Phase 0 with the numbers; the operator decides (Q5).
- Every arm below the in-harness 48-bit control → KILL the pair program, recorded as a finding.

## 4. Codebook quantities

Per subspace, `C_s` also carries:
- the norms `n_c` (256 f32);
- `T_s` as Fisher-z i8 + `FamilyGamma`, gamma fitted on off-diagonals;
- `N_a` (16 u8 per centroid).

`N_a` is derived from `T_s` at load time. It is never stored twice.

Diagonal reads `T(p,p)` return 1.0 by address, and `T_s`'s diagonal is never consulted. That is new code: `FisherZTable::lookup_*` has no guard (`fisher_z.rs:158-172`).

Compose: the analytic codec does not provide compose (`le-contract.md:178-180`: "the semiring COMPOSE keeps its table"). A compose table for pair codes is a non-goal.

## 5. The producer and the harness (two separate things)

**Embeddings.** `bible_vocab_emb96.npy` is not in the release (`data/README.md:9-13`).
- **Option (a):** re-embed with an operator-supplied key passed through the environment. The key never enters a file, commit, brief, CI log or command line, and is never echoed.
- **Option (b):** reconstruct from the old codes. Allowed in a probe only, never for the artifact.

**D-C96P-0 — the measurement harness** (an example with an ndarray dev-dependency; deepnsm-v2 has none today, `Cargo.toml:23-29`):
- **In-harness controls, retrained on the same vectors and split:** the 12-axis PQ, the 48-bit PQ and the RQ point. This is R1's BLOCK fix: under option (a) the vectors differ from those behind the pinned figures, so the gate compares against controls trained on the same vectors. The pinned 0.617 / 0.786 / 0.774 are reported for reference only.
- **Arms:** M, M−t, M−j, A, B, D (§3).
- **One k-means for all arms:** `ndarray::hpc::edge_codec::Codebook::train` (seeded Lloyd's, `edge_codec.rs:69`), so arm differences are not trainer differences. It is f32 with SplitMix init, which is not the Python baseline's f64 numpy Lloyd. That is one more reason the in-harness controls are the bar.
- **Split:** train / validation / eval, from an index file written by the embedding step (data). Arm and formula choices are made on validation, and every gate is reported on eval.
- **B's anchors** are fitted on train.

**D-C96P-4 — the trainer** is a separate example that produces only the winning arm's artifact. It carries no probe arms and no option-(b) path.

## 6. Format, types and the version gate (for M; A-arm formats are harness-internal)

### 6.1 Artifacts
- **`CAM96P01` codebook**, in this order:
  1. `u32` n_sub, `u32` dim, `u32` k;
  2. the centroids;
  3. the norms;
  4. 6 × `FamilyGamma`.

  The i8 `T_s` table is an optional cache section (F3).
- **`CAM96PW1` codes** carry a `u64` digest of the codebook blob, and the loader refuses a mismatch. The digest function is chosen at D-C96P-3.
- **All four magic rejections:**
  - pair codebook ⟂ `CAM96CB1`
  - axis codebook ⟂ `CAM96P01`
  - pair codes ⟂ `CAM96WD1`
  - axis codes ⟂ `CAM96PW1`

  The existing loaders already refuse foreign magics (`codebook.rs:43-45`, `:91-93`). All four get disable runs.

### 6.2 Types
- `Cam96Pair` is a newtype over `[u8;12]` with no public field and no `From<[u8;12]>`.
- **The PW1 loader never hands out raw arrays.** R1's BLOCK: `pub type Cam96 = [u8;12]` is an alias, so a `&[[u8;12]]` would flow straight into `Cam96Space::distance`.
  - It returns a borrowing view `Cam96PairCodes<'a>` over the blob.
  - `get(i) -> Cam96Pair` is a 12-byte `Copy` microcopy (data-flow rule 2).
  - No slice of raw arrays is exposed.
- **The old names stay.** A non-breaking alias `pub type Cam96AxisSpace = Cam96Space;` is added for disambiguation (R2).
  - Justification, downgraded per R1: tesseract-rs builds against lance-graph's default branch with no ref (`rust.yml:35-38`) and reaches `Cam96Space` through `load_cam96_space` (`consistency.rs:63`, `:315-319`).
  - A signature change there would break it. Keeping the names avoids having to check that.

### 6.3 The persisted 12 bytes
- **The row.** `BasinRow.self_code` is raw `[u8;12]` (`episodic_basin.rs:92`) with a total `from_le_bytes` (`:130-144`), in a packed 32-byte row (`:76`, `:120-127`).
  - `EpisodicBasin` is `ValueTenant = 15`, a lane in a `NodeRow`'s 480-byte value slab (`canonical_node.rs:1045-1078`).
  - So **the ClassView key is the owning row's key classid**, read via `classid_read_mode` (F10). Nothing in the payload selects the reading.
- **Resolution.** The code shape is selected per class. Zero means the legacy axis reading. Two candidate mechanisms:
  - (i) a new `ValueSchema` variant on the existing `value_schema` axis;
  - (ii) a new `ReadMode` axis. This trips the fuse at `canonical_node.rs:1629-1633` and touches 12 `ReadMode` struct literals (`:1483-1629`), `hotplug.rs:341` and `lance-graph-ogar/src/lib.rs:467`.

  The choice between them is **`v3-envelope-auditor`'s**, and it is made before D-C96P-7 is written.
- **EMPTY.** `is_empty` is `*self == Self::EMPTY`, i.e. all fields zero (`:114-116`). A real basin with an all-zero code has `member_count > 0` and is not empty.
- **Current producer.** The only producer found in lance-graph/crates writes `[0;12]` (`arigraph/episodic.rs:227`). Nothing pair-coded is persisted today, and that holds by absence; D-C96P-7 is ordered before D-C96P-8.

## 7. Pre-registered gates

Every guard gets a disable run: red with the guard removed, green with it back. Gates are reported on eval; choices are made on validation.

| gate | criterion | fails / outcome |
|---|---|---|
| **G0 information** | Per subspace, report: (i) how often the word's true second-nearest centroid is in `N_a`; (ii) the entropy of `j` given `a`; (iii) the distribution of `t`; (iv) the distribution of the triangle excess `ε` | If (i) is < 0.9 in 4 of 6 subspaces, M's neighbour window is too small: G0 reports it, and window 32 (5 bits j, 3 bits t) is added as an arm. Thresholds are policy pins, labelled as such |
| **G1 fidelity** | The shipping arm's eval ρ ≥ the **in-harness 12-axis** ρ, on the baseline metric | Between the in-harness 48-bit and 12-axis ρ = PARTIAL (operator). Below the 48-bit = KILL |
| G1 report | One row per arm (M, M−t, M−j, A, B, D) plus each in-harness control and the pinned reference figures, each with recon MSE | — |
| **G2 needle** | Unit gate: for every eval word, byte 0 == argmin over `C_s`, and (j, t) == the encoder's minimizer. Report: agreement of byte 0 with an **independently seeded** 48-bit PQ | The unit gate falsifies F7. Swapping the byte order fails it |
| **G3 each field carries information** | Each ablation's ρ gap to M exceeds a **word-blocked bootstrap noise floor** (hand-tuned block size, labelled). The full shuffle of byte 1 across words is kept **report-only** as a ceiling | No gap beyond the floor = that field carries nothing, and the decision table's tie rule applies. G0 is its anti-vacuity partner: it shows the fields vary, so a tie means "uninformative", not "no power" |
| G3v Voronoi | Under a pre-set small perturbation of eval vectors, report the change rate of byte 0, j and t | report only |
| **G4 no frequency in the code** | Structural: `encode` takes the vector and `C_s` only. Test: encoding the same vectors under a permuted `WordId` assignment gives identical codes per word | A code depends on routing. Embedding-side frequency influence is out of scope (measured ρ ≈ −0.07 in `E-CAM96-DISTRIBUTION-MEASURED-1`) |
| **G5 diagonal by address** | `sim(x,x) = 1.0`, and no diagonal read of `T_s` | a lookup answers a needle question |
| **G5b table parity** | ρ from the i8 Fisher-z `T_s` vs ρ from f32 centroid cosines, reported. Tests F11's transfer to this read | report, and a Fisher-z loss > 0.01 ρ is flagged. Threshold is a policy pin |
| **G6 version gate** | The four magic rejections, the digest mismatch refused, a `compile_fail` doctest that the code types do not mix, and no API returns `[u8;12]` from pair codes | silent cross-reading |
| G7 basin | D-SRS-3 gates on pair codes. Averaging is over decoded points only, never over code bytes | report. An unfavourable result keeps those classes on the legacy read mode (§6.3 zero value) |
| G8 downstream | Re-measure `consistency.rs`'s `ABSOLUTE_ENDORSE_THRESHOLD` (0.5, `:289`) on the new scale | report, then re-pin. Its tests track the constant ±0.05 (`:783`, `:791`) and will not catch a scale change. `LOW_CONFIDENCE_THRESHOLD` (`:245`) is LSTM-side and unaffected |

**Scope.** Every verdict is scoped to the KJV held-out vocabulary. Nothing here speaks for other vocabularies or languages.

## 8. Consumers

**In tree, mandatory.**
- deepnsm-v2: `space.rs`; `codebook.rs`; `lib.rs:14`, `:29-35`, `:61`, `:79`, `:102-140`.
- deepnsm-v2 `basin.rs`: gates at `:186`, `:225`, `:263`.
- deepnsm-v2: `lexical.rs:492`, `:566-571`; `spo.rs:51`.
- deepnsm-v2 examples: `examples/bible_wave.rs` and `examples/pop_readout.rs`. The latter contains a direct `Cam96` × `self_code` distance. Line lists are from the code-truth savant and were not re-verified by R1.
- deepnsm-v2 docs: `data/README.md` and `probes/README.md` §4.
- The contract's `episodic_basin.rs`, `canonical_node.rs`, `band_reading.rs:151` and `lib.rs:98-99`.
- `arigraph/episodic.rs:215-233`, `:959`.
- `.claude/v3/soa_layout/tenants.md`, the tenant-table row `EpisodicBasin` (D-ACR-6 rail).

**The codec move (D-C96P-1).**
- `bgz-tensor/Cargo.toml:27`: the contract dependency becomes required.
- Every caller uses the **module path** `bgz_tensor::fisher_z::…`:
  - `shared_palette.rs:36`;
  - `morton_cascade/{mod.rs:29, v3.rs:13, legacy.rs:15}`;
  - `examples/probe_l5_fisherz_amortization.rs:29`, `examples/nnue_palette_cosine.rs:30`;
  - `thinking-engine/examples/cascade_attention_probe.rs:29`.

  So **`fisher_z.rs` itself re-exports** the moved items. The crate-root re-export at `lib.rs:114` is not enough (R1).

**Out of tree, follow-up.**
- tesseract-rs:
  - `consistency.rs:63`, `:245-289`, `:315-319`, `:826-830`;
  - `examples/graph_recovery_demo.rs:61-62` (hard-coded asset names);
  - `README.md:55`;
  - `TOKEN-SEAM-ARCHITECTURE.md:383`, `:465`.
- paperless-rs (git-rev pinned dead copy) is not migrated.

**Plans to annotate.**
- `self-reasoning-substrate-v1.md`.
- `post-teardown-buildup-survey-v1.md`.
- `deepnsm-v2-lexical-evidence-consumer-v1.md`. The new release must republish an **identical** `bible_vocab.txt`.

## 9. Deliverables, ordering and board hygiene

| D-id | deliverable | gate | after |
|---|---|---|---|
| D-C96P-0 | the measurement harness (§5), producing G0/G1/G2-report/G3/G3v/G5b | G0, G1 report | embeddings |
| D-C96P-1 | the analytic Fisher-z codec into `lance-graph-contract` | a **new golden-bytes parity test** vs bgz-tensor's current output. No arithmetic change, so the ρ≥0.999 example certification need not re-run | — |
| D-C96P-2 | `Cam96Pair`, `Cam96PairSpace` (encoder, Gram similarity, diagonal by address, `N_a`) for the arm the decision table picks | G2, G3, G4, G5 | 0, 1 |
| D-C96P-3 | loaders, digest, four rejections, `Cam96PairCodes` view, `Cam96AxisSpace` alias | G6 | 2 |
| D-C96P-4 | the trainer (winning arm only) | G1 | 3 |
| D-C96P-5 | artifact + release, with an identical `bible_vocab.txt` | G1 PASS | 4 |
| D-C96P-6 | `SemanticSpace` documented as routing-addressed, not meaning. **Kept exported, no `#[deprecated]`**: no caller exists, and a deprecation would fail downstream `-D warnings` builds (S4-7's breakage test re-applied) | G4 | — |
| D-C96P-7 | the §6.3 per-class code-shape reading | `v3-envelope-auditor` verdict | — |
| D-C96P-8 | consumer migration, in tree then tesseract-rs | G7, G8 | 5, **7** |

D-C96P-1 in detail:
- **Moved:** `FamilyGamma` encode/decode/fit, table build, `cosine_f32`.
- **atanh clamp:** today inline in `Distance::similarity_z` (`distance.rs:42-52`). It is factored into one contract function used by both. That is a contract change, and it is bit-equivalent (±0.9999).
- **helix's ε = 1e-9 f64** (`helix/src/fisher_z.rs:55-58`) is a separate spatial codec and out of scope.
- **`build_from_palette`** is an inherent method today (`fisher_z.rs:150`). It becomes a **free function** in bgz-tensor, because an inherent impl on a foreign type is E0116. It has no callers.

**Board hygiene (R3 BLOCK fix), binding on every deliverable:**
- Each D-id lands with its board updates **in the same commit**:
  - its `STATUS_BOARD` row flips;
  - `LATEST_STATE` Contract Inventory is updated for D-C96P-1, -2 and -7;
  - measured G0/G1 results go into an `entries/` file plus `python3 .claude/tools/entries_index.py --write`;
  - a `PR_ARC_INVENTORY` entry is added after merge;
  - `python3 .claude/tools/supersession_index.py > .claude/board/SUPERSESSION-INDEX.md` is run **last**.
- Generated files are regenerated, never hand-edited. Append-only rows are never rewritten.
- Sub-agents write only their own tag-file. The orchestrator is the sole writer of shared board files.

## 10. Non-goals

- `PairPalette` (`ISS-PAIRPALETTE-IS-TWO-AXES-NOT-A-PAIR`).
- The classid header mint (Q3).
- The 144-verb basis.
- Vocabulary coverage and per-work lemma codebooks.
- AUTO content cues.
- A compose table for pair codes.
- Retiring the old Cam96 names.
- Migrating paperless-rs.
- Embedding-side frequency effects.

## 11. Findings kept regardless of outcome

- The G0/G1 rows of every arm, including losers, are recorded in an `entries/` file.
- A KILL records the program's negative result, not a silence.

## 12. Escalated to the operator

- **Q1.** Is M the intended reading of `6 × (FisherZ:FisherZ)`? (It is your 2026-09-30 merge; confirmation is still asked.)
- **Q2.** M's similarity reads up to 4 cells per rail plus norms. May le-contract L4's "similarity = ONE table read" be amended?
- **Q3.** Where does the frequency/PoS header live, given the classid canon?
- **Q4.** The embedding key (§5 option a), supplied through the environment.
- **Q5.** If A, B or D beats M outside the noise floor, fork the spec to that reading, or hold?

## 13. Change ledger

### v2 → v3

| # | change | source |
|---|---|---|
| 1 | Reading resolved as the merge M; A/B/D recast as its ablations; decision table added; losers kept as findings (§11) | operator 2026-09-30; R2-§0, R2-§3, R2-§11 |
| 2 | Similarity is the exact Gram reconstruction cosine: it equals the baseline metric, uses the within-word cell, and makes diagonal mixing moot | operator merge; R2-§3/§4; S3-3 |
| 3 | G1's bar is the **in-harness** retrained 12-axis control; pinned figures are reference only | R1-§7 BLOCK |
| 4 | The PW1 loader returns a `Cam96PairCodes` view; no raw arrays | R1-§6 BLOCK, R2-§6 |
| 5 | Same-commit board-hygiene clause, generated-file rule, one-writer rule | R3-§9 BLOCK |
| 6 | G2 unit gate restored (falsifies F7); independent-seed report kept | R2-§7 |
| 7 | G3 is a component ablation with a noise floor; the full shuffle is kept report-only; G0 is its anti-vacuity partner | R2-§7 |
| 8 | G4 rewritten as a structural test (the v2 version was vacuous) | R2-§7 |
| 9 | G5b table-parity report (tests F11's transfer); F11 marked as a storm-probe source | R1-§3, R1-§4 |
| 10 | F1 marked as literal-frozen with a contested reading; F7 relabelled DECISION with its PR source | R2-§1, R1-§1 |
| 11 | F12 added (a static position must be stored) | operator question; `E-PERTURBATION-CONVERGENCE-1` |
| 12 | D as a "jointly trained tile" dropped as unsupported by its source; D kept as the table dual | R1-§3, R2-§3 |
| 13 | Harness (D-0) separated from the trainer (D-4) | R2-§9 |
| 14 | `Cam96AxisSpace` alias added; the breakage justification downgraded | R2-§12, R1-§6.2 |
| 15 | §6.3: the ClassView key is the owning NodeRow's classid (verified at `canonical_node.rs:1045-1078`); `ValueSchema` variant vs new axis goes to the envelope auditor; literal count corrected to 12+2; EMPTY semantics corrected; producer claim scoped to lance-graph/crates | R2-§6, R1-§6.3 |
| 16 | Codec move: `fisher_z.rs` re-exports at the module path; `build_from_palette` becomes a free function (E0116); the atanh clamp is factored out (a contract change); the "~100 lines" estimate removed | R1-§8, R1-§9 |
| 17 | D-C96P-6 keeps `SemanticSpace` exported, with no deprecation | R2-§9 |
| 18 | §2 wording: "independently quantized halves of one slice"; `SemanticSpace` has no live caller; the general-vocab quote is labelled; README figures are recorded outputs | R1-§0, R1-§2 |
| 19 | §5 absolutes softened ("cannot" / "only way"); words vs vectors made explicit | R1-§5 |
| 20 | The key must never reach a CI log, a command line or an echo | R3-§5 |
| 21 | "Jina" ruled exempt as data provenance (an embedding model, not an assistant identifier; CLAUDE.md's model registry uses it), written as "the reference embedding (Jina-v3)" | R3-§2 |
| 22 | Compose wording: "the analytic codec does not provide compose", not "undefined" | R1-§4 |
| 23 | Q5 added | R2-§11 |
| 24 | Basin gate line numbers corrected to `:186`, `:225`, `:263` | R1-§8 |
| — | **Not adopted:** R2's G7/G8 wording ("report-only yet gates D-8") is answered by the legacy-read-mode fallback, not by making them hard gates. Nothing measures a threshold for them yet, and a hard gate without a measured bar would be a policy pin posing as a finding | R2-§7 |

### v1 → v2

See `deepnsm-v2-cam96-pairwise-v2.md` §12.
