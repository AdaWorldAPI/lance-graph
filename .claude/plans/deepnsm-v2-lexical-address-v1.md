# deepnsm-v2-lexical-address-v1 — a word is a 16-bit address into a versioned, baked COCA codebook

> **Status:** PROPOSAL (D-LXA-1..4). Plan only; no code is authorized by this file.
> **Written against:** `main` `0d31c54f` (2026-09-30).
> **Harvested from:** PR #1303 (`deepnsm-v2-cam96-pairwise-v5`), which was **closed without
> merging**. Only four things from it are kept here:
> - its CORRECTION 2 (the lexical-address design);
> - its F23 three-reference measurement;
> - its COCA prior, recast as a bake;
> - one verified code defect.
>
> Everything else stays on branch `claude/brave-mayer-65y3cy` (head `d6c041d7`) and is
> **not** carried: the embedding-subspace design and its gates (already retracted in v5
> itself), the §11/§11R execution socket, and the "baton" doc-comment edits.
> **Board:** `STATUS_BOARD.md` § deepnsm-v2-lexical-address · entry
> `entries/2026-09-30-three-reference-sets-are-not-ordinal-aligned.md` · `ISSUES.md`
> `ISS-CE64-EMIT-INVERSE-BIT2-DISAGREE`.

---

## §0 — In plain words

A facet's 12 payload bytes hold **six words**. Each word is a pair of bytes `[a,b]`,
read together as one **16-bit address**. That address points at one row of an
**immutable, versioned lexical codebook**, and that row says exactly which word it is:
its lemma and part of speech.

- The two bytes carry no separate meaning. They are not two poles, not
  "nearest / second-nearest centroid", and not a slice of an embedding.
- **Relations between words are not in the bytes.** The cognitive shader driver
  supplies them, calibrated through the Fisher-z LUT. Fisher-z governs relation values;
  it never makes a word's identity fuzzy.
- **Frequency, PoS and lemma are baked into the codebook row**, computed once and
  offline. Nothing is counted at runtime.
- Embeddings are at most a comparison instrument. Reconstructing an embedding is **not**
  an acceptance criterion.

Operator, 2026-09-30:

> *"Six lexical slots, not six embedding subspaces. Each slot's two-byte address resolves
> through an immutable, versioned lexical codebook; the driver supplies the selected
> relational reading."*

---

## §1 — Two readings of `[a,b]`, and which one this is

| | reading | status |
|---|---|---|
| (1) | `[a,b]` **identifies a word**: an exact, unique 16-bit address | **the representation** |
| (2) | `[a,b]` names two lexical poles whose cell is a relation | **not** a unit of identity anywhere in this plan |

Earlier Cam96 drafts (v1–v5 on #1303) silently substituted (2) for (1), under the name
"nearest and second-nearest centroid". An artifact check found that nothing shipped,
released, or stored in Tigris ever assigns a word two needles:
- the COCA CAM-PQ gives six per word (one per subspace);
- bgz17 / bgz-tensor `nearest`/`assign` give one.

The phrase originated in #1303's v1 with no source.

---

## §2 — Three reference sets, never interchangeable (F23, measured)

An address means something only against **one named reference**. There are three. They
are neither nested nor aligned, measured over the committed CSVs and the Tigris artifact:

| reference | source | rows | distinct keys | key |
|---|---|---|---|---|
| `COCA4096` | `crates/deepnsm/word_frequency/word_rank_lookup.csv`, ranks ≤ 4096 | 4,096 ranks | 3,559 words | rank: one row per (word, PoS), every rank unique, so a homograph occupies **several** ranks (`to/t` = 6, `to/i` = 12) |
| `COCA5K_LEMMA` | `lemmas_5k.csv` | 5,050 | 4,380 lemmas / 5,050 (lemma, PoS) | (lemma, PoS) |
| `COCA20K_ACAD` | `academic_20k.csv`; Tigris `lance-graph/codebooks/deepnsm-v2-academic-coca-v1/` | 20,845 | 18,559 words / 20,842 (word, Pos) | `word_id` = admission order |

- 5k ∩ 20k = 4,895 of 5,050 (lemma, PoS) and 4,264 words. **116 of the 5k lemmas are
  absent from the 20k.**
- 4096 ∩ 20k = 3,462 of 3,559 words.
- **Only 4 of the 4,264 shared words carry the same ordinal in the 5k and the 20k**:
  `the`, `there`, `care` and `wage`. Ordinals are 0-based first-occurrence order, and
  words are matched exactly. ⊘ #1303 reported "3": that count dropped ordinal 0 (`the`),
  a falsy-zero error; `PaletteVocab` treats id 0 as valid. Lower-casing the 20k words
  moves the shared count to 4,318 and leaves the aligned count at 4.
- Each set handles homographs differently:
  - the 20k collapses 2,286 same-word-different-Pos duplicates into one id;
  - the 4096 gives each (word, PoS) its own rank, so a homograph spans several ranks;
  - the 5k keeps each (lemma, PoS) distinct.
- The Tigris "academic codebook" is a vocabulary carve (`PaletteVocab::from_frequency_ranked`),
  not a trained centroid codebook. Its CSV's sha256 equals the committed file's.

**What an address resolves to depends on the reference's own key:**

| reference | an address resolves to | exactly one (lemma, PoS)? |
|---|---|---|
| `COCA4096` | one rank = one `(word, PoS)` row | yes; ranks are unique per `(word, PoS)` |
| `COCA5K_LEMMA` | one `(lemma, PoS)` row | yes |
| `COCA20K_ACAD` (Tigris carve v1) | one `word_id` = one **word**; the carve merged 2,286 same-word-different-PoS rows | **no.** It resolves to `(word, Ambiguous{PoS set})` |

G-LEX checks each reference against **its own** declared key. It never demands a
(lemma, PoS) from a reference that does not carry one. A PoS-exact academic reference
is possible: its 20,842 distinct `(word, Pos)` rows fit in 16 bits. It would be minted
as a **new reference version**, not by reinterpreting the carve (§5).

**The rule:**
- A reading contract names `ReferenceSet { id, version, sha256 }`.
- An ordinal is stable **within** one reference, never across references.
- Correspondence between references is an explicit (lemma, PoS) map with
  `Ambiguous{n}` and `Missing` as values. It is never assumed to be nested or aligned.
- Reading an address against the wrong reference is **refused**. It is never answered
  with a different word.

---

## §3 — The COCA bake (frequency, PoS and lemma built in; nothing counted at runtime)

There are **two baked tables**, because the two parts of the prior are keyed
differently:
- the **identity table**, one row per `(lemma, PoS)`, which the address resolves to;
- the **surface-form table**, one row per `(surface form, reading)`. The reading share
  `f` belongs here, because it is a property of a *surface form*, not of a lemma. The
  same `record/v` lemma row has a different share for `record`, `recorded` and
  `recording`, so baking `f` onto the lemma row would make it depend on which form was
  picked at build time.

Every field is computed **once, offline**, when the artifact is built. Float arithmetic
is allowed at build time, which is the derived side of the no-float rule. The results
are stored quantized as `u8`:

| field | how it is computed | why |
|---|---|---|
| identity: `lemma`, `PoS` | from `lemmas_5k.csv` (or the reference's own key) | the declared reading the address resolves to |
| identity: **`c`** (u8) | `c = w / (w+1)`, `w = ln(1+freq) / ln(1+freq_max) · K` (`K` is a labelled, hand-tuned pin) | the amount of evidence behind the reading |
| surface form: **`f`** (u8) | the reading's share among that surface form's readings, from `word_forms.csv` (`wordFreq`): e.g. surface `record` → `record/n` 120,048 vs `record/v` 13,014, so `f` ≈ 0.90 and ≈ 0.10. `f = 1` for an unambiguous form (`the`) | an ambiguous form gets `f < 1` without anyone counting at query time |
| unknown | a missing count stays **unknown**, never `0` | zero would assert "no evidence"; unknown asserts "not measured" |

**The canonical `u8` encoding**, so the byte-for-byte gate is reproducible:
- `f` and `c` are in `[0, 1]`. Each is stored as `q = round_half_even(x × 254)`, computed
  in `f64`, which gives `0..=254`.
- **`255` is the UNKNOWN sentinel.** No computed value can produce it, so unknown needs
  no separate flag byte and can never be confused with `x = 1` (`254`).
- `ln` is `f64::ln` and `freq_max` is the maximum over the reference being baked. `K`
  is written into the artifact header next to the three digests, so it is part of what
  is reproduced.

`range` and `disp` are left out. They are collinear with frequency, and `disp` measures
evenness across genres, not evidence.

At runtime the prior `<f, c>` is **two reads**, never a computation: `f` from the
surface-form row that routed the token, `c` from the identity row it resolved to.
Frequency rank stays a routing signal, not meaning (ρ ≈ −0.07, archive F11).

---

## §4 — Deliverables, in build order

| D-id | what | gate (can fire / can stay silent) |
|---|---|---|
| **D-LXA-1** | `LexicalAddress` newtype over `u16` plus `ReferenceSet { id ∈ {COCA4096, COCA5K_LEMMA, COCA20K_ACAD}, version, sha256 }`, in `deepnsm-v2`. Resolving through the wrong reference is a refusal. No bare `[u8; 12]` or `u16` enters the path | **G-LEX:** every declared entry of a reference resolves to exactly its declared reading **under that reference's own key** (§2 table): `(word, PoS)`, `(lemma, PoS)`, or `(word, Ambiguous{PoS set})`. A cross-reference read is refused. A `compile_fail` test proves a bare `u16` is not accepted |
| **D-LXA-2** | Generator for `lexical_correspondence.tsv`: one row per (lemma, PoS) with `id4096 \| id5k \| id20k \| status ∈ {Exact, Ambiguous{n}, Missing}`, the three digests in its header. Generated, never hand-edited | **G-REF:** re-deriving the file must reproduce §2's numbers exactly, including "4 of 4,264" **with ordinal 0 counted** (a falsy-zero generator must fail it). Hand-editing one ordinal must turn the check red |
| **D-LXA-3** | The COCA bake of §3: per reference, an identity table (`lemma`, `PoS`, `c`) and a surface-form table (`form`, reading, `f`), all `u8`, unknown flagged | Re-deriving must reproduce the bake byte for byte. Pinned rows: surface `the` → `the/a` with `f = 1`; surface `record` → `record/n` with `f < 1` and `record/v` with `f > 0`, the two summing to 1 within quantization. An all-unambiguous fixture must give `f = 1` everywhere (stays silent) |
| **D-LXA-4** | The six-slot reading: a ClassView-selected reading of a 12-byte facet as six `LexicalAddress`es under one named `ReferenceSet`. A register in any other shape is refused, never reinterpreted | A facet written under reference X and read under Y is refused. Rotating the six slots changes the resolved words (this proves the six positions are ordered, not a bag) |

**Collision:** D-LXA-3 reads `academic_20k.csv`, which `D-LXC-4` (the academic loader,
currently Blocked on a ruling about three duplicate (word, PoS) pairs) also owns. D-LXA-3
waits for that ruling, or takes it as its own first question. It must not duplicate the
loader.

---

## §5 — Open

- **Where the six slots live:** the row's second facet (bytes 16..32) or a new 16-byte
  `Identity` value tenant. Not decided. Either choice is additive and leaves
  `NODE_ROW_STRIDE` unchanged.
- **Which relational reading(s) the driver selects** between two resolved words, and the
  LUT that carries them. That belongs to the ClassView and is not decided here.
- **Does `COCA20K_ACAD` get its own codebook**, or does it share `COCA4096`'s through the
  correspondence map? And is a PoS-exact academic reference (20,842 `(word, Pos)`
  rows) minted as a new version beside the carve?

---

## §6 — The verified defect carried with this plan

`ISS-CE64-EMIT-INVERSE-BIT2-DISAGREE` (`ISSUES.md`). The driver packs the p64
predicate-plane byte straight into a `CausalMask`: `driver.rs`, emission stage,
`CausalMask::from_bits(h.predicates & 0x07)`. In that byte, bit 2 is **SUPPORTS**
(`p64-bridge`: `SUPPORTS = 2`). But `edge_to_layer_mask` maps mask bit 2 to
**CONTRADICTS**. Emitting a mask and inverting it therefore disagree on bit 2.

This plan does not depend on the defect. It is recorded because it is real and was found
in the same read. The fix belongs to `p64-bridge`: one named mapping used in both
directions, plus a round-trip test that can fail on bit 2.
