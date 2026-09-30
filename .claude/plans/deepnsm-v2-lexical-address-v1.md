# deepnsm-v2-lexical-address-v1 — a word is a 16-bit address into a versioned, baked COCA codebook

> **Status:** PROPOSAL (D-LXA-1..4). Plan only; no code is authorized by this file.
> **Council:** 5+3, ratified v3 (2026-09-30). The change ledger is §7. §3.1 carries one
> **operator escalation** (the alpha-channel fit). D-LXA-3 does not start before it is ruled.
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
> `ISS-CE64-EMIT-INVERSE-BIT2-DISAGREE`, `ISS-LXA-ALPHA-FIT` (the §3.1 escalation).

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

**Prior art: `WordId` is a different key, and it stays.** `deepnsm-v2` already has a
16-bit id: `WordId = u16` in `PaletteVocab` (`vocab.rs:26,47`). It identifies a **surface
form**: `PaletteVocab` is a list of words, readings hang off it (`LexicalReading.form_count`
counts "occurrences of THIS surface form", `lexical.rs:116`), and a lemma "is NOT a
WordId" (`lexical.rs:93`). `LexicalAddress` identifies a **reading**, i.e. a `(lemma, PoS)`
identity. They are two keys:

| type | identifies | keys | exists |
|---|---|---|---|
| `WordId` | a surface form | the surface-form table (`f`) | yes (`vocab.rs:26`) |
| `LexicalAddress` | a `(lemma, PoS)` reading | the identity table | new (D-LXA-1) |

They are joined by `readings(WordId) → [LexicalAddress]`. `WordId` is not wrapped and not
superseded.

`WordId`'s basin/identity split (`vocab.rs:7-21,31-40`) is **not** reading (2). It is a
positional split of one exact id: ids are assigned in frequency order, so the high byte
is a frequency band. The module doc itself says semantic distance "never [comes] from the
id arithmetic itself" (`vocab.rs:19-21`). §0's "the two bytes carry no separate meaning"
is a rule for `LexicalAddress`. It does not govern `WordId`, and nothing here edits
`vocab.rs`.

---

## §2 — Three reference sets, never interchangeable (F23, measured)

An address means something only against **one named reference**. There are three. They
are neither nested nor aligned, measured over the committed CSVs and the Tigris artifact:

| reference | source | rows | distinct keys | key |
|---|---|---|---|---|
| `COCA4096` | `crates/deepnsm/word_frequency/word_rank_lookup.csv`, ranks ≤ 4096 | 4,096 ranks | 3,559 words | rank: one row per (word, PoS), every rank unique, so a homograph occupies **several** ranks (`to` has two rows, at ranks 6 and 12) |
| `COCA5K_LEMMA` | `lemmas_5k.csv` | 5,050 | 4,380 lemmas / 5,050 (lemma, PoS) | (lemma, PoS) |
| `COCA20K_ACAD` | `academic_20k.csv`; Tigris `lance-graph/codebooks/deepnsm-v2-academic-coca-v1/` | 20,845 | 18,559 words / 20,842 (word, Pos) | `word_id` = admission order |

- 5k ∩ 20k = 4,895 of 5,050 (lemma, PoS) and 4,264 words. **116 of the 5k lemmas are
  absent from the 20k.**
- 4096 ∩ 20k = 3,462 of 3,559 words.
- **Only 4 of the 4,264 shared words carry the same ordinal in the 5k and the 20k**:
  `the`, `there`, `care` and `wage`. Ordinals are 0-based first-occurrence order, and
  words are matched exactly. ⊘ #1303 reported "3": that count dropped ordinal 0 (`the`),
  a falsy-zero error; `PaletteVocab` treats id 0 as valid. Lower-casing both lists moves
  the shared count to 4,318 (lower-casing only the 20k gives 4,316). The aligned count
  stays 4 either way.
- Each set handles homographs differently:
  - the 20k collapses same-word-different-Pos rows into one id. 2,286 rows collapse
    away: 2,283 extra `(word, Pos)` keys (20,842 − 18,559) plus 3 exact duplicate rows
    (20,845 − 20,842: `wastewater/n`, `disproportionately/r`, `instill/v`);
  - the 4096 gives each (word, PoS) its own rank, so a homograph spans several ranks;
  - the 5k keeps each (lemma, PoS) distinct.
- The Tigris "academic codebook" is a vocabulary carve (`PaletteVocab::from_frequency_ranked`),
  not a trained centroid codebook. Its CSV's sha256 equals the committed file's.

**What an address resolves to depends on the reference's own key:**

| reference | an address resolves to | exactly one (lemma, PoS)? |
|---|---|---|
| `COCA4096` | one rank = one `(word, PoS)` row | yes; ranks are unique per `(word, PoS)` |
| `COCA5K_LEMMA` | one `(lemma, PoS)` row | yes |
| `COCA20K_ACAD` (Tigris carve v1) | one `word_id` = one **word**; the carve merged 2,283 extra `(word, Pos)` keys and 3 duplicate rows | **no.** It resolves to `(word, Ambiguous{PoS set})` |

G-LEX checks each reference against **its own** declared key. It never demands a
(lemma, PoS) from a reference that does not carry one. A PoS-exact academic reference
is possible: its 20,842 distinct `(word, Pos)` rows fit in 16 bits. It would be minted
as a **new reference version**, not by reinterpreting the carve (§5).

**The rule:**
- A reading contract names `ReferenceSet { id, version, sha256, key_kind }`, where
  `key_kind ∈ {SurfaceKeyed, IdentityKeyed}`. `COCA20K_ACAD` (a `PaletteVocab` carve) is
  surface-keyed. `COCA4096` and `COCA5K_LEMMA` are identity-keyed. The surface-form table
  of §3 exists only for a surface-keyed reference, or through the correspondence map.
- The facet bytes carry **no** reference id and no version byte (F5: content-blind). The
  reference is carried by the facet's **classid**, through the ClassView's reading mode.
  That reading mode pins the **whole** `(id, version, sha256)`. A new codebook version
  therefore means a new classid or reading mode, and a facet written under v1 and read
  under v2 is refused.
- An ordinal is stable **within** one reference, never across references.
- Correspondence between references is an explicit (lemma, PoS) map with
  `Ambiguous{n}` and `Missing` as values. It is never assumed to be nested or aligned.
- Reading an address against the wrong reference is **refused**. It is never answered
  with a different word.
- **The three resolution shapes stay visible in the type.** One `LexicalAddress` type
  resolves to `Reading::Exact(word|lemma, PoS)` or `Reading::Ambiguous(word, PoS set)`.
  Every caller must match both arms; there is no accessor that returns a single PoS
  without handling `Ambiguous`.

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
| identity: **`lemma_evidence`** (u8) | `w / (w+1)`, `w = ln(1+freq) / ln(1+freq_max) · K` (`K` is a labelled, hand-tuned pin) | the amount of evidence behind the reading as a whole. It describes the lemma, not any one surface form's split, so it is **not** the `c` of a truth pair with `f` |
| surface form: **`f`** (u8) | the reading's **listed-reading share**: its share among the readings the source lists for that surface form, from `word_forms.csv` (`wordFreq`) via `LexicalReading.form_count`, not the floored cumulative `coverage` (`lexical.rs:33-39,377-383`). COCA lists are truncated, so `f = 1` means "the only listed reading", not "unambiguous". If any listed reading has no count (`form_count = None`, `lexical.rs:49-50`), the form's share is undefined and the form is marked unknown, never skipped: e.g. surface `record` → `record/n` 120,048 vs `record/v` 13,014, so `f` ≈ 0.90 and ≈ 0.10. `f = 1` for an unambiguous form (`the`) | an ambiguous form gets `f < 1` without anyone counting at query time. The per-form reading list already exists (`LexicalReading.form_count`, `lexical.rs:111-118`; `readings(id)`, `:373`); the bake computes the share from it. The reading-share key follows `deepnsm-v2-coverage-bands-v1` F8 |
| known / unknown | **one bit per row, outside the bytes** (see below and §3.1). Never a byte value and never `0` | zero would assert "no evidence"; an unset bit asserts "not measured". Same principle as `lexical.rs:46-52`, where unknown is `None`, not `0` |

**`f` is a prior, never a selector.** It never picks a reading. D-LXC-1 / D-LXC-11 found
that choosing a tag by frequency is interference. The driver selects the reading; `f` is
only the prior evidence handed to it.

**This is not a NARS truth, and it does not reuse `TruthU8`.** `f` answers "given this
surface form, how often is it this reading?". `lemma_evidence` answers "how much evidence
is there for this reading at all?". They are two statements with two evidence masses, on
two tables. `TruthU8 { frequency, confidence }`
(`lance-graph-arm-discovery/src/translator.rs:57-63`) is a truth about **one** statement,
and its encoding differs from the one below in three ways:
- it quantizes by floor division, `(x*255)/n` (`:77,81`);
- it uses `128` as an "unknown ≈ 0.5" sentinel (`:74-75`);
- it is documented as a substrate value, "not a wire DTO", with no LE codec (`:43-49`).

D-LXA-3 defines its own byte encoding.

`lexical.rs` counts evidence and states "Not truth … no NARS truth" (`lexical.rs:30-31`).
The bake is a **separate, derived artifact** that crosses that line on purpose, offline.
`lexical.rs` itself is unchanged.

**The canonical `u8` encoding**, so the byte-for-byte gate is reproducible:
- `f` and `lemma_evidence` are in `[0, 1]`. Each is stored as `q = round_half_even(x × 255)`, computed
  in `f64`. That uses the full unsigned range `0..=255`, and every byte value is a number.
- **There is no sentinel, and no row is left out.** An unsigned byte carries no
  NaN-like "unknown" value. A row whose value is unknown still holds a canonical fill
  byte, `0`, so the byte-for-byte gate is reproducible. The fill means nothing on its
  own: only the known bit says whether the byte is a value.
- **Measured or not is a bit outside the bytes** (operator ruling F4; the mechanism is
  subject to §3.1). The baked table is the **spine**: complete, one row per reference
  entry, read-only, read by every reader without a lock. Unknown is an **unset bit**. It
  is not a byte value and not a missing row. The table keeps its shape, so addresses
  stay dense and no reader ever handles a hole.
- There is **one known-mask per table**. The surface-form table and the identity table
  are known independently.
- The known bit is forced in the type, like `Exact` / `Ambiguous`. The prior accessor
  returns `Option<Prior>` (the same rule as `lexical.rs:46-52`: unknown is not zero). A
  reader cannot use `f` or `lemma_evidence` without having handled the unknown case.
- **How** the bit is carried is escalated (§3.1). The council found that the shipped
  alpha API does not fit the first wording of this section.
- `ln` is `f64::ln` and `freq_max` is the maximum over the reference being baked. `K`
  is written into the artifact header next to the three digests, so it is part of what
  is reproduced.

### §3.1 — Alpha-channel fit: ESCALATED to the operator (`ISS-LXA-ALPHA-FIT`)

**The ruling, verbatim** (operator, 2026-09-30, on known versus unknown): *"we already have
alpha channel split tunnel trick for that"*.

**What the code does.** `crates/lance-graph-contract/src/alpha.rs`, `alpha_tunnel.rs`:
- **The alpha channel is defined as not a bake.** It has "no bakes.tsv row, no digest",
  and it is discardable whole (`alpha.rs:11-16`, `:857-861`).
- **The split tunnel** (`alpha_tunnel.rs:12-18`) means every lane reads the baked spine
  without a lock, and writes go to an overlay at the same addresses. The overlay records
  where attention went, one lane per rung.
- **`AlphaOverlay` only sits over `NodeRow` tables.** It is built through
  `AlphaAllocation::over`, and only over `&[NodeRow]` (`alpha.rs:531`).
- **`claim` writes only a stamp** (`alpha.rs:683-716`). It has three outcomes:
  - a fresh write of an `AlphaStamp` (cycle, seq, rung, visits) into value slot 0;
  - on a revisit, `Ok { fresh: false }`, which increments `visits` in the existing row;
  - `Unallocated` for an address outside the allocation.

  It never writes a caller's value.
- **The overlay bit means "attended".** It never means "measured".
- **`AlphaMask` is a plain, length-checked bitset.** It has `contains` (`:263`), `and`
  (`:345`), `words` (`:478`) and a public constructor `from_words(words, len)` (`:492`).
  Its single-bit `set` is private (`:255`).
- A sealed per-cycle persistence rule exists only in a plan
  (`.claude/plans/spog-alpha-channel-v1.md`, its frozen decision F1; MedCare-side). It is not in this code.

**The conflict.** Read literally, "known versus unknown is an alpha-channel bit"
contradicts the alpha channel's own definition. "Known" is a baked, digested,
reproducible fact. An alpha bit is ephemeral, undigested, and means "attended". There is
a second conflict with F3: nothing is counted at runtime, so no runtime write could ever
turn a row from unknown into known. The split tunnel's write side has nothing to carry
for this question.

**Options:**
- **(a) A baked coverage mask.** The bake emits the table plus one bitset per table
  (known = 1), built with `AlphaMask::from_words`. This does **not** implement the split
  tunnel the ruling names: it keeps only the `AlphaMask` type and the baked-spine read
  side. The mask is part of the bake (digested), so it must not be called an alpha bit.
- **(b) Codebook entries as `NodeRow`s,** with the known flag or values in a value tenant
  after the stamp. This uses the real overlay, but it costs 512 B per word, needs a new
  tenant, and its bit still means "attended".
- **(c) Extend the alpha API** with a non-`NodeRow` overlay whose bit means "measured".
  This is a contract change that redefines alpha, and it needs its own plan.
- **(d) (a) plus the split tunnel for what it is for.** A baked coverage mask carries known
  versus unknown. The unchanged alpha overlay is kept as the **attention recorder** over
  the codebook spine: which lexical addresses the driver visited, per cycle, per rung.
  Its addresses come from a `NodeRow` projection of the codebook, or later from a
  non-`NodeRow` allocation. That is option (c) scoped to allocation only, with the
  meaning unchanged.

**Council recommendation: (d).** It keeps both legs of what the ruling points at, each
with the meaning its code gives it. It never stores a value in the overlay. The
refinement of `f` / `lemma_evidence` is never an overlay write (F3; `claim` is stamp-only).

**The question to the operator:** confirm that "known" is a baked coverage mask shaped
like `AlphaMask` (options a or d), and that the alpha overlay keeps its "attended"
meaning.

`range` and `disp` are left out. They are collinear with frequency, and `disp` measures
evenness across genres, not evidence.

At runtime the prior is **two reads**, never a computation: `f` from the surface-form row
that routed the token (keyed by `WordId`), and `lemma_evidence` from the identity row of
each candidate reading.

**Where the identity rows come from depends on the reference's key kind:**
- **Identity-keyed reference** (`COCA4096`, `COCA5K_LEMMA`). An address resolves to one
  `(word|lemma, PoS)` row, and that row holds `lemma_evidence`. It is one read.
- **Surface-keyed reference** (`COCA20K_ACAD`). An address resolves to
  `(word, Ambiguous{PoS set})`, so it selects **no single identity row**, and the
  reference has no identity table of its own. Each candidate `(lemma, PoS)` in the set is
  looked up through the correspondence map (D-LXA-2) in an identity-keyed reference. That
  gives one `lemma_evidence` per candidate reading, or unknown for a `Missing` candidate.
  The values are **never aggregated** into one number for the word; picking among them is
  the driver's job, not the bake's (f is a prior, never a selector).
Frequency rank stays a routing signal, not meaning (ρ ≈ −0.07, archive F11).

---

## §4 — Deliverables, in build order

| D-id | what | gate (can fire / can stay silent) |
|---|---|---|
| **D-LXA-1** | `LexicalAddress` newtype over `u16` (a **new** identity key beside the surface-form `WordId`, joined by `readings(WordId) → [LexicalAddress]`, §1) plus `ReferenceSet { id ∈ {COCA4096, COCA5K_LEMMA, COCA20K_ACAD}, version, sha256 }`, in `deepnsm-v2`. Resolving through the wrong reference is a refusal. No bare `[u8; 12]` or `u16` enters the path | **G-LEX** (over typed `LexicalAddress` values; nothing binds raw facet bytes to a reference before D-LXA-4): every declared entry of a reference resolves to exactly its declared reading **under that reference's own key** (§2 table): `(word, PoS)`, `(lemma, PoS)`, or `(word, Ambiguous{PoS set})`. A cross-reference read is refused. A `compile_fail` test proves a bare `u16` is not accepted |
| **D-LXA-2** | Generator for `lexical_correspondence.tsv`: one row per (lemma, PoS) with `id4096 \| id5k \| id20k \| status ∈ {Exact, Ambiguous{n}, Missing}`, the three digests in its header. Generated, never hand-edited | **G-REF:** re-deriving the file must reproduce §2's numbers exactly, including "4 of 4,264" **with ordinal 0 counted** (a falsy-zero generator must fail it). Hand-editing one ordinal must turn the check red |
| **D-LXA-3** | The COCA bake of §3: per reference, an identity table (`lemma`, `PoS`, `lemma_evidence`) for each identity-keyed reference and a surface-form table (`form`, reading, `f`), all `u8`, with the known bit carried as ruled in §3.1. A surface-keyed reference reaches `lemma_evidence` per candidate reading through the D-LXA-2 correspondence map, never as one aggregated value (§3). `lemma_evidence` is baked, never derived from a runtime rung (a rung is not reproducible). **Blocked on §3.1 and on D-LXC-4** | Re-deriving must reproduce the bake byte for byte. Unknown rows hold fill byte `0` with the known bit unset. A form with one `form_count = None` reading must come out unknown (fires); the all-listed fixture must not (stays silent). Pinned rows: surface `the` → `the/a` with `f = 1`; surface `record` → `record/n` with `f < 1` and `record/v` with `f > 0`, the two summing to 1 within quantization. An all-unambiguous fixture must give `f = 1` everywhere (stays silent) |
| **D-LXA-4** | The six-slot reading: a ClassView-selected reading of a 12-byte facet as six `LexicalAddress`es under one named `ReferenceSet`, carried by the classid. A register in any other shape is refused, never reinterpreted. **This is a contract change, gated on its own contract plan (not yet written).** Today no reader can refuse: `SpoFacet::from_register` takes a bare `[u8; 12]` (`awareness_facet.rs:106`), and `Cam96 = [u8; 12]` (`space.rs:163`) appears 68 times in 12 files (grep, counting doc comments). `ReadMode` / `ValueSchema` must first gain a lexical reading. **Existing `Cam96` / `SpoFacet` classids keep their current reading unchanged. The lexical reading exists only under a newly minted classid / reading mode, and nothing re-reads existing rows** (I-LEGACY-API-FEATURE-GATED). D-LXA-1..3 ship without it | A facet written under reference X and read under Y is refused. A facet written under `(X, v1)` and read under `(X, v2)` is refused. Rotating the six slots changes the resolved words (this proves the six positions are ordered, not a bag). These three gates move verbatim into the contract plan; the STATUS_BOARD row carries them until it exists |

**Collision:** D-LXA-3 reads `academic_20k.csv`, which `D-LXC-4` (the academic loader,
currently Blocked on a ruling about three duplicate (word, PoS) pairs) also owns. D-LXA-3
waits for that ruling, or takes it as its own first question. It must not duplicate the
loader.

**Ownership:** the baked tables and coverage masks are read-only artifacts with no
mailbox. Nothing writes them at runtime. If §3.1 rules an alpha overlay over the codebook
(options b, c or d), the overlay's writer is the owning mailbox
(`SoaEnvelope::mailbox_owner`); a consumer never writes as itself.

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
**CONTRADICTS**. Emitting a mask and inverting it therefore disagree on bit 2. The
council confirmed each site: `driver.rs:489`, `p64-bridge/src/lib.rs:123-124`
(`SUPPORTS = 2`, `CONTRADICTS = 3`) and `:74-75`. It found one more writer: inference
type 1 also sets SUPPORTS (`lib.rs:81`). `driver.rs:706-720` keeps its own local
predicate-bit table, which the p64-bridge fix must also route through the one mapping.

This plan does not depend on the defect. It is recorded because it is real and was found
in the same read. The fix belongs to `p64-bridge`: one named mapping used in both
directions, plus a round-trip test that can fail on bit 2.

---

## §7 — 5+3 council change ledger (v1 → draft v2)

The five were prior art, iron rules, code truth, cascade impact and different views.
Code truth reproduced every §2 number from the CSVs (gate G1).

| # | change | source |
|---|---|---|
| L1 | 2,286 split into 2,283 extra `(word, Pos)` keys + 3 duplicate rows | code truth · CodeRabbit |
| L2 | `WordId` / `PaletteVocab` named as prior art | prior art |
| L3 | The reference id lives in the classid (ClassView reading mode), never in the bytes | iron rules · different views |
| L4 | The `Exact` / `Ambiguous` arms are forced in the type | different views |
| L5 | `f` is a prior, never a selector (D-LXC-1/-11) | prior art |
| L6 | `form_count` / `readings` and the coverage-bands F8 key cited | prior art · cascade |
| L7 | The alpha-overlay wording is withdrawn and **escalated** (§3.1): claim is stamp-only, typed over `NodeRow`, ephemeral, means "attended"; it also conflicts with F3 | code truth · iron rules · different views |
| L8 | Mailbox-owner sentence added | iron rules |
| L9 | D-LXA-4 recast as a contract change on its own plan; 1–3 ship without it | cascade |
| L10 | `c` is never derived from a runtime rung | different views |
| L11 | Bit 2: a second SUPPORTS writer (inference type 1) recorded | code truth |

### v2 → v3 (the three reviewers: overclaim-auditor, dilution-collapse-sentinel, firewall-warden)

Verdicts: 1 BLOCK (§3.1, dilution), 22 FIX, the rest PASS. All resolved here.

| # | change | source |
|---|---|---|
| R1 | Two keys, two types. `WordId` identifies a surface form and keys `f`. `LexicalAddress` identifies a reading. Draft v2's "wrap or supersede `WordId`" and its `vocab.rs` supersession note are **withdrawn**: the byte split is positional, not reading (2) | dilution · overclaim · firewall |
| R2 | `ReferenceSet` gains `key_kind`. The ClassView reading mode pins `(id, version, sha256)`, with a version-mismatch refusal gate | dilution |
| R3 | `c` renamed `lemma_evidence`. `(f, lemma_evidence)` is not a truth pair and does not reuse `TruthU8`; the three encoding differences are named | dilution · overclaim |
| R4 | The bake is a derived artifact crossing `lexical.rs`'s no-truth line | overclaim |
| R5 | `f` is the listed-reading share, from `form_count`. A `None` count makes the form unknown, with a two-sided fixture | dilution |
| R6 | Canonical fill byte `0`, one known-mask per table, and `Option<Prior>` forcing the unknown case | overclaim · dilution |
| R7 | §3.1 rewritten: the ruling quoted; alpha is defined as not a bake; the three `claim` outcomes; the sealed-batch sentence moved to its plan source; option (a) described as not a split tunnel; option (d) added and recommended; the conflict stated plainly (**the BLOCK**) | dilution · overclaim |
| R8 | D-LXA-4: legacy-reading invariant stated; `Cam96` count corrected to 68 in 12 files; gates carried on the STATUS_BOARD row until the contract plan exists; G-LEX scoped to typed values | firewall · overclaim · dilution |
| R9 | §2 wording: rank positions for `to`; 4,316 vs 4,318; the three duplicate rows named | overclaim |
| R10 | Bit-2 sites cited; `driver.rs:706-720` local table named | overclaim |
| R11 | Board: `ISSUES` row `ISS-LXA-ALPHA-FIT`, STATUS_BOARD rows updated, bit-2 issue extended | firewall |

**Not adopted:** none. The one frozen-decision conflict (F4 against the alpha channel's
own definition) is escalated (`ISS-LXA-ALPHA-FIT`), not overridden.
