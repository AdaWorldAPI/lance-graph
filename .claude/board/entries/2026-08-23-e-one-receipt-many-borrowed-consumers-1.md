## 2026-08-23 — E-ONE-RECEIPT-MANY-BORROWED-CONSUMERS-1 — the integration half of #1012: one versioned tokenization drives Tantivy, DeepNSM-v2 and a forward surface with zero re-tokenization, and the four gaps that stand between that and a carrier

**Status:** FINDING — [MEASURED] (`PROBE-TOKEN-SEAM-1`, 37 gates, **13
disable-runs each verified red-then-green**; two committed real corpora — the
in-tree KJV Genesis scene `PROBE-TOKEN-BPE-GEOMETRY-1` used, carried verbatim so
the numbers are comparable, and Project Gutenberg's *Alice* split into 300
paragraphs, 75 514 B). The probe lives in `AdaWorldAPI/paperless-rs`
(`crates/paperless-token`, `docs/TOKEN-SEAM-ARCHITECTURE.md`) because it needs a
Tantivy dependency this workspace does not carry; this entry records the
**lance-graph-side** findings.
**This closes the integration half** of
`E-TOKEN-BPE-CAN-FIT-NOT-YET-BUY-1` (#1012). That verdict measured that BPE FITS
the `6×(8:8)` geometry and refused to buy a carrier. It did not ask whether one
tokenization can SERVE several consumers at once. It can.
**Confidence:** High for what is measured at these two corpus scales; the 8-bit
lane saturated at 75 KB, so nothing here is a scale claim.

**Headline.** ONE tokenization per span drove all three consumers and each added
**zero** further tokenizations — not by discipline but by construction. Totals:
313 source tokenizations for 308 spans plus 5 deliberate fixtures, and 1 QUERY
tokenization on a deliberately separate counter (a query is different bytes;
folding it into one number would make the claim a lie).

**The two facts that make it adoptable, and neither was designed for this:**

1. **DeepNSM-v2's library is already tokenizer-free.** `parse_to_spo(&[Tagged])`
   consumes `(WordId, Pos)` pairs and touches no string; the
   `split_whitespace`/`normalise` logic lives ONLY in `examples/bible_wave.rs`
   and `examples/genre_shapes.rs`. The seam needed **no change to the crate**.
   The `(WordId, Pos)` boundary is the shipped seam and nobody had used it as
   one.
2. **Tantivy structurally cannot own offsets.** Its indexer reads `Token::text`
   and `Token::position`, uses `position_length` transiently, and reads
   `offset_from`/`offset_to` NOWHERE outside its own tests. Offsets are consumed
   only by snippet generation, which re-tokenizes the STORED text at query time.
   An index cannot become the ABI here even by accident.

**Measured, and the numbers are the point rather than the verdict:**

| | KJV scene | Alice |
|---|---|---|
| bytes / spans / tokens | 1 126 / 8 / 354 | 75 514 / 300 / 37 149 |
| compression | 3.18× | **2.03×** |
| distinct ids used | 137 | **247 of 255** |
| resident lane bytes | 832 (74 % of source) | 55 572 (74 %) |
| receipt share of resident | 54 % | 30 % |
| particles/span p50/p95/max | 4 / 8 / 8 | 8 / 30 / 43 |
| tokens per lexical unit p50/max | 1 / 7 | 2 / 15 |
| tokens straddling a word boundary | 30 | 58 |

- **The 8-bit lane SATURATES.** 247 of 255 ids on 75 KB of ordinary English, with
  compression already fallen from 3.18× to 2.03×. The canon's answer — the hi
  byte of each `(8:8)` pair as a PAGE lane, two separate bytes, never a widened
  `u16` — is **untested**. Until it is measured no scale claim about token BPE
  should be made, and this supersedes any reading of #1012's 3.35× as a
  corpus-independent figure.
- **The resident lane is ~74 % of the source text, not a fraction of it**, and
  at these span sizes **framing is 30–54 % of it**. A 56-byte receipt against
  12-byte particles means the RECEIPT's column layout matters more than the
  particle's. #1012 could not see this — it had no receipt.
- **Cardinality is not 1:1 in either direction**, so a BPE↔`WordId` projection
  is a real function, not a relabelling. Nothing in the seam assigns a `WordId`
  to a BPE token; that would be a second vocabulary wearing DeepNSM's
  coordinate system.
- **Byte offsets are DERIVED**, by prefix sum over a per-id decoded-length
  table. The receipt stores no offset column at all.

**Four lance-graph-side gaps, each named rather than worked around:**

1. **No shipped token continuation mechanism anywhere.** The nearest precedent
   in SHAPE is `rail_geometry::RailCarving::AxisSlab { reg, cont: Option<usize> }`,
   which chains one register to one continuation and caps at `RAIL_MAX_DEPTH = 24`
   levels — **below the measured p50 of 4 particles**, so it does not fit. The
   probe uses a contiguous run (`first_particle + particle_count + token_count`).
   Stated honestly there are TWO lawful framings and the trade is exact:
   `particle_count` alone bounds the run and a PAD scan inside that bound is
   already exact BECAUSE PAD is reserved (cost: one vocabulary slot, which at a
   255-cap that saturates is not free); or `token_count` costs 4 bytes and frees
   the slot for a full 256-id alphabet. What is unlawful is inferring the end
   from padding with no bound — measured, a lane-wide PAD scan overshoots
   receipt 0 by 10 tokens straight into receipt 1.
2. **`ValueTenant` has no token variant** (16 discriminants, none for text), so a
   lawful resident lane must implement `SoaEnvelope` or land as a new tenant.
   The probe's lane is a probe-local `Vec` and says so.
3. **There is no callable part-of-speech surface, and the reason is a decision
   already taken.** `coca_pos`/`archaic_pos`/`normalise` are byte-identical in
   BOTH deepnsm-v2 examples, above a comment stating `deepnsm_v2::lexicon` was
   DELETED after an audit found `lance-graph-planner`'s `insight_coca_read`
   already grounds it. That grounding does not reach a lean consumer:
   `insight_coca_read` is itself an **example binary**, in a crate carrying
   `serde`/`serde_yml`/`tokio`/`ndarray`, and its master `lexicon.tsv` is absent
   from this checkout. The probe restated the twenty-line tagger rather than
   re-litigate the deletion — recorded so the next consumer has the evidence the
   audit did not.
4. **The semantic half is unexercised.** `cam96_codebook.bin` / `cam96_codes.bin`
   are release assets, absent here, so palette256² DISTANCE never ran. Only the
   lexical/grammar half was measured.

**Polars: refuted, and the honest form is weaker than the question invited.** A
sweep of nine checkouts found **zero** `polars` occurrences in any manifest or
source; every `DataFrame` mention is prose. `paperless-rs` and `tesseract-rs`
declare none of arrow/datafusion/lance/lancedb. There was nothing to remove. The
structured-evidence path is likewise not tabular algebra:
`lance-graph-arm-discovery` takes `Dataset { spec: FeatureSpec, rows:
Vec<Vec<u32>> }` — category-index rows against a schema.

**Method note, and it is the transferable part.** An independent vacuity audit of
the finished probe found FIVE holes: a gate asserting byte counts but never span
counts (the CRLF bug that collapsed 300 spans into 1 would have re-passed it), a
threshold true by construction, an assertion about a type signature rather than
behaviour, an unconditional prefix check, and an unexercised ASCII-vs-Unicode
whitespace divergence. All five are fixed; four gained their own disable-runs;
the fifth is bounded by a measured count of 0. Separately, TWO disable-runs were
themselves wrong first — one relaxed a knob that does not bind on the fixture,
one targeted a mechanism the gate did not actually rest on — and an early batch
reported "no failure" six times in a row because the probe binary path was wrong
and nothing ran. **A knob that does not bind is not a disable; a fixture's SHAPE
is part of a test's coverage; and a null result is a claim about the apparatus
until proven otherwise.**

**⚠ SELF-CORRECTION, same session, after being pointed at `ogar-doc-ir`.** Two
claims above were wrong and are corrected here rather than left standing:

1. **The seam invented an identity that already existed.** The first cut minted
   `source_id`/`span_id` integers. `ogar_doc_ir::DocIr` already answers all
   three questions a tokenization receipt asks — `content_sha256` for WHICH
   document, `(DocPage::number, Region::reading_order)` for WHICH span, and
   `Region::text` for the span's canonical text. The probe was re-cut to read
   them (`docir.rs`; gates `T-DOCIR` / `T-DOCIR-KEY` / `T-DOCIR-SPANS`, 41
   total, 18 disable-runs). Note the crate's own docs CORRECT its plan's first
   sketch on what that hash is: a **per-acquisition dedup key**, not a
   cross-retina identity — which is exactly the right reading for a receipt,
   because you tokenize bytes.
2. **"The OCR boundary supplies no byte offsets" is RETIRED as a gap.** It
   supplies no PAGE-wide offset and does not need to: a region owns its text,
   so an offset is region-local, and `ogar-from-docv1::region_text` is where the
   `leading_space`-aware join already happens. What remains is far smaller — a
   sub-region span needs a non-zero `byte_from`, which the receipt already
   carries and no producer emits.

Also corrected: the `247 of 255` figure quoted above is the count of ids
APPEARING in the lane, not the vocabulary size. Measured, the trained table is
**full at 255/255** on Alice (and on the whole 170 KB file), and **180 of 255**
on the KJV fixture — where, as #1016's own record of that fixture says, the
CORPUS rather than the cap set it. The saturation conclusion holds and is
stronger; the number was the wrong quantity.

**Untouched by this.** HHTL is address geometry and BPE is tokenization —
#1012's measured refutation of the merge tree as a radix prefix partition
stands. Content never travels in classid: the contract id is a FIELD on the
receipt, gated by a grep of the library's own non-comment source.
