# token-value-tenant-v1 — a byte-exact span address inside the 40,767-triple stream

> **Status: PROPOSED — PLAN/BOARD ONLY. Measure-before-carve.** No tenant is
> minted, no code changes, until W1's numbers land (the STOP rule in §4).
> Operator directive (2026-08-28): *"you might need a token value tenant
> inside the 40k"* — the 40,767-triple KJV SPO stream.

## §0 The thesis, one paragraph

Every KJV-scale measurement this workspace has — 40,767 triples, 702
subjects, 35,613 pronoun subjects, the 27,788/7,237/588 SelectionalFit
split, the 49.1% cross-verse chains — was computed over `WordId`, which the
token seam measured at **67.7% byte round-trip** (`tesseract-paperless`
probe; a third of the source bytes do not survive the projection). `TokenId`
is **byte-exact** (`E-TOKEN-BPE-CAN-FIT-NOT-YET-BUY-1`). The tokenization
receipt was re-cut to **mint nothing** — it reads `content_sha256` +
`(DocPage::number, Region::reading_order)` + region-local byte offsets
(`E-ONE-RECEIPT-MANY-BORROWED-CONSUMERS-1`, corrected entry `460d78be`). What
does not exist is a place ON THE ROW where a triple carries the byte-exact
span it was read from. This plan carves that place — as an ADDRESS, never as
content — and re-runs the KJV evidence base over it.

## §1 What is established (verified at HEAD `1d7bc1b1`, not quoted from memory)

- **PROBE-TOKEN-BPE-GEOMETRY-1** (`f3820d84`, probe:
  `crates/lance-graph-planner/examples/probe_token_bpe_geometry.rs`):
  verdict **CAN-FIT, NOT YET BUY**, measured on 1,125 bytes (Genesis 2-3
  scene — the only real corpus in that checkout). Its four bounded claims
  are the ones W1 re-measures at scale: scoped vocab LOST to global (+19%
  tokens/chapter, n=2); vocab saturated at 180/255 ("the corpus set the
  vocab, not the cap" — while Alice measured FULL at 255/255, `460d78be`);
  overflow is the norm (p50=4, max=8 particles/verse); chapter Jaccard 0.32
  (n=1 pair). The scale corpora were "ABSENT — reported absent, never
  simulated."
- **The loci.rs lesson** (`68955ecb`): a resolver that pushes into a
  crate-local `Vec` "never reaches the SoA lane" and gets deleted; the
  shipped path writes through `WitnessLens::write_register` into the
  `NodeRow` value slab — zero-copy, escalate-never-clamp. The token seam's
  `TokenLane` is today exactly such a probe-local `Vec<[u8;12]>` and its own
  docs say so. Production population goes through a lens or it repeats the
  deleted defect.
- **The promoter's V1-mint defect** (same commit): new keys mint via
  `mint_for`, never `NodeGuid::new`.
- **The tarski register is HELD** (`.claude/plans/tarski-markov-hhtl-seam-v1.md`:
  "OPEN QUESTIONS ONLY. This file proposes no mechanism and licenses no
  work"; its earlier stream-order-bound-to-tree-address proposal is
  WITHDRAWN). This plan therefore claims NOTHING about folds, accumulation,
  or `Belief.rung` delegation — see §5.
- **Corpus identity pins** (`6e385c88`): 31,102 verses / 40,767 triples /
  702 subjects / 66 books / 32,357 TOC nodes. Any W-gate that reads the
  corpus asserts these before measuring.

## §2 Substrate facts the carve must obey (file:line at HEAD)

- `VALUE_SLAB_ROW_OFFSET = 32`, `VALUE_SLAB_LEN = 480`
  (`canonical_node.rs:815,817`). Current Full carve ends at row 252 =
  value-slab 220; **260 B free**. A new tenant appends at `row_offset: 252`,
  value-slab `[220, 220+N)`.
- **Discriminant 16**, and the `BoardAggregates` reservation RE-BASES to 17
  — the in-code rule at `canonical_node.rs:1061-1066`: a reservation "is
  re-based by ordinal position, not cancelled"; its offset "is DERIVED
  (`value_offset()`) and must never be written down as a literal again."
- **EXPERIMENTAL status, stated verbatim** — the `CausalWitness = 14`
  precedent: "not in the operator-locked §3 catalogue," appended
  additive / reserve-don't-reclaim, layout-preserving, **no
  `ENVELOPE_LAYOUT_VERSION` bump** (`NODE_ROW_STRIDE` unchanged).
- **Zero-fallback**: an all-zero lane reads *untokenized* — never a wrong
  span, never offset-zero-meaning-position-zero.
- **References, never content** — the `EpisodicBasin = 15` rail is the
  template ("32 B of REFERENCES, never of content"; the fat-concept failure
  §3a forbids). Text NEVER travels in the tenant; `u8:u8` never widened;
  refuse-never-fold on any tier that exceeds its width.
- **Field-isolation matrix tests are MANDATORY** (I-LEGACY-API-FEATURE-GATED)
  on any layout-touching change.

## §3 The two candidate carves — W1's numbers decide, not taste

Both are strawmen. Hard constraints both must satisfy: references-never-
content; zero-fallback; refuse-never-fold; additive at row 252; offset
derived, never literal.

**A — receipt-reference rail (32 B, `EpisodicBasin`-shaped). Favoured by
the measured overflow (p50=4/max=8 particles/verse: one 12-B slot cannot
carry a verse) and by cardinality (40,767 triples over 31,102 verses —
triples from one verse share ONE span, so the row carries a reference and
duplication is structural, not byte-copying):**

| bytes | field | what it is |
|---|---|---|
| `0..4` | `codebook_id` | `u32` registry ref — BPE codebook identity+version; **0 = null/untokenized** |
| `4..8` | `doc_ref` | `u32` registry ref to the receipt's `content_sha256` (32-B hash never inline) |
| `8..10` | `page` | `u16` (`DocPage::number`) |
| `10..12` | `region` | `u16` (`Region::reading_order`) |
| `12..16` | `byte_from` | `u32` region-local offset (the receipt's own coordinate) |
| `16..20` | `byte_len` | `u32` |
| `20..22` | `token_count` | `u16` guard — count > `u16::MAX` → REFUSAL, never truncation |
| `22..24` | reserved | zero |
| `24..32` | reserved | zero (RESERVE-DON'T-RECLAIM) |

**B — 4+12 facet + continuation (16 B, `CausalWitness`-shaped):** classid(4)
+ first particle (12 B), overflow chained. Survives ONLY if W1's at-scale
distribution collapses toward p50=1 — the fixture-scale measurement already
argues against it, but n=2 chapters is not the KJV.

## §4 Waves — model allocation declared up front (Opus filigree / Sonnet grind / Haiku contract-gated churn)

**W0 — corpus reachability (Sonnet; Haiku for re-runs).** `pg10.txt` (PG
#10, public domain) fetched ONCE, sha256-pinned, uploaded to Tigris
`s3://$AWS_S3_BUCKET_NAME/lance-graph/corpora/kjv/` with a `SHA256SUMS`
(the MedCare-rs bakes pattern already in that bucket). FIRST verify what
release `v0.1.0-cam96-data` already carries — no second source of truth for
anything the release already serves. Hydration in probes goes through
`lance-graph-hydrate::HydrationSource::from_env()` (reads the exact
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_ENDPOINT_URL`/
`AWS_DEFAULT_REGION`/`AWS_S3_BUCKET_NAME` set present; path-style; hard
`None` on a missing var) — **never a new fetcher** (operator directive
2026-08-17: the pattern is minted once, consumers "never re-implement it").
- F-TVT-0: flip one byte of a cached artifact → the hydration gate hard-fails
  (verified red-then-green, the `bench/fetch.sh` discipline).

**W1 — the scale re-measure (STOP GATE for everything below; Sonnet arms,
Opus adjudication).** Extend `probe_token_bpe_geometry.rs` with a whole-KJV
arm: `TokenId` treatment vs `WordId` CONTROL (the 67.7% round-trip is
RE-MEASURED as the control, not quoted). The four bounded claims at
n=31,102 verses / ~1,189 chapters:
1. scoped vs global vocabulary (the probe's own words: "run the comparison
   per real corpus");
2. saturation — 180/255 (corpus-bound) vs Alice's 255/255 (cap-bound); the
   KJV at scale is the deciding third point;
3. the overflow distribution (p50/p95/max particles per verse) — **this
   number picks carve A vs B**;
4. chapter token-usage Jaccard against the real TOC address space (32,357
   nodes), replacing the n=1 pair.
- F-TVT-1: byte-exact reconstruction over the WHOLE corpus, or the arm
  fails loudly (no sampling).
- F-TVT-2 (anti-vacuity): the `WordId` control must measurably differ from
  the `TokenId` treatment; if control == treatment the probe measured
  nothing and reports that instead of a result.

**W2 — the carve (GATED on W1's numbers; Opus review, Sonnet
transcription).** `ValueTenant::Token = 16` (BoardAggregates re-bases to
17), EXPERIMENTAL doc-comment verbatim in the `CausalWitness` style ("not
in the operator-locked §3 catalogue"), descriptor appended at
`row_offset: 252`, carve-budget assertion updated `220 → 220+N ≤ 480`.
- F-TVT-3: field-isolation matrix — write the Token lane, assert every
  other tenant byte-unchanged. Disable: remove the descriptor →
  `verify_layout` goes red.
- F-TVT-4: zero-fallback — all-zero reads *untokenized*. Disable: treat
  `codebook_id 0` as a real codebook → red.
- F-TVT-5: refuse-never-fold — `token_count`/`byte_len` overflow → recorded
  refusal, never a truncated value. Disable the guard → red.

**W3 — the lens write path (GATED on W2; Sonnet).** Population goes through
a borrowed lens into the `NodeRow` value slab — the
`WitnessLens::write_register` precedent (`probe_antecedent_binder.rs`),
zero-copy, escalate-never-clamp. The `TokenLane` `Vec<[u8;12]>` stays a
PROBE instrument; the production path has no parallel container (the
loci.rs deletion is the ruling, not a suggestion). `bible_wave` gains a
`--tokens` arm stamping the tenant on the SPO-stream rows — *inside the
40,767* — minting via `mint_for`, never `NodeGuid::new`.
- F-TVT-6: lens-vs-slab identity — an independent read at the DERIVED
  offset returns exactly what the lens wrote. Disable: write to a local
  copy instead of the slab → red.
- F-TVT-7: triples from one verse carry the SAME span reference (dedup by
  address; content never duplicated per-triple).

**W4 — the consumer verdict (Opus).** Re-run the anaphora/selectional and
basin numbers over byte-exact spans vs the WordId baseline, corpus-identity
gates from §1 asserted first. EXIT: an explicit **BUY / NO-BUY** against
PROBE-TOKEN-BPE-GEOMETRY-1's own verdict scale. NO-BUY is a valid outcome:
the tenant stays EXPERIMENTAL-unratified, W3 wiring is parked, and the
numbers are banked either way — a plan whose exit gate cannot say no is
not a gate.

## §5 Non-goals — the fences, so this plan cannot be read past its edges

1. **The LSTM recognizer is untouched** (byte-parity transcode — don't kill).
2. **The COCA tokenizer/lexicon is untouched** (don't kill). BPE is INTAKE
   tokenization; COCA is the grammar/lexical projection. Both live; neither
   replaces the other.
3. **The tarski register stays HELD.** No fold, no accumulation, no
   `Belief.rung`/stamp mechanism is proposed, implied, or enabled here. If
   the span address is ever evidence for that register's questions, that is
   its own future ruling, not this plan's.
4. **A BPE merge tree is NOT HHTL ancestry** — measured (same-depth token
   prefixes overlap). The tenant carries span ADDRESSES, never merge
   hierarchy.
5. **The two machines stay unconnected** — `insight_right_corner_read` (the
   parser) and `CausalWitnessFacet`/G24N4 (the wave); gated on
   BELIEF-ABI-RESTORATION-1 Step 2, not this plan.
6. **paperless-web's S3-backed archive is out of scope** (hydrate-aside
   doctrine vs live-write is unverified; Tantivy needs mmap regardless).
7. **No V1 mints, anywhere.**

## §6 Board hygiene (same commit as this file)

`INTEGRATION_PLANS.md` PREPEND; `STATUS_BOARD.md` D-TVT-0..4 rows (Queued);
`SUPERSESSION-INDEX.md` regenerated via `.claude/tools/supersession_index.py`.
Cross-refs: `E-TOKEN-BPE-CAN-FIT-NOT-YET-BUY-1`,
`E-ONE-RECEIPT-MANY-BORROWED-CONSUMERS-1`,
`E-STREAM-ORDER-VS-PREFIX-TREE-NEITHER-ACCUMULATES-1`,
`E-NO-BUNDLE-STANDING-WAVE-1`.
