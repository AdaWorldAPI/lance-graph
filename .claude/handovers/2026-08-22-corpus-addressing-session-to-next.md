# Handover — corpus addressing (KJV / Composite Gospel / part_of:is_a)

> From: session on `claude/alpha-rung`, 2026-08-22.
> To: whoever picks this up. **Read §1 before §3.** The session produced a
> small set of solid measurements and a large amount of fabricated
> architecture; §1 tells you which is which so you do not inherit the latter.

## §1 Trust boundary — read this first

**Trust the measurements. Do not trust anything this session said about the
canon** except the part explicitly re-derived from the spec in §4, which was
written after the errors below were found.

### Fabricated — do NOT build on

| Claim made | Why it is wrong |
|---|---|
| "HHTL is the Book·Chapter·Verse containment path" | `hhtl.rs:1-38` — HHTL/`NiblePath` is the `subClassOf` (P279) **Abstammung** axis; `mask-inherits-as-delta`, *"walking DOWN the path is IS-A inheritance"*. Book⊃Chapter⊃Verse is **mereology**. `le-contract.md:56` separates `part_of : is_a` as two bytes. |
| "Book·Chapter = HEEL exactly, 6 nibbles free" | Numerology. The 2/2/2 nibble carving was chosen by the session via `ceil(log16 n)` and then reported as a discovery. Ignores that OGAR reads a tier as a 256×256 centroid tile needing hierarchical-4⁴ codebooks. |
| "One address, four witnesses — the translation lane lives in the classid" | No canon support. `ConceptDomain` is ProjectMgmt/Commerce/Ontology/Weather/Osint/Ocr/Health/Anatomy/Auth/Genetics — no translation-lane notion. The session claimed `routing.md` "answers it in one row"; it does not. |
| "Cam96 drops into the key tail" | Content-derived identity. CAM-PQ collisions are the *point* of the codec; identity's job is to discriminate. Correct home already exists: `ValueTenant::EpisodicBasin = 15`, `self_code` at bytes `4..16` — a **value** lane, licensed because it sits at a strictly higher rung than its inputs. |
| "The pericope is an edge set" | It is a collision in a content-addressable index. Nothing stores the parallelism. |
| "MailboxId IS the NiblePath" (quoted as fact) | `tenants.md` §7.3 and `le-contract.md` §5.3 both flag this as **DOC-ONLY**, awaiting a ruling. |

**The pattern, so it is not repeated:** every failure was structure inferred
from a *name* or a *width* instead of from the definition. "HHTL" sounded like
a hierarchical path. Cam96 is 12 bytes read as `6×(8:8)` and so is the key
register, so matching shape was taken for matching role. Read
`.claude/v3/soa_layout/` (4 files, 905 lines) **before** proposing anything
about keys, tenants, or rails.

### Measured — these hold

Every one was produced by running something; the commands are reproducible from
the artifacts in §2.

## §2 Artifacts and where they are

Nothing below is committed; all of it is re-fetchable.

| what | where |
|---|---|
| KJV lane (`bible_kjv.json`, getbible v2, **GPL** transcription) | MedCare-rs Release `corpora-snapshot-2026-07-26` → `source-corpora.tar.gz` |
| 4 PD lanes (luther1545 / elberfelder1905 / bkr / tischendorf) | lance-graph Release `v0.1.0-codebooks-2026-07-26` → `pd-texts-bundle.tar.gz` |
| `versification_map.tsv` + report | same Release → `rosetta-pd-bundle.tar.gz` |
| German rails (`tekamolo.tsv`, `satzklammer.tsv`, `relative_pronoun.tsv`, `lexicon.tsv`) | same Release → `de-bundle.tar.gz` (CC BY-SA 4.0) |
| Cam96 codebook (KJV vocab, Jina-v3 96d) | lance-graph Release `v0.1.0-cam96-data` |
| **NTN-individuals.owl** — persons + locations register | MedCare-rs `corpora-snapshot-2026-07-26` → `restricted-corpora.tar.gz`. **Licence UNVERIFIED (SemanticBible) — local oracle only**, same shelf as PROIEL. |
| **Composite Gospel Index 1.2** | operator-supplied this session; `semanticbible.com/cgi/2004/11/compositeGospel.1.2.rdf`. `dc:source` = RSV, rights reserved — but the file contains **no scripture text**, only references, titles and structure. |

## §3 Measured findings

### 3.1 The versification map's KJV side is exact; 51 of its offsets are not applicable

- `kjv_verse_count` matches the actual KJV lane for **3,567 / 3,567** rows
  (1,189 chapters × 3 lanes). No gaps, no extras. The KJV lane is dense —
  verse count == max verse number in every chapter. The offset census
  reproduces the report exactly (luther1545 36 / elberfelder1905 3 / bkr 8).
- **51 rows declare an offset that cannot be applied end-to-end** — pure
  arithmetic on the map's own columns: 47 have `kjv_verse_count + offset >
  lane_verse_count`; 4 have `1 + offset < 1`.
- The dominant cluster is luther1545 Psalms with `offset=+1` on chapters where
  **both lanes have identical verse counts** — self-contradictory; an endpoint
  must fall off.
- **Cause, worked on Psalm 84** (the report's own receipt): Luther counts the
  superscription as v1, so `+1` is right for KJV v1–v10 — but the lane has
  **dropped KJV v11** entirely, so KJV v12 aligns at `+0`. A single per-chapter
  integer cannot describe a chapter with an interior verse loss.
- Re-scoring head vs tail independently: **22 of the 51 want different
  offsets**. *Caveat: that scorer reuses the original's cheap 5-char
  prefix-anchor signal, so 22 is a diagnosis; the 51 is arithmetic and is
  certain.*
- The report's caveat covers whole-*chapter* divergence only and does not
  mention intra-chapter verse loss. It also explains the thin payoff
  (+21 verses, 0.14 pp): where an offset was found it is partly wrong.

**Operator ruling arising (2026-08-22):** *cross-version reference must use the
verse ADDRESS, never token/verse position.* An `offset` column is position made
portable by arithmetic, and it cannot express either degenerate case an
alignment needs — an address with no witness, or a witness with no address.

### 3.2 Composite Gospel Index — census and defects

355 pericopes (seqID 1–355 contiguous), 601 sources, 3,774 verse-ranges;
Luke 192 / Matt 185 / Mark 122 / John 102; no duplicate titles.
Sources per pericope: 208×1, 65×2, 65×3, **17×4** ⇒ **147 parallel groups**.
Per-gospel `next`/`previousPericopeForSource` chains thread narrative order,
which is *not* verse order.

Defects found:
- **24 of 601** `hasVerseCardinality` values disagree with the range they
  annotate (same-chapter cases only, where the span is unambiguous), in both
  directions — `Matt.1.1-Matt.1.17` says 18, spans 17; `Mark.7.14-Mark.7.16`
  says 2, spans 3. **The reference string is the trustworthy field.**
- **2** `hasReference` values carry a leading space (` Matt.12.38-…`,
  ` Luke.16.18-…`) — a strict parse crashes.
- **4** ranges cross a chapter boundary, so `(chapter, verse)` does not close a
  range: `Mark.8.34-Mark.9.1`, `John.7.53-John.8.11`, `John.15.18-John.16.4`,
  `Luke.22.66-Luke.23.1`.
- **1** malformed reference: `Pericope.102.Mark` carries
  `hasReference="Matt.4.33-Mark.4.34"` under `isPartOf="#Mark"` — mismatched
  books; should read `Mark.4.33-Mark.4.34`.

**600 of 601** RSV ranges resolve inside KJV chapter bounds (the single failure
is the typo above), and gospel chapter counts agree. So RSV↔KJV gospel
versification agrees at range granularity **and no RSV text is needed** — the
references are structure; the content comes from the KJV lane.

**What it is for:** the 147 parallel groups are a labelled falsifier for a
content index — same event, up to four independent retellings — with the 208
single-source pericopes as the paired negatives (the can-it-stay-silent half,
on real narrative rather than padding).

### 3.3 The persons/locations register

`NTN-individuals.owl`: **676 named individuals** — Man 349, Woman 48,
SonOfGod 6, Angel 1; City 88, StateOrProvince 22, LandArea 10, Region 9,
Island 9, FreshWaterArea 8, SaltWaterArea 3, Mountain 3. 25 relation
predicates (`childOf`/`parentOf` 150, `collaboratesWith` 186, `residentPlace`
111, `ethnicity` 143, `spouseOf` 36, `siblingOf` 30, `nativePlace` 28,
`visitedPlace` 38, `knows`, `hasEnemy`, `memberOf`).

**37 surface names collide — `Mary` six times**, Alexander/Simeon/Joseph 5,
Simon/Gaius 4. No positional pointer discriminates those; a register does.

Two defects in the release asset: **`NTN.owl` is not an ontology** — the
195-byte file is a captured HTML error page — and its `owl:imports` target
(`NTNames.owl`) is absent, so the class hierarchy is unresolvable from the
bundle. `theographic-bible-metadata` (CC BY-SA 4.0, whole-Bible people+places,
the publicly-shippable one) is graded PUBLIC-eligible in `EPIPHANIES` but is
**in no Release of any repo**, and the sibling clone did not survive.

## §4 The corrected reading of `part_of : is_a` (re-derived from the spec)

This section was written **after** the §1 errors were found and is the only
architectural content in this handover. Verify it anyway.

- `le-contract.md:56` — **L1 = `part_of : is_a`, 6 × (8:8)**, *"the V3
  mereology:taxonomy key rails; one-byte refs per slot."* Each slot is a
  **pair**: a mereology ref and a taxonomy ref at the same level. L1–L4 share
  `CascadeShape::G6D2` and are differentiated **only by the ClassView**
  (§2 slot purity — labels and positions never live in a payload slot).
- `tenants.md:97` — `TailVariant::V3` is the **cascade-key `(part_of:is_a)`
  8:8 tile**, feature `guid-v3-tail` (default-on; implies `guid-v2-tail`).
  `canonical_node.rs` states the bridge: *"`NodeGuid` and `FacetCascade` are
  the SAME 16 bytes: `classid(4) | 6×(8:8)`."*
  `TailVariant::is_layout_preserving()` returns `true` unconditionally — every
  variant re-reads the same 16 bytes, so no `ENVELOPE_LAYOUT_VERSION` bump.
- **The precedent is already ruled.** `CLASSID_CPIC_V3`, operator directive
  2026-06-26: *"The 6 V3 basins are genomic MEREOLOGY, not labels … a gene's
  identity is its position in the part-of hierarchy (genome → chromosome →
  region → locus → gene), readable as HHTL `(X;Y)` coordinates per
  `(part_of:is_a)` tile — never a flat type tag a HashMap would carry. The
  6-basin + relative location is a substantial address; spending it on labels
  wastes it."*

  Note this also disambiguates the name: **in the V3-tail context "HHTL" means
  the `(X;Y)` coordinate reading of the cascade key**, not `NiblePath`'s
  subClassOf router. One word, two structures.
- **The corpus fits the constraint.** `part_of` is a one-byte ref per slot, so
  each level must be countable *relative to its parent*. Measured on the KJV
  lane: testament-in-canon 2; book-in-testament 39 (OT) / 27 (NT);
  chapter-in-book 150 (Psalms); verse-in-chapter 176 (Psalm 119);
  clause-in-verse 19 (Daniel 5:23, punctuation upper bound — over-counts, so
  conclusive in the safe direction); word-in-clause 46. **Zero levels exceed
  255 anywhere in 31,102 verses.**
- **A Nebensatz is one slot, not an extension.** The `is_a` byte shares the
  slot with its `part_of` byte, so the clause level answers both from one
  16-bit pair: *which* clause of the verse, and *what kind*
  (Hauptsatz / Relativsatz / Adverbialsatz). Currently zero = *not consulted*
  per RESERVE-DON'T-RECLAIM.

### Two things this session deliberately did NOT resolve

1. **Three slots or six.** `TailVariant::V3`'s doc scopes the reading to
   *bytes 10..16* — `leaf·family·identity`, i.e. **3** pairs — while CPIC's
   directive speaks of **"the 6 V3 basins."** Different depths.
   `le-contract.md` §3 flags the key-tail-vs-facet reconciliation as `[H]`
   with *"do not unify silently in code."* **Do not pick one to make a design
   work.** This needs an operator ruling.
2. **The `is_a` byte's referent.** Per §2 that is a ClassView matter and, for a
   taxonomy, an OGAR mint. Not something a consumer invents.

## §5 What changed in the tree

Commit `51bf2547` on `claude/alpha-rung` deletes six `deepnsm-v2` modules —
`tekamolo.rs`, `lexicon.rs`, `toc.rs`, `hydrate.rs`, `promote.rs`, `loci.rs` —
and `examples/toc_hydrate.rs`. Each duplicated shipped code:

- `tekamolo.rs` / `lexicon.rs` → `lance-graph-planner/examples/insight_coca_read.rs`
  already emits all four TEKAMOLO lanes from lemmatised COCA data.
- `toc/hydrate/promote.rs` → the promoter minted V1 keys and wrote a TEKAMOLO
  tenant the planner path already writes.
- `loci.rs` → resolution ships in `deepnsm/examples/spo_anaphora_nibble.rs` and
  `jc/examples/l9_loci_real_text.rs` (measured head-to-head: **0.727** shipped
  vs 0.455 noun-only vs **0.273** agreement-blind, which is what `loci.rs`
  implemented); binding ships in
  `lance-graph-planner/examples/probe_antecedent_binder.rs`, whose header names
  this exact gap and closes it through `WitnessLens::write_register` into the
  `NodeRow` value slab — zero-copy, escalate-never-clamp, five gates.
  `loci.rs` pushed into `WitnessStream`, which that header calls *"a parallel
  local structure that never reaches the SoA lane."*

`deepnsm-v2` after the removal: 114 tests pass, `cargo fmt` clean,
`cargo clippy --all-targets --no-deps -D warnings` clean.

## §6 Suggested next step

If the corpus work continues, the first deliverable is a **falsifier, not a
design**: encode the 147 parallel groups and the 208 negatives from §3.2
against the existing Cam96 codebook and report whether a pericope's witnesses
are mutual nearest neighbours. Note the shipped codebook is **word-level**
(`cam96_codes.bin` = 150,528 B / 12 = 12,544 codes against a 12,542-line
`bible_vocab.txt`; codebook = 6 axes × 256 centroids × 16 f32 = 96d), so
verse-range content addressing needs a pooling decision — and mean-pooling
washes out exactly the rare proper nouns that separate one pericope from its
neighbour. That decision is the real work; do not treat it as free.

Do not resume the §1 line of reasoning under any circumstances.
