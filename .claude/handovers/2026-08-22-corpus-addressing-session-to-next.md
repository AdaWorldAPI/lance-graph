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
| "One address, four witnesses — the translation lane lives in the classid" | **Asserted without a source — not false.** `ConceptDomain` has no translation notion and `routing.md` does not "answer it in one row" as the session claimed, so the assertion was unfounded as made. But D-RCC-2 does specify *"Language = lane discriminant **resolved from** classid — no LanguageDto"* (`plans/rosetta-codebook-convergence-v1.md:112`, `Status: PROPOSED`). Note the wording: **resolved from**, not *lives in* — the classid is an address and the lane is what it resolves to, never content carried in the id. The original error was asserting without reading; retracting it as fabricated was a second error, corrected here. |
| "The pericope is an edge set" | It is a collision in a content-addressable index. Nothing stores the parallelism. |
| "MailboxId IS the NiblePath" (quoted as fact) | `tenants.md` §7.3 and `le-contract.md` §5.3 both flag this as **DOC-ONLY**, awaiting a ruling. |

**The pattern, so it is not repeated:** every failure was structure inferred
from a *name* or a *width* instead of from the definition. "HHTL" sounded like
a hierarchical path, so it was used as one; two structures that share a 12-byte
`6×(8:8)` carving were taken to share a role. Read `.claude/v3/soa_layout/`
(4 files, 905 lines) **before** proposing anything about keys, tenants, or
rails.

**The invariant that bounds all of it** (not a finding — the standing rule):
**identity is an address and is never derived from content.** A quantized code
exists to make similar things collide; an identity exists to keep them apart.
Anything content-derived — a CAM-PQ code, a hash, a pooled embedding — lives in
a **value lane**, and the shipped example is `ValueTenant::EpisodicBasin = 15`,
whose `self_code` (12 B, bytes `4..16`) is licensed precisely because it sits at
a strictly higher rung than its inputs. A content code in the key would also
re-address the whole corpus on any re-bake.

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

## §4 `part_of : is_a` — what is already banked, and the one new thing

> **This section was rewritten by a 5+3 council** (2026-08-22; verdicts
> 2 VIOLATES / 12 GAP / 6 PRIOR-ART-AT / 4 RISK, then 1 BLOCK(P0) + 5 FIX from
> the three reviewers). Its first draft asserted seven propositions. Five were
> already banked canon under E-ids it did not cite, one inverted its source's
> emphasis, and one contradicted an existing design. What follows is what
> survived. The council's own overreach is corrected in place, not hidden.

### Cite these; do not restate them

| what | where it is already banked |
|---|---|
| L1 is a PAIRED reading — `part_of` (hi byte, mereology) : `is_a` (lo byte, taxonomy), two hierarchies in one key | `E-V3-PART-OF-IS-A-TILE`, EPIPHANIES:12266 (2026-06-23) |
| The bytes are content-blind; only the ClassView projects meaning | `E-FACET-8-8-ALWAYS` (:12088), `E-CONTEXT-ROLE-TISSUE-1` (:9517), `knowledge/context-role-traversal-tissue.md:37` |
| Adopting the V3 reading needs no `ENVELOPE_LAYOUT_VERSION` bump | EPIPHANIES:12270 — **and the same entry carries a `NodeGuid::new_v2` 7-group `I-LEGACY-API-FEATURE-GATED` blocker. Do not quote the conclusion without the blocker.** |
| The CPIC basins are mereology, readable as HHTL `(X;Y)` per tile | `E-V3-BASINS-ARE-MEREOLOGY-NOT-LABELS`, EPIPHANIES:12055 |

A second statement of a banked finding under a new name divides the search
surface for every later session. The first draft of this section cited **zero**
E-ids.

### The emphasis runs the other way

`.claude/knowledge/ast-as-partof-isa-address.md:26-28`, verbatim:

> "the *key* carries only the **4-tier routing prefix**
> (`NiblePath::from_guid_prefix_v3`), the complete 6-pair address lives in the
> `FacetCascade` **value facet**."

So the key carries a routing prefix and the complete address is a value facet —
not "the rail is a key tail, not merely a value layout", which is how the first
draft led. `TailVariant::V3` names the tail's *reading*; that is a narrow true
claim, not the headline.

### Three counts, three objects (a question that was malformed)

"Three slots or six?" was one question hiding three. All three are settled:

| count | what it counts | source |
|---|---|---|
| **3** | what the V3 tail re-reads — `leaf·family·identity` | `is_layout_preserving`'s doc scopes the tail to *"bytes 10..16"* (canonical_node.rs:1289-1297); `lance-graph-contract/Cargo.toml:65` — "V3 is a *reading* of the SAME leaf·family·identity 3×u16" |
| **4** | the key's routing prefix | `NiblePath::from_guid_prefix_v3`; ast-as-partof-isa-address.md:26 |
| **6** | the complete `FacetCascade` address, `const _`-asserted | facet.rs; ast-as-partof-isa-address.md:23 |

**Do not cite `TailVariant::V3`'s own doc (canonical_node.rs:1284) for the "3".**
That doc calls V3 *"the `(part_of:is_a)` 8:8 tile"* and names no field count;
`leaf(u16)·family(u16)·identity(u16)` is the **V2** doc one variant above. The
"3" is real but comes from the two sources in the table — and the gap between
V3's aspirational doc and its V2-identical mint is itself worth knowing.

`le-contract.md` §3's `[H]` "do not unify silently" flag is about **L7 helix(48)
vs the CANON key tail** — a different question. It was borrowed for this one; it
is untouched and still open.

### The corpus is already specified — by a PROPOSED design

`.claude/plans/rosetta-codebook-convergence-v1.md` §D-RCC-2 (`E-RCC-1-FOUR-LANES-ONE-KEY-1`,
EPIPHANIES:7499) covers this corpus:

> "Row = versification-normalized verse address (3-byte b/c/v core; scheme map
> … as a lane property). Lanes = witnesses. **Within-row ordinal =
> `clause_index` (verses ≠ sentences).** Absence = `TextAbsent`, never zero.
> Language = lane discriminant **resolved from** classid — no LanguageDto.
> ~31,102 rows."

**It is `Status: PROPOSED (doc-only)`** (that file, line 3) — the plan marks its
shipped pieces explicitly (lines 53-57, 224) and D-RCC-2 carries no such marker.
So it is prior art to build ON, not a ratified contract to defer to. Two
consequences:

- **The measurement stands, and it is not novel.** Every KJV containment level
  fits a one-byte ref relative to its parent — chapter-in-book 150,
  verse-in-chapter 176, clause-in-verse 19 (a punctuation **upper bound**, which
  over-counts), word-in-clause 46; zero spills in 31,102 verses. D-RCC-2's
  3-byte b/c/v core is the same ground. Cite it; do not re-derive it.
- **Clause placement is an open disagreement between two unratified designs.**
  D-RCC-2 puts the clause as a **within-row ordinal**, for a stated reason —
  *verses ≠ sentences*. The first draft of this section put it in a key rail
  slot. Neither is ratified; the divergence is the operator's call, not
  something to settle by picking the newer text.

### The one genuinely new finding

**"HHTL" names two different structures**, and conflating them caused real
errors in this session. In the V3-tail context it is the `(X;Y)` coordinate
reading of the cascade key (`E-V3-BASINS-ARE-MEREOLOGY-NOT-LABELS`); in
`hhtl.rs` it is `NiblePath`, the `subClassOf` (P279) Abstammung router with
DOLCE basins and `mask-inherits-as-delta`. No existing E-id names this
collision — the nearest prior art is a "nibble homonym" table
(`handovers/2026-08-21-2200:110-114`) and a traced collision section
(`ATTENTION_MASK_AUDIT_2026_08_21.md:75`), neither of which covers it.

### Two limitations, stated rather than assumed

- **`is_layout_preserving()` is a hardcoded `true`** (canonical_node.rs:1289-1297),
  not a computed check tied to V3's bit reading. "No version bump" holds by
  convention, not by verification — and by the repo's own falsifiability rule it
  is an assertion no input can falsify.
- **The levels-reading is a ClassView CHOICE, not a property of the bytes.**
  `le-contract.md:122-140` sanctions "area:location in stacked exactness" for the
  identical `6×(8:8)` — six pairs as ONE refinement axis rather than six levels —
  and "one-byte refs per slot" never says what the byte points at (within-parent
  ordinal? palette index? relation-type code?). **No corpus classid has a
  ClassView entry today**, so the reading is unselected.

### Still open (operator, not a session)

- **The `is_a` byte's referent.** Per slot purity a ClassView matter and, for a
  taxonomy, an OGAR mint — not a consumer's invention.
- **Clause placement** — within-row ordinal (D-RCC-2) vs key rail slot.
- **F9 provenance.** A clause boundary derived by splitting text on punctuation
  is content-derived. Whether such a value may serve as an address does **not**
  go away when the clause moves from a key slot to `clause_index` — D-RCC-2's own
  parenthetical (*verses ≠ sentences*) is that same concern. Unanswered against
  `clause_index` today.

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
