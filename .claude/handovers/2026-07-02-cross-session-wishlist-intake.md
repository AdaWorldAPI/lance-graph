# 2026-07-02 — Cross-session wishlist intake (post-#630/#631)

> From: the V3 substrate session (lance-graph + OGAR + fleet-coordination lane).
> To: the three forwarding sessions (ruff/medcare, the second #630 reviewer,
> op-nexgen) and any future session picking up a deferred row.
> APPEND-ONLY. Board pointer: EPIPHANIES `E-V3-XSESSION-INTAKE-1`.
> Every row below is one wishlist item as received, with its disposition.

## What-I-did (executed this arc)

| Item (source) | Disposition | Evidence |
|---|---|---|
| **L1** merge-or-bless `RouteBucketTyped` C6 (op-nexgen) | **MERGED** — nexgen's `vendor/AdaWorldAPI-lance-graph/codegen_spine.diff` applied verbatim to `contract::codegen_spine` (additive trait + `?Sized` blanket bridge; codex PR #8 P2 fix included). nexgen: retire the vendor diff after the next sync. | 12/12 `codegen_spine` tests green incl. 5 new RouteBucketTyped tests |
| **L2** `emission_scan` classid_scan-sibling (op-nexgen) | **MINTED** — `contract::emission_scan`: `TypedForm {Typed, AnyTyped, RecordLink, Stub}` (#[non_exhaustive]) + `classify_ddl_type` (tokenized, precedence Stub > RecordLink > AnyTyped > Typed) + `EmissionCounts` fold. Zero-dep, same design language as classid_scan. nexgen: replace the hand-grep behind the 89.5% figure with this and file corrections against the classifier if the corpus disagrees. | this PR |
| OGAR **item 1** flip fuse (ruff/medcare wishlist) | **DONE** — ogar-class-view test asserting `ogar_vocab::app::{app_of, concept_of}` agree with contract `split_classid`/`CanonHigh` on a literal; #628↔#147 lockstep now mechanical, one-sided reverts fail a test. | OGAR branch, this arc |
| **item 7 / session-2 #7** COUNT_FUSE two-sided | **DONE** — OGAR-side pinning test carrying the literal name COUNT_FUSE; `git grep COUNT_FUSE` now hits in both repos. | OGAR branch |
| **item 2 (partial)** Genetics 0x0E mint + 0x1000 reservation | **DONE** — `ConceptDomain::Genetics (0x0E)` minted (the ledger already committed to `0x0E01_1000` CPIC); `0x1000` pinned RESERVED never-a-port-prefix in the allocation-table test. **q2 APP_PREFIX row NOT done** — blocked on the naming ruling (R-1 below). | OGAR branch |
| **item 4** OGAR post-flip prose sweep | **DONE** (each site verified by Read before edit; discrepancies vs claimed line numbers recorded in the worker report) | OGAR branch |
| **item 5** truncation-disallowed / overflow-as-SoC-reroute mirrored into OGAR DISCOVERY-MAP | **DONE** — appended D-entry citing the lance-graph doctrine + ruff `soc.rs` as shipped implementation. | OGAR docs/DISCOVERY-MAP.md |
| GraphRAG-rs inventory (operator refs) | **DONE** — `.claude/knowledge/graphrag-rs-inventory.md` + E-V3-GRAPHRAG-INV-1. Headline: LanceDBStore = 100% stub; Leiden single-level; cAST example-only; InferenceEngine = doctrine anti-exhibit (AsyncGraphRAG + TypedBuilder are the real exhibits). | this PR |

## RULING-NEEDED (operator checkpoints — recorded, not decided)

| # | Question | My read (advisory only) |
|---|---|---|
| R-1 | **hi-u16 naming**: lance-graph `le-contract.md` spells the canon hi-u16 as `domain:appid`; OGAR general canon reads it as `domain:concept-slot` (appid reading = the OSINT/FMA/CPIC special case). Same u16, two ledger descriptions. | Rule once, record in BOTH ledgers same-arc. Blocks the q2 APP_PREFIX-row guard (two number spaces, colliding small values, no guard today). |
| R-2 | **EdgeBlock canon wording**: lance-graph CANON (locked 06-13) `key(16)\|edges(16)\|value(480)`; OGAR P0 (pinned 06-10) `key(128 bit) + value(3968)`. | Likely reconcilable, not conflicting: edges(16) is a reserved subdivision of OGAR's 496-byte value, and the lance lock is later. But the ask stands: ONE set of consts + size asserts both repos pin, CLAUDE.mds pointing not restating. |
| R-3 | **Per-entry board files** (`board/epiphanies/E-<NAME>.md` + generated index): 4 of 5 recent rebases in a sibling session conflicted ONLY on EPIPHANIES/LATEST_STATE prepend collisions; tax grows with parallel sessions. | Real problem, council-sized governance change (append-only doctrine, hooks, every session's muscle memory). Recommend a council pass before any migration. |
| R-4 | **OGAR probe-ledger Wave A green-light** (PROBE-SUBSTRATE-PROPOSAL §9, stale since 06-10; ~200 LOC parser fully specified; NOT-RUN probe debt 20+). | Cheap, closes a growing debt; no session may self-authorize per §9's own text. |

## Deferred (dispositioned, with landing zones — not dropped)

| Item | Landing zone |
|---|---|
| **L3** Arrow/Lance columnar triple interchange (s p o f c as five parallel columns; retire mid-pipeline ndjson) | W5 consumer wave; pairs with the W1b zero-copy sink (Lance already speaks the format). Design note filed in Addendum-10. |
| **L4** materialization slot for DAG-backed columns ("this field is a cache of DAG node X") | Contract-flag design; belongs with the M19/W5 per-consumer mint reviews. |
| **E5** ruff `Mint` → ndjson/Arrow seam | Correctly waited — the W1b batch-writer/WAL shape it should target now exists (#631). Ruff session owns the emission side; the ingestion side is the cast/descriptor path. |
| OGAR **item 8** `fields_for(classid: u32)` ClassView custom-half routing | First step toward the post-P4 64k ClassView catalogue; needs a design pass (OgarClassView is concept-u16-keyed and prefix-blind today). |
| OGAR **item 9** consume ruff `writes`/`calls` in ogar-from-ruff + F17 body-triage probe | Substantial; endgame-critical for the 85/15→3-bucket measurement; queue as its own arc. |
| **O1–O4** (op-nexgen → OGAR: Rails front-end for ogar-from-schema, surrealdb-core direct AST handoff, compile_graph_ruby, OGIT zone keys) | Acknowledged; compile_graph_ruby (~15 LOC) is a good next OGAR quick-win batch; O1/O2 need the OGAR session's own arc. |
| **item 7 (corpus proof)** run `count_adoption` against a real stored bake + file PROBE-CLASSID-LEGACY-ALIAS with a kill condition | Still blocked in THIS container: no classid-keyed corpora present. Whichever session holds a real bake (q2 osint? nexgen DDL corpus?) should run it; the counting instrument ships since #630. |
| **X1** COORDINATION.md per repo | lance-graph **declines a new file** — the channel already exists: `.claude/board/CROSS_SESSION_BROADCAST.md` (committed, curated, append-only) + `CROSS_REPO_PRS.md`. Minting a third would be the duplication smell the ENTROPY ledger exists to kill. Sessions: broadcast merge events there. |
| **X2** probe-preamble convention (environment facts in subagent fetch/diff prompts) | **ADOPTED** in this session's worker briefs (the GraphRAG worker documented the api.github.com session-denial + raw.githubusercontent workaround instead of misreading the repo as fabricated — the exact failure X2 names). Recommend other sessions copy the pattern; no doc mint needed beyond this row. |

## Blockers / open questions

- ogar-class-view's contract dep floats on `branch="main"` unpinned — the flip fuse
  test closes the semantic side; pinning policy (rev vs branch) is a small follow-up
  the OGAR session may want.
- R-1 blocks: q2 APP_PREFIX row, and the authoritative naming line in both ledgers.

---

## APPENDED 2026-07-02 (later) — synthesis absorption (two synthesis passes + third-session addendum received)

The forwarding sessions produced two synthesis passes and a third-session
addendum over the combined wishlists. Deltas absorbed into THIS arc:

1. **Allocation-table mints serialized (their insight 2 / A-batch):** four
   sessions were queued against ogar-vocab's §2 allocation table (Genetics
   0x0E, OCR 0x08XX, 0x1000 reservation, q2 APP_PREFIX). This arc's OGAR
   batch is the mint vehicle — the 0x08XX OCR mint (class-level concepts
   only: unicharset/recoder/charset; unichars stay content-store rows,
   Osint count=0 precedent) was folded into the in-flight worker batch.
   q2 APP_PREFIX remains blocked on R-1. Future mints: batch or appoint a
   mint-warden; never solo-edit the allocation-table test.
2. **R-1 evidence upgraded + delivery form fixed:** the merged code already
   practices the CONCEPT reading of the hi-u16 (`0x0102` = project_work_item
   shared by openproject `0x0102_0001` and redmine `0x0102_0007` — an appid
   reading cannot express sharing). Suggested ruling on the table: hi-u16 =
   domain byte + concept slot; lo-u16 = app/render prefix; 0x1000 reserved.
   Whatever is ruled: deliver as ACCESSOR RENAMES (`domain_of` + the ruled
   name for the second byte) so the compiler carries the vocabulary — this
   arc counted FOUR instances of order/count prose rotting against code.
3. **R-3 fused:** X1 (COORDINATION.md) formally YIELDS to per-entry board
   files; both go to the council as ONE proposal (per-repo coordination dir
   = merge-event signal + per-entry entries). The measured cost datapoints:
   4-of-5 and 3-of-N rebases conflicting ONLY on board prepends, from two
   independent sessions.
4. **Scan family named as a contract pattern** (A5): classid_scan +
   emission_scan + any future counter share `Form enum + classify_* +
   fold-to-counts`, zero-dep, in the contract. Recorded in Addendum-10 and
   in emission_scan's module doc.
5. **L3/E5 interchange fusion** (A6 + insight 5): one Arrow schema family,
   provenance header with `minter@sha`, ndjson stays the golden/diffable
   layer, ingestion targets the W1b cast shape — no second envelope.
6. **Disposition unification** (A4/insight 5): the disposition ledger
   (`minted | adapter | hand-port | excluded(reason)`) and the 3-bucket DO
   triage are ONE doctrine — buckets = routing decision, ledger =
   conservation accounting; one `Disposition` enum where Mint output lives,
   variant names matching nexgen's RESIDUAL-THREE-BUCKETS.md. Ruff/OGAR
   sessions own the landing.
7. **F17 ratified as the most-agreed next move** — flagged independently by
   all three sessions; one probe, two consumers (ruff fidelity + OGAR
   ActionDef/M25 runtime): run once, both watching, archive the corpus with
   the run (convention 8).
8. **Probe-corpus archival convention adopted** (their insight 6): input +
   generation recipe + hash archived WITH every quotable measurement.
9. **Cross-session citation rule:** epiphany references carry board
   `E-<NAME>` keys or file paths, never per-session ordinals (two sessions'
   "E5" already collided).
10. **O3 stale-item catch acknowledged:** `compile_graph_ruby` already
    exists at ogar-from-ruff/src/mint.rs:99 per the tesseract census —
    the residual is at most flipped-order test expectations. Wishlists rot
    at prose speed; merge events belong on the coordination channel.
11. **X2 lands inside sonnet-worker-guardrails** (not a standalone doc):
    the probe-preamble convention (environment facts, expected 403s,
    authenticity checks in every fetch/diff brief) is a guardrails-§1
    clause family. Queued as a one-paragraph guardrails addition next time
    that doc is touched.

---

## APPENDED 2026-07-02 (operator ruling) — R-1 CLOSED: it was a PHANTOM conflict; the canon already exists and this session authored it

Operator: "Theme 4 is wrong — you did it yourself: domain:appid:classview/concept."
Verified against the shipped canon (authored in this session's own V3 folder,
merged via #629/#630):

- `.claude/v3/soa_layout/le-contract.md:21-30` — the 4-byte prefix IS the
  composed classid: `[domain byte][appid byte][classview u16]`;
  **canon hi u16 = `domain:appid`** (e.g. `0x07:01` = OSINT:q2);
  **custom lo u16 = classview** (hosts the 0x1000 monitor + OGAR §2 app
  render prefixes; post-P4 the 64k ClassView catalogue).
- `.claude/v3/knowledge/v3-substrate-primer.md:94` — hi u16 spelled
  **`CANON concept/domain:appid`**: "concept" NAMES the whole hi u16;
  "domain:appid" is its byte spelling. The le-contract and OGAR-canon
  descriptions were never in conflict — they name the same u16 at two
  granularities.
- The dissolution of the "sharing" argument: `0x0102_0001` (openproject:
  WorkPackage) and `0x0102_0007` (redmine:Issue) share hi `0x01:02` =
  domain 0x01 + **appid byte 0x02** (the canonical app-concept slot) and
  differ in the **lo-u16 ClassView/app-render prefix** (0x0001 OP,
  0x0007 Redmine). The sibling session's "an appid reading cannot express
  sharing" argument conflated the canonical **appid byte** (hi half) with
  the per-vendor **APP render prefix** (lo half) — the word "app" appears
  in both halves with different meanings; THAT homonym, not the layout,
  produced the whole R-1 thread.
- The q2 "collision" dissolves the same way: q2's appid `0x01` lives in
  the HI half inside a domain (`0x07:01` = OSINT:q2); OpenProject's
  `0x0001` APP_PREFIX lives in the LO half. Two registers, positionally
  distinct by construction. The needed "guard" is this naming line in
  both ledgers, not a new number-space rule. The q2 APP_PREFIX row is
  therefore NOT blocked — it is simply a mint to make when q2 renders
  classviews.
- Process lesson (mine): I relayed the sessions' conflict claim into a
  RULING-NEEDED row without grepping my own canon docs first — the exact
  "consult, don't guess" / rule-10 failure the guardrails codify, this
  time on the orchestrator relaying instead of ruling. Calibration
  datapoint for the fleet: the sessions escalated a phantom; escalation
  beats silent divergence, but the FIRST check on any "two ledgers
  disagree" claim is the primer/le-contract, which already carried the
  reconciliation in one line.
- Residual actions: (a) OGAR ledger gets the naming line same-arc (this
  OGAR batch); (b) the sibling sessions' item-3 asks are answered by this
  entry — cite `le-contract.md:26` and `primer:94`, do not re-derive;
  (c) R-2 (EdgeBlock consts) stays open but is downgraded to "probably
  the same homonym class — verify against canon before treating as a
  conflict"; the ask for one pinned const set remains sound engineering.

---

## APPENDED 2026-07-02 (operator rulings 2+3) — R-2 answered empirically; L3-as-schema-design KILLED

**R-2 (EdgeBlock) — operator reframe:** the two spellings were never the
issue; "the question is just if lance-graph is able to pull the second
edges guid separately — if so it's preferable to have edges cheap without
having to load the whole values." Answer from the shipped contract:

- YES at the contract level: `NODE_ROW_COLUMNS`
  (contract/canonical_node.rs:668-687) declares **three separate
  ColumnDescriptors** — Key (offset 0, 16B), **Edges (offset 16, 16B,
  own name_id)**, Value (offset 32, 480B). The envelope contract already
  carves edges as an independently addressable column.
- NOT yet at the materialization level: exhaustive grep
  (`NODE_ROW_COLUMNS` and `NodeRowColumn::Edges` over `crates/`
  `--include=*.rs`) finds **zero consumers outside the contract** — no
  Lance write path yet materializes the three columns as separate Lance
  columns. That wiring IS the W1 sink work.
- **Ruling absorbed as a W1 requirement:** the sink writes NodeRow as
  THREE Lance columns (not one 512B blob), so Lance's columnar projection
  serves edge traversal at 16 B/row without touching the value slab.
  Mechanical gate for the R-2 closure: an edges-only projection test
  (read the EdgeBlock column for N rows; value column not fetched).
- Canon wording settled by the same ruling: `key(16)|edges(16)|value(480)`
  stands (the subdivision is what makes edges cheap); OGAR's
  `key + value(496)` remains true as the coarse "everything-not-key"
  view. The one pinned const set BOTH repos cite = `NODE_ROW_COLUMNS` +
  `NODE_ROW_STRIDE` — OGAR pins by asserting the same numbers, not by
  restating prose.

**L3 interchange — operator ruling: "defining arrow schemas is bullshit
and hallucination because we already have a working SoA schema."**
The L3/E5 "one Arrow schema family (triples batch s/p/o/f/c + facets
batch)" framing — including my own Addendum-10 bullet — is WITHDRAWN as
schema design. There is no new interchange schema to define: the SoA
schema (NODE_ROW_COLUMNS / SoaEnvelope / VALUE_TENANTS / the 16-byte
facet catalogue) IS the columnar schema, and Lance's own columnar I/O
writes LE bytes from the envelope-described backing store. Extraction
output (ruff `Mint`, triples, facets) lands as **node rows + facets in
the existing canon layout through the W1b cast/descriptor path** — the
"interchange format" question dissolves into "write SoA rows".
What survives of L3/E5: provenance stamping (`minter@sha`) and ndjson as
the human-diffable golden layer for PR review — both are about artifacts
AROUND the store, not a second schema. op-nexgen / ruff sessions: do NOT
start a five-column triple-schema design; target the SoA envelope.

---

## APPENDED 2026-07-02 (operator ruling 4) — R-2 requirement RECALIBRATED: the 512-byte SoA schema is NOT touched

Operator: "the SoA schema was 512 bytes before and after and was tested
against surrealdb kv-lance AND batch writer — so i don't see the reason to
touch that." The previous appendix's "W1 sink requirement: write NodeRow as
THREE Lance columns" is RETRACTED as over-reach. Corrected reading:

- The 512-byte row (`key(16)|edges(16)|value(480)`, NODE_ROW_STRIDE = 512)
  is the tested, frozen storage unit — before and after, kv-lance and
  batch-writer verified. Nothing restructures it.
- "Edges cheap without loading the whole values" is served by the layout
  AS IT IS: `NODE_ROW_COLUMNS` already describes the strided view (Edges =
  16 bytes at row_offset 16), so an edge sweep is a strided 16-of-512
  slice read over the existing backing store / mmap — zero-copy per the
  data-flow rule (SIMD/readers slice into the store, never copy). No new
  Lance schema, no column re-materialization.
- Residual gate (read-side only, no storage change): an edges-only strided
  read helper/test over the NODE_ROW_COLUMNS descriptors proving edge
  traversal touches 16 B/row. Whether an additional Lance-level column
  projection ever pays for itself is a MEASUREMENT question for later
  (truth-architect rules apply) — not a schema decision, and not W1.
- R-2 is now fully CLOSED: canon text stands as-is on both sides; the one
  shared const set = NODE_ROW_COLUMNS + NODE_ROW_STRIDE; no repo touches
  the 512-byte unit.

---

## APPENDED 2026-07-02 (ops) — vart in-container unblock: the mirror + re-apply recipe

The `AdaWorldAPI/vart` session-scope 403 (which blocked ALL OGAR workspace
cargo in-container) is worked around, not solved:

- `/home/user/vart` = complete file-by-file mirror of the fork @ main via
  raw.githubusercontent.com (the only channel the proxy leaves open for
  this repo; UI scope-add did not propagate, wget/zipball/api all 403,
  add_repo unreachable — Claude_Code_Remote MCP unauthenticated). Local
  git repo with provenance commits; vart's own suite 117/117; NO upstream
  history — replace with a real clone when scope propagates.
- Re-apply recipe for local OGAR testing (NEVER COMMIT — breaks CI):
  in `crates/ogar-knowable-from/Cargo.toml` swap
  `vart = { git = "https://github.com/AdaWorldAPI/vart", optional = true }`
  for `vart = { path = "../../../vart", optional = true }`, test, then
  `git checkout` the manifest. Note: `[patch."<git-url>"]` does NOT work
  — cargo still fetches the patched-over git source (403s).
- Proven while applied: full OGAR workspace green including the 16
  vart-backend tests (first in-container run ever).
