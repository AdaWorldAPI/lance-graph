# Plan: ruff_charon binary row lanes + header digest (2026-11-02)

**D-id:** D-CODEGRAPH-0-BIN ([STATUS_BOARD](../board/STATUS_BOARD.md)). Operator answers after planning: queries keep graph-wide u32 ids (one id-translation pass per changed crate); the TSV reader stays as a debug `open_tsv`.

DECISION (operator 2026-11-02): each crate becomes fixed-width id lanes + a symbol
table, content digest in the header. Unchanged crate = header read; changed crate =
no text parsing; TSV stays as a debug export. Budget: unchanged refresh µs, batch µs;
a ms result is a failed gate.

## 1. Findings (what happens today)
Writer — ruff_charon/src/store.rs (scratchpad ruffwt checkout):
- :31-44 `ROWS_VERSION=2`, 7 named columns; files.tsv (path,fnv64), residuals.tsv.
- :217-264 `import`: `relations()` -> one `writeln!` per row of TEXT (pred, subject,
  object, GRADE, file, start, end); trailing `CRATE <name> COMPLETE|PARTIAL` row;
  `replace` (exclusive temp + fsync + rename, :168-208) for rows, residuals, files
  (files last = "a crate whose digests are present has its rows", :259).
- :92-148 `read_table`: per line a BTreeMap<String,String> (one alloc per field);
  `rows()` :289 rebuilds `Row` (owned Strings). `stale` :352 = source-file fnv64
  vs files.tsv (unrelated to the rows digest). `dangling` :375 = string filters.
Reader — lance-graph-codegraph/src/artifact.rs:
- :415-495 `read_text_as`: `text.lines()`, `split('\t')` per line, `number()` parse,
  `Memo` + `Symbols::intern` per field, `dispatch_target` string split for MAY rows,
  `P_OTHER` side table. This is the 6.8-10.3 ms.
- :97 `digest` = 8-byte multiply-rotate fold + length, over the whole TSV text.
- graph.rs:521-536 `refresh_crate`: `read_crate_text` (whole 2.5 MB file) then
  `digest(text)` then compare `lanes.digest_of(k)`: the 1.5-1.7 ms unchanged path is
  read + hash of the whole file. graph.rs:416 `open` -> `artifact::read_dir`.
- Precedent for in-place reads: canonical_node.rs:1929 `node_rows_from_le_bytes`
  (one `unsafe` cast, sound only because every field is a byte array — no native-
  endian integers). artifact.rs:115 already uses safe `as_chunks::<8>()`.
- Counting today is hand-rolled: xlog/src/wide.rs:478 `fold_counts` (BTreeMap
  multiset, ADD +1, RETIRE -1 floored at 0, in log order).
- Codebook surfaces read: contract codebook.rs:37-135 (`Codebook` u8, <=256, owns
  `HashMap<String,u8>`; `FamilyCodebookRegistry` family u16 -> Codebook); dt64
  lib.rs:29-68 + 433-468 (labels resolved late from the ClassView; `field_view`
  yields `FieldRow{position:u8,label}` = schema FIELDS, not a symbol table).

## 2. Format `<crate>.rows.bin` (all integers little-endian)
Rule (operator): lanes and every fold carry ONLY ids; names are labels resolved
late, never carried through a query. Names live in their own section(s), never
interleaved with lanes.

Fixed header, 64 bytes (the only bytes an unchanged check reads):
  0  [u8;8] magic  b"CHRNROWS"
  8  u16    major  (=1)    10 u16 minor (=0)
  12 u32    flags  bit0 = crate PARTIAL (mirrors the CRATE row, which stays a row)
  16 u64    digest (definition below)
  24 u64    file_len (whole file; truncation check before any section read)
  32 u32    n_rows   36 u32 n_symbols   40 u32 n_files   44 u32 n_sections
  48 [u8;16] reserved, must be 0
Section directory at 64: n_sections x 16 B {tag u32, elem_width u16, enc u16,
offset u32, len u32}; offsets 8-aligned, ascending, non-overlapping, inside
file_len; for a lane `len == n_rows*elem_width` (presence planes: ceil(n/64)*8).
Sections (tag -> content):
  PRED_MAG u8 lane + PRED_PRES u64 plane  codec9 codeword = predicate codebook id
  PRED_CB  KV (id -> UTF-8) the predicate labels, <=511 entries
  GRADE_MAG/GRADE_PRES + GRADE_CB         codec9, grade codebook (handful)
  FILE_MAG/FILE_PRES + FILE_CB            codec9 when n_files <= 511 (enc=CODEC9);
                                          else FILE u32 lane (enc=U32). "-" (no
                                          site) = codeword 0 = NEVER_TOUCHED, not
                                          a string entry.
  SUBJ, OBJ, MAYT  u32 lanes  crate-local symbol ids (MAYT: NO_ID=u32::MAX for
                   non-MAY rows; the writer precomputes the `Trait::method` id so
                   no reader splits strings)
  START, END       u32 lanes  NO_SPAN = u32::MAX; writer refuses usize > u32::MAX-1
  SYM_OFF + SYM_BLOB  KV table: (n_symbols+1) u32 offsets into one UTF-8 blob;
                   label(i) = blob[off[i]..off[i+1]] borrowed, validated once
                   (`str::from_utf8(blob)` + `is_char_boundary` per offset).
Codec9 mapping (operator: codebooks <=512 use codec9, -255..255, presence bit
flipped): codeword = `EpiLane9::codeword()` (9 bits, presence in bit 8); codeword
0 (= -0, NEVER_TOUCHED) = absent; codeword c in 1..=511 = codebook entry c. Batch
form = u8 magnitude lane + presence plane, 64 lanes per u64 (codec9 batch shape).
u32 stays where ids exceed 511: symbol ids (subject/object/may_target; mask-risc
has 17,155 rows and thousands of names), spans, and files of a crate with >511.
Why not a contract/DT64 surface for symbols: `Codebook` is u8 (<=256) and owns
Strings (a second copy of the label); a FamilyCodebookRegistry split would cut
names into arbitrary 256-buckets with no meaning; DT64 `field_view` labels ClassView
fields (positions <=255), not data symbols. So symbols use the KV fallback section.
Predicate/grade codebooks DO fit: their labels populate a contract `Codebook`
(or are resolved through it) at the graph, exactly as late labels.

Digest: the reader's existing `artifact::digest` fold (artifact.rs:97; 8 bytes per
step, length folded) over bytes [32 .. file_len) — every byte after the digest
and length fields, so it covers counts, directory, codebooks, lanes. Both repos
implement it from the spec (no cross-repo dep: codegraph does not depend on
ruff_charon and must not start); a committed golden vector (fixed bytes -> fixed
u64) is asserted in both test suites. The writer output is deterministic (rows in
`relations()` order, symbols in first-seen order), so equal content = equal digest.
Versioning: reader refuses magic mismatch, major != 1, minor > known, any unknown
tag with bit31 set ("must understand"), a missing required section, reserved != 0,
`file_len != actual size`, digest mismatch, any lane id >= its table size.
Interner (ruling 4, no strings in the hot path: never `Symbols::intern` at
open/refresh): writer adds SYM_HASH (u64 lane, `digest(name bytes)`)
and SYM_OWNER (u32 lane: local id of the name's crate prefix before "::", which
the writer interns into the table) per symbol. The graph-wide interner becomes
`u64 name-hash -> global id` plus, per global id, a label locator
`(crate slot, local id)` into that crate's retained file buffer. On a hash hit the
borrowed blob slices are compared byte-for-byte (no allocation) to refuse a
collision. Owner, `?dynamic` (a reserved global id, not an interned string) and
completeness (header flag bit0, replacing graph.rs:230 `symbols.get("COMPLETE")`)
are all ids/flags; resolve (graph.rs:128-142 `name.find("::")`) reads SYM_OWNER.

## 3. Zero-copy reading
- One `fs::read` of the file into one `Vec<u8>` per changed crate (the kernel copy
  is the intake edge). Lanes are read in place through safe lenses:
  `bytes[off..off+len].as_chunks::<4>()` + `u32::from_le_bytes` at use (the same
  idiom as artifact.rs:115); codec9 lanes as `&[u8]` + `&[u64]`-by-chunks. NO
  `unsafe` cast: the only precedent (canonical_node.rs:1929) is sound because its
  fields are byte arrays; a u32 cast would be native-endian. 8-aligned offsets
  keep a later aligned view possible without a format change.
- What is copied, honestly: the gather into the graph's resident `Vec<u32>` lanes
  (local -> global ids). That is a translation, not a duplicate of the file's
  meaning; true in-place borrowing of SUBJ/OBJ needs crate-local ids to stay the
  query currency (operator question 1). PRED/GRADE/FILE codec9 lanes and START/END
  can be appended as raw LE bytes (no translation needed).
- The retained per-crate buffer is the label source (KV section); refresh swaps
  the buffer; a retired buffer is kept while any global id's locator points into
  it (positions never renumber), dropped at reopen.
- `Lanes` keeps no `other_pred` text side table: predicates are codebook ids.

## 4. Changes
Writer (ruff_charon store.rs):
- `import` builds per crate: codebooks (predicate, grade, file), symbol table
  (first-seen order, plus owner prefixes and MAY targets), lanes; serialises
  `<crate>.rows.bin` via `replace` (same temp+fsync+rename; order rows.bin,
  residuals, files — files last unchanged).
- `rows.tsv` only when `Store::with_tsv_export(true)` / env `CHARON_TSV=1`.
  `Store::rows` reads rows.bin and builds `Row` (debug/test API, strings allowed
  there); `all_rows`, `dangling`, `crates` (files.tsv) keep their signatures.
  `stale` unchanged (source-file digests in files.tsv are a separate concern).
- `ROWS_VERSION` stays 2 for the TSV export; binary has its own major/minor.
Reader (lance-graph-codegraph):
- new `artifact_bin.rs`: `Header::read(&mut File)` (64 bytes, `read_exact`),
  `CrateFile::parse(Vec<u8>) -> Result<_, IntakeRefusal>` (all refusals of §2 as
  new `IntakeRefusal` variants: Magic, Major, Digest, Truncated, Section, IdRange).
- `open`: `*.rows.bin`; TSV reader kept only behind `#[cfg(test)]`/a debug fn
  (operator question 2).
- `refresh_crate`: read the 64-byte header; `digest_of(kid) == header.digest` ->
  return `RefreshStats::default()` without reading the body. Else read, verify,
  gather, then the existing edge-diff path.
- Strings at the edge only: `pos(&str)` in, `symbol(pos)`/`label` out. diag.rs
  already folds row masks (no `row_*` call found; keep it so). query.rs:82,169,250
  use `symbol(x).is_none()` as an existence test -> replace with an id-range test
  (`has_pos`). `row_predicate/row_subject/row_object` (graph.rs:740-751) become
  `#[doc(hidden)]` display/test helpers, documented "not for query paths"; 5 test
  call sites stay.
Counting (ruling 1: Quack's concern, executed by mask-risc on ndarray):
- `xlog/src/wide.rs:478 fold_counts` (BTreeMap) is replaced by two Quack
  `Agg::GroupReduce` calls (quack lib.rs:1224/1263) over a dense edge-ordinal key
  lane: `GroupAgg::Count` over base rows, `GroupAgg::SumI32` over the delta log's
  signed lane; effective count = Count + Sum, present = > 0. Owner: xlog calls
  Quack (as `bag_count`, xlog lib.rs:240, already does); codegraph never counts.
- The delta log's op lane becomes codec9 (+1 add, -1 retire, -0 = untouched).
- Two named gaps, not hand-rolled around: (a) the dense edge ordinal (rank of
  `(src<<32)|dst` among the distinct keys) is not a count — it comes from the
  sorted merge refresh_crate already performs; (b) `SumI32` needs an i32 column,
  and a codec9 lane is mag+presence: Quack needs a codec9-signed column reader
  (GroupFold over sign+magnitude) or one decode to i32 per batch. STOP and land
  that in Quack first if (b) is missing.
- Semantics change to verify: fold_counts floors RETIRE at 0 in LOG ORDER; a sum
  is order-free. They agree iff the log never retires more than is present at
  that point; refresh_crate computes retirements against the old multiset, so it
  holds there — pinned by a falsifier, not assumed.

## 5. Falsifiers (each with its disable run, red-then-green; commit before disabling)
- F1 round trip: writer emits bin + TSV for the same crate; reader resolves every
  row of both to labels; equal row for row (and `Store::rows(bin)==rows(tsv)`).
  Disable: swap SUBJ/OBJ lanes in the writer -> red. Second fixture with a
  P_OTHER-class predicate, a `-` site and a MAY row (shapes differ, assert so).
- F2 header-only unchanged refresh: keep the 64-byte header, overwrite the body
  with garbage -> refresh returns default stats (body provably unread). Disable:
  refresh reads + verifies the whole file -> Digest refusal -> red. Also count
  bytes via a reader that records `read` lengths: == 64.
- F3 refusals: flipped body byte (Digest), truncated file (Truncated), major+1,
  minor+1, missing SUBJ section, lane id >= n_symbols, reserved != 0, offset past
  file_len. Each a separate case; disable each check -> its case red.
- F4 no strings in open/refresh: a counting global allocator over `open` +
  changed `refresh_crate` on a fixture; allocations bounded by lane/buffer count,
  independent of n_symbols (double the symbols -> same count). Disable: intern
  labels -> red.
- F5 hash collision guard: two names forced to one SYM_HASH (test-only hasher
  hook) -> refused, not merged. Silence twin: distinct names never refused.
- F6 codec9 boundaries: codebooks of 511 (codec9) and 512 (refused / U32 enc for
  files); codeword 0 reads as absent, never as entry 0.
- F7 counting parity: GroupReduce(Count)+GroupReduce(SumI32) == old fold_counts
  on random logs that satisfy the no-over-retire precondition; a log that
  over-retires is shown to DIFFER (documents the precondition). Disable: drop the
  Sum term -> red.
- F8 cross-repo golden: committed tiny `*.rows.bin` + `*.rows.tsv` written by the
  ruff writer, read in codegraph tests; digest golden vector in both repos.
- Timing (release, mask-risc store, pre-registered, N>=40, report median+spread):
  unchanged refresh per crate <= 20 µs; changed crate open (17,155 rows) target
  <= 1 ms with the gather + hash map as the expected cost. If it lands >= 1 ms
  that is a failed gate: STOP and report where, do not ship. TSV numbers
  (6.8-10.3 ms / 1.5-1.7 ms) are the baseline quoted beside, not re-derived.

## 6. Steps and PR order
PR A (ruff, writer first):
- SONNET (ruff worker): `crates/ruff_charon/src/store.rs` + new `store_bin.rs`
  only: format writer, codebooks, SYM_HASH/SYM_OWNER, TSV export flag, `rows()`
  via bin, F1/F3(writer side)/F6/F8 fixture generation. No cargo.
- HAIKU (quote-harvester): verbatim header/section constants from store_bin.rs
  for the reader brief.
- ORCHESTRATOR: spec review, cargo test/clippy -p ruff_charon, disable runs, PR.
PR B (lance-graph, after A merges; fixtures copied from A):
- SONNET (codegraph worker): `lance-graph-codegraph/src/artifact_bin.rs`,
  `artifact.rs` (TSV behind test), `graph.rs` (open/refresh/resolve/accessors),
  `query.rs` (has_pos). No cargo.
- SONNET (xlog worker, disjoint file): `lance-graph-xlog/src/wide.rs` fold ->
  Quack GroupReduce, codec9 op lane — only after the Quack codec9-column gap (4b)
  is answered; if missing, a separate PR in lance-graph-quack precedes it.
- HAIKU (census-counter): every `symbols.name`/`.get(&str)`/`intern` call site
  remaining on open/refresh/query paths after PR B (must be edge-only).
- ORCHESTRATOR: gates (-p scoped), F2/F4/F7 disable runs, timing gate, board
  entry (`.claude/board/entries/` + index), PR.

## 7. Operator questions
1. Interner currency: keep graph-wide u32 ids (one gather per changed crate,
   F4-bounded) or make (crate, local id) the query currency so SUBJ/OBJ lanes are
   borrowed in place with no gather (bigger change to positions/cube)?
2. TSV in codegraph: remove the TSV reader from `open` entirely (bin only, TSV
   read only in tests), or keep a debug `open_tsv`?

## 8. Amendment A (2026-11-02): provenance header + per-row function identity

DECISION (operator, 2026-11-02, binding; supersedes nothing above, extends §2):
the format carries the artifact context in a header section and a function
identity per row, so one migration covers both. SCOPE: `<crate>.rows.bin`
and the codegraph reader. REVISIT WHEN: a second frontend (Rust MIR, MLIR)
writes rows.

**Canonical identity is never `FunDeclId` alone.** A Charon `FunDeclId` is an
index into ONE LLBC file's `fun_decls`; two LLBC artifacts reuse the same
numbers. The identity of a function fact is `(artifact, FunDeclId)`, where
`artifact` names the LLBC file and its provenance below.

### 8.1 Header: a `PROV` section (covered by the digest)
Not in the fixed 64 bytes: the unchanged check stays a header read (F2), and
the digest over [32 .. file_len) covers this section, so any provenance
change is a changed crate. One record per LLBC artifact the file was built
from (normally one):

| field | source (read, never assumed) |
|---|---|
| charon version | `CrateData::charon_version` (charon `export.rs:21`; charon-ml compares it for equality, so it is the de-facto LLBC schema key) |
| charon commit | the git HEAD of the charon checkout the ruff_charon build links (`ruff_charon/Cargo.toml` path dependency); captured at build, not at run |
| LLBC schema version | today the same value as `charon_version` (no separate schema field exists in `CrateData`); a separate field is written so a future split does not change the layout |
| LLBC content digest | `artifact::digest` over the LLBC file bytes the writer read |
| Rust toolchain | the `rustc -vV` commit-hash line of the toolchain Charon ran under |
| target + build config | `TranslatedCrate::target_information` keys (target triples) and the relevant `TranslatedCrate::options` (`CliOpts`: features, cfg, flags), serialised in a fixed field order |
| source revision | git HEAD of the analysed source tree plus a dirty flag; `UNKNOWN` when not a git tree, never a guess |

A writer that cannot fill a field writes `UNKNOWN` for it; the reader keeps
UNKNOWN distinct from every value (two UNKNOWN records are not "equal
provenance").

### 8.2 Per-row lanes
- `FUNDECL` u32 lane: the Charon `FunDeclId` of the function the fact belongs
  to, `NO_ID` when no function owns it (a CRATE row, a file-level fact).
- `ARTREF` lane (codec9, elided when the PROV section has one record =
  implicit 0): which PROV record the row's `FunDeclId` indexes.
- Source reference: the existing `FILE` + `START`/`END` lanes.
- Predicate: the existing predicate codebook id. Typed arguments: an
  `OBJ_KIND` codec9 lane says what `OBJ` holds (symbol id, or a word
  reference, below); no object is ever a string to be parsed.
- Evidence class: the existing `GRADE` codebook (COMPILER / … ), renamed in
  docs to "evidence class"; same lane.

### 8.3 `INVOKES_WORD`
A call whose callee is a behaviour-lens word gets its own predicate,
`INVOKES_WORD`, with `OBJ_KIND = WORD` and the object a registered DT64
reference: lens classid (u32 lane `WORD_LENS`) and word (u8, `WORD_IDX`),
i.e. an `AlgebraRef`; a `MethodRef` form is reserved (`OBJ_KIND = METHOD`).
- Registered, not named: the codegraph reader checks each reference against
  the DT64 `BindStore` (`declare_algebra` / the lens basis) and refuses an
  undeclared one as a named intake refusal. No second method catalogue, no
  name convention.
- OPEN, decided in its own reviewable step: where the association "this Rust
  function realises word W" comes from. Candidates: (a) the lens crate
  declares it next to the word (the POLYFILL table, read at build); (b) an
  attribute on the realising function, read by Charon. A name match is not a
  candidate.
- Repeated call sites stay distinct: each row keeps its own span, and the
  edge fold is a multiset (the P1 retry falsifier pins multiplicity).

### 8.4 Falsifiers added to §5
- F9 provenance: changing only the charon commit (or the toolchain, or the
  target) makes an unchanged-source crate a changed crate; an `UNKNOWN`
  field never compares equal to a known one.
- F10 identity: two LLBC artifacts with overlapping `FunDeclId` numbers in
  one crate file resolve to different functions; dropping `ARTREF` merges
  them (disable run).
- F11 `INVOKES_WORD` registration: an undeclared `(lens, word)` is refused at
  intake (disable: skip the check → accepted, red).
- F12 one fact set, two answers: `users_of(W)` (DT64: the method bindings
  whose effects reference W) and `callers(W)` (codegraph: the functions with
  an `INVOKES_WORD` row to W) answer different questions, so they are
  compared through the facts that join them: each binding's realising
  function. On one committed fixture, the realising functions of
  `users_of(W)` equal `callers(W)`. A second fixture adds a Rust caller of W
  with no binding; it must appear in exactly the difference (anti-vacuity:
  the difference is non-empty there and empty in the first).
