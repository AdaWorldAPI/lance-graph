# Ledger revision, 2026-10-23 (e): #1635–#1726

Maps the merges since revision (d) (`LEDGER-REVISION-2026-10-20d.md`, #1592–#1634)
onto the open list. Revision (d) and its predecessors are not edited; this file
supersedes the rows it names. Board entry:
`.claude/board/entries/2026-10-23-ledger-revision-1635-1726.md`.

**Scope.** 85 merged PRs, #1635–#1726, `main` at `19c1324a`. OGAR (`68cb29b`)
and ndarray (`37a3b59`) had no merges since revision (d). Read through a
script extraction of the merged diffs (`facts.md`, per PR: new files, touched
crates, board entries, STATUS rows) and four Haiku `harvest-scout` runs over
named search spaces. Every claim in §2–§4 that names a file:line was read on
the main thread at `19c1324a`; items taken from a board entry's own status line
without reading the code are marked *entry, not re-read*.

## 1. Withdrawn or corrected before they entered the list

| point | reading | now |
|---|---|---|
| "AGENT-CARDS: two cards have no frontmatter" (revision d) | `lance-cache-cartographer.md:1-4` now starts with `---` / `name:` / `description:`; both cards load (they appear in this session's agent list) | closed (C70) |
| "D-CGW-4: nobody calls `Reaction::verdict`" (revision d) | `lance-graph-planner/src/pearl.rs:259-263` `counterfactual_verdict()` calls `r.verdict(consistent)`; tests at `:747-815` | the call exists; what is still missing is the promotion gate (D-CGW-4 row: `Causes` only with `IncreaseEligible` ∧ `Necessary`). No production caller of `counterfactual_verdict()` outside `pearl.rs` tests was found; narrowed, not closed |
| "harvest-scout: `MaskOp::Gather` = 62 lines" | re-counted on the main thread: `grep -rn 'MaskOp::Gather' crates --include=*.rs \| wc -l` = 63 | 63 (revision d had 57); the scout's count was off by one |
| "ARM-B-PREPARE is still open" (revision d) | #1639 `plan_read` / `ReadPolicy` chooses the arm per demand; uniform demand measured | closed for uniform demand (C55); skewed demand remains open as part of D-RU-6 |

## 2. New closed ends (C53–C72)

| id | what closed | evidence |
|---|---|---|
| C53 | **The six decisions are recorded** with binding qualifications: ADMISSION B, D-RA-2 A, WIRE-50 A (presence outside 50..52), ACTION-GATE A (`commit_via` canonical, fail closed, `Public`/`Restricted`), FLOAT-REFS B (fold off by default), O30 A (≤32 `SourceRef`, cold sidecar) | #1640; `entries/2026-10-20-six-decisions.md` |
| C54 | D-RU plan (D-RU-0..8) and its first stages: U1 jc batch probe 6.4×; U2 `contract::set_program::SetProgram` + `relalg::program`; U3 cross-rule CSE | #1641, #1643, #1645, #1649, #1653; entries `2026-10-21-u2-let-planes.md`, `-u3-cross-rule-cse.md` |
| C55 | ARM-B-PREPARE for uniform demand: `plan_read` / `ReadPolicy` picks the addressed or masked arm | #1639 |
| C56 | D-CMR-1: `contract::set_algebra::{SetExpr, Res, Runs, normalize}`; N17 measured closed | #1644, #1646 |
| C57 | D-CMR-2: `lance_graph::fold::set_lower::{SetPlan, SortedBasis, Level, Measured}` | #1650 `f031089`; STATUS row Shipped |
| C58 | D-CMR-6: `contract::set_rules::{Arith, Truth, SumFamily, Population, Semantics, RowRange, Measure}` | #1651; `entries/2026-10-21-d-cmr-6-rule-catalogue.md` |
| C59 | **D-RA-2 shipped:** relalg repointed onto the contract set IR (decision A executed) | #1654, #1655; `entries/2026-10-21-d-ra-2-relalg-symbolic-repoint.md` |
| C60 | **The no-millisecond rule** in `CLAUDE.md` P0, with the tuple-is-a-violation-on-four-levels doctrine; zero-copy joins through a `Store` with lanes and one pointer table `$i` | #1657, #1659; entries `2026-10-21-coresearch-ms-vs-ns-folding.md`, `-zero-copy-join-index-map.md` |
| C61 | Kleene (K3) normalization, `normalize_kleene` / `Logic`, PostgreSQL K3 parity | #1668; `entries/2026-10-21-d-rft-0-kleene-normalization.md` |
| C62 | D-QSP-0 slices 1–2 (scalar aggregates and NULL operands under SQL NULL), with the #1686 / #1687 corrections (NULL is not FALSE or 0; K3 admission) | #1675, #1680, #1682, #1686, #1687 |
| C63 | **D-RFT-1 typed composition, and its cost:** first landing 310 µs (#1681); cost broken down (#1688: fixed per-invocation cost dominates, "fewer invocation boundaries" is the lever); #1685 reads the DR256 register directly: 11.8–12.4 µs, PostgreSQL lane parity kept | #1681, #1685, #1688; entries `2026-10-21-d-rft-1-typed-composition.md`, `-d-rft-1-cost-breakdown.md` |
| C64 | D-TGB-4 receipt, and the correction: a sealed landing cannot be replaced; `landed` requires a `CommitWitness` whose batch fold binds each cohort entry to its thought | #1674, #1684 (`5b62bc9a`); `entries/2026-10-21-d-tgb-4-receipt-replay.md` § Correction |
| C65 | GKAT probe (guarded Kleene algebra) and the D-LFA GKAT amendment | #1676, #1679, #1673 |
| C66 | Operator ruling: direct-register SIMD is best practice; "R2IL IS the SIMD polyfill" | EXO-50 thread after #1685 |
| C67 | Math compiler, stage B / E1: `egglog-r2il-probe`, `ReplayCapsule`, catalogue with oracle refusals and `CheckedRange`; the cold egglog phase is reported in µs, not gated | #1690, #1693, #1696, #1698, #1699, #1705; entries `2026-10-22-math-compiler-gap-matrix.md`, `-stage-b-e1.md` |
| C68 | D-MPS-9: MORK `query_multi` as a differential oracle (lab; not admitted, not wired) | #1690; `entries/2026-10-22-d-mps-9-mork-oracle.md` |
| C69 | **D-DRMC-0..5:** `dr-math-binding-probe` with `FrameStore`-issued `Frame`, separate `Focus` and `Scope`, validity carried by R (X1–X3); refuted before unknown (X4); `RecipeCertificate` that only a certified compiler can issue, so admission ends in `NoCertifiedCompiler` (X15); associativity of Boolean relation composition (D-DRMC-5) | #1700, #1707, #1708; entries `2026-10-22-coresearch-drmc-stack.md`, `-drmc-5-assoc-probe.md` |
| C70 | AGENT-CARDS: the two lance-cache cards have frontmatter and load | #1640 (headers) |
| C71 | Palette256 evidence arc, as measured verdicts: the property sits in evidence accounting plus a decision gate, not in the encoding. P-E2 and P-D3 pass; P-D4 fails, and its base-rate refit (#1725) fails too; P-S2: `ComposeTable` chains do not survive two hops (agreement 0.025 / 0.043; bind results leave the codebook support for 89–100 % of pairs) | #1709–#1726; entries `2026-10-22-coresearch-pal256-evidence-probes.md`, `-probes-pd4-pe2-pd3.md`, `2026-10-23-ps2-compose-chain-preregistration.md` |
| C72 | Stale STATUS rows from revision (d) fixed: D-CGW-9 (SUPERSEDED by D-HL-1 O1), D-CMR-2 (Shipped), D-CGW-0 (readings marked superseded) | STATUS_BOARD rows 137, 271, 128 at `19c1324a` |

Closed from (d): **D-RA-2** (C59), **ARM-B-PREPARE** for uniform demand (C55),
**AGENT-CARDS** (C70), **ROW-D-CGW-9 / ROW-D-CMR-2 / ROW-D-CGW-0** (C72), the
**D-RFT-1 performance gap** (C63). **DEPTH-BUDGET** is narrowed, not closed:
`set_algebra::normalize` and U3's CSE exist; whether depth ≤ 8 slots is now
checked as a rule was not read.

## 3. Open items, revised

Fruit: **L** low, **H** high, **Heavy**. Tier: O = operator decision, S = Sonnet,
H5 = a Haiku card (named). "Decision executed?" refers to C53.

### 3a. The six decisions: recorded, mostly not executed

The finding of this revision: the six rulings of 2026-10-20 exist as a record
(C53), but four of the five that need code have none.

| id | decision | read at `19c1324a` | status | fruit | tier |
|---|---|---|---|---|---|
| **ACTION-GATE** | A: `commit_via` canonical, fail closed, a `Public` / `Restricted` declaration | `lance-graph-contract/src/action.rs:266` `pub fn commit` still exists and checks role names only; `:328` `commit_via` checks roles and `grant_permits`; with `required_role: None` both skip RBAC, pinned by `commit_via_no_required_role_proceeds_regardless_of_rbac`. No `Public`/`Restricted` type exists | **NOT EXECUTED** | H | S (test first: flip the pinning test, then deprecate `commit`) |
| **FLOAT-REFS** | B: the fold path off by default | `crates/lance-graph/Cargo.toml:58` `lance-graph-r2il = { path = … }` is unconditional, and `lance-graph-r2il/Cargo.toml:19-20` pull OGAR `branch = "main"`. No `fold` feature in any manifest (scout over every `crates/*/Cargo.toml`, 39 read in full) | **NOT EXECUTED** | L–H | S |
| **WIRE-50** | A: keep the pyramid order; presence lives outside bits 50..52 (`Option<Maturation3>`) | `lance-graph-planner/tests/maturation_ladder.rs` is EXPERIMENTAL and test-local; no presence contract, no `Option<Maturation3>` outside tests | **NOT EXECUTED** | H | S |
| **ISA-LEGACY** (follows C40) | `isa.rs`'s opcode reading of 46..49 behind a declared legacy reading | no `ISA_LEGACY`, `legacy_opcode`, `LegacyRead` or `activation_i4` in `crates/` (scout, pattern set named in the record); `causal-edge/src/edge.rs` has `fn activation(self) -> i8` | **NOT EXECUTED** | H | S |
| **O30 / PROVENANCE** | A: ≤32 `SourceRef(table_id, version_witness, lane, row_position)` per derived cell, cold sidecar | the only `SourceRef` is `lance-graph-report/src/plan.rs:284` `{id, generation}`, a different concept | **NOT EXECUTED** | Heavy | S after the type is placed |
| **ADMISSION** | B: preparation may establish coordinates; execution operates on co-addressed sets; no hidden relational gather | ruling recorded; `set_lower.rs:9,30` cite it; `OutgoingHop` (`mask-risc/src/graph_hop.rs:55`) stays quarantined. The call-site classification (GATHER-CODE) is not done | ruling done, **code half open** | H | H5 harvest-scout (done, §7) → S classifies |
| **D-RA-2** | A | C59 | **EXECUTED** | — | — |

### 3b. Carried, re-checked

| id | item | status at `19c1324a` | fruit | tier |
|---|---|---|---|---|
| **GATHER-CODE** | Classify each site under ADMISSION B: `Filter::Semijoin` (35 lines), `Pred::EqU32Via` (54), `MaskOp::Gather` (63), `seek_from` (7), `gallop_partition_point` (16), `OutgoingHop` (7). Most are oracle and test sites; the live lowering sites are `lance-graph-quack/src/lib.rs:2077, 2092, 2296, 2390`, `mask-risc/src/exec.rs:569`, `reference.rs:437, 822, 922` | OPEN; candidate list in hand (§7) | H | S per site |
| **SUP-DISPATCH** | `DispatchToG` handles `ConsumerEnvelope::Health` only, everything else `DispatchNotImplemented` (`supervisor.rs:203-213`) | OPEN, unchanged | H | S |
| **SUP-CAPACITY** | `mailbox_capacity` field (`supervisor.rs:42`); `spawn_linked` of a `StubConsumerActor` (`:389`) takes no capacity | OPEN, unchanged | H | S |
| **SUP-DRAIN** | `lance-graph-callcenter/src/drain.rs:97` `mpsc::unbounded_channel()` | OPEN, unchanged | H | S |
| **DURABLEREF** | `DurableRef` / `ThoughtCommitId` only in doc comments (`mork-pathmap-probe/src/rmap.rs:70, :690`; `thought-git-probe/src/receipt.rs:40, :54`). D-TGB-4 now uses `BaseRef` as a stand-in and its entry names D-TGB-7 as the place for the contract type | OPEN, now with two consumers | H | S |
| **D-CGW-4** | narrowed (§1): the verdict is computed; the `Causes` promotion gate (`IncreaseEligible` ∧ `Necessary`) has no production caller | OPEN, Queued | H | S |
| **RECURSION** | D-RU-6 is Queued/Lab: the slot assignment must be argued as an attested frozen-codebook lookup under ADMISSION B; tile, sparse and dirty-frontier probes measured | OPEN | Heavy | O, then S |
| **D-CMR-5** | BLOCKED with a lawful design (preimage as a co-addressed transposed tile fold over a version-witnessed projection, built in preparation) | OPEN, design recorded | Heavy | S |
| **ISS-BF16-POLYFILL-PER-ELEMENT-ALLOC** | ndarray `simd_ops.rs:593-608`: two decode `vec!`s per call and `let mut col = vec![0.0f32; k]` inside the `j` loop, so 2 + 256 allocations per 16×16 call | OPEN, verified | H | S (ndarray) |
| **ISS-MORK-LIBRARY-GLOBAL-ALLOCATOR** | `ISSUES.md:63` OPEN | OPEN | — | O (upstream) |
| ogar_codebook placement, D-ASM re-spec, N16, N18, N20, TD-ONTOLOGY-REGISTRY, ABL, K6, EDGE-DOC, O23, O29, R2–R4, O26, O7/O13/O15, BAND-RETIRE, NAL W3 follow-ups, contract `hamming_256` scalar loop | unchanged from (d) | OPEN, *carried, not re-checked* | as (d) | as (d) |

### 3c. New open items (from #1635–#1726)

| id | item | source | fruit | tier |
|---|---|---|---|---|
| **ROW-D-RFT-1** | STATUS row 9 still reads "follow-up #1685 In PR" and "Open correction #1686"; both merged | STATUS_BOARD:9 | L | H5 spec-editor (applied in this PR) |
| **ROW-D-TGB-4** | STATUS row 394 does not mention the #1684 correction (commit witness, sealed landing) | STATUS_BOARD:394 | L | H5 spec-editor (applied in this PR) |
| **ROWS-MISSING** | No STATUS rows for D-DRMC-0..5, D-QR-SAP / IAM / CAP / POLY (plans #1661–#1663), D-RFT-1-COST, the Palette256 probes (P-D3, P-E2, P-D4, P-S2) | `grep` of STATUS_BOARD (navigation) | L | owner sessions; H5 spec-editor once wording is given |
| **CERT-COMPILER** | A certified-compiler boundary for DR64: a typed relational statement language, egglog laws over it with checked proofs, a lowering check. Until it exists every `RecipeCertificate` admission ends in `NoCertifiedCompiler` | `entries/2026-10-22-coresearch-drmc-stack.md` § Applied | Heavy | O (scope), then S |
| **DRMC-PROBES** | X6 (`validity: None` must refuse or say `Unattested`), X7 (no PopulationReport → premise conversion, compile-fail test), X8 (two certified bindings per key), X16 (a law found on class 1 must not transfer to class 22 unchecked), X18 (a census may veto, never qualify) — not run | same entry | H | S, one per probe |
| **FRAME-SEAL-OWNER** | Who seals the frame for real (non-probe) sources (G1) | same entry, OPEN | — | O |
| **PAL256-G5** | Where the evidence stamp lives (G5); P-D4-B (a separating world); separate π calibration; real labels; Q1 / S1 / S3 / P-C1 not started; whether a calibrated `PaletteLut` composes (P-S2's open point) | entries `-pal256-evidence-probes.md`, `-probes-pd4-pe2-pd3.md`, P-S2 entry | H / Heavy | S (lab) after G5 is placed |
| **D-RU-4** | FAILED GATE on speed under the no-ms rule; correctness measured | STATUS_BOARD:24 | H | S (re-plan as a fold) |
| **D-CMR-2b** | scratch-free `Pred` fold: no win on whole extents; per-run cost 2.5–4× on 200-row runs; the remedy is undecided | STATUS_BOARD:272 | H | O, then S |
| **D-QSP-REST** | remaining D-QSP slices: column-to-column (G4), COALESCE / NULLIF, nullable arithmetic, `NOT IN` subqueries | D-QSP plan | H | S |
| **D-LFA-1/2** | guarded-control verification of a Rubicon flow; PROPOSAL | STATUS_BOARD:11 | H | S |
| **D-MPS-9-REST** | other MORK verbs (`transform_multi_multi`, `remove_sexpr`) and admission | STATUS_BOARD:44 | H | S (lab) |
| **MATH-P0** | catalogue gates: P0 premise consistency; `Exact` → checked-overflow policy; E2–E5 not run | math-compiler entries | H | S |
| **THOUGHT-GIT-END** | the intended end state is about a million RAM explorations, Sudoku admission, revision and a minimal proof under a defined metric, with three artifacts (proof, replay receipt, commit witness). #1684 implements receipt and witness only | operator, EXO-50 | Heavy | O (metric), then S |
| **TUPLE-SITES** | EXO-103 / 104 / 116 / 117 / 118 tuple-site backlog under the no-ms rule; EXO-111 raw AVX-512 intrinsics in contract `mul.rs`; EXO-112 contract → ndarray dependency shape | Linear, *entry, not re-read* | H | S; H5 harvest-scout for the census |

## 4. Connecting the dots

- **K19 — the decisions are a queue, not a state.** C53 recorded six rulings;
  one (D-RA-2) was executed. Every subsequent wave (D-RU, D-CMR, D-DRMC,
  Palette256) ran beside them, not through them. The cheapest high-signal
  work in the repository is now executing rulings that are already made, not
  making new ones: ACTION-GATE, FLOAT-REFS, WIRE-50 + ISA-LEGACY, O30 each have
  their wording in `six-decisions.md` and need no further operator input.
- **K20 — WIRE-50, ISA-LEGACY and the CE64 upper half are one PR.** All three
  touch `causal-edge` / `wire.rs`, all three are field-isolation work under
  I-LEGACY-API-FEATURE-GATED, and doing them separately re-pins the same
  tests three times. This is revision (d)'s K17, still unexecuted.
- **K21 — one missing type behind PROVENANCE, DURABLEREF and THOUGHT-GIT-END.**
  O30's `SourceRef(table_id, version_witness, lane, row_position)`,
  `DurableRef(dataset, branch, version)` and the commit witness's cohort
  entry all identify "which persisted bytes a derived thing read". D-TGB-7 is
  already named as the contract admission point for `DurableRef`. Placing
  `DurableRef` first gives `SourceRef` its `version_witness` field, and the
  commit witness its anchor; the three items then share one type family
  instead of three local stand-ins (`BaseRef`, `{id, generation}`, doc comments).
- **K22 — admission keeps reappearing under different names.** ADMISSION B
  (preparation may establish coordinates), D-DRMC X7 (evidence nominates, a
  validator admits), X15 (`NoCertifiedCompiler`), P-D4 / G5 (evidence
  accounting plus a decision gate), D-RU-6 (slot assignment as an attested
  frozen-codebook lookup) and D-CMR-5 (a version-witnessed projection built in
  preparation) are the same question at four layers: what may be built before
  execution, under what witness, and what may never be promoted from it. A
  single admission vocabulary (witness kinds, who issues them, which
  promotions are refused) would let each layer cite one record instead of
  re-deriving it. This is a candidate, not a ruling.
- **K23 — the direct-register result generalizes as a measurement rule.**
  C63 went 310 → 12 µs by removing invocation boundaries, not by changing
  mask work (#1688: the work is a few ns). D-RU-4 failed its gate on the same
  axis, and the BF16 polyfill measures allocation rather than arithmetic.
  Before any further fold optimization, the first question is the count of
  invocation and allocation boundaries per answer.
- **K24 — Palette256 has a clean negative result to build on.** P-S2 and
  #1721 together show derived XOR-bind laws do not compose unless the
  codebook is closed under the bind. That turns "can palette laws compose?"
  into "fill a `PaletteLut` from observed transitions and test closure", and
  it keeps Z-add (naive Bayes) separate from NARS revision, as ruled.
- **K25 — FLOAT-REFS is a two-line change with a policy shadow.** Decision B
  puts the fold path behind a feature; doing so changes what `lance_graph::fold`
  re-exports by default and therefore every downstream that uses it without
  naming a feature. The change is small; the consumer audit (who imports
  `lance_graph::fold`) is the real work and is Haiku-sized.

## 5. Fruits

### Low-hanging (mechanical, Haiku-led, verified by the orchestrator)

1. **ROW-D-RFT-1, ROW-D-TGB-4** — applied in this PR by `spec-editor` from an
   exact edit spec; diff read before commit.
2. **ROWS-MISSING** — `census-counter` lists every D-id in
   `INTEGRATION_PLANS.md` and plan files with no STATUS row (a closed space:
   two directories); owner sessions supply the wording; `spec-editor` applies it.
3. **FLOAT-REFS consumer audit** — `harvest-scout`: every `lance_graph::fold`
   import in the workspace and sibling repos. Gives K25 its blast radius.
4. **GATHER-CODE triage list** — `quote-harvester` over the eight live
   lowering sites named in §3b, so the Sonnet classification reads quotes,
   not files.

### High-signal (one Sonnet slice each, test first, decision already made)

1. **ACTION-GATE** — flip `commit_via_no_required_role_proceeds_regardless_of_rbac`
   to expect denial, add the `Public` / `Restricted` declaration, route
   `commit` through `commit_via` or deprecate it. Why: an unauthenticated
   action path that the operator has already ruled closed.
2. **WIRE-50 + ISA-LEGACY** (K20) — one PR: presence outside 50..52, the
   legacy opcode reading behind a declared reader, the field-isolation matrix.
3. **FLOAT-REFS** — the `fold` feature, after the audit.
4. **SUP-DISPATCH / SUP-CAPACITY / SUP-DRAIN** — three small, independent
   supervisor slices; capacity first (it is a field already read nowhere).
5. **DRMC-PROBES** X6, X7, X18 — each is a single falsifier on an existing crate.
6. **D-CGW-4** — wire the promotion gate onto `counterfactual_verdict()`.
7. **BF16 outer-product polyfill** in ndarray, then re-run D-RMAP-2 §5a.

### Heavy (needs a placement or scope decision first)

1. **DurableRef → SourceRef → commit-witness anchor** (K21), entering the
   contract at D-TGB-7. Unblocks PROVENANCE, DURABLEREF and THOUGHT-GIT-END.
2. **CERT-COMPILER** for DR64 — typed relational statements, egglog laws with
   checked proofs, a lowering check; first law: associativity of Boolean
   relation composition (C69 already measured it).
3. **RECURSION** under ADMISSION B (D-RU-6) and **D-CMR-5**'s preimage fold.
4. **PAL256-G5** placement, then P-D4-B and a calibrated `PaletteLut` closure test.
5. **THOUGHT-GIT-END** — the proof artifact and its metric.

## 6. Questions for the operator

| # | question | why it blocks |
|---|---|---|
| Q1 | Where does `DurableRef` live (contract at D-TGB-7, as the D-TGB-4 entry proposes)? | K21; PROVENANCE and THOUGHT-GIT-END wait on it |
| Q2 | Scope of the DR64 certified-compiler boundary (CERT-COMPILER) | every `RecipeCertificate` ends in `NoCertifiedCompiler` until it exists |
| Q3 | Who seals frames for real sources (FRAME-SEAL-OWNER, G1) | D-DRMC beyond the probe |
| Q4 | Palette256 stamp home (G5) | P-D4-B and calibrated laws |
| Q5 | The proof metric for THOUGHT-GIT-END | the third artifact |
| Q6 | D-CMR-2b remedy, or close it as measured-no-win | keeps a lab item from going stale |

The six decisions of 2026-10-20 need no further input; they need execution (K19).

## 7. Agent ledger (Haiku harvest tier)

| card | runs | items checked by the orchestrator | wrong | missed |
|---|---|---|---|---|
| substrate-mapper (diff batches b01–b17) | 17 started, all stopped | — | — | the diffs averaged ~110k tokens per 472 lines; replaced by a script extraction. Lesson: raw multi-hundred-KB diffs are not Haiku-sized; use a script for mechanical fields and Haiku for closed questions |
| harvest-scout (FLOAT-REFS manifests) | 1, DONE | `crates/lance-graph/Cargo.toml:51-58` read; no `fold` feature confirmed | 0 | 0 known |
| harvest-scout (WIRE-50 / ISA-LEGACY / SourceRef) | 1, DONE | `maturation_ladder.rs`, `plan.rs:284`, `receipt.rs:40,54` read | 0 | 0 known |
| harvest-scout (gather call sites) | 1, CAPPED at 150 | `MaskOp::Gather` re-counted: 63, scout 62 | 1 (count off by one) | capped tail named by the scout |
| harvest-scout (Reaction::verdict, MORK allocator, BF16) | 1, DONE | `pearl.rs:235-275`, `simd_ops.rs:587-622`, `ISSUES.md:54-63` read | 0 | the scout did not see the `counterfactual_verdict` caller because it searched the type, not the method; the main thread found it |
| spec-editor (STATUS rows 9, 394) | 1 | diff read before commit | see the commit | — |
