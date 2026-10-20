# Ledger revision, 2026-10-20 (d): #1592–#1634

Maps the merges since the last mapping (`LEDGER-REVISION-2026-10-16c.md`,
#1582–#1591) onto the open list. That file and its predecessors are not
edited; this one supersedes the rows it names. Board entry:
`.claude/board/entries/2026-10-20-ledger-revision-1592-1634.md`.

**Scope.** 43 merged PRs, #1592–#1634, `main` at `abfa9e2`. Read through ten
Haiku `substrate-mapper` fact maps over the merged diffs (one per batch), then
checked on the main thread against the code at `main`: every cited file:line in
§2–§4 was read there. Items carried from a session report that were not
re-read are marked *carried, not re-checked*.

## 1. Withdrawn or corrected before they entered the list

Three points from the session report that preceded this revision did not
survive a read of the code:

| point | reading | now |
|---|---|---|
| "STATUS_BOARD shows D-CMR-0 and D-RA-1 as In PR" | both rows read Shipped (#1532, #1546 `c21bd165`) | withdrawn; #1547 fixed it |
| "D-CMR §3.2 says lowering lives in core, which is wrong" | §3.2 says core *depends on* mask-risc, Quack and R2IL; that is #1537 step (b) | withdrawn; the plan is consistent |
| "`ActionInvocation::commit` has an admin bypass" | `action.rs:256`: "Admin is an ordinary role here"; neither `commit` nor `commit_via` bypasses | corrected in ACTION-GATE below |
| "the unbounded drain is a supervisor gap" | `mpsc::unbounded_channel` is at `lance-graph-callcenter/src/drain.rs:97`, not in the supervisor | moved to SUP-DRAIN under the callcenter |

## 2. New closed ends

| id | what closed | evidence |
|---|---|---|
| C39 | Bits 43..45 = `causal_edge::Orientation`; `pack`/`pack_v2` take it; the S/P/O triad removed (D-CGW-1) | #1594 `8f50cd9e`; STATUS D-CGW-1 Shipped |
| C40 | **D-46 decided, the other way from revision (c)'s recommendation:** 46..49 = activation i4 (+7 excitation … −8 inhibition); the operation stays in the running thought; `isa.rs`'s opcode reading becomes a declared legacy reading | `docs/architecture/ce64-semantic-upper-half.md` 2026-10-17 DECISION block; STATUS D-HL-1 (O1 decided) |
| C41 | W (53..58) = the thought layer (0 givens, 1..63 hypotheses); the witness-handle reading and `WitnessTable` as a foreign key are withdrawn; EXO-51 withdrawn | #1597 (2026-10-17), restated #1628 (2026-10-20) |
| C42 | CE64 is an ISA contract, not a storage contract: inside a live thought W names the lane; after a Lance seal it is arbitrary | #1628, entry `2026-10-20-ce64-isa-not-storage.md` |
| C43 | Thought-git lab probe: oracle + signals + slot pool (63 live slots, FIFO / least-used eviction), 19 + 6 tests, 5 oracle disables red | #1630, #1631; `crates/thought-git-probe`; D-TGB-0/1 Shipped (lab) |
| C44 | **The relalg width limit is gone.** `r2il::call_owned` releases consumed slots, so an `Or` / `And` / `And`-of-`Not` of any width runs in 3 slots (was `2k − 1`, 5 atoms refused) | #1634; `lance-graph-r2il/src/lib.rs:588`; tests `call_owned_reuses_consumed_slots_and_call_does_not`, `a_wide_disjunction_keeps_its_rows_in_constant_slots` |
| C45 | Crepe as a cold optimizer in front of relalg, measured: verdict PARTIALLY (static path proven, dynamic rules need a `crepe-core` split); 3.6 µs vs 11.8 ms terminal at 50,000 rows, break-even reuse 0 | #1631–#1634; `crates/crepe-plan-probe`; entry `2026-10-20-crepe-rule-optimizer-spike.md` |
| C46 | Addressed read (Arm B) over Lance 13 / lancedb 0.40: sorted `FacetCascade` keys; per query it scans `3k` rows where the masked arm scans ≥ 150,000, 2.5–12× faster up to 1 % of centers; the one-time attestation is a full scan of both key columns (7–10 ms for 100,002 keys), repaid after 2–3 queries; stale lanes, unsorted tables and foreign datasets refused | #1633; `crepe-plan-probe/src/addressed.rs`; tests b1–b12 |
| C47 | MORK/PathMap probes measured: one `pathmap` package, 0 / 18 upstream files differ (D-MPS-0); differential probe (D-MPS-1); native join over a versioned relationship table, 7 guards disable-verified (D-MPS-8); focus-bound RelationshipMap probe, break-even 76 / 11 / 1.8 reuses (D-RMAP-1A-P) | #1620, #1621; `crates/mork-pathmap-probe` |
| C48 | D-RMAP-2 partly measured: polyfill tier and AMX tile tier; BF16 byte total corrected to 384 KiB; re-run shows absolute times 10–60 % higher (VM load, unverified) with ratios intact | #1623, #1625–#1627 |
| C49 | NAL revision of stratum beliefs equals the union recount (P-pool probe) | #1596; `lance-graph-planner/tests/simpson_revision_pooling.rs` |
| C50 | Experimental ACT-R production kernel (deterministic, refuses NaN, no storage, no effects) | #1615, renamed `crates/lance-graph-act-r` in #1618 |
| C51 | Post-merge recovery: #1601–#1617 merged with no board rows and 17 unresolved threads; reconciled | #1618; catch-up #1627 |
| C52 | Haiku 5.5 limits cited from platform docs on the six cards | #1598 |

## 3. Open items, revised

Fruit: **L** low, **H** high, **Heavy**. Tier: who does the grindwork (O = operator
decision, S = Sonnet, H5 = a Haiku card).

| id | item | status after #1634 | fruit | tier |
|---|---|---|---|---|
| **ADMISSION** (merges GATHER-DECISION, GATHER-CODE's ruling half, ISS-ADMISSION-PREP-BINDING, ISS-GATHER-HOP-1616) | One written ruling on what a read in *preparation* may do. ISS-ADMISSION-PREP-BINDING phrases it: "a counted, witnessed lookup in preparation, folds co-addressed". It settles D-MPS-8, D-RPR-1, D-RST-4/6 and the fate of the quarantined `graph_hop.rs` (#1616). Still no DECISION record of the 2026-10-16 ban itself. | OPEN, operator ruling; **highest** | — | O words it, S records |
| **GATHER-CODE** (code half) | `Filter::Semijoin`, `Pred::EqU32Via`, `MaskOp::Gather` (57 matching lines, AGENT_LOG recount), `seek_from`, `gallop_partition_point`, `is_alpha_acyclic`, `graph_hop.rs` | OPEN, waits on ADMISSION | H | O per item, S |
| **D-RA-2** | Rules into the contract set IR. Now gates four lines: D-CMR-1/6, D-SRAM-2 / crepe IR ownership ("D-RA-2 still gates any IR ownership", #1631), N17's absorption fix, D-RA-3 | Queued (maintainer decision) | — | O |
| **DEPTH-BUDGET** (new; narrows the D-CMR feasibility point) | After C44 the 8-slot budget binds only on nesting depth (`relalg/tests/rewrites.rs:613` `depth_still_costs_slots`). Flattening associative `Or`/`And` is the first rule D-CMR-6 needs. | OPEN | H | S |
| **RECURSION** (merges ISS-RECURSION-DRIVER, D-SRAM-4, D-MTP-3) | D-RA-4's driver is banned; D-SRAM-4 is BLOCKED on it; crepe refuses recursion "needs the D-MTP-3 driver" | OPEN | Heavy | O, then S |
| **PROVENANCE** (merges ISS-RETRACTION-PROVENANCE, D-HL-3, O25 + K2', O30, D-TGA-3, D-RPR-2) | No source bitset or read-provenance plane in `crates/`. Until it exists a changed premise can only be refused, never selectively retracted. C37 (modulo-64 stamp) and AriGraph's stamp-less revision remain the measured failures. | OPEN, O30 decision first | Heavy | O, then S |
| **DURABLEREF** | `DurableRef` / `ThoughtCommitId` exist only in doc comments (`mork-pathmap-probe/src/rmap.rs:70`, `:690`); D-RMAP-1A-V and CPG-00 assume them | OPEN | H | S |
| **WIRE-50** (+ D-MATURATION3-1, O2) | 50..52: pyramid order vs `with_novelty_wisdom` order. The only CE64 upper-half field left without a ruling. | OPEN, decision | — | O |
| **ISA-LEGACY** (new, from C40) | Move `isa.rs`'s opcode reading of 46..49 behind a declared legacy reading; field-isolation tests per I-LEGACY-API-FEATURE-GATED | OPEN | H | S |
| **ROW-D-CGW-9** (new) | STATUS row D-CGW-9 still says "46..49 stays the ISA inference mantissa" — superseded by C40; the 2026-10-17 block in the ce64 doc already says so | OPEN | L | H5 spec-editor |
| **ROW-D-CMR-2** (new) | STATUS row D-CMR-2 reads "Queued (after contract → mask-risc)" — that edge is refused (no contract → mask-risc; lowering through core, #1537) | OPEN | L | H5 spec-editor |
| **D-CGW-4 ↔ C22** | unchanged: call `Reaction::verdict` | OPEN, Queued | H | S |
| **SUP-DISPATCH** | `DispatchToG` handles Health only (`supervisor.rs:203-211`) | OPEN | H | S |
| **SUP-CAPACITY** | `mailbox_capacity` is a manifest field (`supervisor.rs:21-42`) but `spawn_linked` (`:389`) takes no capacity | OPEN | H | S |
| **SUP-DRAIN** | `lance-graph-callcenter/src/drain.rs:97` `mpsc::unbounded_channel` | OPEN | H | S |
| **ACTION-GATE** (corrected) | Two gates, same order (terminal → def-match → RBAC → guard → MUL): `commit` checks role names only; `commit_via` also asks `grant_permits`. With `required_role = None` both skip RBAC. Open: which gate is canonical, and whether `None` should deny by default (capstone item 7). | OPEN, decision | H | O, then S |
| **FLOAT-REFS** (new) | Git branch references: core's default build reaches OGAR `main` through `lance-graph-r2il/Cargo.toml:19-20`; the probes add crepe `ccr-d3ce1263-ied9yh`, datafrog `master`, ndarray `master` (PathMap is `rev`-pinned). A default build can change without a lance-graph commit. | OPEN, policy | L to decide | O |
| **AGENT-CARDS** (new) | `lance-cache-cartographer.md`, `lance-cache-contents-auditor.md` have no frontmatter, so they never load; `.claude/agents/README.md` names 33 cards of 59 | OPEN | L | H5 census-counter + spec-editor |
| **D-TGB** | squash-or-land OPEN; card granularity OPEN; kanban vocabulary OPEN by choice; EXO-98..101 open for any session | OPEN | H | S (lab) |
| **ARM-B-PREPARE** (new) | Semijoin preparation is still O(n) (every row tested once); Arm B removes it below ~1 % selectivity, not above | OPEN | H | S |
| **MORK** | ISS-MORK-LIBRARY-GLOBAL-ALLOCATOR (OPEN); D-MPS-7 OPEN (repository-level clearance incomplete) | OPEN | — | O |
| **ISS-BF16-POLYFILL-PER-ELEMENT-ALLOC** | per-element allocation in the BF16 polyfill tier | OPEN | H | S (ndarray) |
| NAL W3 follow-ups | statement-ordinal key; the w-lane as a second reading of `c`; v4 unmeasured | OPEN, *carried, not re-checked* | H | S |
| `ogar_codebook` placement | 302 hex ids in the contract against the consumer-owned-classid rule | OPEN, decision, *carried* | Heavy | O |
| Contract scalar kernels | `grammar/context_chain.rs:545` `hamming_256` scalar loop | OPEN, *carried* | L | S |
| D-ASM re-spec, N16, N18, N20, TD-ONTOLOGY-REGISTRY, ABL, K6, EDGE-DOC, O23, O29, R2–R4, O26, O7/O13/O15, BAND-RETIRE | unchanged from (c) | OPEN | as (c) | as (c) |
| **ROW-D-CGW-0** (carried from (c), widened) | STATUS row D-CGW-0 still reads "In PR" although #1591 merged, and its description still says "W = Markov arc in `WitnessTable<64>`", superseded by C41 | OPEN (owner session) | L | H5 spec-editor |

Closed from (c): **D-46** (C40), **DOC-WIRE (1)** for 43..45 (C39; 50..52 stays under WIRE-50),
**ARC-GAPS** (done in #1592).

## 4. Connecting the dots

- **K13 — one admission ruling, and it already has a measured example.**
  ISS-ADMISSION-PREP-BINDING asks for "a counted, witnessed lookup in
  preparation, folds co-addressed". Arm B (C46) has that shape, in two
  separate costs. **Attestation** is a one-time full scan of both key
  columns of every row (`addressed.rs:100-117`, `attest` → `read_keys`;
  7–10 ms for 100,002 keys over three tables), done once per table version
  and refused if the table is not sorted; it is what the witness rests on.
  **Each query** then reads only sorted ranges of the table's own key, and
  only these reads are counted (`SideIo.rows_scanned`, `3k` at 50,000 rows);
  a stale lane or a foreign dataset is refused (witnessed). The ruling has
  to name both: the per-query read is counted, the attestation scan is a
  stated one-time cost per version. Writing ADMISSION with Arm B as the worked example and
  `graph_hop.rs` as the counter-example (a hop whose positions come from
  another lane's values) answers D-MPS-8, D-RPR-1, D-RST-4/6 and GATHER-CODE
  in one record.
- **K14 — D-RA-2 has four consumers now.** D-CMR, crepe / D-SRAM-2, N17 and
  D-RA-3 each wait on it. Two rule sets would drift; one decision ends it.
- **K15 — the slot budget is a depth budget.** C44 made width free. A
  normalization pass that flattens associative chains first and then checks
  depth ≤ budget turns feasibility into one testable rule (DEPTH-BUDGET),
  and it is the first rule of D-CMR-6's catalogue.
- **K16 — one missing primitive behind four blocked items.** PROVENANCE,
  D-TGA-3, D-RPR-2 and O25 all need the same thing: which sources a derived
  cell read. D-HL-3 pre-registered it as a source bitset per delta. C41 gives
  a natural candidate width (64 thought slots), but that is a candidate, not a
  ruling; O30 decides.
- **K17 — the CE64 upper half is one field from done.** 43..45 (C39),
  46..49 (C40) and 53..58 (C41) are ruled; WIRE-50 is the last. ISA-LEGACY,
  ROW-D-CGW-9 and the 50..52 ruling belong in one test-first PR.
- **K18 — reproducibility is a policy, not a fix.** FLOAT-REFS is three probe
  references and one default-build reference. The default-build one is the
  only one that changes what `cargo build` gives a consumer.

## 5. Fruits

- **Low:** ROW-D-CGW-9, ROW-D-CMR-2 and ROW-D-CGW-0 (spec-editor); AGENT-CARDS (headers +
  README from a census); dated correction of the session report's withdrawn
  points (already done in §1); the FLOAT-REFS policy question put to the
  operator.
- **High:** ADMISSION written (K13); DEPTH-BUDGET as D-CMR-6's first rule
  (K15); ISA-LEGACY; SUP-DISPATCH, SUP-CAPACITY, SUP-DRAIN; ACTION-GATE after
  the ruling; DURABLEREF; ARM-B-PREPARE; D-CGW-4 → `Reaction::verdict`;
  D-TGB-7..10 in the lab.
- **Heavy:** D-RA-2 and what follows it (K14); PROVENANCE (K16); RECURSION
  inside the ban; D-ASM re-spec; `ogar_codebook` placement.

## 6. Decisions for the operator

| # | question | recommendation | caveat |
|---|---|---|---|
| ADMISSION | what a preparation-time read may do | admit counted, witnessed, same-table key-range seeks (Arm B shape), with the one-time full key scan that attests each table version stated as its own cost; refuse reads at positions taken from another lane's values | `graph_hop.rs` then stays quarantined or is rewritten co-addressed |
| D-RA-2 | where single-table rules live | the contract set IR; relalg keeps lowering | relalg tests must pass unchanged against the contract IR (its own gate) |
| WIRE-50 | order of 50..52 | as (c): keep the pyramid; "not on the ladder" only through D-MATURATION3-1 decision 3 | `wire.rs` re-pin |
| ACTION-GATE | canonical gate; default for `required_role = None` | `commit_via` canonical; `None` allowed only for actions declared public | every `ActionDef` without a role needs a declaration |
| FLOAT-REFS | git branch refs in the default build | pin OGAR by `rev` in r2il, or put the fold path behind a feature | a `rev` pin needs a bump per OGAR change |
| O30 | evidential-base width | unchanged from (c) | K16 makes it the gate for four items |

## 7. Agent ledger (Haiku harvest tier)

| card | runs | items checked | wrong | missed |
|---|---|---|---|---|
| substrate-mapper (one per batch, #1547–#1634, 86 PRs) | 10 (9 COMPLETE, 1 PARTIAL: #1573, a merge-heavy diff, read by grep in three ranges, self-reported) | new-file list of every PR against `new file mode` in the source: equal in 83; in #1571, #1607, #1608 the mapper also listed JSON files that the dump excludes from the diff but shows in the stat, and `git log --diff-filter=A` confirms they are new; claims in map 10 read at `relalg/tests/rewrites.rs:613` and `r2il/src/lib.rs:588` | 0 | 0 known |

## 8. Decisions recorded (2026-10-20, later)

The six questions in §6 were decided the same day: ADMISSION B, D-RA-2 A,
WIRE-50 A, ACTION-GATE A, FLOAT-REFS B, O30 A (32 sources at first), each with
binding qualifications. Record: `.claude/board/entries/2026-10-20-six-decisions.md`.
The §6 table above is kept as the question set; the record supersedes its
recommendation column where they differ (WIRE-50 adds a presence condition
outside bits 50..52; O30 sets 32 and a versioned `SourceRef`).
