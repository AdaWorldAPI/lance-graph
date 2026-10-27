# D-SAG-6-W — Dogfood MCP workbench: a read-only project-graph projection

**Date:** 2026-10-27. **Status:** PROPOSAL + LAB FIXTURE (`crates/sag6-workbench-probe`, read-only, own `[workspace]`, excluded from the root workspace). **Implementation:** W1 lab only; deployment NOT authorized. No production Rust in the root workspace, no ABI, no policy, no storage, no CI and no Railway change is authorized by this document.
**Parent:** D-SAG-6 [#1760](https://github.com/AdaWorldAPI/lance-graph/pull/1760) (merged, 4c8f5b9; `d-sag-6-rust-codegraph-math-logic-tracert-v1.md`). **Coordination:** [Linear EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id), an append-only bus, not an approval.
**Related:** [DT64 board entry](../board/entries/2026-10-26-dt64-holographic-interface.md) (docs merged, no code). Source pins are lance-graph main `aa569bc5c41ca38edf1ce3db945007543b0c82f6`.

## 0. Thesis and authority boundaries

The workbench answers questions about our own project (issues, pull requests, source symbols, debt rows) as **S–P–O facts**:

- **S** is a source-attested identity: a source system, that system's upstream key, and a source revision where the key is revision-scoped.
- **P** is a predicate whose meaning resolves through a `ClassId`. No predicate here has one yet: each is a fixture-local key with `class_id: None` and, where one exists, an OGAR convergence candidate that is reported but not bound.
- **O** is a typed edge to another identity, carried with a witness (the snapshot it was read from).

Every answer is one of five outcomes (§3), never a bare boolean.

**This amendment is NOT:** a second graph engine; a DT64 runtime (DT64 has docs only, `grep -E 'Dt64|DT64|dt64' crates/**/*.rs` finds 0 files); DR64³ (NFR-OP-2 is OPEN, "no DR64³ runtime"); method execution; a writer to GitHub, Linear or source; a minter of OGAR concepts; an approval for deployment.

## 1. Placement in D-SAG-6

D-SAG-6-2 is the Rust MIR/Codegraph projection and stays future. D-SAG-6-W is a **project-graph projection beside it**: the same discipline (source identity, frame-bound ordinals, UNKNOWN is not FALSE) applied to issue, PR and symbol records instead of MIR. It borrows the parent's F-gates (§5) and adds no new tier to the tower.

| D-id | Deliverable | Gate | Status |
|---|---|---|---|
| **D-SAG-6-W0** | this document | scope and authority statements explicit | docs only |
| **D-SAG-6-W1** | lab fixture `crates/sag6-workbench-probe` | §5 gates with independent scalar baseline and disable-red-runs | LAB, 16/16 tests, 8/8 disables red (2026-10-27) |
| **D-SAG-6-W2** | read-only MCP surface (`rmcp`, streamable HTTP) over the W1 query contract | all W1 gates unchanged; no write tool exists | PROPOSED, separate approval |
| **D-SAG-6-W3** | `/board` view over the same query contract | same contract; no second query path | PROPOSED, separate approval |
| **D-SAG-6-W4** | Railway deployment | separate operator authorization (§7) | NOT AUTHORIZED |

## 2. Existing-code reuse matrix

Verdicts: REUSE-BY-DEPENDENCY, REUSE-SHAPE (copy the shape, not the type), CONVERGE (align with an existing owner, bind later), NOT-REUSABLE, MISSING.

| Concern | Existing code | Verdict | Note |
|---|---|---|---|
| Frame seal and staleness | `dr-math-binding-probe/src/lib.rs` `FrameStore{new, seal, advance, check -> Result<(), Refusal>}`; `Refusal::{ForeignFrame, SourceChanged, MapChanged, CoordinatesChanged, FrameResealed, ...}` | REUSE-BY-DEPENDENCY | Frames compare by identity of issue (`Weak::ptr_eq`), not value. Pulls egglog (git) and sha2; `cargo check` 53 s cold. |
| One-hop evaluation | `lance-graph-quack` `Filter::{cmp, and, ...}`, `Cmp::EqU32`, `Agg::{Count, Rows, ...}`, `lower(&Query) -> Result<Program, LowerError>`; mask-risc `execute_into(program, &Planes{n_rows, masks, lanes}, &Foreign::NONE, &mut Scratch, Out)` | REUSE-BY-DEPENDENCY | Edge table with subject/predicate/object ordinal lanes plus `Filter::and[EqU32, EqU32]`: admitted, no gather, no foreign plane. |
| Multi-hop | `Filter::Semijoin` (quack `lib.rs:387`, builder `:419`) unclassified, ISSUES.md `ISS-ADMISSION-B-CALL-SITE-CLASSIFICATION-NOT-STARTED` (:74), Linear EXO-140; mask-risc `graph_hop.rs:55` `OutgoingHop` quarantined behind off-by-default feature `quarantined-gather-hop`, `ISS-GATHER-HOP-1616` (ISSUES.md:206) | MISSING / REFUSED | Two independent grounds. STATUS_BOARD D-RST-8: "multi-hop over a large domain | BLOCKED on admission A2". |
| Answer shape | dir-sim `RoleEvidence {Held{grant,path}, NotHeld, Undetermined{unread}, NotDeclared}` (`actors.rs:84-102`) | REUSE-SHAPE | Mirrored by the §3 answer enum. |
| Refusal shape | contract `epistemic_state5` `Epi5ReadError{UndeclaredClass(ClassId), GenerationMismatch, UnknownProvenance}` | REUSE-SHAPE | Typed refusal reasons, not strings. |
| Deterministic digest | contract `hash::fnv1a` | REUSE-BY-DEPENDENCY | Frame and catalogue digests. |
| Set semantics | contract `set_rules` `Truth{TwoValued, ThreeValued}`, `Population{Distinct, Bag}` | REUSE-SHAPE | Edge relation is `Distinct`; `NotHeld` is meaningful only under declared coverage. |
| Identity minting | contract `NodeGuid::mint_for(TailVariant, classid, ...)` (`canonical_node.rs:387`) | NOT used | Needs a `classid`; none exists for these entities, and the fixture does not invent one. |
| Project-work concepts | OGAR `ogar-vocab` (main 68cb29b): `PROJECT_WORK_ITEM 0x0102`, `PROJECT_STATUS 0x0105`, `PROJECT_REPOSITORY 0x010A`, `PROJECT_RELATION`, `PROJECT_CHANGESET`, via `RedminePort`/`OpenProjectPort` aliases (`ports.rs:225-258`) | CONVERGE | Recorded as convergence candidates, unbound. |
| GitHub / Linear port; PullRequest, SourceSymbol concepts | none in `ogar-vocab` | MISSING | An OGAR change per `HOTPLUG-MIGRATION-GUIDE`; out of scope here. |
| String-id graph | contract `literal_graph` (String ids) | NOT-REUSABLE | Identity must not be a string on the hot path. |
| DT64 binding | docs #1794-#1799, #1804; 0 code files | MISSING | Not consumed; W1 does not claim DT64 conformance. |
| DR64² relation tile | existing lab surface | optional | Not needed for W1; Quack lanes suffice. DR64² only after cost proof. |
| DR64³ | NFR-OP-2 OPEN | MISSING | Not used. |
| DurableRef / ThoughtCommitId | doc comments only (`mork-pathmap-probe/src/rmap.rs:70`, `:690`); EXO-148, D-RST-1 | MISSING | No durable-replay claims. |
| MCP transport | rmcp 1.x `StreamableHttpService`, `surrealdb/surrealdb/mcp/src/service.rs:941-943` | REUSE-BY-DEPENDENCY at W2 | Precedent only; W1 has no transport. |
| `adarail_mcp` as host | Python/FastAPI at `mcp.exo.red`; write tools `ingest`, `post`, `message`, `upload_image`; ~30 secret env names incl. `GITHUB_TOKEN` and a writable Upstash | NOT-REUSABLE | A read-only workbench must not share a process with write tools or write-scoped secrets. |

## 3. Data contract (conceptual; lab types, not ABI)

**GlobalId** = (source system, upstream key bytes, source revision for revision-scoped keys). Linear and PR keys are stable, but their content is snapshot-scoped. Source symbols are revision-scoped (repo, commit, path, item path), so the same symbol name at two revisions is two identities (F01). A cold intern table maps GlobalId to a frame-local ordinal; strings never enter the hot path.

**SourceSnapshot** = (system, upstream revision, object read, fetched_at). A Linear edge's snapshot is the issue it was read from at that issue's `updatedAt`; a source symbol is pinned at the commit it was checked at.

**Canonical records (W1).** The `Record`s in `src/fixture.rs`, each carrying its snapshot, upstream locator and grade, plus the declared coverage (`Corpus::mark_complete`). Everything a `Workbench` holds (interned ids, the four `u32` lanes, the frame) is derived from them and rebuilt on load. Storing the raw upstream payloads as fetched is a W2 sync-job deliverable, not part of W1.

**Source-to-witness boundary (what a W1 receipt is NOT).** A W1 snapshot is a *read receipt*: which object was read, at which upstream revision string (a Linear `updatedAt`, a commit SHA), on which date. `Corpus::mark_complete` records that *this reader* read an edge set in full. Neither is an owner-issued physical snapshot pin, a permission or ACL attestation, or a complete negative observation across sources, so `NotHeld` here means "empty in what this fixture read", nothing wider. The `FrameStore` frame refuses stale *lab coordinates* only; it is not a production source snapshot and is not proposed for DT64-Q0. Reusing any W1 assertion for DT64-Q0, Report destination binding (#1811) or EXO-46 admission needs its own attested source, permission and coverage witness; an equal ordinal or digest alone must refuse.

**Frame.** `Workbench` owns a `dr_math_binding_probe::FrameStore`. On each load it advances to `FrameRevisions { source, relation_map, coordinates }`, each an `fnv1a` digest (of the snapshots, the predicate catalogue and the interned identity list). A `LocalRef` carries the `Frame` it was issued under; frames compare by identity of issue, not by value, so a coordinate from another workbench is `ForeignFrame` even at equal numbers, and any reload makes earlier coordinates `SourceChanged` / `MapChanged` / `CoordinatesChanged` / `FrameResealed` (F07, F11).

**Predicate catalogue** (fixture-local ordinals; not DT64 slots, not classids):

| Predicate | Shape | Symmetric | Inverse (projection, never stored) | ClassId convergence candidate |
|---|---|---|---|---|
| `blocks` | issue → issue | no | `blocked_by` | `PROJECT_RELATION` |
| `related_to` | issue ↔ issue | yes | — | `PROJECT_RELATION` |
| `mentions_symbol` | issue → source line at a commit | no | `mentioned_by` | none (no SourceSymbol concept) |
| `tracked_by_debt_row` | issue → `ISSUES.md` row at a commit | no | `tracks` | none |
| `cites_pr` | issue → pull request | no | `cited_by` | none (no PullRequest concept) |
| `authored_in_session` | artifact → Claude Code session id | no | `authored` | none |

A `merged_as` (PR → commit) predicate would converge onto `PROJECT_CHANGESET`; it is not in W1 because no fixture record needs it.

**Evidence grades** (`Grade`): `ObservedSource`, `CompilerDerived`, `LogicallyDerived`, `Tested`, `IndependentlyVerified`, `Hypothesis` — the parent plan §2 list. W1 uses two: every record is `ObservedSource` except one inferred edge, which is `Hypothesis` and is listed by `list_unverified_claims`. Stale evidence is not a grade: it is a frame refusal.

**Answer** (never `bool`): `Witnessed { witnesses, complete }` | `NotHeld` (only when the edge set was declared complete) | `Undetermined { unread }` | `Refused(Reason)` | `Unsupported(op)`. `Reason` = `Frame(FrameRefusal)` | `MultiHopNotAdmitted` | `PredicateUnbound` | `Executor(..)`. This mirrors dir-sim's `RoleEvidence`.

## 4. Execution path

| Question shape | Path | Verdict |
|---|---|---|
| One-hop forward ("what does X block?") | edge table; `Filter::and[subject == X, predicate == blocks]` lowered by Quack, run by mask-risc `execute_into` | ADMITTED |
| One-hop inverse ("what blocks X?") | `Filter::and[object == X, predicate == blocks]`: `blockedBy` is a column read of `blocks`, never stored twice. Cross-checked against Linear's own `blockedBy` list as an oracle. | ADMITTED |
| Symmetric (`related_to`) | one stored direction, queried on both columns | ADMITTED |
| Multi-hop, transitive blocker chains | would need `Semijoin` (EXO-140, ISS-ADMISSION-B...:74) or `OutgoingHop` (ISS-GATHER-HOP-1616, ISSUES.md:206) | REFUSED: `Refused(MultiHopNotAdmitted)`, citing both |
| Provenance of an identity (every edge touching it) | `Filter::or[subject == X, object == X]` | ADMITTED |
| Unverified claims | `Filter::cmp(grade == Hypothesis)` over the grade lane | ADMITTED |
| Method inspection / binding for a predicate | no prepared methods exist; no predicate has a classid | `inspect_predicate` → `Unsupported`; `prepare_method` → `Refused(PredicateUnbound)` for any classid |
| Snapshot comparison | set difference of the canonical records | cold audit, not a query path |

The independent scalar baseline is a plain loop over the original edge records, **test-only**. It is never a fallback inside a served path.

## 5. Falsifiers — W1 results

Run: `cd crates/sag6-workbench-probe && cargo test` — **16 passed, 0 failed** (2026-10-27, lance-graph `aa569bc` + this crate). Every test names what makes it fail; the baseline oracle compares by `GlobalId` over the raw records and shares no code with the ordinal lanes or the Quack program.

| # | Falsifier | Parent gate | Test (`tests/falsifiers.rs`) |
|---|---|---|---|
| 1 | Same path at two revisions is two identities; each answers only for its own revision | F01 | `the_same_symbol_at_two_revisions_is_two_identities` |
| 2 | A coordinate from another workbench, same corpus and same ordinal, is `ForeignFrame`; the silence twin accepts it where issued | F07 | `a_coordinate_from_another_frame_is_refused` |
| 3 | An unread edge set answers `Undetermined`, an incomplete one `complete: false` | F06 | `missing_coverage_is_undetermined_not_false` |
| 3b | Silence twin: a fully read empty edge set answers `NotHeld` | F06 | `a_fully_read_empty_edge_set_is_not_held` |
| 3c | An identity read in full with no edge at all still resolves and answers `NotHeld` (codex review #1812) | F06 | `a_fully_read_identity_with_no_edges_is_not_held` |
| 4 | Reload refuses earlier coordinates: identical corpus `FrameResealed`, changed snapshot `SourceChanged` | F11 | `a_reload_invalidates_every_earlier_coordinate` |
| 5 | No predicate binds a prepared method, even with a real OGAR classid (`0x0102`) | F04 / F14 | `an_unbound_predicate_refuses_every_prepared_method` |
| 6 | Queries leave the canonical records unchanged (W1 has no write path; all query methods take `&self`) | F13 | `queries_do_not_change_the_canonical_records` |
| 7 | The inferred edge is listed as unverified and keeps grade `Hypothesis` in answers | F15 | `an_inferred_edge_is_reported_as_a_hypothesis` |
| 8 | Quack + mask-risc equals the scalar baseline for every identity × predicate × direction (≥ 20 non-empty comparisons, asserted) | F10 | `quack_matches_the_independent_baseline_everywhere` |
| 8b | `blocked_by` equals Linear's own `blockedBy` lists; only the 2 `blocks` rows are stored | DT64-3 inverse parity | `blocked_by_is_a_projection_that_matches_linear` |
| 9 | Derived state is disposable: a workbench rebuilt from the records alone answers identically. W1 has no Redis; this is the property a cache must preserve | new W-gate | `derived_state_is_disposable` |
| 10 | Importing the same records twice changes neither the corpus nor the snapshot diff | new W-gate | `a_replayed_import_is_idempotent` |
| — | Multi-hop is refused; the one-hop parts of `explain_blocker` still answer | F04 / D-RST-8 | `multi_hop_is_refused_not_emulated` |
| — | `compare_snapshots` reports exactly an added record | — | `compare_snapshots_reports_exactly_the_change` |
| — | Every witness carries a snapshot revision and an `https://` locator | F15 | `every_witness_carries_its_source` |

**Disable runs** (each patch asserted to apply exactly once, then restored with `git checkout`; tree clean after):

| Disable | Tests that went red |
|---|---|
| D1 coverage always complete | `missing_coverage_is_undetermined_not_false`, `multi_hop_is_refused_not_emulated` |
| D2 coverage never complete | `a_fully_read_empty_edge_set_is_not_held`, `blocked_by_is_a_projection_that_matches_linear` |
| D3 frame check removed | `a_coordinate_from_another_frame_is_refused`, `a_reload_invalidates_every_earlier_coordinate` |
| D4 symbol identity drops its revision | `the_same_symbol_at_two_revisions_is_two_identities` |
| D5 inverse reads the forward column | 6 tests incl. the baseline and the Linear projection check |
| D6 `query_chain` answers `NotHeld` | `multi_hop_is_refused_not_emulated` |
| D7 witness grade laundered to `ObservedSource` | `an_inferred_edge_is_reported_as_a_hypothesis` |
| D8 `prepare_method` returns `Ok` | `an_unbound_predicate_refuses_every_prepared_method` |

**Not claimed:** no timing (the fixture is 19 records; a timing would measure nothing); no claim beyond this bounded slice; no DT64 conformance.

## 6. Technical-debt and EXO-50 references

| Item | Status / relation (read 2026-10-27) |
|---|---|
| EXO-111 | Backlog; blockedBy EXO-112; related EXO-50 |
| EXO-112 | Backlog; blocks EXO-111; related EXO-50; label "CC: Operator Decision" |
| EXO-140 | Backlog; related EXO-46, EXO-114, EXO-135; "CC: Operator Decision"; names `Filter::Semijoin`; cites #1640, #1732 |
| EXO-148 | Backlog; blocks EXO-55; related EXO-45, EXO-50; cites #1730; DurableRef / ThoughtCommitId do not exist |
| EXO-50 | Done (2026-10-26); append-only bus, not an approval |

| ISSUES.md row | Line | Relevance |
|---|---|---|
| `ISS-DURABLEREF-THOUGHTCOMMITID` | :176 | OPEN, owned by D-RST-1; blocks any durable-replay claim |
| `ISS-ADMISSION-B-CALL-SITE-CLASSIFICATION-NOT-STARTED` | :74 | `Semijoin` unclassified; blocks multi-hop |
| `ISS-GATHER-HOP-1616` | :206 | `OutgoingHop` quarantined; blocks multi-hop |

STATUS_BOARD: **D-RST-1** (DurableRef, "LAB ONLY ... Queued", :281), **D-RST-6** (queued on A1), **D-RST-8** (multi-hop, BLOCKED on admission A2).

## 7. Open prerequisites (owner decisions)

| Owner | Question | Blocks |
|---|---|---|
| OGAR owner | Add a GitHub / Linear port and PullRequest / SourceSymbol concepts, or confirm convergence onto the `PROJECT_*` ids? | binding predicates to ClassIds (W2 method inspection) |
| Operator (EXO-140) | ADMISSION B classification of `Filter::Semijoin` | multi-hop answers (any stage) |
| D-RST-1 owner (EXO-148) | Home for DurableRef / ThoughtCommitId | durable replay claims |
| Operator | Domain: `mcp.exo.red` is occupied by `adarail-production`; proposed `workbench.mcp.exo.red` | W4 |
| Operator | Railway workspace and service ownership; a separate service, not co-hosted with `adarail_mcp` | W4 |
| Operator | Read-only GitHub and Linear tokens, held by a separate sync job, not by the query service | W2 sync, W4 |
| Operator | Is Redis/Upstash used at all? If yes: cache only, read-only key namespace, rebuildable | W2 |
| Operator | Resolve `railway.toml` (DOCKERFILE, hc 100 s) vs `railway.json` (NIXPACKS, hc 10 s) disagreement if either file is reused | W4 |

## 8. Proposed first implementation PR (separate from deployment)

**Title (PROPOSED):** `D-SAG-6-W0/W1: dogfood workbench amendment + read-only lab fixture`.
**Scope:** this document; `crates/sag6-workbench-probe` (own `[workspace]`, excluded from root; deps `lance-graph-quack`, `lance-graph-mask-risc`, `lance-graph-contract` for `hash::fnv1a`, `dr-math-binding-probe` for `FrameStore`); its tests; board rows (STATUS_BOARD D-ids W0-W4, an `entries/` finding, regenerated indexes in the same commit).
**Not included:** any MCP server or transport; `/board` view; Railway or domain change; GitHub/Linear sync or tokens; Redis; OGAR concept mint or ClassId; multi-hop; DR64³; DurableRef; writes to any external system; changes to `adarail_mcp`; CI. The root `Cargo.toml` gains one `exclude` entry for the probe and nothing else.

**This amendment is NOT a deployed service, a DT64 or DR64³ runtime, a second graph engine, an OGAR concept mint, a multi-hop reasoner, or an authorization to merge, deploy or write to GitHub, Linear or source.**
