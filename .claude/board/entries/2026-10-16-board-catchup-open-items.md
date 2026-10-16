# 2026-10-16 — Board catch-up: open items named by merged PRs #1388–#1588

**Status:** OPEN (ledger of pointers; nothing decided here).

The catch-up that added 115 missing `PR_ARC_INVENTORY.md` entries also read the
"open", "follow-up" and "deferred" lines of those PR descriptions. This entry
lists the ones still open, with where each is already recorded. It adds no new
claim: every row points at the PR body and at the doc that owns the item.

Items the bodies called open that have since closed are listed last, so they
are not re-filed.

## Integration-plan debt (a plan names the step; the step has not run)

| item | source PR | owner doc |
|---|---|---|
| D-RPF-4 population revision blocked on four decisions (normative vs pairwise chain, the c = 255 defect, admission as mask, witness scope) and a T1 gap (LUT-weighted strided sum) | #1411, #1420 | `plans/2026-10-08-resident-projection-fold-mask-v1.md` |
| D-RPF-1 needs the minimised cover size per recipe measured on the real law tables before it ships | #1420 | same plan |
| D-SQLP: which SQL text parser (DataFusion is in its grace period); the SAP cost comparison is unmeasured until W3/W4 | #1476 | `plans/sql-polyfill-r2il-v1.md` |
| An executable `PROC_DEF`/`PROC_CALL` contract is missing on the OGAR side; W2 procedures are inlined Rust functions until it exists | #1484 | same plan, §4 |
| A sorted-compare lowering needs an order witness, which exists only for facet lanes | #1484 | entry `2026-10-11-teleport-probe-source-restriction.md` |
| D-CMR seven OPEN items for `ProposedMethod`'s action anchor: source version, policy revision, simulated impact, approval vs authorization, `UnknownOutcome`, re-observation, the second gate (`commit`) | #1532 | `plans/d-cmr-concept-chain-normalization-v1.md` |
| Supervisor: normal `DispatchToG` forwarding not implemented for non-`Health` envelopes; declared mailbox bound unenforced; realtime drain unbounded | #1543 (guardrails) | `docs/architecture/OUTER_MEMBRANE_GRAPH_LAB_VERTICAL_SLICE.md` |
| D-SMD-0 source audit (12×12 palette / VerbRoleTable, HHTL, Moore) against HEAD before any code | #1545 | `plans/semantic-morton-qualia-do-falsifier-plan.md` |
| `performed_by`, `duration`, `tenant` stay `Hypothesized` until OGAR mints their targets (9 of 12 `BillableWorkEntry` edge targets unminted) | #1459 | `plans/cross-glove-business-parity-v1.md` §C.9.2 |
| `changed_coordinates` representation of the sparse-append pattern | #1466 | `knowledge/split-tunnel-sparse-append-resident-fold.md` |

## Decisions owed to the operator (named in the PR, not answered on the board)

| item | source PR | owner doc |
|---|---|---|
| Should `WoaPort` resolve `ogit.WorkOrder:TimeSheet` to `0x0103`, with "Stundenzettel" only as its label? (also D-HPR O2) | #1462, #1464 | D-XGP-8 row; `plans/v3-mandatory-hotplug-reading-v1.md` |
| D-HPR O1: which ClassView a new canon mint uses instead of `0x1000` | #1464 | same plan |
| D-HPR O3: `0x0101`, `0x0202`, `0x0A01` are both codebook concepts and code-minted appid heads, so `concept_override` gives a plugging consumer the head's storage reading | #1471 | same plan; entry `2026-10-11-synergy-ledger-classid-hotplug.md` |
| X13, X14, Y7 of the instance-authorization map | #1468 | entry `2026-10-11-coresearch-instance-authz.md` |
| K1, O25 policy, O7, O23, O26 (and O4, O13, O14, O15 from #1492) | #1492, #1554 | `.claude/research/2026-10-recovery/CONSOLIDATED-LEDGER.md` |
| The five canon-high residue sites | #1430 | `TECH_DEBT.md` `TD-CANON-HIGH-AUDIT-RESIDUE-1` |

## Innovation debt (research proposed, probe not built or not run)

| item | source PR | owner doc |
|---|---|---|
| NARS-Reasoning benchmark falsifier probe (arXiv 2604.18873) | #1492 | `.claude/research/2026-10-recovery/PAPER-2604.18873-nars-reasoning-benchmark.md` |
| N1–N14, sharpest N1: `syllogize` writes bits 50..52 as `0b111` and zeroes witness and epistemic bits, so a derived edge inherits none of its premises' state | #1554 | `CONSOLIDATED-LEDGER.md` |
| D-SMD-1..6 (scalar Morton vectors, held-out grammar regions, qualia ablations) | #1545 | D-SMD plan |
| D-FLOW-COG F0–F6 (flow/cog/Rubicon convergence); draft for architectural review | #1543 | `docs/architecture/FLOW_COG_RUBICON_ONA_CONVERGENCE.md` |
| The 14 plans of 2026-10-12..16 recorded in `INTEGRATION_PLANS.md` (2026-10-16 catch-up entry) are proposals or pre-registrations; none is authorized | many | each plan |

## Called open in a body, closed since (do not re-file)

- Group secondary SMTP address delivered nowhere (#1480): closed by #1483.
- `from_graph` Graph groups needed OGAR #353 (#1473): OGAR #353 merged.
- OGAR authority did not set the class default field mask (#1556): closed by OGAR #358 (`GrantSource::default_field_mask_of`), per `LATEST_STATE.md`.
- Board gaps named in #1554 (no arc entries for #1494–#1504; no STATUS rows for D-RESIDUE-FP-0 / D-MATURATION3-0): both present now.
- Supervisor restart backoff (#1539): resolved by #1540.
- #1427 "compilation not yet observed": the STATUS row now records a CI line.
