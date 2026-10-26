# D-DT64-4: Edge×Edge dependency algebra, Thought-Git, time and hypothesis projection

**Date:** 2026-10-26. **Status:** RESEARCH PLAN / docs only. **Parents:** DT64-0–3 proposed, D-RFT/EXO-126, D-DRMC/EXO-127, D-HL/EXO-78, D-TGB/EXO-98 and EXO-148, #1792 owner/snapshot findings.
**Do not:** create write path, replay persistence, a knowledge-confidence engine or learning without separate authority.

## Semantic levels, not one giant cube

L0: observed source facts at versioned and authorized coordinates.
L1: directed relationship/inverse queries, same facts, local-view focus.
L2: **edge×edge** relations: dependency, support, contradiction, entailment candidate, and policy-source grant. Input is **edge identity with provenance**, not a naked pair of local ordinals.
L3: **Thought-Git**: `(source frame, hypothesis slot, logical version, time)` indexes provisional derivations; slot 0 represents givens and 1..63 are a *capacity ceiling*, not guaranteed occupied hypotheses (EXO-78 / D-HL ruling).
L4: time-series-derived features and numerical ONNX/TFLite leaf computations, only after DT64-5 admits methods. Prediction/recommendation does not upgrade L0 observations by itself.

Never equate `DR64³` coordinate axis with `w_slot`, temporal dimension or ONNX feature dimension. These are separately typed overlays/links; representation choice follows demand and witnesses. `ThoughtCommitId` and `DurableRef` do not exist as production code (EXO-148); use lab-local opaque handles without durable-contract claims.

## 1. Edge identities and dependencies

Proposal: an observed edge has `(source, dataset/table, version, stable source edge key or canonical typed triple + occurrence ID, predicate catalogue identity, domain/range endpoints, valid interval, observed_at, policy scope, evidence)`. For derived edge D, record the set of supporting *input edge identities* and the law/recipe identity, not just a Boolean result. A canonical Boolean derived answer is true with ≥1 valid support path. Path count and independent evidence provenance require separate Bag/provenance fold.

Example IAM diamond:
```text
e1: user U direct_member group G1
e2: G1 security_member nested G0
e3: G0 grants ROLE_R
e4: U direct_member G2
e5: G2 grants ROLE_R
d1: U has_effective ROLE_R, supports {e1,e2,e3} AND {e4,e5}
```
When e1 or e2 is retracted, d1 remains with support path 2; when e4 also disappears, d1 is not supported. UNKNOWN if a group-role policy was unread, not FALSE. Support witnesses are not indiscriminate transitive closure over mail-enabled lists. Use `dir-sim/actors.rs::RoleEvidence` and D-RFT derived relations as oracle.

## 2. Temporal coordinates are independent

| Coordinate | Meaning | Source | Gate |
|---|---|---|---|
| `valid_from,to` | when the real business fact applies | Odoo `hr.version`, source DTO | not a Lance version |
| `observed_at` | when source extractor saw it | adapter observation | not claim validity |
| `source_revision` | source owner snapshot | source owner | cannot compare across unrelated sources |
| `dataset_version` | physical immutable Lance checkout | `checkout_version` | not merely `QueryReference::at` |
| `query_horizon` | which knowledge may be admitted | `QueryReference::at` | source checkout independent (#1792) |
| `thought_slot` | provisional hypothesis context | D-HL `w_slot`, 0 givens | not proof or global ID |
| `thought_commit` | future durable state identity | EXO-148 OPEN | must not invent as existing |

A fact may be true for a past interval but first observed today. One source revising a record does not rewrite another source's version line. A temporal projection needs explicit as-of policy and independently grounded source snapshots.

## 3. Counterfactuals, replay, learning

Scenario `H1`: user is hypothetically removed from G1. Apply a provisional delta only in hypothesis 1, compute effective role R. Its new result is NOT an observed production permission change; hypothesis must never write AD/Entra/Odoo/SAP. Scenario `H2`: revoke another group edge; calculate distinct support diff. Compare combined staged alternatives with original source snapshot and rebase conflicts.

Define as a **lab** `ThoughtDelta = {base_snapshot_ref, hypothesis_slot, affected_edge_keys, diff, rule_digest, output_digest, provenance}`. Do not assert `ThoughtCommitId`/DurableRef shipped. Use existing Thought-Git branch/diff/status/squash/discard/replay plan vocabulary and source-pinned receipts. No new persisted WAL or runtime actor.

Learning is another separate lane: metrics over predicted versus subsequently observed outcomes may produce *calibration or suggested policy*, never retroactively mutate the original observation. A derived Boolean `edge×edge` relation may be exact; a confidence/weight or neural score must occupy a separate typed numerical carrier with model version and error envelope, not overwrite truth.

## 4. Timeseries + ONNX

Typed input `(entity, property, valid/observation times, value, source version)` yields explicitly ordered scalar/float/i8 time series. Window semantics, missing samples, irregular timestamps, monotonicity, feature normalization, dtype/rounding/saturation, model digest and permissions stay explicit. Native `ndarray` arithmetic and D-NAM/EXO-152 method binding may execute a deterministic model; ONNX is *external notation*, not a new runtime. A model output is `ComputedResult` with receipt, not `ObservedFact`.

Example: forecast licence exhaustion from licensed assignment event series; it is a forecast, not an assignment. Example: suggest access anomaly after group change; keep posterior score separate from effective role derived by exact SID/policy rule.

## 5. Falsifier ladder

- T0: history dual clocks: same valid_at and different observed_at must be distinguishable.
- T1: two independent support paths, retract one → still TRUE; retract both and complete policy → FALSE.
- T2: missing grant policy → UNKNOWN, regardless of absence in Boolean tile.
- T3: delta in hypothesis slot 1 never changes slot 0 givens; source snapshot immutable.
- T4: same `thought_slot` reused at later base source snapshot refuses stale replay unless explicit rebase.
- T5: same local ordinal in two source frames refuses edge identity alias.
- T6: `QueryReference::at` horizon alone does not attest checkout; different dataset same version refuses.
- T7: deterministic re-execution over same evidence+method yields identical result digest; no clock/randomness hidden.
- T8: test a dependency cycle and two contradictory assertions. No naive automatic "repair", retain conflict.
- T9: role computation vs neural predictor: ML score cannot cause authority bit to become TRUE.
- T10: output with missing time-series data is refused or explicit modeled missingness, never silently zero-filled.
- T11: provenance storage cost, baseline direct re-evaluation vs incremental dependency tracking, including failed preparation economics.

Negative controls must independently fail; original source DTOs never changed. Real thought-git durable/replay gates depend on EXO-148 owner's decision, EXO-55/58 writer receipts, and explicit authorization. Do not patch those adjacent components.

## 6. Stage decision records and follow-ups

DT64-4A: Source-linked edge identity/provenance oracle, no new storage.
DT64-4B: Exact Boolean and counted-support derivation oracle, dependency retraction.
DT64-4C: 64-slot thought/hypothesis replay probe (lab only), independent base version.
DT64-4D: Temporal join and counterfactual as-of querying.
DT64-4E: Model-output receipt and prediction-versus-fact separation after DT64-5.
A/B may start as independent lab plans; C/D require source/focus authority; E requires numerical method admission. **No durable implementation until EXO-148 is resolved and operator authorizes**.

**Non-goals:** no emotion/awareness claims from relationship inversion, no "hologram" as automatic learning, no all-to-all edge matrix from conjecture, no new persisted graph engine or hidden cross-system policy writes.
