# SAG capstone alignment — v1

**Status:** CURRENT-CONTRACT for the documents it aligns; nothing here is
implemented by this PR. Lands as #1523; supersedes the separate review threads of #1510,
#1511, #1512, #1513, #1515 and #1517, which are merged here unchanged except
for the edits listed in §6.

**Read this before any of the plans it aligns.** They were written in
parallel against the #1509 base. They agree on most of their architecture,
but they used six names for one concept, five status vocabularies, and
several claims that later PRs made stale.

## 1. The aligned documents

| D-id series | PR | Plan(s) | Suffix means |
|---|---|---|---|
| D-SAG-2-0..5 | #1510 | `software-awareness-graph-v2-ruff-ona-closed-loop.md` + addendum + provenance-parity knowledge | waves |
| D-SAG-3-0..15, D-SAG-3.1 | #1511 | `sag-wisdom-frontier-ona-hyperon-v1.md`, `sag-wisdom-frontier-hard-problems-v1.md` | falsifiers |
| D-SAG-4A..G | #1513 | `software-awareness-agent-context-ona-v1.md`, `…-deferred-contracts-v1.md` | deferred contracts |
| D-SAG-5-0..10 | #1515 | `sag-reflexive-cognition-from-1509-v1.md` | probes |
| D-IAM-SAG-0-*, 1-* | #1512 | `iam-substrate-sag-dir-sim-v1.md`, `iam-substrate-frontier-gaps-v1.md` | deliverables |
| D-SAG-CF-1 | #1517 | `inverted-temporal-ablation-reasoner-oracles-v1.md` + `inverted_temporal_ablation_oracle.py` | one RFC |

The suffix meaning differs per series, and that is acceptable because the
series are disjoint. **New SAG work takes the next free series number and
uses the suffix for deliverables.** No collisions exist between the six.

## 2. `ProposedMethod` is the canonical name

**DECISION.** One name, `ProposedMethod`, replaces `CandidateProposal`,
`Proposal` (#1510), `ProposedChange` (#1512), the returned `Candidate[]`
(#1513), "H1 proposal" / "optimization proposal" (#1515) and "any
optimization is a proposal" (#1511).

> A `ProposedMethod` is an **inert, evidence-carrying method candidate**. It
> names what would change and what must be shown first. It can never execute,
> and no stage downstream of it may treat its existence as authorization.

It has exactly two shapes. ⊘ Both anchors named below were removed from
main by the method-layer reset (#1524, `D-CMR-RESET`; the code is kept on
`backup/session-011BMJ-2026-10-12`). The name and the two shapes stand; the
code anchor is re-decided when the class-method rebuild (D-CMR) lands:

| shape | former anchor (removed by #1524) | what it carries | what it is not |
|---|---|---|---|
| **rewrite** — a pure, structural optimization | `awareness::Candidate { kind, methods, obligations }` ("never a rewrite") | the candidate, its open `Obligation`s, evidence tier and provenance, and a read-only verification query | not applied: it still needs a semantic-equivalence oracle (NULL / bag / set), a cost measurement and an operator decision |
| **action** — an external effect | `EffectClass::Propose` on `MethodBody::Action` ("builds an inert proposal"); every call refuses with `NotPure` | the effect, its preconditions (source ETag/version, policy revision, target identity) and its simulated impact | not dispatched: it needs fresh OGAR authorization, then provider receipt and re-observation |

### 2.1 Anchors after the reset (2026-10-13)

**DECISION.** The action shape anchors on the action gate that already exists:
a proposal is an `ActionInvocation` in state `Pending`, and the gate that may
advance it is `ActionInvocation::commit_via` (the `ClassRbac` grant registry,
no admin bypass). This is a requirement, not yet a property of the code: the
older `ActionInvocation::commit` (`action.rs:265`) makes the same transitions
with an admin bypass, is public, and `lance-graph-ogar/src/actions.rs`
directs consumers to it (OPEN item 7).
**SCOPE.** The action shape only. The rewrite shape anchors on the rebuilt
D-CMR fold binding (`classid × WideFieldMask → Compiled` program) and waits
for it.
A rewrite-shape `ProposedMethod` is then the output of concept-chain
normalization: an equivalent, cheaper form of a chain, proven against the
unrewritten form and a row oracle (rules in
`entries/2026-10-12-session-reset-inventory.md`, "Concept chains").
**BASIS.** The removed `glove-parity::method::EffectClass` documented that only
OGAR's action gate runs `ExternalMutate`, so `Propose` sits in front of that
gate. Read in code:

| piece | where | what it already gives a proposal |
|---|---|---|
| the gate | `lance-graph-contract/src/action.rs` `ActionInvocation::commit_via` (`:327`; `commit` at `:265` is the older variant with an admin bypass) | sticky terminal states, then def match → RBAC grant (`ClassRbac`, `Operation::Act`) → `StateGuard` → MUL `GateDecision`; `Pending → Committed / Failed / Cancelled`, and `Hold` keeps it `Pending` |
| the inert record | `ActionInvocation::pending(..)` (`:218`) | target as a full `NodeGuid`, `cycle`, `idempotency_key`, `trace_id`; `emitted_at_millis` stays `None` until commit |
| OGAR's IR form | `ogar-vocab` `ActionDef` / `ActionInvocation` / `ActionState` (same four states, String identities, plus `parent_invocation` and `failure_reason`) | the producer-side record |
| building a `Pending` invocation from a request | `ogar-from-schema/src/action_ws.rs` `submit_to_invocation` | HIRO `submitAction` → `Pending`; the doc comment says it then passes `commit_via` before `Committed` |
| running it | `ogar-action-handler` `CapabilityExecutor` impls; the doc comments say `commit_via` runs upstream of them, in rs-graph-llm's `graph-flow-action` (not read here) | execution plus `Receipt` and `sendActionResult` |

Not related: `ogar-proposal` (schema `MappingProposal` drafts for the
ontology registry) and the Cohen's d `EffectClass` in `lance-graph-cognitive`.

**What the gate does not carry yet** (OPEN, against the life cycle below):
1. *Source version / ETag precondition.* `StateGuard` is `field == value` on
   the current instance; no version of the source the proposal was made
   against.
2. *Policy revision.* RBAC is evaluated at commit against the current grants;
   the proposal records no revision.
3. *Simulated impact.* The MUL `GateDecision` is an input to `commit_via`,
   not stored on the invocation.
4. *Separate approval and authorization.* The life cycle keeps "approved"
   and "authorized" as two facts; `commit_via` decides RBAC and impact in one
   call.
5. *`UnknownOutcome`.* `ActionState::Failed` covers both an unauthorized actor
   and a runtime failure (`action.rs:50`), and there is no state for a
   timed-out external call, which the life cycle requires never to read as
   "not applied".
6. *Re-observation.* Nothing records the target's state after the effect.
7. *A second gate.* `commit` (admin bypass) can advance the same `Pending`
   invocation without the `ClassRbac` check. Either retire, privatize or
   redirect it, or document both gates and their different authorization.

None of the seven is built by this note; each lands, if at all, in the gate's
own types.
**REVISIT WHEN** the D-CMR rebuild lands (rewrite anchor), or when any of the
seven gaps is closed.

The life cycle is shared, and each arrow is a separate fact with its own
receipt:

```text
ProposedMethod -> verified (oracle / simulation) -> approved (operator / policy)
               -> authorized (fresh OGAR) -> applied -> re-observed -> Verified | Unverified | UnknownOutcome
```

`proposed != verified != approved != authorized != applied != re-observed`.
A failed or timed-out external call is `UnknownOutcome`, never "not applied".

**Not a `ProposedMethod`:** choosing what to look at next. #1511's
`Proposal {…}` record (scores, rubicon_state), #1510's `CandidateProbe`,
#1515's attention proposal and #1517's candidate cut X all select work and
change nothing. They are renamed **`SchedulerDecision`** where a type was
spelled (#1511); prose is left as is. A scheduler decision is `Pure` or
`ReadVersion`.

**Name check.** `lance_graph_ontology::MappingProposal` is unrelated (an
ontology mapping); it keeps its name.

**When code lands,** `ProposedMethod` is one type, in the crate the D-CMR
rebuild puts class methods in (`lance-graph-glove-parity::awareness` no
longer exists after #1524). No other crate defines its own proposal type.

## 3. One status vocabulary: unknown is not false

The six PRs define five enums for the same idea. The canonical carrier is the
four-state status already shipped by dir-sim (`Held / NotHeld / Undetermined
/ NotDeclared`), extended only with the qualifiers the PRs show are needed:

| canonical | maps from |
|---|---|
| **Witnessed** (positively observed) | `Confirmed`, `Known`, `Held`, `Witnessed`, `Complete` |
| **Refuted** (positively contradicted) | `Refuted`, `NotHeld`, `AbsentUnderClosedWorld` *(only with a closure certificate)* |
| **Undetermined** (not read, not readable, or not finished) | `Pending`, `NotObservable`, `Incomplete`, `Unread`, `Missing`, `Undetermined`, `Partial`, `NotIndexed`, `InsufficientEvidence` |
| **Conflicted** | `Conflict`, `Conflicted` |

dir-sim's `NotDeclared` (the directory declares nothing for the subject)
is not a truth value: it maps to **Undetermined** with reason
`NotDeclared`, so a caller can tell "nobody said" from "not read yet"
without a fifth state.

`Stale` is not a fifth state: it is any of the four read against an expired
revision, carried as a separate revision field. `Unsupported` /
`UnsupportedMultiHop` are typed refusals, not statuses.

**Consequence fixed in #1511:** `missing = required & ~(established |
refuted | unread)`. An unread prerequisite is `Undetermined`, not missing
debt. The plan and the prompt now agree with `hard-problems` H1.

Building this enum is Step 4 of the SAG queue (one receipt edge family + one
four-state status); it is not built here.

## 4. Where the PRs agree (no action)

- Belief, intent and permission are separate. A commitment grants nothing.
- ONA, OpenNARS, PLN and Hyperon are oracles and references, never engines.
- NARS f/c is not a probability; EIG needs a calibrated distribution.
- Copies and pipeline layers add no evidence; fission needs contributor IDs.
- The H1 fusion numbers agree everywhere (1.59–1.78×; 603.5 vs 379.1 µs).

## 5. Contradictions resolved

| topic | PRs | resolution |
|---|---|---|
| Ruff's role | #1510 "independent source oracle"; #1515 "Python lint only, not an equivalence oracle" | Both hold for different questions: Ruff's polyglot SPO harvest is a **source-structure** oracle (what calls/reads what); it is **never** a semantic-equivalence oracle for a rewrite `ProposedMethod`. |
| ONA pin | #1510 "not pinned"; #1513/#1515 `dc4efd0`; #1511 v0.9.3 | Pin **`dc4efd0`**. v0.9.3 is the release that contains it for reading only. |
| Rubicon states | three chains (#1511 twice, #1510) | Use §2's life cycle for proposals; `hard-problems` H5's chain for scheduler commitment. Commitment never implies authorization in either. |
| Execution class names | #1512 `PureRead / VersionPinnedRead / Propose / ExternalMutate` "RFC name only" | They exist: `EffectClass::{Pure, ReadVersion, Propose, ExternalMutate}`. #1512 now uses them. |
| Receipt signing | #1515 "signed witness"; #1511 "not cryptographically signed" | Receipts are content-addressed and auditable, not signed. "Signed" in #1515 means "attributed". |

## 6. Stale claims

These are true at the #1509 base and false now. The plans keep their text
(history); this list is the correction.

1. **"Second hop refused" / "multi-hop blocked" / `NoRelationCarrier` /
   `SecondHop` — NOT stale; the refusal stands.** An earlier draft of this
   list said #1522 removed it. #1522 (`MaskOp::GatherChain`) and its ndarray
   primitive were withdrawn unmerged (closed 2026-10-12, kept on
   `backup/session-011BMJ-2026-10-12-chain2`; `D-CMR-RESET`): a hop is a
   coordinate range, and chained hops are the intersection of ranges, not a
   chain of key reads. `gremlin::compose` still returns
   `GremlinRefusal::SecondHop`, so #1517 F13's "invalid two-hop query" probe
   remains valid as written.
2. **SAG defects A and B remain UNVERIFIED** (#1510 plan is right). The
   fixes were written in #1519, which was closed unmerged with the method
   layer reset (kept on `backup/session-011BMJ-2026-10-12-sagfix`); the
   `awareness.rs` itself was removed by #1524, so both failure modes must be
   re-tested on whatever replaces it.
3. **OGAR #357 status.** Still unverified upstream; the plans' "RFC" is the
   safe reading.

## 7. Open points (not resolved here)

- #1517's oracle: `test_f0_pristine_replay_is_exact` compares a pure
  function with itself, `test_f7_contingency…` repeats F3, and F8 has no
  test. They are **claimed, unverified** until rewritten.
- No STATUS_BOARD or INTEGRATION_PLANS rows exist for D-SAG-2, D-SAG-3,
  D-SAG-4, D-IAM-SAG or D-SAG-CF; they are research series with no
  deliverable in flight. Rows are added when a series starts building.
- `ProposedMethod` and the four-state status are named here, not built.
