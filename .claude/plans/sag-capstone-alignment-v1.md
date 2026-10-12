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
