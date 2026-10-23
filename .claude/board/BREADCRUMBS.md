# Breadcrumbs — the trail below IDEAS (append-only)

> **Append-only.** Prepend new crumbs. Never delete or rewrite a crumb; regrade
> its `**Cluster:**` line only. Added 2026-10-23 (operator: "breadcrumbs of
> something bigger").

## What a crumb is

An observation that seems to rhyme with another one, and claims nothing more.
It fails every other gate on its own: it is not a lesson, not an idea, not
torque, not an insight. Before this file existed, it died at closeout, so the
bigger pattern it belonged to could never form, because only the trail across
sessions shows it.

A crumb costs one entry and is never gated. It needs:

- `**Observation:**` one or two sentences, stated as observed, not explained;
- `**Receipt:**` a PR, board entry, probe or file:line;
- `**Rhymes-with:**` other crumb ids (may be empty);
- `**Cluster:**` `UNCLUSTERED` | `RHYME (<trail id>)` | `NAMED → IDEAS <date title>` | `KILLED (<counterexample receipt>)`.

## The clustering rule

When three or more crumbs from **independent sources** link to each other,
someone must try to name the shared mechanism. There are two outcomes:

- **A mechanism is named.** Prepend an `IDEAS.md` Open row that cites the
  crumbs, and set each crumb's `Cluster:` to `NAMED`.
- **No mechanism is found.** The trail gets the verdict `RHYME`, the
  `cross-domain-synthesizer` test (does the trail share a mechanism, or only
  a surface?). The crumbs stay. A RHYME is not a failure, just a trail
  without a name yet.

A crumb that a counterexample refutes is regraded `KILLED`, with the
counterexample as its receipt. It is never deleted.

Coordination happens in Linear (`CC: Thread` label `Breadcrumb`, milestone
*Co-research: Breadcrumbs → Torque*). This file is the record.

---

## Crumbs

### BC-2026-10-23-14 — where a state sits inside its cell decides where it goes next
**Observation:** In EXO-129, about half of the members of a palette cell go to
a different next cell than the cell's modal result (purity ≈ 0.47). A table
learned from reached states does worse on population starts than on the
states it saw. In the XOR-group world (purity 1.0) the exact law composes
even under noisy snapping.
**Receipt:** lance-graph #1731,
`entries/2026-10-23-exo129-learned-lut-preregistration.md`.
**Rhymes-with:** BC-2026-10-23-1, BC-2026-10-23-4
**Cluster:** NAMED → IDEAS 2026-10-23 "Lumpability as the closure check"

### BC-2026-10-23-13 — a replay receipt is sealed before landing, and its witness is issued only by the write path
**Observation:** D-TGB-4's receipt is built and sealed at stage time; a
landing is accepted only with a `CommitWitness` that `VersionStore::commit_cycle`
alone produces, and its absence refuses with `Refusal::NoCommitWitness`. The
module states that the receipt proves replay equals the original run, not that
the derivation is admissible; that promotion is D-TGB-7's. Same three parts as
trail T-ADMISSION: built before (sealed receipt), witness with an issuer
(the cycle commit), refused promotion (replay → proof).
**Receipt:** `crates/thought-git-probe/src/receipt.rs` on main `8f610a85`
(module doc, `pub struct CommitWitness`, `Refusal::NoCommitWitness`); lance-graph #1674, #1684; GPT /05 three-artifact
note (EXO-50, `session_019fbZ92dFKQ5BJHrA3YcH9A/007`).
**Rhymes-with:** BC-2026-10-23-8, BC-2026-10-23-9 (draft #1730)
**Cluster:** RHYME (trail T-ADMISSION, Linear EXO-135)
**Correction (2026-10-23, operator):** the witness is not issued by the write
path. It is the lane dimension: identical NodeGuids across lanes give a
vertical read of which thoughts wrote a node. The probe's `CommitWitness`
restates that read as a lab stand-in. The issuer question raised on EXO-135
(component or version) is answered: neither, it is the lane.

### Trail T-ADMISSION — what may be built before execution, under what witness, and what may never be promoted from it?
Six crumbs (BC-2026-10-23-7 … -12) from six sources: a decision record, two
probe reviews, a co-research map and two plan rows.
**Shared shape (candidate, not adopted):** each crumb has (1) something built
before execution (coordinates, a projection, a slot table, a recipe), (2) a
witness naming who issued it and against which version, and (3) a promotion
that is refused (evidence → premise, census → qualification, replay → proof,
hidden gather → co-addressed read).
**Kill:** two crumbs whose witnesses need incompatible semantics; that makes
the trail a RHYME only.
**Rhymes-with trail:** T-CLOSURE (its refusals `BagNeedsMultiplicity` and
`Complement` under K3 may be the same refused promotion seen from the algebra side).
**Verdict:** RHYME, 2026-10-23; naming attempt open. Linear EXO-135.

### BC-2026-10-23-12 — D-CMR-5's preimage is a projection built in preparation and witnessed by version
**Observation:** the lawful design for `f⁻¹(S)` is a co-addressed tile fold
over a version-witnessed projection of `f`, built in preparation by a
sequential same-table scan.
**Receipt:** STATUS_BOARD row D-CMR-5; `entries/2026-10-21-d-cmr-5-rescope.md`.
**Rhymes-with:** BC-2026-10-23-7, BC-2026-10-23-11
**Cluster:** RHYME (trail T-ADMISSION, Linear EXO-135)

### BC-2026-10-23-11 — D-RU-6's slot assignment must be argued as an attested frozen-codebook lookup
**Observation:** the recursion driver stays Queued/Lab until its slot
assignment is shown to be an attested lookup in a frozen codebook under
ADMISSION B.
**Receipt:** STATUS_BOARD row D-RU-6.
**Rhymes-with:** BC-2026-10-23-7, BC-2026-10-23-12
**Cluster:** RHYME (trail T-ADMISSION, Linear EXO-135)

### BC-2026-10-23-10 — Palette256 puts its property in evidence accounting plus a decision gate
**Observation:** the co-research map found the property in evidence
accounting and a decision gate, not in the encoding; where the evidence stamp
lives is open (G5).
**Receipt:** `entries/2026-10-22-coresearch-pal256-evidence-probes.md`.
**Rhymes-with:** BC-2026-10-23-8
**Cluster:** RHYME (trail T-ADMISSION, Linear EXO-135)

### BC-2026-10-23-9 — a recipe certificate only a certified compiler may issue; none exists for DR64
**Observation:** after X15, recipe admission in `dr-math-binding-probe` ends in
`NoCertifiedCompiler`; a replay capsule is evidence, not the issuing authority.
**Receipt:** `entries/2026-10-22-coresearch-drmc-stack.md` § Applied, lance-graph #1703.
**Rhymes-with:** BC-2026-10-23-8
**Cluster:** RHYME (trail T-ADMISSION, Linear EXO-135)

### BC-2026-10-23-8 — evidence nominates, a separate validator admits
**Observation:** D-DRMC X7: no conversion from a `PopulationReport` to a law
premise may exist; a population-derived premise can at most be a named,
undischarged hypothesis.
**Receipt:** `entries/2026-10-22-coresearch-drmc-stack.md` (X7, probe not run).
**Rhymes-with:** BC-2026-10-23-9, BC-2026-10-23-10
**Cluster:** RHYME (trail T-ADMISSION, Linear EXO-135)

### BC-2026-10-23-7 — preparation may establish coordinates; it may not conceal a relational gather
**Observation:** ADMISSION B separates preparation (coordinates) from
execution (co-addressed sets) and forbids an unaccounted gather in either.
**Receipt:** `entries/2026-10-20-six-decisions.md`, lance-graph #1640.
**Rhymes-with:** BC-2026-10-23-11, BC-2026-10-23-12
**Cluster:** RHYME (trail T-ADMISSION, Linear EXO-135)

### BC-2026-10-23-6 — Boolean composition is admitted only within compatible domains
**Observation:** D-DRMC-5 admits `BooleanCompose` only over two-valued
Distinct operands with compatible ClassView domains and refuses commutation.
Closure here is "closed within a sort"; this counts as a T-CLOSURE crumb only
if sort typing is accepted as closure. Same probe, near-miss counterexample:
14 association orders give the same relation (closed and correct) at
6 414–15 310 row-ORs, so closure decides correctness, not cost.
**Receipt:** lance-graph #1707, `entries/2026-10-22-drmc-5-assoc-probe.md`.
**Rhymes-with:** BC-2026-10-23-4, BC-2026-10-23-5
**Cluster:** NAMED → IDEAS 2026-10-23 "Lumpability as the closure check" (trail T-CLOSURE, Linear EXO-132)

### BC-2026-10-23-5 — two-valued rewrites break on a carrier that holds NULL
**Observation:** `set_algebra::normalize` (De Morgan, lane covers) is sound on
{T, F} only. On nullable lanes the carrier is {T, F, U} and the rewrites turn
UNKNOWN into TRUE. Extending the carrier to Kleene K3 closed it;
`Complement` and `RelationalDifference` still refuse under `ThreeValued`.
**Receipt:** lance-graph #1668, `entries/2026-10-21-d-rft-0-kleene-normalization.md`;
`cargo +1.99.0 run --release -p lance-graph-contract --example k3_pg_parity`:
two-valued `normalize` wrong on 474 of 2000 expressions against PostgreSQL
16.14, `normalize_kleene` equal on 2000/2000. Same shape in #1686 / #1687
(`bool_or`/`bool_and` read NULL as FALSE).
**Rhymes-with:** BC-2026-10-23-1, BC-2026-10-23-4
**Cluster:** NAMED → IDEAS 2026-10-23 "Lumpability as the closure check" (trail T-CLOSURE, Linear EXO-132)

### BC-2026-10-23-4 — the same join is right under Distinct and wrong under Bag
**Observation:** D-RFT-1's composition gives the right answer under Distinct
(set semantics) and the wrong one under Bag unless multiplicity is carried as
a bit product on the support. The carrier's closure under the multiplicity
operation decides correctness.
**Receipt:** lance-graph #1685.
**Rhymes-with:** BC-2026-10-23-1, BC-2026-10-23-2, BC-2026-10-23-3
**Cluster:** NAMED → IDEAS 2026-10-23 "Lumpability as the closure check" (trail T-CLOSURE)

### BC-2026-10-23-3 — Warshall closure is well defined because the register is closed under ∨ and ∧
**Observation:** DR256's transitive closure over the 256×256 bit register
needs no snapping. Every intermediate stays a 256×256 bit matrix.
**Receipt:** `dir-rmap-probe` Qr register (D-QR plans, INTEGRATION_PLANS
catch-up 2026-10-23).
**Rhymes-with:** BC-2026-10-23-1, BC-2026-10-23-2, BC-2026-10-23-4
**Cluster:** NAMED → IDEAS 2026-10-23 "Lumpability as the closure check" (trail T-CLOSURE)

### BC-2026-10-23-2 — spectral radius is degenerate on Boolean and byte-map carriers
**Observation:** On the Boolean DAG nest ρ = 0 (nilpotent), and on the finite
byte-to-byte maps ρ = 1. Spectral radius only carries information on Bag
(counting) composition. Whether an operator can grow depends on the carrier
it acts in.
**Receipt:** evaluated in session 01G22yT6htkcdyXsihxxXdrg on 2026-10-23
(review of an external relay); no PR.
**Rhymes-with:** BC-2026-10-23-1, BC-2026-10-23-3, BC-2026-10-23-4
**Cluster:** NAMED → IDEAS 2026-10-23 "Lumpability as the closure check" (trail T-CLOSURE)

### BC-2026-10-23-1 — a compose table collapses because its codebook is not closed under bind
**Observation:** `ComposeTable` agreement with exact composition drops to
0.025 / 0.043 at k = 2. 100% / 88.9% of XOR-bound pairs fall outside the
codebook's support, so every hop snaps.
**Receipt:** lance-graph #1726,
`entries/2026-10-23-ps2-compose-chain-preregistration.md`.
**Rhymes-with:** BC-2026-10-23-2, BC-2026-10-23-3, BC-2026-10-23-4
**Cluster:** NAMED → IDEAS 2026-10-23 "Lumpability as the closure check" (trail T-CLOSURE)

### Trail T-CLOSURE — is the carrier closed under the operation the plan composes with?
Four crumbs from four independent sources: a probe, a review, a register
design and a typed-composition fix. Three or more crumbs is enough to
trigger a naming attempt.
**Candidate mechanism (not adopted):** a plan-time `closed(carrier, op)`
check that decides whether a composition may stay a lookup, must snap, or
must refit the carrier.
**Kill:** a closed carrier on which the composition still fails.
**First probe:** TORQUE row T-2026-10-23-1 (EXO-129).
**Verdict:** RHYME, 2026-10-23.
**Naming attempt, 2026-10-23 (after crumb BC-2026-10-23-14):** *lumpability* (Kemeny–Snell).
A partition of the state space composes under an operation exactly when it is
lumpable for that operation, i.e. when every member of a cell goes to the
same cell. Purity is its data-only observable. Named, not adopted. See
IDEAS 2026-10-23. Whether crumbs 2 and 3 (spectral radius, Warshall over
∨/∧) are instances of it, or only rhyme with it, is still open.
Scope limit from session 013z (crumbs BC-2026-10-23-5 and -6 on this trail,
#1730, NULL/K3 carrier and sort-closed composition): closure decides
CORRECTNESS, not cost. Those crumbs' `Cluster:` lines are regraded to NAMED (#1730 is on main).

**Evidence appended 2026-10-23 (session_013z2e1YXkm6o6HNrDn5BPWA):** crumbs
BC-2026-10-23-5 (K3 NULL, independent of the palette and DR256 lines) and
BC-2026-10-23-6 (sort-closed composition, weaker). Naming attempt, HYPOTHESIS,
not adopted: a rewrite or composition is sound only when the carrier is closed
under every operation it assumes; otherwise snap (lossy, P-S2), extend the
carrier (K3, Bag multiplicity) or refuse (`Complement` under K3,
`BagNeedsMultiplicity`). Scope limiter from BC-2026-10-23-6: closure decides
correctness, not cost. Proposed regrade RHYME → NAMED left to the trail owner.


**Evidence appended 2026-10-23 (session_019fbZ92dFKQ5BJHrA3YcH9A):** BC-4 is
a near-counterexample to the word "closed". A Distinct carrier IS closed under
join (a set joined with a set is a set), yet the Bag answer is wrong; K3 is the
same: {T, F} is closed under ∧ ∨ ¬, the failure is that NULL has no image in
it. Candidate refinement, HYPOTHESIS, not adopted: the condition is that the
map from the true domain into the carrier commutes with the operation and with
the measure read at the end (`h(a ⊕ b) = h(a) ⊕' h(b)`, abstract-interpretation
soundness); closure of the carrier is necessary, not sufficient. Cheap test it
implies: the one-step commutation defect rate (P-S2: 100% / 88.9% of bound
pairs outside support) should predict k-step agreement without running k hops.
**Kill for the refinement:** a case where the map commutes at one step and the
k-step composition still fails.
