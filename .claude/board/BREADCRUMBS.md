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

### BC-2026-10-23-4 — the same join is right under Distinct and wrong under Bag
**Observation:** D-RFT-1's composition gives the right answer under Distinct
(set semantics) and the wrong one under Bag unless multiplicity is carried as
a bit product on the support. The carrier's closure under the multiplicity
operation decides correctness.
**Receipt:** lance-graph #1685.
**Rhymes-with:** BC-2026-10-23-1, BC-2026-10-23-2, BC-2026-10-23-3
**Cluster:** RHYME (trail T-CLOSURE, Linear EXO-132)

### BC-2026-10-23-3 — Warshall closure is well defined because the register is closed under ∨ and ∧
**Observation:** DR256's transitive closure over the 256×256 bit register
needs no snapping. Every intermediate stays a 256×256 bit matrix.
**Receipt:** `dir-rmap-probe` Qr register (D-QR plans, INTEGRATION_PLANS
catch-up 2026-10-23).
**Rhymes-with:** BC-2026-10-23-1, BC-2026-10-23-2, BC-2026-10-23-4
**Cluster:** RHYME (trail T-CLOSURE, Linear EXO-132)

### BC-2026-10-23-2 — spectral radius is degenerate on Boolean and byte-map carriers
**Observation:** On the Boolean DAG nest ρ = 0 (nilpotent), and on the finite
byte-to-byte maps ρ = 1. Spectral radius only carries information on Bag
(counting) composition. Whether an operator can grow depends on the carrier
it acts in.
**Receipt:** evaluated in session 01G22yT6htkcdyXsihxxXdrg on 2026-10-23
(review of an external relay); no PR.
**Rhymes-with:** BC-2026-10-23-1, BC-2026-10-23-3, BC-2026-10-23-4
**Cluster:** RHYME (trail T-CLOSURE, Linear EXO-132)

### BC-2026-10-23-1 — a compose table collapses because its codebook is not closed under bind
**Observation:** `ComposeTable` agreement with exact composition drops to
0.025 / 0.043 at k = 2. 100% / 88.9% of XOR-bound pairs fall outside the
codebook's support, so every hop snaps.
**Receipt:** lance-graph #1726,
`entries/2026-10-23-ps2-compose-chain-preregistration.md`.
**Rhymes-with:** BC-2026-10-23-2, BC-2026-10-23-3, BC-2026-10-23-4
**Cluster:** RHYME (trail T-CLOSURE, Linear EXO-132)

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
