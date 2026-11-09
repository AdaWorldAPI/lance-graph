# Plan: the GitHub-analogue Thought-Git lifecycle as a shared contract (2026-11-02)

**D-id:** D-TGL ([STATUS_BOARD](../board/STATUS_BOARD.md) rows D-TGL-0 … D-TGL-G12).
**Status:** CONTRACT, docs only. Nothing here is wired; no workflow engine is built.
**Sources:**
- P4 review [entry](../board/entries/2026-11-02-p4-thought-git-codegraph-dt64-review.md);
- operator decisions and lab result [entry](../board/entries/2026-11-02-p4-operator-decisions-lifecycle-lab.md);
- `thought-git-api-v1.md` (D-TGA) and `thought-git-basics-v1.md` (D-TGB);
- `crates/thought-git-probe`.

DECISION (operator, 2026-11-02): Thought-Git maps the processes of Git/GitHub onto DR³ and DT64. It does not store physical Git branches.

- W0 is the canonical main.
- W1..63 are generation-qualified, reusable working slots.
- A durable `ThoughtCommitId` is independent of the slot.
- **Lifecycle:** `branch → change → diff → PR → checks → review → rebase/squash → merge/land → receipt/replay`.
- **On refusal:** `rework` or `discard`.
- **Scope:** one shared process semantics for Codegraph, DT64, XLOG and later participants.
- **Exclusions:** no 63 graph copies, no extra semantic store, no new method authority.
- **REVISIT WHEN:** EXO-148 lands its types, or a participant cannot express a step with the primitives below.

## 1. Steps as existing primitives

A participant (Codegraph, DT64, XLOG …) supplies four things:
- what a change is;
- how it is checked;
- how it is recomputed against a newer W0;
- how it is applied.

Thought-Git owns the rest: identity, slot, provenance, admission and order.

| step | Thought-Git primitive (owner) | participant hook | state today |
|---|---|---|---|
| branch | `SlotPool::branch` → `Slot{index, generation}`; identity `ThoughtCommitId` | none | slot pool shipped (lab, D-TGB-1); identity MISSING (EXO-148) |
| change | thought write (D-TGA-1) | Codegraph: a source revision; DT64: a route overlay (G11); XLOG: a delta under its own rules | Codegraph lab shipped (input = source rows); others open |
| diff | `diff` mask (D-TGA-1) | Codegraph: `PreparedRefresh` (the checked op batch) | Codegraph lab shipped |
| PR | enqueue in `MergeQueue` (D-TGA-6). A PR record = `ThoughtCommitId` + base `DurableRef` + receipt plan | none | MISSING (D-TGA-6, EXO-148) |
| checks | deterministic, rerunnable gates, each a named refusal (§2) | participant validation | partly shipped (lab + code) |
| review | authorization, never computation: MUL/Rubicon gate and revision gate G1–G7 | none | gate exists (probe `the_mul_gate_decides_whether_the_write_crosses`); wiring open |
| rebase | D-TGA-3 (DRed) for derived facts; replace-and-recompute only under §3 | participant recompute against the current W0 | Codegraph lab shipped (F-LAND); DRed MISSING |
| squash | `squash_gate` / `protected_squash` (D-TGA-5/5D), under G8 for bag cells | none | value cells shipped (lab); bag cells contract only (§4) |
| merge/land | D-TGA-4 (up to date before landing) + D-TGB-3 (`LanceCycleWriter`), published per G10 | participant apply (infallible after its fence) | Codegraph apply + fence shipped (lab); durable land MISSING |
| receipt/replay | D-TGB-4 `Receipt`, `MetadataTable`, `CommitWitness` | participant program id + provenance (Codegraph: PROV, `binary-lanes` §8) | lab shipped; durable MISSING (EXO-148) |
| rework | same slot, new thought commit whose parent is the refused one; back to `change` | none | contract only |
| discard | `SlotPool::free`; the generation moves on and old handles go `Stale` | none; W0 untouched (F-DISCARD) | slot pool + Codegraph F-DISCARD shipped (lab) |

**PR states are readings, not a stored state machine.** Each one follows from records that already exist:
- `Draft`: the slot is live and the thought is not enqueued.
- `Open`: the thought is enqueued.
- `Checking` / `ChecksFailed`: the check folds are absent or refused.
- `Approved`: the review gate passed.
- `Behind`: the base is not the current W0.
- `Landed`: a sealed metadata row exists.
- `Discarded`: the slot was freed with no landing.

No additional table holds these states.

## 2. Checks (each a named refusal, never an empty answer)

- **Oracle:** a contradiction with a given, or a divergence between landings (`next_version`).
- **Squash gate:** R1 and R2; a protected reader the retained data cannot answer (`ProtectedRefusal`).
- **Codegraph:**
  - `prepare_refresh` refusals: intake, NeedsReopen, a cube stage refusal, a closure stage refusal;
  - `commit_refresh` → `Stale`.
- **DT64:**
  - `admit` (method, `sig_rev`, guard);
  - the D-DT64-AUTH census for production words;
  - `route` reservation (validated in prepare, G11).
- **Writer:** `Fenced`, `HashConflict`, `InvalidArtifact` (`persist_sink.rs`).
- **Receipt:** every `receipt::Refusal` arm.

## 3. Replace-and-recompute instead of DRed (conditional exception)

DECISION (operator, 2026-11-02): allowed for Codegraph facts that can be reconstructed completely and deterministically from source artifacts with an immutable identity (§8 of `d-codegraph-0-binary-lanes-v1`: PROV with LLBC digest and source revision).

**Not** allowed for:
- derived XLOG facts;
- incremental conclusions;
- DT64 routing.

These keep D-TGA-3 (four masks, DRed).

## 4. G8: bag semantics across thoughts

DECISION (operator, 2026-11-02): Thought-Git merges **source revisions**, never raw edge ADDs. For a source S, its previous contribution is removed entirely and its new contribution is recomputed entirely.

- Identical revisions of the same source count once (the second lands as a no-op: equal digest).
- Different sources may contribute the same edge several times.
- Several real call sites stay counted several times.
- Competing revisions of the same source need an explicit conflict decision. A checked three-way merge comes later.
- Squash loses no real multiplicity and applies no identical contribution twice.

The bag algebra stays with XLOG/DR (multiset delta log, merge on read). Thought-Git manages identity, provenance and admission.

**Pinned in the lab:** F-LAND's oracle counts 5 call sites of `m::b` from two sources after both branches land.

## 5. Production gates (all OPEN; lab results never satisfy them)

- **G9 Lifecycle.** Every step in §1 is expressible with the listed primitives, and the PR states are readings. Closes when D-TGA-1/3/4/6 and D-TGB-2/3/5 are delivered against this table.
- **G10 Atomic publication.** "Codegraph commit → `BindStore::route` → `LanceCycleWriter`" is NOT a valid production land: a later `Fenced` would leave published partial state. The required order:
  1. **Prepare everything.**
     - Codegraph: `PreparedRefresh`.
     - DT64: route overlay validated against the `BindStore`: the class is registered and the position is inside its reservation. **`BindStore` has no prepare/validate route API today; that is a named gap.**
     - The land record and its batch hash.
  2. **Durable commit first:** `commit_cycle` with the land record.
     - `Fenced` / `HashConflict` → nothing applied, regenerate.
     - `Committed` / `Reconciled` → continue.
  3. **Apply W0 and DT64 in memory from the durable land record,** with no error path. The participant's own fence cannot fire, because the writer is W0's sole mutator.
  4. **Crash between 2 and 3:** replay from the durable land record at start-up (the `recover_and_apply` shape in `persist_sink.rs`).
- **G11 Branch-local routing.** A branch never calls `BindStore::route` and never uses a DR position minted in the branch as a durable routing key.
  - Routing changes live in the branch's route overlay; registration of a derived classid is additive and global, but stays unreachable until a land routes to it.
  - Positions are assigned by W0 at land (the Codegraph analogue is F-LAND).
- **G12 Identities stay separate:** W-slot, slot generation, `ThoughtCommitId`, `DurableRef`, `RouteVersion`, Lance version, source Git SHA.
  - For Codegraph the source Git SHA is provenance (PROV), not a thought identity.
  - Joins go only through explicit records: receipt, `CommitWitness`, `MetaKey`, land record.
  - The Codegraph generation fence is local and in memory, and is none of these.
- **EXO-148.** `DurableRef` and a never-reused `ThoughtCommitId` are required before any durable land, replay or PR record.

## 6. Order

1. F-LAND and F-DISCARD (done, lab).
2. G8 recorded (this plan §4, board).
3. This contract.
4. G9–G12 + EXO-148 registered as production gates.
5. Only then D-TGB-7: the shared crate (`lance-graph-thought-git`), with Codegraph as its first participant.

No automatic merge. No production land on the strength of lab tests.
