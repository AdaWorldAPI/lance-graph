# D-FOUNDRY-PARITY-1 — DR × DT64 as an operational ontology: evidence matrix and first lab

> **Status:** ACTIVE. Level R lab merged (#1991, #1995); G6 in #1996. Governance
> (G2–G4) is composition work over existing pieces (§0, corrected 2026-11-10).
> **Lab:** `crates/foundry-parity-lab` (own workspace root, not a member).
> **Contract read first:** `.claude/knowledge/dr-dt64-locality-execution-contract.md`
> (its own evidence table grades Foundry parity HYPOTHESIS; this plan does not
> change that grade).
> **Verify:** `cargo test --release --manifest-path crates/foundry-parity-lab/Cargo.toml`
> and `cargo run --release --manifest-path crates/foundry-parity-lab/Cargo.toml --bin bench`.

## 0. Report (current, 2026-11-10)

**Verdict.** No Foundry parity is claimed; nothing was compared against a
running Foundry. On the read side, DR + DT64 + mask-RISC answer operational
questions over source-owned planes with exact oracle parity, correct
refusals and zero copies. On the governance side, the parts exist in other
crates but are not composed on this path.

**What the lab shows (32 tests, every guard disable-verified):**

| Question | Path | Measured (64 rows) |
|---|---|---|
| `CanCook` | arm B, hand-written mask-RISC | 470 ns, 0 B copied |
| `CanCook` as a registered method | G6: `admit` → bind → relalg lowered once → execute | 359–400 ns |
| per-store cost, 64 × 8 | arm C, INT8 packed GEMM (AMX) | 0.65–1.17 µs |
| per-store maximum | arm B layer-cake sweep | ~15–19 µs (8× slower than objects: no min/argmin) |
| same over objects | arm A | ~1.8 µs + 21 µs and 51 968 B to build |

**What exists elsewhere and what is missing (audited 2026-11-10):**

| Area | Exists | Missing on this path |
|---|---|---|
| Authorization (G2) | `ogar-auth` → `ogar-rbac` → `lance-graph-rbac::ScopedDecision`: row `ScopeSpec` + `WideFieldMask`, delegation; DT64 `mask_where` over markings | Apply it to a DT64 binding: per-row scope lane → resident plane; `visible_fields()` limits bound fields |
| Scenarios (G3) | thought-git branches, `next_version`, squash gate; xlog `stage_closure` preview | An overlay frame a DT64 binding can read |
| Actions / writeback (G4) | thought-git Rubicon outcomes, `commit_cycle` + `CommitWitness`, `stage`/`seal`; xlog `stage`/`commit`; rbac Write/Act; #1993 durable map writes | One authorized, preconditioned, committed, witnessed action end to end |
| Provenance | binding witnesses, thought-git receipts + replay, commit witnesses, rbac audit identity | One record per answer joining them |
| Algebra (G1) | counts, sums, GEMM | element-wise min/argmin (decision P1b) |
| Relations (G5) | — | ADMISSION B decision (EXO-140) |

**Next step with the most value:** G2 in the lab, using the existing
`authorize_scoped` decision instead of the stand-in.

Architecture maps: `.mermaid/` (D-FOUNDRY-MAP-1).

## 1. Overview

The question is not "can we clone Foundry". It is: can source-owned SoA planes,
addressed through DR and interpreted through DT64, answer *operational*
questions about an organisation — what can it produce, where, at what cost,
under whose authorization — and how far that is from Foundry's documented
Ontology capabilities. Nothing here was compared against a running Foundry
instance; Foundry columns cite its public documentation only, as evidence of
*documented* capability, never of internal implementation or performance.

## 2. Checklist

- [x] Read the DR/DT64 locality contract, `lane.rs` (`LaneBinding`,
      `borrow_all`), the #1987/#1989 lab, mask-RISC IR (`Pred`, `MaskOp`,
      `Terminal`), `MethodBinding`/`MethodRef`, `contract::scenario`, `lance-graph-rbac`.
- [x] Evidence matrix (§4).
- [x] Lab: deterministic 64-store × 64-category fixture, source owner, 211-field class.
- [x] Independent scalar oracle + materialized-object null model (arm A).
- [x] `CanCook`, `CanFulfill`, `NetworkCapacity`, network ingredient cost through DT64 + mask-RISC (arm B).
- [x] Named cases 01–09, refusals, negative controls (§5).
- [x] Disable runs: five guards broken one at a time, suite red each time, tree restored.
- [x] Bench in ns/µs (§6).
- [x] P1a (2026-11-10): per-store cost through the INT8 packed GEMM (ndarray#418), arm C — oracle-exact, staged A counted (§5a, §6).
- [ ] P1b: element-wise cross-lane arithmetic in an admitted operator (per-store max without the sweep, limiting resource) — see §7 G1.
- [x] P2 / G6 (2026-11-10): `CanCook` registered as a DT64 `MethodBinding` and executed from its `MethodRef`; the WHEN guard is lowered by relalg once per binding and executed per question, no lab lowering (§5b, §6).
- [ ] P2 / G5: recipe demand as bound relation lanes (§7 G5; waits on ADMISSION B, EXO-140).
- [ ] P3: scenario overlay frame (§7 G3); governed action / writeback gate (§7 G4).
- [ ] Production IAM in the read path instead of the lab stand-in (§7 G2).

## 3. Architecture of the lab (what is where)

```
fixture (only value definition)
   │ ingest (cold, once)
   ▼
Network  ── source owner: 211 typed planes in ITS row order + its FrameStore
   │  LaneSource::lane_of / borrow   (addresses only; never a copy)
   ▼
World    ── DT64: ClassView basis (211 fields), route, BindStore
   │  bind_lanes(demanded fields only)   → LaneBinding   (cold, ~5.5 µs)
   │  borrow_all (one check, N lookups)  → pointer table (~0.26 µs)
   ▼
ItemQuery ── mask-RISC programs over the borrowed planes, per question
   CanCook(q)        = visible ∧ offered ∧ equip ∧ ⋀_k(known_k ∧ stock_k ≥ d_k·q) ∧ staff ≥ ⌈q/u⌉
   CanFulfill(q,t)   = CanCook(q) ∧ slots ≥ 1 ∧ minutes ≤ t − prep
   NetworkCapacity   = Σ_q Count(CanCook(q))         (lawful by monotonicity, checked)
   network cost(q)   = Σ_k d_k·q · MaskedSumI32(CanCook(q), cost_k)
   per-store max     = layer-cake readback of CanCook(q) masks
```

X = 64 stores (rows), Y = 64 categories (stock / known / cost lanes), Z =
typed operational interpretation (staff skills, equipment, offered items,
delivery slots and minutes, the identity lane). Z is not one homogeneous axis
and nothing is materialized as a 64 × 64 × 64 tensor: the dense Cartesian
product never exists. The source holds 35 168 bytes; arm B copies 0 of them.

Identity: positions become stores only by reading the bound `storeId` lane at
a result's set bits; results are keyed by global identity, so a source in
another row order answers identically (tested).

Authorization: a **lab stand-in** — the reader is denied global id 1006,
expressed as `NeU32(storeId, 1006)` inside the program. DR's `Scope` is
checked only as a binding consistency witness (locality contract, boundary
5); it never authorizes. The stand-in is not IAM.

## 4. Foundry parity evidence matrix

Statuses: IMPLEMENTED (source + tested) · LAB (executable, oracle-validated) ·
PARTIAL · PROPOSED · BLOCKED · ABSENT · UNKNOWN. Verdicts: PARITY-DEMONSTRATED ·
FUNCTIONALLY-RELATED · LOWER-LEVEL-ADVANTAGE-HYPOTHESIS · MISSING · BLOCKED ·
NOT-COMPARABLE · UNKNOWN. No row claims PARITY-DEMONSTRATED: nothing was
compared against Foundry.

| # | Foundry capability (documented) | DR/DT64 counterpart | Source evidence | Status | Missing contract | Smallest falsifier | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Object types + typed properties | `ClassView` basis + `ClassSchema` + typed source lanes | `lance-graph-dt64/src/lib.rs` (`register`, `ClassSchema`), `lane.rs`; lab `bind.rs` (211 fields) | LAB | Property types are the source's, not declared in DT64 (lane kind checked only at use: `ArmRefusal::PlaneKind`) | Bind a mask field where an i32 lane is expected → refused | FUNCTIONALLY-RELATED |
| 2 | Stable identity + provenance | DR route + witnessed coordinate; global id lane | `RouteAt`, `CoordSpace`, `Frame`; lab reorder test | LAB | No per-value provenance (who wrote this stock, from which system) | Two sources disagree on a value; no witness says which wins | PARTIAL → MISSING for provenance |
| 3 | Link types / traversal (search-around) | Relation lanes; `graph_hop::OutgoingHop` | mask-RISC `graph_hop` is feature-gated OFF (value-directed gather banned, ADMISSION B) | BLOCKED | Admitted relation traversal over source-owned planes; recipe→ingredient is a fixture function here, not a bound relation | Bind recipe demand as a relation lane and traverse without Gather | BLOCKED |
| 4 | Object sets, filtering (WHERE) | Scoped WHERE population = mask program | lab `cook_mask`, `fulfill_mask` | LAB | — | Parity suite (8 items × 3 q × 64 stores) | FUNCTIONALLY-RELATED; LOWER-LEVEL-ADVANTAGE-HYPOTHESIS (0 bytes copied, 470 ns / question at 64 rows) |
| 5 | Computed properties / functions | `MethodBinding` (ON×WHEN×DO) + `MethodRef`; WHEN guard lowered by relalg to one mask-RISC program | `lib.rs:215`, `:353`; lab `method.rs` (§5b) | LAB (read-only guards) | Production owner of a prepared program; DO effects still not executed | `every_registered_can_cook_method_matches_the_oracle` | FUNCTIONALLY-RELATED (read-only) |
| 6 | Versioned method behaviour | `SigRev`, `GuardWitness`, `Stamp` checked by `admit` before execution | `lib.rs:345-370`; lab `method.rs` | IMPLEMENTED (identity) / LAB (execution) | Same as 5 | `a_reference_with_another_guard_is_refused` (forged guard, other `SigRev`), foreign store refused | FUNCTIONALLY-RELATED (read-only) |
| 7 | Actions / state changes | thought-git write path; xlog two-phase edge writes; rbac write/act grants | `thought-git-probe` `Outcome::{Commit,Plan,Prune,Veto}`, `VersionStore::commit_cycle` (one sealed version + `CommitWitness` per cycle), `MetadataTable::stage`/`seal`; `lance-graph-xlog` `Cube::stage` (checks, no version) / `commit` (cannot refuse); `lance-graph-rbac` `Operation::{Write,Act}`, `PermissionSpec::can_write`/`can_act`; durable map writes in #1993 (open) | LAB (probe) / PROPOSED (`thought-git-lifecycle-contract-v1`: CONTRACT, docs only) | Not composed for this domain: no store-network action passes authorize(Write/Act) → stage → commit → witness; no precondition on a source value (only frame / sealed-version refusals) | Reassign order 12→17: authorized stage, stale-precondition refusal, commit, witness | PARTIAL (pieces exist, not composed) |
| 8 | Multiple sources + sync | `LaneSource` per owner, `FrameStore` per source | `lane.rs` `source_owns`; lab uses one source | PARTIAL | Cross-source composition; G2L across differing spaces is refused, not translated | Two sources, different orders, one question | BLOCKED (G2L OPEN in the locality contract) |
| 9 | Source ownership, logical materialization | One canonical owner; borrowed lanes | lab: arm B copies 0 bytes; arm A copies 51 968 | LAB | — | `only_demanded_fields_are_bound_and_borrowed_once` | LOWER-LEVEL-ADVANTAGE-HYPOTHESIS |
| 10 | Security / authorization | `ogar-auth` `establish` → `VerifiedLogin` → `ogar-rbac` `authorize_user` / `authorize_identity` → `lance-graph-rbac` `ScopedDecision` / `MembershipDecision`: row scope (`ScopeSpec`: tenant + path, restrictive AND) and field projection (`WideFieldMask`, union; `visible_fields()` empty on refusal); delegation (`acting_party`); DT64 `Resolved::mask_where` over `PropertySpec::marking` | `lance-graph-rbac/src/authorize.rs:165-250`; `ogar-rbac/src/lib.rs:155,327`; `contract::selection::walk_rails_authorized` consumes the field mask | IMPLEMENTED (decision) / ABSENT (applied in a DT64 binding) | Apply the decision on this path: `ScopeSpec` needs a per-row scope lane and becomes a resident plane (Quack `Binder::live`); `visible_fields()` ∩ the marking mask restricts which fields `bind_lanes` may bind | Lab: replace the `NeU32` stand-in with an `authorize_scoped` decision; a narrowed role binds fewer fields and sees fewer stores | PARTIAL (decision built; read-path wiring missing) |
| 11 | History / replay | `FrameStore` revisions; Lance versions; thought-git receipts | lab stale-frame refusal + rebind; `thought-git-probe` `Receipt` + `replay` / `replay_landed` (recompute from a sealed base, refusing `SourceChanged` / `ConfigChanged` / `OutputMismatch`) | LAB (invalidation; replay in the probe) | Bind a DT64 lane at a historical frame | Answer at frame N after the source moved to N+1 | PARTIAL |
| 12 | Derived state + invalidation | `LaneBinding::check` on every open | lab `a_stale_frame_is_refused_and_a_rebind_sees_the_restock` | LAB | Incremental recompute of only the changed subset (lab recomputes the whole question; it is 64 rows) | — | FUNCTIONALLY-RELATED |
| 13 | What-if scenarios (beta in Foundry docs) | `contract::scenario::ScenarioBranch`; thought-git branches; xlog staged preview | `thought-git-probe` `SlotPool::branch` (from main or a live thought), `next_version` (from-scratch oracle recording contradictions and divergences), `squash_gate`; `lance-graph-xlog` `Cube::stage_closure` (the derived result as it would be after commit, changing nothing) | LAB (probe) | A branch as an overlay a DT64 binding reads (overlay frame witness, §7 G3) | Store 12 closed on a branch: answer = oracle, main's frame unchanged | PARTIAL |
| 14 | Aggregations / analytics | mask-RISC `Count`, `MaskedSumI32`, group terminals; ndarray INT8 packed GEMM | lab `network_capacity`, `network_cost`; arm C `per_store` (per-store cost) | LAB | Per-row min / argmin across lanes (§7 G1) | Limiting resource through an admitted operator | PARTIAL |
| 15 | Operational applications | a2ui-rs / consumer CustomObject | not exercised here | UNKNOWN | — | — | NOT-COMPARABLE here |
| 16 | Audit / decision provenance | Binding witnesses; thought-git receipts and commit witnesses; rbac identity | `LaneBinding` (route, frame, basis); `thought-git-probe` `Receipt` (base, program, inputs, seed, config, source fold, roots, output), `CommitWitness` (data version, cycle, cohort); `lance-graph-rbac` `ResolvedIdentity::auth_classid` (doc: "for the audit"), `acting_party` | PARTIAL (pieces in the probe and the rbac crate) | One record per answer joining who asked (identity, delegation), which binding witnesses and which result | — | PARTIAL |
| 17 | Data connections / egress | Source adapters (OGAR, Lance) | outside this lab | UNKNOWN | — | — | NOT-COMPARABLE here |
| 18 | AI-assisted, governed decisions | none on this path | — | ABSENT | Depends on 7, 10, 16 being composed | — | MISSING |

Correspondence hypotheses (§4 of the brief), graded: `ClassView` ↔ object
type: FUNCTIONALLY-RELATED, but ClassView alone carries no security, search,
history or writeback. `MethodRef` ↔ Action Type: **rejected** — a `MethodRef`
names a read-only method revision; it is not a transaction, has no
preconditions, idempotency or commit. Object set ↔ WHERE mask: LAB-supported.

## 5. The lab, case by case

| Case | Expected | Arm B | Test |
|---|---|---|---|
| Store 01 | feasible, max 30, limit staff | ✓ (limit oracle/arm A only) | `the_named_cases_answer_as_designed` |
| Store 02 | max 0, cheese | ✓ | same |
| Store 03 | max 0, no cook | ✓ | same |
| Store 04 | max 0, oven | ✓ | same |
| Store 05 | q=1 yes, q=2 no | ✓ | same |
| Store 06 | no → yes after restock + rebind; stale binding refused | ✓ | `a_stale_frame_is_refused_and_a_rebind_sees_the_restock` |
| Store 07 | NotAuthorized, never "cannot cook" | ✓ | named cases + `bypassing_authorization_reveals_the_denied_store` |
| Store 08 | same answers in another row order, by identity; canonical selection refused | ✓ | `another_row_order_*` |
| Stale frame | refused (`SourceChanged`) | ✓ | stale test |
| Store 09 | unknown cheese ≠ zero; bypass reads the stale 1000 as stock | ✓ | `unknown_is_not_zero`, `bypassing_the_known_guard_*` |

Also refused: a lane serving another scope; a field the source does not map;
a question needing a field that was not bound (`network_cost` without cost
lanes); quantity outside `1..=128`. Disable runs (each red, then restored):
staff predicate threshold → 0; denied mask → empty; ingredient threshold off
by one q; unknown mask → empty; fulfil deadline → `i32::MAX`.

### 5a. Arm C — per-store cost as one INT8 GEMM (P1a)

`C[store][item] = Σ_k cost[store][k] · demand[k][item]` through
`ndarray::simd::int8_gemm_packed_strided` (ndarray#418, AMX `TDPBUSD` on the
lab host). Only the cost lanes of categories some item demands are bound (plus
`storeId`). **A is staged**: the source owns one `i32` lane per category (A's
transpose) and the kernel reads `u8` rows, so the bound lanes are copied once
per binding into a 64 × 64 `u8` buffer (4 096 B, counted). A cost that does not
fit `u8` is refused, never truncated. B (the menu's demand matrix) is packed
once and reused only under the same menu revision — the identity-and-revision
rule #1992's `ReuseKey` applies to its operands, never a digest.

Tests (7): oracle parity for 64 stores × 8 items × q ∈ {1, 3, 128} and against
the materialized null model; the sum over arm B's `CanCook` mask equals arm B's
`network_cost` (two independent arms); demanded lanes only, one borrow each, A
staged once; `u8` overflow refused and 255 admitted exactly; a stale binding
refused after a price change; another menu revision and out-of-range q
refused; reversed row order answers by identity. Disable runs (each red, then
restored): truncate instead of refuse; ignore the denied mask; demand in the
wrong column; admit any menu; stage only the first cost lane.

Not exercised: the scalar tier of the kernel (the host has AMX); ndarray#418
tests both tiers against each other. The limiting resource (argmin) remains
BLOCKED — it is not a matrix product.

Not exercised: the `NotMonotone` and `Ceiling` guards of the layer-cake sweep
are defensive; no admitted fixture can reach them (feasibility predicates are
all `≥ threshold(q)` with thresholds non-decreasing in q). They are labelled,
not claimed as tested.

### 5b. G6 — `CanCook` as a registered method (P2, part)

`crates/foundry-parity-lab/src/method.rs`. `World::new` registers one
`MethodBinding` per (item, quantity), quantity ∈ {1, 3}: the quantity is a
constant of the guard, and DT64 guards take no parameters, so it is part of
the method id. The WHEN guard says "this store can make `q` of item `m`" as a
`SetExpr` over field positions: mask fields as `Sym(field)`, value fields as
residuals on lane `field` (`stock_k ≥ d_k·q`, `staff ≥ ⌈q / per_staff⌉`,
offered, equipment, every demanded category known). `reads` is exactly the
fields the guard names; `effects` is empty (read-only).

`PreparedMethod::prepare` admits the `MethodRef` (`BindStore::admit`), binds
`reads` + `storeId`, borrows them once, and renames field positions to the
bound planes. The reader scope (IAM stand-in) is conjoined there, never stored
in the method. `prepare` then lowers the guard ONCE through relalg
(`rel_from_set` + `run_with_program`) and keeps the returned mask-RISC
program; `run` only executes it (`execute_into`, `Out::Mask`). The lab writes
no lowering. `rel_from_set` is the one relalg addition: a public wrapper over
the conversion `run_program` already applied (`from_set`, crate-private).
`prepare` refuses a lowering of more than one fold (`Folds`), since relalg
keeps only the last fold's program; that tripwire cannot fire today, and a
test pins that every guard shape lowers to one fold.

Tests (8, in `tests/integration/method.rs`): every registered method equals
the oracle's feasible set and count; another row order answers identically by
identity; the reader scope is load-bearing and outside the method; a reference
with another guard (`GuardMismatch`) or signature revision is refused, as is
one issued by another store (`ForeignStore`); a method whose `reads` misses a
field its guard names is refused (`NotBound`), never answered with the field
read as absent; repeated runs of one prepared method equal the first (the
scratch is reused); every guard shape lowers to one fold. Disable runs, each
red and restored: drop the staff conjunct; drop the known conjuncts; stock
threshold off by one; drop the reader scope; skip the identity mapping; lower
`All` instead of the guard; treat an unbound field as `All`.

**Measured:** the first version called `run_program` per question, which
re-lowers every time: ~21 µs per question. Lowering once per binding brings a
question to 359–400 ns, level with arm B's hand-written program (474 ns),
and moves the one lowering into `prepare` (~15 µs, was ~9 µs).

## 6. Measurement (MEASURED)

`cargo run --release --manifest-path crates/foundry-parity-lab/Cargo.toml --bin bench`;
one VM, Intel Xeon @ 2.30 GHz, 4 vCPU, single thread, median of 41 warm runs,
three runs, ranges across runs:

| Stage | ns |
|---|---|
| cold: `World::new` (register + route) | 24 770–25 351 |
| cold: `bind_lanes`, 15 demanded of 211 fields | 5 500–5 511 |
| cold: `ItemQuery::open` (`borrow_all` + pointer table) | 254–272 |
| warm: `CanCook(q=1)`, one program, 64 stores | 470–474 |
| warm: `CanFulfill(q=2, 45 min)` | 569–577 |
| warm: per-store answers (layer-cake max, ~31 programs) | 15 146–15 199 |
| warm: `NetworkCapacity` (Σ_q Count) | 13 666–13 944 |
| warm: network cost q=3 (3 × MaskedSum) | 1 367–1 375 |
| arm A: build 64 objects (51 968 B second copy) | 20 818–23 216 |
| arm A: `CanCook(q=1)` over 64 objects | 1 792–1 844 |
| arm A: `NetworkCapacity` | 1 811–1 922 |
| arm C cold: bind 23 cost lanes + `storeId` | 5 567–5 953 |
| arm C cold: open (`borrow_all` + stage A, 4 096 B) | 1 139–1 562 |
| arm C cold: prepare demand band (once per menu revision) | 452–629 |
| arm C warm: per-store cost, 64 stores × 8 items, AMX `TDPBUSD` | 654–1 170 |
| arm A: per-store cost, 64 × 8 (recipes read once) | 1 188–1 661 |
| G6: `method_ref` + `admit` | 5 054–5 254 |
| G6 cold: `PreparedMethod::prepare` (admit + bind + borrow + one lowering) | 14 874–15 193 |
| G6 warm: `CanCook(item 0, q 3)` run, cached mask-RISC program | 359–400 |

Reading, not spin: a single WHERE-shaped question is ~4× cheaper in arm B and
copies nothing; but the per-store maximum and the network capacity are ~8×
**more** expensive in arm B than walking already-built objects, because the
admitted algebra has no element-wise min across lanes and must sweep q. Arm
A's build cost (21 µs) amortises after roughly two capacity questions. At 64
rows every mask is one word, so mask-RISC's fixed per-program cost (program
construction + validation, not separated here) dominates; scaling is OPEN.

Arm C (separate run of the same bench, three runs): per-store cost for all 8
items is ~1.5–1.8× cheaper than arm A's walk once arm A reads its recipes
once. A first comparator called `fx::recipe` per (store, item) and measured
7.7–15 µs; that was 512 allocations, not the objects, and is not the
comparison. At 64 × 64 × 16 the GEMM is tiny; the per-call fixed cost
(tile configuration, output mapping) likely dominates, and scaling is OPEN.

G6 (four runs; one more under load measured 600 ns and is excluded): a
registered method's question costs the same as arm B's hand-written program
once relalg's program is cached per binding (§5b). `admit` is ~5 µs (the
guard witness is normalized on issue and on admit). `World::new` now registers 16
methods and costs ~48 µs (was ~25 µs with none); not separated further.

## 7. Missing-contract register

| Id | Missing operation | Why the current contract cannot express it | Lawful alternative today | Smallest test | Needs |
|---|---|---|---|---|---|
| G1 | Element-wise cross-lane arithmetic: `min_k ⌊stock_k / d_k⌋` per row, argmin (limiting resource). (Per-row weighted sum — per-store cost — is now LAB via arm C, §5a.) | mask-RISC terminals reduce over rows; predicates compare a lane to a constant; min/argmin is not a matrix product | Layer-cake over monotone masks (max only, ~30× programs); none for argmin | Per-store max and limiting resource in one program, equal to oracle | Decision + mask-RISC/Quack PR (H2) |
| G2 | Row-level authorization from IAM in the read path | The decision exists (`ogar-rbac` → `lance-graph-rbac::ScopedDecision`: row `ScopeSpec` + `WideFieldMask`); nothing applies it to a DT64 binding, and the lab has no per-row scope lane | Lab stand-in predicate on the identity lane | Lab: `authorize_scoped` decision as a resident plane plus `visible_fields()` limiting bound fields | A per-row scope lane on the source; where the decision is applied (Quack `Binder::live` is the existing seam) |
| G3 | Scenario overlay over a binding | Branches exist in the thought-git probe (`SlotPool::branch`, `next_version`) and as xlog staged previews; no frame witness lets a DT64 binding read a branch | Oracle only | Store 12 closed on a branch: overlay answer = oracle, canonical frame unchanged | Overlay frame contract joining a thought-git branch to `FrameStore` |
| G4 | Governed action / writeback | Pieces exist: thought-git Rubicon outcomes, `commit_cycle` (one write path, `CommitWitness`, `AlreadySealed` / `DuplicateThought` refusals), `MetadataTable::stage`/`seal`; xlog `Cube::stage`/`commit`; rbac `Operation::{Write,Act}`. Not composed: no action over this domain is authorized, staged, checked against a source-value precondition, committed and witnessed | none in this lab | Preview vs commit of "reassign order 12→17" with a stale-precondition refusal | Composition over the existing pieces; `thought-git-lifecycle-contract-v1` (CONTRACT) is the contract |
| G5 | Bound relations (recipe → ingredient, menu → recipe) | Value-directed traversal is Gather, banned outside ADMISSION B | Relations as fixture functions (lab) | Bind recipe demand as lanes and answer without Gather | ADMISSION B decision |
| G6 | Methods executed from `MethodRef` | DT64 binds behaviour, executes nothing. LAB since §5b: admitted, bound, lowered once by relalg, executed per question at arm B's cost | — | Done in the lab: oracle parity, refusals, 359–400 ns per question | Production: who holds the prepared program across questions (no owner named) |

## 8. The three hypotheses

- **H1 (semantic contract suffices for read-only):** *partly supported.*
  WHERE-shaped feasibility, fulfilment, capacity and network cost are exact
  through existing contracts with no copy. Falsified for anything needing a
  per-row value combining several lanes.
- **H2 (missing algebra):** *supported by G1, G5, narrowed by arm C.* The
  per-store cost IS expressible, as an INT8 GEMM over staged A (§5a). The
  limiting resource still cannot be expressed; the maximum only by a sweep.
- **H3 (infrastructure):** *narrowed (2026-11-10 audit).* The first matrix
  said authorization, scenarios and actions had no contract; that was written
  without reading `ogar-auth`, `ogar-rbac`, `lance-graph-rbac`,
  `thought-git-probe` or `lance-graph-xlog`. They exist: the authorization
  decision is implemented (row scope + field mask), and branches, a single
  write path with commit witnesses, receipts with replay and a two-phase edge
  write exist as probes. What is missing is COMPOSITION on this path:
  applying the decision to a DT64 binding (G2), letting a binding read a
  branch (G3), and one governed action end to end (G4).

None is chosen over the others; all three hold in part.

## 9. Roadmap

- **P0 (this PR):** the lab and this matrix.
- **P1a (done, 2026-11-10):** per-store cost via the INT8 packed GEMM (arm C).
- **P1b:** G1 remainder — decide whether element-wise cross-lane min/argmin
  enters mask-RISC/Quack, then re-run per-store max and limiting resource
  against the oracle.
- **P2:** G6 + G5 — `CanCook` as a registered method lowered from its
  `MethodRef` (LAB, §5b, lowered once per binding); recipe demand as bound
  relation lanes (waits on ADMISSION B).
- **P3:** G3 scenario overlay frame; G4 action gate (preview vs commit) as a
  separate workstream; G2 IAM in the read path.
- **P4:** real source connectors, a2ui screens, deployment.
- **P5:** AI-assisted decisions, only after P3's governance exists.

No effort estimates: there is no basis for them.

## 10. Foundry documentation used

Fetched 2026-11-10 (HTTP 200, no redirects); paraphrased:

- `docs/foundry/ontology` — the Ontology is an operational layer over
  datasets, virtual tables and models, with semantic elements (objects,
  properties, links) and kinetic elements (actions, functions, dynamic
  security).
- `docs/foundry/functions/api-object-sets` (TSv1) — an object set is an
  unordered collection of objects of one type; supports filtering, Search
  Arounds along link types, and aggregations.
- `docs/foundry/object-edits/overview` — an Action is a single transaction
  changing properties of one or more objects; edits add/remove links and
  create/delete objects.
- `docs/foundry/action-types/permissions` — view / edit / apply are separate
  permissions; row and column controls filter what is read when invoking an
  action and do not extend to the action's write.
- `docs/foundry/ontology/overview-ontology-scenario` — **Beta**; a sandbox of
  edits applied by actions over Ontology data; not a data-versioning tool.
- `docs/foundry/object-backend/overview` — Object Storage v2 is the canonical
  store backing the Ontology; v1 (Phonograph) marked for deprecation.

Consequences for the matrix: Foundry scenarios are built *from actions*, so
row 13 depends on row 7 here too; Foundry separates read controls from write
authorization, which is the same split G2 / G4 keep apart. Nothing here claims
knowledge of Foundry internals or performance.
