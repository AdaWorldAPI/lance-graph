# D-BAG-FOLD: decision memo, the zero-copy pointer lens (Arm L)

Read-only. Target: lance-graph PR #1937 at `ae885f06a`, source `514948203`. No source,
plan, Linear or PR state changed.

## Authorities read, by kind

| record | kind | what it says that bears on Arm L |
|---|---|---|
| `entries/2026-10-20-six-decisions.md` §1 ADMISSION B | BINDING DECISION | Preparation may do "counted, version-witnessed, same-table key-range seeks over an attested sorted key" and "construct a projection for subsequent co-addressed evaluation". "A position derived from another relational column or table is not admissible as a read address". "Execution remains co-addressed." `Semijoin` / `EqU32Via` / `Gather` "need call-site classification". |
| `ISSUES.md` ISS-ADMISSION-PREP-BINDING (RULED) | BINDING (applies B) | The admitted evidence shape: the lookup runs in counted preparation, and "its fold program contains no `Gather` and no scatter terminal (checked on the program, disable-verified)". |
| `ISSUES.md` ISS-GATHER-HOP-1616, GATHER-DECISION | BINDING | `OutgoingHop` (`Gather` over `frontier[sources[i]]`) stays quarantined. It closes only by a rewrite as a co-addressed read `Thought(t)[T_a[i] : T_b[i]]`. |
| #1590 (`entries/2026-10-16-ledger-revision-1582-1591.md`) | BINDING (ban), no DECISION record | Value-derived positional reads banned. Existing gather code untouched. |
| `CLAUDE.md` P0 "There are no boundaries (2026-10-21)" | BINDING DOCTRINE, later than B | Zero copy everywhere. The one materialization is the pointer table `$i`. Everything else is "read in place through `$i`". Whole-template form `"{$i[0]}..{$i[15]}" -f $table{$i[0]}..`: a per-column pointer. |
| `.claude/knowledge/zero-copy-lens-law.md` (2026-07-28/29) | BINDING (operator rulings); several rows CONJECTURE | "Cross-tenant pointers are legitimate; cross-tenant values are not." Pointer projection (`target = cur + off`, computed) vs pointer chasing (loaded, then dereferenced). A stored value that a cast reproduces is a projection and must not be stored. |
| `mask-risc/src/ir.rs` `MaskOp::Gather` doc, `ScatterOrU32` survival | EXECUTABLE SOURCE | `Gather` = `dst[i] = foreign.words[lane_u32[i]]`. **Out of range is the zero-fallback, never an error.** A scattered mask is never an intermediate. |
| `quack/src/lib.rs` `Filter::Semijoin` | EXECUTABLE SOURCE | The foreign plane must be RESIDENT, never a mask another Program produced. `Gather` "may survive as a masking op only when its source is already legitimate resident state, its output is tile-local, and no dynamically-created population mask feeds it". |
| `contract/src/ordered_lane.rs` | EXECUTABLE SOURCE | `SealedFacetLane::seal` / `attest_sorted` (non-decreasing, so **duplicates allowed**); `OrderedLaneWitness {version, n_rows, digest of the key sequence, lens}`; `seek_from` = the witnessed same-table seek; `bound` validates the witness against the lane and checks the lens. |
| ISS-WITNESSED-RANGE-DOES-NOT-ATTEST-PLANE-ORDER | OPEN ISSUE | The witness attests the lane, not the execution row order of other planes. |
| EXO-140 | OPEN, no owner | Classification of the Semijoin / EqU32Via / Gather call sites not started. The classification must say whether each site relies on a frozen-codebook vertical read or on EXO-114's per-version order attest. |
| EXO-46 | OPEN | The counted global→focus binding membrane needs separate production admission. "Do not silently admit or forbid all indexed navigation as one category." |
| EXO-114 | OPEN | Order contract: identity / contiguity / total-order classes. Seek needs a per-sealed-version witness. Lance keeps no order. |
| EXO-105 | OPEN | mask-risc frozen. |
| #1791 | UNRESOLVED PROPOSAL | Witnessed derived population: the only route to reusing a scattered mask. Not authorized. |
| EXO-50 since 2026-10-30 | correspondence | No operator correction touches ADMISSION B, the gather rulings or the zero-copy `$i` rule (checked to 2026-11-03 11:50). |
| PR #1937 Arm L | LAB EVIDENCE | Oracle-equal on 8 cases; checks are probe-local. |

## A. Recommendation

**KEEP ARM L LAB-ONLY, REQUEST OWNER CLASSIFICATION.**

Reasons, from the records above:

1. **Admission B's text excludes Arm L as written.** B admits indirection in
   preparation only: "execution remains co-addressed". The admitted evidence pattern
   requires a fold with no `Gather`. Arm L's `Gather` runs in execution and reads
   `R_in[ib[i]]` with `ib[i] ≠ i`.
2. **The later doctrine points the other way.** CLAUDE.md (2026-10-21) makes `$i` the
   single sanctioned materialization and reads columns "in place through `$i`", one
   pointer per column. That is Arm L's shape. Neither text supersedes the other.
3. **The index's provenance is undecided, not forbidden.** `ib` comes from the row's
   own key, by address arithmetic plus a same-table witnessed seek. It is not an edge
   column like `OutgoingHop`'s `sources[i]`. Whether "another relational column"
   covers a key-derived pointer lane is exactly EXO-140, which has no owner.
4. **Arm L also fails on two enforcement points today:**
   - `Gather` turns an out-of-range pointer into a silent `false`;
   - no contract witness binds the presence planes or the pointer table to a
     version.

   Admitting it now would admit a silent wrong answer.

**Arm P is weaker, not stronger.** Its shape is the one B admitted (projection in
preparation, gather-free fold), but it copies each presence bit into a second plane.
The zero-copy law calls that a projection stored beside the first. A cast through
`$i` reproduces it, and no rung is gained. So Arm P conflicts with the later rule.
The decision is therefore not "P or L". It is how the 2026-10-21 `$i` rule and B's
"execution remains co-addressed" are reconciled, and Arm L is the shape that
reconciliation would admit.

## B. Access provenance

| access form | index producer | index consumer | read position decided by | existing classification |
|---|---|---|---|---|
| Direct co-addressed fold | none: every operand is row `i` | `MaskOp::And/Or/Ternlog`, `Pred::*` | the row itself | ADMITTED (B: "execution remains co-addressed"). v1 method C and v2 Arm P execution. |
| Prepared address lens `$i` | `SealedFacetLane::seek_from` (or `bound`) keyed by the row's own key arithmetic (`parent(key)`), witnessed, counted, in preparation | `MaskOp::Gather` via `Filter::Semijoin` / `EqU32Via`, in execution | the row's own declared address | **AMBIGUOUS.** B admits the seek in preparation and not the indirect read in execution. CLAUDE.md 2026-10-21 admits reading through `$i`. EXO-140 / EXO-46 open. |
| Value-directed Gather | a relational value lane (`sources[i]`, an fk, a scattered mask) | `MaskOp::Gather`, `ScatterOrU32` hop | another column's or table's value | FORBIDDEN (#1590; GATHER-DECISION; B "not admissible as a read address"; survival law for produced masks). |

The machine instruction is the same in rows 2 and 3. Only the producer differs, so
the classification has to key on the producer, not on the opcode.

## C. Witness completeness

| invariant | producer | witness type | verifying consumer | status today | negative control |
|---|---|---|---|---|---|
| 1. `$i` derived from the candidate's declared address | probe `parent()` + seek | none | none | **probe-only** | not built: a pointer lane filled from a value lane must be refused |
| 2. seek counted and version-witnessed | `SealedFacetLane::seek_from` | `OrderedLaneWitness` | `SealedFacetLane::validate` | **enforced** for the seek itself (lens + digest + version) | stale witness: `a_prefix_aperture_under_a_stale_witness_sweeps_and_says_why` |
| 3. frame / ClassId / codebook / scope | `OrderedLaneWitness.lens`; `relcomp::Chain` (lab); `r2il::prepare` → `guards.check(frames, now)` | `SemanticLens`, `FrameStore` | `bound` (lens check); `prepare` (no production caller) | **partial**: lens enforced, frame lab-only | v2 case 7 (probe); `LensMismatch` tests |
| 4. source planes resident, not produced `Keep` | caller | none (doc precondition) | none | **missing** (documented obligation on `Filter::Semijoin`) | not built: a produced mask passed as foreign must be refused |
| 5. no intermediate / composed relation | program shape | none | none | **probe-only** (assert no `Gather` in Arm P; Arm L has none to assert) | v2 by construction |
| 6. no mask from one program feeds another | program shape | none | none | **probe-only** | survival-law tests cover scatter only |
| 7. pointer within attested bounds | seek result | none | `Gather` **zero-fallback** | **MISSING, and silent**: out of range reads as `false` | not built: `ib = n_rows` must refuse, today it answers `false` |
| 8. physical plane order = certified order | `seal` sorts its own keys | `OrderedLaneWitness.digest` (keys only) | none for planes | **MISSING** (ISS-WITNESSED-RANGE…) | v2 case 5 unchecked: `(false,0)` vs `(true,1)` |
| 9. deletion / version invalidates pointer + witness | `LanceVersion` in the key witness | key lane only | none for presence planes or `$i` | **MISSING** for presence | v2 case 8 unchecked: stale `(true,1)` |
| 10a. unique keys in the admitted population | `attest_sorted` | none (non-decreasing passes) | `CountKeyRunsU32` checks order for its own terminal | **MISSING** as a lens precondition | v2 case 6 forged: `(true,2)` vs `(true,1)` |
| 10b. edge refines its source by one tier | mint / storage | none | none (no guard in quack, mask-risc or lance-graph-xlog) | **probe-only** | v2 case 7 (`NonRefiningEdge`) |

A key-order witness is not a presence-version witness. Rows 8 and 9 fail
independently: the key lane can be untouched while a presence plane changes (case 8).

## D. Minimal operator ruling (draft, for decision; not in force)

> **ADMISSION B — address-lens clarification.** In execution, a fold may read a
> resident plane of the same table at `$i[j]` only if every one of these holds:
>
> 1. `$i` was produced in preparation, by `SealedFacetLane` seeks keyed by
>    arithmetic on row `j`'s own key under the witnessed lens;
> 2. `$i`, the plane and the key lane carry one version, and the fold refuses on
>    any mismatch;
> 3. every `$i[j]` is checked to be `< n_rows` before the fold. An out-of-range
>    pointer is a refusal, never `Gather`'s zero-fallback;
> 4. the key lane is strictly increasing, so keys are unique;
> 5. the read plane is resident, never a mask a program produced;
> 6. the program has one terminal, and its output feeds no other program.
>
> **Remains forbidden:**
> - an index produced from any value lane (edge endpoints, foreign keys, codebook
>   values) or from another table;
> - an index computed from a mask a program produced, or carried between programs;
> - a scattered mask as an intermediate (`ScatterOrU32` survival unchanged);
> - `OutgoingHop` (quarantine unchanged);
> - any recursive or variable-depth use (EXO-73);
> - any Bag multiplicity claim (EXO-108).
>
> This admits no new opcode and no copy. Arm P's prepared presence copy is not
> admitted by this clause.

## E. Independent falsifier

Smallest admissibility probe: the v2 fixture, with Arm L's checks moved out of the
probe into the shape the ruling names.

Every one of these disables must turn the probe red. Each has been measured today as
a silent answer, or is not yet built:

| # | disable | expected after the ruling | today |
|---|---|---|---|
| 1 | permute a presence plane but not the key lane | refusal | silent false (case 5 variant; key digest does not see it) |
| 2 | delete a presence bit without a version bump | refusal | silent stale (case 8 unchecked) |
| 3 | `ib[j] = n_rows` | refusal | silent `false` (zero-fallback) |
| 4 | duplicate key | refusal at attest | silent count 2 (case 6 forged) |
| 5 | pass a `Keep` output as the foreign plane | refusal | accepted (doc-only precondition) |
| 6 | fill `ib` from an edge-endpoint value lane | refusal (provenance) | accepted |

Silence twin: the 8 v2 cases stay equal to the nested-loop oracle with every check
on. The admitted-shape parity is Arm L ≡ the direct co-addressed fold of v1
method C on fixed endpoints.

Disable 6 is the one that separates the lens from a value-directed gather. Unless it
is refused by something other than the probe's own code, the ruling cannot be
checked.

## F. Unresolved boundaries

- **EXO-140:** who classifies `Filter::Semijoin` / `EqU32Via` / `Gather` call sites,
  and whether "another relational column" covers a key-derived pointer lane.
  Owner: none named.
- **EXO-46:** whether the counted global→focus pointer table may be used in
  production execution, and under which witness. It must not be decided as "all
  indexed navigation".
- **EXO-114:** per-version order attest. Seek needs total order. Uniqueness is a
  separate property no class currently states.
- **EXO-105:** mask-risc frozen. Rule 3 above (refuse out-of-range rather than
  zero-fall back) must be met outside mask-risc, by a pre-fold bounds check,
  unless the operator unfreezes it.
- **DT64 behaviour admission:** `r2il::prepare` refuses guarded bindings
  (`GuardNotExecutable`, `binding.rs:290-291`), and `xlog::bag_count_admitted`
  admits only `[GroupCount, GroupCount, MulI64]` (`lib.rs:399`). The smallest
  connection is an admitted DO word shape `presence-conjunction → Any` whose
  preparation step is this lens. Its falsifier: the same binding refused under a
  stale frame (`guards.check`) and under each disable above. ON stays a change
  trigger, never the existence test.
- **Out of scope by construction:**
  - arbitrary graph edges (no key arithmetic for the parent);
  - variable depth and recursion (EXO-73);
  - multiple paths to one endpoint and Bag multiplicity (EXO-108);
  - SCCs.

## Answer to the key question

Yes, in principle. A certified address lens names where each relation's presence
witness already lives, so the fold proves the relationship without it ever being
stored, and no presence bit is copied. Only the pointer table is materialized.

Today's source does not certify the lens. It certifies the key order of one lane.
Of the eleven rows in table C, five are missing, four are enforced only by the probe,
one is partial, and only the seek itself is enforced. The
interpretive question is unowned. Arm L stays lab-only until the operator rules on
D and an owner takes EXO-140.
