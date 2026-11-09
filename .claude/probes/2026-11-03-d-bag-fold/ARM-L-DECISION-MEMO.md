# D-BAG-FOLD: decision memo, the zero-copy pointer lens (Arm L)

**Status:** DECISION PREPARATION. Documentation only. The ruling in §D is
**DRAFT / NOT IN FORCE**. Nothing here admits Arm L, a `MaskOp::Gather` call site, or
any implementation.

Evidence: this folder (`README.md`, `REPORT.md`, the two probes). Source:
lance-graph `514948203`. Probe commit: `ae885f06a`.

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
| This folder, Arm L | LAB EVIDENCE | Oracle-equal on 8 cases; every check is probe-local. |

## A. Recommendation

**KEEP ARM L LAB-ONLY, REQUEST OWNER CLASSIFICATION.**

1. **ADMISSION B's text excludes Arm L as written.** B admits indirection in
   preparation only: "execution remains co-addressed". The admitted evidence pattern
   requires a fold with no `Gather`. Arm L's `Gather` runs in execution and reads
   `R_in[ib[i]]` with `ib[i] ≠ i`.
2. **The later doctrine points the other way.** CLAUDE.md (2026-10-21) makes `$i` the
   single sanctioned materialization and reads columns "in place through `$i`", one
   pointer per column. That is Arm L's shape.
3. **The two have not been reconciled.** No record states whether reading a resident
   plane through `$i` counts as co-addressed execution, and neither text supersedes
   the other. This memo does not reconcile them. That is the operator question at the
   end.
4. **The index's provenance is undecided, not forbidden.** `ib` comes from the row's
   own key, by address arithmetic plus a same-table witnessed seek. It is not an edge
   column like `OutgoingHop`'s `sources[i]`. Whether "another relational column"
   covers a key-derived pointer lane is exactly EXO-140, which has no owner.
5. **Structural derivation is necessary, not sufficient.** That an index is derived
   from structure does not admit the `MaskOp::Gather` that consumes it. Every call
   site still needs EXO-140's classification and the witnesses in §C.
6. **Today's source would admit silent wrong answers.**
   - `Gather` turns an out-of-range pointer into a silent `false`.
   - No contract witness binds the presence planes or the pointer table to a version.

**Arm P is a negative architectural comparison, not a fallback.** Its shape is the one
B admitted (projection in preparation, gather-free fold), but it copies each presence
bit into a second plane. The zero-copy law treats that as a projection stored beside
the first: a cast through `$i` reproduces it, and no rung is gained. Arm P is kept in
the probe only to show that the conjunction holds without `Gather`. It is not a route
to implement if Arm L is refused.

## B. Access provenance

| access form | index producer | index consumer | read position decided by | existing classification |
|---|---|---|---|---|
| Direct co-addressed fold | none: every operand is row `i` | `MaskOp::And/Or/Ternlog`, `Pred::*` | the row itself | ADMITTED (B: "execution remains co-addressed"). v1 method C and v2 Arm P execution. |
| Prepared address lens `$i` | `SealedFacetLane::seek_from` (or `bound`) keyed by the row's own key arithmetic (`parent(key)`), witnessed, counted, in preparation | `MaskOp::Gather` via `Filter::Semijoin` / `EqU32Via`, in execution | the row's own declared address | **AMBIGUOUS, NOT ADMITTED.** B admits the seek in preparation and says nothing to admit the indirect read in execution. CLAUDE.md 2026-10-21 describes reading through `$i`. EXO-140 / EXO-46 open. |
| Value-directed Gather | a relational value lane (`sources[i]`, an fk, a scattered mask) | `MaskOp::Gather`, `ScatterOrU32` hop | another column's or table's value | FORBIDDEN (#1590; GATHER-DECISION; B "not admissible as a read address"; survival law for produced masks). |

The machine instruction is the same in rows 2 and 3. Only the producer differs, so
a classification must key on the producer, not on the opcode. Row 2 is not thereby
admitted: a lawful producer is one of several required conditions (§C), and none of
them is enforced in source today except the seek itself.

## C. Witness completeness

Five properties are easily conflated, and each needs its own witness:

- **provenance:** where `$i` came from;
- **bounds:** whether `$i[j]` lies inside the plane;
- **ordered-lane identity:** whether the planes are in the order the key witness
  certifies;
- **key uniqueness:** whether one address names exactly one row;
- **presence-version invalidation:** whether a presence change retires `$i` and its
  witness.

A key-order witness answers only ordered-lane identity, and only for the key lane.
A key lane can stay unchanged while a presence plane changes (v2 case 8).

| invariant | producer | witness type | verifying consumer | status today | negative control |
|---|---|---|---|---|---|
| 1. **Provenance:** `$i` derived from the candidate's declared address, never from a value lane | probe `parent()` + seek | none | none | **probe-only** | disable 6 (not built) |
| 2. Seek counted and version-witnessed | `SealedFacetLane::seek_from` | `OrderedLaneWitness` | `SealedFacetLane::validate` | **enforced** for the seek itself (lens, digest, version) | stale witness: `a_prefix_aperture_under_a_stale_witness_sweeps_and_says_why` |
| 3a. **Frame** consistency: `$i`, planes and key lane read in one frame / focus | `FrameStore`; `relcomp::Chain` (lab); `r2il::prepare` → `guards.check(frames, now)` | frame issue / focus | `prepare` (no production caller) | **lab-only** | not built for the lens |
| 3b. **ClassId / codebook** consistency: one classid, one codebook, one lens | `OrderedLaneWitness.lens` | `SemanticLens` | `bound` / `bound_aperture` (`LensMismatch`) | **partial**: lens enforced on the seek; nothing ties the presence planes to that lens | v2 case 7 (probe); `LensMismatch` tests |
| 3c. **Scope** consistency: every pointer target inside the declared scope | `relcomp::Chain` scope check (lab) | scope | none in quack / mask-risc | **probe-only / lab-only** | v2 case 7 cross-class edge refused (probe) |
| 4. Source planes resident, not produced `Keep` | caller | none (doc precondition) | none | **missing** (documented obligation on `Filter::Semijoin`) | disable 5 |
| 5. No intermediate or composed relation | program shape | none | none | **probe-only** | v2 by construction |
| 6. No mask from one program feeds another | program shape | none | none | **probe-only** | survival-law tests cover scatter only |
| 7. **Bounds:** every pointer `< n_rows` | seek result | none | `Gather` **zero-fallback** | **MISSING, and silent**: out of range reads as `false` | disable 3 |
| 8. **Ordered-lane identity:** physical plane order = certified order | `seal` sorts its own keys | `OrderedLaneWitness.digest` (keys only) | none for planes | **MISSING** (ISS-WITNESSED-RANGE…) | disable 1; v2 case 5 |
| 9. **Presence-version invalidation:** a deletion or version change retires `$i` and its witness | `LanceVersion` in the key witness | key lane only | none for presence planes or `$i` | **MISSING** for presence | disable 2; v2 case 8 |
| 10a. **Key uniqueness** in the admitted population | `attest_sorted` | none (non-decreasing passes) | `CountKeyRunsU32` checks order for its own terminal only | **MISSING** as a lens precondition | disable 4; v2 case 6 |
| 10b. Each edge refines its source by exactly one tier | mint / storage | none | none (no guard in quack, mask-risc or lance-graph-xlog) | **probe-only** | v2 case 7 (`NonRefiningEdge`) |

Totals: one enforced (2), one partial (3b), four missing (4, 7, 8, 9), one missing
as a precondition (10a), six probe-only or lab-only (1, 3a, 3c, 5, 6, 10b).

## D. Minimal operator ruling — **DRAFT / NOT IN FORCE**

> **DRAFT / NOT IN FORCE — for operator decision only. Grants no admission and
> authorizes no implementation.**
>
> **ADMISSION B — address-lens clarification (proposed).** In execution, a fold may
> read a resident plane of the same table at `$i[j]` only if every one of these holds
> and is checked by the contract, not by the caller:
>
> 1. **Provenance:** `$i` was produced in preparation by `SealedFacetLane` seeks keyed
>    by arithmetic on row `j`'s own key, never by any value lane.
> 2. **Frame, ClassId, codebook and scope:** `$i`, every plane it reads and the key
>    lane belong to one frame, one classid, one codebook (lens) and one declared
>    scope. Any mismatch is a refusal.
> 3. **Presence version:** `$i`, every plane and the key lane carry one version, and
>    a presence change retires `$i`. The fold refuses on any mismatch.
> 4. **Ordered-lane identity:** every plane is attested to be in the key witness's
>    order, not only the key lane.
> 5. **Bounds:** every `$i[j]` is checked `< n_rows` before the fold. An out-of-range
>    pointer is a refusal, never `Gather`'s zero-fallback.
> 6. **Uniqueness:** the key lane is strictly increasing.
> 7. **Residency:** the read plane is resident, never a mask a program produced.
> 8. **Single program:** one terminal, whose output feeds no other program.
>
> **Remains forbidden:**
> - every `MaskOp::Gather` outside this clause. A structurally derived index is not
>   enough on its own;
> - an index produced from any value lane (edge endpoints, foreign keys, codebook
>   values) or from another table;
> - an index computed from a mask a program produced, or carried between programs;
> - a scattered mask as an intermediate (`ScatterOrU32` survival unchanged);
> - `OutgoingHop` (quarantine unchanged);
> - any recursive or variable-depth use (EXO-73);
> - any Bag multiplicity claim (EXO-108).
>
> Adds no opcode and no copy. Does not admit Arm P's prepared presence copy.

## E. Independent falsifier

### The eight-case oracle results (measured, `v2_witnessed_existential`, 0 mismatches)

Both arms agree with each other and with a nested-loop oracle in every checked case.
Prefix-only is shown for contrast.

| case | Arm P / Arm L | oracle | prefix-only |
|---|---|---|---|
| 1 all three edges present | (true, 1) | (true, 1) | true |
| 2a R deleted | (false, 0) | (false, 0) | **true** |
| 2b S deleted | (false, 0) | (false, 0) | **true** |
| 2c T deleted | (false, 0) | (false, 0) | false |
| 3 three candidates, one valid | (true, 1) | (true, 1) | — |
| 3b three candidates, two valid | (true, 2) | (true, 2) | — |
| 4 three candidates, none valid | (false, 0) | (false, 0) | **true** |
| 5 permuted rows | refused (`OrderNotAttested`); unchecked arm (false, 0) | (true, 1) | — |
| 6 duplicate terminal key | refused at seal (`DuplicateKey`); forged witness (true, 2) | (true, 1) | — |
| 7 another class holds the only valid chain | (false, 0); cross-class edge refused (`NonRefiningEdge`) | (false, 0) | — |
| 8 S deleted after preparation | stale refused (`StaleVersion`) in both arms; unchecked stale (true, 1); re-prepared (false, 0) | (false, 0) | — |

Measured disable runs: removing arm L's `Semijoin` on `S_in`, or the one on `R_in`,
trips the arm-agreement assertion (red).

### The six disable runs the ruling must turn red

The smallest admissibility probe is the v2 fixture, with Arm L's checks moved from the
probe into whatever enforces §D.

| # | disable | required after a ruling | today |
|---|---|---|---|
| 1 | permute a presence plane but not the key lane | refusal | silent false (a key digest cannot see it) |
| 2 | delete a presence bit without a version bump | refusal | silent stale (case 8 unchecked) |
| 3 | `ib[j] = n_rows` | refusal | silent `false` (zero-fallback) |
| 4 | duplicate key | refusal at attest | silent count 2 (case 6 forged) |
| 5 | pass a `Keep` output as the foreign plane | refusal | accepted (doc-only precondition) |
| 6 | fill `ib` from an edge-endpoint value lane | refusal (provenance) | accepted |

Silence twin: the eight cases above stay equal to the oracle with every check on.
Admitted-shape parity: Arm L ≡ the direct co-addressed fold (v1 method C) on fixed
endpoints.

Disable 6 is the one that separates a lens from a value-directed gather. Unless it is
refused by something other than the probe's own code, the ruling cannot be checked.

## F. Unresolved boundaries

- **EXO-140:** who classifies the `Filter::Semijoin` / `EqU32Via` / `Gather` call
  sites, and whether "another relational column" covers a key-derived pointer lane.
  Owner: none named.
- **EXO-46:** whether the counted global→focus pointer table may be used in
  production execution, and under which witness. Not to be decided as "all indexed
  navigation".
- **EXO-114:** per-version order attest. Seek needs total order. Uniqueness is a
  separate property no class currently states.
- **EXO-105:** mask-risc frozen. §D condition 5 (refuse out-of-range rather than zero
  fall-back) must be met outside mask-risc, by a pre-fold bounds check, unless the
  operator unfreezes it.
- **DT64 behaviour admission — independently gated.** Even if Arm L were admitted, a
  DT64 method could not yet use it:
  - `r2il::prepare` refuses guarded bindings (`GuardNotExecutable`,
    `binding.rs:290-291`);
  - `xlog::bag_count_admitted` admits only `[GroupCount, GroupCount, MulI64]`
    (`lib.rs:399`).

  The smallest missing connection is an admitted DO word shape
  `presence-conjunction → Any` whose preparation is this lens. Its falsifier is the
  same binding refused under a stale frame (`guards.check`) and under each disable
  above. ON stays a change trigger, never the existence test. This route needs its own
  operator decision; a ruling on §D does not grant it.
- **Out of scope by construction:**
  - arbitrary graph edges (no key arithmetic for the parent);
  - variable depth and recursion (EXO-73);
  - multiple paths to one endpoint, and Bag multiplicity (EXO-108);
  - SCCs.

## Answer to the key question

In principle, yes. A certified address lens names where each relation's presence
witness already lives, so the fold can prove the relationship without the relationship
ever being stored. No presence bit is copied; only the pointer table is materialized.

Today's source does not certify such a lens. It certifies the key order of one lane,
the seek over it, and the lens of that seek. The interpretive question of whether
reading through `$i` is co-addressed execution has no owner. Arm L stays lab-only.

## Operator question (OPEN)

**Does a preparation-time, witnessed, same-table address mapping `$i` qualify as
co-addressed execution when the fold subsequently reads resident presence planes
through `$i`, despite lowering to `MaskOp::Gather`?**

- **If yes:** which exact witnesses are required for provenance, frame (with
  ClassId, codebook and scope), ordered-lane identity, presence version, key
  uniqueness and bounds? §D is one draft answer.
- **If no:** which existing law excludes it (ADMISSION B's "execution remains
  co-addressed", the #1590 ban, or another), and what lawful zero-copy route remains
  for multi-hop existence? Arm P's copied projection is not such a route under the
  zero-copy law.
