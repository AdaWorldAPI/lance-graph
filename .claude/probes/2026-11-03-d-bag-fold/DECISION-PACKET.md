# D-BAG-FOLD: operator decision packet (EXO-140 / EXO-46)

**Status: DRAFT / PROPOSED / NOT IN FORCE.** This is decision preparation. It grants no
admission and authorizes no implementation. No code, probe, Linear status or
execution path is changed.

**Evidence:**
- `main` @ `119787fc4`;
- #1937 (virtual composition, Arm P vs Arm L);
- #1938 (v3, production witness gaps, 26/26);
- #1939 (v4, construction-time attestation and candidate-gated execution, 21/21);
- this folder: `ARM-L-DECISION-MEMO.md`, `REPORT-V3.md`, `REPORT-V4.md`.

EXO-50 was read to 2026-11-03 11:50. No operator message supersedes ADMISSION B, the
#1590 gather ban or the 2026-10-21 `$i` doctrine.

## Source facts (read, not inferred)

1. **ADMISSION B** (`.claude/board/entries/2026-10-20-six-decisions.md` §1):
   - preparation may do "counted, version-witnessed, same-table key-range seeks over
     an attested sorted key" and "construct a projection for subsequent co-addressed
     evaluation";
   - "a position derived from another relational column or table is not admissible as
     a read address";
   - "Execution remains co-addressed";
   - `Semijoin` / `EqU32Via` / `Gather` "need call-site classification".
2. **ISS-ADMISSION-PREP-BINDING** (ruled): the admitted evidence shape is a fold program
   with no `Gather`.
3. **`CLAUDE.md`, 2026-10-21:** the only materialization the inversion makes is the
   pointer table `$i`; everything else is "read in place through `$i`", one pointer per
   column.
4. **`OrderedLaneWitness::forged(version, n_rows, digest)` is `pub`**
   (`#[doc(hidden)]`, documented "falsifier-only", `crates/lance-graph-contract/src/ordered_lane.rs`).
5. **`lance_graph_dt64::dr::Frame` has no public constructor**
   (`crates/lance-graph-dt64/src/dr.rs`).
   - It is issued only by `FrameStore::seal`, and its identity is the issuing store
     (`Weak` pointers; `same_issue`, `issued`).
   - `Frame::revisions()` and `FrameStore::revisions()` expose the plain-data
     `FrameRevisions { source, relation_map, coordinates }`.
   - `FrameStore::check` accepts only the current seal and names only the first changed
     revision (source → map → coordinates).
6. **mask-risc** `MaskOp::Gather { under: Option<Operand> }`: an out-of-range index is the
   zero-fallback. **Quack** `Node::Gather { under: Option<Mask> }`: `gate_walk` gates an
   `And`'s conjuncts by one resident plane only.
7. **v4 lab evidence:**
   - construction-time presence witness and pointer lens;
   - a forged pointer table whose digest is re-sealed passes digest verification (A7c);
   - presence verification costs more than gated execution;
   - register-gated execution scales with the candidate count.

## 1. Authority is not integrity

**Integrity** answers "is this the value that was recorded?" A digest or an
`OrderedLaneWitness` answers it.

**Authority** answers "was this produced by the party allowed to produce it, by the
allowed method?" Neither a digest nor an `OrderedLaneWitness` can answer it:

- `OrderedLaneWitness::forged` is public. Any crate can build a witness with any
  version, row count and digest.
- A digest can be recomputed by anyone who holds the bytes (v4 A7c).

So an ordered-lane witness, or any digest, **must not be treated as an authorization
token.** It is an integrity check that a party already holding authority uses.

**What the certified lens needs instead:**

- **Issuer-controlled construction:** no public constructor. Only the key-lane owner can
  mint one, the way only `FrameStore::seal` can issue a `Frame`.
- **Immutable provenance:** the lens records the issue it was minted under and cannot
  be changed afterwards.

`Frame` is the existing model for that pattern. `OrderedLaneWitness` is not.

**Current dependents of `forged`.** Classified here; nothing is changed in this PR.

| site | kind | needs `pub`? |
|---|---|---|
| `crates/lance-graph-contract/src/ordered_lane.rs` lines 618, 624, 630, 808 | the contract's own `#[cfg(test)]` module | no: same crate |
| `crates/lance-graph-quack/src/lib.rs` lines 4589, 4705 | `#[cfg(test)] mod diamond_lowering_tests` | yes: cross-crate test |
| `crates/lance-graph-nal/tests/w2.rs` line 235 (`a_stale_witness_refuses_to_bind`) | integration test | yes: cross-crate test |

No production caller exists. The two cross-crate test paths are the only compatibility
dependence. If the contract owner later restricts `forged`, those tests need another
route to a stale witness, for example a feature-gated test constructor or a real
reseal. That is the owner's call, not part of this PR.

## 2. Pointer and presence lifetimes

| evidence | depends on | invalidated by | should not be invalidated by |
|---|---|---|---|
| pointer lens `$i` | coordinate / key layout | a `coordinates` revision advance; a key reseal | a source-only (presence) mutation |
| presence attestation | the current relation facts | a `source` (or `relation_map`) revision advance | — |

A source-only mutation therefore retires presence evidence without reminting `$i`.

**Today's admission is coarser, and that is the binding behaviour.**
`FrameStore::check` refuses the whole frame on any change, which retires both. v4 A4′
showed exactly that.

**The narrower per-axis route is a proposal, not an existing admission:**

- **Revision counters are not authority.** `FrameRevisions` is plain data, public
  fields and `Copy`, so anyone can build one. A consumer comparing
  `Frame::revisions()` against numbers it holds would be trusting values it cannot
  authenticate.
- **The issuer must decide per axis.** A sound per-axis check is an *issuer-side*
  operation. Given a `Frame` that `same_issue` / `issued` show to be its own, the
  `FrameStore` answers "is the coordinates axis of this frame still current?"
  separately from "is the source axis current?". That needs a new issuer method, or a
  lens and attestation each holding their own issued token per axis.
- **Owner:** DT64 owner (`FrameStore`), with the contract owner for the lens types. It
  is not built and not admitted.
- **Unknown:** whether such a per-axis answer stays sound across a reseal at equal
  revisions (`FrameRefusal::FrameResealed`) is unchecked.

**Blocked part.** No existing owner model forces a presence writer to advance `source`.
The planes are separate memory, with no write path through the frame store. Until the
plane / mailbox owner binds every presence write to the advance, a silent writer is
caught only after the fact, by the key-bound presence witness (v4 A5). It is not
refused at the write. **BLOCKED pending that owner.**

**Cost classes, kept separate:**

| class | cost |
|---|---|
| key-version mint | cold, once per coordinates revision |
| presence attestation | per source revision; v4 crossed 1 ms at ~14k rows with a full re-sort |
| per-query validation | O(1): issue identity plus revision checks |
| hot fold | µs, scales with candidates (v4 B3) |

Cheap per-query validation is sound only if a mutation cannot bypass the advance.
Until that binding exists, cached verification is a correctness liability. Whether
mint and attestation costs fall under the no-ms rule's cold-cost exception is
**unclassified**; that is an operator decision and is not granted here. Copying
presence planes remains excluded.

## 3. Recommended rulings — PROPOSED / NOT IN FORCE

> **PROPOSED / NOT IN FORCE. Enacts nothing.**
>
> **EXO-140 (recommended).** A same-table, coordinate-derived, issuer-certified `$i`
> should be **eligible for classification as co-addressed execution**, although it
> lowers to `MaskOp::Gather`, provided all of these hold:
>
> 1. **Resident source planes:** only planes of the same table, never a mask any
>    program produced.
> 2. **Verified provenance:** minted by the key-lane owner, through witnessed
>    `SealedFacetLane` seeks keyed by arithmetic on each row's own key. Authority
>    comes from issuer-controlled construction (§1), never from a digest.
> 3. **Correct frame and scope:** an issue of the right store; ClassId and prefix scope
>    checked at mint.
> 4. **Bounds:** every pointer checked at mint. An out-of-range pointer is a refusal,
>    never `Gather`'s zero-fallback.
> 5. **Generation:** valid only while its coordinates revision is current; the presence
>    it reads is attested for the current source revision (§2).
> 6. **In-program gating:** the gather runs gated inside one program, and its result
>    never leaves the program as a mask.
>
> **Remain prohibited:**
> - value-derived `Gather` (an index from any value lane or another table);
> - reuse of produced masks;
> - copied presence projections (Arm P);
> - scattered intermediates;
> - `OutgoingHop`;
> - recursion (EXO-73);
> - Bag multiplicity (EXO-108).
>
> **EXO-46 (recommended).** The `$i` pointer table is admissible in production
> execution only as such a certified lens:
>
> - **Mint:** the key-lane owner only.
> - **Mutate:** nobody; the lens is immutable.
> - **Invalidate:** the issuer, on a coordinates revision.
> - **Consume:** any fold that presents it, after the issuer-side validity check.

**Reconciliation required before this can bind.** ADMISSION B says "execution remains
co-addressed", and its admitted pattern contains no `Gather`. The interpretation above
becomes binding only once **the operator explicitly reconciles that wording**. If the
operator reads "co-addressed" as "same row index", ADMISSION B excludes the
recommendation as written. The only zero-copy route left would then be a layout with
ancestor presence already at the candidate's row, which needs a derived population
(#1791, not authorized).

**Strongest counterargument.** At execution time a certified read and a value-directed
gather are the same instruction. Their difference lives entirely in the mint, and
today no contract type records or enforces that provenance:

- `OrderedLaneWitness` is forgeable;
- the certified lens type does not exist.

Classifying the instruction before that type exists would admit it on trust. That is
why the ruling is phrased as *eligibility*, conditional on the lens type.

## 4. Ownership and implementation gates

Every implementation gate stays closed until its owner and the EXO-140 / EXO-46
interpretation have acted.

| owner | decides / owns | gate |
|---|---|---|
| **EXO-140 / EXO-46 operator** | the admission interpretation and its scope (§3); reconciliation of ADMISSION B; who owns call-site classification; whether mint and attestation costs fall under the cold-cost exception | closed: no ruling yet |
| **Contract owner** | the certified lens type (issuer-controlled, immutable); the presence attestation type; key uniqueness in sealing; lens identity (a second `SemanticLens`, or out of scope); the future of `forged` | closed until the operator rules |
| **DT64 owner** (`FrameStore`) | an issuer-side per-axis validity check; presence-writer binding to `source`, with the plane / mailbox owner | closed; per-axis route is only a proposal |
| **Quack owner** | candidate-gated lowering: gate each `Semijoin` conjunct of an `And` by the accumulated in-program register | closed until the lens type exists |
| **DT64 owner, later** | method-binding admission for a three-hop existence method (`GuardNotExecutable`, restricted DO shape) | separate gate; not touched |

Quack lowering detail, for the owner:

- sound for `And` (`x ∧ y = x ∧ (y under x)`);
- not sound inside `Or`;
- inside `Not`, the gate passes through but is never dropped;
- the register must come from conjuncts already lowered;
- mask-risc unchanged (`under: Some(Operand::Scratch)` already accepted).

Tests the Quack owner would need:

- existing lowering suites unchanged;
- a randomized `And` / `Or` / `Not` differential against a row oracle;
- v4 B1 parity and both B2 disables through Quack;
- a program-shape assertion;
- a no-ms timing comparison against v4 B3.

## 5. Admission falsifiers

| control | must be refused at | today | evidence |
|---|---|---|---|
| plane misaligned at construction | presence mint / first verify | lab only | v4 A2 |
| plane permuted after construction | presence verify (per generation) | lab only | v4 A3 |
| stale generation via the API | issuer check | production (whole frame) | v4 A4 |
| presence write without advance | the write (owner) | not enforceable | v4 A5 |
| duplicate key | seal / mint | lab only | v3 4a, v4 A6 |
| forged pointer table | mint authority | not enforceable (digest re-seal passes) | v4 A7c |
| pointer out of bounds | mint | lab only | v4 A7a/b |
| foreign ClassId pointer | mint | lab only | v4 A7d |
| wrong frame | issuer check | production | v4 A8a |
| stale key version | issuer check | production | v4 A8b |
| wrong lens | mint | not constructible | v4 A8c |
| value-derived index | mint | lab only | v3 7a, v4 A7c′ |
| produced mask as source plane | lowering / execution | not enforced | — |

Silence twins: v4 B1 parity and the B2 disables must stay green under every check.

## 6. Remaining decisions

**Operator, decidable now:**

1. EXO-140: is a certified `$i` read eligible for classification as co-addressed, and
   how is ADMISSION B's wording reconciled?
2. EXO-46: does lens authority come from issuer-controlled construction by the key-lane
   owner?
3. Who owns the EXO-140 call-site classification.

**Owners, after 1 and 2:** the contract, DT64 and Quack items in §4.

**Lacking evidence (do not decide yet):**

- cold-cost classification of minting and attestation;
- whether an order-independent presence digest is acceptable (collision and
  provenance not analysed);
- soundness of an issuer-side per-axis check across `FrameResealed`.

## Next engineering step: OWNER DECISION

- **No implementation is authorized or ready.** The experimental evidence is complete.
  The remaining step is the operator's explicit ruling on §6 items 1–3.
- **If the operator rules yes:** the first engineering step would be for the contract
  owner to *specify* the certified-lens and presence-attestation types against §5,
  still without changing execution.
- **Until those types exist:** Quack lowering and any production `Gather` through `$i`
  stay **NO-GO**.

This packet closes the D-BAG-FOLD investigation.
