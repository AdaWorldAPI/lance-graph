# D-NFR — native representation contract: Quack → DR64³ → Palette256³ → execution

**Status:** D-NFR-0 MEASURED-IN-SOURCE (this document, docs only). D-NFR-1.. PROPOSED, none authorized. Read at `origin/main` `f0b046b9f`.
**Scope:** which binding seams of the intended native representation are shipped, lab-only, proposed or missing. It does not implement any of them, change an ABI or an execution contract, or decide the #1744 / #1745 / #1750 S0 ownership dispute.
**Companion:** `.claude/plans/d-native-first-dependency-correction-v1.md` (the canonical dependency order and the PR corrections). This contract comes first; the correction cites it.

## 0. The two invariants

1. **ONNX is an external mathematical compatibility specification.** The native starting point is the existing R2IL polyfill (`lance-graph-r2il::procs` + `FoldDialect`, executed by mask-risc over `ndarray::simd`). No ONNX graph, Tract runtime or Optimum dependency is in the internal execution path.
2. **Quack binding + DR64³ + Palette256³ is the intended native representation.** DR64³ owns the proposed local relational / focus geometry. Palette256³ owns the proposed associated numerical representation. Existing R2IL, mask-risc, BGZ, Helix and ndarray primitives are reused. This document separates what is implemented from what is only an integration contract.

The kernel inventory does not define this architecture. A shipped kernel is evidence about EXECUTION only; it does not establish that a seam exists, or what its semantics are.

## 1. Status vocabulary

| status | meaning |
|---|---|
| SHIPPED | in a production crate on `main`, with tests |
| LAB | in a lab/probe crate on `main` (stand-in, not the owner); no production caller |
| PROPOSED | named by a plan; no type exists |
| MISSING | required by the target architecture; no type and no plan owns it |

Coverage of an OPERATION is reported separately on four axes: **semantic** (its mathematical contract is written down), **executable** (an existing composition computes it), **approximate** (it is computed with a stated error, not exactly), **certified** (an applicable transformation has independently checkable evidence). An operation can be executable without being semantically specified, and specified without being executable.

## 2. Seam table

Every row cites the defining source. "Read" means the body was read for this document, not only searched.

| # | seam | status | source (read) | what it does / does not do |
|---|---|---|---|---|
| S1 | names and literals → numeric identity | SHIPPED | `lance-graph-quack/src/bind.rs`: `Binder`, `Draft::bind`, `BindError` | Resolves table, field and textual literal once through a consumer-owned `Binder` (catalog + codebook). Refusals: `UnknownTable`, `UnknownField`, `UnknownValue` (nothing minted), `KindMismatch`, `OutOfRange`, `WrongTable`. Three-valued NULL handling through `BoundField.validity`. |
| S2 | bound query reused without re-resolution | SHIPPED | `bind.rs` module doc; `lance-graph-mask-risc` `CompiledProgram` | A `Query` "can be lowered and executed any number of times without touching a name, a label or a codebook again". The lowered program can also be kept (`CompiledProgram`, recognition once). |
| S3 | ordered-lane version witness | SHIPPED | `lance-graph-quack/src/lib.rs` (`lane_version`, `OrderedLaneWitness`), contract `ordered_lane` `WitnessError::VersionMismatch` | A range bound on an ordered lane carries the lane's version and order-sensitive digest; a forged or stale witness makes the leaf sweep instead and says why (`a_prefix_aperture_under_a_stale_witness_sweeps_and_says_why`). Covers one lane's order, not a whole binding. |
| S4 | table catalog / physical registration | PROPOSED | `bind.rs`: `Registrar`, `TableDeclaration::register` | "no crate owns a table catalog or storage allocation yet, so `Registrar` has no production implementation". |
| S5 | **binding store**: persisted bindings, frame validity, invalidation | MISSING in Quack | `bind.rs`, `lib.rs` (searched closed: no snapshot / frame / epoch type) | Quack is a binding *membrane*, not a store. The consumer's `Binder` owns ID meaning and "what survives a rename or a re-observation". No Quack type records under which frame a `Query` was bound. |
| S6 | frame: source × relation map × coordinates revisions, seal identity, stale refusal | LAB | `dr-math-binding-probe/src/lib.rs`: `FrameRevisions`, `FrameStore`, `Frame` | "Lab stand-in for the owner of a frame." Frames compare by issue (`Weak::ptr_eq`), never by value; a stale frame names the first revision that moved (`SourceChanged`, `MapChanged`, `CoordinatesChanged`, `FrameResealed`); a foreign store is `ForeignFrame`. No production owner. |
| S7 | focus and scope as separate guards | LAB | same file: `Focus { revision }`, `Scope { id }`, `Guards`, `bind_dr64` | Checked by equality at borrow time (`FocusChanged`, `ScopeChanged`). "Checks seal identity and equality of focus and scope, not their authenticity; a production binder also needs the ClassView registry, the tenant ACL and the source owner." |
| S8 | typed axes, truth and population per method | LAB | same file: `MethodSignature { owner_class, method_ordinal, row: Axis, column: Axis, truth, population }`, `AxisKind::{Identity, Property, Community, Capability}` | `bind_dr64` refuses `MethodChanged`, `AxisChanged`, `SemanticsChanged`, `ThreeValuedNeedsValidity`, `TwoValuedHasNoValidity`, `BagNeedsMultiplicity` (Bag is refused, not supported). |
| S9 | the DR register | LAB | `dr-math-binding-probe/src/relcomp.rs`: `Rows = [u64; 64]`, `compose`, `Chain`, `RelExpr` | One borrowed 64 × 64 Boolean relation register (512 B): two axes (row × column). Composition is OR/AND over set bits. **This is DR64², not DR64³**: no type carries a third axis. |
| S10 | DR64³ third axis (identity × identity × property/relation) | PROPOSED | (no type; closed search over the probe and `dir-rmap-probe`) | The cube is a target, not a type. A stack of S9 registers indexed by a relation/property ordinal is the obvious reading; nothing on `main` fixes it. |
| S11 | global → local focus | LAB | `dir-rmap-probe/src/focus.rs`: `Focus::slots`, `Focus::project`; `prep.rs`: `Witness` | Binds up to 256 group ordinals to local slots for one chain and projects the full B+ closure onto them (4 words per group). Refuses `OutOfFocus`, `Stale`, `TooLarge`, `Duplicate`; completeness through unfocused groups is tested. `Witness` is tied to dir-sim's `VersionStore`, so `dr-math-binding-probe` does not reuse it. A DR256 local domain over one domain (IAM membership), not a general one. |
| S12 | relational certification | LAB | `relcomp.rs`: `admit_composition_assoc`, `AdmittedLaw`, `prove_reassociation`, `EggProof`; `recipe.rs`: `RecipeCertificate`, `admit_recipe` | Associativity of Boolean composition admitted per law shape and proved by egglog, kept as egglog's own `ProofStore` + `ProofId`. Applies to Boolean / Distinct relations only. |
| S13 | Palette256³ codebooks | SHIPPED | `bgz17/src/palette.rs`: `Palette { entries: Vec<Base17> }`, `PaletteEdge { s_idx, p_idx, o_idx }`; `scope.rs`: `codebook_s/p/o`; `palette_semiring.rs`, `distance_matrix.rs` | ≤256 Base17 (`i16[17]`) archetypes per plane; an edge is three u8 indices, one per S/P/O plane. **The third axis already has a meaning: S, P, O.** Distance matrix, compose and identity via `PaletteSemiring`. |
| S14 | palette quality contract | SHIPPED (stated, per crate) | `bgz17/src/lib.rs` layer table; `bgz-tensor/src/palette.rs` CLAM `radii`; `codebook_calibrated.rs` | bgz17 states measured rank correlation per layer (ρ 0.937 scent, 0.965–0.992 palette, 1.000 exact planes). bgz-tensor keeps a CLAM worst-case radius per entry and a γ+φ-calibrated u8 table. A correlation is a quality measurement, not an error bound; only the CLAM radius is a bound. |
| S15 | codebook runtime identity (family, generation, digest) | MISSING | `Palette`, `CalibratedCodebook` (no such field); `bgz-tensor/src/manifest.rs` `ModelEntry.sha256` | The only digest is per hydrated shard file. Nothing binds a palette index to the codebook generation that issued it, so a stale or foreign codebook cannot be refused at use. |
| S16 | DR coordinate → palette-valued property | MISSING | (no type) | Nothing says which DR axis carries which palette plane, under which frame. ADMISSION B stands: a DR ordinal is never a palette index and a palette index is never a positional gather into another table. |
| S17 | native execution | SHIPPED | `lance-graph-r2il::procs` (~30 procedures), `FoldDialect`, `lance-graph-mask-risc` terminals, `ndarray::simd` | Selection, Boolean combination and integer folds (count, any, all, sum, min, max, group sum, blend, keep). The DR compose kernel in S9 is NOT on mask-risc; it is the probe's own loop. Floats and approximations run in existing ndarray / bgz kernels, outside the fold algebra. |
| S17a | R2IL route per ClassId (DIRECT / COMPOSE / LOWER / REFUSE), resolved once and held in the binding | LAB | `math-polyfill-probe/examples/classid_route.rs` (D-MPF-4) | No production ClassId-keyed table exists. LOWER targets `r2conc::SlabState::step`, machine semantics only. |

`lance-graph-contract::alpha_focus` and `rubicon_witness` use "focus" for an ATTENTION mask (rung × tenant). That is a different concept from DR's spatial focus; do not cite them for S7 or S11.

## 3. Reusable bindings versus mutable state

| kind | what | rule |
|---|---|---|
| reusable, validated once | a bound `Query` (S2); a `CompiledProgram`; a `MethodSignature` declaration | may be cached and reused across calls |
| mutable, re-checked at every use | frame seal (S6), focus revision and scope (S7), lane witness (S3), codebook generation (S15, once it exists) | never cached away. `bind_dr64` re-runs all three checks per borrow, and each is an O(1) pointer or value comparison. |

The D-MPF-2 measurement (#1752) attributes ~73–78 ns per call to R2IL procedure composition and ~45–63 ns to an ONNX-shaped wrapper that rebinds every call. Reusing a validated binding can only remove the RESOLUTION and PROGRAM-BUILD part of that. It cannot remove the guard checks, and it must not. Until the guard checks are timed separately from program construction, no claim that a boundary can be removed is made.

> ⊘ (D-MPF-4, #1752 plan §10) "R2IL procedure composition" above is the FoldDialect strategy re-run per call, not R2IL routing. For a fixed query, the existing `last_program()` → `CompiledProgram` facility removes it. A ClassId-prebound route runs at the direct `ndarray::simd` cost plus 27–28 ns for the frame/focus/scope guard. That guard is the one per-run check that must remain, and it is a versioned-identity check, not resolution.

## 4. The smallest missing seam

**S5 × S6: a Quack-bound query does not know its frame.** S1–S3 resolve and reuse a binding; S6–S8 guard a DR register against a stale frame, focus or scope. Nothing connects the two. A `Query` executed after its source, relation map or coordinate assignment moved is not refused by anything except S3, which covers only an ordered lane's order.

Proposed shape, not built (owners: Quack for the binding, the DR owner for the frame, no new ABI):

- the consumer's `Binder` names the frame revisions it resolved under (source, relation map, coordinates), exactly `FrameRevisions`;
- binding returns the `Query` together with a sealed `Frame` from the frame owner;
- every execution of a stored binding runs the frame check before lowering, and refuses with the existing `Refusal` variants.

This is one adapter between two types that already exist, not a new store, AST or runtime. It is also the precondition for S15 (a codebook generation is one more revision in the same seal) and S16.

## 5. Next stages (smallest experiment each; none authorized)

| D-id | experiment | gate |
|---|---|---|
| D-NFR-0 | this contract | shipped / lab / missing per seam, cited to source |
| D-NFR-1 | the S5 × S6 adapter as a lab probe: bind a Quack `Draft`, seal a frame, execute, advance one revision, execute again | the second execution is refused with the matching `Refusal`; the first equals an independent oracle |
| D-NFR-2 | S15: carry a codebook generation in the frame seal | a palette value bound under generation g is refused under g+1 |
| D-NFR-3 | S16: bind one DR axis to one palette plane (S/P/O) under a frame | an independent Cartesian oracle for coordinates and an independent numerical oracle for values both agree; a mismatched codebook is refused |
| D-NFR-4 | the end-to-end acceptance fixture (§6) | all nine steps, each measured alone |
| D-NFR-5 | one binding and one focus reused across several operations, against rebinding each time and against a fused scalar / SIMD baseline | the reuse saving is measured, guards still run on every use |

## 6. The acceptance fixture (design only)

1. Start from a Quack-bound snapshot (S1, S2) under a sealed frame (S6, via D-NFR-1).
2. Resolve typed identities and properties through the consumer `Binder`.
3. Project a local DR view: S11 focus slots, S9 register(s).
4. Select an existing Palette256 value operation (S13) through the D-NFR-3 binding.
5. Execute on mask-risc / ndarray (S17); palette lookups on bgz17.
6. Check coordinates against an independent Cartesian oracle.
7. Check values against an independent numerical oracle, against the stated error contract (S14), never as exact unless the layer is exact.
8. Refuse a stale frame, a moved focus, a changed scope and a mismatched codebook.
9. Report binding, focus construction, folding, SIMD execution and materialization separately.

No step adds a runtime.

## 7. Open points

| id | question | owner | status |
|---|---|---|---|
| NFR-OP-1 | who owns the production frame store (S6): Quack, ClassView, or the source owner? Narrowed 2026-10-26 (`dr256-where-capstone-v1.md` Part E): in-borrow execution needs none (the source owner, through the borrow); persisted reuse is `QueryReference::at`; open only for escaped reuse — will the mailbox owner issue a generation every write path advances | operator / mailbox write-path owner | OPEN (narrowed) |
| NFR-OP-2 | what the DR64³ third axis indexes: relation ordinal, property ordinal, or palette plane | DR owner | OPEN |
| NFR-OP-3 | whether `dir-rmap-probe`'s `Witness` and `dr-math-binding-probe`'s `Frame` become one type | DR owner | OPEN |
| NFR-OP-4 | the exact-versus-approximate admission rule for palette values inside a native fold | bgz owner | OPEN |
