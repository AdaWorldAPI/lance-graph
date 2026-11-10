# DR × DT64 Locality and Execution Contract

> READ BY: any session that touches DR coordinates, DT64 bindings
> (`LaneBinding`, `ClassView`, `WideFieldMask`, `VirtualObject`), source-owned
> SoA lanes, Quack / mask-RISC lowering, or an ndarray / AMX execution path
> fed from bound lanes; every reviewer of such a PR; `layer-boundary-warden`,
> `fold-carrier-scientist`, `zero-copy-warden`, `v3-mailbox-warden`.
>
> Status: CURRENT-CONTRACT for the invariant and the five boundaries.
> The evidence section is graded per item; read the grade, not the tone.
> Linear decision: EXO-220 in P-EXO-11.
> Born 2026-11-10 from D-DT64-GEMM-SHAPE-1/2/3 (#1987, #1988 and follow-ups).

## Central invariant

**Resolve identity and location once per valid binding, execute against
source-owned ordered planes locally, and only re-resolve, re-derive or
materialize when an explicit semantic, version or hardware requirement makes
it necessary.**

The governing question for every PR:

> What does this implementation recompute or reconstruct per operation that it
> could lawfully resolve once per binding?

The test: *changing the menu should not require rebuilding the restaurant.*

The goal is amortizing knowledge of location, not merely avoiding copies. A
zero-copy path that repeats a lookup per element or per field is still the
phone book carried into the kitchen. (Measured instance below: a zero-copy
resolve that was O(N²) because it called a linear-scan lookup per field.)

## The five boundaries

Each has one owner. A design that moves a responsibility across one of these
lines needs an explicit decision, not a convenience.

### 1. DR: witnessed coordinate and address fabric

DR says *where* a value is: positions in a coordinate space, routed through a
store whose frame, route and coordinate revisions are witnessed. It carries no
ISA, ndarray or AMX meaning.

- Global-to-local (G2L) mapping is an **optional, admitted binding
  operation** used only when coordinate spaces differ. It is never a
  compulsory step before each operation, and never applied to coordinates
  that are already aligned.
- A lane in another row order is refused (`LaneRefusal::CoordinateSpace`),
  not silently translated.
- G2L does not grant Gather. Cross-table, value-directed position reads stay
  under ADMISSION B (#1640) and the Gather restrictions; a prepared map does
  not admit itself.

### 2. DT64: logical shape, interpretation and versioned binding

DT64 is the menu. `ClassView`, `WideFieldMask` and a CustomObject-style
`VirtualObject` say which fields a consumer wants, in what order and shape,
and how properties and methods are interpreted. A binding (`LaneBinding`)
holds addresses only and is checked against route, frame, schema basis,
mapping, scope and coordinate space.

- Reordering or reshaping the menu changes the view, never the kitchen.
- A valid binding is reused; it is re-made only when one of its witnesses
  changes.
- DT64 is not a value store and not a matrix store.

### 3. Source-owned SoA: one canonical value owner

The source owns the inventory: ordered planes in its own layout and its own
frame store. Consumers borrow lanes; they do not copy them into a parallel
object universe.

- No per-object physical copy of source-owned values.
- A copy is legitimate only as a bounded staging step for execution
  (boundary 4), and it is counted.

### 4. Quack / mask-RISC / ndarray / ISA: lawful execution

Quack defines and lowers the lawful operation; mask-RISC and ndarray (through
`ndarray::simd`) execute it; the ISA does the arithmetic.

- Read operands in place when the source geometry fits the instruction
  (FACET STRIDE: read positions where they sit).
- Stage only what the instruction cannot consume directly, and only the tiles
  it needs.
- Hardware-required layouts (AMX VNNI packing of B) are execution
  requirements, not a second canonical representation. Prepare them once and
  reuse them under authoritative operand identity and version; never re-derive
  them per operation without a reason.
- Typed arithmetic contracts are preserved (AMX `TDPBUSD`: unsigned u8 A,
  signed i8 B; no implicit reinterpretation).

### 5. IAM: independent authorization

Authorization is its own decision, checked at the read or execution boundary.
It is **never inferred from a matching DR scope**: `Scope { id }` is a value
the binding checks for consistency, not proof that the reader may read. A
binding that type-checks does not authorize; an authorized reader still needs
a valid binding.

## What must not be introduced silently

- Another canonical identity authority.
- A per-object physical copy of source-owned data.
- A compulsory G2L conversion for already aligned coordinates.
- Repeated binding or lookup for an unchanged valid context.
- An ndarray or AMX meaning inside the DR address contract.
- Unauthorized Gather, cross-space reads or stale-frame reuse.
- Reuse of a prepared operand keyed by a hash alone. Reuse must rest on
  authoritative identity and version (selection equality, frame issue,
  operand identity and revision, shape), not on a digest that can collide.

Witnesses to keep: scope, coordinate space, route, source frame, schema basis,
field mapping, operand identity and version.

## Evidence (graded)

| Item | Grade | What it shows |
|---|---|---|
| #1987 — BF16 virtual 64×64 and 256×256 objects over source-owned lanes | MEASURED, merged | One binding feeds direct strided AMX (0 bytes of A copied), staged, CBLAS and oracle arms with exact parity; refusals for frame, scope, mapping, basis, coordinate space, foreign lanes and out-of-range selections |
| #1988 — `LaneBinding::borrow_all` | VERIFIED-IN-CODE, merged | One check, then every bound lane; N source lookups instead of N² |
| Lab resolve on `borrow_all` (follow-up) | MEASURED, not merged | Resolve at 256 fields 1.6–2.0 µs vs 13.8–13.9 µs (#1987 as first measured) and 26–28 µs (`main` after the provenance fix); the old cost was `LaneBinding::addr`, a linear scan called per field |
| INT8 vs BF16 over the same object (follow-up) | MEASURED, not merged | `u8 × i8 → i32` end to end 66–101 µs vs BF16 103–160 µs at 256 fields on one VM; INT8 still copies A (5–7 µs) and re-packs B per call (23–29 µs, 36–38% of its GEMM), because ndarray has no strided-A / prepacked-B INT8 entry |
| #1987's "B preparation ≈ GEMM" | REJECTED-BY-MEASUREMENT | That timer included the synthetic operand's hash; from a table built once, prepared B costs about a third of the GEMM |
| G2L across different source orders | OPEN | Not demonstrated: every fixture shared one coordinate space, and differing spaces were refused, not translated |
| Foundry operational-ontology parity | HYPOTHESIS | Research goal; nothing here establishes it |

All timings: one VM (Xeon with AMX-BF16, 4 vCPU), single-threaded, medians of
41 warm runs, three runs per figure, ranges reported; run-to-run variation up
to about 35%. Not certified on dedicated silicon.

## Applying it in review

For each PR touching these boundaries, answer in the PR body:

1. What is resolved once per binding, and what per operation?
2. Which witnesses invalidate the reuse, and are all of them checked?
3. Which bytes are copied, and which boundary requires the copy?
4. Which claims are measured (with the command and host), and which are
   hypotheses?

## Recorded

- Linear decision: [EXO-220](https://linear.app/exo-red/issue/EXO-220) in P-EXO-11.
- Source PRs: #1987 and #1988 (merged), and the two lab follow-ups that cite
  this document.
