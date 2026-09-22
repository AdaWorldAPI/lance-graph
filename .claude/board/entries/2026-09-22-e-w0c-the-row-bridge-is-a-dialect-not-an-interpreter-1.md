## 2026-09-22 — E-W0C-THE-ROW-BRIDGE-IS-A-DIALECT-NOT-AN-INTERPRETER-1 — a merged relational op carried as loco program data reaches the fused executor with no population crossing; the enum explosion is upstream of mask-risc

**STATUS: measured | SCOPE: `crates/r2il-mask-abi-probe/tests/row_bridge.rs` (W0C, the row bridge W0B named and left open) | BASIS: capstone audit "stop hand-rolling semantics", read against HEADs of ogar-loco, ogar-r2il, mask-risc, quack, ndarray `simd_masking_ops`**

**The question.** Can `COUNT(*) WHERE p.country = v THROUGH l.partner_id AND l.amount > t`
— today three parallel enums (`mask_risc::{Pred,Terminal}`, `quack::{Filter,Agg}`,
LGJ `LgjOpDesc`) — exist instead as BYTES, `(EQ_VIA:fk,key,v) (GT_I32:lane,t) (AND) (COUNT)`,
and reach the substrate through machinery that already exists, with the result
bit-identical to the native path and nothing row-sized written outside tile scratch?

**Answer: yes, and the seam is a `Dialect`, not an interpreter.** loco's `Interpreter`
dispatches; a `Vocabulary` above `DOMAIN_FLOOR` under its own classid declares
arity/pushes as a table; a `Dialect` whose stack `Value` is a mask-risc `Operand`
(a SLOT NAME, never words) turns interpretation into PROGRAM CONSTRUCTION;
`execute_into` then runs it tile-fused. loco refuses `FOR_EACH`/`FOR_RANGE`, so it
structurally cannot sweep rows — the correct property. A per-row R2IL replay
(the `zipper_hop_parity` shape) is the WRONG seam for this: scalar, materialising.

**Measured (5 tests, all green; rows 1/63/64/130/1000/65536; four disable arms red-then-green — table in the test's module doc):**

| metric | value |
|---|---|
| logical calls in the body | 4 |
| physical facade passes, loco dialect | **2** (`EqU32Via`, `GtI32` gated under it) |
| physical facade passes, `quack::lower` | **3** — a redundant trailing `And{a,b}` after `b` was already gated under `a` |
| loco-side bytes allocated (ops Vec + stack + body) | 208 at n=1,000 and 208 at n=65,536 — row-independent |
| executor bytes (tile-local scratch) | 136 at both n — not a population (65,536 rows would be 16 KiB/slot) |
| result | `Value::Count` equal to quack's program AND a scalar walk at every n |
| code surface of operation #2 (`GT_I32`) | one `match` arm + one vocabulary row; no enum variant, no ABI symbol, no status code |

**Classification the audit asked for.** A (physical irreducible, one facade call per
tile): all gated compares, `Range`, `Match*`, Boolean ops, `Ternlog`, `Gather`,
**`EqU32Via`** (address-through read, no source mask ever built), **`GroupSumViaI32`**
(segmented fold through the fk; the composition writes a population), **`CountKeyRunsU32`**
(run-boundary count with a 12-byte carry; `distinct.rs` already proves the alternative
consumes a population), rotation (a re-READ, no op). D (demanded sinks): `Count/Any/All`,
`MaskedSum/Min/Max`, `Keep`, `BlendI32`, `ScatterOrU32`. **No B/C found inside mask-risc.**
The masquerade is UPSTREAM: `quack::Filter`↔`Pred` and `quack::Agg`↔`Terminal` are 1:1
mirrors, and LGJ mirrors both again with a refusal arm per status. Three spellings, one meaning.

**What this does NOT do.** It promotes nothing: the vocabulary lives in the test.
It does not fix quack's third pass (its own wave, its own falsifier — pinned here
two-sided so the gap cannot close silently). It does not build a row-space R2IL
executor (none exists; R2IL's `CallMask` indexes ≤180 call slots and its branches are
addresses no engine honours). Wide compare literals (>255) go through loco's constant
pool, unexercised here. Direction settled, not migration: quack `Filter`/`Agg` and
`LgjOpDesc` become deletable in principle when callers hand loco bodies instead of enums.

**OPEN:** quack's redundant AND (measured, not fixed); constant-pool literals; whether
`R2ILVocabulary` and this mask-fold vocabulary should share a registry root or be
two classids (both routes exist today, neither is wired to the other).
