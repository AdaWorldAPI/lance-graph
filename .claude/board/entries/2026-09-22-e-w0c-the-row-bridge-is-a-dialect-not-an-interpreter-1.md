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

---

## AMENDMENT (2026-09-22, same day) — the local vocabulary is GONE; W1a landed, so the probe now runs OGAR's real table

**STATUS: measured | SCOPE: same file, rewritten | BASIS: OGAR #306 merged (main `7c6ddba`), band `0xE2..=0xED` verified byte-identical on merged main**

The entry above says *"It promotes nothing: the vocabulary lives in the test."* That
is now **superseded**, and the OPEN item *"whether `R2ILVocabulary` and this mask-fold
vocabulary should share a registry root or be two classids"* is **RESOLVED**: there is
no second vocabulary. `MaskFoldVocabulary` is deleted. `tests/row_bridge.rs` consumes
`ogar_r2il::R2ILVocabulary` — ONE arity table in the tree, two concept ids
(`CONCEPT_R2IL_MACHINE` / `CONCEPT_R2IL_FOLD`), because `VocabularyRegistry::plug`
copies `*v.table()` and refuses only a taken id. The packed `EQ_VIA fk,key,v`
spelling is gone with it: operands arrive on the stack via `NUMBER`, and `Load`'s
immediate carries the lane KIND as P-Code's address SPACE.

**Measured, three frontends through ONE dialect:**

| | value |
|---|---|
| A (quack-shaped, join leaf first) | 2 facade passes |
| B (Blockly-shaped, local filter first) | 2 facade passes — order-independent |
| `quack::lower`, same query | 3 (the redundant trailing AND, still not fixed) |
| dialect-side allocation | **728 B at n=1,000 AND at n=65,536** — identical |
| executor scratch at n=65,536 | 65 words against a 64-word tile cap; a population slot would need 1,024 |
| C (Mathcad-shaped) `SUM(amount)-SUM(cost)` | margin 8721, **`programs_run == 2`** |

The Mathcad case forced the design's one real refinement: a mask-risc `Program` has
ONE terminal, and no R2IL op has `body_refs > 0` (so no multi-body route), therefore a
scalar-producing fold FINALIZES the accumulated ops, RUNS them, and pushes a real
`Scalar`. `INT_SUB` then combines two computed scalars. loco still composes and
mask-risc still executes; only the WHEN moves, to each fold boundary. Nothing
row-sized ever reaches the stack — only `Scalar`, `Slot`, `Address`, `Sink`.

**Four defects this pass found in the version above, three of them in prose I wrote:**

1. **The fixture comment claimed a zero fallback that does not exist.** It said "the
   zero-fallback for an out-of-range fk is part of the contract." `eq_u32_via_to_mask`
   is `if addr < table.len() && table[addr] == v` — it **DROPS**. The zero-fallback
   belongs to `mask_gather_u32`, a different kernel. The oracle was right all along;
   the comment imported one kernel's addressing contract onto another's. A reviewer
   proposed "fixing" the oracle to `unwrap_or(0)`, which would have made it disagree
   with the kernel at `v == 0` — an oracle that stops reproducing the operation under
   test. Rejected with the kernel body as evidence; the reviewer conceded and recorded
   the distinction. **Both kernels are now documented side by side so they cannot blur
   again.**
2. **A vacuous disable arm, found by RUNNING it rather than reasoning about it.** The
   arm "peephole fires on a non-last op" is **structurally unreachable** in this
   vocabulary: with no DUP or SWAP primitive, the value on top of the stack was always
   produced by the last emitted op, so scanning the whole `ops` vec instead of
   `last_mut()` changed nothing and all tests stayed green. The load-bearing guard is
   the **UNGATED** half (`under: None`), which does go red (3 ops instead of 4). The
   vacuous row is replaced by the real one and the finding is documented rather than
   the row being quietly kept.
3. **`(T - 40) as u8` silently became 233, not -23.** `NUMBER` decodes unconditionally
   UNSIGNED, so there is no signed inline range in this file at all. The module doc now
   states the range per decode instead of one blanket sentence — the blanket sentence
   ("a compare literal past 255") is how this drifted, since it is wrong for exactly
   the half of the ops it appears to cover.
4. **The allocation instrument was unsound and documented as sufficient**, which is
   worse than an unexamined bug. A process-global `AtomicUsize` with a
   min-over-eight-samples mitigation cannot yield an uncontaminated figure if every
   sample is contaminated, and different contamination at the two row counts makes the
   equality assertion flaky rather than loud. Now a `thread_local!` `Cell` with a
   **`const` initializer** — a non-`const` one takes a lazily-boxed path that would
   recurse through the allocator being measured.

   **⊘ CORRECTION, same day, and it is about EVIDENCE not about the fix.** The line
   that stood here claimed "the disable run that validated the fix measured 992 vs
   17,120 bytes." That is a WRONG ATTRIBUTION, repeated from a worker summary without
   checking: those numbers belong to the ROW-DEPENDENCE arm (scratch allocated per fold
   instead of once), which guards a different property. Measured directly — a valid
   three-anchor patch reverting the counter to the exact process-global `AtomicUsize`
   shape it superseded, all three anchors assertion-checked — **all 12 tests stay GREEN
   at 728 bytes both times.** So the thread-local change has **no deterministic
   falsifier**, and it never could: the contamination it removes is scheduling-
   dependent, so a single run cannot exhibit it.

   That does not make the change wrong; it makes it a **structural soundness fix**
   whose correctness is an argument (the allocator runs on the allocating thread, so a
   thread-local cannot receive another thread's bytes) rather than a measurement. It is
   recorded as such here, because "eight arms red-then-green" would otherwise count a
   green run as evidence for a guard it cannot test — the exact shape of the vacuous
   arm found one bullet above. Seven arms are red-then-green; this one is structural.

**Seven disable arms red-then-green, plus one structural change with no deterministic
falsifier** (the thread-local counter, see the correction above), restored from a file
copy rather than `git checkout` (the work was uncommitted; `git checkout` would have eaten it — the
lesson this workspace already paid for once).

**OPEN, narrowed.** quack's redundant third pass (measured, unfixed, pinned two-sided).
Only `VIA` and `SUM` of the twelve fold bytes are implemented; the other ten are
refused BY NAME as `Unimplemented`, so **`GROUP_SUM`'s selection claim — one byte
choosing `GroupSumI32` vs `GroupSumViaI32` off its key operand's kind — is declared in
OGAR's arity table but NOT yet exercised by any dialect.** The table only ever claimed
arity, so that is not an overclaim, but the selection behaviour is unproven and should
be the next arm. `CONSTANT` stays unwired (nothing here needs a literal past 255).
`FIRST` still unminted, still waiting on a `receive`-shaped falsifier.
