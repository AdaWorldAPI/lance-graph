# Folding doctrine — the map is not the city

> READ BY: simd-savant, kernel-membrane-warden, zero-copy-warden,
> lens-migration-engineer, truth-architect, and any session about to write a
> shuffle, transpose, gather/scatter, lane rotation, "canonicalize" step,
> scratch buffer, or crypto round layout.
> READ BEFORE: implementing any algorithm whose description contains
> rotate / transpose / swap axes / reverse / shear / diagonalize / permute /
> "visit columns instead of rows".
>
> Status: WORKING-MODEL, with MEASURED support (sources cited per claim).
> DECISION: fold by default; materialize only where total downstream cost is
> lower. SCOPE: SIMD kernels, crypto rounds, masks, tensors, layouts, graph
> coordinates. BASIS: the shipped examples and measurements in §6.
> REVISIT WHEN: a measured case shows a folded path losing to materialization
> that the break-even law in §3 did not predict.

## 1. The picture

Think of the data as a city and the algorithm as a route map. A transformation
of the map (rotate 90°, transpose, reverse an axis, shear rows, reinterpret
lanes) is first a change of **coordinates**:

```text
physical value  +  logical coordinate
```

The coordinate may change while the value stays where it is. That is
**folding**. Carry the transformed coordinate system forward for as long as
it is cheaper than moving the data.

A 90° rotation is the textbook case: swap axes plus reverse one axis.

```text
LEFT / CCW   B[r,c] = A[c, W-1-r]
RIGHT / CW   B[r,c] = A[H-1-c, r]
```

Where the consumer accepts a strided view, that is O(1) metadata, not O(N)
movement. Where contiguous output is eventually needed, fuse the index map
into the kernel that was going to read the data anyway (one traversal, not
rotate-then-consume).

**Teleportation** is a physical move (shuffle, transpose, gather, copy). It is
necessary only when a real rendezvous is required: two values must meet in
one operation, and no register renaming, operand selection, index remap,
addressing mode, carried permutation or permutation composition can make
them meet. That is the materialization boundary.

Named transposes and rotations in an algorithm describe **relationships**, not
instructions. Implement the dependency graph, not the drawing.

**Canonicalization is suspicious.** In `transform → compute → transform back`,
ask why the step back exists. It is justified only by a real consumer or
interface boundary, never by how the next stage is traditionally drawn.

## 2. Folding is not "avoid movement at all costs"

It is: **delay physicalization until you can show it is cheaper than carrying
the abstraction.** Teleports are one tax among four:

| cost class | examples |
|---|---|
| 1. routing / materialization | shuffles, gather/scatter, transposes, copies, intermediate writes |
| 2. control | per-tile loops, call/dispatch overhead, front-end pressure |
| 3. interpretation | composing maps, recognizing folds, storing and re-reading runtime mapping state |
| 4. access / backend | strided access, gather latency, cache locality, backend-specific instruction expansion |

The optimization target is **minimum total physicalization cost** across all
four, not minimum teleports. §6.1 is the measured case where counting only
routing would have missed the dominant term.

## 3. The break-even law

With `U` future uses of the representation:

```text
carry the fold:       C_compose     + U · C_carried
materialize once:     C_materialize + U · C_canonical
```

Materialize when

```text
C_materialize  <  C_compose + U · (C_carried − C_canonical)
```

- Canonical access is not free, so the comparison uses the **extra** cost of
  carrying the map, not its total access cost.
- `C_compose` includes recognizing the fold. If that happens per call, it is
  paid per call (§6.1: the fold lost at small extents for exactly this
  reason).
- A transpose that will be read hundreds of times through strided or gathered
  access can be cheaper to do once. Deferral is the default, not an absolute.

## 4. The best fold disappears at compile time

A folded map should become operand selection, register naming, addressing
modes, or a fixed backend sequence, all decided at compile time. If carrying
the map creates runtime metadata that must be decoded on every operation, the
map has been reified into a new small city. Measure that cost explicitly
(class 3), and prefer type-level or `const` coordinate transforms.

**Crypto rule.** Only public, compile-time coordinate transforms may be folded
freely. A secret-dependent remap that changes memory addresses, branches or
other observable timing is not a free map; it breaks constant-time execution.
(Data-dependent *block* addressing that an algorithm specifies, like Argon2d's
reference-block choice, is part of the algorithm, not a layout fold.)

## 5. The backend decides how much of the map is physical

The map is semantic; whether it is free is backend-specific. The same logical
relation can be one instruction on one tier and a sequence on another (§6.1:
every 3-input truth table is one VPTERNLOG on AVX-512, a composed sequence on
AVX2). Measure per tier; a cost claim without its target tier is an anecdote.

**Composed maps need the same falsifiers as moved data.** An index composition
can overflow, alias or silently go out of range. Test it with the discipline
used for kernels: a check that fails when the guard is removed (§6.3).

**Relation to the zero-copy law** (`zero-copy-lens-law.md`). The break-even law
governs **transient** routing (registers, tile scratch) and a terminal's
elected output (a `Keep` bitmap, a `materialize*` result). It never licenses a
second owner of substrate bytes. That remains forbidden without exception, so
this doctrine operates inside the zero-copy law, not beside it.

## 6. Shipped examples and measurements

### 6.1 Program collapse probe — control cost dominates, backend decides

`.claude/board/entries/2026-09-23-collapse-probe-v4-and-dispatch-split.md`
(`crates/lance-graph-mask-risc/examples/program_collapse_probe.rs`; MEASURED,
N = 1 048 576 rows, x86-64 only):

- 90–94 % of the tiled path's time is the tiled−bulk gap: 33–54 ns per op per
  tile, from running each op per 8-word tile. Not data movement (class 2, not
  class 1).
- At v4 (AVX-512) every collapsible 3-plane Count costs ~6.2 µs whatever its
  truth table; at v3 (AVX2) the same folds cost 8.6–17.3 µs (§5).
- At 1 % of the population the bulk path beats the fold, because the fold
  re-runs `validate` and the recognizer on every execute. Caching the
  recognized form on `Program` is OPEN (class 3, §3).

### 6.2 The terminal elects materialization — Range ∩ plane → Count/Any

`.claude/board/entries/2026-09-23-terminal-elects-materialization-range-plane.md`
(commits `9982e9c`, `daa1585`). The range never becomes a bitmap: the fused
terminal reads the resident plane's touched words plus two register-masked
edge words, writes 0 derived words and carves 0 scratch slots. `Keep` is the
one terminal that elects a bitmap. It needed no new primitive, only a
lowering rule. MEASURED (`range_fused_probe`): at N = 1 048 576 with a tiny
range, 138 620 ns materialized against 56 ns fused.

### 6.3 Two-column GROUP BY — carry the composed address

ndarray `f6f3e26e` (`GroupKeyAddr::Pair`): the group address is `hi · stride
+ lo`, computed on the fly; no combined key column exists. Its guards are
tested at the edges (`lo == stride − 1` kept, `lo == stride` and
`lo == u32::MAX` dropped, `hi = stride = u32::MAX` without overflow).
`2456a840` shows why composed maps need falsifiers: the first stride-drop test
was vacuous (removing the `lo >= stride` guard left it green) until dropped
rows were made to alias real slots.

### 6.4 Boolean chain collapse — permutations cancel, logic edition

ndarray `7695fa68` (`mask_ternlog_popcount` / `mask_ternlog_any`) and
`Program::fused_ternlog` (`crates/lance-graph-mask-risc/src/ir.rs`): several AND/OR/NOT steps
fold into one 3-input truth table over the input planes, and the chain ends in
Count or Any with no mask written.

### 6.5 Argon2 compress — count before choosing a layout

ndarray #332 (`U64x8::transpose8`) and its blackboard entry. Moves counted
(a word "teleports" when its lane changes between stages; 128 words, 8 lanes,
register choice free):

| layout | load→row | in row pass | row→col | in col pass | col→store | total |
|---|---|---|---|---|---|---|
| vertical (lane = permutation) | 112 | 0 | 112 | 0 | 112 | 336 |
| horizontal (diagonalize by lane rotation) | 64 | 96 | 112 | 96 | 112 | 480 |

- The textbook "diagonalize / undiagonalize" drawing is register renaming in
  the vertical layout (0 moves). Implementing the drawing (horizontal) costs
  96 moves per pass.
- Row → column is the irreducible rendezvous: a column `G` reads words from
  four different lanes. That is the one place for a physical transpose.
- Load and store moves exist only because blocks are kept in canonical order.
  Keeping Argon2's blocks transposed carries the map across calls (224 left).
  That is a proposal, not shipped: it changes what `Block::as_ref()` exposes.
- Shipped as password-hashes #3: the block stays in registers from load to
  store, with 8 `transpose8` calls (2 load, 2 row→col, 4 store).

MEASURED (password-hashes #3, Argon2id 64 MiB t=3 p=1, best of 9, three runs;
instruction counts from `objdump` of the one function holding `vpmuludq`,
`Argon2::compress`, into which `compress_simd` is inlined):

| tier | scalar | old SIMD (gather/scatter) | in-register | insns | `vpmuludq` | stack ops |
|---|---|---|---|---|---|---|
| v3 (AVX2) | 121–130 ms | 118–124 ms | 121–128 ms | 2683 | 128 | 478 |
| v4 (AVX-512) | 138–148 ms | 111–115 ms | 91–95 ms | 709 | 64 | 64 |

- The same logical routing is a win on one tier and a wash on another (§5).
  On v3 `U64x8` is two `ymm` halves, so 16 live vectors need 32 registers
  against 16: about 480 of 2683 instructions are spill traffic (class 4).
  On v4 the 16 vectors fit the 32 `zmm` registers, and `vprolq`/`vpternlog`
  also fold rotates and 3-way XORs (class 2).
- Routing (class 1) was not the v3 bottleneck; register pressure was. A
  "fewer teleports" count alone would have predicted a win on both tiers.

## 7. Checklist

1. Write the dependency graph, not the diagram. Which values must meet?
2. For every transform, ask: can it be renaming, index composition, addressing,
   or a carried permutation?
3. For every `… → transform back`, name the consumer that forces it.
4. Count all four cost classes on each target tier; apply §3.
5. Make the carried map compile-time where possible; in crypto, require it.
6. Give every composed index map a falsifier, like any kernel.
