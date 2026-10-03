# Co-research: chained hops over masks — exploration map (2026-10-03)

**Status:** OPEN — exploration map; ratifies nothing. Question from #1308. Harness:
`.claude/agents/coresearch-council.md` (first run).

**Question (re-asked after the premise gate, PREMISE-SPLIT).** "A mask produced by one
program feeds the next program's Gather" was three questions:

- **Q1:** a fixed k-hop pull chain with no intermediate mask;
- **Q2:** reachability `*1..k`, where frontier and `visited` are loop-carried state;
- **Q3:** what `ForeignPlane` *is*.

**Anchors held:**
- A1, the tiling law (`mask-risc/src/lib.rs:16-22`);
- A2, the scatter survival condition (`ir.rs:306-314`);
- A3, the Gather doc;
- A4–A6;
- A7, reference frame vs motion.

## Council

**Scouts:**
- code cartographer;
- internal prior art;
- literature (7 papers, SECTION-READ);
- systems (DuckDB, Kùzu, GraphBLAS/LAGraph, PostgreSQL, Soufflé);
- concepts (semijoin algebra, DBSP, Kleene and Knaster–Tarski).

**Co-architects:** bridge, firewall, falsifier. The crosswalk has 18 rows (X1–X18); raw output is banked in the session scratchpad.

## What the code says (VERIFIED-IN-CODE)

- **No loop construct.** A program is straight-line, pre-filled scratch is refused (`ir.rs:509-528`, `:519-522`), and `Out` has no loop variant (`value.rs:44-55`).
- **`ForeignPlane` is read whole by key on every tile** (`exec.rs:1714-1732`). Its fields are `pub` and it wraps a plain `&[u64]` (`ir.rs:86-92`). So **a scattered `Out::Mask` can be re-fed as a `ForeignPlane` today**. The only fence is the doc wording "resident validity" (`ir.rs:82`) plus quack's doc precondition (`quack/src/lib.rs:360-370`).
- **The deepest composed read is depth 2, over values** (`EqU32Via`, `GroupKey::Via`; ndarray `*_via`). There is no depth-k bit gather, and `mask_gather_u32` (`simd_masking_ops.rs:1099`) is a single-index scalar read.
- **Every existing graph traversal materialises its frontier:**
  - `hdr_bfs`;
  - `multi_hop`;
  - `CsrIndex::bfs`;
  - the sibling `lgj_hop`, whose `Graph.hop().hop()` re-feeds a scattered mask (pre-A1; X16).

## Convergent outside evidence

- **BFS step.** BFS is one masked mxv step, `f ← Aᵀf .∗ ¬v`. GraphBLAS push/pull, Datalog semi-naive evaluation, DBSP, the CTE working table and Kùzu IFE all describe the same two-state loop.
- **Operand reuse** (`Aᵀv .∗ ¬v`, Yang–Buluç–Owens) needs one carried mask, not two.
- **DBSP types loop state** as a `z⁻¹` back edge bracketed by `δ0…∫`. This is distinct from constant inputs (resident) and from the output (demanded). A linear step needs exactly one accumulator.
- **Bounded `*1..k`** is a finite union of fixed chains. Loop state is semantically necessary only for unbounded `*`.

## Co-architect disagreement (stricter verdict kept)

The bridge architect read loop state as lying *outside* A1's conditional clause ("when the next fold can consume directly"). The firewall critic answered **TRAP**: A1's last sentence is unconditional — *"the only population-sized writes are the DEMANDED sinks"*. An iterate `v_t` with `t < k` is population-sized and not demanded.

Reading "can consume directly" as "with today's primitives" would also invert the missing-capability STOP rule. **So loop state needs an operator amendment to A1. It is not a reading of A1.**

## Exploration map

| idea | verdict | why / shape |
|---|---|---|
| **Tile-local gather chain** `start[fkʲ(i)]` (X9 reframed, X10) | **PROBE → substrate-first** | Answers Q1, and bounded `*1..k` over FUNCTIONAL (in-row fk) hops, with **no amendment**. State is tile-sized (`cur: [u32; TILE_ROWS]`, `acc`) and every source is resident. Firewall PASS, STOP-gated. Needs one ndarray primitive (`mask_gather_chain_u32` / `mask_gather_via_u32`, W1a contract), then a mask-risc `MaskOp::GatherChain`. A multi-valued hop is a join and is refused here. Probe: P-COMPOSE. |
| **`LoopPlane` / LOOP-STATE class** (X3+X5+X6+X8) | **PROBE, then operator decision** | Fits multi-valued hops and unbounded `*`. A driver loops a straight-line step (`Gather{src: Loop}` → `AndNot` → `Keep`), with one owner (the driver frame) and a lifetime of one invocation. Scatter is refused as the step terminal, and it never escapes before ∫. It **requires amending A1's sentence 2**. Probes P-REUSE and P-DETERMINISM. |
| Double buffering (X8) | **PROBE** (part of the above) | In-place `v ∨= G(v)` over-reaches under bounded k: with ascending tile order a chain advances many hops per step. It is order-free only for unbounded `*` (monotone closure). Safe Rust already forbids aliasing the Gather source with its `dst`, which settles X18. Consequence: bounded k costs 2 population buffers either way. |
| **`ForeignPlane` provenance type** (X15) | **PROBE (defect pin first)** | A doc fix alone would delete the only fence: the doc should describe the mechanism ("key-addressed, read whole"), but provenance must be a TYPE in mask-risc. Make the fields private and add named constructors `resident(..)` and a loop-state variant, with no conversion from `Out`. This makes the step named, not provable. Probe: P-Q3. |
| Semi-naive Δ/I (X4) | SKIP | Superseded by operand reuse; a second undemanded population. Measured: it buys nothing span-bound (ndarray D-GTM-1m). |
| Push carried across iterations, per-step direction switch (X2) | SKIP | CONFLICTS-ANCHOR A2. Push stays a final demanded sink. Forward reachability needs a declared reverse lane (RF-TRANSPOSE). |
| Pointer doubling / composed `fkʲ` lanes (X9) | SKIP (TRAP) | Population-sized u32 intermediates; forbidden by A1 *a fortiori*. P-COMPOSE C3 measures it only as a contrast. |
| Yannakakis full reducer / predicate transfer (X12) | PARK | Exact sets for every variable (answers v2 H-2), but only as declared demanded sinks, i.e. multi-sink at the consumer. Bloom filters are refused (approximate). |
| MS-BFS 64-lane bitsets (X14) | PARK | The bit axis would be sources, not rows; it would need a separate plane class. |
| 2-phase law (X11), Kùzu S-Join (X13) | ALREADY-HAVE | External corroboration of A2 / of Gather over a resident plane. |
| Waben "elected reusable frontier" (X17) | PARK | Broader than A1 allows; would be narrowed to the LOOP-STATE bracket if that is adopted. |
| `lgj_hop` push chain (X16) | out of scope | A tension for lance-graph-java's board, not this repo. |

## Probes (pre-registered by the falsifier designer)

**P-REUSE** — ndarray example, about 3 h.
- **Fixture:** functional fk, n = 4,096, seed `0xC0FFEE`, with a 3-cycle through start, a self-loop, a tail-into-cycle, a 40-chain and 5 out-of-range fks.
- **Check:** operand reuse == reuse+ANDNOT == semi-naive == two independent oracles, for every t ≤ 45.
- **Kill:**
  - the wrong seed `v_0 = start` must differ from the oracle at t = 2 (can-it-fire);
  - anti-vacuity: `kept*3 < n`.
- **Cost half:** 65,536 rows, v4 and v3, kill if reuse is more than 1.25× semi-naive.

**P-DETERMINISM** — about 2 h.
- **Fixture:** a 63-chain across 64 tiles; ascending, descending and 1,000 shuffled tile orders.
- **Expected:**
  - in-place at t = 1, ascending: `|v| == 63` where the oracle has `1` (fires);
  - double buffering is identical across all orders (silent);
  - in-place unbounded equals the closure.

**P-COMPOSE** — about 4 h plus about 1 day for the primitive.
- **Target:** `start[fkᵏ(i)]` for k ∈ {0,1,2,3,7,16,64}, n ∈ {2¹⁶, 2²⁴}, with an fk out of range at depth 2 only.
- **Arms:** chained masks (forbidden contrast), composed walk, pointer doubling, and the union of depths (must equal P-REUSE at t = k).

**P-Q3** — about 2 h.
- **Defect pin:** `out_mask_refeeds_as_foreign_plane_today` passing is the defect, recorded OPEN.
- **Guard:** private fields; trybuild compile-fail on the struct literal plus a source fence.
- **Silent half:** all existing tests unchanged except for renamed constructors.

## Not searched or not read

- Beamer SC'12 and MS-BFS (SECONDHAND only);
- DuckDB mark/semi-join build pipelining;
- RedisGraph/FalkorDB internals;
- whether quack's `ForeignPlane(pub u16)` needs the same fence.

## Open, for the operator

The premise gate was re-run on the final option set; it includes the "belongs elsewhere" option. The options:

- **(a) Tile-local gather chain** — substrate-first, needs no amendment. Covers Q1 and bounded functional `*1..k`.
- **(b) A1 amendment naming LOOP-STATE** as the one non-demanded population class (bracketed, single owner, never escapes before ∫). Covers multi-valued hops and unbounded `*`.
- **(c) `ForeignPlane` provenance type** — ⊘ REVISED in Round 2 below: `resident(..)` only, no loop-state variant.
- **(d) Recursion belongs elsewhere:** mask-risc stays non-recursive, and `*` stays refused (RF-*) or lives in a consumer-side driver.

(a) and (c) do not need (b).

## Round 2 (after operator review, same day)

The operator asked for two challenges before anything is chosen: make the falsifier try to break **(a)+(d)**, and audit (c) against the Foreign contract's own history, not Rust type compatibility.

### (c) — premise audit, PREMISE-WRONG on "a computed `Out::Mask` is a `ForeignPlane`"

**The history** (commit text quoted):
- `663b8792` (2026-09-21) sanctioned the identification: `ScatterOrU32` output "a follow-on program then reads it back as a [`ForeignPlane`]".
- `a9a3e9d4` withdrew it the same day: "`foreign` must name a RESIDENT plane the caller holds … never a mask another program produced". That commit also deleted the Keep → foreign plane → Gather pipeline in favour of the factored `EqU32Via`.

**Signatures.** Foreign is caller-held standing state (frame-like). A computed mask and a loop iterate are motion: the writer is the executor or a loop, and they live for one run or one step. Tests 1–4 all fire.

**Mechanical vs semantic.** Physical compatibility (`&[u64]`, pub fields, `ir.rs:86-92`) is a found contract smell, not an established equivalence.

**Repair, correctly named:**
- **R1:** `ForeignPlane::resident(..)` is the ONLY constructor, and the fields become private. Precedent: dir-sim's `Kept` `compile_fail`.
- **R2:** a computed same-query plane gets no type and stays refused; chains are lowered in factored form.
- **R3:** only on measured need, a separate motion type that never converts to `ForeignPlane`.

⊘ The earlier "loop-state variant ON `ForeignPlane`" would have re-sealed the withdrawn claim. It is struck.

**Live use, checked:** the only `ForeignPlane {` literals are in `mask-risc/tests/{foreign,extent}.rs`, quack's test module and dir-sim doctests. All of them wrap resident fixture planes. So the gap is open, but no production caller uses it.

### (a)+(d) — falsifier round 2: nothing requires mask-risc to OWN loop state

| case | verdict | reason |
|---|---|---|
| bounded `*1..k`, multi-valued / reverse lane | SURVIVES via (d) | (a) costs L^k per row; a double-buffered kernel loop is deterministic. Push-only relations stay refused (A2 / RF-CHAIN) under every option, (b) included. |
| reachability, then `AND pred` and `GROUP BY` | **breaks (d) as worded; SURVIVES as d′** | If the kernel returns `v_k`, the next fold could have consumed that population tile by tile: A1 by its letter. d′: for bounded k the kernel returns `v_{k-1}` (a true pipeline breaker), and mask-risc fuses the last step with pred and the fold. For unbounded `*`, returning the converged iterate is legitimate (convergence is a global test). |
| cycle-sensitive `*1..` lower bound | SURVIVES | Seed `v_0 = ∅`, `v_1 = G(start)`. Already in the P-REUSE fixture. |
| §4.2, D-CML-5/5a, OQ-CML-4 | SURVIVES | §4.2 is a (d) driver. 5a and OQ-CML-4 are prerequisites for both. |
| **node-filtered reachability** (`ALL(n IN nodes(p) WHERE P(n))`), not in v2's table | **breaks (a)+(d)** | Under (d), Quack must sink P as a whole plane that the step could consume tile-locally: A1. Calling the kernel "demanding" P would make A1 a guard that never fires. |

**What (b) would still buy:** nothing measurable predicted; only A1 conformance at the seam. The two breaking cases need **b-lite**: a kernel-owned iterate READ by mask-risc through a named type that is neither `ForeignPlane` nor `Kept`. That is exactly the auditor's **R3**, and a type that names provenance cannot prove it.

**New probes** (pre-registered):
- **P-SEAM:**
  - n = 65,536, functional fk plus a 4-lane variant, predicate plus group key, k = 8.
  - Arms: S1 (kernel returns `v_k`, then fold) vs S2 (d′).
  - Exact against scalar BFS plus scalar group sum.
  - S1 holds exactly one more population buffer.
  - Kill d′ if S2 needs a raw `ForeignPlane` literal.
  - Kill the claim that (b) or b-lite buys cost if S1 − S2 < 2 %.
- **P-FILTERED:**
  - A 4-lane pull carrier, P at 50 % on a row field.
  - Arms: F1 (sink P whole) vs F2 (P recomputed per tile per step).
  - Exact against BFS on the P-induced subgraph.
  - Prediction: F1 is faster. So b-lite buys conformance, not speed.

**Open:**
- Whether Quack can express node-filtered variable length at all. If it cannot, that case is hypothetical.
- Whether a kernel-owned whole iterate passed into mask-risc is itself A1-clean. The falsifier calls it a pipeline breaker, but A1 says "a caller-owned bitmap is still materialisation". That is the operator's ruling, and it is the same question as (b) at smaller scope.

### Option set after Round 2 (premise gate: includes "belongs elsewhere")

- **(a) GatherChain** — strong candidate. Substrate-first, no A1 change. Covers fixed k and bounded `*1..k` over functional hops.
- **(c′) `ForeignPlane::resident` only** (R1, plus R2 refusal) — independent, closes the gap without naming a false equivalence.
- **(d′) recursion in an external kernel** — complements (a); not an alternative to it. Bounded k hands `v_{k-1}`; mask-risc fuses the last step.
- **b-lite (= R3)** — a typed, kernel-owned iterate readable by mask-risc. Only if P-SEAM or P-FILTERED shows need. It is a smaller decision than (b), but still a contract decision.
- **(b) full loop state inside mask-risc** — no case found that needs it. Held.
