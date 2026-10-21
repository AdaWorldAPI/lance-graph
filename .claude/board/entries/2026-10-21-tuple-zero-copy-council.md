# 2026-10-21 — 5+3 council: the tuple is a zero-copy violation (blast radius)

**Status:** FINDING. A policy ruling plus the classified sites it reaches. Nothing is implemented here. D-RU-6 stays Queued/Lab. This entry changes nothing in mask-risc and proposes no new op; any new op needs a law-6 insufficiency proof first. EXO-103 to EXO-118 are the follow-up issues on the Linear coordination board (EXO-50).

## The model

PowerShell `"{0}..{15}" -f $lane.split`: a format template names the positions of one value and reads them in place; it builds nothing. A ClassView carving (`6×(u8:u8)`, `4×(u8:u8:u8)`, `3×(u8:u8:u8:u8)`) is that template over the 16-byte facet. A fold acts on all positions at once (one compare → one mask). A **tuple** is the value split into owned, row-shaped value objects.

## Decisions (policy, not derived facts)

- **F1** (DECISION, lance-graph `CLAUDE.md` P0): no millisecond result anywhere. A single operation runs in nanoseconds, a batch in at most microseconds.
- **F9** (DECISION, operator 2026-10-21, `CLAUDE.md` P0): a tuple, meaning an owned row-shaped copy of lane values, is a zero-copy violation. That holds as an intermediate and as a terminal, on any production or lab path, at any speed. F1 and F9 are separate grounds: copying breaks F9, and the per-row alloc/clone/sort separately breaks F1.
- F3 and F7 are read through F9: a terminal's elected or demanded output is **coordinates** (a kept mask, `materialize*` row ids with O(n) stated), never value rows. Basis: `folding-doctrine.md` §1 and §5; `mask-risc/src/ir.rs` doc on `ScatterOrU32`: "the scattered mask is legal only when it IS the externally demanded result … never an intermediate".
- mask-risc is the reference folding algebra and is frozen until new algebra proves itself in ns.

## What is legal and what is not

Named so no leg collapses into another:

| leg | status |
|---|---|
| owned row-shaped value copy (`Vec<Vec<_>>`, `Vec<(Vec<_>,_)>`, tuple sets of values) | **violation** (F9) |
| row-id population consumed by a further fold, filter, count or join | **violation** (F7) |
| coordinates out (kept mask, `materialize*` row ids) | legal |
| borrowed lens: an iterator or callback yielding coordinates or borrowed lane views, never owned values | legal |
| mask/word scratch and transient tile routing (`folding-doctrine.md` §5) | legal; NOT a tuple |
| fold of multiplicities (count(A×B)=count(A)·count(B), law 7) | legal |
| enumeration in an independent test oracle (not the code it checks) | legal only there |
| columnar lane copy at a cross-process or serialization membrane | **OPEN** (operator) |

## Resolution

- **R1.** In non-test code, no row-of-values or row-id population feeds a further fold, filter, count or join.
- **R2.** No terminal returns copied row values; scalars (Count, Sum, Min, Max) and coordinates are fine. When a demand genuinely returns rows, the kept mask or row ids plus lane descriptors cross, and the consumer reads the lanes in place, or a borrowed lens yields coordinates or borrowed views.
- **R2a.** No genuine row-demand gate exists yet. In cypher-quack `demand.rs`, classify fails closed to `Demand::Bindings` on unrecognised shapes. Bindings is also returned for non-row shapes, including `count(DISTINCT c)` with a cross-variable filter. These were observed at those sites; no closed search was run. So Bindings means "refused" today, not "rows requested". Refusal stays the default (`cypher-quack-seam.md`, CURRENT-CONTRACT). → EXO-106.
- **R3.** Enumeration is licensed only as an independent test oracle (`zero-copy-lens-law.md`: old bodies kept as `#[cfg(test)]` oracles; coresearch shared-oracle rule). Gate it with `#[cfg(test)]` where possible. Where tests/ and examples/ link the normal library, as they currently do in crepe-plan-probe and mork-pathmap-probe, use a named-exemption guard in the style of mask-risc's `exactly_one_materialiser` (`mask-risc/src/exec.rs`). Searched: crepe-plan-probe `src/`; `tuples_over_mask` carries no cfg gate. mask-risc's public `reference_execute` / `reference_scratch` (`lib.rs` re-exports) is an operator decision only. → EXO-105, EXO-109.
- **R4.** UPHELD for count, exists and total multiplicity: these need no tuple.
  - `count(A×B)=count(A)·count(B)` is already implemented for non-emitting sides at `crepe-plan-probe/src/native.rs:446-449`.
  - Distinct across factors uses `ScatterCountU32` (`ir.rs:364`), or `CountKeyRunsU32` on a key-ordered lane (`ir.rs:389`).
  - OPEN for three cases (law 6: recorded, not designed; → EXO-108):
    - GAP-A1, scalar distinct count over a projection;
    - GAP-A2, set-semantics distinct output (a representative kept mask);
    - GAP-B, bag count over emitting sides, the sum over set slots of the product of per-side fan-outs (only one-lane `MaskedSumI32` exists).
- **R5.** crepe-plan-probe (lab):
  - `cse.rs:393-424` and `pipeline.rs:189-219` answer Exists / Count(Set) / Count(Bag) from a tuple `BTreeMap`. This breaks R1 and R4 inside the lab. → EXO-103, EXO-116.
  - `native.rs:420-479` (`tuples_over_mask`) is a violation in full, on four levels (see the correction below). No half of it is kept. → EXO-104.
  - `NativeAnswer` (`native.rs:338`) has no reference outside its definition (searched: `crates/**/*.rs`).
  - Nothing here is promoted out of the lab.

## Replacements that exist (reuse, law 6)

| need | mechanism |
|---|---|
| count / exists | mask-risc `Terminal::Count` / `Any`; relalg `Measure::Count` |
| total multiplicity | law 7 product; mork-pathmap-probe `JoinExec::bag_count` (a scalar per-slot loop, not a mask-risc terminal) |
| distinct across factors | `ScatterCountU32` (`ir.rs:364`); `CountKeyRunsU32` (`ir.rs:389`, key-ordered lane) |
| coordinates out | `Terminal::Keep` (`ir.rs:321`); `materialize_rows` (mask-risc `exec.rs`, O(n) stated) |
| coordinates-first ordering, precedent | lance-graph-report `render.rs` (Keep → row ids → lanes). Its final hydration into values is itself on the follow-up list. |
| borrowed-lens shape, precedent | mork-pathmap-probe `for_each_answer` (2 sides only); lance-graph-report `exec.rs` `for_each_tuple` |

None found in the crates searched: a public streaming set-bit iterator (crepe-plan-probe calls a local `for_each_set`), a multi-side multiplicity lens, and fan-out for multi-valued emitting sides. → EXO-107.

## Blast radius

The sweep covered non-test `src/` of every crate under `crates/` (548 pattern hits, 73 crates). Rows marked *read* were read in context; *hit-depth* rows were classified from the hit line and its surroundings. This is not a proof of absence.

**Mandatory: crepe-plan-probe only** (3–5 src, 1 test, 5–8 example files). Its sites are listed under R5. The oracles are `reference.rs` and `lib.rs`. Examples time Count(Bag) through the tuple map (`examples/addressed.rs`, `addressed_read.rs`) and time oracle decode (walk_*, measure). Consumers searched: no `Cargo.toml` under `/home/user/*` depends on it, and no `.rs` file in `crates/` names `crepe_plan_probe` outside it.

**No tuple site found by the sweep** (read) in mask-risc, relalg, quack, cypher-quack and lane-fold. Their `Vec<Vec<u64>>` uses are mask scratch. mask-risc's public reference executor is a licensed row-at-a-time oracle (R3). cypher-quack refuses Bindings.

**Follow-up**, hit-depth unless marked:
- **lance-graph-dir-sim `store.rs` `effective`** (read): Keep mask → `materialize_rows` → `BTreeSet<(Guid128,Guid128)>` → set difference. That is a row population feeding a set operation, in a non-test library path; whether a shipped binary reaches it was not checked. → EXO-117.
  - **Priority inversion.** This is the only non-lab site, yet it is follow-up only. It is outside the lab refactor because versions have different row spaces plus an overlay, so mask AndNot does not apply directly. Its priority should be decided against the lab work, not after it.
  - **OPEN (operator):** is a `(Guid,Guid)` pair set a coordinate set or a value tuple?
- **mork-pathmap-probe:** `join.rs` `answers` → `BTreeMap<Row,u32>` (lab); `rmap.rs` edge/escape sets.
- **lance-graph-report:** `result.rs` pivot `keys()` → `Vec<Vec<u32>>` (flat_map-clone shape).
- **lance-graph-contract:** `graph_render.rs` `CypherExecutor` returns `Vec<Vec<CypherValue>>`. No implementor was found in this workspace or in the sibling repos checked: a dormant cross-repo surface.
- **legacy:** lance-graph-cognitive `cypher_bridge.rs`; lance-graph core `graph/versioned.rs` edge-set difference and arigraph paths; holograph `query/executor.rs`; lance-graph-planner `adjacency/` (mostly build scratch).

→ EXO-118. Eight more crepe-mentioning files were not read line by line.

**Board/docs carrying ms-denominated figures from the tuple paths.** These get an append-only correction note, never a rewrite. The tuple-vs-fold timings stay as dated comparison evidence:
- `entries/2026-10-21-u3-cross-rule-cse.md`;
- `entries/2026-10-21-coresearch-ms-vs-ns-folding.md`;
- `entries/2026-10-20-crepe-rule-optimizer-spike.md`;
- `plans/crepe-rule-optimizer-spike-v1.md`;
- STATUS_BOARD D-RU-3 and D-RU-6;
- the three `entries/2026-10-21-d-ru-6-*.md`.

→ EXO-110.

## Rejected and narrowed objections (kept on record)

- **"Someone pays the split at the membrane anyway."** Rejected in process: in the target design the consumer reads lanes in place. Narrowed across processes: at a serialization membrane (wire clients) lane bytes physically move. Columnar lane transport, never row tuples, is the only candidate shape there, and whether it is an F9 exception is OPEN for the operator.
- **"A lens terminal is legal"** (v1). Narrowed: legal only when it yields coordinates or borrowed views. A lens that yields owned values is a violation.
- **"A tuple is a violation because it is slow."** Superseded: it is a violation because it copies. Slowness is a separate ground (F1).

## Gates

- **G1:** every R has a verdict with file:line.
- **G2:** H answered: ruled by F9 for owned value copies; legal legs named in the table.
- **G3:** every site classified, with read depth stated; 8 files unread.
- **G4:** this entry quotes no timing; board figures are named by file only.
- **G5:** each violation names an existing replacement or a GAP (A1, A2, B, the set-bit iterator, the multi-side lens).
- **G6:** no replacement yields owned values.

## Council run

| phase | who / what |
|---|---|
| 5 savants (Sonnet; S4 on Opus) | prior art, iron rules + fold laws, code truth, cascade / blast radius, different views |
| 3 reviewers (Sonnet) | overclaim-auditor (2 BLOCK: absolute wording in §2 and §6, both re-worded and re-graded); dilution-collapse-sentinel (0 BLOCK; borrowed-lens and membrane legs restored); firewall-warden (0 BLOCK, all non-negotiables clean) |

v1 → v2 changes: F9 added; R2 replaced; R2a, R3 independence and the gaps added; blast radius filled in.

v2 → v3 changes:
- **Wording:** absolutes scoped; F9 defined as owned row-shaped value copies.
- **Legal legs:** borrowed lens and the membrane question restored; legal legs tabled. (v3 R5 said "emitting half"; corrected below.)
- **Gaps:** GAP-A split into A1 and A2.
- **Evidence:** citations corrected (`ir.rs:364`, `:389`); ms figures named by file only; scratch-file citations removed; dir-sim priority inversion flagged.
- **Corrections:** board ms figures get append-only notes, never rewrites.

## Correction (operator, 2026-10-21) — the tuple violates on four levels; no half is kept

v3 said `tuples_over_mask` was a violation "in its emitting half" and that its count multiplication "survives as a fold". That was wrong. The function is a violation in full, on four levels:

1. **Zero copy.** It copies lane values into owned rows, a second owner of substrate bytes (F9).
2. **Folding.** It enumerates where a fold belongs. Count, exists and multiplicity are answered by decoding rows, not by folding mask words into a terminal.
3. **Natural order.** The substrate already holds its rows in a natural order: slot order of the mask and the lanes, readable in place at zero copy. `out.sort()` plus merge discards that order and imposes a value order on decoded copies, so the substrate's own order is ignored. It also costs O(n log n). Order belongs to the substrate's coordinates, never to decoded rows.
4. **Time.** The per-row alloc, clone and sort is what produced the millisecond results (F1).

The multiplication of side counts at `native.rs:446-449` is not a part of the tuple builder to keep. The fold it computes (law 7, count(A×B)=count(A)·count(B)) exists on its own. It belongs at the terminal, over the kept mask in natural order, not inside a function that builds rows. "The fold-shaped half can be kept" was a collapse in the other direction: it filed a valid law under an invalid carrier.

## Correction (operator, 2026-10-21): no boundaries

The legal-legs table above is superseded where it allows anything to cross a
boundary. There are no boundaries; the whole substrate is zero copy.

- Struck: "coordinates out … legal", "borrowed lens … legal" as a
  boundary leg, and the OPEN row "columnar lane copy at a cross-process or
  serialization membrane". There is no membrane at which values are copied.
- The only "exception" is the inversion, global to local, and it is still
  zero copy: it materializes the pointer table `$i` and nothing else.
- A "tuple" is a zero-copy fold template: `"{0}{1}" -f $laneA[$i],$laneB[$i]`.
  A `(Guid, Guid)` pair held as a value (dir-sim, EXO-117) is the violation;
  two lanes read at `$i` is the correct shape.
- Oracles included: tests compare answers by folds through the template
  (crepe-plan-probe `Answers::{total, matching, for_each}`), never by
  expanding rows.
- Open, found by the same rule: `mork_pathmap_probe::join::Side::prepare`
  copies the non-key column into `vals`. Under this ruling it should hold
  only `$i` (row positions per slot) and read `rows[$i]` in place. Its
  `prepare` stage fails the no-ms gate at n = 10 000 and 50 000 (the
  `no-semijoin` and `none` arms of `crepe-plan-probe` example `measure`).

