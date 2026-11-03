# D-GLOBAL-WHERE P2 — report

Date: 2026-11-03. LAB ONLY. No production crate changed. Nothing here is
authorized as a design; S-INGRESS and D4 stay open.

Labels used on every number:

- **in-memory**: the P2 probe (`probe/`), data generated in memory, no storage.
- **Lance-backed**: `crates/crepe-plan-probe/examples/where_lance_arm.rs`, a
  real Lance 13 dataset on local disk. Bytes are Lance's `bytes_read`, served
  from the page cache, not from disk.
- **partial integration**: a stage joined to the next one by a given input
  instead of the real producer (the DR→Lance step, see D4).
- **FAILED-GATE**: a warm in-fold number ≥ 1000 µs. Reported, never hidden.
- **COLD/IO (reported)**: prepare, attest, storage reads. Reported, not gated
  (operator ruling 2026-10-22, CLAUDE.md scoped exception).

Answer parity is checked separately from timing: every arm's count and
per-column sums are compared to an oracle before any timing is printed.

## 1. Frozen evidence, independent of Lance

Frozen at `49f02414d`; re-checked this pass.

| probe | what it checks | result |
|---|---|---|
| `falsify_p1` | 37 falsifiers (refusals, composition laws) | 37/37 |
| `where_k` 1M / 10M | WHERE over K criteria, K distribution 34/46/14/6 | every answer equals the oracle; 353 / 648 FAILED-GATE values reported |
| DuckDB 2.0.0 parity (`duckdb/compare_where.py`) | count + EXISTS vs DuckDB, threads 1 and 4 | every DuckDB answer equals the oracle; coverage is enforced in both directions |
| `project` 1M / 10M | 64 logical properties over 32 lanes, layouts U/P/S | all answers match; refusals R1–R3 fire; 23 / 41 FAILED-GATE |
| `compose` 1M / 10M | repeated DR composition beyond 64K | all composed answers equal the oracle; all refusals fire |

Known gap: the k4/k16 sparse and mixed AND cases in `project` have oracle
count 0, so they cannot catch a wrong non-zero answer.

DuckDB timing, regular grid (39 cases), weighted by the K distribution:

| N | threads | count (ours vs DuckDB wall) | EXISTS (ours vs DuckDB wall) |
|---|---|---|---|
| 1M | 1 | 14.17× WIN | 1.27× WIN |
| 1M | 4 | 5.81× WIN | 1.73× WIN |
| 10M | 1 | 7.00× WIN | 0.43× LOSS |
| 10M | 4 | 1.96× WIN | 0.16× LOSS |

These are answer-parity runs with timings attached. They are not an
engineering-parity verdict; see §6.

The EXISTS loss at 10M has one cause, verified in code: `Terminal::Any` scans
every tile with no early exit (`mask-risc/src/exec.rs:1633`, `:1793-1794`).
DuckDB stops at the first match.

## 2. In-memory corrections (`probe/src/bin/corrections.rs`)

The original timings stay as baselines. Each correction is a separate arm on
the same fixture (`--sel 20000`); every arm matches the oracle.

**C1 — route once vs per query.** Prepare (routing) is COLD:
162,177 µs at 1M and 164,813 µs at 10M (independent of N, it depends on the
selection). After prepare, a query costs:

| Q queries | per query, re-routing each time | per query, routing once |
|---|---|---|
| 1 | 163,962 µs | 158,198 µs |
| 16 | 168,809 µs | 10,421 µs |
| 256 | 172,018 µs | 1,012 µs |
| warm, no prepare | — | 381–494 µs |

Routing once pays off at Q ≥ 2. The stale guard refuses a binding after
`replace_tile`. With the guard disabled the stale binding would be accepted,
so the guard is load-bearing.

**C2 — existing ndarray kernels vs the hand-written packed u8 compare.** There
is no ordered u8 compare over a packed u32 lane (NO-EXISTING-KERNEL):

- `lt_u8_to_mask` reads a contiguous `&[u8]`.
- Quack `Cmp` has only `EqU32` / `NeU32` / `MatchU32` and signed `I32`
  orderings.

The kernel arm composes existing words (`ternary_match_u32_to_mask`,
`mask_or_assign`, `mask_not`). It runs one ternary pass for each 2-bit prefix
the threshold needs:

| predicate | passes | 1M: hand / kernel | 10M: hand / kernel |
|---|---|---|---|
| f0 < 64 | 1 | 289 / 172 µs | 3,219 / 1,949 µs |
| f1 ≥ 64 | 1 | 312 / 161 µs | 3,257 / 2,096 µs |
| f2 ≥ 170 | 4 | 291 / 649 µs | 3,230 / 8,424 µs |
| f3 < 255 | 8 | 336 / 1,300 µs | 3,214 / 16,384 µs |

The existing kernels win when the threshold needs one pass. They lose by up to
5× when it needs several, because the cost scales with the pass count. Every
10M row is FAILED-GATE: a full 10M-row lane scan is over the µs budget on
either path. The real fix is an ordered `u8`-in-`u32` compare word. That is a
missing kernel, not missing wiring, so it is filed for the mask-risc / ndarray
owner. It was not built here.

**C3 — borrowed lane reads vs `gather_lane` copies** (sel 20,000; 80,896 B per
copy):

| threshold | 1M: copy / borrowed | 10M: copy / borrowed |
|---|---|---|
| lt=100 | 43.1 / 43.2 µs | 122.5 / 120.5 µs |
| lt=500 | 60.7 / 148.9 µs | 111.3 / 261.1 µs |
| lt=900 | 44.7 / 59.1 µs | 111.3 / 110.0 µs |

Borrowing removes the copy and its 80,896 B allocation with no loss at
selectivity 10% and 90%. At 50% the borrowed path is 2.3–2.5× slower. One
possible cause is unpredictable branches in the in-place compare; that is not
yet verified. This is filed open rather than explained.

## 3. Lance selective-read arm (`where_lance_arm`, Lance-backed)

The arm reuses the #1633 addressed read; no new read API was added. The edit
to the lab crate `crepe-plan-probe/src/addressed.rs`:

- makes the existing `ranges` / `range_mask` public;
- adds `range_tree`, so ingress bytes are measured on the tree Lance receives;
- adds `check_fresh`; `current_lane` now calls it, so there is one freshness
  rule.

The example constructs an `AddressedRelation` and calls these directly.

Fixture: `id` plus `p0..p15` as i32, written as one fragment. Each column
carries a `predicate_iri`. The physical order is permuted (`(5j+3) mod 16`)
and the physical names are not the labels. Attest asserts `_rowid == _rowaddr`
row for row: the prefilter is keyed in `_rowid` space and the lane holds
`_rowaddr`. With stable row ids that assertion would fail, and the arm would
refuse.

Arms:

| arm | what it does |
|---|---|
| A-full | full scan of WHERE + demanded columns, fold over the WHERE mask |
| B-allow / B-block | scan the WHERE columns, build a RowAddrMask from the result, masked scan of the demanded columns |
| C-exact | ids given (BLOCKED_D4) → lane ranges (`ranges`) → `range_mask` → masked scan of demanded columns only |
| C-default | as C, with `ReadPolicy::DEFAULT` (coalesce, or full scan when the ranges cover too much) |

All four arms give the same count and sums at every cell (oracle checked).

### 1M results (median of 5 warm runs, µs; bytes = Lance `bytes_read`)

Selected cells (full table: `results/lance_arm/n1000000/lance_arm.log`):

| layout | sel | k | arm | rows scanned | bytes (% of A) | mask bytes | route | ingress | Lance read |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| clustered | 0.1% | 1 | A-full | 1,000,000 | 100% | 0 | 0 | 0 | 10,944 |
| clustered | 0.1% | 1 | C-exact | 1,000 | 0.1% | 27 | 65.5 | 0.5 | 620 |
| clustered | 10% | 4 | A-full | 1,000,000 | 100% | 0 | 0 | 0 | 25,139 |
| clustered | 10% | 4 | C-exact | 100,000 | 6.8% | 47 | 6,951 | 1.6 | 2,180 |
| scattered | 0.1% | 1 | C-exact | 988 | 33.3% | 2,124 | 331 | 49.1 | 1,631 |
| scattered | 0.1% | 16 | C-exact | 988 | 99.8% | 2,124 | 487 | 50.3 | 12,930 |
| scattered | 10% | 4 | A-full | 1,000,000 | 100% | 0 | 0 | 0 | 25,411 |
| scattered | 10% | 4 | B-allow | 1,100,164 | 120% | 126,420 | 0 | 3,201 | 67,939 |
| scattered | 10% | 4 | C-exact | 100,164 | 80% | 126,420 | 9,533 | 4,593 | 54,872 |
| scattered | 10% | 16 | C-exact | 100,164 | 100% | 126,420 | 9,136 | 4,572 | 222,968 |
| scattered | 50% | 16 | A-full | 1,000,000 | 100% | 0 | 0 | 0 | 83,343 |
| scattered | 50% | 16 | C-exact | 500,613 | 100% | 377,194 | 43,569 | 10,888 | 945,373 |

What this shows:

1. **The mask always cuts rows scanned** to exactly the selection (C) or
   n + selection (B, which first scans the WHERE columns). Filtering is not
   missing anywhere.
2. **Bytes follow page locality, not selectivity.** Clustered selections read
   0.1–9% of the full scan's bytes. A scattered selection of 0.1% already
   reads 33% of the bytes at k=1 and 99.8% at k=16. Every page holding at
   least one survivor is read whole. This repeats H14c/H14d (#1573/#1575) on
   the #1633 path. It is a physical I/O limit, not a wiring defect.
3. **On scattered selections a masked read is slower than a full scan.**
   - At 10% and k=16, C-exact takes 222,968 µs against A's 88,913 µs; B takes
     255,226 µs.
   - At 50% and k=16, C-exact takes 945,373 µs against A's 83,343 µs.
   - Across scattered 10–50%, C-exact is 1.4–11× slower than A, and B is
     2.4–11× slower.
   - `ReadPolicy::DEFAULT` sees this coming and falls back to a full read on
     every scattered 10–50% cell. That is the #1489 "cost choice", already
     implemented. It still pays for the route stage before deciding.
4. **Mask ingress is a copy, and its cost scales with the number of ranges,
   not rows.**
   - One clustered range: 27–144 B of tree, 0.4–5.4 µs.
   - 91,527 scattered ranges: 126,420 B, ~4,600 µs.
   - 326,222 ranges: 377,194 B, ~9,400–10,900 µs.

   Timing covers building the one tree Lance receives. The serialized size is
   measured outside the timer. The first run of this report timed two tree
   builds per C read and doubled the C ingress figures (Codex finding on
   #1950); every number here is from the corrected code.

   It is never zero-copy: `insert_range` builds a `RowAddrTreeMap` that Lance
   owns (S-INGRESS).
5. **Route (ids → lane ranges by galloping seek) is the dominant non-I/O
   cost.** It is about 70–95 ns per id: 6,900–9,500 µs per 100k ids and
   34,000–44,000 µs per 500k. It is FAILED-GATE on every cell ≥ 10k ids. It is linear in the
   number of ids because each id is sought separately.

### 10M results (median of 3 warm runs, µs)

Every oracle and refusal check passes. Attest is COLD: 809,110 µs, once per
version. Full table: `results/lance_arm/n10000000/lance_arm.log`.

| layout | sel | k | arm | rows scanned | bytes (% of A) | mask bytes | route | ingress | Lance read |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| clustered | 0.1% | 1 | A-full | 10,000,000 | 100% | 0 | 0 | 0 | 133,908 |
| clustered | 0.1% | 1 | C-exact | 10,000 | 0.04% | 27 | 758 | 1.3 | 1,196 |
| clustered | 10% | 4 | A-full | 10,000,000 | 100% | 0 | 0 | 0 | 284,073 |
| clustered | 10% | 4 | C-exact | 1,000,000 | 6.4% | 242 | 73,993 | 11.6 | 16,348 |
| clustered | 10% | 4 | C-default | 10,000,000 | 64.3% | 0 | 71,509 | 0.1 | 319,741 |
| scattered | 0.1% | 1 | C-exact | 9,858 | 33.3% | 20,960 | 4,702 | 583 | 10,024 |
| scattered | 0.1% | 16 | A-full | 10,000,000 | 100% | 0 | 0 | 0 | 1,001,637 |
| scattered | 0.1% | 16 | C-exact | 9,858 | 100% | 20,960 | 4,721 | 560 | 126,264 |
| scattered | 10% | 4 | C-exact | 1,001,669 | 80% | 1,317,366 | 96,773 | 57,455 | 773,128 |
| scattered | 10% | 16 | C-exact | 1,001,669 | 100% | 1,317,366 | 98,639 | 55,939 | 2,814,172 |
| scattered | 50% | 16 | A-full | 10,000,000 | 100% | 0 | 0 | 0 | 1,029,055 |
| scattered | 50% | 16 | C-exact | 5,006,105 | 100% | 4,776,680 | 416,595 | 123,162 | 11,019,172 |

The 1M pattern holds at 10M, and grows:

- A clustered 0.1% read is 112× faster than the full scan.
- The worst scattered read (50%, k=16) is 11× slower than the full scan.
- Even a 0.1% scattered selection at k=16 reads 100% of the bytes, yet it
  still beats the full scan 7.9× on time, because it decodes 9,858 rows
  instead of 10M.
- Ingress for 3.26M ranges is a 4.8 MB tree and 123,162 µs.

**New: `ReadPolicy::DEFAULT` misroutes clustered demand.** `plan_read`
(`addressed.rs:212-232`) decides on covered rows alone (`covered*1000 >=
n*full_permille`, 10%). It never looks at how many ranges there are, and its
own doc says "Clustered demand … is not measured". A single contiguous 10%
range is therefore sent to a full scan:

| N | C-default | C-exact | penalty |
|---|---:|---:|---:|
| 1M | 35,606 µs | 9,310 µs | 3.8× |
| 10M | 428,883 µs | 93,994 µs | 4.6× |

The scattered cells, where the fallback is right, have tens of thousands to
millions of ranges. Range count, not coverage, separates the two. This is
recorded as a probe item and the policy is not changed. It belongs to the
#1633 owner.

A defect found on the way. Lance 13 compares the file row limit as
`params.max_rows_per_file as u32` (`lance-13.0.0/src/dataset/write.rs:1077`).
The fixture set `1 << 40`, which truncates to 0. Every input batch therefore
closed a file, and the 10M table came out in many fragments. The attest guard
refused it as `NotContiguous` (log kept: `results/lance_arm/n10000000/notcontiguous.log`).
That is the #1633 refusal working as designed. The fixture now uses
`u32::MAX`. This upstream truncation is recorded here and not patched.

## 4. ClassView → WideFieldMask → physical columns (Lance-backed)

`resolve()` maps each demanded field position through:

1. `ClassView::fields(class)` → `FieldRef::predicate_iri`, then
2. the Lance schema field metadata `predicate_iri`, to the physical column name.

It never maps position to ordinal or label to column name. Two traps prove a
shortcut would be caught:

- **Ordinal shortcut:** mapping by ordinal gives the wrong column on 16 of 16
  positions.
- **Label shortcut:** mapping by label finds no physical column on 16 of 16.

Each refusal below is paired with a success control:

| refusal | trigger |
|---|---|
| AbsentField | the position is not in the presence `FieldMask` |
| Unauthorized | the position is outside the caller's `WideFieldMask` |
| UnknownProperty | no column carries the IRI |
| AmbiguousProperty | two columns carry the IRI (a separate dataset) |
| StaleSchema | the dataset was appended after resolution |

A refusal returns no column list, so the column list cannot widen. Projected
bytes follow k exactly at A-full: 3.8 MB (k=1), 6.3 MB (k=4), 20.3 MB (k=16).
C reads only the resolved columns.

Kept distinct throughout:

| mask | answers |
|---|---|
| WHERE mask | which rows pass |
| `WideFieldMask` | which fields are demanded or permitted |
| DR mask | which identities are selected |
| Lance `RowAddrMask` | which row addresses to read |

The code never passes one where another is expected.

## 5. D4 / S-INGRESS boundary

`BLOCKED_D4`: no type binds a DR `Frame` to a Lance dataset version.

- `lance-graph-dt64` has no Lance dependency.
- The #1788 single-authority decision is open.

So arm C takes its ids as a given input, the oracle's sorted identities. It
never composes a DR ordinal with `_rowid`. No third coordinate map was built.

The freshness checks that do exist are #1633's own, now one function:

- a stale lane is refused after an append;
- a lane read against another dataset is refused;
- a forged newer-version witness is refused by `seek_from`.

RowAddrMask construction is measured, not assumed free; see §3 point 4.

## 6. Cost columns

One row per stage for three representative 1M cells, k=4, µs. Labels per
stage: **IM** in-memory, **LB** Lance-backed, **PI** partial integration.

| cell | WHERE evaluation | ClassView resolution | DR routing | mask ingress | Lance read | projection | FOLD | total |
|---|---|---|---|---|---|---|---|---|
| clustered 10%, C-exact | 0 (ids given, PI) | 10.3 LB | 6,951 IM (lane seek; DR→Lance BLOCKED_D4) | 1.6 LB | 2,180 LB | inside Lance read; 6.8% of A's bytes LB | 64.6 LB | 9,310 PI |
| scattered 10%, C-exact | 0 (PI) | 15.7 LB | 9,533 IM | 4,593 LB | 54,872 LB | inside Lance read; 80% of A's bytes LB | 65.9 LB | 72,587 PI |
| scattered 10%, A-full | 790 LB (over loaded columns) | 14.4 LB | 0 | 0 | 25,411 LB | inside Lance read; 100% LB | 2,066 LB | 29,432 LB |

Not measured in the same cell, because the DR→Lance edge is BLOCKED_D4:

- in-memory DR WHERE evaluation over resident planes (`where_k`): 138–1,530 µs
  at 1M;
- DR routing per composition (`compose`, route column): 35,914 µs out of a
  50,178 µs total at sel 50k, D=1.

A full end-to-end row, DR selection → Lance read in one process, does not
exist yet. It is the integrated path the #1788 register already lists as
unmeasured.

## 7. Three parity verdicts

**Semantic parity: REACHED.**

- Every count and EXISTS answer equals the oracle and DuckDB at 1M and 10M.
- Every Lance arm equals the oracle and the full-projection oracle at 1M and
  10M.
- Three-valued NULL semantics match.
- Refusals fire and are paired with controls.

**Performance parity: SPLIT.**

| operation | ours vs DuckDB | status |
|---|---|---|
| count | 14.2× (1M, 1 thread) down to 1.96× (10M, 4 threads) | WIN |
| EXISTS | 0.43× / 0.16× at 10M | LOSS: no early exit |

On storage, a clustered selective read beats the full scan by 11.5–17.6× at
1M and 17–112× at 10M. A scattered one at 10–50% loses to the full scan by
1.4–11×, and the existing
`ReadPolicy` already knows to fall back.

**Engineering parity: NOT REACHED on strategy, reached on representation**
(`ENGINEERING-PARITY.md`; source-backed Lance trace: `LANCE-TRACE.md`):

- **Selection reuse: ALREADY WORKS.** A versioned DR mask lives longer than
  DuckDB's per-chunk selection vector.
- **Chunk compaction: INTENTIONALLY DIFFERENT.** Word folds have no underfull
  chunks.
- **Late materialization: INTENTIONALLY DIFFERENT.**
- **Evaluation direction: MISSING CAPABILITY.**
- **EXISTS early exit: MISSING WIRING.**
- **Physical-read avoidance: MISSING** the single `_rowid` authority and
  bind-time constant folding.
- **Routing: MISSING WIRING.** Route once, then reuse (C1 shows Q ≥ 2 pays).

WHEN vs WHETHER: DuckDB optimizes *when* a property is materialized. We can
skip the read entirely for:

- COUNT / EXISTS / ALL over a bound plane;
- presence questions;
- empty demand;
- DISTINCT over a sealed map.

We cannot skip:

- the first evaluation of a value predicate;
- payload aggregates;
- non-resident sources;
- a generation change (502–758 µs).

## 8. Verdict

Format per item: producer → missing edge → consumer → test → measured cost →
owner → PR.

**ALREADY WORKS**

1. Lance 13 masked read with column projection.
   - Path: #1633 `AddressedRelation` → `range_mask` → `with_row_addr_prefilter` + `project` → fold.
   - Test: `where_lance_arm`, all oracle checks.
   - Cost: clustered 0.1% = 1,176 µs read and 0.1% of the full scan's bytes.
   - Owner: lab. PR: this one.
2. `ReadPolicy` cost choice.
   - Path: `plan_read` → `Read::Full` on scattered 10–50%.
   - Test: C-default cells.
   - Cost: avoids the 1.4–11× scattered-read loss. On clustered demand it
     picks wrongly; see item 7.
   - Owner: lab #1633.

**WIRE NOW** (lab only, done here)

3. ClassView / WideFieldMask → physical Lance columns.
   - Path: `ClassView::fields` → `predicate_iri` → schema metadata → `Scanner::project`.
   - Test: 5 refusals + 2 traps + projected bytes.
   - Cost: resolve 7–40 µs.
   - Owner: lab, this PR. Production wiring needs an owner decision; not done.

**PROBE**

4. EXISTS early exit.
   - Path: `Terminal::Any` → break per tile (`exec.rs:1793`) → Quack EXISTS.
   - Test: the 10M EXISTS cells, plus a can-stay-silent test on an empty mask.
   - Cost: today 0.16× DuckDB at 10M.
   - Owner: mask-risc. PR: separate, not opened.
5. Ordered u8-in-u32 compare word.
   - Path: ndarray / mask-risc → `Pred` → the `project` layouts.
   - Test: C2.
   - Cost: up to 5× slower than the hand loop when composed from existing words.
   - Owner: ndarray / mask-risc. Not built.
6. Route once, reuse per frame.
   - Path: P1 arm C index → a frame-scoped binding → the compose route stage.
   - Test: C1.
   - Cost: 158,198 → 1,012 µs per query at Q=256.
   - Owner: DRF / D-GLOBAL-WHERE.

7. `ReadPolicy` decides on coverage only.
   - Path: `plan_read` → also weigh the range count → C-default.
   - Test: the clustered 10% cells.
   - Cost: today 3.8× (1M) and 4.6× (10M) slower than the exact read.
   - Owner: lab #1633. Not changed here.

**OWNER DECISION**

8. One version-checked DR → `_rowid` authority, plus bulk ingress.
   - Path: dt64 `Frame` → ONE witness (DR route version + Lance `ds.version()` + uri) → `RowAddrTreeMap`.
   - Test: the C arm, with ids from DR instead of given.
   - Cost: ingress 0.4–123,162 µs, measured.
   - Owner: S-INGRESS / D4. BLOCKED_D4.

**PARK**

9. Scattered selective reads on a row-major page layout.
   - Physical limit: bytes follow page locality, not selectivity.
   - Not a code change.
   - Revisit with a clustered layout or a different page size.

**SKIP**

10. A new mask store, trie, hash table or paging engine. Nothing measured here
   needs one.

## Not verified

- The cause of the 50% slowdown on the borrowed path (C3).
- Disk reads: every Lance byte count came from the page cache.
- DuckDB internals: the source is not on disk (see `ENGINEERING-PARITY.md`).
