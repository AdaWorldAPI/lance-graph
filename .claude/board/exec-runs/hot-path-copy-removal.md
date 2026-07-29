# Hot-path copy removal — what was done, what was removed (2026-07-29)

Operator order: *"copies are forbidden, borrows are only for the same mailbox"*,
*"only cognitive achievements > tenant"*, *"we're talking about copy in hot
path"*, *"fix"*.

Commits: `b3515ba` (WitnessLens), `eba7ed6` (7-agent Copy census),
`12a23d3` (ontology carve-out), `0310122` (hot-path allocations).

---

## 1. What was WRONG (stated plainly)

PR #868 changed the `window:` parameter of 7 `meta_basin` functions from a
gathered `&[(usize, CausalWitnessFacet)]` to a `WitnessLens`, and its body
claimed **"the materializing path is GONE."** That claim was false. The input
parameter changed; the copy machinery *below* it was never looked at.

Two distinct defects, both mine, both authored the same day:

| # | defect | where | status |
|---|---|---|---|
| A | `WitnessLens` derived `Clone, Copy` — a borrow that duplicates silently and can leave its mailbox | `witness_fabric.rs` (introduced `df69d87`, THIS session) | FIXED `b3515ba` |
| B | `GradedRow` materialised and re-copied through the whole pipeline | `meta_basin.rs` | FIXED `0310122` (partially — see §4) |

Defect A is the one the operator had to name twice. It was created hours earlier
under a task titled *"`#[repr(transparent)]` + borrowed view — make the cast
real"* — i.e. the zero-copy fix itself shipped the copy.

---

## 2. What was REMOVED

### `WitnessLens` — `#[derive(Clone, Copy)]` (b3515ba)
A `Copy` borrow can be stored beside the original and carried out of the
compartment owning the rows with **no move and nothing in a diff to point at**.
That is exactly how a same-mailbox borrow escapes its mailbox. Removing the
derive forces pass-by-reference; reach is now bounded by the borrow it was built
from. Contract gates clean with it gone — nothing was relying on duplicating it.

### The head-of-chain allocation (0310122)
`grade_rows` returned `Vec<GradedRow>` → now a **lazy iterator**. Grading is a
projection over addresses; there was never anything to accumulate.

### The second, redundant allocation (0310122)
`tail(graded: &[GradedRow]) -> Vec<GradedRow>` did `.iter().copied().collect()`
— a full second copy of rows that had just been built. It now consumes the
grading iterator directly. **The chain allocates ONCE**, at the clustering
boundary.

### The per-probe re-grading allocation (0310122)
`stable_under_perturbation` rebuilt a whole `Vec<GradedRow>` inline, and
`stability_sweep` calls it **once per budget** — 11 allocations per basin on the
default range. Replaced by `grade_shapes_at`, which is lazy **and** skips the
quorum sweep entirely: `meta_cluster` reads only `.trajectory`, so the expensive
half was being computed and discarded on every probe.

### A redundant `#[must_use]` (0310122)
The returned iterator already carries it; clippy `-D warnings` caught the
duplicate.

### 8 `derive(Clone, Copy)` on borrow-carrying types (eba7ed6)
Across contract / callcenter / cognitive / holograph / planner / deepnsm, from
the 7-agent census.

---

## 3. What was DELIBERATELY NOT removed

- **`Copy` on value types.** `.claude/rules/data-flow.md` §2 *requires* it for
  reasoning microcopies (`TruthValue`, `Fingerprint`, `u64`, `Band`, `CpuCaps`,
  `ScanParams`). Stripping those breaks the law from the other side. The census
  ruled 26 LEGITIMATE on exactly this ground.
- **13 ELEVATED sites** — a value at a strictly higher rung than every input it
  derives from earns its store (`Locus::Quorum` / `Contradiction` precedent).
- **`lance-graph-ontology`** — operator-carved: *"copying ontology is desired,
  RDF is converted to KV."* RDF is an external format at a membrane; the
  hydrators' copy is the FIRST projection, not a second one. Banning it would
  forbid ingest. See the carve-out in `copy-derive-blast-radius.txt`.
- **The one remaining collect at the clustering boundary.** `meta_cluster` and
  `density_scores` need random access over the tail; that is a genuine
  accumulation, not a projection.

---

## 4. What is STILL a copy (honest remainder)

`MetaBasin.members: Vec<GradedRow>` and `MiniBasin.members: Vec<GradedRow>`
still hold **copies of rows that already exist in the tail buffer**. The
correct shape is indices into that buffer. Not done here: it cascades through
`meta_cluster` / `mini_basins` / `coarse_flags` and every test fixture, and
after today's record I am not landing a refactor that size unreviewed in the
same pass as the fix.

`reasoning.rs` has `PremiseBundle.premises: Vec<..>` and
`arena(&self) -> BeliefArena`.

> **⊘ SELF-CORRECTION (same session, before acting on it).** An earlier revision
> of this line called `arena()` "a per-call materialisation" and listed it as a
> copy to remove. **That was wrong, and it was the same reflex that nearly
> mis-flagged `lance-graph-ontology`:** seeing an owned return value and reading
> it as a duplication without asking what it derives from.
>
> `arena()` is a **fold**, not a copy. It takes premises and produces pooled
> beliefs with NARS-revised truth — `TruthValue::revise` on disjoint stamps,
> CHOICE on overlapping ones. The output is at a strictly higher rung than every
> input: no premise carries pooled confidence; the arena is where that comes into
> existence. That is exactly the ELEVATED carve-out (`Locus::Quorum` /
> `Contradiction` precedent) and it EARNS its store. Removing it would delete the
> reasoning, not a duplication.
>
> **The real finding, which is smaller and different:** `resolve()` (`:176`) and
> `differential()` (`:208`) each call `self.arena()`, so a caller that does both
> — which is the expected consumer shape — folds the same premises **twice**.
> That is redundant computation, not a copy. `.claude/rules/data-flow.md`
> permits the fix (*"caches use interior mutability (`RwLock`, `LazyLock`) or are
> built once"*), so a `OnceCell<BeliefArena>` is in-doctrine. NOT done here: it
> changes a public struct's shape, and it is a different category of defect from
> the one this record is about. Filed, not smuggled in.

`PremiseBundle.premises` remains a stored `Vec` and is genuinely unexamined.

---

## 5. Measurements, not claims

- Sweep path: **2 allocations per graded row → 1**.
- Perturbation probe: **11 allocations per basin → 0** on the default budget
  range, plus the quorum computation dropped from every probe.
- Scan cost characterised separately (Codex P2): **4608 `visible` probes at
  N=512/k=8 vs 64 peer comparisons gathered** — the Θ(N·k) predicate form. That
  is NOT fixed here; the operator's ruling is that the peer domain should be the
  **address list**, which bounds it by k. Tracked in
  `TD-LENS-QUORUM-SCANS-THE-WHOLE-LENS` with the corrected framing.

## 6. Gates

`cargo test -p lance-graph-planner` **325 passed / 0 failed** ·
`clippy --all-targets -- -D warnings` clean · `cargo fmt --check` clean ·
`cargo test -p lance-graph-contract --lib` 1134 passed.

## 7. Standing status

**PR #868 remains draft / DO-NOT-MERGE.** The hot-path allocations are fixed;
the `MetaBasin.members` copies and the `arena()` materialisation are not, and
the PR's original "materializing path is GONE" claim is retracted in its body.

---

## 8. HOW it was redone without the copy — and why the result is BETTER, not merely equal

The instinct when told "remove the copy" is to fear a trade: fewer allocations,
more indirection, same work. That is not what happened. In every case the copy
was **carrying nothing**, so removing it removed work rather than moving it.

### 8.1 `grade_rows` — the copy was a materialised map

**Before.** `(0..len).filter(visible).enumerate().map(grade).collect()` — build
every `GradedRow`, put them in a heap buffer, hand the buffer back. The caller
then walked the buffer.

**After.** Identical expression, `.collect()` deleted, return type
`impl Iterator<Item = GradedRow> + 'a`. The `map` closure is unchanged.

**Why better, not equal.** A `map` over a range IS the projection; `collect` was
adding a heap buffer between two loops that were already fused. Removing it
gives: no allocation, no second pass, and — the real win — **grading now happens
lazily, so a consumer that short-circuits never pays for the rows it does not
reach.** `tail` filters on `quorum <= tail_below`; under the old shape every row
was fully graded (quorum sweep AND chain walk) before the first was tested.

**The lifetime is the whole trick.** `grade_rows<'a>(lens: &'a …, visible: &'a impl Fn…)`
with `move` on both closures ties the iterator to the borrows it reads. No
`Copy`, no clone, no owned capture — the iterator is a *description* of work
over borrowed state, which is exactly the lens argument one level up.

### 8.2 `tail` — the copy was a filter that reallocated

**Before.** `tail(graded: &[GradedRow], …) -> Vec<GradedRow>` doing
`.iter().copied().filter(..).collect()` — a full second buffer, from a buffer
built one line earlier.

**After.** `tail(graded: impl IntoIterator<Item = GradedRow>, …) -> Vec<GradedRow>`.

**Why better.** Taking `IntoIterator` instead of `&[T]` means the filter fuses
into the grading iterator: **one pass, one allocation, for what was two of each.**
And it is strictly more general — a slice still satisfies `IntoIterator`, so
nothing that could call it before cannot call it now. The remaining `Vec` is
kept deliberately: `meta_cluster` and `density_scores` need random access, and
an accumulation that a real algorithm requires is not a copy in the sense the
law forbids.

### 8.3 `stable_under_perturbation` — the copy hid a computation nobody read

This is the one that produced a genuine algorithmic win rather than an
allocation win.

**Before.** Inline, per probe:
```
let reperturbed: Vec<GradedRow> = (0..lens.len()).filter(visible).enumerate()
    .map(|(idx,pos)| GradedRow { idx, pos, quorum: 0, trajectory: … }).collect();
```

**After.** `grade_shapes_at(lens, visible, locus, hops).collect()` — a named,
lazy, shape-only grading.

**Why better.** Naming the operation exposed what the inline version obscured:
`meta_cluster` reads **only `.trajectory`**. The inline code already knew this —
it hardcoded `quorum: 0` — but the *general* `grade_rows` next to it did not,
and `stability_sweep` calls this **once per budget**, 11 times on the default
range. Extracting it made the asymmetry legible and let the quorum sweep be
skipped structurally rather than by a magic literal.

So the change removes, per basin per sweep: **11 heap allocations** AND **11
full quorum passes** — where a quorum pass is the expensive half (it scans peers;
the chain walk is `lens.at` lookups). The copy was not the cost. The copy was
hiding the cost.

### 8.4 The pattern worth keeping

In all three the copy was **an artefact of how the code was written, not of what
it had to compute**. A `collect` between two loops that are already fused; a
filter that reallocates what it filters; an inline rebuild that obscures which
half of the work is dead. None of them was a trade-off being paid for
correctness — which is why removing them costs nothing and buys laziness,
generality, and one dead computation deleted.

**This is the substance of the operator's ruling.** "The lens is the performance
floor — a materialization is strictly worse on BOTH axes" is not a slogan about
memory. Here it was literally true: the copies were slower AND they concealed
that a quorum was being computed 11 times per basin and thrown away.

**Measured, both directions:**

| path | before | after |
|---|---|---|
| grading → tail | 2 allocations/row, eager | 1 allocation total, lazy |
| perturbation sweep (default range) | 11 allocs + 11 quorum passes per basin | 0 allocs, 0 quorum passes |
| generality of `tail` | `&[GradedRow]` only | any `IntoIterator` (slices still work) |

Gates unchanged and green throughout: **325 planner tests**, clippy
`-D warnings` clean, fmt clean. No test was weakened to accommodate the change —
the equivalence and anti-vacuity tests from #868 still pass against the same
oracles.
