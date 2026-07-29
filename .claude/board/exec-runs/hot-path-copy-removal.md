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

`reasoning.rs` also still has `PremiseBundle.premises: Vec<..>` and
`arena(&self) -> BeliefArena`, which **builds and returns a whole arena per
call**. Documented in #866 as "the escape hatch"; it is a per-call
materialisation and is unexamined.

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
