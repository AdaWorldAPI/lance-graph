# EXO-229 Gate A — lance-graph-dt64 through Charon → ruff_charon → Codegraph

## Identity
- source revision: lance-graph c70e7d8 (dt64 + contract trees identical to origin/main 9f4563b)
- charon: AdaWorldAPI/charon c69e231, toolchain nightly-2026-09-17; cargo-miri absent → default sysroot (std/core bodies opaque)
- crate instance: lance-graph-dt64 lib target only, default features, x86_64-unknown-linux-gnu
- ruff_charon: `import` (rows.tsv #version 2, files.tsv fnv64, rows.bin, effects.tsv, residuals.tsv)
- reader: `lance-graph-codegraph` via the new read-only example `cg_query`

## Cost
- charon LLBC: 29 s; ruff_charon import: 1.2 s; `Codegraph::open` + one query: 4–6 ms
- full sweep of `impact` over all 735 positions: 3.7 s wall, i.e. ~5 ms per process launch

## Facts recovered
| predicate | rows |
|---|---|
| CALLS_RESOLVED | 1355 |
| DEFINES | 517 |
| PUBLIC | 370 |
| IMPLEMENTS | 347 |
| SYNTHETIC | 26 |
| MAY_DISPATCH | 10 |
| REFERENCES | 8 |
| CRATE | 1 |
residuals: 0 rows (header only). effects: present (e.g. `Overflow:Mul` asserts).

Observed files: `src/dr.rs`, `src/lane.rs`, `src/lib.rs`. **Unobserved:** `tests/*.rs`, `examples/`, benches, every other crate.

## Cube limits (measured, not assumed)
- 735 positions (517 defined + external callees), `open` did not refuse.
- `impact` over all 735: **0 refusals** (no ClosureWidth(64), no TooMany(256) at this size); 381 non-empty, 354 empty.
- Conclusion: the limits are not exercised by a 500-symbol crate. A larger crate or a multi-crate store is needed to observe them; this run cannot be cited as "limits never bind".

## Answer-key checks
| question | Codegraph | source | verdict |
|---|---|---|---|
| callers of `LaneBinding::check` | `lane` (lane.rs:360), `borrow_all` (lane.rs:386), CONFIRMED | same 2 in-crate; plus ≥8 in `tests/lane_socket.rs` (227, 277, 314, 373, 403, 407, 424, …) | in-crate exact; test callers **unobserved and not reported as such** |
| callees of `LaneBinding::check` | `admit_lane`, `source_owns`, `current_route`, `FrameStore::check`, `BindStore::resolve_counted`, `LaneAddr::eq`, `Vec::into_iter` (25 sites) | matches lane.rs body | correct |
| impact of `BindStore::resolve_counted` | `check`, `lane`, `borrow_all` | correct transitive in-crate set | correct |
| callers/impact of `BindStore::admit`, `method_ref` | **empty**, rc=0 | ~140 call sites across ~20 files in other crates (quack `bind.rs`, r2il tests, report, dir-rmap, probes, labs …) | **false unaffected** |
| dynamic dispatch | 10 MAY_DISPATCH rows, e.g. `ClassView::fields_at via Clause`, `LaneSource::borrow via Clause` | trait-object / generic calls in lane.rs | kept POSSIBLE, never CONFIRMED |

## The finding
261 of 370 `PUBLIC` symbols return an empty `impact` set (339 of 517 defined symbols overall).
The store already carries both facts needed to say "open beyond this crate" — the `PUBLIC` row and the
observed-files list — but `Codegraph::impact` returns `Graded { confirmed, possible, unknown }` with
`unknown` empty. Under crai-rs CR-H0 ("UNKNOWN is a value, not an absence"; "false unaffected is more
severe than UNKNOWN") this is the most serious defect found: an empty answer that reads as "nothing
depends on this".

Fix shape, not applied (no Codegraph API change in this mission): a reader-side rule — a `PUBLIC`
symbol whose crate artifact is the only one loaded gets `unknown = {EXTERNAL_CALLERS_UNOBSERVED}`; any
query result carries the observed-file set so test callers show as not covered. `cg_query` can print
this today from `PUBLIC` + `files.tsv` without touching the library.

## Not covered by this gate
- test and example targets (charon was run on the lib target only)
- other crates' calls into dt64 (needs one artifact per consumer crate)
- macro-expanded bodies are in LLBC, but their spans point at the macro call site
- std/core bodies (opaque: no miri sysroot)

# Gate B — four-crate store (dt64 + contract + quack + r2il), same charon/ruff_charon config

| crate | charon s | import s | harvested | represented | residual |
|---|---|---|---|---|---|
| lance-graph-contract | 21 | 7 | 31015 | 31014 | 1 (`REFERENCE_NOT_NAMED rung_schedule::requires_of`) |
| lance-graph-quack | 14 | 1 | 1796 | 1796 | 0 |
| lance-graph-r2il | 7 | 1 | 2319 | 2319 | 0 |
| lance-graph-dt64 (Gate A) | 29 | 1.2 | — | — | 0 |

- 9,138 positions; `open` + `symbols` 159 ms. In-process sweep of `impact` + `callers` over all
  positions: **1,224 ms, 0 refusals** (9,569 answered, 8,707 empty).
- **Correction to the brief:** `ClosureWidth(64)` and the 256-id tile bound belong to the
  single-tile `Cube::closure`. Codegraph on main builds `Cube::new_tiled` and answers with
  `closure_wide` (`lance-graph-xlog/src/wide.rs`, `MAX_POSITIONS = 2^20 − 256`). They are not
  the limits a Codegraph query meets. The sweep's refusal arm is not disable-proven by this store
  (nothing refuses); the crate's own `Inject::Closure` / `Inject::Stage` tests cover that plumbing.
- **Cross-crate edge recovered:** `lance_graph_r2il::binding::prepare → BindStore::admit`, CONFIRMED.
  `method_ref` still has no observed caller (its callers are in tests/probes/labs, unloaded).
- **Grep false positive caught by the compiler arm:** `lance-graph-quack/src/bind.rs` has five
  `.admit(` calls; they are quack's own test-only typed code space, and quack does not depend on
  dt64. A name grep counts them as DT64 users; Codegraph correctly does not.
