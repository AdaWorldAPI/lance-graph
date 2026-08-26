## 2026-08-23 — E-RECIPE-DISPATCH-BRIDGE-1 — the `FnIndex -> kernel(id)` seam is built and identity-preserving for all 34 ids; one receipt now spans both instruction ranges

**Status:** FINDING (measured — `PROBE-RECIPE-DISPATCH-BRIDGE-1`, run against
real code). **Confidence:** High; reproducible from the commit.

**Closes the seam `PROBE-RECIPE-EXECUTION-1` (#995) left open.** #995 called
`kernel(id)` directly — never through an `ogar_loco::Call`. This probe
(`crates/lance-graph-ogar/examples/recipe_dispatch_bridge_probe.rs`) builds
the missing arm: given a `Call` whose `FnIndex` is a minted recipe op,
`recipe_of(f)` resolves the id, `kernel(id)` is invoked via `run_with`, and
the result is recorded alongside ordinary shared-core execution in ONE trace.

**Three falsifiers, all green:**
- **Part 1 — identity, all 34 ids.** For every `id in 1..=34`: routed through
  the bridge (`op_of(id)` → `Call` → interpret → `recipe_of` → `kernel(id)`)
  vs. called directly on an identical starting `ThoughtCtx`, bypassing the
  bridge entirely. `Outcome` (`PartialEq`) and the resulting `ThoughtCtx`
  (`Debug`-compared) are identical on **all 34** — no id resolves to the
  wrong kernel, no operand corruption, no gate skipped.
- **Part 2 — determinism under replay**, bridged, 4-slot focus battery:
  byte-identical traces across two runs, 4 sampled ids.
- **Part 3 — one canonical receipt spanning both instruction ranges**: a
  single 7-call program interleaves shared-core arithmetic (`NUMBER`, `ADD`)
  with two recipe dispatch calls (`RTE`, `ASC`), producing ONE ordered trace
  — `shared-core, shared-core, shared-core, recipe, shared-core, shared-core,
  recipe` — the shape #992's `TraceEvent` and #995's per-kernel signature
  never combined into.

**Layering held, as directed:** `ogar_loco` was NOT touched (stays zero-dep,
recipe-blind); `lance_graph_contract::recipe_kernels` was NOT touched (stays
zero-dep, loco-blind). The bridge is a small adapter living entirely in
`lance-graph-ogar` — the one crate the workspace already lets depend on both
— not a new generic `DomainDispatch` trait (deferred until a second domain
vocabulary needs one).

**What remains explicitly unbuilt, by design:** the recipe operand's real
address resolution against the basin-local attention-focus codebook
(`recipe_vocab.rs`'s own module doc: *"the basin that owns the prefix owns
the resolution"*). This probe stands in with a small, honestly-labelled
focus-slot array — the identity/determinism/composition questions it answers
do not depend on how the focus address was resolved.

Cross-ref: `E-RECIPE-EXECUTION-SEPARABILITY-1` (the semantics this bridges
to), `E-OGAR-LOCO-INTERPRETER-RUN-1` (the shared-core half).

