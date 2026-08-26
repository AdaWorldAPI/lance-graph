## 2026-08-23 — E-RECIPE-EXECUTION-SEPARABILITY-1 — the 34 recipes already have real, tested execution semantics (`kernel(id)`/`Tactic::apply`); 23/34 are separable by state transition, 11 collapse into 15 coarse-signature collisions

**Status:** FINDING (measured — `PROBE-RECIPE-EXECUTION-1`, run against real
code). **Confidence:** High; reproducible from the commit.

**Correction to `PROBE-LOCO-INTERPRETER-1`'s KC1 framing** (`.claude/
brainstorms/2026-08-22-behavioral-ir-fathoming.md` §F1, `AdaWorldAPI/
lance-graph` PR #992): "the recipes' semantics live in `ThoughtCtx`/
`recipe_dispatch` wiring, out of scope" read as "not built yet." They are
built and tested: `lance-graph-contract::recipe_kernels.rs` carries all 34 as
real `impl Tactic` blocks (`apply(&mut ThoughtCtx) -> Outcome`), dispatchable
via `kernel(id: u8) -> Option<&'static dyn Tactic>` / `all_kernels() ->
[&'static dyn Tactic; 34]`, id space `1..=34` — the SAME id space
`lance-graph-ogar::recipe_vocab::op_of`/`recipe_of` uses for the `FnIndex`
mapping (`FnIndex(0x90 + id - 1)`), verified by a round-trip assertion, not
assumed.

**KC1 reframed and measured**, `crates/lance-graph-ogar/examples/
recipe_execution_probe.rs`: given the SAME starting `ThoughtCtx` (a 4-context
battery — hot/cold/empty/neutral), do different recipe ids produce
observably different state transitions? Signature deliberately coarse (fired
/ Δconfidence sign / which fields changed / candidate-count-delta sign) so
raw float noise cannot manufacture false separability. Result: **23/34
distinguishable across the whole battery; 11 collapse into 15 pairwise
collisions** (e.g. `ARE`/`ZCF`/`HKF`/`MCP` all "fire" but change nothing
observable on any context tested; `RCR`==`IRS`==`TCA`). Measured, not quoted:
31/34 `KernelMaturity::Operational`, 14/34 can move `confidence` at all;
restricting to Operational-only kernels barely moves the rate (23/31).

**The generalization:**

> A crate's own hedge ("out of scope for this run") can read, to a later
> session, as "does not exist" rather than "not pulled into THIS probe." The
> two are different claims with different next actions — the first says
> "build it," the second says "wire it in" — and conflating them costs a
> session the correct next step.

**What this does NOT establish:** an `ogar_loco::Call`/`FunctionBody` →
`kernel(id)` dispatch bridge. No `FnIndex` in `RECIPE_OP_BASE..RECIPE_OP_END`
is invoked by any interpreter today (`PROBE-LOCO-INTERPRETER-1`'s interpreter
still covers only the shared core). That bridge is now a small, well-scoped
task — the id mapping is already verified consistent, so it is wiring, not
missing semantics.

Cross-ref: `E-OGAR-LOCO-INTERPRETER-RUN-1` (the KC1-untested finding this
corrects), fathoming report §F1a.

