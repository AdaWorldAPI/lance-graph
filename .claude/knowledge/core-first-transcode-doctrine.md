# Core-First Transcode Doctrine — the Core empowers the AST to be thin

> **READ BY:** `core-first-architect`, `adapter-shaper`, `core-gap-auditor`,
> `family-codec-smith`, `integration-lead`, `truth-architect`, and ANY agent
> proposing C++→Rust transcode / codegen / AST-DLL / adapter / "port Tesseract"
> work.
> **Status:** CONJECTURE (2026-06-16) — the unification is architecturally
> coherent and fits the operator-locked OGAR canon, but it is **not a FINDING**
> until the `unicharset` adapter-parity probe (below) runs green. Label every
> downstream citation accordingly.
> **Trigger phrases:** "transcode", "codegen", "AST DLL", "C++ → Rust",
> "port Tesseract", "DO/DTO adapter", "generated Rust", "tesseract-rs source".

---

## The one sentence

**A generated AST / adapter / codegen layer is only ever as clean as the Core
it targets — so the Core must be a deliberate, reusable, hand-built foundation
(OGAR), NOT the leftover residue of "what we couldn't codegen."** The Core is
what *empowers* the generated layer to be elegant; without a rich Core, every
generated unit drags its own world (types, state, structure) and the output is
dirty — and the diff-gate just fails everything.

## The inversion (this is the whole trick)

```
Naive transcode:   C++  ──codegen──►  BLANK target
                   → every method re-implements its own types/state/structure
                   → fat, self-contained, dirty Rust; more cleanup than hand-port

Core-first:        C++  ──codegen──►  RICH target (the OGAR Core)
                   → each method becomes a THIN adapter that ASSUMES the Core
                   → classid for identity, SoA columns for state, ClassView for
                     composition; the adapter is a shape, not a re-implementation
```

The intelligence is in **recognizing the simplification** — shaping the Core
(and its movable parts) first so the generated layer collapses to thin shapes.
A residue-Core never lets the adapters be thin.

## The Core = OGAR's movable parts (the assume-contract)

OGAR is **operator-locked canon** (`CLAUDE.md` § CANON, `canonical_node.rs`,
locked 2026-06-13) — it is the hand-built reusable Core. Each movable part is a
thing the generated adapter gets to **assume**, which is exactly what keeps the
adapter thin:

| An adapter may ASSUME… | …because the Core provides | …so the adapter does NOT |
|---|---|---|
| "a class is a `classid`" | `canonical_node` key (`classid` u32) | carry type identity / define a struct hierarchy |
| "state lives in SoA value tenants" | helix `Signed360` / turbovec / palette columns; the per-axis widths are the **#511 `SoaMemberSpec`** calibration | define its own data structures |
| "relations are the edge block" | `EdgeBlock` (12 in-family + 4 out) | build its own graph / pointer web |
| "composition/inheritance is `classid → ClassView`" | ClassView preset selection (PR #498), driven by the harvest's `has_function`/`inherits_from`/`virtually_overrides` manifest | implement inheritance, vtables, MRO |
| "invocation is a `UnifiedStep`" | `OrchestrationBridge` / `UnifiedStep` (contract) | be a method bolted onto a god-object |

Concretely: `unichar_to_id` stops being "a method on `UNICHARSET`" and becomes a
thin fn — *read the unicharset tenant columns → table lookup → return id.* The
Core does identity, state, composition, dispatch; the adapter is the lookup.

## The two halves are one system (the harvest is NOT orthogonal to codegen)

Earlier framing called the SPO harvest "orthogonal" to codegen. **That was
wrong.** They are the two halves of one mechanism:

- **`ruff_cpp_spo` SPO harvest** (`has_function` / `inherits_from` /
  `virtually_overrides`) = the **method-resolution manifest** — *which* adapters
  a `classid`'s ClassView composes, and in what override order.
- **`tesseract-rs-ast-dll-codegen-v1`** = the **adapter bodies** — the Rust the
  ClassView dispatches to.
- **`classid → ClassView`** = the "inheritance" — composition of the adapter set
  the manifest names. **No new layer, no new `ValueSchema` variant.**

## The execution-model ladder (v1 → v2 → v3)

The doctrine above is **v1**: a Core targeted by thin adapters, codegen'd once at
build time. The operator's forward design (2026-06-16) extends it along the
**execution model** — *how* the adapter body is compiled and *where it lives* —
**without touching the Core-first invariant.** Each rung is **CONJECTURE**; the
gating probes are below. The striking part: rungs 2–3 land on substrate that is
already shipped, not greenfield.

| Rung | What it adds | Already-shipped substrate it lands on | The new edge(s) |
|---|---|---|---|
| **v1 — Core-first codegen** | thin classid-keyed adapters target OGAR; bodies codegen'd at build | `canonical_node`; `classid → ClassView` (#498); `tesseract-rs-ast-dll-codegen-v1` | — (this doc) |
| **v2 — two-tier compile** | ONE Elixir-shaped adapter source, TWO backends: **existing → compile-time** (Askama→Jinja: a proc-macro monomorphises to Rust, zero runtime cost); **new → JIT** (jitson/Cranelift) | `contract::jit` (`JitCompiler` / `JitTemplate` / `KernelHandle`); ndarray jitson/Cranelift; n8n-rs `CompiledStyleRegistry` | the `defadapter!` build-time macro (the Askama half); a JITSON lowering of the adapter shape (the JIT half) |
| **v3 — elixir-tissue over a fixed Core** | Core stays immutable; the DO-shaped business logic is **replaceable tissue** (BEAM hot-swap heritage) living in the **AST-DLL**, persisted + served + hot-swapped via **SurrealDB's API**; a **Kanban orchestration** reacts to **Odoo shapes** and dispatches the tissue | `contract::kanban` (`KanbanMove` / `KanbanColumn` / `StepDomain::Kanban`); `surreal_container` (`view`/`read` = projection, `write` = commit); `E-SUBSTRATE-IS-THE-SCHEDULER` (substrate emits the schedule, surreal LIVE reactive) | an **Odoo→kanban ingest** (Odoo model/stage shapes → `UnifiedStep{step_type:"kanban.*"}`); the AST-DLL **tissue store + hot-swap** over `surreal_container` |

**Why Elixir is the right syntax to steal (the deep reason, not just ergonomics).**
Elixir/BEAM's defining property is **hot code swapping of a running system** —
which is *exactly* "replaceable tissue on a fixed Core." Stealing the syntax
(multi-clause heads → `match classid`; `|>` → method chain / `defadapter!`;
`with` → `?`; behaviours → traits; `@spec` → types) buys the ergonomics; the
architecture buys the swap. "Tissue" is the workspace's own word (`CLAUDE.md`:
"AriGraph / episodic / SPO / CAM-PQ are thinking tissue — not storage"): the
Core is the skeleton, the elixir-adapters are organs hot-swappable around it.

**The Core-first invariant is unchanged across all three rungs.** Whether a body
is build-time-monomorphised (v2 Askama), JIT'd (v2 Cranelift), or hot-swapped
from the SurrealDB AST-DLL (v3), it is STILL a thin adapter that targets the OGAR
Core (classid / SoA tenants / ClassView / `UnifiedStep`). A tissue adapter that
needs state the Core can't hold is STILL a **Core gap → EXTEND-CORE**, never an
adapter-state-leak. The execution model changes; the iron guard does not.

### Probes that gate the new rungs

```
PROBE-COMPILE-TWO-TIER (P1, gates v2)
  Hypothesis: one Elixir-shaped adapter source produces byte-identical behaviour
              whether lowered by the build-time macro (defadapter!) or the JIT
              (JITSON → Cranelift).
  Pass:  macro-compiled and JIT-compiled adapter agree byte-for-byte on the
         unicharset corpus (and both match the libtesseract oracle).
  Fail:  divergence ⇒ the shape's semantics aren't backend-independent; fix the
         shared lowering before shipping two backends.

PROBE-SURREAL-TISSUE-SWAP (P2, gates v3 tissue layer)
  Hypothesis: a DO adapter served from the surreal_container AST-DLL can be
              hot-swapped (replace the body) WITHOUT rebuilding the Core, and the
              post-swap invocation still hits byte-parity.
  Pass:  swap body A→A' via the surreal API; ClassView dispatch picks up A';
         Core (classid/SoA/ClassView) untouched; parity holds for both.
  Fail:  the swap forces a Core change ⇒ the boundary isn't where v3 claims.

(the Odoo→kanban ingest is the APPLICATION rung; gated on both probes above plus
 a separate Odoo-shape → KanbanMove mapping spec, not yet written.)
```

All three rungs inherit the v1 falsifier (`PROBE-OGAR-ADAPTER-UNICHARSET`, below)
as their floor: no execution-model elaboration matters until one leaf adapter
hits byte-parity through a ClassView at all.

## Scope boundary (where it holds vs. where it must NOT be forced)

- **Holds cleanly** for the *mechanical, data-shaped leaf* methods — unicharset
  id↔utf8, recoder encode/decode, dawg membership, weight-matrix walks. These
  become clean DO-in/out adapters. (This is the exact subset the codegen plan
  already scopes to.)
- **Does NOT collapse** for the *intrusive / stateful / virtual-dispatch-heavy*
  core — ELIST/CLIST raw-pointer mutation, the BiLSTM / recodebeam numeric
  kernels. Forcing these into the DO-adapter mold is the **Frankenstein
  flattening** the workspace explicitly forbids (`frankenstein-checklist.md`).
  They stay raw-pointer hand-ports (codegen plan §5).
- Therefore this is the holy grail for the **integration shape** (how
  transcoded behaviour plugs into the substrate), **not** a free pass on the
  **transcode difficulty** (the hard kernels remain hard).

## The one iron guard (this is what keeps it honest)

> **A Core gap shows up as an adapter that needs state, or a dispatch, the Core
> cannot hold. When that happens, EXTEND THE CORE deliberately — NEVER hack the
> adapter.**

The moment an adapter starts carrying its own state, the elegance is gone and
you are back to the dirty parallel port. A Core gap is a *signal to grow the
deliberate Core* (a new value tenant, a ClassView capability), filed and
reviewed — not an excuse to fatten one adapter.

## The falsifier (CONJECTURE → FINDING gate)

Per `truth-architect` discipline, this doctrine is a CONJECTURE until measured.
The cheapest end-to-end probe:

```
PROBE-OGAR-ADAPTER-UNICHARSET (P0)
  1. Transcode 1–2 unicharset leaf methods (unichar_to_id / id_to_unichar) as
     classid-keyed DO-in/out adapters.
  2. Mint an OGAR classid whose ClassView composes them, using the harvested
     has_function manifest.
  3. Invoke through the ClassView.
  Pass:  byte-parity with libtesseract (FFI oracle) on a fixed corpus.
  Fail / leak: the adapter needs state the SoA tenants can't carry, or a
     dispatch the ClassView can't express → a Core gap found cheaply, BEFORE
     building the whole transcode.
```

Until this runs green, "the OGAR Core makes the transcode clean" is a
CONJECTURE. Do NOT scale the adapter approach across modules until it passes.

## Anti-patterns this doctrine exists to catch

- **Residue-Core** — treating the Core as "the parts we couldn't codegen"
  instead of the deliberate foundation designed first. Yields fat adapters.
- **Parallel-Object-Model** — building a second, standalone Tesseract-rs object
  model (structs + impls) instead of growing OGAR with classid-keyed adapters.
- **Adapter-State-Leak** — an adapter carrying its own state because the Core
  doesn't offer the tenant. Fix the Core, not the adapter.
- **Universal-Adapter-Flattening** — forcing intrusive/stateful methods into the
  DO-adapter shape (Frankenstein). Route them to raw-pointer hand-port instead.
- **Harvest-is-orthogonal** — forgetting the SPO graph IS the ClassView
  method-resolution manifest; treating harvester polish and codegen as unrelated.

## Cross-references

- `CLAUDE.md` § CANON — OGAR `canonical_node` (key / edge / value), the locked Core.
- `canonical_node.rs` — `NodeGuid` / `EdgeBlock` / `NodeRow`; classid / family / ClassView guards.
- PR #498 — `classid → ClassView` resolution (the composition mechanism).
- `crates/perturbation-sim/src/columns.rs` + PR #511 — `SoaMemberSpec` value-tenant calibration (Core-shaping: which column carries which value).
- `AdaWorldAPI/ruff` `ruff_cpp_spo` — the SPO harvester (the method-resolution manifest source).
- `.claude/plans/tesseract-rs-ast-dll-codegen-v1.md` — the codegen (adapter-body) plan; §5 module routing (codegen vs hand-port).
- `AdaWorldAPI/ruff/.claude/plans/cpp-spo-probes-v1.md` — the harvester gating probes.
- `lance_graph_contract::orchestration` — `OrchestrationBridge` / `UnifiedStep` (the adapter invocation surface).
- `.claude/knowledge/frankenstein-checklist.md` — composition-failure / flattening guard.
- `crates/lance-graph-contract/src/jit.rs` — `JitCompiler` / `JitTemplate` / `KernelHandle` (the v2 JIT tier; ndarray jitson/Cranelift compiles, n8n-rs caches).
- `crates/lance-graph-contract/src/kanban.rs` — `KanbanMove` / `KanbanColumn` / `StepDomain::Kanban` (the v3 orchestration seam; planner emits, ractor drives, surreal projects).
- `crates/surreal_container/` — the SurrealDB tier (`view`/`read` = projection, `write` = commit) that would host the v3 AST-DLL tissue store + hot-swap.
- `.claude/board/EPIPHANIES.md` `E-SUBSTRATE-IS-THE-SCHEDULER` — substrate emits the schedule (surreal LIVE reactive); the v3 Odoo→kanban reaction extends it. `E-TRANSCODE-EXEC-LADDER-1` records this ladder.
- `AdaWorldAPI/odoo` (`/home/user/odoo`) — the v3 shape source (Odoo model/stage shapes → `KanbanMove`).

## Provenance / credit

OGAR (the `canonical_node` / `classid → ClassView` / value-tenant Core) is
**operator-locked canon**, designed by the operator + prior sessions. What this
doc contributes is the **C++-transcode ↔ OGAR unification** (transcode as thin
classid-keyed adapters composed by ClassView, the SPO harvest as the
method-resolution manifest) and the **Core-first inversion** principle — that
the Core's deliberate shaping is *precisely* what lets the generated layer be
thin. Captured 2026-06-16, before it dilutes.
