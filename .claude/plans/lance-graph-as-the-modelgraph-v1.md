# lance-graph AS the ModelGraph — v1

> **Status:** PROPOSAL (operator-set endgame, 2026-09-16). No code yet; this
> document is the survey that makes the next wave a wiring job instead of a
> design job.
>
> **Operator:** *"endgame should be to wire lance-graph as ModelGraph"*, arrived
> at through: *"sql() polyfill is next then / doublecheck and work on
> quack<>duckdb parity / always use ruff cpp to analyze the duckdb code"*,
> *"ruff has at least 2 codegen arms"*, *"lance-graph-arm-discovery supports
> pandas like table > spo based on frequency and confidence"*.

## The claim

`ModelGraph` is today a **transient Rust object** — `ruff_spo_triplet::ir::ModelGraph`
is `{ namespace: String, models: Vec<Model> }`, built by a harvest, consumed by
a codegen pass, then dropped. The endgame is that it stops being transient and
**becomes addressed rows in lance-graph's V3 SoA**: one substrate that the
harvest mints into, that `arm-discovery` mines, that codegen projects from, and
that `sql()`/quack queries.

The payoff is that these four stop being four pipelines over three formats and
become four readers of one addressed surface.

## The direction constraint — this is what decides the design

**No ruff crate may compile against lance-graph.** The edge is forbidden and the
tree takes real trouble to respect it:

- `ruff_cpp_codegen` — *"names `lance_graph_contract` types as text, never
  compiles against lance-graph"*; the rendered Rust is type-checked downstream
  in the consumer repo.
- `ruff_spo_address` — mints a 16-byte facet *"layout-identical to
  `lance_graph_contract::facet::FacetCascade`"*. Layout-identical, **not
  imported**.

Verified 2026-09-16: `grep -rn "lance.graph" ruff/crates/*/Cargo.toml` finds the
string only inside two `description =` fields. Zero dependency edges.

So "wire lance-graph as ModelGraph" can only mean **lance-graph becomes the
HOME**, fed across a byte boundary — never ruff importing the spine. This also
matches `assembler-vs-storage-substrate.md`: OGAR assembles, lance-graph is the
spine, Lance calcifies.

## The chain, and it is ALREADY BUILT except one step

```
 C++ corpus (e.g. AdaWorldAPI/duckdb)
   │
   ├─(A) ruff_cpp_spo::extract_tree ─────────► ModelGraph          [ruff]
   │        CppClass / CppFunction / CppEnum
   │        has_function · inherits_from · virtually_overrides
   │
   ├─(B) ruff_spo_address::mint ─────────────► Facet, 16 B         [ruff, no dep]
   │        part_of = has_field/has_function  (hi chain, mereology)
   │        is_a    = inherits_from/rdf:type  (lo chain, taxonomy)
   │        byte-identical to FacetCascade, CascadeShape::G6D2
   │
   ├─(C) ogar-from-ruff::mint ───────────────► CompiledClass       [OGAR]
   │        THINK arm Class + DO arm ActionDef + the rail Facet
   │
   ├─(D) ogar-from-ruff::lance_sink ─────────► NodeRow 512 B
   │        compiled_class_to_facet   (a reinterpret no-op)
   │        compiled_class_to_noderow (CANON key|edges|value)
   │        compiled_classes_to_le_bytes  ──► as_le_bytes()
   │
   ├─(E) ✗✗✗ THE ONE MISSING LINK: Dataset::write ✗✗✗
   │        lance_sink.rs names it out of scope BY DESIGN:
   │        "stops at as_le_bytes(), exactly as network::to_facet stops at
   │         the facet … that column write needs the lance engine / ractor
   │         runtime."
   │
   └─(F) the readers, once (E) exists
            quack / sql()     — query it
            arm-discovery     — mine it, write truth back as edges
            ruff_cpp_codegen  — project Rust manifests FROM it
```

**(A)–(D) are live, unstubbed code.** `lance_sink.rs` is 263 lines with zero
`todo!`/`unimplemented!`. The endgame is not a rewrite; it is **(E) plus
pointing (A) at a second corpus**.

## What arm-discovery contributes, confirmed in source

`lance-graph-arm-discovery` is the **table → SPO** arm, exactly as the operator
described. `translator.rs`, verbatim:

- ARM **confidence** = `cooccur / antecedent_count` → NARS **frequency**
- ARM **evidence** `m = cooccur` → NARS **confidence** `c = m / (m + k)`
- canonical carrier is `TruthU8` (`frequency`/`confidence` as `u8`) — *"exactly
  the `CausalEdge64` wire, float-free"*; the `f32` `NarsTruth` exists only as an
  edge *"because the downstream `spo::truth::TruthValue` and
  `ruff_spo_triplet::Triple` are themselves f32"*.

That last clause matters: **arm-discovery already names `ruff_spo_triplet::Triple`
as its downstream.** The seam is half-acknowledged in code today.

Its input is `Dataset { spec: FeatureSpec, rows: Vec<Vec<u32>> }` — a
categorical table, one code per feature per row. That is precisely the shape of
the harvest ore (`events.tsv`: method × ordinal × EventKind × scope × …), so
the DuckDB event ore can be mined without inventing a format. **The encoder
from ore-TSV to `Dataset` does not exist yet** and is the cheapest real
deliverable in this plan.

## The codegen arms — two, and the second is not the obvious one

| arm | in | does |
|---|---|---|
| `ruff_cpp_codegen` | ruff | ModelGraph → `ClassManifest` → **Rust source** (`MethodSig` targeting the OGAR Core), gated by a `decompile ∘ project` signature-plane round-trip |
| `ruff_spo_address` | ruff | SPO triples → **`(part_of:is_a)` rank-mint** → the 16-byte address. Its own doc: *"the one genuinely-new brick between the `ruff_*_spo` SPO harvest and the lance-graph `(part_of:is_a)` GUID SoA … the carrier is already there; this crate fills the mint."* |

`ruff_python_codegen` is **not** one of these — it is upstream ruff's
`Generator`/`Stylist` (Python AST → source) and unrelated to the SPO path.

## Honest gaps, in dependency order

| # | gap | size |
|---|---|---|
| G-A | `extract_tree` has **never been run on DuckDB** — only the behavioural `harvest_events` arm has. The structural arm is what yields the operator class tree. | small: run it |
| G-B | No ore-TSV → `arm_discovery::Dataset` encoder. | small |
| G-C | **(E)** — no `Dataset::write`. Everything upstream stops at `as_le_bytes()`. | the real one |
| G-D | `sql()` does not exist on the Java surface. Per `lance-graph-java/CLAUDE.md` it is the whole consumer story, and it needs a substrate to query. | gated on G-C |
| G-E | **quack has no DuckDB in its loop at all** — zero dependency; every "DuckDB" in it is a doc comment, and its oracles are hand-written Rust. What is called parity today is quack ↔ its own oracle. Task #8's "DuckDB semantic differential fixtures" remains genuinely unstarted. | independent, real |

## Wave order (proposed, not started)

1. **W1 — structural harvest of DuckDB** (G-A). Run `extract_tree` over the
   operator/expression headers; commit the triples beside the event ore, same
   provenance discipline. Falsifier: the class tree must contain the types the
   translation matrix reasoned about by hand (`Vector`, `SelectionVector`,
   `ValidityMask`, `ht_entry_t`, the `ScalarExecutor` sinks) — if the harvest
   misses them, the structural arm has DuckDB's template problem too and that
   is the finding.
2. **W2 — ore → Dataset encoder** (G-B), then mine the DuckDB event ore and
   read the rules. Falsifier: rules must be non-vacuous — a rule set whose
   confidence is ~1.0 everywhere is measuring the harvester's own grammar, not
   DuckDB.
3. **W3 — the write** (G-C). The single highest-value step; unblocks G-D and
   makes every earlier stage durable instead of transient.
4. **W4 — real DuckDB differential** (G-E). Independent of 1–3 and honest to
   start now: it is the only thing that makes the word "parity" true.

## What this plan does NOT claim

It does not claim (A)–(D) work on DuckDB — they are proven on the Tesseract/
C# corpora, and DuckDB is template-heavy in a way that already defeated one
harvest arm (`duckdb-to-v3-translation-matrix-v1.md` §6: seven TUs at 100 %
`Empty`). W1's falsifier exists precisely because that may recur.

It does not claim (E) is small. `lance_sink.rs` names the blocker in its own
words, and `symbiont`'s tombstone write is recorded BLOCKED for the same
reason.

---

## The OLD ModelGraph sank to the WRONG place — and the reason names this plan's risk

Operator, 2026-09-16: *"OLD ModelGraph was sinking to surrealql AST DLL arm by
mistake which later was deprecated and switched to ogar IR with a wrapper to
wire adapters and storage and Klickwege parity via askama in ERB redmine ERB
fieldview ergonomics."*

Confirmed in primary sources, and the deprecation's own words are the point:

> `OGAR/crates/ogar-adapter-surrealql/src/lib.rs:3` —
> **"⊘ DEPRECATED (operator ruling 2026-07-22 — a separation-of-concerns
> error)"**, and *"The reason is not primarily 'it doesn't work'."*

`docs/SURREAL-AST-AS-ADAPTER.md` records what went wrong: the behavioural arm
(`ActionDef` + `ActionInvocation` + `KausalSpec` + lifecycle) has **no
equivalent vocabulary in DDL**, so encoding it there means *"hijacking `DEFINE
EVENT … WHEN … THEN …`"* — §0 calls this the **negative-beauty workaround**:
*"it works in a narrow sense and rots immediately."*

**Why this is load-bearing for THIS plan rather than history.** The
corrected shape is exactly the chain in the diagram above — ModelGraph sinks to
**OGAR IR**, and a **wrapper** then wires three separate things that the DDL
sink had collapsed into one:

| the wrapper wires | today |
|---|---|
| **adapters** | `ogar-adapter-*` (SurrealQL demoted to one membrane among several, never the spine) |
| **storage** | `ogar-from-ruff::lance_sink` → `NodeRow` → `as_le_bytes()` — this plan's G-C |
| **Klickwege parity** | askama fieldview — `ogar-render-askama::render_field_view`, consumed by `a2ui-rs`; the ERB/redmine lineage |

So G-C is not merely "add a write". It is the storage leg of a **three-leg
wrapper whose whole reason for existing is that collapsing the legs was the
original error.** A G-C implementation that reached back into adapter or render
concerns would be the 2026-07-22 mistake with a different target, and the
`layer-boundary-warden` / `bbb-warden` cards are the gate.

**Direct consequence for `sql()` (G-D):** the same trap in its Java clothes.
`sql()` is an ADAPTER leg. It must never become the place behaviour is
expressed — which is precisely what `lance-graph-java/CLAUDE.md`'s ruling
already says from the other side (*"behavior travels by ADDRESS … `onClick:
<lambda>` in a component tree is the same hijack as `DEFINE EVENT` in DDL"*,
a2ui-rs T2). Two repos, two rulings, one error.

## The THIRD codegen arm — `ruff_python_dto_check`

Operator: *"ruff dto crate is meant to deduplicate routes and … also to propose
Separation of concern optimizations."*

It exists and is substantial — `bundle`/`calibrate`/`codegen`/`config`/
`contract`/`emit`/`extractors`/`matcher`/`observations`/`preflight`, its own
`bin`. Self-described as *"config-driven extractor over `ruff_python_parser`
that harvests structured DTO/route/handler facts … with a preflight subcommand
that proposes a config from the tree itself"*, and **additive** — it depends on
`ruff_python_parser`/`_ast`/`ruff_source_file` and modifies no other crate.

So the count is **three**, not two, and the plan's table above is corrected:

| arm | emits |
|---|---|
| `ruff_cpp_codegen` | Rust `MethodSig` manifests from a C++ ModelGraph |
| `ruff_spo_address` | the `(part_of:is_a)` rank-mint → 16-byte facet |
| **`ruff_python_dto_check::codegen`** | **handler + view template** — `codegen/{dto,columns,jinja,pipeline,target}.rs`, kind-generalized by `KindRecipe` so *"the other 10 kinds slot in by adding a recipe entry — not new Rust per kind"* |

**`jinja.rs` is the same fieldview ergonomics leg** the wrapper wires as
Klickwege parity — the template side of the askama/ERB lineage, reached from
route/handler facts instead of from a `DocIr`. That is a convergence worth
probing, not assuming: two producers, one view-template surface.

Its guardrail is already the right one and should be inherited by anything this
plan generates: *"Generated code goes to a draft directory; it is never wired
into a build and never emits `unimplemented!()`/`todo!()` into a compiled
production path (the PR #102 failure guardrail)."*

**Route dedup + SoC proposals are the capability to reuse, not rebuild.** Note
what the pairing means: the SurrealQL sink was deprecated *as a
separation-of-concerns error*, and this crate's stated job includes *proposing
separation-of-concern optimizations*. The tool that proposes the fix and the
mistake it would have caught are the same concept at two ends of the arc — so
pointing it at a corpus is a cheap, direct check on whether a sink is
collapsing legs, rather than a new analysis to invent.

**Scope honesty:** it is a **Python** extractor (`ruff_python_parser`). DuckDB
is C++, so it does not apply to W1 directly; its relevance here is (a) the
third codegen arm for the count, (b) the view-template convergence with
askama fieldview, and (c) route-dedup/SoC as an existing capability for the
Python-shaped consumers (odoo-rs, woa-rs, the `list_for_tenant`/`soft_delete`
recipes it already implements end-to-end against the woa-rs oracle).
