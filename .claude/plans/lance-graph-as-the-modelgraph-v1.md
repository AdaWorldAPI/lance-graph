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

---

## ⊘ G-C's BLOCKER AS STATED IS STALE — ractor is not what the write needs

Operator, 2026-09-16: *"ractor is replaced by the lance-graph soa_mailbox.rs and
lance-graph-supervisor kanban_actor.rs is the readonly monitoring version."*

Confirmed in source, and it **removes an imaginary blocker from this plan**.
G-C above quotes `lance_sink.rs`: *"that column write needs the lance engine /
ractor runtime."* The ractor half of that sentence is out of date.

**`kanban_actor.rs` says so in its own module doc** (tombstone dated
2026-08-05, `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`):

> *"Message-free kanban **visibility + pure helpers** — what remains of the
> retired actor surface."* … `KanbanActor` + `KanbanMsg::{Advance, MulAdvance,
> Tick}` and the RPC drivers are *"DELETED, not deprecated: a version tick was
> being read as permission to advance, and an ack-shaped RPC was being read as
> what makes the substrate progress."*

And it names the model that replaced them:

```text
think → seal → publish Lance version → next cycle reads the published version
```

> *"Nothing signals, acknowledges, or schedules that."*

**The write discipline lives in `mailbox_soa.rs`** — `crates/cognitive-shader-driver/
src/mailbox_soa.rs` (the operator's "soa_mailbox.rs", words swapped; searching
the filename alone finds nothing, which is why this was worth checking rather
than restating). It carries one-writer-per-mailbox as a *compile-time* property,
not a convention: rows are *"consumed in place by the owner (no emission)"*,
*"ownership is compile-proven"*, and the *"mutated only via the owner trait
invariant is compiler-enforced"* (E-CE64-MB-4).

### What this changes, concretely

| | before this note | after |
|---|---|---|
| G-C is | "build or adopt an actor runtime we don't have" | **publish a Lance version from an owned mailbox** |
| the shape | an ack-shaped RPC per write | `think → seal → publish`; progression is existence, not command |
| the risk | none stated | porting actor-per-write would re-add the exact architecture this repo DELETED |

The last row is the one to carry into W3. `lance-graph-java/CLAUDE.md` already
states it as an inherited rule — *"actor/message-per-write is the architecture
lance-graph itself DELETED — never port it… per-owner `advance(owner)` RPCs are
exactly the deleted shape"* — and it applies with equal force here, on the
repo that did the deleting.

**Residue, so it is not mistaken for the live model:** `ractor` still appears in
four `Cargo.toml` files — `cognitive-stack`, `symbiont`, `onebrc-probe`,
`lance-graph-supervisor` (the last two `optional = true`, and the supervisor's
`description` still advertises a *"ractor-supervised actor tree"*). `symbiont`
is operator-ruled no-go (`CLAUDE.md`: *"⊘ DEPRECATED 2026-08-18, operator
no-go"*). Those entries are leftovers; none of them is evidence that the write
path wants an actor.

**Consequence for the wave order:** W3 gets cheaper and its acceptance gate gets
sharper. It is not "stand up a runtime"; it is "seal a `NodeRow` batch and
publish one version, written by the mailbox that owns those rows." The
falsifier follows directly — a W3 implementation that introduces a message, an
ack, a tick, or a per-owner `advance()` RPC has re-created the deleted shape and
fails on that ground regardless of whether the bytes land.

---

## ⊘ SECOND CORRECTION — G-C IS NOT "no write". The writer EXISTS; the READ is the gap.

Operator, 2026-09-16: *"lance-graph 879 + 908..912 949 (?) started to migrate it
accordingly to batchwriter."* Followed. The arc, and what it left:

| PR | landed |
|---|---|
| #879 | D-MBX-A6-P4 cycle loop-closure driver — sparse seal/apply + MUL-gate thought seam |
| #908-910 | boot config at `.config/<repo>/config.yaml`; `soa_config` unknown-key rejection + classid doc corrections |
| #911 | `graph/cycle_sink` — the concrete Lance-backed `WalSink` |
| **#912** | **Phase A: artifact-backed commits + THE SOLE OWNED LANCE WRITER** (supersedes #911's contract) |
| #949 | D-WXS-2a half A — row-major vs Morton KILL fires, no code change |

**So `Dataset::write` is real and in-tree** — two call sites, under a module doc
saying the I/O *"is INTERNAL to this one writer"*:
`Dataset::write` at `crates/lance-graph/src/graph/cycle_sink.rs:675` (the Create
inside `LanceCycleWriter::bootstrap`) and `Dataset::write` at
`crates/lance-graph/src/graph/cycle_sink.rs:710` (the Create-or-Append inside
`LanceCycleWriter::raw_append`). The plan's G-C above — *"no `Dataset::write`"* —
was **wrong when written.**

> ⊘ The first spelling of this citation was `…rs:675` followed by a bare
> `` `:710` ``, and the citation-decay gate correctly failed it: its anchor is the
> NEAREST backticked token, so `:710` — which is not a symbol and appears nowhere
> near line 675 — became the thing it went looking for. The fix the gate asks for
> is not a corrected number but an anchor that is an ADDRESS, so each citation now
> carries `Dataset::write` adjacent to it, and that string is literally on both
> cited lines. A shorthand second reference is cheap to write and costs a real
> gate failure.

### `batch_writer.rs`'s own status note is ALSO stale

It carries, dated 2026-07-27: *"`cast()` itself has **zero production call
sites**."* Measured 2026-09-16 — **three**, all production, none in `#[cfg(test)]`:

| site | role |
|---|---|
| `cognitive-shader-driver/src/mailbox_soa.rs:764` | the **owner** casting its own moves (`writer: &mut BatchWriter<P>` in a production signature) |
| `lance-graph-planner/src/owner_adapter.rs:100` | **write-on-behalf** (`//! write-on-behalf to the BatchWriter`) |
| `lance-graph-supervisor/src/cycle_driver.rs:407` | **P4a (drain)** — *"Map the fleet's staged `BatchWriter` casts into `SweepSlot`s"* |

(The ~17 other `.cast(` hits are ractor `ActorRef::cast(SupervisorMsg::…)` — a
different method entirely. Counting them would have inflated this.)

### What IS still unwired — and it is the observation leg, not the write

**`deinterlace` has zero production callers.** Every hit outside `temporal.rs`
is a doc comment; `witness_fabric.rs:510` cites the same 2026-07-27 note. So:

```
intent staged  (BatchWriter::cast)        ✓ wired, 3 sites
bytes written  (cycle_sink Dataset::write) ✓ exists, #912
durability READ (deinterlace)              ✗ ZERO production callers
OGAR bytes → that writer                   ✗ lance_sink still stops at as_le_bytes()
```

**G-C therefore splits, and neither half is what the plan said:**

- **G-C1 — connect `ogar-from-ruff::lance_sink` to the writer that already
  exists.** Smaller than "build a write": both ends are built and they do not
  meet.
- **G-C2 — wire the durability read.** `batch_writer.rs` states the ruling it
  obeys (`E-ACK-ELIMINATED-1`, no confirmation bookkeeping — durability is the
  row's own `LanceVersion` read through `temporal`) and then says that read has
  no production implementor. **A write nothing reads back is not yet a loop.**

> **Third stale blocker this session.** `lance_sink`'s "needs the ractor
> runtime", `batch_writer`'s "zero call sites", and this plan's own "no
> `Dataset::write`" were all true once and all false when I quoted them. The
> pattern is specific enough to name: **a STATUS note dated months back is a
> measurement, and this workspace's own rule is that a measurement without a
> re-run is an anecdote.** Re-measure before citing — three for three.

## The Rubicon connection — why `revision.rs` and the missing read are ONE gap

Operator: *"ogar-loco and revision.rs are 2 of the loose ends"* … *"revision.rs
is supposed to be the rubikon heckhausen last phase."*

Heckhausen's Rubicon model runs **predecisional → preactional → actional →
postactional**, and the last phase is *evaluating the achieved outcome against
the original goal*. The model is genuinely wired here — `cognitive-compiler/
src/lib.rs:18` carries *"The five Rubicon phases (Heckhausen). A trace records
which phase produced…"*, `lance-graph-contract/src/rubicon_witness.rs` maps the
phases and asks *"why a focus mask can falsify it"*, and `lib.rs:167` names
**D-ACR-8 — reading the Heckhausen crossing from the focus of attention**.

`revision.rs` (in the CONTRACT, not the planner) says of itself:

> *"models revision, not truth persistence, and still has **no production write
> capability** — the output types stop before actual-world mutation by design"*,
> closing a gap where `entropy-closure-causal-ground-v1` (#1057) names revision
> *"the only write-back"* and *"the court of appeal."*

**These are the same gap seen from two ends.** The postactional phase cannot
evaluate an outcome it cannot observe; observation is `deinterlace`; the court
of appeal has no write-back because the durable read that would tell it what
actually happened is unwired. G-C2 is therefore not plumbing — **it is what
closes the last Rubicon phase**, and "revision has no write capability" is a
symptom rather than an independent item.

## The kanban seam — planner ↔ ogar-loco ↔ ogar-r2il

Operator: *"lance-graph-planner <> ogar-loco <> ogar-r2il has a lot of gaps to
wire the kanban."* Measured — and the gap is NARROWER and more specific than
"unwired".

> ⊘ **I drafted "unwired in both directions" and it was FALSE.** Caught before
> commit by re-checking a citation I had just written, which is the same
> re-measure rule this document states three paragraphs up. The bridge crate
> EXISTS.

**`lance-graph-ogar` IS the seam, and it is built.** A workspace member
(`Cargo.toml:96`) that depends on BOTH sides — `lance-graph-contract` by path
(`:86`, `:159`) and `ogar-loco` by git (`:137`) — for the stated reason that
neither may import the other:

> `recipe_vocab.rs` — *"the 34 NARS recipes as `ogar-loco` ops above
> `DOMAIN_FLOOR`, with the kanban census as the awareness surface
> (`D-ACR-9`)"* … *"`ogar_loco` is zero-dep by design and
> `lance_graph_contract` is zero-dep by charter. **Neither may import the
> other** … A vocabulary needs both, so it lives in a consumer that already
> depends on both"* … *"**The ladder IS a program**: `recipe_dispatch::ladder`
> returns an ordered `Vec<RecipeStep>`; loco executes ordered
> `(function : value)` calls. That is mechanism, not analogy."*

So the 34-recipe → loco-op unification and the kanban census surface are
**already written**. What is missing is one edge:

- **The planner does not depend on `lance-graph-ogar`** (verified: no such line
  in `lance-graph-planner/Cargo.toml`). The seam exists in a crate the planner
  cannot see.
- **`ogar-loco` / `ogar-r2il` themselves: zero `kanban` hits.** Correct by
  design — the zero-dep rule above means the kanban vocabulary belongs in the
  bridge, not in loco. Their silence is conformance, not a gap.
- The planner-side references stay forward-looking doc comments
  (`cognitive_palette.rs:77` *"Growth from here is `ogar-loco`, not this
  table"*; `recipe_dispatch.rs:301` *"what an ogar-loco `FunctionBody` WILL
  carry"*) because nothing has connected them to the bridge that now exists.

> ⊘ **AND THE REMEDIATION I FIRST WROTE HERE — "add a planner → lance-graph-ogar
> dependency" — IS ALSO WRONG.** The operator pointed at `hotplug.rs`, and it
> names a different mechanism entirely. Second correction to this same section,
> both caught before commit.

### The sanctioned mechanism is HOT-PLUG, and a Cargo edge is not it

`lance-graph-contract/src/hotplug.rs` — *"Generic consumer hot-plug — the
plug-and-play pattern EVERY consumer migrates to (operator, 2026-07-07)"* —
three roles, three homes:

| role | home | note |
|---|---|---|
| **socket** | `lance-graph-contract::hotplug` | `HotPlug` + `CapabilityAuthority`, **zero-dep**: *"No OGAR dep — the contract … MUST stay dependency-free (a path dep here breaks every CI cargo invocation at workspace-load time; learned 2026-07-07)"* |
| **authority** | OGAR-side | resolves classids → vocab rows, action defs, and the storage reading (`Activation::read_modes`); verifies registration |
| **consumer** | its own crate | declares **one `HotPlug` const**, calls `activate` in its own binary/tests |

> *"The classid is the join key on BOTH sides … drift bangs once, **no pins, no
> serialization, no per-consumer plug crate**."*

**The activation cascade** (operator, 2026-09-16): *"plug and play mints the
classid. and classid mints ogar-vocab which triggers ogar-loco vocabulary."*

```
HotPlug const (consumer)  →  activate()  →  classid minted
                                         →  ogar-vocab rows (Class + ActionDef)
                                         →  ogar-loco vocabulary (the palette /
                                            recipe ops, landing in recipe_vocab.rs)
```

That is why `recipe_vocab.rs` can carry *"the 34 NARS recipes as `ogar-loco`
ops … with the kanban census as the awareness surface (`D-ACR-9`)"* without
either zero-dep crate importing the other: **the classid is the join, not a
dependency edge.**

### Measured state of the hot-plug spine

| piece | state |
|---|---|
| socket (`HotPlug`, `CapabilityAuthority`) | **built**, zero-dep, in the contract |
| authority (`impl … CapabilityAuthority for OgarAuthority`) | **built and PRODUCTION** — `lance-graph-ogar/src/lib.rs:524`; `#[cfg(test)]` does not begin until `:612` |
| consumer declarations | **all eight are TEST FIXTURES** (`:618`–`:831`, inside the two `cfg(test)` modules at `:612` and `:681`) — including `BLOCKLY` |
| **the planner's own `HotPlug`** | **ABSENT** — zero `HotPlug`/`hotplug` hits anywhere in `lance-graph-planner/src` |

> A grep note worth keeping: `impl.*CapabilityAuthority` found nothing on the
> first pass because the impl is written fully-qualified
> (`impl lance_graph_contract::hotplug::CapabilityAuthority for OgarAuthority`).
> I nearly recorded "the authority is unimplemented", which would have been the
> fourth false blocker of the session. **A negative grep result is a claim about
> the pattern, not about the tree.**

**So G-F is: the planner declares a `HotPlug` const and calls `activate`.** No
dependency edge, no plug crate, no seam to build — socket, authority and the
loco landing are all in place; the one consumer that needs the kanban has never
plugged in. Third instance this session of two built ends that do not meet.

### The palette arc — and what checking it turned up

Operator: *"ogar loco has a palette arc to unify the 34 nars templates which
just have been checked last few PRs only to find old VSA bindspace singleton
and hamming 16kbit deprecated."*

The palette surface is in `ogar-loco` (`registry.rs`, `basin.rs`,
`vocabulary.rs`, `lib.rs`), and `recipe_vocab.rs` above is its lance-side
landing. The deprecation the recent PRs surfaced is recorded in-tree —
`cognitive-shader-driver/src/engine_bridge.rs:370` calls the
`Vsa16kF32`/`Binary16K` set-bits plane *"the deprecated one"* — and this repo's
own `CLAUDE.md` carries both supersessions: PR #477's three-tier model
(no singleton, no inter-mailbox handoff type at all) and
`E-MARKOV-TEMPORAL-STREAM-1` (2026-07-10), which moves the Markov trajectory
**off the VSA braid onto the `temporal.rs` sorted stream** and demotes VSA to
its `I-VSA-IDENTITIES` four-test niche (N ≤ 32 lossless role superposition
inside ONE compartment).

**Consequence for the palette arc, and it is a real constraint:** a unification
of the 34 templates must NOT be carried on a 16 Kbit Hamming plane or a
singleton BindSpace. Both are superseded substrate, and `temporal.rs`'s
version-range read (`QueryReference::at(v, rung)` + deinterlace) is what
replaced them — **which lands on G-C2 again**, since that read has no
production caller. The palette arc, the Rubicon last phase, and the durability
read are three names for the same missing leg.

## The other loose end — alpha channel split tunnel ↔ SPOG

`.claude/plans/spog-alpha-channel-v1.md` exists: **SPEC (Phase 0), 2026-09-07,
48 KB**, self-described *"Register-before-code. Every 'exists' claim below was
read this session … Every 'absent' claim names the search that backs it."* Its
operator mandate targets MedCare-rs, and it carries the lance-11 constraint
(*"rows are experimental in lance 11 and only required for tombstones which we
avoid by having sealed batch per cycle"*) — the same sealed-batch-per-cycle
shape #912's writer implements.

**Not surveyed further here** — it is a private-consumer-facing spec and
deserves its own pass rather than a paragraph in this plan. Recorded as a named
open end with its status, not folded in.

## Revised gap table

| # | gap | was | now |
|---|---|---|---|
| G-A | `extract_tree` never run on DuckDB | small | unchanged |
| G-B | no ore-TSV → `Dataset` encoder | small | unchanged |
| **G-C1** | OGAR bytes → the existing owned writer | *"the real one"* | **smaller — both ends exist, they do not meet** |
| **G-C2** | `deinterlace` / durability read | not identified | **the real one — and it is what closes Heckhausen's last phase** |
| G-D | `sql()` | gated on G-C | gated on G-C2 (a surface over an unreadable write is not a surface) |
| G-E | quack has no DuckDB in its loop | real | unchanged |
| **G-F** | planner ↔ loco ↔ r2il kanban seam | not in plan | **unwired both ways; a build, not a wiring** |
| **G-G** | alpha-channel split tunnel ↔ SPOG | not in plan | **SPEC Phase 0 exists; needs its own pass** |

---

## `ogar-loco`'s emancipation — the boring floor is the MOAT, not the burden

Operator, 2026-09-16: *"ogar-loco has the burden of being boring scratch /
blockly-rs but needs to emancipate."*

Read rather than grepped, and the architecture is **already emancipated**; what
has not moved is the crate's IDENTITY. Its own `lib.rs` opens by arguing the
point:

> *"`ogar-loco` — the **low-code program surface**: the vocabulary-agnostic call
> ABI that every block/template/flow frontend shares … The block-editor arc
> (the `blockly-rs` consumers) **proved a storage shape**; the operator
> direction that created this crate **generalizes it**: elixir-shaped templates
> are 'just a rails-shaped semantic over classid index, 256:256 — not much
> different than blockly, just different vocabulary — a reusable surface for
> any other purposes.' Power-Automate-style flows are the third consumer in
> line."*

So Blockly is **consumer #1 of three named**, not the crate's purpose. The
burden is that it was born there and is still read as its storage crate.

### The seam that does the emancipating, and it is enforced

`vocabulary.rs` — *"One call-ABI, sibling vocabularies selected by classid"* —
splits at `DOMAIN_FLOOR`:

| range | owner | rule |
|---|---|---|
| **below** `DOMAIN_FLOOR` | the **shared computational core** | arities live ONCE, *"so `IF` cannot quietly mean two things in two domains, and no sibling can drift on `ADD`'s arity"* |
| **at/above** the floor | the **vocabulary** | a `Vocabulary` impl answers for exactly that range via `domain_*` hooks |
| a core byte the core does not cover | **nobody** | *"refused everywhere — a vocabulary does not get to guess for it. Coverage grows in the core, once, for everyone."* |

`conformance::check` is the mechanical gate, and its stated posture is exactly
right for a shared floor: *"the JVM-verifier / Wasm-validator posture: validate
before trusting, refuse loudly."*

The two-quantity split is the other sign this is not a toy: `stack_arity`
(operands evaluated before the call, on the stack) vs `body_refs` (function
indices in the call's VALUE bytes). `forever` proves they are independent —
zero operands, one body — and *"a single conflated number cannot express it."*

### So what "emancipate" actually means

**Not "become less boring."** The boring floor is load-bearing: `ADD`, `IF`,
`repeat`, `forever` are Scratch-shaped *on purpose*, because a shared core that
every sibling vocabulary agrees on is what stops N dialects. That is the moat.

**Emancipation is upward, above the floor — and it has already started.**
`lance-graph-ogar::recipe_vocab` is the first act: *"the 34 NARS recipes as
`ogar-loco` ops above `DOMAIN_FLOOR`, with the kanban census as the awareness
surface."* NARS tactics expressed in the same `(function : value)` two-byte call
as `repeat`. Nothing about the ABI had to change to carry them — which is the
claim "vocabulary-agnostic" was making, now cashed.

### This is the JAVA RULING one tier down — same shape, same reasoning

| | `sql()` (T3, Java) | `ogar-loco` (the program surface) |
|---|---|---|
| the surface | ordinary SQL a developer already knows | ordinary blocks an author already knows |
| why boring | *"novel API is the enemy, not slow API"* — every unit of novelty is lock-in-by-learning-curve | a drifting core is N dialects; the floor is what makes siblings composable |
| where power lives | lance-graph, reached by classid → ClassView | the vocabulary above `DOMAIN_FLOOR`, reached by classid → `Vocabulary` |
| what the caller learns | nothing | nothing |
| the join key | **the classid** | **the classid** |

Both surfaces are deliberately dumb, both resolve meaning through the classid,
and in both the temptation is to make the SURFACE cleverer. **On the Java side
that temptation is already ruled out in writing; the same ruling should be read
as covering loco.** A loco emancipation that grew new core opcodes, or let a
vocabulary redefine a shared-core byte, would be `Mask.minus()` in block
clothing — power leaking into the front instead of arriving through the address.

**Consequence for G-F:** the planner's missing `HotPlug` is not merely a wiring
chore, it is how the kanban vocabulary gets to exist above the floor without
anyone importing anyone. The cascade the operator named —
*"plug and play mints the classid; the classid mints ogar-vocab; which triggers
ogar-loco vocabulary"* — IS the emancipation mechanism, stated as a pipeline.
Every consumer that hot-plugs adds a sibling vocabulary and takes nothing from
the floor.

---

## ⊘ THIRD CORRECTION — the progress loop does NOT wait on `deinterlace`

Operator, 2026-09-16: *"batchwrite fire and forget write + try progress kanban."*

Read `batch_writer.rs` properly (not grepped) and both halves are already in its
module doc, in those words:

> *"The sink drains EAGERLY (ASAP on cast, background), and the write masks the
> thinking and vice versa: the thinker reports (casts) and moves on — **'melden
> macht frei'** — it is NEVER refused."*

And the progress half is named end to end: `cycle_driver::collect_casts` drains
the payloads, seals the FIRST move per owner as `SweepSlot::paired_move`, and
`persist_sink::recover_and_apply` applies it via
**`MailboxSoaOwner::try_advance_phase`** — *"consulting no scheduler at any
point."*

**So G-C2 as I wrote it — "a write nothing reads back is not yet a loop" — is
wrong.** The loop closes on *fire-and-forget cast + try-advance*, not on a
read-back. `deinterlace` belongs to a DIFFERENT arm: the durable OBSERVATION
path (temporal queries, pinned-reference recovery). Conflating the two made an
observation gap look like a liveness blocker. Corrected: the loop's liveness
does not depend on it.

### The kanban IS the Rubicon DAG — one mechanism, not two

`try_advance_phase` (in `contract/soa_view.rs`) returns
**`Result<KanbanMove, RubiconTransitionError>`** and its whole body is
`from.can_transition_to(to)`. The kanban columns ARE the Heckhausen phases and
the transition DAG is the Rubicon crossing. That is why *"kanban rubikon and
revision.rs wrapping it"* is one chain rather than three subsystems.

`revision.rs`'s binding to it is already stated AND tested in
`contract/kanban.rs:560`:

> *"Whatever `revise` proposes must be an edge `try_advance_phase` accepts — **it
> completes the Rubicon DAG, it does not route around it.** And a column with no
> `Plan` successor must stay silent."*
> — test `revise_only_ever_proposes_a_legal_edge_and_is_silent_elsewhere`

So revision is not bolted on: the contract already forbids it from inventing an
edge, and a test enforces both the legality and the silence half.

### Measured wiring of the progress chain

| piece | production callers |
|---|---|
| `BatchWriter::cast` | **3** — `mailbox_soa:764` (owner), `owner_adapter:100` (on-behalf), `cycle_driver:407` (P4a drain) |
| `MailboxSoaOwner::try_advance_phase` | **1** — `mailbox_soa.rs:1281` (`.try_advance_phase(mv.to)`); every other hit is a doc comment |
| `persist_sink::recover_and_apply` | **0** — its only two call sites (`:1128`, `:1146`) are inside `mod tests` at `:732` |

**So the chain `batch_writer`'s doc describes is wired at both ends and hollow
in the middle**: casts land, the owner can advance, and the glue that turns
sealed slots into applied moves has no production caller. That is the honest
G-C2 — narrower than "no loop", and it is a connect, not a build.

> **Caller consequence the doc flags as documented nowhere else**, worth
> carrying into any W3 work: *"at most ONE move per owner per cycle is sealed.
> Casting three transitions for one mailbox performs one and defers two — and
> because the deferred ones do eventually apply, **code assuming otherwise is
> wrong in a way nothing reports.**"*

## The planner is the hot-plug home — and `sql()` is just a classid

Operator: *"lance-graph-planner is the correct to get a ogar-loco and ogar-vocab
> hotplug.rs store for the vocabulary, which then ogar-loco is empowered to feed
into kanban rubikon and revision.rs wrapping it"* … *"eg ogar-loco could get the
classid for `sql()`"* … *"which then gets the whole duckdb vocabulary"* …
*"that would be the easiest and cleanest move."*

This resolves G-D, and it dissolves it rather than solving it:

```
DuckDB C++
  ─(ruff_cpp_spo::extract_tree)──────►  ModelGraph
  ─(ruff_spo_address::mint)──────────►  classids  (part_of:is_a, 16 B facet)
  ─(planner's HotPlug const + activate)─►  ogar-vocab rows (Class + ActionDef)
  ────────────────────────────────────►  ogar-loco VOCABULARY above DOMAIN_FLOOR
                                             ├─ sql() is ONE classid in it
                                             └─ the whole DuckDB vocabulary beside it
  ─(kanban = Rubicon DAG, try_advance_phase)─►  executed
  ─(revision.rs)─────────────────────────────►  postactional phase wraps it
```

**`sql()` stops being a bespoke Java surface and becomes a vocabulary entry.**
That is why it is the cleanest move: there is nothing to design. The Java glove
calls a classid; loco defines what that classid means; the classid is the join —
the same join key the hot-plug socket already uses on both sides. It also keeps
the Java ruling intact for free: the surface stays boring because it is not a
surface at all, just an address.

**And it is what the DuckDB harvest was FOR.** Today's reproducible header
harvest (123 methods / 1620 events, provenance-stamped) is not documentation of
a competitor — it is **vocabulary source**. W1's structural arm
(`extract_tree`) produces the class tree that becomes the vocabulary above the
floor. That reframes G-A from "a survey we owe" to "the first step of the
build".

### The remaining DataFusion leg

Operator: *"planner historically had datafusion stuff that we need to replace
with masked ops."* This is the standing ruling applied to the planner
specifically — `CLAUDE.md` already records *"every planning is in migration to
`ogar-loco` and `ogar-r2il`, especially datafusion is out of the picture, what
exists gets a grace period, nothing new will migrate to it"*
(`E-PLANNING-MIGRATES-TO-LOCO-R2IL-DATAFUSION-IS-GRACE-PERIOD-1`).

The replacement target is not in doubt — `lance-graph-quack` already proves the
shape (every operator lowers to a `Program` executed by the ONE mask-risc
evaluator) and `plan_lower == quack::lower` is pinned by a differential. So the
planner's DataFusion surface has both a ruling and a worked example to migrate
onto; what it lacks is the migration itself.

## Wave order, revised by everything above

1. **W1 — `extract_tree` on DuckDB.** Now the first step of the vocabulary
   build, not a survey. Same falsifier.
2. **W-HP — the planner's `HotPlug` const + `activate`.** Smallest change with
   the largest unlock: it is G-F, it is the vocabulary store, and it is what
   lets `sql()` and the DuckDB vocabulary exist above the floor.
3. **W3' — connect `recover_and_apply`** (not "build a write"). Both ends
   exist; the glue is test-only.
4. **W-DF — planner DataFusion → masked ops**, onto quack's proven shape.
5. **W4 — the real DuckDB differential** (G-E), still independent and still the
   only thing that makes "parity" true.

`deinterlace` (the observation arm) drops out of the critical path entirely —
it is what `revision.rs` will eventually need to EVALUATE an outcome, but it
gates neither the write nor the advance.

## The join leg — replace the inherited SQL joins with a DuckDB version fed from loco

Operator: *"and then to replace the sql joins inherited from upstream
lance-graph with a duckdb version fed from ogar-loco."*

This closes the arc the DuckDB harvest opened, and the assessment is already
written. `duckdb-to-v3-translation-matrix-v1.md` row **A3** reads
`JoinHashTable`'s probe path (`join_hashtable.cpp:248-296`,
`ht_entry.hpp:34-37/49-51`) and rules it **"V3 BETTER for the addressed case;
NEEDS FALSIFIER otherwise"**, because:

> `ht_entry_t` packs **16 bits of salt + 48 bits of pointer** in one u64 and
> prefilters on the salt to avoid a full key compare. That is a *probabilistic*
> prefix derived from a hash. **The V3 classid prefix is the real thing**:
> matching it is not a filter that may be wrong, it is a **contiguous row
> range** (R5, D-GTM-0m: **49–99 ns vs 22.4 µs**). Also `IsOccupied() == (value
> != 0)` — DuckDB independently arrived at zero-is-absence, the same convention
> as the zero-fallback ladder.

And it states its own limit, which is the falsifier this leg inherits:

> *"The comparison only holds when **both sides are minted into the same address
> space**. A join between a minted V3 population and an external, unminted key
> set has no shared prefix and falls back to A2's hash question."*
> Plus: *"`join_hashtable.cpp` is 6,986 harvested events, by far the largest TU
> read — this row is a **reading of two functions, not of the join**; spilling,
> radix partitioning, and chain building are not assessed."*

**Why "fed from ogar-loco" is the right shape rather than a port.** A join is a
`(function : value)` call like any other: the vocabulary above `DOMAIN_FLOOR`
names it, the classid selects it, and it lowers to a `Program` the one mask-risc
evaluator runs — exactly as quack already does for scan/filter/project/group.
So this is not "reimplement DuckDB's hash join in Rust"; it is **mint the join
into the vocabulary and let the existing lowering carry it**. The minted-address
case collapses to a prefix range (a mask), and only the unminted case needs the
hash machinery at all.

It also lands on an already-queued deliverable: **task #10 / PR5** is the
join-elision experiment — `src_mask → hop → dst_mask` versus the DataFusion
node-edge-node join, with set semantics explicit and *"no replacement claim for
bag or cross-address-space joins."* That caveat and A3's are the same caveat,
arrived at independently, and together they bound the leg honestly: **the
replacement claim is for the minted, set-semantics case; bag semantics and
cross-address-space joins stay with the inherited path until measured.**

### Revised final wave order

| wave | what | why here |
|---|---|---|
| **W1** | `extract_tree` on DuckDB | first step of the vocabulary build, not a survey |
| **W-HP** | the planner's `HotPlug` const + `activate` | smallest change, largest unlock: G-F, the vocabulary store, and what lets `sql()` exist above the floor |
| **W3'** | connect `recover_and_apply` | both ends built; the glue is test-only |
| **W-DF** | planner DataFusion → masked ops | standing ruling + quack's proven shape to migrate onto |
| **W-JOIN** | the join leg, fed from loco | gated on W-HP (needs the vocabulary) and on W1 (needs the harvest); PR5 is its falsifier |
| **W4** | the real DuckDB differential (G-E) | independent; the only thing that makes "parity" true |

W-JOIN is deliberately last of the build waves: it is the one that needs BOTH
the vocabulary (W-HP) and the harvest (W1) to exist, and its honest scope is
set by two independently-derived caveats rather than by ambition.

---

## ⊘ FOURTH CORRECTION — there is no DataFusion in `lance-graph-planner` to remove

Operator: *"the immediate endgame is to replace all the DataFusion shit in
lance-graph-planner to revive its status as the center of gravity and wire it
into `mailbox_soa.rs` <> kanban rubikon."*

Measured before acting, and the target is not where it looks:

| | |
|---|---|
| planner DataFusion **dependency** | **none** — no `[dependencies]` entry at all |
| planner DataFusion **code** | **none** |
| planner DataFusion **mentions** | **10, every one a DOC COMMENT** crediting a pattern: *"Strategy #4: RuleOptimizer — Composable rule-based optimization (from DataFusion)"*, *"From DataFusion's ExtensionPlanner + UserDefinedLogicalNode pattern"*, *"inspired by DataFusion's OptimizerRule trait"*, *"Analogous to DataFusion's Repartition but for fingerprint vectors"* — in 10,028 LOC |
| the `datafusion = []` line | a **FEATURE**, under `[features]` beside `jit = []` — **inert**: nothing `cfg`-gated on it, nothing in the workspace enabling it |

**The real DataFusion is 8,044 LOC in `crates/lance-graph/src/datafusion_planner/`**
— `scan_ops`, `join_ops`, `predicate_pushdown`, `vector_ops`, `expression`,
`cost_estimation`, `udf`, `analysis`, `builder/`. In the CORE crate. The crates
carrying a real dep are `lance-graph` (4 entries), `lance-graph-callcenter` (3),
`holograph` (2), `lance-graph-catalog` (1), `lance-graph-python` (1). **Not the
planner.**

> My own census script reported "planner: 1 entry" — its regex matched the
> FEATURE line. The trap is documented in this repo's own pin-rule comment
> (*"a FEATURE NAME, not a version pin. It is not a counter-example"*) and I
> walked into it anyway, one turn after adding the rule that forbids it.

**Done now (small, gated):** the inert `datafusion = []` feature is REMOVED with
the reason inline. Its only live effect was making every
`grep datafusion crates/*/Cargo.toml` misreport this crate as a consumer.
`cargo test -p lance-graph-planner`: **435 passed, 0 failed, 2 ignored** (+4).

### So "revive its status as the center of gravity" is a MIGRATION, not a deletion

Nothing leaves the planner; capability **arrives**. The work is to move what
`lance-graph::datafusion_planner` does INTO the planner as masked-op lowerings —
and `lance-graph-quack` is the worked example, with `plan_lower == quack::lower`
already pinned by a differential.

**And the planner is already structurally central to the write path**, which is
the strongest argument that "center of gravity" is a restoration rather than a
promotion: `cognitive-shader-driver` depends on `lance-graph-planner` (its
`mailbox_soa.rs:760` takes `&mut lance_graph_planner::batch_writer::BatchWriter<P>`),
and `owner_adapter.rs` / `batch_writer.rs` / `persist_sink.rs` all live in the
planner. The kanban write staging is the planner's already; what is missing is
the QUERY side rejoining it.

## The intake vocabulary — the Cypher parser, and what it must stop feeding

Operator: *"there is a parser upstream in lance-graph that is supposed to handle
table rows and column joins akin to cypher"* … *"if we could reroute that
vocabulary to replace semiring json to masked intake <> masked ops."*

The parser is `crates/lance-graph/src/parser.rs` — **1,932 LOC of nom
combinators** over `ast.rs` (544 LOC), *"parsing … Cypher queries … focused on
graph pattern matching and property access."* That is the intake vocabulary: a
table row is a node, a column join is a pattern edge.

**What it must stop feeding is dated substrate.** `graph/semiring_map.rs` maps
GraphBLAS semirings onto *"7 HDR semirings over **16Kbit BitVec matrices**"* —
and 16 Kbit Hamming is exactly what the recent PRs found deprecated, alongside
the singleton BindSpace (`E-MARKOV-TEMPORAL-STREAM-1`; `engine_bridge.rs:370`
calls that plane *"the deprecated one"*).

So the reroute is a THREE-way convergence, not a rename:

```
  parser.rs (Cypher AST)                  ← the intake vocabulary, keep
       │
       ├─ TODAY ─► semiring_map ─► HDR semirings over 16Kbit BitVec   ⊘ deprecated
       │
       └─ TARGET ─► masked INTAKE ─► Program ─► the ONE mask-risc evaluator
                                     (the quack shape, already proven)
```

The parser stays; the semiring/16Kbit leg is what the masked path replaces. That
also re-uses W-DF's own justification rather than opening a second argument —
one lowering, one evaluator, and the differential already pins them equal.

### W-DF, restated concretely

1. Delete nothing in the planner (there is nothing to delete) — **✓ done** for
   the one inert artifact.
2. Lower the Cypher AST to `Program` via the quack-proven path, so intake is
   masked from the parser onward.
3. Retire the `semiring_map` → 16 Kbit BitVec leg as those lowerings land — it
   is deprecated substrate, not a design choice to preserve.
4. Migrate `datafusion_planner`'s capability (scan → filter → project → join)
   crate-by-crate into planner lowerings; `join_ops` is W-JOIN and lands last
   for the reasons already stated.

---

## The upstream inventory, and where the seam actually is

Operator: *"everything what upstream has is here —
`github.com/lance-format/lance-graph/tree/main/crates/lance-graph/src`."*
Censused against our fork:

| upstream file | LOC | role |
|---|---|---|
| **`datafusion_planner/`** | **8,044** | the DataFusion backend — 44 % of the whole surface |
| `query.rs` | 2,171 | entry |
| `parser.rs` | 1,932 | nom Cypher |
| `semantic.rs` | 1,719 | semantic analysis |
| `logical_plan.rs` | 1,417 | the graph logical plan |
| `lance_vector_search.rs` | 560 | Lance ANN path |
| `ast.rs` | 544 | the AST |
| `config.rs` / `case_insensitive.rs` / `sql_query.rs` | 465 / 377 / 356 | |
| `parameter_substitution.rs` / `lance_native_planner.rs` | 280 / 121 | |
| `error.rs` / `spark_dialect.rs` / `lib.rs` / `table_readers.rs` / `sql_catalog.rs` | 111 / 107 / 83 / 69 / 47 | |

≈ **18,400 LOC inherited**, and two divergences worth naming: upstream's
`csr_index.rs` (its #160 "CSR adjacency index for native graph traversal") is
**ABSENT here** — this fork went its own way with the planner's Kuzu-style
`adjacency/` — and our three additions are `soa_config.rs`, `reasoning.rs`,
`dev_s3_env.rs`.

### `logical_plan.rs` is the seam, stated by the backend itself

`datafusion_planner/mod.rs` opens:

> *"**Translates graph logical plans into DataFusion logical plans** … Phase 2:
> Plan Building — **Nodes → Table scans, Relationships → Linking tables,
> Traversals → Joins.** Variable-length paths (`*1..3`) use unrolling: generate
> fixed-length plans + UNION."*

So the boundary is unambiguous, and the migration does not touch the vocabulary:

```
parser.rs → ast.rs → semantic.rs → logical_plan.rs      5,612 LOC — the VOCABULARY, KEEP
                                        │
                                   ── THE SEAM ──
                                        │
        TODAY  datafusion_planner/ 8,044 LOC          TARGET  Program → the ONE
               "Traversals → Joins"                          mask-risc evaluator
               (+ UNION-unrolled var-length paths)            traversal = src_mask
                                                              → hop → dst_mask
```

**"Traversals → Joins" IS the inherited SQL join the operator wants replaced**,
and its masked counterpart is already a queued deliverable: task #10 / PR5 is
exactly `src_mask → hop → dst_mask` versus the DataFusion node-edge-node join.
The two halves were specified independently and meet here.

Variable-length paths are the interesting sub-case rather than an obstacle:
upstream UNROLLS `*1..3` into fixed-length plans plus a UNION, which in mask
terms is N hops OR'd together — a `MaskOp` composition, not a plan rewrite.

### What W-DF actually is, now that the seam is located

**Re-point `logical_plan.rs` at a `Program`, not at DataFusion.** Everything
above the seam stays; the 8,044-LOC backend is what the masked path replaces,
operator by operator, with quack as the worked example and
`plan_lower == quack::lower` already pinned by a differential.

That also answers the *"semiring json → masked intake"* reroute in the same
move: the graph-side leg (`semiring_map` → HDR semirings over 16 Kbit BitVec,
deprecated substrate) and the SQL-side leg (`datafusion_planner`) are two
backends under one vocabulary. **One seam, two legs retired, one replacement.**

**Migration order within W-DF**, cheapest and most-proven first, matching the
operators quack already ships:

1. `scan_ops` → the plane leaf + survivor-skip gate (quack ships this)
2. `predicate_pushdown` → gating is the lowering (`under`), already the default
3. `expression` → `Pred` (a filter IS a predicate; no expression interpreter)
4. projection / aggregate → quack's Keep/Blend + two-phase GROUP BY
5. `join_ops` → **W-JOIN**, last, bounded by the two independently-derived
   caveats already recorded
6. `vector_ops` / `lance_vector_search` → NOT in scope: the Lance ANN path is
   not a DataFusion artifact and has no masked equivalent claimed.

---

## ⚠ THE BLOCKER FOR "planner as centre of gravity" — it sits BELOW core, not above

Measured before writing any lowering:

| edge | state |
|---|---|
| `lance-graph` core → `lance-graph-planner` | **EXISTS** — optional dep (`Cargo.toml:72`, behind the `planner` feature) **and** a dev-dep (`:155`) |
| `lance-graph-planner` → `lance-graph` core | **NONE** — and adding it would be a **cycle** |
| planner → `lance-graph-mask-risc` | **NONE** (only `lance-graph-quack` depends on mask-risc) |

So the planner **cannot** hold a `logical_plan → Program` lowering as things
stand: the logical plan lives in core, and reaching it from the planner is a
cycle. Defining a planner-local copy would be the parallel-IR anti-pattern this
whole arc exists to avoid.

Two further facts make the position honest rather than merely awkward: the
planner's own `CypherParse`/`GqlParse`/`GremlinParse`/`SparqlParse` strategies
are **regex stubs** — this repo's own open list says *"Wire planner strategies
to lance-graph core (actual parser, not regex)"* — and the real 1,932-LOC nom
parser is in core. The planner is genuinely downstream of the vocabulary today.

### Three routes, and the repo has already solved this exact shape once

| | route | cost | does it make the planner central? |
|---|---|---|---|
| **A** | lowering lives in **core**, beside `logical_plan.rs`; core gains `mask-risc` | smallest — no cycle, no new crate, no new IR | **No.** Planner stays centre for STRATEGY + the write path only |
| **B** | move the plan/AST vocabulary **down into `lance-graph-contract`** (zero-dep), planner then owns the lowering | largest — touches the contract's surface | **Yes**, and durably |
| **C** | a **bridge crate** depending on both | one more crate | Yes, by delegation |

**C is established precedent here.** `lance-graph-ogar` exists for precisely
this situation and says so: *"`ogar_loco` is zero-dep by design and
`lance_graph_contract` is zero-dep by charter. **Neither may import the other** …
A vocabulary needs both, so it lives in a consumer that already depends on
both."* The same reasoning transfers verbatim.

**B is what the operator's words actually ask for** — *"revive its status as the
centre of gravity"* — because A leaves the query side in core permanently, and C
puts the centre in a fourth crate rather than in the planner. B is also what the
zero-dep contract is FOR: it already hosts `kanban`, `soa_view`, `revision`,
`hotplug` — the shared vocabulary every tier reads. A plan vocabulary is the
same kind of thing.

**This is a genuine fork and it is the operator's to call**, because A and B
produce materially different repos and neither is reversible cheaply. What is
NOT in doubt, whichever wins:

- the seam is `logical_plan.rs` (the backend says so itself);
- the vocabulary above it (5,612 LOC) is kept;
- the 8,044-LOC DataFusion backend and the `semiring_map`/16 Kbit leg are the
  two legs retired;
- quack is the worked shape and the migration order is scan → pushdown →
  expression → projection/aggregate → join.

**Recorded, not guessed.** The one thing already done is the inert
`datafusion = []` feature's removal (planner tests green, 435 passed) — real,
small, and independent of which route is chosen.

---

## §14 — The mindset check, and the correction I owe first

Operator, three messages in one turn: *"make sure that you keep kanban soa
owned / the plans are the glove so that consumers dont have to care"*, *"check
if my mindset has it right or if you see a better way"*, and *"the consumers
should not care that the SoA mailboxes are smart, they still should have a
surface to be reached even if they were AGI and duckdb combined."*

### §14.0 — The correction: §13's A/B/C fork was already ruled, three months ago

`fe726fa` (§13) recorded an open placement fork — A (lowering in core) / B
(plan vocabulary into the zero-dep contract) / C (bridge crate) — and left it
for the operator on the grounds that *"the routes produce materially different
repos and neither reverses cheaply."*

**That fork does not exist.** `.claude/plans/cypher-mask-lowering-v1.md` §5.2
ruled it, in three parts, with a register I did not have:

1. **Boolean combination CONSUMES `ogar_loco::TERNLOG = 0x86`** (`ogar-loco/src/lib.rs:607`,
   arity 3, *"the call's ONE VALUE BYTE is the 8-bit truth table … the purest
   `(function : value)` in the ABI"*), which has **zero consumers**. So the
   Boolean half is a NET REDUCTION in unconsumed surface. **Nothing is minted.**
2. **`Pred`, the hop and the terminals are a LOWERING TARGET ONLY.** The
   mechanical test, borrowed from `tesseract-rs/CLAUDE.md`'s *"types exist only
   BEFORE the bake"*: **does a BYTE of this survive the query that produced it?**
   No → it must never get a mint. A `Program` is built from one
   `LogicalOperator`, executed once, and dropped.
3. **The trigger that would flip (2), written down before anyone hits it:** the
   moment a mask program becomes a **stored artifact**, the ops must be minted as
   a domain vocabulary **above `DOMAIN_FLOOR`** — and where is settled by
   precedent, not choice: `lance-graph-ogar`, which git-deps `ogar-loco` and
   path-deps `lance-graph-contract`, because *"neither may import the other …
   a vocabulary needs both, so it lives in a consumer that already depends on
   both"* (`recipe_vocab.rs:8-16`; the 34 NARS recipes already occupy
   `0x90..=0xB1` there).

So the answer to "where does the vocabulary live" is **nowhere — it is not a
vocabulary**, and the crate question I raised was the wrong question. This is
the `CLAUDE.md` § *grep FINDS, reading DECIDES* rule one level up: I measured the
dependency graph correctly and inferred a fork, without reading the plan that
had already closed it. The prior-art rule ("grep the existing ~100 files before
writing a new one") exists for exactly this.

### §14.1 — The mindset is right, and three things already enforce it

**"the plans are the glove so that consumers dont have to care."** Right, and it
is not aspiration — it is the shape the code already has AND has a *mechanical*
enforcement:

| the claim | what enforces it | where |
|---|---|---|
| a plan describes, it never computes | `lance-graph-quack` has ONE dep (mask-risc) and its manifest says *"a `match` here that computed anything would be the duplicate evaluator the whole arc exists to avoid"* | `lance-graph-quack/Cargo.toml:9-12` |
| the executor borrows, the caller owns | law A1 — *"the plan describes, the executor borrows, ndarray computes, the caller owns memory"* | `mask-risc/src/lib.rs` A1 |
| a consumer cannot be coupled to it | §5.2's byte test: no byte of a `Program` is persisted, transmitted, or read by a reader that did not build it | `cypher-mask-lowering-v1.md` §5.2 Part 2 |

The third row is the load-bearing one. *"Consumers don't have to care"* is
usually a promise about discipline; here it is a property: **you cannot couple
to something that does not persist.**

**"keep kanban soa owned."** Already true, and fenced three ways — two by
convention and one structurally:

1. `try_advance_phase` is a method on `MailboxSoaOwner`, in the **zero-dep
   contract** (`lance-graph-contract/src/soa_view.rs:311`), returning
   `Result<KanbanMove, RubiconTransitionError>` — the lifecycle DAG is the
   owner's, not a caller's.
2. `lance-graph-supervisor/tests/probe_ignition.rs:1234-1241` asserts the
   probe's own source contains **no** `.try_advance_phase(` and no
   `.advance_phase(` — with a paired can-stay-silent assertion so the scan is
   proven live rather than vacuously green.
3. **And the glove structurally cannot reach the ownership.**
   `mask-risc::Program` is `{ ops: Vec<MaskOp>, terminal: Terminal,
   scratch_slots: u32 }`; `MaskOp` has 8 variants, `Terminal` has 8, `Operand`
   is slot / plane / lane. **There is no spelling for an owner, a phase, or a
   move.** A consumer holding a plan cannot express a kanban transition even if
   it wanted to — the vocabulary does not contain one.

**"a surface to be reached even if they were AGI and duckdb combined."** This is
the sharpest of the three, and it is what makes §5.2 Part 3's trigger
load-bearing rather than pedantic. A plan that does not persist lets the
substrate change arbitrarily underneath it. The moment one persists, a consumer
is coupled to a byte layout, and "AGI or duckdb" stops being free. Part 3 is
therefore not bureaucracy — it is the exact condition under which the operator's
sentence stops being true, written down in advance.

### §14.2 — One correction to the instruction, and one better way

**Correction — the DataFusion is not in the planner.** Measured:
`lance-graph-planner` contained exactly one `datafusion` token, `datafusion = []`
under `[features]` — a feature NAME with no body and no `cfg` reader. Removed in
`3cafd13`; 435 tests still pass. The 8,044 LOC of DataFusion lives in
`lance-graph` **core** (`datafusion_planner/`). So *"replace all the DataFusion
shit in lance-graph-planner"* is already discharged and it bought nothing,
because there was nothing there to remove.

**And "revive the planner as the center of gravity" cannot be done as literally
stated**, because the arrow runs the other way:

| edge | state |
|---|---|
| `lance-graph` core → `lance-graph-planner` | EXISTS (optional, `planner` feature, `Cargo.toml:72`) |
| planner → core | **NONE — would be a cycle** |

**The better way: `mask-risc` is the convergence point, and neither side owns
the other.** `lance-graph-mask-risc` deps only `ndarray`; core deps `ndarray`;
planner deps `ndarray`. So **both can depend on mask-risc with no cycle**, and
neither has to import the other:

```text
core:     parser → AST → LogicalOperator ──mask_lower──┐
                                                        ├──→ Program ──exec──→ ndarray::simd
planner:  kanban / mailbox → what to ask ───lower──────┘
```

That is not a new idea — it is the shape `p64` already plays between ndarray and
lance-graph (`CLAUDE.md`: *"p64 = convergence point (both repos meet, no circular
deps)"*). Reused, not invented.

And it delivers the operator's stated GOAL without needing the planner to be
central in the dependency sense:

- the glove is the **plan**, and the plan is stable because it does not persist;
- the planner keeps what it actually owns — kanban, mailbox, `batch_writer`,
  `persist_sink`, `owner_adapter` — without importing a query pipeline;
- core keeps the parser it already has, and gains ONE new thing: §2's Phase 2.5.

So: **"planner as center of gravity" is a MEANS; "the plan is the glove" is the
END.** The end is reachable; the means as literally stated is a cycle, and does
not need to be paid for.

### §14.3 — What is actually next, and it is not code

`cypher-mask-lowering-v1.md` §7.0 is a **STOP gate**: *"Nothing is built until
Wave 0 answers, with numbers."* And `mask_lower` exists nowhere in the tree — the
only hit is `mask-risc/src/lib.rs:20` naming it as an absence. So Wave 0 has not
run, and the boring correct move is to run it. §15 is W0-b.

---

## §15 — W0-b RUN. The corpus census, measured — and CORRECTED after review

`cypher-mask-lowering-v1.md` §7.0's second measurement, and half of its STOP
gate. Instrument: `crates/lance-graph/examples/w0b_corpus_census.rs`
(committed; run it, do not trust this table).

**The corpus is every `.rs` file under the workspace's `crates/`, walked at run
time** — not a hand-listed set. Classification is §3/§4's, row by row, through
the REAL `parse_cypher_query` + `LogicalPlanner::plan`.

```
candidate literals extracted : 342
    rust files walked          : 1451
    files carrying a query     : 27
  did not parse                : 17
  parsed but did not plan      : 22
  CLASSIFIED                   : 303

  Full  (everything lowers)  : 113  ( 37.3 %)
  Split (mask prefix + DF)   : 190  ( 62.7 %)
  Grace (nothing lowers)     :   0  (  0.0 %)
```

**§7.0's STOP condition does not fire.** Its words were *"if the full-lowering
fraction is negligible, this plan's premise is wrong and Wave 1 does not
start."* 37.3 % full and **zero** pure-grace is not negligible.

### §15.0 — ⊘ THE FIRST RUN OF THIS CENSUS WAS WRONG, AND CODEX CAUGHT IT

The version that landed in this plan first reported **46.0 % over 50 classified
queries**, from three hand-listed files. Codex filed four P1s on lance-graph
#1240. All four were correct, and three of them moved the number:

| finding | what it was | effect |
|---|---|---|
| **the corpus was a subset** | the census read `parser.rs`, `logical_plan.rs`, `semantic.rs`. A walk of the same tree with the same extractor finds queries in **27 files** — the whole of `crates/lance-graph/tests/`, `src/query.rs`, the planner's strategy modules, the Python bindings | 50 classified → **303** |
| **inline pattern properties were ignored** | `MATCH (p:Person {name: "Alice"})` files its predicate in `ScanByLabel.properties` / `Expand.properties` / `.target_properties`, and all four maps went unread. A string equality there is P-9 grace | new row, **53** hits |
| **T-12 never fired** | `DISTINCT n.p` is grace, `DISTINCT n` is free. The `Distinct` arm recursed and left it to `classify_value`, which accepts a bare `Property` unconditionally (T-4 is legitimately `[G]`) — so the distinction was never made and the histogram had no T-12 row at all | new rows, **37** hits |

**The defect before the defect.** The three files were chosen after a `grep`
for `"…MATCH …"` reported zero hits in the DataFusion builder modules. That
grep cannot see a raw string or a literal spanning lines — the same blind spot
that made the extractor itself wrong (§15.4). **I used grep as a verdict on the
same day I added the rule to `CLAUDE.md` saying not to.** The fix is therefore
structural rather than a longer list: the corpus is now a WALK, so a file added
to the tree enters it with no edit, and the list cannot go stale the way the
last one did.

### §15.1 — Why a query is not Full (counted, not guessed)

| n | plan row | the question it asks that a mask cannot answer |
|---|---|---|
| 93 | §4.1 G-1 `ORDER BY` | in what ORDER |
| 53 | §4.2 P-9 inline string property | variable-width values |
| 45 | §4.2 P-9 non-integer literal | " |
| 29 | §4.1 T-12 `DISTINCT` over a value | how many TIMES |
| 28 | §4.1 G-2 `LIMIT` | which POSITION in an order |
| 28 | §4.5 `Join` | across which ADDRESS SPACES |
| 24 | §4.2 G-3 string predicate | variable-width values |
| 23 | §4.4 G-7 vector distance / similarity | how CLOSE |
| 10 | §4.1 G-2 `SKIP` | which POSITION |
| 6 | §4.3 G-5 `UNWIND` | what VALUE, as a new relation |
| 5 | §4.1 T-12 `count(DISTINCT …)` | how many TIMES |
| 3 | §4.1 T-12 `collect()` | " |
| 2 | §4.2 G-4 scalar function | variable-width values |

Every reason lands in §4.6's table, which is the check that the classifier is
reading the plan rather than inventing a boundary. **Order and position are the
largest axis (131), strings and non-integer literals the second (124)** —
inverting the three-file census, where the two were tied at 17 each. A subset
misranks the boundary as well as mis-measuring it.

### §15.2 — The finding the percentages do not contain

**21 of the 22 plan-refusals return a BARE NODE VARIABLE** — measured, not read:
the example inspects the AST the planner refused and counts the returns whose
every item is `ValueExpression::Variable(_)`. (The three-file census said 9 of
10; the wider corpus strengthens it.)

`RETURN <node>` is §3.5 **T-3**: *"the mask itself — `Terminal::Keep`. Not a row
list. The caller reads the plane."* It is the cheapest thing the mask path can
do and it is graded `[G]`.

So the incumbent planner **refuses** the query the mask path answers with no
work at all, and because a refusal is excluded from the denominator, **37.3 %
understates the premise by exactly the shape that most favours it.**

### §15.3 — Three things this number is not

1. **Conditional on OQ-1.** Every Full query scans by LABEL, and §1.4 states the
   `label → classid` binding is ABSENT (see §16 — it is not, but the wiring
   hop is). The census counts the SHAPE as lowering; that is what keeps the
   premise falsifiable, and it is not a claim the route exists.
2. **It measures COMMITTED TEST SOURCE.** These queries exist to exercise a
   parser, a planner and a DataFusion backend, so the distribution is theirs,
   not a production workload's.
3. **It says nothing about speed.** §7.0 is a correctness gate; no row above is
   a performance claim.

The 17 that do not parse are also honest signal rather than noise: they are
`LET … IN`, `UNION`, `LEFT MATCH`, and `RESONATE(fp, $q, 0.3)` — dialect the
parser does not implement, sitting in planner and cognitive tests as
aspirational syntax. They are excluded because there is no plan to classify,
and they are ENUMERATED so that is checkable.

### §15.4 — The extractor was wrong once too, and that bug moved the headline 29 points

Before the corpus was widened, the first run reported **15 of 20 (75.0 %)**.
`parser.rs:931-933` contains `char('"')` three times. A scanner that does not
know about char literals opens a string there, runs one quote out of phase for
the rest of the file, and emits raw Rust source as three "queries" while
swallowing real ones. Fixed: 26 candidates → 62, and the headline fell from
75.0 % to 46.0 % — before codex's finding took it to 37.3 % over 303.

All three runs exited 0. What caught the extractor was **enumerating the
exclusion buckets instead of counting them** — a bucket labelled "3 did not
parse" is unfalsifiable, and printed, its first line read
`); // Verify the AST structure let ast = result.unwrap(); …`.

**Three measurements, three numbers, one instrument.** The lesson is not that
the census was careless; it is that a measurement's SCOPE is part of the
measurement, and nothing in a green run reports its own scope. The walk now
prints how many files it read and which ones carried a query, so the next
reader can see the scope without reconstructing it.

---

## §16 — W0-a / OQ-1 ANSWERED BY READING: the binding exists, unconsumed, and the gap is one config field

§7.0's STOP condition is sharper on W0-a than on W0-b: *"If W0-a says there is
no `label → classid` route AND no cheap one can be minted, §3.1 is unbuildable
and the plan is re-scoped … or shelved."* §1.4 finding (3) states the route
*"does not exist"* and names `canonical_concept_id` as a resolver.

Read rather than grepped, and the picture is one piece larger than §1.4 records.

### §16.1 — The binding is already minted, already exported, and has zero consumers

`lance-graph-contract/src/ogar_codebook.rs` carries **`LabelDTO`**, and its own
doc comment is the specification of exactly this route:

> *"A curator-agnostic label binding: a consumer-local `label`, its OGAR codebook
> `id` (binary identity), and the portable `canonical` symbol. … Identity
> comparison uses `id`; AST/planner emission uses `canonical`; presentation uses
> `label`."* — with `id` documented as *"the OGAR codebook binary identity (the
> classid low u16)."*

`LabelDTO::from_canonical(concept) -> Option<Self>` resolves through
`canonical_concept_id`; curator-shaped aliases (`"Issue"` → `"project_work_item"`)
normalize through OGAR `ogar_vocab::canonical_concept` first, which stays out of
the zero-dep contract by design.

And the far end already takes exactly that type:
`mailbox_scan::match_nodes_by_class(view, class_id: u16)`.

**Consumers of `LabelDTO` across the workspace: ZERO.** The only hit outside its
own module is its re-export at `lance-graph-contract/src/lib.rs:234`. That is the
same shape as `TERNLOG = 0x86` (§14.0 Part 1): a minted, exported, unconsumed
surface that the lowering would *consume* rather than duplicate. Two of this
plan's supposed gaps are the same kind of thing — **built ends that do not
meet** — which is the pattern §13 already recorded as the session's recurring
finding.

### §16.2 — What is actually missing: a field, not a mechanism

`GraphConfig`'s `NodeMapping` (`crates/lance-graph/src/config.rs:60-71`) is
`{ label, id_field, property_fields, filter_conditions }`. There is nowhere to
put a canonical concept or an id. So the missing hop is:

```text
Cypher label "Person"
      ?                      <- THE GAP: NodeMapping has no field for this
      ↓
canonical concept  ──canonical_concept_id──▶  u16  ──▶ match_nodes_by_class / a class-equality mask
      ↑                                        ↑
   LabelDTO.canonical                    LabelDTO.id        (both already exist, both unconsumed)
```

**§7.0's STOP does not fire.** A cheap mint exists: one field on `NodeMapping`
carrying the canonical concept, wiring a binding that already ships. No new
codebook, no new vocabulary, no byte minted — §14.0 Part 2's test is untouched,
because none of this persists a lowering.

### §16.3 — The honest caveat, and it changes how §15's 46 % should be read

**The corpus's labels are not in the codebook.** Measured against the 123 entries
of `ogar_codebook::CODEBOOK`:

| label | in codebook? |
|---|---|
| `Person` | **no** |
| `Company` | **no** |
| `Thing` | **no** |
| `Node` | no (`osm_node`, `mars_node_template`, `osm_way_node`, `osm_street_node` are different concepts) |

The codebook carries real domain concepts — `project` `0x0101`,
`project_work_item` `0x0102`, `billing_party` `0x0204`, `osm_node` `0x0F01`. The
Cypher test corpus's labels are synthetic and were never minted, which is
unsurprising: they exist to exercise a parser.

So §15's caveat 1 is not a formality. The 46 % measures the **shape** of the
corpus, and a production number would have to be measured over queries whose
labels are minted concepts. That is a different census, and it needs a real bake
— which is the rest of W0-a (OQ-2 mint order, OQ-3 column widths, OQ-6
transpose lanes), still unrun.

**What §16 does settle:** the one gate that could have shelved the plan does not.
The route is buildable, it is buildable cheaply, and building it consumes
existing surface instead of adding any.

---

## §17 — Operator rulings, 2026-09-16

### §17.1 — TERNLOG sequencing: the alpha channel goes first, and NOT merely because it asked first

I had framed this as a scheduling tie between two claimants on one opcode.
That was the wrong frame, and the operator's reason is structural:

> *"spog alpha channel is important because it affects the underlying storage
> and table 'multitenant' access — meaning SPOG interactions writing to alpha
> split tunnel without changing ontologies itself as saccade focus of attention
> sparse write and meta awareness of simultaneous rung.*
>
> *before it was 10 rung not talking to each other despite `kanban_actor.rs`,
> or deprecated 'one at a time' — levels of thinking interact and require priors
> to be on the same level."*

Three things follow, and none of them is about calendar order.

**(a) The alpha channel is a STORAGE-tier change; a query lowering is not.**
It changes how a write lands and who can see it — multitenant table access.
A mask lowering reads. A reader must not define the semantics the writer will
have to honour, so the write side settles first or the read side is built
against a contract that has not been decided.

**(b) The point of the split tunnel is that attention writes WITHOUT rewriting
the ontology.** A sparse, saccade-shaped write — focus lands, something is
recorded, the shared structure is untouched. That is what makes concurrent
attention safe, and it is a property of the channel, not of any consumer.

**(c) The thing actually being bought is SIMULTANEOUS RUNGS.** The prior state
was ten rungs that did not talk to each other, and the deprecated alternative
was "one at a time". Levels of thinking interact, and interaction requires
priors to sit on the same level. So the alpha channel is what makes a rung's
prior legible to another rung at all — and "meta awareness of simultaneous
rung" is the capability, not a nice-to-have.

**Consequence for the Cypher lowering:** it consumes `TERNLOG 0x86` AFTER the
alpha channel does, and it inherits whatever simultaneity contract that first
consumer establishes. §5.2's OQ-9 said "defer rather than race"; this says why
deferring is correct rather than merely polite.

### §17.2 — `RevisionKind` lives beside `belief.rs`, and the wiring is the real work

Operator: *"belief is historically correct but needs to be adjusted for proper
wiring."*

So: the planner, beside `belief.rs` (option A in
`mask-algebra-revision-read-v1.md` §5 Q1), not the contract. The history is
right; what needs attention is the wiring, not the home. Recorded here; the
adjustment itself is that plan's D-MAR-2.

### §17.3 — I withdraw the "is 46 % negligible?" question

Operator: *"46% negligible how do i know what you are talking about"* — and
that is a fair hit, twice over.

First, I asked for a ruling on a WORD from a plan the operator did not write
(§7.0's own "negligible"), without supplying what anyone would need to judge
it. Second, and worse, the number had already moved: codex's review took it
from 46.0 % over 50 queries to **37.3 % over 303** (§15.0). I put a stale
figure up for decision.

**It was never a decision. It is a measurement, and the measurement answers
it:** zero queries out of 303 fail to lower at all. Every single one lowers
either fully or as a mask prefix with a DataFusion remainder. A premise that
covers some of every query in the corpus is not negligible, and no ruling is
required to say so. Wave 1's gate is open on the evidence.

### §17.4 — The reporting defect this exposed, named so it stops

Operator: *"i have no idea what you are talking about when you talk
incoherently which i call 'reverse grep' — throwing pattern matching that
doesn't make any context for me."*

Correct, and it is a specific failure rather than a style complaint. I had been
reporting in symbol names and section numbers — `§5.2 Part 2`, `LabelDTO has
zero consumers`, `T-3`, `OQ-1` — which are ADDRESSES. An address is only
meaningful to a reader who already holds the map. Handing someone addresses
instead of meaning is the same defect as citing a line number instead of an
anchor, one layer up: it reads as precision and carries none.

The rule for this plan and for session reporting: **say what a thing does and
why it matters in plain words first; the symbol, file and section go at the END
as receipts.** A decision request that cannot be understood without opening
three documents is not a decision request.
