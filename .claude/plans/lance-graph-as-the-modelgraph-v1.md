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

**So `Dataset::write` is real and in-tree**: `crates/lance-graph/src/graph/cycle_sink.rs:675`
and `:710`, under a module doc saying the I/O *"is INTERNAL to this one writer."*
The plan's G-C above — *"no `Dataset::write`"* — was **wrong when written.**

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
