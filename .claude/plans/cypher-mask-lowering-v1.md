# cypher-mask-lowering-v1 — Cypher lowers to a MASK PROGRAM over `ndarray::simd`; DataFusion keeps only what cannot be Boolean

> **Status:** PLAN (pre-5+3-council). No code. No mint. No cargo run in this repo
> during authoring.
> **Register discipline:** every "exists" claim carries `file:line`. Anything
> argued rather than read is marked **[claimed, unverified]**. Measured numbers
> are quoted with their source; nothing here is a performance claim.
> **Builds on, does not contradict:** `.claude/plans/cypher-kanban-ast-unification-v1.md`
> (one IR, four relationships; `Backend::MailboxSoa`; F1 backend parity).

---

## §0 — Why this plan exists, in one operator sentence

`CLAUDE.md:1017-1025` — **OPERATOR RULING 2026-09-05**, verbatim: *"Every planning
is in migration to ogar-loco and ogar-r2il, especially datafusion is out of the
picture, what exists gets a grace period, nothing new will migrate to it."*
Board: `E-PLANNING-MIGRATES-TO-LOCO-R2IL-DATAFUSION-IS-GRACE-PERIOD-1`.

That ruling names the destination (`ogar-loco` / `ogar-r2il`) and the disposition
of the incumbent (grace period, maintained, never extended). It does **not** say
how a `MATCH` becomes a program. This plan says how, for the one query surface
this repo owns end to end.

The thesis, and it is not new here — it is `mask-risc-lowering-v1.md` §2's
semiring fence read from the Cypher side:

> **A graph pattern is a Boolean question about a population. A Boolean question
> about a population is a mask. Therefore a Cypher `MATCH` is a mask program, and
> the only parts that are not are the parts that were never Boolean.**

`ndarray/.claude/blackboard.md` entry **D-GTM-0j** (§12.4 of the ternlog plan; the blackboard is prepend-ordered, so cite the entry id, never a line) is the type boundary in
measured form: *"masks win whenever the relation is Boolean; a relation that
carries VALUES needs a value-aware algorithm"* — explicitly a TYPE boundary, not
a density one (no crossover: 745× at 0.02 % density, 297× at 100 %). §3 below is
that boundary applied construct by construct; §4 is everything on its far side,
and that is exactly the grace path.

### §0.1 — The litmus tests this plan is written against

`CLAUDE.md` § The Click:

> *Does this add a free function on a carrier's state, or a method on the carrier?
> → Free function = reject. Method = accept.*

Applied: a mask program is not a `plan(ast, config, substrate)` free function. It
is `MaskProgram::lower(&CypherQuery)` → `program.execute(&Planes)` — the program
carries its own ops and its own terminal, and the executor borrows planes it does
not own (`lance-graph-mask-risc/src/lib.rs:20-26`). §6 pins this.

`CLAUDE.md` § The falsifiability rule: *"An assertion implied by the code it tests
is not a test."* §7 carries a can-fire / can-stay-silent pair per lowering rule, an
anti-vacuity bound per filter, and a disable-run per guard — and §7.5 records the
two ways a disable can silently not be one.

---

## §1 — WHAT EXISTS (documentation register; every row cites source)

### §1.1 — The Cypher path as it runs today

| piece | where | what it actually does |
|---|---|---|
| The nom parser | `crates/lance-graph/src/parser.rs:23` `parse_cypher_query` | the real parser; 66 647 bytes of source. Produces `ast::CypherQuery` |
| The AST | `crates/lance-graph/src/ast.rs:15` `CypherQuery`; `:137` `NodePattern`; `:166` `RelationshipPattern`; `:192` `LengthRange`; `:236` `BooleanExpression`; `:314` `ValueExpression` | full surface: labels, inline properties, direction, var-length, WHERE, WITH, RETURN, ORDER BY, SKIP/LIMIT |
| Semantic analysis | `crates/lance-graph/src/semantic.rs:74` `analyze`; `:587` `validate_property_reference`; `:733` `validate_length_range` | validates against `GraphConfig`; substitutes parameters |
| The graph logical plan | `crates/lance-graph/src/logical_plan.rs:19` `LogicalOperator` (12 variants); `:168` `LogicalPlanner::plan` | `ScanByLabel` · `Filter` · `Expand` · `VariableLengthExpand` · `Project` · `Join` · `Distinct` · `Sort` · `Offset` · `Limit` · `Unwind` |
| **The seam this plan intercepts** | `crates/lance-graph/src/datafusion_planner/builder/mod.rs:30-105` `build_operator` | ONE `match` over all 12 `LogicalOperator` variants → `datafusion::logical_expr::LogicalPlan` |
| The three-phase driver | `crates/lance-graph/src/query.rs:920-952` `create_logical_plans` | Phase 1 semantic → Phase 2 `LogicalOperator` → Phase 3 DataFusion. **Phase 2.5 is where a mask program goes** |
| The DF lowering bodies | `datafusion_planner/builder/{basic_ops.rs,expand_ops.rs,aggregate_ops.rs,join_builder.rs}` (2 476 lines total) | `Expand` → a join; `VariableLengthExpand` → unroll + union |
| The backend router | `crates/lance-graph/src/graph/graph_router.rs:46-62` `Backend` | `DataFusion` · `Blasgraph` · `Palette` · **`MailboxSoa`** — the fourth already landed per the kanban plan's Inc 0 |
| The MailboxSoa node scan | `crates/lance-graph/src/graph/mailbox_scan.rs:77` `match_nodes_by_class` | **the defect this plan exists to fix — see §1.4** |

### §1.2 — The T1 masking vocabulary (`ndarray::simd`, the ISA membrane)

Three-layer contract, operator-ruled, `ndarray/.claude/blackboard.md:181-195`:
consumers → `simd_masking_ops.rs` (ergonomics, never an ISA) → `simd.rs`
(architecture-agnostic lane types) → `simd_{avx512,avx2,neon,wasm,scalar}.rs`
(peers). **POLYFILL LAW** (`:196-202`): every public primitive has compile-time
implementations for all five backends; **scalar is a peer, not a fallback**;
*"`TERNLOG` stays semantic above the backends"*. **BACKEND LAW** (`:203-207`): no
shared runtime body the backends delegate into.

Every symbol this plan's lowering table names, with its line:

| group | symbols (`ndarray/src/simd_masking_ops.rs`) |
|---|---|
| compare → mask, contiguous | `eq_u32_to_mask:164` · `ne_u32_to_mask:1044` · `eq_i32_to_mask:1017` · `ne_i32_to_mask:975` · `gt_i32_to_mask:340` · `ge_i32_to_mask:920` · `lt_i32_to_mask:881` · `le_i32_to_mask:947` |
| compare → mask, **strided** (the 512-byte row) | `eq_u32_strided_to_mask:231` |
| ternary / care-masked (TCAM) | `ternary_match_u32_to_mask:1285` · `ternary_match_u64_to_mask:1338` · `ternary_match_strided_to_mask:1415` (`pattern`/`care` are `&[u8;12]` — the V3 register, exactly) |
| mask algebra | `mask_and:377` · `mask_or:410` · `mask_xor:1139` · `mask_andnot:508` · `mask_not:1072` (takes `n_rows` — see §5.3) · `mask_ternlog::<IMM>:588` · `mask_ternlog_assign::<IMM>:623` |
| tests | `mask_any:1215` · `mask_all:1242` |
| masked reductions | `masked_sum_i32:692` · `masked_min_i32:1502` · `masked_max_i32:1523` · `masked_strided_group_sum:779` |
| selection without compaction | `blend_i32:1582` |
| popcount | `ndarray/src/bitwise.rs:274` `popcount_batch_u64` |

Truth-table immediates, `ndarray/src/simd.rs:588-611` — and they live on the
**facade**, not a backend, deliberately (`:584-586`: *"a backend-resident module
is compiled out whenever another backend is selected, which is exactly what
happened to this module's first home in the scalar backend"*):

`AND3 = 0x80` (`:590`) · `AND2_ANDNOT = 0x40` (`:592`) · `AND_ANDNOT2 = 0x10`
(`:594`) · `OR2_AND = 0xA8` (`:596`) · `XOR3 = 0x96` (`:598`) · `MAJ3 = 0xE8`
(`:600`) · `AND2 = 0xC0` (`:602`) · `OR3 = 0xFE` (`:604`) · `XOR_AND = 0x28`
(`:608`) · `AND2_OR = 0xEA` (`:611`).

Index convention, `simd.rs:579-581`: `index = (a << 2) | (b << 1) | c`, result bit
`(IMM >> index) & 1`. **That is a complete 3-input Boolean basis**: any function of
three leaves is one immediate, hence one pass. §3.3 is built on exactly that.

### §1.3 — The mask side of the house

| piece | where | state |
|---|---|---|
| `lance-graph-mask-risc` op vocabulary | `crates/lance-graph-mask-risc/src/ir.rs` — `Operand:7` · `LaneRef:16` · `Planes:44` · `Pred:56` · `MaskOp:82` · `Terminal:108` · `Program:132` · `OpHistogram:186` | **CODED and coherent.** `Pred` has 10 variants incl. `MatchU32/MatchU64`; `MaskOp` has 8; `Terminal` has 8 |
| …its four laws | `lib.rs:18-39` | *"The plan describes, the executor borrows, ndarray computes, the caller owns memory"*; masks-choose-admissibility; `TERNLOG` is semantics; reference semantics are independent |
| …**and it did not build** (as of this read — ⊘ fixed by PR2 `c095dcc`: only `ir` is declared now, the crate is a workspace member, builds, and is clippy/fmt/test-gated in CI) | `lib.rs:56-61` declared `pub mod exec; pub mod fuse; pub mod hop; pub mod ir; pub mod reference; pub mod ternlog_table;` — **only `ir.rs` and `lib.rs` existed on disk** (verified by directory listing) | five of six modules were absent; `pub use exec::…` / `fuse::…` at `:63-64` could not resolve |
| …**and it was not in the workspace** (as of this read — ⊘ PR2 adds it to `members`) | `Cargo.toml:2-28` members, `:29-…` exclude — `lance-graph-mask-risc` appeared in **neither** list | an orphan directory at the time: nothing compiled it, nothing gated it |
| Shipped T1 consumer inside this repo | `crates/lance-graph-planner/src/nested_bands.rs:32` imports `gt_i32_to_mask, le_i32_to_mask, mask_and, mask_ternlog, popcount_batch_u64`; `:161` `mask_ternlog::<AND_ANDNOT2>`; `:439` `::<AND2>` (⊘ now `mask_and`, the exact name, after the PR2 council); `:623` is a `#[cfg(test)]` alias of `AND2`, not a third immediate | **the precedent: `lance-graph-planner` already depends on `ndarray` (`Cargo.toml:24`) and already calls T1 by name.** No new dependency edge is needed for the planner half |
| `AlphaMask` | `crates/lance-graph-contract/src/alpha.rs:224-230` — `words: Box<[u64]>`, `len: u32`, tail bits PHANTOM and every op that could raise them must clear them | the contract-side mask carrier, same LSB-first order as `ndarray::simd` (`mask-risc/src/lib.rs:4-6`) |
| The hop, as an ABI | `lance-graph-java/native/lgj-abi/src/exports.rs:2053` `lgj_hop(store, edge_classid, facet_mask, decode_mode, src_mask, dst_mask)` | the `src_mask → hop → dst_mask` shape, shipped |
| …its selection algebra | `exports.rs:2026-2032` + `:2104-2118`: `selected_f = ternlog<AND3>(class_f, src, struct_f)`, `dst = ⋁_f scatter(selected_f)`; *"No row is examined to decide whether it participates"* | **the pattern §3.4 reuses verbatim** |
| …and its scar tissue | `exports.rs:2034-2044`, `:2118-2136`: three prior shapes each traded algebra for arithmetic; a per-row gather *"measured faster on the AoS store and shipped briefly; the operator ruled it out"* | read this before proposing a walk |
| The one-crossing chain | `exports.rs:1762-1782` `lgj_plan_eval` — *"accumulator starts as all rows set; each op is evaluated and combined per its `combine` field"*, monotone `V(k+1) ⊆ V(k)` | the accumulator-starts-all-set convention §3.1 row N1 adopts |

### §1.4 — Three findings that change what this plan has to be

**(1) The planner's `cypher_parse` strategy does not parse Cypher.**
`crates/lance-graph-planner/src/strategy/cypher_parse.rs:35-52` is
`let q = input.context.query.to_uppercase()` feeding `has_graph_pattern = q.contains("MATCH")`
and its sibling `contains`/`matches` calls; its own
comment at `:67-68` says *"Real implementation: call lance-graph's
parser::parse_cypher_query() to produce a full AST. For now, feature detection is
the output."* `arena_ir.rs:40` says the same (*"Real implementation receives AST
from CypherParse strategy"*), and `collapse_gate.rs:34-45` / `sigma_scan.rs` are
comment-only bodies returning `input` unchanged.

Consequence: **there is exactly one real Cypher lowering in this repo** (the absence half of this is §1.5's `[claimed, unverified]` grep), and it
runs `parser.rs` → `semantic.rs` → `logical_plan.rs` → `datafusion_planner`. A
mask lowering must attach to THAT chain (`query.rs:920-952`), not to the planner's
strategy list. The strategy list is where a mask program gets *selected*, later,
and only once it exists.

**(2) `Backend::MailboxSoa`'s node scan is a materialised population.**
`mailbox_scan.rs:77-91`:

```rust
pub fn match_nodes_by_class<V: MailboxSoaView>(view: &V, class_id: u16) -> Vec<NodeMatch> {
    let classes = view.class_id();
    classes.iter().take(view.n_rows()).enumerate()
        .filter_map(|(row, &c)| (c == class_id).then_some(NodeMatch { row, backend: Backend::MailboxSoa }))
        .collect()
}
```

`clam_contained:115`, `cakes_nearest:136`, `members:285` are the same shape. Each
is a per-row scalar loop that ends in `.collect()` into a `Vec` of row indices —
i.e. the exact thing `lance-graph-java/CLAUDE.md`'s mask-native invariant forbids
as normal execution state (*"a `long[]` of selected row IDs is still a
materialised population"*), reproduced here in Rust. The doc comment's
zero-value-decode claim is true and is not the issue; **the output type is.**

So the mask lowering is not merely "a faster backend". It is the correction of a
shape defect that already shipped on the backend the kanban plan opened.

**(3) There are TWO classid widths and the label→classid binding does not exist.**

- `MailboxSoaView::class_id() -> &[u16]` (`lance-graph-contract/src/soa_view.rs:100-102`)
  is a **default method aliasing `entity_type()`** (`:89`) — a `u16`.
- `class_view.rs:54` `pub type ClassId = u16`.
- But `le-contract.md:12-30` §1 says the facet's 4-byte prefix **is the composed
  classid u32** (canon hi u16 = domain:appid, custom lo u16 = the ClassView
  selector), and the resolvers are `ogar_codebook.rs:757`
  `canonical_concept_id(concept: &str) -> Option<u16>` and `:398`
  `render_classid_for_concept(app, concept) -> Option<u32>`.
- And `GraphConfig` (`crates/lance-graph/src/config.rs:35-57`) maps a label to a
  **table name plus an `id_field: String`** — `NodeMapping` at `:60-71` has
  `label`, `id_field`, `property_fields`, `filter_conditions`. **No classid
  anywhere.**

`MATCH (n:Person)` therefore has no mechanical route to a classid today. That is
OQ-1 (§8), it is the first thing Wave 0 measures, and nothing in §3's node rows
can be built before it is answered.

### §1.5 — Absence claims, graded honestly

**[claimed, unverified]** — these rest on greps this document cannot re-run, and a
Wave-0 worker re-runs them before anything is built:

- No mask lowering from `LogicalOperator` exists anywhere in `crates/`.
- No consumer of `ogar_loco::FnIndex::TERNLOG` exists in any of the three repos.
  The only OGAR-side occurrences are the mint itself (`ogar-loco/src/lib.rs:607`),
  its arity row (`vocabulary.rs:98`), its name row (`vocabulary.rs:347`) and two
  comments (`lib.rs:596`, `vocabulary.rs:92`, `vocabulary.rs:1431`) — i.e. the
  opcode is minted and **unconsumed**.
- No `u16`- or `u8`-width compare-to-mask exists in T1. Supported by
  `ndarray/.claude/blackboard.md:49-52`, which states the limit from the other
  side: *"The T1 compare is i32-wide, so the u8 permeability column is widened 4×
  for `gt_i32_to_mask` … a u8/u16 compare-to-mask is a T1 addition."*

---

## §2 — THE SEAM (where a mask program attaches, and to what)

```text
  parser.rs:23                    ast::CypherQuery                    UNCHANGED
        │
  semantic.rs:74                  validated + parameters substituted  UNCHANGED
        │
  logical_plan.rs:168             LogicalOperator (12 variants)       UNCHANGED
        │
        ├──────── NEW: Phase 2.5 ───────────────────────────────────────────────
        │         mask_lower(&LogicalOperator, &Binding) -> LoweringOutcome
        │              ├─ Full(MaskProgram)        every construct lowered
        │              ├─ Split(MaskProgram, LogicalOperator)
        │              │                            mask prefix + DF residue
        │              └─ Grace                     nothing lowered
        │
        ├─ Full/prefix → MaskProgram::execute(&Planes) → mask + terminal
        │
        └─ Grace/residue → datafusion_planner/builder/mod.rs:30 build_operator
                                                     ← UNCHANGED, maintained
```

Four properties of this seam, each deliberate:

1. **It is BELOW the four front-ends.** Gremlin / SPARQL / GQL already converge on
   the same planner IR per `cypher-kanban-ast-unification-v1.md`'s own council
   verdict (*"all 4 polyglot parsers already return one `PlanInput`/`QueryFeatures`
   IR"*, line 126). Attaching at `LogicalOperator` means they inherit the lowering
   without a second lowering table. **We claim nothing further about those three
   front-ends in v1** — see §6 N-9.
2. **It is ADDITIVE.** `build_operator` is not edited. A `Grace` outcome leaves
   today's behaviour byte-identical, which is what makes §7's Wave-0 differential
   possible at all — both paths have to be runnable side by side.
3. **`Split` is the load-bearing variant.** The realistic query is not "all mask"
   or "all SQL": it is `MATCH (n:Person) WHERE n.age > 30 RETURN n.name ORDER BY
   n.name`. The `MATCH` + `WHERE` half is a mask; the `ORDER BY` is not. §4 is the
   list of what lands on the right of that split, and §3 row R-5 is why the mask
   half can be handed across it without materialising.
4. **It respects the grace period literally.** DataFusion gains no new behaviour,
   loses no existing behaviour, and is still the default for every query the
   lowering declines. "Out of the picture" is executed as *stops growing*, which
   is what the ruling's own next clause says (*"what exists gets a grace period"*).

---

## §3 — (a) THE LOWERING TABLE

Column key: **Cypher construct** → **mask op(s)** (T1 symbol by name) → **status**.
Status is `[G]` grounded (every symbol cited above exists), `[H]` hypothesis
needing a measurement named in §8, `[GRACE]` routed to DataFusion (§4 holds the
reasons in full).

### §3.1 — Node patterns and label selection (7 rows)

| # | Cypher | lowers to | status |
|---|---|---|---|
| **N-1** | `MATCH (n:Label)`, rows minted in classid order | **a RANGE mask** — set bits `[lo, hi)` directly, no compare at all. `ndarray/.claude/blackboard.md:18-22`: a contiguous prefix reveal measured **49–99 ns** vs **22.4–22.8 µs** for the general `ternary_match_u32_to_mask` sweep — **228–462×**; *"a minted, Morton-keyed tenant never pays it"* | `[H]` — mechanism `[G]`, the ordering precondition is **OQ-2** |
| **N-2** | `MATCH (n:Label)`, arbitrary row order | `eq_u32_strided_to_mask(bytes, 0, 512, n_rows, classid, out)` — offset 0 because `le-contract.md:12-30` puts the composed classid u32 at the facet's byte 0; stride 512 = the CANON row (`CLAUDE.md` § Minimal SoA node: 16 B key + 16 B edges + 480 B value) | `[H]` — mechanism `[G]`, the label→classid route is stated ABSENT by §1.4 (3); **OQ-1** |
| **N-3** | `MATCH (n)` — no label | the all-ones mask. Adopts `lgj_plan_eval`'s convention verbatim (`exports.rs:1768`: *"the accumulator starts as all rows set"*), so an unlabelled scan costs a memset, not a sweep | `[G]` |
| **N-4** | `MATCH (n:A:B)` — multi-label | two label masks + `mask_and`. Three labels = **one** `mask_ternlog::<AND3>` pass, not two ANDs — the `lgj_hop` spelling (`exports.rs:2026-2032`) | `[G]` |
| **N-5** | `(n:Label {k: v})` — inline property | the label mask and the property mask are two leaves; **fuse them with the WHERE tree** (§3.3) rather than ANDing eagerly. `ast.rs:143` `properties: HashMap<String, PropertyValue>` is the source | `[G]` |
| **N-6** | `()` / `(n)` inside a path, unbound | all-ones; contributes no pass | `[G]` |
| **N-7** | subtree / ancestry selection (`is_a`-shaped label hierarchies) | a **prefix range mask** over the HHTL key, not a compare. `mailbox_scan.rs:115` `clam_contained` is today's scalar spelling of this; `contract/src/hhtl.rs` `NiblePath` + `rail_geometry.rs:178-180` `is_ancestor_of` are the arithmetic. Enumerable at tier granularity: 3 tiers × 3 ternary states = 9 primitives, 18 further cells an AND of three (`mask-risc-lowering-v1.md:459-486` §14.1-14.2) | `[H]` — depends on N-1's ordering and on a real HHTL-populated bake |

### §3.2 — Property predicates (9 rows)

`PropertyRef` is `ast.rs:220-225` (`variable`, `property`); the comparison
operators are `ast.rs:303-312`.

| # | Cypher | lowers to | status |
|---|---|---|---|
| **P-1** | `n.p = <int>` on an `i32`-aligned lane | `eq_i32_to_mask:1017` | `[G]` |
| **P-2** | `n.p <> <int>` | `ne_i32_to_mask:975` | `[G]` |
| **P-3** | `n.p > / >= / < / <=` | `gt_i32_to_mask:340` · `ge_i32_to_mask:920` · `lt_i32_to_mask:881` · `le_i32_to_mask:947`. Signed by construction — `ir.rs:57` names it (*"ordered compares are signed"*) | `[G]` |
| **P-4** | `n.p = <int>` on a `u32` lane (ids, classids) | `eq_u32_to_mask:164` · `ne_u32_to_mask:1044` — exact bitwise (`ir.rs:19`) | `[G]` |
| **P-5** | any of P-1..P-4 where the value lives **inside the 512-byte row** rather than in a contiguous column | `eq_u32_strided_to_mask:231` at `(first_offset, 512, n_rows)`. This is the normal case for the V3 substrate and the reason the strided form exists | `[G]` |
| **P-6** | a **bit-pattern** predicate — "these bits of the register, don't-care the rest" | `ternary_match_u32_to_mask:1285` / `ternary_match_u64_to_mask:1338` / `ternary_match_strided_to_mask:1415` (whose `pattern`/`care` are `&[u8;12]` — the V3 payload exactly). Corresponding IR: `Pred::MatchU32/MatchU64` (`ir.rs:74,76`). **No Cypher surface syntax reaches this today** — it is what a rail/facet predicate lowers to once §8 OQ-4 picks a spelling | `[H]` |
| **P-7** | `n.p IN [a, b, c]` | N compare→mask sweeps, `mask_or`-accumulated. **This is the shipped crosswalk shape**, `spog-alpha-channel-v1.md:176-189` §3.2: *"`eq_u32_strided_to_mask` on the FK column … for each needle of the incoming survivor key set, `OR`-accumulated into `sweep`"* | `[G]` · cost is **OQ-8** |
| **P-8** | `n.p` on a `u16` / `u8` lane — **including `MailboxSoaView::class_id()`, which is `&[u16]`** (`soa_view.rs:100`) | **no T1 op of that width exists.** Two routes: widen the column 4× and use `eq_i32_to_mask` (the measured workaround, `blackboard.md:49-52`, which makes any cost figure an UPPER bound), or add a T1 primitive. **The STOP rule fires** — see §6 N-10 | `[H]` · **OQ-3** |
| **P-9** | `n.p = <float>` / `<string>` / `n.p` on a variable-width lane | **[GRACE]** — §4.2 | `[GRACE]` |

### §3.3 — Boolean combination: truth-table fusion (9 rows)

`BooleanExpression` is `ast.rs:236-283` — `And` / `Or` / `Not` over `Comparison`
leaves. Every row below is **one physical pass over the mask words**, because
`simd.rs:579-581`'s index convention makes any 3-input Boolean one immediate.

| # | Cypher WHERE shape | immediate | T1 call | status |
|---|---|---|---|---|
| **B-1** | `A AND B` | `AND2 = 0xC0` (`simd.rs:602`, `c` ignored) or `mask_and:377` | 1 pass | `[G]` |
| **B-2** | `A OR B` | `mask_or:410` | 1 pass | `[G]` |
| **B-3** | `NOT A` | `mask_not:1072` — **takes `n_rows`**, see §5.3 | 1 pass | `[G]` |
| **B-4** | `A AND B AND C` | `AND3 = 0x80` (`:590`) | **1 pass, not 2** | `[G]` |
| **B-5** | `A AND B AND NOT C` | `AND2_ANDNOT = 0x40` (`:592`) | **1 pass, not 3** | `[G]` |
| **B-6** | `A AND NOT B AND NOT C` | `AND_ANDNOT2 = 0x10` (`:594`) | **1 pass, not 4** | `[G]` |
| **B-7** | `(A OR B) AND C` | `OR2_AND = 0xA8` (`:596`) | **1 pass, not 2** | `[G]` |
| **B-8** | `(A AND B) OR C` | `AND2_OR = 0xEA` (`:611`) | **1 pass, not 2** | `[G]` |
| **B-9** | **any** other 3-leaf Boolean | evaluate the subtree's truth table over the 8 assignments → one `IMM: u8` → one `mask_ternlog::<IMM>:588`. `mask-risc/src/lib.rs:34-37` already names the mechanism (*"turns any Boolean subtree over three leaves into one `MaskOp::Ternlog` by evaluating its truth table"*) | 1 pass | `[G]` mechanism · the fuser module is **absent on disk** (§1.3) |

**The fusion rule, stated so it is checkable:** a WHERE tree of `k` leaves lowers
to `k` predicate sweeps plus `⌈(k-1)/2⌉` combination passes, never `k-1`. Three
predicates is **one** combination pass. This is the single largest structural
claim in §3 and §7's F-B1 is its falsifier.

**What this is NOT.** `blackboard.md:26-28` measured the ratio on a real chain:
`step = x·ternlogq + n` with `ternlogq = 291 ns/pass` and `n = 17.3 µs`, so at
`x = 1` *"the chain is 1.7 % of the step"*. Fusing 3 passes into 1 saves ~580 ns
against a ~17 µs step. **Truth-table fusion is a correctness and shape property,
not a speed argument** — the speed lives in N-1's range reveal and in not
materialising. Any PR body claiming otherwise is over-selling it.

### §3.4 — Relationships: `src_mask → hop → dst_mask` (9 rows)

`RelationshipPattern` is `ast.rs:166-177`; `RelationshipDirection` is `ast.rs:181`;
`LogicalOperator::Expand` is `logical_plan.rs:46-66`.

The shape is `lgj_hop`'s, verbatim (`lance-graph-java/native/lgj-abi/src/exports.rs:2053`),
and its selection algebra is the thing to copy (`exports.rs:2026-2032`):

```text
selected_f = mask_ternlog::<AND3>(class_f, src, struct_f)      // ONE pass per facet
dst        = ⋁_{f ∈ participation} scatter(selected_f)
```

— *"No row is examined to decide whether it participates; participation is computed
for the whole population and intersected."* The only walk left is the scatter, and
only because the destination index is **decoded** from the selected row, making it
*"the operand of a permutation, not a decision about which rows take part"*.

| # | Cypher | lowers to | status |
|---|---|---|---|
| **R-1** | `(a)-[:R]->(b)` | `src_mask` (= a's mask) → hop over R's edge facets → `dst_mask`; then AND with b's label mask | `[G]` shape, `[H]` in-repo — the symbol is in `lgj-abi`, not `lance-graph` (§6 N-11) |
| **R-2** | `(a)<-[:R]-(b)` | the hop over the **TRANSPOSE**, never the same relation read backwards. `blackboard.md:414-417` is the measured trap: *"the GEMM arm computes `{i : srcs(i) ∩ active ≠ ∅}` while the mask arm was unioning `srcs(i)` over active i — those agree only for a SYMMETRIC relation … The mask arm must union the TRANSPOSE"* | `[G]` as a rule · `[H]` whether a transpose lane exists in a real bake — **OQ-6** |
| **R-3** | `(a)-[:R]-(b)` undirected | R-1 ∪ R-2: two hops, `mask_or`. **Not** one hop over a "both" flag | `[G]` shape · `[H]` in-repo (inherits R-1/R-2) |
| **R-4** | `-[:R1\|R2]->` multi-type | one hop per type, `mask_or`-accumulated into `dst`. Mirrors `lgj_hop`'s own per-facet `⋁_f` | `[G]` shape · `[H]` in-repo (inherits R-1/R-2) |
| **R-5** | `-[r {k: v}]->` relationship property filter | an extra leaf in the per-facet conjunction: `ternlog::<AND3>(class_f, src, prop_f)` and then AND `struct_f` — the SAME two-predicate pattern `lgj_hop` already runs (`class_f` at facet base +0, `struct_f` at base +12) with one leaf substituted | `[G]` |
| **R-6** | `(a)-[:R]->(b)-[:S]->(c)` — fixed 2-hop | chain: `dst₁` becomes `src₂`. Target-label masks AND in **between** hops, never after, so hop 2's frontier is already narrowed | `[G]` shape · `[H]` in-repo (inherits R-1/R-2) |
| **R-7** | `-[:R*1..k]->` bounded variable length | iterate R-1 `k` times over a **DELTA frontier**, not the accumulated state: `frontier ← mask_andnot(dst, state)`, `state ← mask_or(state, dst)`, stop when `!mask_any(frontier)`. `blackboard.md:33-36` measures exactly this — *"the **NNUE reading** — spread from the DELTA frontier (`scratch & !state`), never from the accumulated state — gives the identical closure (gate green) at **8.8 µs (−48 %)**"* | `[G]` mechanism · `[H]` on graph shape |
| **R-8** | `-[:R*]->` unbounded | R-7 to fixpoint. Termination is `mask_any(frontier) == false` (`mask_any:1215`) — a **bit test over a 8 KiB plane**, not a visited-set lookup. Today's DF path unrolls and unions (`builder/expand_ops.rs`, `logical_plan.rs:72-91`: *"implemented by unrolling into multiple fixed-length paths and unioning them"*); a mask fixpoint has no unroll bound at all | `[G]` mechanism · `[H]` |
| **R-9** | `WHERE` applied to a hop result | the survivor mask is just another leaf: `mask_ternlog::<AND3>(dst, pred_a, pred_b)`. **This is the crosswalk's own shape** — `spog-alpha-channel-v1.md:182-184`: *"`mask_ternlog::<AND3>(&sweep, &tenant_mask[n], &rung_gate, &mut survivors)` — the survivors' key set is the needle set of hop n+1"* | `[G]` shape · `[H]` in-repo (inherits R-1/R-2) |

**The forbidden move, quoted because it is one line away from every one of these
rows.** `spog-alpha-channel-v1.md:186-187`: *"The forbidden move is a mask-`AND`
across two tables."* A mask is an index into ONE population at ONE address space.
R-1's post-hop `AND` with b's label mask is legal **only** because `dst` and the
label mask index the same row store. The moment a pattern spans two address spaces
it is §4.5 `[GRACE]`, not a clever mask.

### §3.5 — Returns and terminals (12 rows)

`ReturnClause` is `ast.rs:409-415`; `ValueExpression::AggregateFunction` is
`ast.rs:329-336`; `classify_function` is `ast.rs:374-382` (`count|sum|avg|min|max|collect`
are aggregates). `Terminal` is `mask-risc/src/ir.rs:108-128`.

| # | Cypher | lowers to | status |
|---|---|---|---|
| **T-1** | `RETURN count(*)` | `popcount_batch_u64` (`ndarray/src/bitwise.rs:274`) over the final mask. `Terminal::Count` (`ir.rs:110`) | `[G]` |
| **T-2** | `RETURN count(n)` | identical to T-1. **Because there is no NULL**: `mask-risc/src/lib.rs:44-47` — *"absence in the V3 substrate is a zero-fallback, never a validity bit, so DuckDB's three-valued AND/OR collapses to Boolean algebra"*. `count(n)` and `count(*)` cannot differ | `[G]` |
| **T-3** | `RETURN n` | **the mask itself** — `Terminal::Keep` (`ir.rs:127`). Not a row list. The caller reads the plane | `[G]` |
| **T-4** | `RETURN n.prop` | **a masked projection that stays `(mask, lane_ref)`** — the pair, never an index list. `Planes::lanes` (`ir.rs:50`) + `LaneRef` (`ir.rs:16-23`) is the carrier. Materialisation exists but is **named**: exactly one symbol whose name starts with `materialize`, O(n) stated in its doc (`lance-graph-java/CLAUDE.md`'s Materialisation exception) | `[G]` shape · the named materializer is **new code**, §7 W1 |
| **T-5** | `RETURN sum(n.p)` on an `i32` lane | `masked_sum_i32:692` — widened to `i64`, *"carry-safe for every `i32` input"* (`mask-risc/src/lib.rs:49-50`) | `[G]` |
| **T-6** | `RETURN min(n.p)` / `max(n.p)` on `i32` | `masked_min_i32:1502` / `masked_max_i32:1523` — `Option`, `None` on an empty mask | `[G]` |
| **T-7** | `RETURN avg(n.p)` on `i32` | `masked_sum_i32` + `popcount_batch_u64`, divide at the terminal. Two reductions, one pass each, no intermediate | `[H]` — the result type is float; whether the caller wants exact rational or `f64` is **OQ-7** |
| **T-8** | `RETURN sum(…)` over a **grouped** lane inside the 12-byte register | `masked_strided_group_sum:779` (`group_bytes` 1..=4, asserted at `:783`) | `[G]` |
| **T-9** | `EXISTS { MATCH … }` / any existence test | `mask_any:1215` — one early-exiting word scan, no count | `[G]` |
| **T-10** | `CASE WHEN <pred> THEN a ELSE b` | `blend_i32:1582` — **no compaction**; `Terminal::BlendI32` (`ir.rs:124`) writes into a caller buffer | `[G]` |
| **T-11** | `RETURN DISTINCT n` (a node variable) | **free — the identity.** A mask IS a set: a row is in it or it is not, and there is no multiplicity to collapse. `DISTINCT` over a node variable lowers to nothing at all | `[G]` |
| **T-12** | `RETURN DISTINCT n.p` / `count(DISTINCT n.p)` / `collect(…)` | **[GRACE]** — §4.1. Distinct over VALUES needs value identity, which a population mask does not carry | `[GRACE]` |

### §3.6 — Constructs that do not lower (7 rows, pointer only — reasons in §4)

| # | Cypher | disposition |
|---|---|---|
| **G-1** | `ORDER BY` (`ast.rs:427-440`) | **[GRACE]** §4.1 |
| **G-2** | `SKIP` / `LIMIT` (`ast.rs:29,33`) | **[GRACE]** §4.1 — with one carve-out stated there |
| **G-3** | `STARTS WITH` / `ENDS WITH` / `CONTAINS` / `LIKE` / `ILIKE` (`ast.rs:257-281`) | **[GRACE]** §4.2 |
| **G-4** | `toLower` / `toUpper` and every scalar string fn (`ast.rs:323` `ScalarFunction`, classified at `:377`) | **[GRACE]** §4.2 |
| **G-5** | `UNWIND` (`ast.rs:119-124`, `logical_plan.rs:28-35`) | **[GRACE]** §4.3 |
| **G-6** | `WITH` as an aggregation boundary (`ast.rs:398-406`) | **[GRACE]** §4.3 |
| **G-7** | `vector_distance` / `vector_similarity` (`ast.rs:343-356`) | **[GRACE]** §4.4 — and it is the *interesting* one |

**Row count: 7 + 9 + 9 + 9 + 12 + 7 = 53 lowering rows.** Counted by the status in
each row's own cell: **31 `[G]`**, **13 `[H]`** (⊘ PR2 council: R-3/R-4/R-6/R-9 are compositions of the `[H]` rows R-1/R-2 and N-2's precondition is stated ABSENT by §1.4, so five rows regrade down; nine of the thirteen name a measurement in §8 — R-1, R-7, R-8, R-9 and N-7's bake precondition are answered by OQ-12),
**9 `[GRACE]`** (the 7 rows of §3.6 plus P-9 and T-12, which reach the same fence
from inside the predicate and terminal groups).

---

## §4 — (b) WHAT CANNOT LOWER, AND WHY

The organising principle is the blackboard's D-GTM-0j entry: **the boundary is a
TYPE boundary, not a density one.** A mask answers *which rows*. It cannot answer
*in what order*, *how many times*, or *what value*. Everything below is one of
those three questions wearing different syntax.

### §4.1 — Order and multiplicity: `ORDER BY`, `LIMIT`, `SKIP`, `DISTINCT <value>`, `collect`

A mask is an unordered set with no multiplicity. `ORDER BY` demands a total order
over values; `SKIP`/`LIMIT` demand a *position* in that order; `collect()` demands a
sequence. None is expressible in the vocabulary at all — `mask-risc/src/lib.rs:51-52`
says so as a scope statement: *"Nothing else — no strings, no dictionaries, no ORDER
BY, no bag-semantics joins."*

**The carve-out, and its fence.** `LIMIT k` with **no** `ORDER BY` is "any k
survivors", which a bounded bit-walk answers. That walk is a materialiser and must
be the named one (T-4), it must state its `O(k)` cost, and it must never be reached
implicitly. `LIMIT` **with** `ORDER BY` is `[GRACE]`, full stop: the k that survives
depends on the order, so the walk would return a different answer than DataFusion
and §7's differential would — correctly — go red.

`count(DISTINCT x)` is the sharp case: `count(*)` is a popcount, and the two differ
by exactly the multiplicity a mask threw away. Do not let the first tempt anyone
into the second.

### §4.2 — Strings and variable-width values

Every T1 compare is fixed-width over a numeric lane (`i32` / `u32` / `u64`).
`STARTS WITH` is a prefix test over variable-length bytes; `CONTAINS` is a search;
`toLower` is a transformation producing a new value. None of these is a
fixed-offset fixed-width comparison, and faking one — hashing the string into a
`u32` and comparing — changes the ANSWER (collisions admit rows the predicate
excludes), which is a correctness change dressed as an optimisation.

**A real seam, named but not claimed:** an interned-dictionary column would make
string equality a `u32` compare and `STARTS WITH` a range over the dictionary's
sort order. That is a substrate change (a dictionary lane is a new lane), it is
governed by the STOP rule (§6 N-10), and it is **not in this plan**.

### §4.3 — Bag semantics: `UNWIND`, `WITH`-as-aggregation

`UNWIND` turns one row into n rows — the population grows, and a mask over a fixed
population cannot express a population that changes size. `WITH` as an aggregation
boundary produces a *new* relation whose rows are groups, not the original rows;
the mask indexes the old population and has nothing to say about the new one.

This is the same fence `mask-risc/src/lib.rs:24-26` draws for `SelectionVector`:
*"there is no per-row object, no hidden rowset, no second row-index universe — a
`SelectionVector` cannot be expressed in this vocabulary at all."* A grow-the-rows
operator would require exactly that second universe.

### §4.4 — Values that are not Boolean: vector distance, similarity, NARS truth

This is the row that matters most, because it is the one a future session will try
to collapse. `E-TOPOLOGY-MASKS-MAGNITUDE-COMPOSE-NEVER-COLLAPSE-1`
(`.claude/board/EPIPHANIES.md`, cited **by E-id, not by line** — see §9's note), operator
ruling, binding:

> *Topology chooses neighborhood, masks choose admissibility, BLAS chooses
> magnitude.* … *the MQ hexagon does not become a TERNLOG immediate; the immediate
> does not become a neural weight; BLASGraph is never used for Boolean elimination
> because a matmul can encode it.* … **mask bit = projection bit** (true by the
> immediate's index construction) — **NEVER mask bit = weight**.

And the falsifier it hands the reviewer (same entry): *"A design in
which one of the three does another's job … is the collapse this entry forbids;
the reviewer's question on every PR is 'which of the three is this, and does it do
only that'."*

So `vector_distance(n.v, $q) < 0.3` **splits**: the distance is a scoring kernel over
a value lane (BLAS side), the `< 0.3` is a threshold producing a mask
(`lt_i32_to_mask`-shaped, once the score exists), and the composition is two ops,
not one. `mask-risc-lowering-v1.md:121-123` §2's semiring table is the same fence
from the other direction: Boolean and XOR semirings are exact; *"HammingMin,
SimilarityMax, Resonance, NarsTruth — **no**: a value per cell, not a bit"*, and
*"Anyone claiming HammingMin is an AND fails this fence."*

`[GRACE]` here means: the threshold half is a mask, the scoring half is not, and v1
routes the whole construct to DataFusion rather than building half a splitter.

### §4.5 — Cross-address-space joins

`LogicalOperator::Join` (`logical_plan.rs:100-104`) over two disconnected patterns
in **different** row stores is `[GRACE]` by the forbidden-move rule quoted in §3.4:
masks from two populations have no common index, and ANDing them is a silent
type error that produces a plausible-looking wrong answer.

A `Join` over two patterns in the **same** population is not a join at all — it is
a conjunction, and it lowers as B-1. The lowering must distinguish these by asking
the binding which store each side indexes, never by whether the operator is spelled
`Join`.

### §4.6 — The honest summary of the boundary

| the question the construct asks | mask can answer | example |
|---|---|---|
| which rows? | **yes** | `MATCH`, `WHERE`, hops |
| how many rows? | **yes** (popcount) | `count(*)` |
| is there any row? | **yes** (`mask_any`) | `EXISTS` |
| what is the total / min / max of a lane over those rows? | **yes** (masked reduction) | `sum`, `min`, `max` |
| in what ORDER? | no | `ORDER BY` |
| which POSITION in that order? | no | `SKIP`/`LIMIT` after `ORDER BY` |
| how many TIMES (multiplicity)? | no | `collect`, `count(DISTINCT)` |
| what VALUE, as a new relation? | no | `UNWIND`, `WITH`-aggregation |
| how CLOSE / how MUCH? | no — that is a different mechanism | vector distance, NARS truth |
| across which ADDRESS SPACES? | no | cross-store `Join` |

---

## §5 — (c) THE PLACEMENT RULING: where the lowered opcodes live

### §5.1 — The question, stated so it can be answered wrongly

`lance-graph-mask-risc/src/ir.rs` declares `MaskOp` (8 variants, `:82-103`), `Pred`
(10 variants, `:56-77`) and `Terminal` (8 variants, `:108-128`). `ogar-loco`
declares a one-byte function codebook (`FnIndex`, `lib.rs:375`) with a **permanent**
split at `DOMAIN_FLOOR = 0x90` (`lib.rs:348`, const-asserted at `:353-357`:
*"DOMAIN_FLOOR is stored-byte ABI: moving it reinterprets every persisted program;
mint inside the existing ranges instead"*). Below the floor is the shared
computational core; at and above it is the classid-selected vocabulary
(`lib.rs:98-104`).

Is `ir.rs` therefore **a second vocabulary beside `ogar-loco`** — the very thing
`ogar-r2il` was written to avoid (`ogar-r2il/src/lib.rs:86-89`: *"mirroring `r2il`'s
77 variants here would create the second vocabulary this crate exists to avoid, and
every mirror is a drift surface"*)?

### §5.2 — THE RULING, in three parts

**Part 1 — Boolean combination CONSUMES the already-minted `TERNLOG = 0x86`. It does
not mint.**

`ogar-loco/src/lib.rs:607` `pub const TERNLOG: FnIndex = FnIndex(0x86)`, arity 3
(`vocabulary.rs:98`), documented at `lib.rs:600-606` as *"the call's ONE VALUE BYTE
**is the 8-bit truth table** … one FnIndex covers all 256 stacked-mask combinators,
which is the purest `(function : value)` in the ABI."*

That is §3.3 exactly — nine rows, one opcode, the immediate as the value byte. And
it is **shared core**, so it means the same thing in every vocabulary
(`lib.rs:612-617` `is_shared_core`), which is precisely the property a cross-domain
Boolean combinator needs.

It currently has **zero consumers** — §1.5's grep, independently re-run across lance-graph, lance-graph-java and OGAR by the PR2 council's boundary audit (2026-09-14): every `TERNLOG` hit is `VPTERNLOG`/`mask_ternlog` or ogar-loco's own tables, never the `FnIndex` constant; verified. `spog-alpha-channel-v1.md:215-219` §3.5
already names the first one as future work — *"the first `ogar-r2il` consumer
through `lance-graph-ogar` = `RANK` to admit + `TERNLOG 0x86` per hop, with the
falsifier that a lifted crosswalk program yields the hand-written chain's survivor
mask bit-for-bit"*. **This plan is the second claimant on the same slot, and it
should defer to that one rather than race it** (§8 OQ-9).

So: the Boolean half of the lowering is a NET REDUCTION in unconsumed surface, not
an addition. Nothing is minted for it.

**Part 2 — `Pred`, the hop, and the terminals are a LOWERING TARGET ONLY. No byte,
no codebook, no mint.**

The mechanical test, borrowed intact from a sibling repo's own ruling (`tesseract-rs/CLAUDE.md`
§ *"Types exist only BEFORE the bake. Afterwards there are only classes"* — *"The
mechanical test: does this type still exist after the bake?"*), restated for this
boundary:

> **Does a BYTE of this survive the query that produced it?**
> If yes, it is a vocabulary and needs a mint.
> If no, it is a lowering target and must never get one.

A `MaskProgram` is built from one `LogicalOperator`, executed once, and dropped. No
byte of it is persisted, transmitted, or read by a foreign reader. `ir.rs`'s own
header already asserts the weaker half (`:1-4`: *"Every op names its operands by
**slot** … nothing in the IR owns bytes"*).

That makes it the same kind of object as DataFusion's `Expr` — an in-process Rust
enum consumed inside one crate — and minting a persistent one-byte codebook for it
would be the layout-reclaim hazard `DOMAIN_FLOOR`'s const-assert exists to forbid:
bytes minted with no producer are exactly what was torn down at `0x87..0x8B`
(`ogar-loco/src/lib.rs:593-598`: *"the pair-specific band minted there on
2026-09-01 was retracted on 2026-09-02 with the model it encoded (reserve, don't
reclaim …)"*; the five names — `BELNAP_JOIN`, `INFO_GAIN`, `SIGMA_TENSION`,
`ACCUMULATE`, `STANCE_ENTROPY` — are catalogued in `mask-risc-lowering-v1.md:78` as
*"no producer, no basin"*).

**The insufficiency argument the STOP rule demands** (`rubicon-loco-rung-cognitive-fabric-v1.md:584`
`F-RLR-2`, quoted in `mask-risc-lowering-v1.md:96`: *"a new carrier is proposed
before `ogar_loco` is proven insufficient — automatic STOP"*), stated explicitly
rather than assumed:

| need | is `ogar_loco` sufficient? |
|---|---|
| 3-input Boolean over masks | **YES** — `TERNLOG` 0x86, all 256 tables in one opcode. Consume it (Part 1) |
| compare a value lane to a constant → mask | **NO.** loco's `Call` carries at most 3 immediate bytes (`MAX_VALUES_PER_CALL = 3`, `ogar-loco/src/lib.rs:635`), and an `i32` threshold does not fit. A wide literal spends the value byte as a **constant-pool index** (`lib.rs:91-95`), which is expressible — but the lane reference, the width, the signedness and the stride are three more operands. Not sufficient **as one call** |
| hop over an edge lane | **NO** — no such op below the floor, and none should be: a hop's operand is an adjacency structure, not an immediate |
| masked reduction | **NO** below the floor |

Conclusion: loco is sufficient for the part this plan consumes and insufficient for
the parts it keeps internal — which is the correct shape, because the insufficient
parts are precisely the ones that should not be persisted bytes.

**Part 3 — the TRIGGER that would flip Part 2, written down before anyone hits it.**

The moment a mask program becomes a **stored artifact** — cached in a version-keyed
trie, written to Lance, sent across a process boundary, or read by any reader that
did not build it — Part 2 is void and the ops MUST be minted as a **domain
vocabulary above `DOMAIN_FLOOR`**, not below it (a graph-query lowering is
domain-specific; it is not *"everything that is the same no matter what the bytes
mean"*, `ogar-r2il/src/lib.rs:10-12`).

Where it would live is already settled by precedent, not by choice:
`crates/lance-graph-ogar/src/recipe_vocab.rs:8-16` states the hosting rule —
*"`ogar_loco` is zero-dep by design … and `lance_graph_contract` is zero-dep by
charter. **Neither may import the other** … A vocabulary needs both, so it lives in
a consumer that already depends on both"* — and demonstrates it: the 34 NARS
recipes occupy `0x90..=0xB1` in `lance-graph-ogar`, which git-deps `ogar-loco`
(`Cargo.toml:137`) and path-deps `lance-graph-contract` (`:159`). A mask-query
vocabulary takes the next free domain band in its own `Vocabulary` impl, in that
same crate, under its own classid.

### §5.3 — Three substrate laws the op set must obey (each has cost a session real time elsewhere)

**(a) The tail law.** `mask_not:1072` takes `n_rows` as a parameter, and
`AlphaMask` documents why (`alpha.rs:226-229`): *"Bits at and past `len` are PHANTOM
and every op that could raise them ([`Self::not`]) must clear them — a complement
that forgets the tail word invents up to 63 addresses the spine never had."* Every
`NOT` in §3.3 (B-3, B-5, B-6, and any `IMM` whose truth table maps `(0,0,0) → 1`)
raises tail bits. **`Program` must carry `n_rows` and the executor must clear the
tail after any complementing op** — `ir.rs:96` says `Not` clears it; a fused
`Ternlog` with a complementing immediate does not say so, and that is a gap to close
in Wave 1, with a falsifier (§7 F-X3).

**(b) The one-population law.** Restated from §3.4: a mask indexes ONE address
space. The forbidden move is `mask_and` across two tables
(`spog-alpha-channel-v1.md:186-187`). Encode it in the type: an op's operands carry
a population id, and combining two different ids is a compile-time or
construction-time refusal, never a runtime surprise.

**(c) The transpose law.** An incoming hop unions the transpose, not the forward
relation read backwards (`blackboard.md:414-417`, where the mistake *"failed
immediately (912 vs 930)"* and agreed only for a symmetric relation). The lowering
must name which lane it hops over; "reverse the direction flag" is the shape that
was measured wrong.

---

## §6 — (e) NON-GOALS (explicit, each with its why)

**N-1 — This does not deprecate DataFusion.** The ruling says *grace period*, and
`cypher-kanban-ast-unification-v1.md:107-108` says *"**datafusion is NOT deprecated**
(#540). The router gains a backend; it does not lose one."* §4's `[GRACE]` rows are
maintained, not migrated, and not extended.

**N-2 — No new set algebra.** `dismech-causal-replay-v1.md:70` — *"No second set
algebra: Mengenlehre = `EvidenceMask` ops"* — owns set algebra repo-wide. The mask
program must be shown to BE that algebra, not a second one. If it cannot be, that is
a finding, not a licence.

**N-3 — No mask cache, trie, or memo.** `mask-risc-lowering-v1.md` §14's voxel cube
and §6's trie are a **sibling repo's** plan with its own gates (D-MRL-0b′, G9). This
plan builds a lowering; caching what it produces is a separate wave with a separate
falsifier and is explicitly out of scope. Note the reason it must stay out:
`mask-risc-lowering-v1.md:65` records that the `mammal` reuse claim was written
*"in the present indicative about a thing that does not exist"*, and the corrected
reading is that the ALGEBRA is measured 7/7 while the REUSE half is unmeasured.

**N-4 — No layout change.** No new node or edge type, no stride change, no
`ENVELOPE_LAYOUT_VERSION` bump. Everything in §3 reads the CANON 512-byte row
(`CLAUDE.md` § Minimal SoA node) and the 4+12 facet (`le-contract.md:12-30`) as they
stand. `le-contract.md:335-338` RESERVE-DON'T-RECLAIM applies unchanged.

**N-5 — No SIMD in `lance-graph`.** Every vector op comes from `ndarray::simd`
(POLYFILL LAW, `blackboard.md:196-202`). Zero `core::arch`, zero `#[cfg(target_feature)]`,
zero `target_feature`, zero ISA branches in any crate this plan touches. A missing
primitive is §6 N-10, never a local intrinsic.

**N-6 — No performance claim without a bench.** §3.3's own text already refuses the
obvious one (fusing 3 passes into 1 saves ~580 ns against a ~17 µs step,
`blackboard.md:26-28`). Wave 0 and Wave 1 are **correctness** gates. Nothing in this
plan is licensed to say "faster" until a committed bench says a number.

**N-7 — No semiring beyond Boolean and XOR.** `mask-risc-lowering-v1.md` §2's table
is binding: Boolean and XorBundle/XorField are exact; HammingMin, SimilarityMax,
Resonance and NarsTruth are values and stay values. *"Anyone claiming HammingMin is
an AND fails this fence."*

**N-8 — No mutation.** `CREATE`, `SET`, `DELETE`, `MERGE` are out of scope entirely.
The kanban plan already rules that a board mutation routes through the DO arm's
commit gate (`cypher-kanban-ast-unification-v1.md:134`: *"a move = `ActionInvocation`
through the commit gate (def-match→RBAC→state-guard→MUL), NOT a raw `MATCH…SET`
edge-rewrite"*). A mask lowering for reads must not become a back door around that.

**N-9 — Gremlin / SPARQL / GQL are not claimed.** The seam sits below them
(§2 property 1) so they *should* inherit, but "should" is not a measurement. v1
lowers Cypher, and the other three are a Wave-4 falsifier (§7), not a v1 claim.

**N-10 — A missing substrate primitive is a STOP, never a local workaround.** If the
lowering needs a `u16` compare-to-mask (P-8), a dictionary lane (§4.2), or a
transpose lane (R-2), the answer is: **stop; the capability lands substrate-first
in `ndarray::simd`** (or in the contract), with its own parity test across all five
backends, and only then does the consumer gain a name for it. The consumer never
grows the membrane. Three real gaps were closed this way before, per
`lance-graph-java/CLAUDE.md` § Missing-capability STOP rule.

**N-11 — `lgj_hop` is not importable from here.** It is a C ABI export in a
different repository (`lance-graph-java/native/lgj-abi/src/exports.rs:2053`). R-1's
`[H]` grade is exactly this: the SHAPE is shipped and proven, the SYMBOL is not in
this tree. Wave 2 either re-expresses it over `MailboxSoaView`'s edge accessors or
records that it cannot — it does not link against lgj-abi.

**N-12 — `lance-graph-mask-risc` is adopted as a COMPILING SKELETON, not as the crate this read saw.** ⊘ PR2 `c095dcc` made it a workspace member declaring only `ir`; `exec`/`fuse`/`hop`/`reference`/`ternlog_table` stay PR3. As read, it did not build (five of
six declared modules absent) and is in neither workspace list (§1.3). Wave 0 decides
whether to complete it or to host the lowering in `lance-graph-planner`, which
already has the `ndarray` dependency and a shipped T1 consumer
(`nested_bands.rs:32`). **Adopting a crate that has never compiled is not a
shortcut** — this is §8 OQ-10.

---

## §7 — (d) WAVES, EACH WITH ITS FALSIFIER

The gate discipline is `CLAUDE.md` § The falsifiability rule, applied without
exception: *"An assertion implied by the code it tests is not a test. Before a test
lands, answer: what input would make this fail?"* Plus its two corollaries — a
filter needs an **anti-vacuity** test (the excluded set is non-trivial), and a guard
needs **both** a can-it-fire and a can-it-stay-silent test over non-trivial input.

### §7.0 — Wave 0 — MEASURE. No production code. (the plan's STOP gate)

Nothing is built until Wave 0 answers, with numbers, the questions §1.4 and §8 raise.

**W0-a — the substrate census.** Against a real bake (not a synthetic fixture):
does a `label → classid` binding exist or must one be minted (OQ-1)? Are rows minted
in classid order (OQ-2)? Which candidate property columns are `u32`/`i32`-aligned vs
`u16`/`u8` (OQ-3)? Is there a transpose lane for any relationship (OQ-6)? Each answer
is a number or a "no", never a plan.

**W0-b — the corpus census.** Take a real Cypher corpus (the parser's own 44 tests
are the floor; the four builder test modules add more —
`datafusion_planner/builder/basic_ops.rs`, `expand_ops.rs`, `join_builder.rs`,
`aggregate_ops.rs`). Classify every query by §3/§4: **what fraction lowers fully,
what fraction splits, what fraction is pure grace?** (OQ-5.) If the full-lowering
fraction is negligible, this plan's premise is wrong and Wave 1 does not start.

**W0-c — THE DIFFERENTIAL HARNESS. This is the falsifier the whole plan hangs on.**

Shape, stated precisely enough to build and to break:

```text
for each fixture query q in F:
    dataset D              a REAL, committed, immutable dataset (never /tmp)
    lhs = DataFusion path:   query.rs:920 create_logical_plans → execute → rows
    rhs = mask path:         mask_lower(logical_plan) → execute → mask
    assert  set_of(lhs.row_ids)  ==  materialize_rows(rhs)      // SET equality
```

Five properties, each of which a weaker harness would drop:

1. **SET equality, not cardinality.** `spog-alpha-channel-v1.md:331` gate (a) is the
   precedent and states why in its own words: *"equal cardinality on disjoint sets is
   consistent with a bijection and proves none, so the pass condition stays the set
   equality against the scalar reference … the count is the cheap early filter in
   front of it, never a substitute."* Count first (20 ns), set second (the proof).
2. **The DataFusion side is the REFERENCE, and it is the one that already works.**
   This is the only window in which that is true — after the grace period ends there
   is no second implementation to diff against. Wave 0 is therefore not merely first;
   it is the **only** time this comparison is available, which is why it is a wave and
   not a task.
3. **Anti-vacuity per fixture.** Every fixture must exclude a non-trivial set:
   `survivors * 3 < population`, asserted. A fixture whose `WHERE` admits everything
   proves the harness runs, not that the lowering is right. (This is the
   `elimination_rate() > 0.0` defect from `E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1`,
   pre-empted.)
4. **The harness must be able to FAIL, demonstrated.** Pre-register a *wrong*
   lowering and watch it go red: substitute `AND2 = 0xC0` where `AND3 = 0x80` belongs.
   `spog-alpha-channel-v1.md:336` gate (f) is the precedent — *"a wrong immediate
   (`0x80` vs `0xC0`) differs on real data"* — and the fixture set must contain at
   least one query on which those two immediates genuinely disagree. If they do not,
   the fixture is too weak, not the check.
5. **Fixtures are COMMITTED.** No `/tmp`. `tesseract-rs/CLAUDE.md` records the cost
   of the alternative: two tests red for 13 days behind a skip-guard, *"a test fixture
   under `/tmp` is a time bomb with a skip-guard for a fuse — the skip hides it exactly
   where a fresh CI would have caught it."*

**W0 STOP condition.** If W0-a says there is no `label → classid` route AND no
cheap one can be minted, §3.1 is unbuildable and the plan is re-scoped to §3.2+§3.3
over an explicitly-supplied classid, or shelved. Say so; do not build N-2 against a
classid that does not exist.

### §7.1 — Wave 1 — the Boolean core, ONE population, no hops

Scope: §3.1 N-2/N-3/N-4, all of §3.2 except P-6/P-8/P-9, all of §3.3, and
§3.5 T-1/T-2/T-3/T-5/T-6/T-9/T-11. Output: `Full` or `Grace` only — **no `Split`
yet**, so Wave 1 cannot hide a bug in a residue.

Falsifiers, each disable-verified red-then-green:

| id | assertion | its DISABLE |
|---|---|---|
| **F-B1** | a 3-leaf `WHERE` produces **one** combination pass, not two | make the fuser emit two `And`s; assert the op histogram (`ir.rs:186` `OpHistogram`, `:201` `mask_passes`) goes 1 → 2 |
| **F-B2** (can-fire) | `A AND B AND NOT C` admits rows that `A AND B` alone does not exclude — i.e. C actually excludes something | drop the `NOT C` leaf; survivor set must GROW |
| **F-B3** (can-stay-silent) | the same lowering over a fixture where C is empty leaves the survivor set **unchanged** | force `AND2_ANDNOT` → `AND3`; the sets must now differ |
| **F-X3** (the tail law, §5.3a) | a fused complementing immediate does not raise phantom tail bits: `popcount(program) == popcount(scalar reference)` on a population whose `n_rows % 64 != 0` | remove the tail clear; on `n_rows = 100` the count must jump by up to 28 |
| **F-A1** (anti-vacuity) | every Wave-1 fixture excludes `> 2/3` of the population | — (a bound, not a disable) |
| **F-R1** (reference independence) | the scalar reference evaluator is written without `ndarray` and diffed on every backend (`mask-risc/src/lib.rs:38-39`: *"the oracle every executor is diffed against, on every backend"*) | — |

**Also in Wave 1, because T-4 needs it:** exactly ONE materialiser, its name starting
`materialize`, its O(n) cost in its doc comment, and a test asserting no other public
symbol returns `Vec<usize>` / `Vec<u64>` / `&[u64]`-of-row-ids from the lowering's
surface. That test is the structural enforcement of the mask-native invariant inside
this repo; without it, §1.4 finding (2) reappears in a new crate.

### §7.2 — Wave 2 — the hop

Scope: §3.4 R-1, R-3, R-4, R-5, R-6, R-9. Depends on Wave 0's OQ-6 answer and on
N-11 (re-express, do not link).

| id | assertion | its DISABLE |
|---|---|---|
| **F-H1** | one hop from a src mask equals the DataFusion `Expand` join's target set, bit-for-bit | — (it IS the differential) |
| **F-H2** (the transpose law) | `<-[:R]-` over an **asymmetric** fixture differs from `-[:R]->` | build the fixture symmetric; the test must then go GREEN under the wrong lowering — which is exactly why the fixture must be asymmetric, and the test asserts asymmetry first (`blackboard.md:414-417`) |
| **F-H3** (can-stay-silent) | a hop from an EMPTY src mask writes an empty dst and touches no row | — |
| **F-H4** (one-population law) | a hop whose dst mask indexes a different store is REFUSED, not computed | remove the population-id check; the call must now succeed and produce a wrong answer |
| **F-H5** (no walk) | the selection half examines no row to decide participation — assert via an op histogram, not by reading the code | — (`exports.rs:2118-2136` records three prior shapes that each lost this) |

### §7.3 — Wave 3 — the fixpoint

Scope: §3.4 R-7, R-8.

| id | assertion | its DISABLE |
|---|---|---|
| **F-F1** | the delta-frontier closure equals the accumulated-state closure, exactly | — (this is `blackboard.md:33-36`'s own gate, re-run here) |
| **F-F2** | `*1..k` equals the DataFusion unroll-and-union for every `k` in the fixture | — |
| **F-F3** (termination) | an unbounded `*` over a **cyclic** fixture terminates, and the step count equals the graph's eccentricity from the seed | remove the `mask_any(frontier)` test; it must hang or over-count |
| **F-F4** (can-stay-silent) | `*1..k` on a fixture with no R edges returns the seed set unchanged for `min=0`, empty for `min=1` | — |

### §7.4 — Wave 4 — the seam, `Split`, and the grace path

Scope: Phase 2.5 in `query.rs:920-952`, the `Split` outcome, and the router.

| id | assertion | its DISABLE |
|---|---|---|
| **F-S1** | with the lowering DISABLED, every existing test is byte-identical to today | — (the additive-seam gate; this is the one that must be run before any merge) |
| **F-S2** | a `Split` query returns the same rows as the pure-DataFusion path | — |
| **F-S3** (the grace list is real) | every §4 construct in the fixture set is classified `[GRACE]` and takes the DataFusion path — asserted by the classifier's own output, not by the result being right | force the classifier to accept `ORDER BY`; the differential must go red |
| **F-S4** (N-9) | the same lowering, reached from a Gremlin or SPARQL query that produces the same `LogicalOperator`, produces the same mask | — |

### §7.5 — Two ways a disable silently is not one (read before running any of the above)

Both cost real time in sibling repos, and both apply directly to a knob-heavy plan
like this one.

**(a) A disable that does not apply is indistinguishable from a guard that is not
load-bearing.** Both look like "the test still passes". Always assert the
replacement actually landed:

```python
assert s.count(old) == 1, "anchor moved"
```

**(b) Turning a knob is only a disable if the knob BINDS.** Zeroing a constant proves
nothing when the guarded quantity can reach the same outcome by another route.
Concretely here: setting an immediate to `0` does NOT disable B-4..B-9 — `IMM = 0x00`
is the constant-false table, which changes the answer for a different reason than the
one under test. The correct disable for a fusion rule is **unhooking the fuser**, and
for a law it is **removing the check**, never re-tuning a number.

**And commit before you disable.** The restore is `git checkout <file>`, which
reverts to the last COMMIT — so uncommitted work under test is deleted by the
restore. Order: commit, disable, checkout.

---

## §8 — (f) OPEN QUESTIONS — each answered by a MEASUREMENT, not a discussion

Every row names the instrument and the shape of the answer. A row answered in prose
is not answered.

| # | question | how it is answered | what a wrong answer costs |
|---|---|---|---|
| **OQ-1** | **`MATCH (n:Label)` → which classid?** `MailboxSoaView::class_id()` is `&[u16]` aliasing `entity_type` (`soa_view.rs:100-102`); `class_view.rs:54` is `ClassId = u16`; `le-contract.md:12-30` says the facet prefix is a composed **u32**; `ogar_codebook.rs:757` resolves a concept to `u16` and `:398` renders a `u32`. `GraphConfig`/`NodeMapping` (`config.rs:35-71`) carries **none** of them | census a real bake: how many distinct labels, do any collide under `u16`, which width is actually in the row at byte 0 | building N-2 against the wrong width produces a sweep that silently matches the wrong rows — the confident-and-wrong quadrant |
| **OQ-2** | **are rows minted in classid order?** N-1's 228–462× range reveal (`blackboard.md:18-22`) is conditional on it | measure on the bake: is `{rows with classid c}` contiguous for every `c`? Report per-class, not in aggregate | assuming yes when no = a range mask that returns the wrong set. Assuming no when yes = paying a 22 µs sweep for a 99 ns read |
| **OQ-3** | **`u16` / `u8` compare-to-mask: T1 addition, or widen the column 4×?** `blackboard.md:49-52` names the workaround and marks its own numbers as UPPER bounds | measure both on a real `u16` class column: the widened `eq_i32_to_mask` sweep vs a prototype narrow compare. If the delta is small, do not add a T1 primitive (POLYFILL LAW means one addition is five backend implementations + a parity test) | adding a primitive nobody needed; or shipping a 4× widening on the hottest column in §3.1 |
| **OQ-4** | **what Cypher surface reaches `ternary_match_*` (P-6)?** The op takes `pattern`/`care` as `&[u8;12]` — the V3 register exactly (`simd_masking_ops.rs:1415-1417`) — and no Cypher syntax addresses a rail today | pick ONE candidate spelling, lower it, and measure whether a rail predicate actually occurs in the W0-b corpus. If it occurs zero times, do not build it | building a TCAM path with no query that reaches it |
| **OQ-5** | **what fraction of a real corpus lowers fully vs splits vs grace?** | W0-b, counted per query over the committed corpus, reported as three integers | this is the plan's premise; a negligible full-lowering fraction means Wave 1 should not start |
| **OQ-6** | **does a transpose lane exist for any relationship?** R-2/F-H2 depend on it | inspect the bake: for each relationship class, is there a lane indexed by target? If not, `<-[:R]-` is `[GRACE]` in v1 and must be listed there | the measured 912-vs-930 failure (`blackboard.md:414-417`), reproduced |
| **OQ-7** | **`avg()` result type** — exact rational over `(masked_sum_i32, popcount)`, or `f64`? | diff both against the DataFusion path's own `avg` on the fixture set; whichever is bit-identical wins, and if neither is, `avg` is `[GRACE]` | a silent numeric divergence that the SET-equality differential cannot see, because it is in the value, not the population |
| **OQ-8** | **`IN [a..z]` (P-7): at what list length does the OR-accumulated chain lose to a hash-set scan?** | sweep list length on the real column; report the crossing, or report that there is none up to the corpus's longest list | a lowering that wins on 3 needles and loses on 300, chosen by guess |
| **OQ-9** | **who is the first `TERNLOG 0x86` consumer?** `spog-alpha-channel-v1.md:215-219` §3.5 already claims that slot for a lifted crosswalk program, with its own falsifier | ask; do not race. If spog lands first, this plan's Part-1 consumption is a second caller of a proven path, which is strictly better | two first-consumers of one opcode, each proving it differently |
| **OQ-10** | **host the lowering where?** ⊘ the compile half is answered — PR2 makes `lance-graph-mask-risc` a member that builds and is CI-gated. What is still open: complete the five missing modules THERE, or host the lowering in `lance-graph-planner` (which has `ndarray`, `Cargo.toml:24`, and a shipped T1 consumer, `nested_bands.rs:32`) | decide in Wave 0 by writing the executor against the IR and measuring whether a second crate edge pays for itself | a crate that compiles but never executes anything, and a planner that grows a second evaluator |
| **OQ-11** | **does the `Split` residue actually save anything?** A mask prefix handed to DataFusion must arrive as a predicate DataFusion can push down, or the split is pure overhead | measure `Split` vs pure-grace on the same query. If `Split` never wins, drop the variant and simplify to `Full` / `Grace` | a three-way outcome enum where two values suffice |
| **OQ-12** | **what does a hop cost on THIS substrate?** `lgj_hop`'s measured numbers are on a facet-major store in another repo (`exports.rs:2042-2044`: *"the hop runs 3.3–4.8× over AoS"*); the CANON row here is AoS-shaped (16 B key + 16 B edges + 480 B value) | measure the re-expressed hop on the real bake, both layouts if both exist | importing a sibling repo's ratio as if it were this repo's — the same class of error as citing a debug-profile number as release |

---

## §9 — Cross-references

> **How citations in this file are to be read (a finding, not a formality).**
> While this plan was being written, `.claude/board/EPIPHANIES.md` and
> `lance-graph-java/native/lgj-abi/src/exports.rs` both MOVED under it: the
> `E-TOPOLOGY-…` entry shifted 670 → 709 because EPIPHANIES is **prepend-only**,
> and `lgj_hop` shifted 1703 → 2053. Every line number here was re-verified after
> that, but the lesson generalises: **a board entry is cited by its `E-id` and a
> symbol by its NAME; the line is an as-of reading, never the identity.** A future
> session that finds a line number stale should re-grep the id or the symbol, not
> conclude the claim is wrong — and should not "fix" a citation by deleting the
> claim.


**Operator rulings this plan is subordinate to:**
`E-PLANNING-MIGRATES-TO-LOCO-R2IL-DATAFUSION-IS-GRACE-PERIOD-1` (`CLAUDE.md:1017-1025`) ·
`E-TOPOLOGY-MASKS-MAGNITUDE-COMPOSE-NEVER-COLLAPSE-1` (`.claude/board/EPIPHANIES.md`) ·
`E-V3-FACET-4-PLUS-12` (`le-contract.md:8`) · the CANON node (`CLAUDE.md` § Minimal SoA node) ·
POLYFILL LAW / BACKEND LAW (`ndarray/.claude/blackboard.md:196-207`) ·
`F-RLR-2` (`rubicon-loco-rung-cognitive-fabric-v1.md:584`).

**Plans this builds on, and how it relates to each:**

| plan | relation |
|---|---|
| `cypher-kanban-ast-unification-v1.md` | **builds on.** Its Inc 0 opened `Backend::MailboxSoa` (`graph_router.rs:53-61`) and its F1 is backend parity. This plan supplies the mask-shaped scan that Inc 0's `match_nodes_by_class` (`mailbox_scan.rs:77`) currently does with a `Vec`. **No contradiction:** that plan says *"datafusion is NOT deprecated … the router gains a backend; it does not lose one"* (`:107-108`) and so does §6 N-1 |
| `lance-graph-java/.claude/plans/mask-risc-lowering-v1.md` | **sibling, different layer.** That plan lowers *loco verbs* to a mask RISC behind a Java membrane and owns the cache/trie question (§14 voxelmasking). This plan lowers *Cypher* to the same RISC and owns no cache (§6 N-3). §2's two-axis model and §2's semiring fence are cited, never re-claimed |
| `spog-alpha-channel-v1.md` | **supplies the chain shape** (§3.2 crosswalk, `:176-189`), the forbidden move (`:186-187`), the gate-(a) set-equality discipline (`:331`), the gate-(f) wrong-immediate falsifier (`:336`), and the prior claim on `TERNLOG 0x86` (`:215-219` — OQ-9) |
| `ndarray/.claude/plans/gemm-ternlog-mask-consolidation-v1.md` | **upstream of the measurements** cited from `blackboard.md` (D-GTM-0j type boundary, D-GTM-0m range reveal + NNUE delta frontier) |

**Code this plan names, by crate:**
`lance-graph`: `parser.rs`, `ast.rs`, `semantic.rs`, `logical_plan.rs`, `query.rs`,
`config.rs`, `datafusion_planner/builder/mod.rs`, `graph/graph_router.rs`,
`graph/mailbox_scan.rs`. ·
`lance-graph-contract`: `alpha.rs`, `soa_view.rs`, `class_view.rs`, `ogar_codebook.rs`,
`hhtl.rs`, `facet.rs`. ·
`lance-graph-planner`: `nested_bands.rs`, `strategy/cypher_parse.rs`. ·
`lance-graph-ogar`: `recipe_vocab.rs` (the domain-vocabulary hosting precedent). ·
`lance-graph-mask-risc`: `ir.rs`, `lib.rs` (orphan; OQ-10). ·
`ndarray`: `simd_masking_ops.rs`, `simd.rs`, `bitwise.rs`. ·
`OGAR`: `ogar-loco/src/{lib.rs,vocabulary.rs}`, `ogar-r2il/src/lib.rs`.

---

## §10 — Board hygiene this plan owes on landing

Per `CLAUDE.md` § Mandatory Board-Hygiene Rule, a PR landing any wave of this plan
must, **in the same commit**:

- add the wave's D-ids to `.claude/board/STATUS_BOARD.md` (Queued → In progress → In PR → Shipped);
- PREPEND this plan to `.claude/board/INTEGRATION_PLANS.md`;
- PREPEND any finding or correction to `.claude/board/EPIPHANIES.md` — and §1.4's three
  findings are corrections that already qualify;
- regenerate `.claude/board/SUPERSESSION-INDEX.md` **LAST, after the board writes**
  (`python3 .claude/tools/supersession_index.py > .claude/board/SUPERSESSION-INDEX.md`),
  because the board is one of its inputs and a run made before the EPIPHANIES prepend
  produces a byte-identical file that reads as current while CI goes red on exactly
  the rows the new entry's D-id citations moved.

The termination clause applies: a PR whose entire content is board hygiene for prior
PRs generates none of these obligations.

---

## §11 — Summary of the three deliverables the prompt asked for

1. **The lowering table (§3): 53 rows** — 7 node/label, 9 property predicate, 9
   Boolean-fusion, 9 relationship/hop, 12 return/terminal, 7 explicit non-lowering.
   **31 `[G]`, 13 `[H]`** (regraded by the PR2 council — compositions inherit their components' grade; nine name a measurement in §8), **9 `[GRACE]`**.
2. **The placement ruling (§5): three parts.** Consume the already-minted, so-far
   unconsumed `TERNLOG = 0x86` (`ogar-loco/src/lib.rs:607`) for all Boolean
   combination — do not mint. Keep `Pred` / hop / terminals as a **lowering target
   only**, no byte, no codebook, because no byte of a mask program survives the query
   that built it. If that ever changes, they become a **domain vocabulary above
   `DOMAIN_FLOOR`** hosted where `lance-graph-ogar::recipe_vocab` is hosted, and the
   trigger is written down in advance.
3. **The grace path (§4): nine construct families** stay on DataFusion — order and
   multiplicity (`ORDER BY`, `SKIP`/`LIMIT`-after-order, `DISTINCT <value>`,
   `count(DISTINCT)`, `collect`), strings and variable-width values, bag semantics
   (`UNWIND`, `WITH`-aggregation), non-Boolean values (vector distance/similarity,
   NARS truth), and cross-address-space joins. Maintained, never extended.
