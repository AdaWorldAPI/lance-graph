# AST-as-(part_of:is_a)-address — sinking compiled semantics into the GUID

> **READ BY:** integration-lead, truth-architect, core-first-architect,
> family-codec-smith, baton-handoff-auditor
> **Status:** CONJECTURE (design; the **carrier is now SHIPPED** — see V3
> alignment below — only the rank-minter is unbuilt/unmeasured).
> **Cross-ref:** `crates/lance-graph-contract/src/facet.rs` (the **SHIPPED**
> `FacetCascade` substrate, #613/#614), `canonical_node.rs`
> (`TailVariant::V3` / `mint_for` / `CLASSID_OSINT_V3`, #615),
> `guid-canon-and-prefix-routing.md` (the GUID canon),
> `core-first-transcode-doctrine.md` (harvest → ClassView → codegen),
> OGAR `SURREAL-AST-AS-ADAPTER.md` + `SURREAL-AST-TRAP-PREFLIGHT.md`
> (DDL is an adapter, never the spine), `encoding-ecosystem.md`.

> **V3 alignment (2026-06-26).** This doc was first drafted without awareness that
> the V3 substrate **already shipped** (#613/#614 `FacetCascade`, #615 `mint_for`).
> Three corrections, applied throughout:
> 1. **The slot-count "open gate" is CLOSED.** `facet.rs::FacetCascade` =
>    `facet_classid(4) | 6×(8:8) = 16 B` — **6 tiers** (`HEEL·HIP·TWIG·LEAF·family·
>    identity`), `const _`-asserted. That IS the full-key 6-pair / 12-slot answer;
>    the *key* carries only the 4-tier routing prefix
>    (`NiblePath::from_guid_prefix_v3`), the complete 6-pair address lives in the
>    `FacetCascade` **value facet**. Nothing left to ratify.
> 2. **The carrier is NOT missing.** `FacetCascade` is **content-blind** and
>    already lists `(part_of:is_a)` as a consumer projection: `hi_chain()` =
>    part_of, `lo_chain()` = is_a, with `hi_distance`/`lo_distance` the two
>    orthogonal prefix metrics. AST-as-(part_of:is_a) is one reading of this shipped
>    substrate; the only genuinely-new brick is the deterministic **rank-minter**.
> 3. **The classid row is `0x1000_0700`-shaped** (shipped `CLASSID_OSINT_V3`: V3
>    marker in the HIGH u16, domain routed on the LOW u16). Its `(part_of:is_a)`
>    *ordering* is **pending the operator's Canon:Custom correction** (canon→high) —
>    flagged on the row, not settled here.

## The claim

The structural AST of a transcode *source* (C#/Python/C++/Ruby) can be stored
**as the (part_of:is_a) GUID address itself**, in the lance-graph SoA — not as a
SurrealQL AST/DDL, and not as a raw syntax tree. The GUID *is* the AST node's
structural identity; the value columns + `CausalEdge64` edges hold the behavior.
This closes the loop from the `ruff_*_spo` harvest to the OGAR Core, and makes
an LSP (`ruff-lsp`) the natural read/serve surface.

```
source (C#/…) ──ruff_*_spo harvest──► SPO triples ──► AR-shaped Model / ClassView
                                                            │  rank-mint (NEW brick)
                                                            ▼
                                              (part_of:is_a) GUID  ──► lance-graph SoA
                                                            │  serve
                                                            ▼
                                                        ruff-lsp ──► editor
```

## The purpose — the three-layer thesis (the *why*, grounded in shipped code)

The *what* above (store the AST as a `(part_of:is_a)` address) exists to serve
one economic end: **make minting an ERP in any consumer as cheap as importing
pre-minted class primitives.** Three layers, each already backed by shipped
contract code (verified this session):

### Layer 1 — rails-shaped semantic AST at *assembler* cost

The structural AST's queries collapse to **one hardware instruction each**,
because the node's identity *is* a 128-bit register (`facet::FacetCascade`) and
the semantic operations *are* its four SIMD lanes:

| LSP / semantic query | `(part_of:is_a)` operation | assembler (facet.rs four-lane table) |
| --- | --- | --- |
| `definition` (symbol → node) | full-facet equality | `vpcmpeqd` + `vmovmskps` (row lane) |
| `documentSymbol` (part_of tree) | `hi_chain()` compare | `vpcmpeqw` / `pshufb` (tile lane) |
| `typeHierarchy` (is_a lattice walk) | `lo_chain()` longest-common-prefix | `vpxor` + `tzcnt` (prefix lane) |
| ancestry / quadrant containment | Morton nibble prefix | GFNI `vgf2p8affineqb` (nibble lane) |

A graph traversal becomes a register XOR + trailing-zero count. **Why the rails
shape is the precondition, not a style choice:** a *declarative* class body (a
flat bag of typed `part_of | is_a` declarations) maps onto fixed-width 6×(8:8)
tiles; an *imperative* syntax tree (variable-arity, positional) does not. Only
the declarative **THINK arm** flattens to the address.

### Layer 2 — static OGAR shape · dynamic ClassView · askama row-view

`class_view.rs` already states this exact ladder (its own doc comment):

```
SoA row            = the XML document   (agnostic bytes, no meaning)
class / ObjectView = the XSD schema      (the shape: which fields, in order)
ClassView          = the parser+schema   (projects row → typed view, late-bound)
askama template    = the XSLT            (renders the projected view)
```

Two shipped invariants make the static shape *reusable* and let it evolve like
Redmine's 17-year row view without breaking persisted data:

- **C2 — presence, NEVER semantics.** `has(n)` = "field n is populated here,"
  never "field n behaves differently here." Per-app variation lives in the
  ClassView projection + the askama template (the *render*), never in the bits.
- **N3 — append-only field positions.** "Once instances persist, a field's bit
  position never moves and retired bits are never reused." *This is the
  mechanism* behind a row view that grows columns for 17 years without
  invalidating a single stored row.

**View and action are duals of one ClassView.** A node's SoA value slab holds
**N × `(facet_classid | 6×(8:8 part_of:is_a))`** facets (the value-tenant
migration's multiple tenants per node) plus a *little* DO. The ClassView projects
those same static facets **two** ways: row → typed **view** (THINK → askama
render) **and** row → conditional **classaction** (DO → an `ActionDef` fired under
a `KausalSpec` `StateGuard{field,value}` that reads the SoA's own bits). So
behavior is conditionalized *at the cost of one ClassView classaction dispatch* —
a guard-read (assembler-cheap, exactly like the view projection) + an `ActionDef`
lookup by `classid`. Render and behavior are the two projections of one ClassView
over one static SoA; the DO arm is as cheap and as reusable as the THINK arm
because it is the **same** projection mechanism — only the *little* DO needs to
ride along, conditionalized, not a parallel engine.

**In the consumer, DO is a classaction *pointer*, not logic.** Because the
ontology→class conversion is lossless, a class identity is preserved bijectively
*across* consumers — so its behavior need not be re-implemented, only *pointed
at*. With separation of concerns + a DTO-carried invocation, the DO arm collapses
in the consumer to an **object-oriented reusable classaction pointer**: `classid
→ ActionDef` (the target = shared behavior, minted once), dispatched via an
`ActionInvocation` DTO (`realizes → ActionDef.identity`; carries
`object_instance`, `idempotency_key`, `state`). The consumer holds only the
*pointer* (`classid` address) + its per-app `KausalSpec` guard (the *when/which*)
+ its own content; the *what* (the `ActionDef` body) lives once at the resolution
target. This is exactly the **OGAR consumer doctrine** — *the `classid` is pure
address; the class-magic (`ActionDef`+`KausalSpec`) is a property of the Core node
the address resolves to, never of the address* (`ogar-consumer-preflight.md`). A
vtable/strategy slot whose implementation is a universal OGAR primitive — and the
lossless bijection is precisely what guarantees the pointer resolves to the
*same* target for every consumer.

`openproject-nexgen-rs` (`op-codegen-projection`) and `woa-rs` already pull
`askama 0.12` — the render end is live.

### Layer 3 — OGAR as importable ERP-primitive stdlib · lance as the compiler

`ogar_codebook.rs` is the import surface — a curated `(concept, u16)` codebook,
**wire-compatible, zero-dep** (a consumer *stamps* an id, never links a Core):

| economic step | shipped surface |
| --- | --- |
| **import** a primitive | `canonical_concept_id("account.move") → u16` → `classid` → `ClassView` |
| **customize** for an app | `render_classid_for_concept(app_prefix, concept)` — compose shared concept (lo u16, canon) with the consumer's render skin (hi u16, custom) |
| **compile** to a running app | lance-graph (OGAR-as-IR): `classid → ClassView → askama` |

**Cost collapse:** transcode goes from O(whole app, hand-ported) to O(the
consumer's *deltas* from the imported primitive). `account.move` /
`res.partner` / OpenProject `Issue` / MedCare `Patient` are minted **once** in
OGAR (harvest → `facet_mint`) and reused by every consumer.

**The magnitude:** OpenProject is **~500K LOC** of Rails. Under this model a
consumer's *marginal* cost collapses from "re-transcode ~500K LOC" to "import the
OGIT class primitives + wire classaction pointers + per-app content & `KausalSpec`
guards." The primitive-mint is a real *one-time* cost, but it is **amortized
across every consumer** — MedCare / WoA / SMB / Odoo / OpenProject-nexgen all draw
the *same* OGIT patterns (the regulatory `NTO/{Audit, Compliance, Legal}` set
imported once; cf. boundary #1). That magnitude — "much cheaper than 500K LOC" —
is precisely the CONJECTURE's payoff, and the brick-3 probe (MedCare harvest →
mint → SoA → LSP query, MedCareV2 oracle) is what turns it from a claim into a
measurement.

**Headline target (CONJECTURE — the probe measures it):** OpenProject + Odoo
together as **~2 MB of GUID-encoded `(part_of:is_a)`** instead of ~20 MB / ~250K
LOC — a ~10× collapse. Dimensionally credible: at 16 B/`FacetCascade`,
2 MB ≈ **131K class/member/field nodes**, enough to hold both ERPs' structural
skeleton. The honest caveat: that 2 MB is the **THINK-arm structure + classaction
pointers**; the DO-arm `ActionDef` *bodies* don't vanish — they are minted **once**
in OGAR and shared (amortized across every consumer), so the figure is the
*per-consumer marginal* footprint over a shared primitive library, not the whole
system reduced to 2 MB. The 10× ratio is the hypothesis the MedCare probe gives
the first real datapoint for.

**What the primitives actually are: laws and regulations, not CRUD shapes —
content stays with the consumer.** An ERP's hard value is *compliance* (tax law,
audit/GoBD immutability, SKR04 accounts, sanctions/AML, HIPAA), and regulation is
**universal** (jurisdiction-wide) and **reusable** (every consumer in that
jurisdiction needs it). So the importable primitive is the *law as a static
pattern* — its legally-required fields/relations as an `ObjectView`, its
regulatory rules as `ActionDef`/`KausalSpec` guards — minted once, pulled by
every consumer. The **content** (this company's invoices, this clinic's patients)
stays with the consumer (per `I-VSA-IDENTITIES`: bundle identities, store content
in the consumer's own stores); only the **pattern** is shared. This is grounded:
OGIT already carries regulation-as-pattern in `NTO/{Audit, Compliance, Legal}`
(e.g. `Compliance/{SanctionsEntry, legalBasis, sanctionedUnder, financiallySupports}`).

The enabling result (operator-proven; attributed, not re-verified here):
**classes convert bijectively and losslessly between OWL (the W3C ontology
standard regulatory ontologies ship in — e.g. FIBO) and OGIT.** The mechanism is
structural — OGIT's `{entities, verbs, attributes}` *is* the `(part_of:is_a)`
shape (entity → class, attribute → `part_of` field, verb → `is_a`/relation) — so
a standard regulatory ontology published in OWL imports losslessly into the OGAR
`ClassView`. **Honest scope:** the bijection is over **classes** (the THINK-arm
structure — the legally-required fields); the regulatory *rules* (when a tax
applies, an audit guard fires) ride the DO arm (`ActionDef`/`KausalSpec`,
wireable + membrane-governed per #2 below). What's local-verified is OGIT's
regulatory namespaces + its native `(part_of:is_a)` shape; the formal
OWL↔OGIT bijection proof is the operator's prior result, cited not re-run.

### The three honest boundaries (so this isn't a promise it can't keep)

1. **Mechanism shipped; the pattern *source* is imported — the *mint* is the
   remaining step.** Earlier framings said "stdlib mostly empty"; that's now
   too pessimistic. **Complete OGIT is imported into OGAR** at
   `OGAR/vocab/imports/ogit/` — **~1,940 TTL** across the full `NTO` (incl.
   `Audit` / `Compliance` / `Legal` / `Security`) **plus** the `SDF` automation
   schemas (`Automation/{event, change, incident, requirement}` — the HIRO
   ActionHandler lineage, i.e. DO-arm source too). So the regulatory-pattern
   *source* is in place. What remains is **minting** that source into
   `FacetCascade` codebook primitives (`facet_mint` → publish to
   `ogar_codebook`): the bricks all exist (`ruff_*_spo`, `facet_mint`, the OGIT
   TTL/JSON input) — the **source → codebook wire** is the work, not the
   harvest.
2. **The DO arm is *also* wireable — behavior has an OGAR shape too.** (An
   earlier framing treated behavior as bespoke; that is too pessimistic.) The DO
   arm has its own OGAR IR — `ActionDef` / `ActionInvocation` / `KausalSpec`
   (`OGAR-AST-CONTRACT.md`) — and the **arago/HIRO ActionHandler** model
   transcodes into exactly it (the OGIT ontology *is* arago/almato's — TTL
   creator `chris.boos@almato.com`). An action handler becomes an `ActionDef`
   keyed by `classid`, realized as an `ActionInvocation` with a lifecycle
   `ActionState` machine, guarded by `KausalSpec`. So behavior is
   mint-and-import-able on the **same** mechanism as the THINK arm — `odoo-rs`'s
   `od-posting` becomes a *consumer* of imported `ActionDef`s, not hand-written
   logic. (Consistent with the boundary section below: behavior is keyed *by* the
   address as `ActionDef`, never flattened *into* it.) **The remaining honesty —
   not a wall:** (a) same stdlib-population maturity as #1 — the IR + transcode
   path exist; the *published* per-domain action library is the work; (b) DO
   composition is **`KausalSpec`-membrane-governed** (StateGuard / lifecycle /
   dependency-path) by design — behavior is reusable but its wiring is gated and
   auditable, not free code import. That governance is the feature, not the gap.
   The *conditionalization* (which `ActionDef` fires, when) is classaction-cheap
   — a guard-read + a `classid` dispatch; only an `ActionDef`'s *internal* compute
   (e.g. a tax algorithm) is still its content, imported as a primitive rather
   than re-derived per consumer.
3. **Cheapness requires pull-don't-reconstruct.** Holds only if consumers
   **import** (`canonical_concept_id` / `*Port::class_id`) rather than
   re-harvest or copy the codebook — the `ogar-consumer-preflight` iron rule.
   The moment a consumer rebuilds the class graph locally, the cost is paid
   twice (the do-it-twice / SURREAL-AST trap this whole design rejects).

## Why this is the convergence, not a detour

1. **An AST has exactly two structural relations, and they ARE the two tile axes.**
   - **is_a** (taxonomy / typing): `Patient is_a DbBase`, `kdnr is_a Property`,
     `Save is_a Function` ← harvest `inherits_from` + `rdf:type` ← the **is_a
     byte (TISSUE / what)** of a tier.
   - **part_of** (mereology / membership): `kdnr part_of Patient`,
     `Patient part_of MedCare.Models` ← harvest `has_field` / `has_function`
     (inverted) ← the **part_of byte (PLACE / where)** of a tier.

2. **SurrealQL is an adapter, not the spine.** Storing the AST as
   `DEFINE TABLE`/`DEFINE FIELD` DDL is the "negative-beauty hijack"
   `SURREAL-AST-AS-ADAPTER.md` §0 rejects — DDL can't carry the behavioral arm
   and it makes the schema the spine. The (part_of:is_a) GUID + lance SoA is the
   spine; SurrealQL is at most a query projection over it.

## The class wrapper + the rails-shaped (declarative) AST

Two requirements, one insight:

- **Class wrapper** = the OGAR `ClassView` / the `ruff_spo_triplet::Model` IR.
  The GUID is its *address*; the ClassView is the resolved declaration set the
  address points at ("the key prerenders nodes, zero value decode").
- **Rails-shaped, not syntax-tree.** You do NOT sink a raw `CSharpSyntaxTree`
  (imperative, positional — wrong shape). You sink the **declarative class-body
  shape** — exactly what `ruff_ruby_spo` harvests as the ActiveRecord shape: a
  class flattened to a *bag of typed declarations*, every one of which is a
  part_of or is_a edge. That is why `ruff_spo_triplet`'s IR is a `Model` with
  sibling declaration `Vec<…>` fields rather than a syntax tree — it is the
  language-agnostic, already-(part_of:is_a)-shaped class wrapper that every
  frontend (Python/C++/Ruby/C#) fills.

## The GUID decomposition — 6×(part_of:is_a) (RESOLVED — the shipped `FacetCascade`)

Operator framing (session): the GUID carries **6 (part_of:is_a) pairs = 12
slots**, read across the *whole* key, giving six levels of (composition × type)
resolution — enough to encode a deep AST node's full structural path, each level
capturing both where it sits (part_of) and what it is (is_a):

| GUID region | proposed (part_of : is_a) reading |
| --- | --- |
| `classid` (4B) | **shipped** as hi u16 : lo u16 = custom/render : canon/concept (`CLASSID_OSINT_V3 = 0x1000_0700` — V3 marker hi, domain routed on lo). **OPEN (Canon:Custom):** the operator's correction flips this to canon(hi):custom(lo) so the prefix sorts by shared concept, not render skin — so the classid `(part_of:is_a)` half-order is **not settled** (orthogonal to the per-tier minting). |
| `HEEL` / `HIP` / `TWIG` (2B each) | namespace·root-type / class·base / member-slot·member-kind |
| `basin·leaf` + `identity` (6B) | basin `po_rank` : `ia_rank` (OGAR `family = (po_rank3<<8)|ia_rank3`) |

**Reconciliation — RESOLVED by shipped code** (was framed as "the gate before any
packer"). `facet.rs::FacetCascade` settles it: `facet_classid(4) | 6×(8:8)` =
**6 tiers** (`HEEL·HIP·TWIG·LEAF·family·identity`), i.e. the **full-key 6-pair /
12-slot** reading — *not* path-only. The "path = 6 bytes = CAM-PQ 6×256 = 3 tiers"
canon describes the **key routing prefix** (`NiblePath::from_guid_prefix_v3`,
4-tier); the *complete* 6-pair address is the 16-byte `FacetCascade` **value
facet** (it does not fit the 64-bit key — facet.rs says so explicitly). So
"path-only vs full-key" is answered **both**: the key carries the routing prefix,
the FacetCascade carries the full address. The rank-minter therefore targets the
6 `FacetCascade` tiers (`tier.hi` = part_of chain, `tier.lo` = is_a chain) — no
allocation is left to ratify, and the only open ordering is the classid half-order
above (Canon:Custom), which is orthogonal to per-tier packing.

## The missing brick — a deterministic part_of/is_a rank-minter

Between the SPO harvest and the GUID is the **one** genuinely-new component — the
carrier (`FacetCascade`) is **already shipped**, so this is the only brick to
build: a pure-Rust, dependency-free rank-minter. Given the corpus's part_of edges
+ is_a edges, assign each node its `(po_rank, ia_rank)` at each tier and write
them into the 6 `FacetCascade` tiers (`tier.hi = po_rank`, `tier.lo = ia_rank`);
`FacetCascade::to_bytes()` is then the 16-byte facet and `hi_chain()`/`lo_chain()`
the two prefix-routable hierarchies — no new layout, no new type.

- **part_of tree** (namespace → class → member): deterministic sibling/
  topological rank.
- **is_a lattice** (class → base, member → kind): deterministic type rank.
- For a **finite, known** AST this is deterministic assignment — **not** learned
  PQ centroids — so it is exact and roundtrip-lossless (no quantization error on
  a known class graph). Iron-rule clean per **I-VSA-IDENTITIES**: it encodes
  *identity positions* (which class, which base, which slot), never bundles
  content.

## The boundary — skeleton in the address, muscle in the edges

The (part_of:is_a) address holds the **THINK arm**: the class/member/type graph,
the ClassView method-resolution manifest — precisely the harvest's *structural*
predicates. It does **not** hold the **DO arm**: method bodies, control flow, the
transcode logic. Those are not part_of/is_a relations; they live in
`ActionDef`/`KausalSpec` and the harvest's behavioral edges (`reads_field`,
`raises`, `traverses_relation`) as `CausalEdge64`, **keyed by** the GUID, not
encoded in it. (Lance compresses the value/edges arbitrarily; the key stays
transparent — compression never costs addressability.)

## The serve surface — ruff-lsp (and it's language-agnostic)

`AdaWorldAPI/ruff-lsp` today is a vanilla fork of the deprecated Python
ruff-lsp (no SPO/lance wiring) — a clean slate for the *read* end. The core LSP
operations **are** part_of/is_a queries:

| LSP request | graph query | axis |
| --- | --- | --- |
| `textDocument/typeHierarchy` | walk the is_a lattice | **is_a** |
| `textDocument/definition` | resolve symbol → ClassView node (GUID) | address lookup |
| `references` / `callHierarchy` | walk part_of / behavioral edges | **part_of** + edges |
| `documentSymbol` / `workspace/symbol` | the class→member part_of tree | **part_of** |

Backed by the (part_of:is_a) lance store, an LSP is no longer a Python-only
linter surface — it is a **language-agnostic semantic-navigation surface** over
whatever frontend filled the graph (Python / C++ / **C#-MedCare** / Ruby),
because at that layer it is all the same `Model`/ClassView keyed by GUIDs.

## Next bricks (ordered; each gated on the prior)

1. **Slot allocation — LOCKED (shipped).** The 6-pair / 12-slot layout is the
   shipped `FacetCascade` (#613/#614); no ratification needed. The ONLY remaining
   ordering decision is the classid `(part_of:is_a)` half-order (the operator's
   Canon:Custom correction) — orthogonal to per-tier ranking, so it does **not**
   block brick 2.
2. **Build the deterministic rank-minter** (`ruff_spo_address`, pure std): SPO
   graph → `(po_rank, ia_rank)` per level → packed slots. Verifiable in
   isolation.
3. **Probe on MedCare**: `ruff_csharp_spo` harvest → mint → lance SoA →
   `typeHierarchy`/`definition` query → diff the class graph against the C#
   original, with `MedCareV2` as the parity oracle.

The per-tier layout is **locked** (the shipped `FacetCascade`), so brick 2 (the
rank-minter) can proceed now — it writes into existing tiers, inventing no type.
Only the classid half-order (Canon:Custom) remains a decision, and it is
orthogonal to per-tier packing, so it does not gate the minter.
