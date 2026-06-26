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
