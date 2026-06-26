# AST-as-(part_of:is_a)-address — sinking compiled semantics into the GUID

> **READ BY:** integration-lead, truth-architect, core-first-architect,
> family-codec-smith, baton-handoff-auditor
> **Status:** CONJECTURE (design, co-developed in session; not built/measured)
> **Cross-ref:** `guid-canon-and-prefix-routing.md` (the GUID canon),
> `core-first-transcode-doctrine.md` (harvest → ClassView → codegen),
> OGAR `SURREAL-AST-AS-ADAPTER.md` + `SURREAL-AST-TRAP-PREFLIGHT.md`
> (DDL is an adapter, never the spine), `encoding-ecosystem.md`.

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

## The GUID decomposition — 6×(part_of:is_a) (OPEN: reconcile slot count)

Operator framing (session): the GUID carries **6 (part_of:is_a) pairs = 12
slots**, read across the *whole* key, giving six levels of (composition × type)
resolution — enough to encode a deep AST node's full structural path, each level
capturing both where it sits (part_of) and what it is (is_a):

| GUID region | proposed (part_of : is_a) reading |
| --- | --- |
| `classid` (4B) | app-render skin **part_of** : shared concept **is_a** (per OGAR consumer doctrine: hi u16 = render/ClassView, lo u16 = concept/RBAC) |
| `HEEL` / `HIP` / `TWIG` (2B each) | namespace·root-type / class·base / member-slot·member-kind |
| `basin·leaf` + `identity` (6B) | basin `po_rank` : `ia_rank` (OGAR `family = (po_rank3<<8)|ia_rank3`) |

**Reconciliation to lock (the gate before any packer is built):** the canon
(`guid-canon-and-prefix-routing.md`) describes the **path** as `6 bytes =
CAM-PQ 6×256`, i.e. **3 tiers × 256×256 tile = 3 (part_of:is_a) pairs over
HEEL/HIP/TWIG**. The 12-slot reading extends part_of:is_a to the *full* GUID
(classid + cascade + basin), not just the path. Whether part_of:is_a is
**path-only (3 pairs / 6 bytes)** or **full-key (6 pairs / 12 bytes)** is the
single decision that must be ratified against the V3 mint canon (PR #615
lineage) before the rank-minter packs anything — building against the wrong
allocation is the Frankenstein/`I-LEGACY-API-FEATURE-GATED` risk.

## The missing brick — a deterministic part_of/is_a rank-minter

Between the SPO harvest and the GUID is one new, pure-Rust, dependency-free
component: given the corpus's part_of edges + is_a edges, assign each node its
`(po_rank, ia_rank)` at each level and pack into the (locked) slot layout.

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

1. **Lock the slot allocation** — path-only (3 pairs) vs full-key (6 pairs / 12
   slots) — against the V3 mint canon. *Decision, not code.* (epiphany-council.)
2. **Build the deterministic rank-minter** (`ruff_spo_address`, pure std): SPO
   graph → `(po_rank, ia_rank)` per level → packed slots. Verifiable in
   isolation.
3. **Probe on MedCare**: `ruff_csharp_spo` harvest → mint → lance SoA →
   `typeHierarchy`/`definition` query → diff the class graph against the C#
   original, with `MedCareV2` as the parity oracle.

Until (1) is ratified, no packer is written — the layout is the contract, and a
packer built against an unlocked contract is the exact anti-pattern this
workspace gates.
