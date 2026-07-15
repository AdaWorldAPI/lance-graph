# Assembler vs storage substrate — OGAR assembles; Lance calcifies; nobody compiles on the hard disk

> READ BY: layer-boundary-warden, codegen-flow-cartographer, integration-lead,
> truth-architect, any session about to file an issue, name an "ingest API",
> or assign a responsibility across the OGAR ↔ lance-graph boundary.
>
> Born 2026-07-14 from a real failure chain: a Klickwege live-edge issue filed
> at lance-graph (#691, closed) because "SPO store" pattern-matched to the
> query engine's triple store; then two more wrong re-framings before the
> operator's rulings landed. Companion doc:
> `compilation-vs-runtime-substrate.md` (the compile-time/runtime split).

## The distinction

**OGAR is the assembler substrate** — the V3 compiler layer. It owns the IR
and the minting: `Class` / `ActionDef` / `ActionInvocation` (ogar-vocab), the
classid canon and content-blind facet, the frontends (`ogar-from-*`), the
adapters/emitters, codegen into consumer repos. It is where structure is
*assembled*. It is not a database.

**lance-graph is the spine; Lance is the disk.** lance-graph carries the
zero-dep *type* contracts (`lance-graph-contract` — which OGAR itself
consumes, e.g. `ogar-class-view` implements `lance_graph_contract::ClassView`)
and the query/planner machinery. Lance columnar storage is **where facts
calcify**: versions, tombstones, episodic history. Persistence is the
substrate's own calcification — Lance's columnar I/O writes LE bytes from the
in-place store ("zero-copy from creation to Lance tombstone"). **There is no
ingest API**: nothing "sends data to storage" as a request, and no producer
design should ever name one.

**The operator's two rulings, verbatim spirit:**
1. "There's a difference between storage and assembler substrate."
2. "Nobody would be stupid enough to use a hard-disk substrate for a
   compiler."

## Ownership quick-map (where does X live?)

| X | Home | Why |
|---|---|---|
| `Class` / `ActionDef` / `ActionInvocation` IR | OGAR (`ogar-vocab`) | assembler owns the IR |
| classid canon, facet, minting | OGAR | assembler owns identity |
| codegen / frontends / emitters | OGAR (`ogar-from-*`, `ogar-emitter`) | assembler produces code + SPO facts |
| zero-dep type contracts (`ClassView`, masks, sequencing primitives) | lance-graph-contract | the compile-time handshake OGAR + consumers build against |
| query engine, planner, temporal reads | lance-graph | the spine |
| persistence, versions, tombstones | Lance (via lance-graph) | calcification, not ingestion |
| live edges / new facts *landing* | the assembler-side graph of active record | facts are assembled, then calcify |

## Failure signatures (all observed, one session, 2026-07-14)

- **Wrong-repo filing:** "it's SPO-shaped → lance-graph `graph/spo/`". No —
  the triple store is query machinery; the *landing* of new facts is
  assembler-side. (lance-graph #691 → OGAR.)
- **"Ingest API" vocabulary:** any design that says a producer "ingests into
  storage" has inverted calcification into a service call.
- **"V3 substrate = lance-graph":** lance-graph hosts V3 *type contracts and
  docs*; the substrate being described — the graph of active record — is
  OGAR. Reading board docs' location as ownership is the trap.
- **Compiler-on-disk:** designing assembly/compile steps as if they run
  against storage. See the companion doc's door-knocking-compiler test.

## The 30-second checklist

- **Is this thing IR/minting/codegen?** → OGAR. **Types both sides compile
  against?** → lance-graph-contract. **Query/plan over persisted state?** →
  lance-graph. **Durability?** → calcification, not an API — stop designing
  an endpoint.
- **Before filing a cross-repo issue:** open the target repo's CLAUDE.md +
  doc family FIRST (OGAR's `docs/` index; consult-don't-guess is P0). The
  #691 failure was filing from pattern-match without reading OGAR at all.
- **Naming test:** if the design contains "ingest", "send to storage", or
  "ask lance-graph to", rewrite until those words are gone or the design
  moved to the runtime-engine document where it belongs.

Cross-refs: OGAR `CLAUDE.md` (canonical GUID/node canon, doc family),
`docs/OGAR-AS-IR.md` (the compiler framing), lance-graph
`E-LAYER-CONFUSION-OGAR-VS-SPINE-1`, MedCare-rs `E-MEDCARE-30`,
`compilation-vs-runtime-substrate.md`.
