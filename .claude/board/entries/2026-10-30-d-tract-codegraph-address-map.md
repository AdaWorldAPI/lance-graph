# D-TRACT-CODEGRAPH-ADDRESS (for DT64): how a codegraph gets an address, and where the codegraph already is

**Date 2026-10-30 · D-id D-TRACT-SHAPE-1 · Status OPEN (map, no code) · For the DT64 session**

The question: Tract compiles by rewriting one typed op graph (`declutter` then per-op `codegen` patches). We want the same codegraph for every disassembled program, addressed in the substrate. How is that address minted, where can the graph be inspected today, and should `lance-graph-r2il` get a second classid for it?

"KPEA" appears nowhere in lance-graph, OGAR or ndarray (searched; a search only finds candidates). This map reads it as "the codegraph Tract keeps and burn does not". Confirm or correct the term.

## 1. R2IL already has two classids for one table

`ogar-r2il` declares one opcode table under two provisional concepts in the BinaryLifting domain (`0xC4XX`):

| Concept | Reading | Source |
|---|---|---|
| `0xC400` `CONCEPT_R2IL_MACHINE` | scalar machine semantics (r2sleigh interpreter) | `ogar-r2il/src/lib.rs:536` |
| `0xC401` `CONCEPT_R2IL_FOLD` | the same table over populations (`lance-graph-r2il::FoldDialect`) | `ogar-r2il/src/lib.rs:545` |

Both are plugged with `VocabularyRegistry::plug(concept, &checked)` (`ogar-r2il/src/lib.rs:1491`). `compose` samples the table once, so arity, pushes and body refs are identical under both. Both are marked PROVISIONAL: not in `ogar-vocab`'s `CODEBOOK`, and no persisted GUID may carry them until they are.

A codegraph reading would be a third concept on the same table, e.g. `0xC402`. Its nodes are R2IL ops, its edges are def-use, and it is scoped to one function. It is the same pattern again, with no new vocabulary.

## 2. Minting the address

Per the 32-bit address (concept 16 | local 16):

- **Concept half (16 bits):** OGAR mints it. The three BinaryLifting concepts go into `ogar-vocab`'s `CODEBOOK` together, then through hotplug activation. Until then all three stay lab-only. This repo does not mint.
- **Local half (16 bits):** this is the app lens (`ogar-vocab/src/app.rs`). Per-op signatures are fields of that lens, exposed through `WideFieldMask` blocks of 256. That is what `tract-harvest-probe` already does for ONNX (65 fields, `Registry::revision()` digests the exposed set).
- The "behaviour-slot region `0xF000..0xFFFF`" in the architecture diagram is marked as a candidate. I found no code for it. Treat it as a proposal.

## 3. Where the codegraph can be inspected today

| Source | Graph | Rewrites / provenance |
|---|---|---|
| r2sleigh `r2ssa` (disassembled programs) | `SsaGraph` with `BlockId` / `InstId` / `ValueId` / `UseSite` (`graph.rs:13-244`), `CFG` (`cfg.rs:503`), `def_use` (`defuse.rs:123`), dominator tree, phi placement; `SsaArtifact` per function (`function/mod.rs:379`) | `OpArena` + `OpId`, with a `Pass` recorded for every op a pass minted or removed (`arena.rs:53,116`). This is the rewrite history. |
| r2sleigh `r2rewrite` | hash-consed term arena over `r2ssa` expressions | certified rewrites, each rule a proven equivalence: the analogue of Tract's declutter |
| r2sleigh `r2dec` | renders from sealed facts | codegen with an obligation ledger |
| Tract | `TypedModel` = `Graph<F,O>` (`core/src/model/graph.rs:32`); `tract dump --audit-json` | `declutter()` + `optimize()` (`typed.rs:185`); per-op `TypedOp::codegen` → `TypedModelPatch` (`ops/mod.rs:376`) |
| lance-graph `tract-harvest-probe` | `dr3::Tiled` op/dep graph (DR64, global to local, ≤ 65,536 positions) | none yet: it binds ops, it does not rewrite |
| `ogar-loco` `Program` | functions by index, bodies by reference byte (`program.rs`) | call-graph level only, no dataflow |
| burn-onnx | none at run time (emits Rust source + weights) | none |

So the codegraph for disassembled programs already exists. Every function that goes through `r2ssa` becomes an SSA graph with def-use edges, dominators and a pass log. What it lacks is a substrate address and a resident form.

## 4. `lance-graph-r2il`: second classid or second consumer?

`lance-graph-r2il` describes itself as the fold reading only: `FoldDialect` folds R2IL calls into mask-risc programs and holds no graph. A third reading belongs in a sibling consumer, not in that crate:

- **(a) recommended:** a new consumer reads `0xC402`. It imports an `SsaArtifact`'s ops and def-use edges as positions in a DR64 `Tiled` graph, keyed by (function GUID, `OpId`), with `Pass` provenance as a field. The fold dialect stays unchanged.
- **(b)** giving `lance-graph-r2il` a second classid would mix the fold reading with a graph reading in one crate whose doc says it is one reading.

## 5. Constraints this must respect

- **"R2IL is EXECUTED, never pre-converted"** (`ogar-r2il/src/lib.rs:26-35`, 2026-08-26). The `ruff_r2il` SPO pre-pass is ruled out as a lossy static shadow. A codegraph reading must therefore be an index over the same executed ops: addresses of `OpId`s, never an exported projection that replaces them. This is the main risk and needs OGAR's call.
- **DT64-5 scope:** no new runtime, no ONNX interpreter, no tensor AST. The codegraph is an addressed view, never an engine.
- **Zone:** `r2ssa` derives `serde` on its artifacts. The import must read the arena in place, not round-trip through a wire format.
- **Cost of "every program":** making a graph mandatory for admission (no substrate address without an `SsaArtifact`) is a gate on `r2engine`'s request scheduling. Its cost has not been measured.

## 6. A collision found while mapping

The probe's lab lens `ONNX_LENS = 0x0F00` sits in the Geo domain `0x0FXX` (`ogar-vocab/src/lib.rs:1160`). It is never persisted, but it must move. ONNX is not binary lifting, so OGAR decides where it belongs:

- an ONNX graph lowered onto the codegraph reading where its ops map to R2IL/fold ops, or
- its own concept.

## Open for DT64

1. What does KPEA mean: the Tract-style codegraph, or something else?
2. One table, three readings (`0xC402`), or a separate concept for the graph?
3. OGAR to mint `0xC400`–`0xC402` together?
4. Which crate owns the `r2ssa` import, and does it satisfy the execute-not-pre-convert ruling?
5. Where does the ONNX lens move to?
