# Firefly Frame & Cognitive Fabric

**Location**: `crates/lance-graph-cognitive/src/fabric/`

**Role**: The **unified execution substrate** (Cognitive Fabric) and its portable binary instruction format (**Firefly Frame**). This is the "machine code" / ISA for the Ada Cognitive CPU — a single 16384-bit (2 KiB) frame format that dispatches to multiple cognitive paradigms (NARS, Causal, VSA Memory, Lance vectors, Cypher graphs, Quantum-inspired, Control flow, Traps) with zero-copy semantics and built-in qualia + truth context.

This is the low-level hot-path encoding that makes the Resonance-Based Cognitive System, L1-4 cycles, and multi-language reasoning executable in a unified way.

## Firefly Frame Format (16384 bits / 256 u64 words)

The frame is a carefully packed binary structure designed for:

- **Zero-copy** execution directly on BindSpace
- **Multi-paradigm dispatch** via 4-bit language prefix
- **Rich context** (qualia, NARS truth <f,c>, temporal version, causal correlation ID)
- **Error resilience** (ECC field, currently simplified XOR parity with BCH planned)
- **Routing** across lanes, hives, sessions for distributed execution

### Bit Layout (from the spec)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FIREFLY FRAME (16384 bits / 2048 bytes)              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Bits 0-63    │ HEADER: Auth + Routing                                       │
│              │   Magic: 0xADA1, Version, Session ID, Lane ID, Hive ID, Seq  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Bits 64-127  │ INSTRUCTION: Opcode + Dispatch                               │
│              │   Language Prefix (4 bits) + Opcode (8) + Flags (4)          │
│              │   Dest / Src1 / Src2 registers (16-bit each, 8+8 addr)       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Bits 128-255 │ OPERAND: Extended payload (TRAP args, Cypher patterns, etc.) │
├─────────────────────────────────────────────────────────────────────────────┤
│ Bits 256-639 │ DATA: Payload fingerprint (384 bits)                         │
│              │   Truncated fingerprints, serialized patterns, vectors       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Bits 640-1023│ CONTEXT: Execution state (384 bits)                          │
│              │   Qualia [i8; 8], Truth (u8,u8), Version u64, Correlation ID │
├─────────────────────────────────────────────────────────────────────────────┤
│ Bits 1024-16381│ ECC + Payload extension (BCH planned)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Bits 16382-16383│ TRAILER: 0b11 delimiter                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Language Prefix Dispatch (4 bits → 256 ops each)

| Prefix | Paradigm       | Example Ops                                      | Connection to Prior Work |
|--------|----------------|--------------------------------------------------|--------------------------|
| 0x0    | **Lance**      | RESONATE, INSERT, HDR cascade, quantization      | Vector similarity / BindSpace |
| 0x1    | **SQL**        | SELECT, JOIN, AGGREGATE (via DataFusion)         | Relational layer |
| 0x2    | **Cypher**     | MATCH, CREATE, SHORTEST_PATH, TRAVERSE           | Hot-path Cypher in planner |
| 0x3    | **NARS**       | DEDUCE, INDUCE, ABDUCE, REVISE, ANALOGY, ATTEND  | NARS integration, truth values |
| 0x4    | **Causal**     | SEE (corr), DO (intervene), IMAGINE (counterfact), BUTTERFLY | Pearl rungs + Butterfly detection |
| 0x5    | **Quantum**    | SUPERPOSE, COLLAPSE, ENTANGLE, INTERFERE         | Superposition over CausalEdge64? |
| 0x6    | **Memory**     | BIND (XOR), UNBIND, BUNDLE, PERMUTE, CRYSTALLIZE | VSA / BindSpace core |
| 0x7    | **Control**    | JUMP, BRANCH, CALL/RET, FORK/JOIN, YIELD, LOOP   | MetaOrchestrator / lanes |
| 0xF    | **Trap**       | HALT, PANIC, IO, AUTH, CHECKPOINT, DEBUG         | System / FFI boundary |

This unifies what was previously discussed as separate "polyglot Cypher/Gremlin/NARS/SQL" into one compact tagged instruction stream.

## Key Rust Types (High-Signal API)

### Core Structs
- `FrameHeader` — Magic 0xADA1, session/lane/hive/sequence routing
- `LanguagePrefix` enum — The 9 dispatch targets above
- `Instruction` — prefix + opcode + ConditionFlags + dest/src1/src2
- `ExecutionContext` — qualia:[i8;8], truth:(u8,u8), version:u64, correlation_id:u64
- `FireflyFrame` — The 20-word (WORDS=20) complete frame with header, instruction, operand, data, context, ecc

### FrameBuilder (Ergonomic Construction)
Convenience methods that auto-increment sequence and produce ready-to-dispatch frames:

```rust
let mut frame = FrameBuilder::new(session_id)
    .lane(5)
    .hive(2)
    .resonate(dest, query_fp, k)           // Lance 0x00
    .cypher_match(dest, pattern_addr)      // Cypher 0x00
    .nars_deduce(dest, premise1, premise2) // NARS 0x00
    .bind(dest, a, b)                      // Memory 0x00
    .branch(condition, target)             // Control
    .trap(service, arg1, arg2);            // Trap
```

Also `nop()`, `halt()`.

### Encoding / Decoding
- `encode(&mut self) -> [u64; 20]` — Computes ECC on the fly
- `decode(&[u64; 20]) -> Option<Self>` — Verifies/corrects ECC, validates magic
- Header/Instruction/Context all have `encode`/`decode` helpers

**Note on ECC**: Currently simplified XOR + parity. Comment explicitly says real impl should use BCH(1247,1024) or similar for the large frame.

## Cognitive Fabric Architecture (from mod.rs)

The `fabric` module is the **unified substrate where all subsystems resonate**.

It re-exports:
- `butterfly`, `executor`, `firefly_frame`, `gel`, `mrna`, `shadow`, `subsystem`, `udp_transport`, `scheduler`, `zero_copy`

High-level flow (from module docs):
1. Firefly Compiler (Python side?) → parses programs → Graph IR → packs into 16384-bit Firefly Frames
2. Executor (Rust) → dispatches frames to the correct language runtime (zero-copy on BindSpace)
3. No external Redis/message queue needed — in-process execution with lanes/hives
4. UDP transport + scheduler for distributed/multi-lane execution
5. mRNA / CrossPollination + Butterfly detection for dynamic field interactions and sensitivity analysis

This directly implements the "cognitive-shader-driver is The SoA" and the L1-4 closed loop at the instruction level.

## Integration Points (Cross-References)

- **NARS**: Direct opcodes + truth values carried in `ExecutionContext`. Matches `CausalEdge64` + NARS truth discussion.
- **Causal / Pearl**: SEE/DO/IMAGINE + dedicated BUTTERFLY opcode. Ties to `butterfly.rs` and `CausalEdge64`.
- **BindSpace / VSA**: BIND/UNBIND/BUNDLE/PERMUTE/CRYSTALLIZE opcodes. Core of Memory prefix and `zero_copy.rs`.
- **Lance / Resonance**: RESONATE + HDR cascade ops. Connects to holograph/resonance layers.
- **Cypher hot-path**: Dedicated prefix + pattern embedding in operand/data.
- **MetaOrchestrator / Thinking Styles**: Control flow (FORK/JOIN/YIELD) + lanes provide the execution substrate for style-modulated cycles.
- **Qualia**: Carried in every frame context — enables qualia-aware dispatch and promotion membrane decisions.
- **Zero-copy & Scheduler**: `ZeroCopyExecutor`, `FireflyScheduler`, `DispatchPlan` — hot-path execution without serialization.

## Epiphanies & High-Signal Insights

1. **One ISA to Rule Them All** — Instead of separate interpreters for NARS / Cypher / SQL / VSA, everything compiles down to the same 2KB Firefly Frame. This is the ultimate unification of the "polyglot query layer" discussed earlier. Dispatch is a simple 4-bit prefix + table lookup.

2. **Context is First-Class** — Every instruction carries qualia + NARS truth + causal correlation ID. This makes meta-cognition, emotional steering, and causal tracking intrinsic to the instruction stream — not bolted on later. Perfect for the "meta awareness in every cycle" requirement.

3. **Butterfly as Primitive** — Having a dedicated BUTTERFLY opcode in Causal + a whole `butterfly.rs` module shows deep integration of chaos/complexity theory (small changes → large effects) into the causal reasoning engine. This is rare and powerful.

4. **Lanes + Hives + Sessions** — The routing fields enable massive parallelism (lanes) and clustering (hives) while keeping causal/session integrity. Combined with FORK/JOIN/YIELD, this is a full distributed cognitive runtime model.

5. **Zero-Copy + Fingerprint Everything** — Data is carried as fingerprints or truncated vectors. Combined with `zero_copy.rs`, this is designed for the exact hot-path constraints (20-200 ns) discussed throughout the project.

## Technical Debt & Open Items

- **ECC is placeholder**: Simplified XOR. Real BCH or LDPC needed for production 16K frames. Comment acknowledges this.
- **Compiler side missing**: The "Firefly Compiler (Python)" that turns high-level programs into frames is referenced but not in this Rust crate (probably external or in another repo).
- **Full opcode tables incomplete**: The dispatch table in the doc comment is illustrative; actual executor implementation in `executor.rs` may have more or different ops.
- **Quantum prefix**: Currently more inspirational than implemented (superposition over what state?).
- **Cross-crate consistency**: Need to ensure `CausalEdge64`, `SpoHead`, NARS truth packing, and Firefly `ExecutionContext` stay in sync.
- **Performance validation**: The encode/decode + ECC path needs benchmarking against the 20-200 ns target.

## Recommended Next Exploration

- Read `executor.rs` and `scheduler.rs` for dispatch logic
- Read `butterfly.rs` for the BUTTERFLY detection implementation
- Read `mrna.rs` and `gel.rs` for the mRNA / cross-pollination / GEL compiler aspects
- Check how `FrameBuilder` is used from higher layers (planner, cognitive-shader-driver)
- Map the 256 opcodes per language more completely (many are still "TBD" in spirit)

This module is **foundational** — it is the concrete realization of the multi-paradigm cognitive execution model that the entire lance-graph-cognitive crate (and likely the broader system) is built around.

---
*Generated from direct source analysis of `firefly_frame.rs` + `fabric/mod.rs` + directory structure. Fits perfectly into the Resonance + Causal + VSA + NARS architecture.*
*Low-entropy, high-signal documentation for session continuity.*