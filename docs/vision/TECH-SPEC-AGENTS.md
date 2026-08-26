# Tech Spec — The Lowering Substrate (for agents & engineers)

> **Audience:** engineering agents and contributors working inside the
> AdaWorldAPI workspace. This is the mechanical reference: what exists, what
> is proven, what is conjecture, and where the next work is. Every claim is
> tagged **[SHIPPED]**, **[PROVEN]** (a green falsifier exists), or
> **[CONJECTURE]** (design only). Do not cite a **[CONJECTURE]** as evidence.

## 0. One sentence

Lower any code — a C64 ROM, a C++ binary, legacy Java, an OWL rule set — to
one intermediate language (**R2IL**, Ghidra p-code transliterated), address
it by ordinal in **ogar-loco**, and execute it **zero-copy** through a
mask-native slab executor (**r2conc**); everything above that is a policy or
render *glove*, not a new engine.

## 1. The pipeline

```
INTAKE (once, C++ allowed)        RUNTIME (pure Rust, zero C, zero-copy)
────────────────────────          ──────────────────────────────────────
bytes ──libsla──► p-code          R2IL body ──r2conc.step──► machine state
  └─ Ghidra SLEIGH spec             │  register = borrowed slab
     (compiled .sla)                │  ram      = borrowed slab (the GUID is the pointer)
        │                           │  unique   = owned scratch
   translate → R2IL                 └─ Control::{Next,Jump,Call,Return}
        │
   ogar-loco: classid + lane ordinal   (address IS the program — §7.8 zipper)
```

- **INTAKE is serialization; RUNTIME is not.** The only place bytes are
  parsed is the lift. After that it is codebook-index masking algebra. This
  is the same cut the workspace already draws for every other intake arm.
- **Only `libsla` decode remains C++ at runtime.** r2sleigh already
  reimplements the downstream arms in Rust: `r2ssa` (SSA/heritage), `r2dec`
  (decompiler), `r2types` (type inference), `r2sym` (symbolic exec),
  `r2conc` (concrete exec). **[SHIPPED]**

## 2. R2IL is Ghidra p-code, transliterated — not a similar IL

Measured: **72 Ghidra p-code opcodes vs 82 R2IL variants.** The distinctive
tells — `Multiequal` (Ghidra's φ), `Indirect`, `PtrAdd/PtrSub`, `SegmentOp`,
`Cast`, `Extract/Insert` — are p-code names. `Varnode {space, offset, size}`
is `VarnodeData` field-for-field. The 10 extras are deliberate extensions
(`Fence`, `LoadLinked`, atomics, `Nop`, `CpuId`, `Breakpoint`). **[SHIPPED]**

> **HAZARD — `Negate` means opposite things on the two sides.** Ghidra
> `CPUI_INT_2COMP` = arithmetic `-x` maps to R2IL `IntNegate`; Ghidra
> `CPUI_INT_NEGATE` = bitwise `~x` maps to R2IL `IntNot`. Same word, swapped
> meaning. Nothing guards this today; pin it before porting between the two.

## 3. r2conc — the concrete slab executor **[SHIPPED]**

`SlabState` **borrows** `register` and `ram` as `&mut [u8]` (machine state
lives with the caller, never copied), owns `unique` scratch, registers
`Custom(n)` slabs explicitly. Values are transient LE `u64` loads/stores —
nothing keyed, hashed, cloned, or reified. Everything outside the concrete
subset (floats, atomics, `CallOther`, SSA-analysis forms) is **refused
loudly**, never silently skipped. Semantics anchored to `r2sym` for
differential runs. r2sleigh PR #5 (merged).

**Correctness boundary → security boundary:** the loud-refusal discipline is
a *correctness* boundary today. Under the zero-trust glove it becomes a
*security* boundary — see §6.

## 4. The falsifier ledger

| probe | claim | status | evidence |
|---|---|---|---|
| **ZIPPER-HOP-PARITY** | a rail descent replayed as explicit R2IL ops yields a **bit-identical mask** to the native hop — navigation and execution are one algebra (pure fragment) | **[PROVEN]** | OGAR `ogar-r2il/tests/zipper_hop_parity.rs`; 5 disable runs red-then-green |
| **LIVE-REGFILE** | a Ghidra-lifted 6502 routine executed by r2conc matches an **independently written** oracle on the full architectural state | **[PROVEN]** | r2sleigh `crates/r2conc/tests/live_regfile.rs`; 18/18, 7 disable runs; feature `probe-6502` |
| **R12 (flatten)** | Ghidra's p-code vocabulary **cannot flatten** into Valhalla (payloads 0/2, ordinals 3/3) — the Java seam must carry ordinals, not objects | **[PROVEN]** | lance-graph-java `R12_GhidraPcodeVocabularyVsCliff.java` |
| **R7 (zero-alloc)** | 10⁹ group projections allocate **exactly 960 B**, byte-identical across runs — zero per-op allocation | **[PROVEN]** | lance-graph-java `R7_BillionOpsZeroAlloc.java`; throughput **not** banked (1.76–3.48 s spread) |
| **LANES** | one R2IL body over N masked register slabs — the throughput claim | **[CONJECTURE]** | not built; cheap now (r2conc already borrows slabs) |
| **OWL-RL-FIXPOINT** | an OWL RL rule set lowered to R2IL, run to fixpoint (empty delta mask), matches the bake's own closure | **[CONJECTURE]** | not built |

## 5. LIVE-REGFILE found a real Ghidra defect — and the discipline that caught it

Ghidra's 6502 `ADC` (a slaspec dated 2006) computes carry-out **ignoring the
carry-in** and assigns `V := C`. Both wrong for a real chip. The probe drives
this deliberately and asserts the two sides **DISAGREE** (its can-it-fire
half). It also **corrected its own written prediction**: the multiply routine
was predicted immune (it `CLC`s before every `ADC`, never *reads* `V`) —
premises true, conclusion false, because `ADC` still *writes* `V`. Lesson
pinned: *"the buggy path is unreachable" is a claim about the whole
architectural state, not just the computed result.* **[PROVEN]**

## 6. The mask-native / zero-trust invariant (fix before any glove)

For any glove that runs **untrusted** code:

> **scan-then-execute, never execute-then-scan.** The malware-scan mask runs
> over the lifted R2IL *before* `step`; the sandbox executes only
> `lifted ∩ whitelist`; a lifted op never reaches a real effect without
> passing the policy mask first.

Whitelisting is a mask AND; malware detection is a masked pattern-match over
lifted R2IL; alerting is the alpha plane firing on an anomaly. Same
propose/verify machinery as the decode tier, pointed at "is this a known-bad
shape" instead of "which op is this."

## 7. The three-tier JIT (V4 = V3 + executable content) **[CONJECTURE]**

`V4 = V3 + executable content` — same bytes, same 512-byte stride, same
`ENVELOPE_LAYOUT_VERSION`; the version lives in the resolver's capability. A
classid can resolve to an R2IL body in the content tier.

| tier | engine | trigger | status |
|---|---|---|---|
| 0 decode | hexagon recall + exact ogar-r2il table verify | every op | tables **[SHIPPED]**; hexagon recall **[CONJECTURE]** |
| 1 interpret | r2conc over masks (one op, N lanes) | every (program, mask) | r2conc **[SHIPPED]**; the masked sweep is LANES **[CONJECTURE]** |
| 2 native | `ndarray::hpc::jitson_cranelift` | HOT programs | Cranelift engine **[SHIPPED]**; profile-drive **[CONJECTURE]** |

The profiler already exists: **`AlphaStamp.visits` is the hotness counter.**
Meta-awareness and profile-guided optimization are one mechanism at two rungs.

**Index-never-authority:** a learned encoding (hexagon) *proposes*; the exact
table *verifies*; the disagreement tail fails loud and becomes training
signal, never a silent mis-execute. The 99.6% hexagon figure is
operator-reported and **its artifact is not located in this workspace** —
treat as **[CONJECTURE]** until a reproducer lands.

## 8. What to build next, in order

1. **Pin the `IntNegate`/`IntNot` swap** with a test (§2 hazard). Cheap, high value.
2. **`OpBehavior` differential for r2conc** — `opbehavior.cc` is Ghidra's own
   executable per-opcode reference, *already compiled into the binary via
   libsla*. Diff every r2conc arm against it, exhaustively, **while the
   dependency still exists**. The result outlives libsla.
3. **PROBE-R2IL-LANES** — one body, N masked register slabs. The throughput
   number that makes "realtime" and the zero-trust sweep credible.
4. **PROBE-OWL-RL-FIXPOINT** — the projected-rule lens (predicate classid →
   R2IL template, axiom-row facets → operands).

## 9. Canon cross-refs

- lance-graph `r2il-machine-semantic-contract-v1.md` §7.7 (coupled cold/hot
  materials), §7.8 (V4, three-tier JIT, zipper isomorphism, OWL RL fence).
- lance-graph `EPIPHANIES.md`: `E-V4-EXECUTABLE-CONTENT-THREE-TIER-JIT-1`.
- lance-graph-java `EPIPHANIES.md`:
  `E-ONE-SUBSTRATE-FIVE-GLOVES-GHIDRA-IS-THE-GLOVE-NOT-THE-MODEL-1`,
  `E-LGJ-GHIDRAS-SEAM-IS-AN-INTERFACE-...` (R12).
- r2sleigh `ROADMAP.md` (Phase 2: "Decompiler Quality — match Ghidra, exceed it").
