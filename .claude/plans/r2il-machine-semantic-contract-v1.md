# R2IL as the machine-semantic contract — documentation, plan, status, integration

> **Status:** PLAN (v1, 2026-08-25). One measured finding underneath
> (`E-R2IL-MACRO-VOCABULARY-TRANSFERS-ACROSS-COMPILER-AND-LANGUAGE-1`);
> everything else here is PROPOSED and labelled as such, line by line.
> **Nothing in this document has been built.** No mint performed, no
> layout bump, no crate created.
>
> **Origin:** an operator sketch of the dependency layering (§2), plus a
> sibling session's live question — *"wie soll die Session R2IL
> speichern?"* — answered in §4. This plan exists so that question is
> answered once, from shipped code and measurement, instead of
> re-derived per session.

---

## §0 — The three registers, and why they are separate

The single most common failure in this workspace is a sentence that is
half description and half proposal, so a later reader cannot tell which.
This plan keeps them apart mechanically, and the separation is the
deliverable as much as the content is:

| register | answers | where it lives | rule |
|---|---|---|---|
| **DOCUMENTATION** | *what IS, today, in shipped code* | §3 here; module docs; `docs/` | every claim carries `file:line` or a measured number. If it cannot, it is not documentation |
| **PLAN** | *what is PROPOSED, and what would falsify it* | §5–§6 here; `.claude/plans/` | every item carries a falsifier and a kill condition. A plan item with no way to fail is a wish |
| **STATUS** | *where we actually are* | `.claude/board/STATUS_BOARD.md` | D-id + one of Queued / In progress / In PR / Shipped / Closed. Never prose |
| *(evidence)* | *what was measured, once* | `.claude/board/EPIPHANIES.md` | append-only; a finding is cited from here, never restated |

**The test, applied to this very document:** §2 is an operator sketch —
it is neither documentation nor measurement, and it is marked as such.
§3 is documentation and every row cites source. §4 is an ANSWER derived
from §3 + the evidence, so it inherits their status, not a higher one.
§5 onward is plan.

---

## §2 — The dependency layering (OPERATOR SKETCH, not yet built)

Stated by the operator 2026-08-25. Reproduced because its *correction of
an earlier framing* is the load-bearing part:

```
                    GHIDRA
          UI / plugins / analyzers / API   (Java, unchanged)
                      │  consumes a Java compatibility surface
                      ▼
             lance-graph-java              (Valhalla descriptors / views)
                      │  Panama FFM — crossed ONCE
 ═════════════════════╪═════════════════════════════════
                      ▼
                  r2sleigh                 (SLEIGH/P-code → R2IL translator)
                      │  consumes the r2il crate
                      ▼
                    R2IL                   (faithful machine-semantic ISA)
                      │  lowers onto
                      ▼
                  SoA V4                   (lanes / masks / SIMD / ClassView)
                      │
                 lance-graph
```

**The correction, in the operator's own terms.** An earlier framing had
`r2sleigh` owning the native representation:

```
   WRONG:  Ghidra Java → Panama → r2sleigh → "R2IL SoA"
                                  ^^^^^^^^ owns a private object graph,
                                           serializes some R2IL later

   RIGHT:  r2sleigh USES R2IL as its machine-semantic contract,
           and R2IL itself is backed by SoA V4.
```

Consequence: **r2sleigh does not invent a second storage physics.** It
becomes primarily the SLEIGH/P-code → R2IL translator. That is not a
stretch — it is what its own README already describes (§3).

**The Ghidra half — compatibility FACADE, not compatibility DATA MODEL.**
`instruction.getPcode()` keeps working, but the implementation becomes
`InstructionRef → block handle → SoA slice/mask → lazy materialization`.
The cost of manufacturing `PcodeOp[]` is paid ONLY when old Java asks for
it; new analysis operates on masks and blocks and never pays. This is the
same doctrine `lance-graph-java`'s own mask-native policy already states
for `where`/`hop`/`compute` — Java-surface convenience never dictates
substrate representation.

---

## §3 — DOCUMENTATION: what ships today (every row cited)

| claim | source |
|---|---|
| r2sleigh's own pipeline is `.sla (Ghidra) → libsla → P-code → r2il → {ESIL, SSA, decompiler, type inference}` | `r2sleigh/README.md` |
| `r2il` is the typed IL (60+ opcodes); `r2sleigh-lift` is the SLEIGH/P-code translation layer | `r2sleigh/README.md`, `crates/{r2il,r2sleigh-lift}` |
| **`libsla` is a dependency of exactly ONE crate** | `crates/r2sleigh-lift/Cargo.toml:10` — the whole workspace's Ghidra-native coupling is isolated to that line |
| `ruff_r2il` consumes `r2il`/`r2ssa` by path and melts them into flat facet-addressed rows | `ruff/crates/ruff_r2il/{Cargo.toml:19, src/lib.rs}` |
| The R2IL address today is `VarnodeFacet` = 16 bytes, `classid \| offset_lo \| offset_hi \| size` | `ruff_r2il/src/facet.rs:40,232` |
| The 12-byte content-blind register carving is `CascadeShape::{G6D2,G4D3,G3D4}`, all `G·D=12` | `lance-graph-contract/src/facet.rs:359-431` |
| …and is deliberately MIRRORED in OGAR as `LaneShape::{Pairs,Triples,Quads}` | `OGAR/crates/ogar-loco/src/lib.rs:1-100` ("mirroring the LE contract's CascadeShape"); graded ESTABLISHED by the LG #1023 audit |
| `FacetCascade` decode is a pointer reinterpret, not a copy | `facet.rs:93` `#[repr(C, align(16))]`; `as_bytes`/`ref_from_bytes`; test `reinterpret_is_a_no_op` asserts POINTER IDENTITY |
| …and that zero-copy path has **zero non-test consumers** — every real caller uses the copying `from_bytes` | measured 2026-08-25: `ref_from_bytes` appears nowhere outside `facet.rs` |
| The macro vocabulary space is `System[256] \| Learned[≤256] \| Explore[≤256]`, palette-indexed per lane | `lance-graph-contract/src/soa_view.rs:41`; `E-BPE-OVER-DEFUSE-CHAINS-BEATS-LINEAR-AND-FITS-LOCO-1` |
| R2IL's real opcode census is 9 RISC-like opcodes — **not** a bitmask ALU | LG #1023 audit Challenge 1, 143 fns / 17,557 rows |
| "IR lands in V4, physically identical to V3 — no `ENVELOPE_LAYOUT_VERSION` bump" | operator-stated, recorded in the POC entry |
| `CONTENT NEVER TRAVELS IN CLASSID. CLASSID SELECTS THE READING.` | ratified PR #1012, `E-CONTENT-NEVER-TRAVELS-IN-CLASSID-1` |

**One documented defect, found 2026-08-25 and not yet filed as work.**
`facet.rs:232` computes `classid = (concept << 16) | space_discriminant`.
For the FIXED spaces (`Register`/`Ram`/`Const`/`Unique`) that is
legitimate — the space genuinely selects how the remaining 12 bytes are
read, which is exactly what a classid is for. For **`SpaceId::Custom(n)`**
it is not: `n` is a per-binary ordinal lifted out of the program, i.e.
content in the reading selector, against both the ratified law above and
OGAR's canon (*"lo u16 = APP render prefix — NEVER a shape ordinal"*).
The open item O3 currently reads *"does `Custom(u32)` fit the 16-byte
projection?"*; under this reading the question is not whether it fits but
that it does not belong. **CONJECTURE** — the falsifier is in §6.

---

## §4 — THE ANSWER: how a session should store R2IL

The sibling session's question, answered from §3 + the evidence, in five
rules. Status inherits from the sources: rules 1–3 are documentation,
rules 4–5 are plan.

**R1 — Do not build a private object graph and serialize R2IL later.**
That is the exact shape the operator's correction rejects, and the
Firewall (ADR-022/023) forbids serialization in the hot path regardless.
`to_le_bytes`/`from_le_bytes` ARE the format.

**R2 — R2IL lands as a V4 tenant that is PHYSICALLY V3.** 16-byte
content-blind facet, `classid` selects the reading, no
`ENVELOPE_LAYOUT_VERSION` bump. V4 is an additive sibling tier for R2IL's
100%-coverage requirement (operator ruling 2026-08-21,
`E-V4-IS-THE-100-PERCENT-TIER-V3-UNCHANGED-1`); V3 continues unchanged
for everything it already carries.

**R3 — The 12 bytes are a dumb register the ClassView projects.** Carve
per `le-contract.md` §3 — `6×(8:8)` rails / `4×(8:8:8)` / `3×(8:8:8:8)`.
`u8:u8` stays two separate bytes, never widened. Reads go through
`ref_from_bytes` (a reinterpret), not `from_bytes` (a copy) — see the
§3 row saying nobody currently does this.

**R4 — The macro vocabulary is a PALETTE, not a `FnIndex` per macro.**
`ogar-loco` ROUTES microcode into the `System`/`Learned`/`Explore`
palettes; the "90 free slots" number is one encoding's headroom, not a
vocabulary ceiling. §5's evidence says one palette can serve several
programs and two toolchains, so the palette is shared, not per-binary.

**R5 — Behaviour travels by ADDRESS, never inline.** The macro id is an
ordinal into a class's set; the class's `ClassView` resolves it. An
inline lambda on the surface is the SURREAL-AST trap in its R2IL edition.

---

## §5 — THE EVIDENCE that makes the shared palette defensible (2026-08-25)

`E-R2IL-MACRO-VOCABULARY-TRANSFERS-ACROSS-COMPILER-AND-LANGUAGE-1`, in
one table. Trained on the two gcc `stress_test` binaries only; scored by
macro-hit density per def-use chain (scale-free); two pre-registered
nulls, 20 seeds each, multiset assertions on every draw:

| held-out corpus | density | vs train (2.529) | strict (column) null | REAL in range? |
|---|---|---|---|---|
| unseen C program, same gcc | 2.515 | −0.6% | 1.969 [max 1232 hits vs real 1529] | no |
| unseen Rust (rustc/LLVM) | 2.409 | −4.7% | 2.009 [max 17461 vs real 20861] | no |

The vocabulary is not a gcc idiom; crossing the language boundary costs
4.7%, not a collapse. The pre-registered kill condition (REAL inside the
column-null range ⇒ split the palette per toolchain) did not fire. Fences
carried, not waived: x86-64 only, pass-1 seven-opcode convention,
chain-length 3, the Rust sample capped at 200/548 functions.

**Why this matters to §4/R4:** a palette entry is only mintable as SHARED
vocabulary if it means something outside the binary it was learned from.
That is now measured on the axes available in this workspace. What it
does NOT establish: that the shape is the same THOUGHT across languages —
co-occurrence is measured, semantics is not.

---

## §6 — PLAN: the wave order, each with its falsifier

No wave starts before its predecessor's falsifier is green. Model policy
per workspace rule (grindwork → Sonnet; accumulation/orchestration/gates
→ Opus; gates run centrally, once).

| wave | deliverable | falsifier / kill condition |
|---|---|---|
| **W0** | `Custom(n)` census (closes §3's defect + O3): over the 4-binary corpus, count distinct `Custom(n)` and whether the same `n` denotes the same thing across binaries | same `n`, different meaning across binaries ⇒ `Custom(n)` is CONTENT ⇒ it moves OUT of classid into payload/edge before ANY mint. Same meaning everywhere ⇒ reading, stays |
| **W1** | the R2IL V4 tenant spec: ClassView carving of the 12 bytes for Op / Varnode-ref / macro-ref rows, written against `le-contract.md` §3 | field-isolation matrix test (I-LEGACY-API-FEATURE-GATED): write each field, assert all others unchanged. Any aliasing ⇒ re-carve |
| **W2** | the `0xC4 BinaryLifting` mint (ruff PR3 arc, O5) — container concepts only, gated on W0's verdict for the space axis | mint request names concepts; a concept that encodes per-binary content is rejected by W0's rule |
| **W3** | palette wiring: macro-id byte → `{Learned,Explore}Palette[id]` → R2IL×BPE microcode, `ogar-loco` routing | B4-equivalent byte-exact round-trip through the palette; admission stays MUL's/the triangle's, NEVER the wiring's |
| **W4** | `r2sleigh` seam: `r2sleigh-lift` emits INTO the tenant (R1 — no private graph then serialize). libsla STAYS — it is the oracle | parity: lift the same binary via today's path and via the tenant; byte-equal FlatFact streams. libsla removal is LAST (see below), never first |
| **W5** | `lance-graph-java` facade: `PcodeOp`/`Instruction`/`Varnode` as lazy views over handles + masks | the mask-native gates (GraphHopTest allowlist, allocation gates) extended to the new views; `getPcode()` pays materialization ONLY when called — measured, not asserted |

**libsla is removable scaffolding, and deliberately LAST.** While it is
in place, both sides read the same SLEIGH truth, which makes it the
byte-parity oracle for every wave above (the tesseract-rs method). It is
removed only after W4's parity is green — the "final removable
scaffolding", never an early surgery. This also keeps ruff PR4
("libsla-optional") correctly sequenced: it is a *candidate* after W4,
not a prerequisite of anything.

**Open questions this plan does NOT decide** (inherited from the V4
ruling, still with the operator): V4's exact special-needs set beyond
R2IL; the V3-vs-V4 routing rule; whether the W2 mint is registered as V3
or V4.
---

## §7 — ADDENDUM 2026-08-25: the white/grey reading, hex demoted to a testable overlay, and the demotion gate

> Register: PLAN + vocabulary. One operator-converged framing, recorded
> because it names shipped structure correctly and turns the one unbacked
> piece (hexagonal topology) from a rejected reading into a falsifiable
> experiment. Nothing here changes W0–W5.

### 7.1 The layer vocabulary (naming, not new machinery)

```
PHYSICS       SoA V4                       (shipped)
SEMANTICS     R2IL contract                (shipped; ruff_r2il consumes it)
ADDRESSING    Cartesian / Morton / HHTL    (shipped; 256×256 per rail, 4⁴ centroids)
HOLOGRAPHY    bounded VSA superposition    (shipped, NICHE: N ≤ √d/4 ≈ 32,
                                            I-VSA-IDENTITIES; not a storage story)
VOCABULARY    transferable BPE macros      (measured 2026-08-25: −0.6% gcc→gcc,
                                            −4.7% gcc→rustc, both outside the
                                            marginal-preserving null)
COGNITION     learned topology + resonance (System/Learned/Explore lanes shipped;
                                            the TOPOLOGY half is UNBUILT)
```

White matter = PHYSICS…VOCABULARY's exact half (deterministic, addressed,
replayable; `expand(macro) → exact R2IL` is B4, already green). Grey
matter = the plastic half (resonance, counterfactual lane, revision —
`PROBE-METACOGNITIVE-TRIANGLE-1`, PR #1013's veto). The bridge invariant,
already ratified in the POC entry: **a macro never becomes a second
truth.**

### 7.2 Hexagonal topology — demoted correctly, and thereby testable

The #1023 audit's FALSE/net-new verdict on "6 directional 16-bit facets"
stands UNTOUCHED: hex is **not** the storage geometry, and no rereading
of the 96-bit register claims otherwise. The corrected proposal is an
**optional learned neighbourhood graph ON TOP of the Cartesian crystal**:

```
Cartesian resident crystal → addressed cells/masks/edges
    → optional learned neighbourhood overlay → 6-neighbour resonance
```

**PROBE-GREY-TOPOLOGY-AB (unbuilt, gated).** Two topologies, same macros,
same tasks, same activation rules:
  A = Morton/Cartesian neighbourhood (the substrate's native reading)
  B = six-neighbour graph overlay
Metrics: pattern completion, transfer, steps-to-convergence, false
resonance, memory footprint, counterfactual quality.
Kill condition: B must beat A on ≥2 metrics without losing on footprint
by more than it gains — otherwise the crystal stays square.

**Honest gate:** the experiment as specced requires a learned-topology
layer that does not exist. The cheapest real first version needs no new
tissue: the 33-macro co-occurrence graph over the existing 4-binary
corpus IS a learned neighbourhood — run completion/false-resonance on it
under both adjacency readings before building anything. If even that
cheap form shows no B-advantage, the full probe is not worth its cost.

### 7.3 Crystallization has a measurable admission criterion now — and needs a DEMOTION gate

Admission (Explore→Learned→System) stays MUL's + the triangle's. What
2026-08-25 adds is the evidence shape: **a System candidate must
(a) B4-round-trip, (b) fit a loco lane, (c) fire across program AND
toolchain boundaries outside the marginal-preserving null.** On today's
corpus: 25 macros qualify, 7 stay lane-local, 2 (`fits-NONE`, both
binary-local) never crystallize.

**The missing half, required before any promotion ships:** the demotion
path. A System macro that falls INSIDE the null range on a new corpus
goes back to Learned — same gate, run in reverse. Without it,
crystallization is the 150/150 guard one level up
(`E-ANTI-EIGENVALUE-…-1`): a vocabulary that can only grow carries
progressively less information per entry. Promotion and demotion are one
mechanism read in two directions, and the falsifier for the pair is:
inject a deliberately corpus-local macro into System, present the new
corpus, assert it is demoted (can-fire) — and assert the 25 transferable
ones are NOT (can-stay-silent).
