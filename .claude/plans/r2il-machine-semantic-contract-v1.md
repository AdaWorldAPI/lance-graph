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

> **⊘ RESOLVED 2026-08-25 by W0 — and the conjecture above named the WRONG
> mechanism.** `Custom(n)` is NOT per-binary: the table is built by
> `CustomSpaceTable::from_arch` from `ArchSpec.spaces`, i.e. per
> ARCHITECTURE, so within one arch the ordinal is stable across binaries.
> The real defect is one level up — `ordinal_of` returns
> `CUSTOM_ORDINAL_BASE + rank(raw)`, so the lo-u16 is a RANK inside a table
> the classid never names, over a raw id that is itself an upstream
> registration-order counter. Two facets from two arches are byte-identical
> while denoting different spaces (demonstrated against the shipped API).
> Census: **0 custom spaces in 94,536 rows across all four binaries** — the
> case is LATENT on x86 and fires first on the 6502/C64 arc.
> **⊘ AND IT NOW FIRES (same day):** the arch census over the real SLEIGH
> specs finds the 6502 minting TWO custom spaces — `OTHER` → `Custom(0)`
> and, because the alias map is case-SENSITIVE, its own main memory `RAM`
> → `Custom(1)`. The 6502's RAM is not `SpaceId::Ram`. Latent → live;
> verdict unchanged, urgency changed. See
> `E-W0-IS-LIVE-ON-THE-6502-AND-ITS-MAIN-MEMORY-IS-NOT-SpaceId-Ram-1`.
> Verdict: fixed
> spaces 0–3 stay; the custom axis is BLOCKED from the `0xC4` mint as
> carved. Full entry:
> `E-W0-THE-SPACE-ORDINAL-IS-A-RANK-RELATIVE-TO-A-TABLE-THE-CLASSID-NEVER-NAMES-1`.
> Prior art: `facet.rs:18-20`'s own `⚠ Known tension` note flagged the
> shape-ordinal problem first; W0 supplies the measurement and mechanism.

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
| **W0** | ~~`Custom(n)` census~~ **RUN 2026-08-25 — see the ⊘ note in §3.** Census: 0 custom spaces in 94,536 rows / 4 binaries (latent). Mechanism falsified in code: the lo-u16 is a rank in an unnamed table; two arches collide byte-identically | **VERDICT: fixed spaces 0–3 stay; the custom axis is BLOCKED from the `0xC4` mint as carved.** The kill condition fired on a third possibility the row did not anticipate — neither "content" nor "reading", but a reading relative to a table absent from the address |
| **W1** | *(status 2026-08-25: **MAY PROCEED**, independently of W0's verdict — insofar as its carving work does not depend on the custom-space decision. Where it does touch that axis, it inherits W2's block.)* the R2IL V4 tenant spec: ClassView carving of the 12 bytes for Op / Varnode-ref / macro-ref rows, written against `le-contract.md` §3 | field-isolation matrix test (I-LEGACY-API-FEATURE-GATED): write each field, assert all others unchanged. Any aliasing ⇒ re-carve |
| **W2** | ⛔ **BLOCKED-BY-OGAR-MINT-DECISION (W0 ran 2026-08-25).** The gate is no longer *"W0 decides how the space axis is carved"* — W0's verdict is that the **custom axis must not enter the mint in its current carving at all** (the lo-u16 is a rank inside a table the classid never names; two arches collide byte-identically). Fixed spaces 0–3 are unaffected and stay. **Do not resume W2 on the space axis** until OGAR rules on one of the three named-but-undecided options (arch in the address / raw SLEIGH identity / custom space outside the classid). Container concepts that do not touch the space axis are not blocked by this. | the original row's falsifier stands for the concept half; for the space half the block is the verdict, not a test to re-run. Entry: `E-W0-THE-SPACE-ORDINAL-IS-A-RANK-RELATIVE-TO-A-TABLE-THE-CLASSID-NEVER-NAMES-1` |
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

### 7.4 Operator refinement, same day: the reversible-trust state machine

Converged wording, recorded verbatim in substance because it is the
demarcation the rest of the plan leans on:

> **White Matter ist Wahrheit und Zwang. Grey Matter ist Hypothese und
> Plastizität. Crystal ist der reversible Vertrauensstatus gelernter
> Muster. Hexagon ist noch gar nichts außer einem Kandidaten für lokale
> Rechengeometrie.**

`System` therefore means *"currently confirmed strongly enough to be
treated as a fixed palette"* — long-term memory, never truth. Truth is
carried by white matter only (the CLAIM boundary, B4 expansion, the
contract veto). Crystal is a STATUS of a pattern, not a third storage
technology.

```
             promote                    promote
Explore ───────────────→ Learned ───────────────→ System
   ↑                        ↑                        │
   └──────── demote ────────┴──────── demote ────────┘
```

**Symmetric criteria, one epistemic yardstick in both directions:**

```
PROMOTE if   B4 exact round-trip
             AND fits loco geometry
             AND transfer density stable
             AND REAL outside the marginal-preserving null
             AND the contract falsifier passes

DEMOTE  if   B4 breaks
             OR the contract falsifier fails
             OR a new held-out corpus falls into the null range
             OR transfer density collapses beyond admitted tolerance
```

**Hysteresis is REQUIRED, not optional:** `promote threshold > demote
threshold` — so a single exotic binary cannot flap the state machine at
a boundary case. This is an anti-flutter measure, not leniency; the
tolerance values are POLICY PINS to be measured when the first promotion
ships, not defended.

**Hex burden of proof, sharpened.** The question is never "can a hexagon
do association" (the shipped systems already do). It is: *can a
6-neighbour topology beat the existing Morton/Cartesian structure
measurably* — same macros, same corpus, same promotion rules, same
resonance rule. Metrics extend §7.2's list with: promotion stability and
demotion rate. If hex wins, it wins as a COMPUTE topology — never
retroactively as an explanation of the 96-bit register.

### 7.5 Calibration-before-demotion — the instrument must qualify before it may judge (operator-hardened, 2026-08-25)

> **A corpus may falsify a crystal only after it has falsified the
> hypothesis that it is an incompetent instrument.**

**⊘ The defect this corrects was in MY first draft of the positive
control, and the operator caught it.** The draft rule — "a corpus may
only demote if the 25 established macros do NOT fall into its null
range" — contains an immortality loop: the very macros a corpus might
legitimately demote also decide whether the corpus is allowed to demote
at all. `25 fall into the null → corpus ruled bad → the 25 may not be
demoted.` A self-immunizing vocabulary is the anti-eigenvalue failure
wearing a lab coat. Recorded rather than silently fixed.

**The rule:** corpus qualification must be INDEPENDENT of the candidate
it judges.

```
CORPUS QUALIFICATION (all six, before any demotion verdict)

1. sufficient chains / effective sample size
2. no degenerate opcode monoculture
3. the null model's variance/range has not collapsed
4. known POSITIVE controls separate from the null      (known signal → seen)
5. known NEGATIVE controls stay in/near the null       (known noise → not seen)
6. the candidate under evaluation takes NO part
   in its own corpus qualification
```

Point 5 is load-bearing: a positive control alone cannot catch a corpus
that makes EVERYTHING look strong. The instrument must demonstrate both
halves — `known signal → erkannt, known noise → nicht erkannt` — before
it may rule on an unknown.

**Cheap immediate form — leave-one-out calibration:**

```
demote M17?
  qualify the corpus using  {established macros} \ {M17}  + frozen negative controls
  only then evaluate M17
```

M17 can now actually die.

**Target form — a frozen calibration panel, NOT identical to the System
palette:**

```
CAL+   a few provably portable R2IL macros (frozen at panel creation)
CAL−   shuffled / corrupted / contract-invalid macros
       (seed material exists: B4's corruption demo already constructs a
        contract-invalid macro and proves the check catches it)

held-out corpus → CAL+ separates? → CAL− stays null? → corpus is
COMPETENT → may contribute demotion evidence
```

**And even a qualified failure is EVIDENCE, not a verdict** — the §7.4
hysteresis applied to demotion:

```
qualified failure → demotion evidence accumulates
                  → crosses the measured lower threshold
                  → System → Learned
```

Otherwise "never demote" is merely traded for "one strange Rust binary
in a bad mood empties the library." The accumulation threshold is a
POLICY PIN, measured when the first real demotion case exists.

This is the same philosophy as every falsifier in this workspace, one
level up: not only the specimen must be falsifiable — **the instrument
must prove it can currently measure.** (Precedent, small scale: the
shuffle runs of 2026-08-25 asserted their multiset/marginal preservation
on every draw — each null was itself checked before its verdict counted.)

### 7.6 §7.2's probe gets its real task: ONTOLOGY-MORPHOGENESIS (operator-approved + leakage-hardened, 2026-08-25)

Supersedes the "cheap first version" in §7.2 (the 33-macro co-occurrence
graph) as the PRIMARY form of `PROBE-GREY-TOPOLOGY-AB`: independently
curated open ontologies (MONDO / HPO / GO / ChEBI — different projections
of one biological reality) are a far stronger oracle than internal
machine patterns held against each other. The baked corpus already
exists on the consumer side (the open-ontology SoA bake with
`(classid, identity)` lanes and `part_of:is_a` rails); the data half of
this probe lives in that repo's own ledger, only the topology half here.

```
PROBE-GREY-TOPOLOGY-AB / ONTOLOGY-MORPHOGENESIS   (unbuilt, gated)

INPUT:
  exclusively intra-ontology structure:
  is_a, part_of, roles, evidence, causal edges.
  LEAKAGE FENCE (operator-added, load-bearing): not only explicit
  crosswalks are withheld — DERIVED features that already betray them
  are banned from the fold rule too: no xref-derived labels, no
  pre-normalized shared IDs, no bridge features that carry the answer.

WITHHELD ORACLE:
  the known cross-ontology mappings (xrefs, curated anatomy mappings,
  curated axis crosswalks). Hidden during folding; scored against after.

COMPARE (same data, same fold rule, same budgets):
  A = the substrate's native Morton/Cartesian neighbourhood
  B = the experimental local topology (hex, if it applies)

NULLS (all four; each must be qualified per §7.5 before it may veto):
  degree-preserving rewiring          (kills hub bias)
  relation-type shuffle
  label/identity shuffle
  is_a-DEPTH-PRESERVING leaf permutation   (kills common-root attraction —
      without this, "everything near the root looks alike" gets celebrated
      as morphogenesis)

GATE:
  A fold is interesting only if it survives destruction of the semantics
  that supposedly caused it AND recovers withheld mappings better than
  every qualified null/control.
```

**§7.5 applies to the oracle itself:** a null model or held-out set may
demote a fold only after proving it can separate known withheld
relationships as its positive control. A broken oracle gets no veto.

**Sequencing, deliberately two-phase:** first the probe must RECOVER
what curators independently already knew, despite it being hidden —
self-supervised structure discovery with uncontaminated ground truth.
Only THEN do the novel basins (folds with no existing crosswalk) become
admissible as candidates — and they enter as Explore-status crystals
under §7.4/§7.5, never as findings. The ontologies themselves stay
separately true throughout: the geometry learns PROXIMITY, white matter
keeps the explicit edges, provenance, and truth.

**Both outcomes pay:** if B/hex does not beat A on withheld recovery,
false resonance, and promotion/demotion stability, hex is dead and the
crystal stays square — a real answer. If it wins, the honeycomb has its
first empirical lease.

**Responsibility boundary (operator, same day — the cut that keeps this
plan from owning MedCare):**

```
lance-graph plan:   defines the topology experiment + falsifiers.  NOTHING ELSE.
MedCare-rs:         owns corpus, bake state, withheld mapping lists, receipts
                    (its own ledger, RAIL_OFFENE_POSTEN).
```

§7.6 defines the FORM of the experiment. The moment a future edit here
starts specifying ontology ingestion, bake artifacts, or concrete
crosswalk lists, the clean cut has failed — move that text to the
consumer repo and leave a pointer.

### 7.7 ADDENDUM 2026-08-26 — no executive chooser; the two materials are COUPLED, not controller-and-controlled

> Register: PLAN + vocabulary correction. Operator-ruled. Scoped to this
> plan's own boundary (§7.6): the FORM of the mechanism and its topology
> falsifier. The consumer-side half — expected-vs-observed firing over a
> real corpus, focus dynamics, candidate pruning — belongs to MedCare-rs
> and is recorded there, not here.

**The rejected formulation, recorded because it is the tempting one.** A
session proposed closing the activation loop with

```
chooser(spine, alpha) -> next_address
```

— an executive selector holding both planes and deciding. **Rejected.**
It reintroduces a scheduler layer the architecture does not have, and it
makes the transition an act of selection rather than a consequence of
substrate state. The wording never reached a document; it is written down
here so it is not re-proposed as an obvious move.

**The corrected mechanism — two materials running simultaneously:**

```
COLD  readonly ontology/SoA + baked executable routes
      → where activation is EXPECTED to conduct
HOT   local reaction field: activation · focus · signed conductance
      → where it ACTUALLY conducts, and where resolution is concentrated

cold route fires → hot field reacts → hot field changes local conductance
                → another cold route fires → repeat
```

Neither material controls the other. The next firing is a consequence of
the coupled state. This is a strictly stronger claim than the open-loop
diagnosis it replaces, and a strictly weaker one than a chooser: it must
be shown that the coupling *alone* produces a transition.

**The local field carries TWO readings, not one.** Activation (which
material is firing) and focus (where gain/resolution is concentrated) are
co-located at the same coordinate. Focus is **not** a selector — it is
modulation, `effective ≈ activation × focus`, so a wide region may be
active while attention occupies one aperture. A saccade is then the
movement of the focus field over the active field, never a pick from a
candidate list. **Do not hard-code this as an ABI struct**; the
requirement is semantic co-location at one address, and §7.2's verdict
that the geometry itself is unearned is untouched by it.

**Signed learned conductance is a hypothesis with a separation
requirement.** Token/macro identity stays a stable id; the *contextual*
contribution may be a signed `i8` (positive retain, near-zero dormant,
negative suppress-from-frontier). Suppression removes a candidate from
the current frontier and **never** deletes its canonical identity. The
ordering that follows — **prune before hydration**: compact weights →
mask/threshold/top-k → hydrate only survivors — is an efficiency claim
and owes the null in `F-HOT-6` (recorded consumer-side).

**Unchanged and re-affirmed:** `R2IL carries fidelity, BPE carries
repetition` — `expand(macro)` must be byte/operation-identical to the
underlying sequence (B4). Hot learned wiring is **not** a world-causal
claim: it may never create or mutate a CE64 edge, and the promotion
ladder of §7.3/§7.4 (with its demotion half and hysteresis) governs any
crystallization back into cold material. Frequency is not truth.

**Topology falsifier, extending §7.2's control set:**

```
F-HOT-9   Replace the local geometry with random-6 / square-4 / square-8 /
          k-NN under an otherwise identical coupled loop.
          If behaviour is unchanged, the GEOMETRY is rejected while the
          hot-field abstraction survives.
```

That split is the point: this addendum increases what the hot field is
asked to *do* without granting the honeycomb one inch of unearned lease.

### 7.8 ADDENDUM 2026-08-26 — V4 = V3 + executable content; the three-tier JIT; OWL lowered to IR

Operator-ruled, same day as §7.7. Three sentences of doctrine, then the
architecture they imply. All CONJECTURE-graded until the probes at the end
run; the doctrine sentences themselves are operator rulings, not findings.

**The doctrine (pin verbatim):**

1. **V4 = V3 + executable content.** Nothing about the bytes moves — same
   4+12 facet, same 512-byte stride, same `ENVELOPE_LAYOUT_VERSION`. What
   changes is what a classid can RESOLVE to: a ClassView whose content tier
   holds executable R2IL. A V4 row IS a V3 row; a V3 reader just doesn't
   know it can run. The "version" lives in the resolver's capability —
   exactly where I-LEGACY-API-FEATURE-GATED wants it. No migration, no
   layout gate, no bit reclaim.
2. **The learned encoding is an INDEX, never the AUTHORITY.** The hexagon
   substrate's measured 99.6% op-decode (3 training programs, muscle-memory
   arc) is a recall number, and execution cannot ride on recall — a
   mis-decoded op that executes is silent corruption. Two-tier decode:
   hexagon PROPOSES (associative, ~always right), the exact ogar-r2il
   tables (ARITY/MNEMONICS, O(1)) VERIFY. Agree → execute. Differ → table
   wins, the disagreement becomes a training example. The 0.4% tail fails
   LOUD into the slow lane and teaches; it never mis-executes. Same
   relationship CAM-PQ has to the full fingerprint (search tier vs truth
   tier, I-VSA-IDENTITIES).
3. **Execution is population-masked, never per-row.** The interpreter's
   unit of work is (program, mask), not (program, row). One instruction,
   N lanes: the varnode read/write is an ndarray SIMD sweep over the
   facet-register columns; dispatch amortizes over the mask's popcount.
   Divergence = split the mask, run both arms masked, re-join (an
   `and_not`, not an architecture). The interpreter never sees a `long[]`
   of rows — mask-native law applied to interpretation.

**The three-tier JIT (the operator's Cranelift framing, grounded):**

| tier | engine | trigger | status |
|---|---|---|---|
| 0 — decode | hexagon recall + exact-table verify | every op | hexagon measured 99.6%; verify tables SHIPPED (ogar-r2il) |
| 1 — interpret | vectorized r2sym (r2sleigh × ndarray masks) | every (program, mask) | r2sym exists row-at-a-time; vectorization is the build |
| 2 — native | `ndarray::hpc::jitson_cranelift` via `lance-graph-contract::jit::JitCompiler` | HOT programs | Cranelift engine SHIPPED behind `jit-native`; trait SHIPPED |

**The profiler already exists and it is the alpha channel.** Tier-up needs
a hotness counter; `AlphaStamp{cycle, seq, visits}` IS one. A program body
whose alpha visit count crosses threshold gets handed to jitson;
`KernelHandle` caches it. The meta-awareness concern and profile-guided
optimization are the SAME mechanism read at two rungs — the open-loop
activation plane (§7.7's hot material) doubles as the JIT profile with
zero new machinery. This is the strongest structural argument yet that
alpha belongs in the contract tier (beside `attention_facet`), consumed by
whatever hosts the interpreter.

**OWL/RDF lowered to IR — the scope fence that keeps it honest:**

The lowering target is **OWL RL** (the forward-chaining rule profile),
NOT full OWL DL. OWL RL rules compile to R2IL naturally: a SubClassOf
axiom = a masked facet check + mask union; a property chain = a masked
hop sequence; materialization = run rules to fixpoint, where the fixpoint
test is FREE in mask algebra (the delta mask is empty). Full DL needs
tableau reasoning — do not promise it, do not build it.

This completes the compiler picture with the tier the stack already has:

- **AOT** = the ontology BAKE (MedCare's closure tables — `is_a` rails,
  ancestor closures). What is known at bake time is compiled ahead.
- **Interpreter** = R2IL over masks, for rules that arrive AFTER the bake
  (dynamic axioms, per-tenant rules) — no re-bake needed.
- **JIT** = jitson for the dynamic rules that alpha proves hot.

So "lower OWL into IR like Cranelift" is not a metaphor: bake/interpret/
tier-up is literally the JVM/V8 model with the alpha plane as the
profiler and the ontology bake as AOT.

**Space binding + write-back (from the same-day co-architecting, so it
doesn't dilute):** R2IL `register` space = the node's own 12-byte facet
register; `ram` = classid-prefixed SoA lanes (the GUID is the pointer);
`unique` = never-persisted scratch. `machine_memory_map` (0xC604) is the
per-class binding table — which rails a program may read/write:
ClassView for behavior. The interpreter's `Store` never writes through;
it emits into the open cycle image (mailbox-owned) — R2IL executes hot,
lands cold, one writer preserved.

**Probes, in dependency order (each independent; 1–2 need no ndarray
wiring):**

- **PROBE-R2IL-DECODE-AUTHORITY** — hexagon-proposes/table-verifies over
  the trained corpus: the 0.4% must be CAUGHT (disagreement detected),
  zero silent mis-executes. The safety case for everything above.
- **PROBE-R2IL-LIVE-REGFILE** — one harvested 6502 routine, N=1,
  facet-register byte-parity vs a reference emulator after N instructions.
  Proves the space binding.
- **PROBE-R2IL-LANES** — same routine, N=64K masked lanes through the
  ndarray sweep. The throughput number that makes "realtime" credible.
- **PROBE-OWL-RL-FIXPOINT** — one OWL RL rule set lowered to R2IL, run to
  fixpoint over a baked population; inferred-mask parity vs the bake's own
  closure for the axioms both know. Gates the AOT/interpreter seam.

**Java/Valhalla arm, settled:** a Java interpreter (even value-class
flattened) ends where lance-graph-java's own law points — compute lands
through Panama into ndarray kernels anyway, so it buys a second
interpreter that drifts. Java STEERS (which mask, which program address,
when): one ABI call `(program_ref, mask_handle) → mask_handle`, bulk and
lifecycle-clean. Rust EXECUTES. One interpreter, two front-ends.

**§7.8 grading note (same day, measurement-skeptic pass):** the 99.6%
hexagon decode figure is **operator-reported; the artifact is NOT located
in this workspace.** Grep over this board + plan finds no such
measurement, and §7.2's own honest gate (2026-08-25) states the learned
hex-topology layer "does not exist" here. So either the measurement lives
in another session/repo (to be linked when named), or it is remembered
imprecisely. Either way the doctrine is UNCHANGED — sentence (2) is
precisely the rule that makes the architecture safe under an unverified
recall number: the exact tables are the authority regardless of what the
proposer's true recall is. Consequence for sequencing:
PROBE-R2IL-DECODE-AUTHORITY's *verify seam* is buildable now against a
pluggable proposer; the hexagon plugs in when its artifact is located.
PROBE-R2IL-LIVE-REGFILE needs no proposer at all and is buildable today.
---

## §8 — The plasticity falsifier ladder (operator brief merged, 2026-08-26): one queue, pre-registered first experiment

> Register: PLAN + PRE-REGISTRATION. Merges the operator's resume brief
> with the four open steps this arc already carried. Board-only unless
> an experiment justifies code; the first experiment (Q1) is run with a
> TEMPORARY instrument over the shipped probe, reverted after — same
> method as every measurement in this arc.

### 8.1 Doctrine lines (restated so no experiment drifts past them)

- **R2IL carries fidelity. BPE carries repetition. Plasticity changes
  association strength or residency — it must NEVER rewrite executable
  truth.** Every learned macro stays byte-exactly expandable to R2IL
  atoms, in every arm of every experiment, or the arm aborts.
- **Coverage is never the headline.** With 7 atoms and chain length 3,
  "≥1 macro fires" saturates (the strict null already reaches 92.8%).
  Density per chain + null separation is the primary transfer metric.
- **"Association demonstrated" ≠ "plasticity demonstrated."** The first
  is measured (batch, 4 binaries, two nulls). The second requires Q1
  below, and until it runs the tissue is a slide, not a culture.
- **Do not quietly optimize around a failure. Measure it.** Negative
  findings land append-only.

### 8.2 The unified queue

| # | item | status/gate |
|---|---|---|
| **Q1** | interference × saturation × order harness (§8.3, ONE instrument, three arms + sabotage) | pre-registered below; **runs in this PR arc** |
| Q2 | 6502 lift `SpaceId` measurement (does a real lifted memory access carry `Custom(1)` or `Ram`?) | cheap; r2sleigh-side; settles the E-W0-IS-LIVE conditional |
| Q3 | OGAR mint decision (three custom-space options) | EXTERNAL gate; W2 stays blocked |
| Q4 | W1 op-row carving, non-space subset | may proceed |
| Q5 | ontology-morphogenesis probe (§7.6) | data half owned by the consumer repo |
| **Q6** | **hex A/B topology experiment** | **GATED on Q1's results.** Role hypothesis (NOT cell=concept, NOT six-directions=six-categories): hex cell = addressable learned residence; six neighbours = local candidate associations; potentiation/depression = strengthen/weaken a local relation; plastic frontier = the region where Explore may alter Learned; locality = a bound on how far one learning event may perturb existing knowledge. Falsifiable question: *can local hexagonal plasticity learn the SAME new associations as the global palette while producing LESS interference, controlled forgetting, and bounded propagation?* Metrics held constant vs the global baseline: held-out density before/after learning B, interference on A, changed-entry count, propagation radius, saturation behavior, order sensitivity, exact R2IL expandability, white-matter veto rate, recovery after demotion. **A hex topology that merely looks brain-like but does not reduce interference FAILS.** |

### 8.3 Q1 PRE-REGISTRATION (written and committed BEFORE the run)

**Corpora.** A = the two gcc `stress_test` binaries (train). H_A =
`vuln_test` (held-out, unseen C — the A-domain probe). B =
`build-script-build` (rustc). Same ore, same seven-opcode convention,
same chain extractor as the shipped probe.

**Incremental protocol.** "Learn B after A" = keep A's SymTable, apply
A's merges to B's raw streams in learned order (standard encode-then-
continue; skipping this would re-mint A-equivalent pairs as duplicate
ids), then continue `bpe_merge` on B's streams under the SHARED cap.

**Arms and pre-registered predictions:**

- **INT (interference).** density(V_{A→B} on H_A) vs baseline 2.515.
  **Structural prediction: it cannot drop** — the store is additive and
  the matcher monotone, so new merges only add hits. A measured drop >1%
  falsifies the additivity model itself (major finding). Confirmation is
  a CONTROL result and must be recorded as *"no destructive interference
  is possible yet because no destructive mechanism exists"* — explicitly
  NOT a plasticity success. Real interference first becomes possible at
  saturation, which is why SAT is the load-bearing arm.
- **SAT (saturation).** Forced cap = 16 (< the 33 A-macros), two naive
  policies: REFUSE (A's first 16 merges keep their slots; B learns
  nothing) and EVICT-AMORTIZED (learn uncapped, keep the top-16 by
  measured training occurrences). **Kill threshold, from the already-
  measured column null on H_A (max 1232/608): a policy FAILS if its
  held-out-A density ≤ 2.03** — i.e. the vocabulary becomes
  indistinguishable from the marginal-preserving null on A.
- **ORD (order dependence).** V_{A→B} vs V_{B→A}; primary metric =
  Jaccard over the macros' atom-pattern sets. Pre-registered reading:
  ≥0.8 order-robust · 0.5–0.8 path-sensitive (consolidation needed) ·
  **<0.5 = "the vocabulary" is a path artifact.**
- **SABOTAGE (harness validity, can-fire + can-stay-silent).** Deleting
  A's top-5 macros (they carried 1,063 of 1,529 H_A hits = 69.5%) must
  drop H_A density below 1.8; deleting the bottom-5 must move it <2%.
  If either half fails, the harness — not the store — is broken, and no
  arm result counts.
- **BYTE-EXACT gate, every arm:** decode(streams) == original atoms for
  every training stream under every table. Any mismatch aborts that arm.

**⊕ RUN 2026-08-26 (append-only results pointer).** Q1 was executed
after this section's commit (`7b00847`); results and per-arm reading
against the thresholds above are on the board:
`E-Q1-THE-ADDITIVE-STORE-CANNOT-INTERFERE-YET-AND-THE-VOCABULARY-IS-ORDER-ROBUST-1`.
One-line verdicts: SABOTAGE valid (0.766 / 2.510) · INT = CONTROL
(2.554, no drop — additivity holds, plasticity NOT demonstrated) ·
ORD robust (Jaccard 0.872) · SAT both policies clear the 2.03 kill
(REFUSE 2.222, EVICT-AMORTIZED 2.365, evict keeps 15/16) · byte-exact
in every arm. The instrument was reverted; this experiment justified
NO source change (board-only outcome, per the brief). **Q6 is now
ungated.**

### 8.4 The learning loop as control law (operator-stated; placement, not prose)

Plasticity is the LAST gate, never the first reflex:

```
experience → attention (Alpha) → reason → hydrate-if-missing (cognitive
Maslow) → counterfactual → commit (Rubicon) → act ← veto window (Libet)
→ observe → revise → qualify evidence (§7.5 instrument) → plasticity
gate → MAYBE learn
```

never `experience → immediately change weights`. Q1's harness tests the
final two stations; everything upstream already ships (#879, #998,
elevation/cycle, kanban census).

**Goal line, verbatim intent:** not to imitate a biological brain — to
build digital associative tissue with properties biology does not
naturally provide: exact provenance, reversible learned macros,
qualified evidence for promotion/demotion, and a falsifier attached to
every claim of plasticity.

---

## 9. Q6 — the hex A/B topology experiment (PRE-REGISTERED 2026-08-26, before any harness existed)

Q1 ungated this: it showed the current store *cannot* interfere,
because it is additive and unbounded — "did not forget" was an
architectural property, not a learning rule. Interference first becomes
possible under **bounded capacity with eviction**. Q6 therefore does not
compare "hex vs no-hex" on the Q1 store; it builds the smallest store in
which forgetting is *possible at all*, and asks whether hexagonal
locality makes that forgetting better-behaved than a global pool.

### 9.1 The question, restated as something that can fail

> At equal capacity and equal newly-learned associations, does confining
> eviction to a hex neighbourhood produce LESS interference on prior
> knowledge than a global pool — and is any advantage due to *hexagonal
> content addressing*, or merely to *partitioned capacity*?

The second clause is the one that kills a pretty result. It is why the
random-partition arm below is not optional.

### 9.2 Corpora (identical ore to Q1, `ore_all.tsv`, 94,536 rows)

- **A** (prior knowledge, train): the two gcc `stress_test*` binaries.
- **H_A** (held-out A probe, interference is measured here): `vuln_test`.
- **B_train / H_B**: `build-script-build` (rustc), split at FUNCTION
  granularity — distinct function names sorted, every 5th to H_B — so no
  chain leaks across the split. H_B is where *learning* is measured.

Learning B after A uses Q1's incremental protocol verbatim (apply A's
merges to B's raw streams in learned order, then continue).

### 9.3 The three arms (same capacity C, same corpora, same order)

1. **GLOBAL — the control.** One pool of C slots. On overflow, evict the
   globally lowest-utility macro (utility = measured training
   occurrences; Q1's EVICT-AMORTIZED, which won there).
2. **HEX.** C slots as 7 cells of ⌊C/7⌋ (a radius-1 hex tile: centre +
   ring of 6). A macro's cell is its **first atom's A-frequency rank**
   (rank 0 = centre; ranks ≥7 wrap to cell `(rank mod 6) + 1` — the ore
   carries 9 opcodes, and this wrap is a documented convention affecting
   only the two rarest, not a tuned parameter). Adjacency: centre is
   adjacent to all six; ring cell *i* is adjacent to {centre, i−1, i+1}.
   On overflow in cell *c*, eviction may take **only** from
   N(c) = {c} ∪ neighbours(c), lowest utility first; if all of N(c) holds
   higher-utility macros, the new macro is **REFUSED**. One learning
   event may therefore perturb at most one hop.
3. **RAND — the topology null.** Identical cell count, per-cell capacity,
   adjacency, eviction and refusal rules; only the *address* changes: the
   cell is a SplitMix64 hash of the macro's atom pattern
   (`0x9E3779B97F4A7C15`, the workspace seed). Capacity partitioning and
   eviction locality are preserved; content locality is destroyed.

### 9.4 Capacities

C ∈ **{14, 21, 28}** — below the 33 macros A learns uncapped (an
already-published Q1 number, so no measurement precedes this
registration), and divisible by 7 for per-cell capacity 2 / 3 / 4.

### 9.5 Metrics, per arm per cap

`d_A_postA` (that arm's own H_A baseline after A alone — this exposes
any handicap the partitioning itself imposes) · `d_A_postB` ·
**interference = d_A_postA − d_A_postB** (within-arm delta, so an arm is
neither credited nor punished for its own baseline) · `d_B_postB` on H_B
(**learning**) · `evicted_A` (changed entries) · `refused` ·
`max_hop` (propagation radius) · byte-exact decode.

### 9.6 Pre-registered gates — read in this order

- **G0 — inertness guard.** A cap counts only if GLOBAL actually evicts
  ≥1 A-macro AND shows interference > 0 there. A cap where the control
  does not forget cannot measure forgetting; such a cap is excluded and
  reported as excluded, not quietly dropped.
- **G1 — equal-learning gate.** HEX passes only if
  `d_B_postB(HEX) ≥ 0.95 × d_B_postB(GLOBAL)`. **Buying less
  interference by learning less is the trivial failure and is an
  automatic FAIL**, no matter how good the interference number looks.
- **G2 — headline.** HEX must show `interference(HEX) <
  interference(GLOBAL)` at ≥2 of the counting caps, with G1 met there.
- **G3 — topology null.** Even with G2 passed, the finding is
  *"partitioning helps; hexagonality is irrelevant"* unless
  `interference(HEX) < interference(RAND)` at ≥2 counting caps with G1
  also met against RAND.
- **G4 — byte-exact.** decode(streams) == original atoms in every arm at
  every cap; any mismatch aborts that arm. R2IL stays the sole truth.

### 9.7 Predicted outcome, stated in advance and against the hypothesis

Structurally, HEX should evict fewer A-macros — cells B never addresses
are unreachable. **The risk is G1**: refusals mean HEX may simply learn
less of B, in which case its low interference is the trivial failure and
Q6 FAILS by the operator's own rule (*"a hex topology that merely looks
brain-like but does not reduce interference FAILS"*). G3 is genuinely
uncertain: if A and B are dominated by different opcodes, content
addressing separates them and HEX earns its result; if they share
dominant opcodes, HEX collides precisely where it hurts and should land
on RAND. A FAIL is a publishable result and is recorded append-only,
exactly like a pass.

**⊕ RUN 2026-08-26 (append-only results pointer).** Q6 was executed
after §9's commit (`6d6f3c6`). **Result: FAIL at every HYPOTHESIS
gate (G1–G3) at every cap** — hex learns less (G1) *and* interferes
2.8–4.2× more (G2), and a random partition with the identical locality
rule beats it (G3). The two VALIDITY gates **passed**, which is what
makes the failure trustworthy rather than an artefact: G0 confirmed all
three caps count (the control genuinely forgets) and G4 byte-exact held
in all nine runs.
Mechanism and the generalized finding (*content addressing under a
heavy-tailed distribution is capacity-destroying*) are on the board:
`E-Q6-HEX-FAILS-CONTENT-ADDRESSING-IS-CAPACITY-DESTROYING-UNDER-A-SKEWED-DISTRIBUTION-1`.
The instrument was reverted; board-only outcome, no retune attempted.

---

## 10. Q7 — frequency-sized cells, and what a 2-byte rail should carry (PRE-REGISTERED 2026-08-26, before any harness existed)

Two arms. **Q7a** runs the disposal Q6's own mechanism analysis named
(cells sized by content mass), so the hex line closes on evidence rather
than on one badly-sized construction. **Q7b** asks the question the
canon actually needs answered: the V3 facet register is `6×(u8:u8)`
(`le-contract.md` §3) — so **what should a 2-byte rail carry?**

### 10.1 Q7a — frequency-sized cells

Q6's HEX held 6/9/12 macros of a nominal 14/21/28 because uniform
`cap/7` cells met a heavy-tailed address distribution: cells 0–2
saturated, 3–6 were never addressed. HEX-FREQ apportions the same total
capacity **proportional to each cell's A-mass** (largest-remainder;
minimum 1 slot for a cell with nonzero mass, **0 for a cell with none** —
a zero-mass cell is never addressed, so a slot there is pure waste).
Everything else — adjacency, neighbourhood-bounded eviction, refusal,
utility — is Q6's, unchanged. Arms: GLOBAL · HEX-FREQ · RAND, caps
{14, 21, 28}, gates G0–G4 exactly as §9.6.

**Pre-registered degeneracy check, and it binds a PASS as hard as a
FAIL.** Report `max_cell_share` = largest cell capacity ÷ cap. If it is
**≥ 0.80**, HEX-FREQ has degenerated into a global pool with a fringe,
and a passing G2 must be read as *"capacity utilization was the whole
story; hexagonal locality contributed nothing"* — it does **not**
resurrect the hex hypothesis. Naming this in advance is what stops a
degenerate pass being sold as a win.

### 10.2 Q7b — three carriers, the same two bytes

Equal budget in **bytes**, not entries — and all three entries happen to
cost exactly 2, which is what makes this a fair fight:

| carrier | the 2 bytes hold | matching rule | what it bets on |
|---|---|---|---|
| **BPE** | a merge pair `(l, r)`, symbol ids < 256 | its atom pattern occurs as a contiguous span in the chain | symbolic identity, variable length, reusable wherever it occurs |
| **PAL** (`palette256:palette256`) | `(code(x,y), code(y,z))`, codes from A's frequency-ranked adjacent-pair codebook capped at 256 | exact equality on the chain | categorical identity of the whole chain — high specificity |
| **I8** (`i8:i8`) | `(clamp₈(y_pos−x_pos), clamp₈(z_pos−y_pos))`, saturating ±127 | exact equality on the chain | metric def-use distance — signed, SIMD-native |

Budgets N ∈ {14, 21, 28} entries (28/42/56 bytes). Train on **A**;
measure on **H_A** (`vuln_test`, unseen C) and **H_B**
(`build-script-build`, rustc — cross-language transfer). Vocabulary =
top-N by training frequency (BPE: the first N merges).

**The comparison metric is HIT-RATE** (fraction of held-out chains with
≥1 match), *not* density: PAL and I8 match a whole chain exactly, so
their density is capped at 1.0 by construction while BPE's is not.
Density is reported but is only comparable within BPE. Comparing the two
across carriers would be an artefact, not a result.

### 10.3 Gates

- **C0 — null separation, mandatory, per carrier per budget.** Rebuild
  the vocabulary from a **per-column shuffled** training set (each of the
  three chain columns — and, for I8, the three position columns —
  permuted independently across chains, preserving every column marginal
  exactly), 20 seeds, SplitMix64 `0x9E3779B97F4A7C15`; measure on the
  unshuffled held-out sets. A carrier carries structure only if its real
  H_A hit-rate lies **strictly outside** `[null_min, null_max]`. Per
  `I-NOISE-FLOOR-JIRAK` the separation is read distribution-free
  (range non-overlap), never as a σ claim.
- **C1 — ranking.** Rank carriers by H_A hit-rate at each budget.
- **C2 — transfer.** `hit(H_B) / hit(H_A)`: ≥0.8 language-portable ·
  0.5–0.8 partial · <0.5 corpus-specific.
- **C3 — no unqualified claim.** No carrier may be called "better"
  at a budget where it failed C0.
- **C4 — byte-exact** for the BPE arm, as always.

### 10.4 Predictions, stated in advance

1. **BPE transfers best** — already-measured precedent (density fell
   only 4.7% gcc→rustc, `E-R2IL-MACRO-VOCABULARY-TRANSFERS-…-1`).
2. **PAL** buys specificity at the cost of coverage: lower hit-rate at
   equal budget, and worse transfer, because a whole-chain signature is
   more corpus-specific than a reusable 2-opcode fragment.
3. **I8 is the wildcard and my named risk**: I predict it beats its null
   on H_A and **degrades most on H_B** — *metric def-use distance is
   compiler idiom; symbolic identity is language-portable.* If I8
   instead transfers well, that falsifies my model of what positional
   deltas encode, and I record it as such.
4. **Q7a**: with the addressed cells now sized to their mass, HEX-FREQ
   should hold near its full capacity — so Q7a is a clean test of
   whether *neighbourhood-bounded eviction itself* helps once the
   capacity waste is removed. I expect it to land on GLOBAL, not beat
   it; if so, the hex line closes.

**⊕ RUN 2026-08-26 (append-only results pointer).** Q7 executed after
§10's commit (`95ea634`). **Q7a:** degeneracy check did not fire
(max_cell_share 0.286/0.333/0.357); frequency sizing lifted utilization
from Q6's 43% to 79–86% and **G1 now PASSES 3/3** — so Q6's
"learns less" half was a construction artefact — but **G2 still fails
(1/3)** and the arm's own resolution limit (n=1, between-arm spread ≤
within-arm spread) forbids claiming HEX-FREQ is *worse*, only that it
misses the bar. **Q7b:** PAL and I8 pass C0 at every budget; PAL
transfers 0.99–1.00 (**falsifying prediction #2**), I8 transfers
0.65–0.67 with a ~10× null margin (**confirming #3**); **BPE failed C0
because hit-rate — the metric I chose to make the carriers comparable —
saturates and inverts**, so C3 binds and BPE is unranked. Full reading,
the post-hoc density sweep, and the follow-up a valid comparison needs:
`E-Q7-FREQUENCY-SIZING-RESCUES-THE-LEARNING-GATE-BUT-NOT-THE-INTERFERENCE-CLAIM-AND-THE-2-BYTE-RAILS-ARE-COMPLEMENTARY-NOT-COMPETING-1`.
Instrument reverted; board-only outcome.

**§7.8 continuation (operator-ruled, same day) — RDF intake IS the V4
program source; the lowering is a PROJECTION, not a compilation:**

The lance-graph-ontology RDF intake arm produces V3 (baked axiom rows).
**Treated as V4, the SAME rows are the execution projection + behavior
learning** — no second pipeline, no emitted artifact:

- The predicate classid SELECTS an R2IL template; the row's own facet
  bytes ARE the operands. `SubClassOf` → masked-check-and-union;
  property chain → masked hop sequence. The program is a ClassView-style
  projection over rows that already exist — the zero-copy law extended
  one rung (*the array is a projection* → *the program is a projection*).
  Re-bake and the "code" updates for free; there is no second artifact
  to drift.
- This is the shape `lance-graph-contract::jit` already has:
  `JitCompiler::compile(&JitTemplate)` — **one template per RULE KIND,
  never per axiom**; instances are rows. The template seam was pre-built.
- Consequence for the earlier open question (content-tier vs ActionDef
  field): it only ever applied to the HARVEST arm. Two program sources,
  one interpreter: **projected** (ontology rules — stored nowhere) and
  **stored** (harvested imperative bodies — content tier, keyed by
  action address). The intake pipeline changes not at all; V4 costs the
  bake nothing.
- **"Behavior learning" defined:** execution of projected rules deposits
  alpha (visits per template × population); hot templates tier up
  through jitson; the decode-disagreement stream trains the learned
  overlay. The substrate learns WHICH OF ITS OWN AXIOMS IT USES and gets
  faster at exactly those. The ontology stays declarative cold truth
  (CE64-fenced per §7.7 — learned wiring never creates/mutates an edge);
  what is learned is the conduction pattern over it (hot). Coupled
  materials and the JIT are one mechanism.

PROBE-OWL-RL-FIXPOINT is unchanged in intent but sharpened in form: it
runs the PROJECTED reading (template × axiom-row operands) against the
bake's own closure — proving the lens, not a compiler.

**§7.8 second continuation (operator-compressed, same day) — the zipper
isomorphism: R2IL is stacked masks over codebooks, and a masked rail
zipper is the same object (pure fragment only):**

The operator's compression: *"R2IL is a bunch of stacked masks of
codebooks; if you stack masked rail as a zipper it might work
similarly."* Named mechanism, not rhyme: both are **prefix-scoped
codebook selection**. An R2IL opcode is a codebook index that scopes how
the following slots decode; a rail byte-pair is a codebook index whose
codebook the prefix selected (longest-prefix binding, OGAR canon). SLEIGH
decode is the same automaton (bit prefix → constructor table → operand
fields). Consequences:

- **Tier-0 decode is ONE mechanism** — `decode(prefix_state, bytes) →
  (op, state)` serves the 6502 stream, the GUID rails, and the R2IL
  stream. The muscle-memory training trains one learned object: the
  prefix-scoped codebook family. Address recall and op recall are the
  same skill.
- **An address IS a program.** One zipper descent step = one
  op-application = the hop law `Mask × ClassView → Mask`. A rail path is
  a maximally compressed straight-line program (the level implies the
  operation; the per-level op vocabulary is the ClassView carving).
  Navigation needs no special case in the interpreter.
- **A baked ancestor closure IS a JIT-compiled navigation program** —
  the composed mask of a hot descent, precomputed. §7.8's "bake = AOT"
  is literal under this reading, with rail paths as the source language.

**THE FENCE (load-bearing):** the isomorphism holds for the PURE fragment
only — selection, load, mask algebra. Rail descent has no `Store`; R2IL
does. Addresses are pure straight-line programs; the effectful fragment
(stores → open cycle image) is R2IL-only. Without this fence, "an address
with side effects" is the SURREAL-AST trap in a new coat — behavior
riding the address, T2 violated.

**PROBE-ZIPPER-HOP-PARITY (new, and the CHEAPEST in the queue — run it
first):** take one real rail path, replay it as explicit R2IL ops through
the masked interpreter, and require the resulting mask to be
BIT-IDENTICAL to the native hop path's mask over the same population.
Green → navigation and execution are one algebra, proven. Red → the
zipper reading is demoted to poetry before anything is built on it.
Needs only existing pieces (masks, ClassView carving, ogar-r2il tables) —
no r2sleigh, no hexagon, no emulator.

**PROBE-ZIPPER-HOP-PARITY: RUN, GREEN (2026-08-26 — OGAR
`crates/ogar-r2il/tests/zipper_hop_parity.rs`, commit 66946e8, PR #286).**
The zipper isomorphism's pure fragment is promoted **CONJECTURE →
FINDING**: a 3-level rail descent over a 197-row `FacetCascade` slab,
answered as (a) one fused u128 prefix compare (the compiled/closure
shape) and (b) an op-at-a-time interpretation of a straight-line
`Load`/`IntEqual`/`BoolAnd` program dispatched through the real `R2ILFn`
table, is **bit-identical** — with a trap population making per-level,
per-byte, in-order matching load-bearing (rows for AND→OR, lo-byte-only
compare, skipped level, out-of-order pairs, half-pair), all five disable
runs red-then-green. The fence is executable: `Store` is refused by the
pure-fragment interpreter; a core byte below the domain floor is refused
by the table. Zero-copy end to end (reinterpret views, in-place register
reads, unique-space scratch, masks both sides) per the operator nudge:
*V3 and V4 are both zero-copy — serialization exists only in the intake
arm; after that it is codebook-index masking algebra.* Scope honesty:
proven for the PURE fragment at depth ≤ 3 over a synthetic slab; the
effectful fragment and real baked populations remain with
LIVE-REGFILE / LANES / OWL-RL-FIXPOINT.

**§7.8 third continuation (operator paradox + census, 2026-08-26) — the
Valhalla/Panama paradox resolves; r2sleigh has NO zero-copy executor yet
but its IR is already zero-copy-shaped; freeze/thaw are the zipper's two
directions:**

Operator: *"it might be faster to use Java Panama Valhalla as a C64
emulator than running it in JITson, because Valhalla knows ABI-shaped
masking and Panama knows zero copy… JITson on the other hand becomes
materialization."*

**Census (r2sleigh fork, measured):** `Varnode {space, offset, size}` is
LITERALLY a MemorySegment slice descriptor — the IR was born zero-copy.
The ONLY executor is `r2sym`, and it materializes everywhere BY DESIGN
(string-keyed `HashMap` register file; `HashMap<u64,u8>` per-byte
memory; `Clone` values boxing Z3 ASTs; `step → Vec<SymState>` whole-state
forks). Correct for symbolic path exploration; structurally the opposite
of realtime execution. **No concrete interpreter exists in the fork.**

**Paradox resolution:** Java would be fast there because Panama (state =
slab addressed by offset/size, never a keyed map) + Valhalla (values
scalarized, never reified) + HotSpot PGO accidentally implement the
three properties this arc already ruled: zero-copy slabs, masked value
flow, alpha-profiled tier-up. Rust provides the first two DELIBERATELY
(`&mut [u8]` per space; `Copy` + monomorphization — Rust has had
Valhalla since 1.0) and the third is the alpha/jitson tier. **Verdict:
no JVM; build the concrete slab executor in the fork** — an `r2conc`-
shaped crate beside r2sym, sharing the IR: `SlabState` (one flat slab
per space: register = the facet register slab, ram = the lane slab,
unique = scratch) + `step(&mut self, op: &R2ILOp)` over the concrete
subset. This is LIVE-REGFILE's prerequisite and its natural first
deliverable.

**"jitson becomes materialization" — affirmed, precisely:** a compiled
kernel is a materialized second projection of behavior, legitimate ONLY
as an ELEVATED cache (profile-derived, `KernelHandle`-held, discardable,
regenerable), never authority. Index-never-authority applies at tier 2
identically: the IR / axiom rows stay truth; the kernel is a cache entry
with a demotion path.

**"Can behavior with R2IL and hexagon become frozen and learned?" —
YES, and the zipper FINDING makes it well-defined:**
- LEARNED (soft) = hexagon decode recall + alpha's conduction pattern,
  grown in production from the disagreement stream.
- FROZEN (crystallized) = a hot behavior compressed into its composed
  form. Three freeze targets, ascending: a composed MASK (result
  frozen, the baked-closure shape) · a jitson KERNEL (execution frozen)
  · a new straight-line R2IL BODY (trace-JIT: the hot path through
  branches frozen branchless).
- **Freeze = program → address; thaw/learn = address → program** — the
  two directions of the zipper isomorphism, whose pure-fragment round
  trip PROBE-ZIPPER-HOP-PARITY just proved exact.
- Gates already shipped: §7.3 crystallization admission + DEMOTION gate,
  §7.4 reversible trust. A frozen artifact that disagrees with the
  interpreted authority under sampled differential execution is demoted
  (re-thawed) — the tier-0 propose/verify structure applied at tier 2.

**r2conc SHIPPED (2026-08-26 — r2sleigh PR #5, branch
claude/c64-6502-falsifier-shztkk restarted from master per merged-PR
rule).** The concrete slab executor the §7.8 census called for:
`SlabState` borrows `register`/`ram` (`&mut [u8]` — machine state lives
with the caller), owns `unique` scratch, registers `Custom(n)` slabs
explicitly (the 6502's case-sensitive `RAM` alias → `UnknownSpace` when
forgotten, never conjured). Transient LE u64 values, nothing reified;
semantics anchored to r2sym for differential runs (carry-out, sign-fill
ashr, byte-offset subpiece with OOR as refusal, trunc-toward-zero sdiv);
direct branch targets = varnode OFFSET (p-code), indirect = value;
everything outside the concrete subset fails LOUD. 12 falsifiers, 6
disable runs red-then-green (logged in the module doc), clippy/fmt/
workspace-check green. PROBE-R2IL-LIVE-REGFILE is now unblocked: lift a
6502 routine → run through SlabState → register-file byte parity vs a
reference emulator.

**PROBE-R2IL-LIVE-REGFILE: RUN, GREEN (2026-08-26 — r2sleigh
`crates/r2conc/tests/live_regfile.rs`, feature `probe-6502`; plan
`probe-r2il-live-regfile-v1.md`).** A hand-assembled 6502 multiply routine
lifted by **Ghidra's own compiled SLEIGH spec** and executed op-at-a-time
through `SlabState` matches an **independently written** reference 6502 on
the full architectural state, across operand pairs whose products exceed
255. 18/18; seven disable runs red-then-green. Three legs keep it from
being an echo: Ghidra's semantics (external), an isolation-enforced oracle
(never saw the R2IL side), and arithmetic itself.

Four things it MEASURED rather than assumed, each pinned:

1. **6502 memory lifts to `SpaceId::Ram`, not `Custom(n)`** — completing
   the §7.8 correction. The case-sensitive alias finding is TRUE of
   `LiftContext`/`ArchSpec` and FALSE of the `Vec<R2ILOp>` stream, which
   dispatches on libsla's `AddressSpaceType`. Both sites corrected.
   *A finding about a data structure is not automatically a finding about
   every path that touches it.*
2. **The facet binding is a PROJECTION, not a byte-identity.** SLEIGH's
   6502 register space spans **55 bytes** (each status flag is its own
   byte register); the *semantic* register file is **7 bytes** and fits a
   12-byte V3 facet register with room to spare. §7.8's binding stands,
   with that word corrected from identity to projection.
3. **Ghidra's 6502 `ADC` is defective**: carry-out ignores the carry-in,
   and `V := C`. Not our bug; the probe drives it deliberately and asserts
   the two sides DISAGREE, which is what proves the harness can fire.
4. **The probe corrected its own prediction.** The plan predicted the
   routine was immune (it `CLC`s before every `ADC`, never *reads* `V`) —
   premises true, conclusion false, because `ADC` still *writes* `V`. On
   `255 × 255` both sides agree on the product `0xFE01` and every other
   field while `V` diverges. **"The buggy path is unreachable" is a claim
   about the whole architectural state, not just the computed result.**

Also forced by building it: `r2conc` now **refuses a `Const`-space branch
target** (a p-code-relative branch within one instruction, not an address)
rather than misreading the displacement as an address — a latent
silent-wrong, now loud. The 6502 emits none, so the gap was invisible
until the probe went looking.

Remaining in the §7.8 queue: PROBE-R2IL-LANES (the masked-population
sweep) and PROBE-OWL-RL-FIXPOINT (the projected-rule lens).


---

## 11. Q8 — PROBE-GREY-TOPOLOGY-AB, the cheap form (PRE-REGISTERED 2026-08-31, before any harness existed)

§7.2's probe is the one hex question Q6/Q7 did NOT answer. They measured
plasticity and interference under capacity; this measures **pattern
completion and false resonance** — different metrics, still unrun. §7.2's own
honest gate names the cheapest real version: *the macro co-occurrence graph
over the existing corpus IS a learned neighbourhood — run
completion/false-resonance on it under both adjacency readings before
building anything.* That is what this runs. No new tissue.

**Grey against white, in parallel** (§7.1 vocabulary): the R2IL exact table is
the white-matter authority; the learned overlay is the grey-matter proposer.
Both arms propose; the table verifies. G4 below is that invariant as a gate.

### 11.1 Corpus — and a stated limitation

`r2il-pass1.ore.tsv` (AdaWorldAPI/ruff release `r2il-harvest-pass1`):
**17,560 rows, 2 binaries** (`stress_test`, `stress_test_opt`), 143 distinct
functions. This is NOT Q6/Q7's `ore_all.tsv` (4 binaries, 94,536 rows) —
it is that corpus's A-side. Consequences, stated in advance:

- the macro count will differ from §7.2's "33"; whatever it is, it is
  reported, not matched to a remembered number;
- no B-corpus and no held-out C, so **no learning/interference claim is
  available here** — that is Q6/Q7's territory and this probe does not
  re-enter it;
- the opcode skew is present (`copy` 6753, `int_add` 4768, `load` 2394,
  `store` 1968), so Q6's capacity mechanism can express itself if it applies.

### 11.2 The three arms, same macros, same task, run in parallel

1. **A — WHITE / Cartesian.** Adjacency = the substrate's native reading:
   nearest neighbours in Morton order over the macro's own address.
2. **B — GREY / hex overlay.** Adjacency = the 6 highest co-occurrence
   neighbours of each macro, the learned neighbourhood.
3. **RAND — the topology null.** 6 neighbours drawn uniformly, same degree,
   same completion rule. Q6's lesson, carried forward without argument: an
   arm that cannot beat a random partition with the identical rule has shown
   nothing.

### 11.3 Task and metrics

Task: **cued completion.** Hold out one atom of a macro, present the rest,
let each arm propose a completion by resonance over its own adjacency.

- **completion** — fraction of held-out atoms correctly proposed (higher better)
- **false resonance** — fraction of proposals that are confident and WRONG
  (lower better); confident = the arm's top proposal, no abstention
- **steps** — mean adjacency hops to reach the proposal (lower better)
- **footprint** — bytes of adjacency actually stored per arm

### 11.4 Pre-registered gates, read in this order

- **G0 (validity/inertness).** The task must be non-trivial: no arm at 100%
  completion, and RAND must score materially below the better of A/B. If G0
  fails the run measures nothing and NO hypothesis verdict is reported.
- **G1 completion:** B ≥ A.
- **G2 false resonance:** B < A.
- **G3 topology null:** B beats RAND on completion AND false resonance.
- **G4 (authority invariant):** every proposal is verified against the exact
  R2IL table; a wrong proposal is REFUSED, never executed. Byte-exactness of
  the R2IL layer holds throughout. This is a validity gate, not a hypothesis.
- **KILL (§7.2 verbatim):** B must beat A on **≥2** of {completion, false
  resonance, steps} **without losing on footprint by more than it gains** —
  otherwise the crystal stays square.

### 11.5 Prediction, stated in advance and against the hypothesis

Q6/Q7 found hexagonal locality bought nothing on plasticity, and the
mechanism they named — content addressing under a heavy-tailed distribution
starves the addressed cells — is a property of the CORPUS, not of the task.
The same skew is present here. **I predict B fails the kill condition**, and
that co-occurrence adjacency will nonetheless beat RAND on completion (the
graph is real information) while NOT beating Cartesian on false resonance.
If B does clear the kill condition, that is a genuine positive and the
`E-Q6/E-Q7` line is narrower than it currently reads.

Instrument is a throwaway example, reverted after the run; the outcome is
board-only, whichever way it goes.

### 11.6 OUTCOME (run 2026-08-31)

**G0 caught my first instrument.** Completion was 0.0000 for every arm at
every cap — the task asked for a macro sharing the cue's left atom and scored
it correct only if the right atom matched too, which would make it the same
macro. Structurally unwinnable; per §11.4 no hypothesis verdict was reported.
Rebuilt as held-out next-macro prediction (115 train chains / 28 held out).

**The gates then passed, against my prediction — and a degree ablation
withdrew the result.** B beat A and RAND on completion and false resonance at
all three caps with a smaller footprint, so under §7.2's kill condition it
survives. But the task uses only each arm's FIRST neighbour, and at
`DEGREE = 1` B scores **identically to four decimals with 5.5× less memory**
(0.2730 / 0.1790 / 0.1498; footprint 164→30, 336→60, 632→120). The
six-neighbourness does nothing; B is a bigram successor table.

**The methodological finding, which outlives this run:** Q6 taught the
topology null and it was included — but RAND varies the WIRING at fixed
degree, so it cannot see that degree itself is inert. **A locality claim
needs a DEGREE ablation, not only a wiring null.** §7.2's probe as specced
lacks one; any "B wins" it produced would have been unattributable.

Full reading, the weak-A caveat, and the convergence with
`E-PALETTE256-IS-A-NEEDLE-THE-COLON-IS-THE-DISTRIBUTION-1` (the information
is in the PAIR, not the neighbourhood's shape):
`E-Q8-THE-SIX-DOES-NO-WORK-A-DEGREE-ABLATION-COLLAPSES-THE-HEX-OVERLAYS-ENTIRE-ADVANTAGE-1`.
Instrument reverted; board-only outcome.
