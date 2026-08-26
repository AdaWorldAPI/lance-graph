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
> case is LATENT on x86 and fires first on the 6502/C64 arc. Verdict: fixed
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
