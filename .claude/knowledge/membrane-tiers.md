# Membrane Tiers — the 3+ layer abstraction law, and the entropy ledger

> READ BY: kernel-membrane-warden, bbb-warden, simd-savant,
> preflight-drift-auditor, layer-boundary-warden, integration-lead.
> READ BEFORE: designing any public signature, ABI symbol, mask kernel, or
> consumer-facing surface in any language; briefing a worker whose file sits
> on a membrane.
>
> Status: DOCTRINE (operator-directed 2026-09-04). The tiers formalise the
> lance-graph-java "mask-native invariant" and the ndarray polyfill as ONE
> pattern, repo-wide.

## The principle

**A tier may only know the vocabulary of the membrane directly beneath it.
Nothing crosses a membrane except by NAME.** Three membranes minimum; the raw
substrate beneath them is T0.

| Tier | What it is | May know | Crosses UP as | The membrane (gate) |
|---|---|---|---|---|
| **T0 substrate** | bytes, lanes, Lance columns, SoA v3 rows; `simd_{avx512,avx2,neon,scalar}.rs` | strides, offsets, carvings, intrinsics, alignment | — | none; T0 is where truth lives |
| **T1 primitive** | TWO SIBLING ALGEBRAS (2026-09-07, below): **population** — `ndarray::simd` facade, `lgj-abi/kernels.rs`, `mask_*`, `eq_*_to_mask`, `ternlog`, `popcount`; **epistemic** — `TruthU8`, revision, deduction, abduction, … *(RULED, NOT YET RESIDENT — see § "What is ruled vs what is coded" below: these four are coded only in `lance-graph-planner`'s `nars_engine.rs`, and are ABSENT at the lgj-abi T1 membrane)* | `&[u64]`, `&[u8]`+`(offset,stride)`, `IMM`, `TruthU8` | a mask, a count, a lane descriptor, **a truth lane DESCRIPTOR (the `TruthLaneId` shape D-BBB-NARS-3 would mint — prescriptive, 0 code sites today) — never the population** | **polyfill rule** (simd-savant): no intrinsic, no `#[cfg(target_arch)]` above this line |
| **T2 behavior** (was "selection") | ABI exports; `where`/`hop`/`plan_eval`; `Mask × FieldMask → Mask` *(the shipped type; `WideFieldMask` does not cross the ABI — lgj `fixture.rs:1-6` calls wiring it "a later slice", and `class_view_provider.rs:64,235` wires plain `FieldMask`)*; **and the epistemic siblings, named through the same `plan_eval`** | handles, `classid`, `FieldMask` (fields by NAME), version, **a truth LITERAL (`TruthLiteral`) — never a truth population** | a handle, a count, a status | **no hand-composed T1 op, no computed geometry** (kernel-membrane-warden) |
| **T3 intent** | Java facade; R2IL / OGAR `ActionDef`; low-code | names: class, edge, field, version | an outcome | **no byte position** (bbb-warden + ApiSurfaceTest) |
| **R2IL** | emits T3 artifacts | T3's vocabulary (names, outcomes) | an outcome | its ceiling IS T3's; door-knocker test (layer-boundary-warden) |

`where()` is T2 precisely because it is the alias of a glove: Java owns the
NAME `where`, T2 owns the descriptor, T1 owns the mask op, T0 owns the bytes.
The same word lives at every tier and means one thing at each.

## The polyfill is the worked instance

ndarray's `simd.rs` (T1 membrane) → `simd_ops.rs` (staging) → `simd_{arch}.rs`
(T0) IS this pattern at T0/T1. A consumer that reaches into `simd_int_ops` or
hand-writes a compare-and-pack loop has punched T1 — the violation
`simd-savant` exists to catch. lgj-abi stacks the same shape at T1/T2:
`exports.rs` names `kernels::ternlog::AND3`, never `ndarray::simd` directly.

## T1 has TWO sibling algebras — the doctrine is a behavior membrane, not a selection pipeline

**Operator ruling, 2026-09-07.** An audit found NARS truth arithmetic nowhere on
the Java side of the membrane — not exported, not imported, not present — and
concluded from that: *"NARS is off the ladder entirely; the ladder has no tier
for scoring."* **The conclusion was wrong and the diagnosis was the wrong axis.**
The distinction that matters is not *selection vs scoring*. It is **syntax vs
execution** — which is the distinction every other tier here is already built on.

The ladder does not need a sixth tier. **T1 was described too narrowly.** It
contains two sibling primitive algebras:

```
T1
├── population algebra          └── epistemic algebra
│     mask                            TruthU8
│     ternlog                         revision
│     eq → mask                       deduction
│     popcount                        abduction
```

Both are primitive behavior. **T2 may name either. T2 may not hand-compose
either. T3 may express intent in either.** Every rule already written applies
unchanged to the second column — `kernel-membrane-warden`'s HAND-COMPOSED
verdict covers a T2 that spells `revision` out of smaller truth ops exactly as
it covers a T2 that spells `AND3` as two `mask_and`s.

**That claim was FALSE when first written, and is true only because the same
commit made it true.** `kernel-membrane-warden`'s trigger and its method named
masks only; a T2 hand-composing `revision` would have walked straight through
the gate this paragraph invoked. Codex caught it on #1222 (P2), and the card
now carries the epistemic algebra in its trigger and **method step 2b**. Two
sibling gaps of the same shape were closed with it: the T1 row above said a
truth LANE crosses up while the shape table below says a population never does
(now: the DESCRIPTOR, never the population), and `bbb-warden`'s method
classified only signature shapes, so a public helper with a legal
`TruthLiteral` signature computing revision in its BODY passed every step while
doing exactly what `F-BBB-NARS-1` forbids (now: method step 4, an explicit
body-and-import audit). **A doctrine sentence that names a gate must cite the
step that makes it true** — the same defect as the G11 fence being prose until
2026-09-03, found three times in one commit and fixed in the same one.

### The lowering, end to end

```
T3  Java / R2IL / low-code
      │  NARS operation NAME + opaque handles
      ▼
T2  plan_eval — the behavior membrane
      │  resolved bulk operation
      ▼
T1  epistemic primitives (beside the population primitives)
      │  substrate-native execution
      ▼
T0  TruthU8 lanes / rows / history / state
```

**Target route, not an available one.** T3 will say
`Truth.Revision(lhs_handle, rhs_handle)`, and **will not know how revision
works** — that part is the ruling and is permanent. The rest is not yet built:
once the structural gate exists and `plan_eval` carries a Truth opcode (it
carries `{EQ_U32, GT_I32}` today and no Truth opcode; both gate on
D-BBB-NARS-2/-3), T2 will resolve the name, T1 will execute the arithmetic, and
T0 will own every resulting `TruthU8`. The membrane above is shipped; the
operation that would travel it is not. See § "What is ruled vs what is coded".

### Extend the plan language, NOT the ABI surface

The tempting fork — mint `lgj_score_*` beside `lgj_hop` — is rejected. It grows
a second semantic API next to `plan_eval`, and the end state is predictable:
`where()`, `hop()`, `score()`, `nars_revision()`, `nars_deduction()`, … with
Java knowing progressively more about the behavior graph. **The membrane starts
growing little computational fingers.**

`lgj_plan_eval` exists precisely so a whole behavioral expression crosses ONCE.
(That rule is not new here — it restates lgj `docs/abi.md` §6, which already says the
fused-plan call exists "precisely so that `.where(...).where(...).count()` is **one**
crossing regardless of how many predicates or rows." What is new is extending it to
the epistemic column.)
NARS becomes another named plan operation, not another export:

```
Plan                       ← ILLUSTRATIVE. Not a type that exists today.
 ├── Select(…)
 ├── Hop(…)
 ├── Ternlog(…)
 └── Truth(…)
      ├── Revision
      ├── Deduction
      ├── Abduction
      └── …
```

**Read that as the shape `D-BBB-NARS-3` would mint, not as a description of the
code.** Measured 2026-09-10 across both checkouts: `enum Plan` / `Plan::Select` /
`Plan::Hop` / `Plan::Ternlog` / `Plan::Truth` have **zero hits**; `lgj_hop`
(`exports.rs:1703`) is an ABI function, not a tree variant. `lgj_plan_eval` IS
shipped and tested (`exports.rs:1421,1522,1547`; `abi.rs:246-251,337`), but its
`LgjOpDesc` is a **flat array with a combined AND/OR — not a tree** — and its
opcode set is exactly `{LGJ_OP_EQ_U32 = 1, LGJ_OP_GT_I32 = 2}` (`abi.rs:250,252`),
with **no Truth/Revision opcode**. So "a whole behavioral expression crosses ONCE"
is CODED for population predicates and RULED-BUT-UNBUILT for the epistemic side.

### `TruthU8` is the canonical SUBSTRATE representation — not automatically the wire form

These are two different claims and the workspace had been conflating them.
`TruthU8 { frequency: u8, confidence: u8 }`
(`lance-graph-arm-discovery/src/translator.rs:25-33`) is canonical **at T0**.
What crosses is decided separately, and by shape:

| shape | crosses? | as |
|---|---|---|
| a truth LITERAL, `TruthLiteral(192, 217)` | **yes — but never as a bare pair** (⊘ 2026-09-10, § "LE is the universal DTO layer" below): it is meaning the caller supplies, syntax, T3's to state — and its KIND is bound by a versioned DTO schema with canonical little-endian layout, or by an opaque typed handle whose registry binds the same kind and schema | a versioned typed DTO (schema + version + canonical LE byte order), or a typed handle — never `(u8, u8)` on its own |
| a truth POPULATION, `[TruthU8; 65536]` | **never** | `TruthLaneId(u64)` — an opaque 8-byte descriptor |

This is the same rule `bbb-warden` already enforces for masks (*"a `long[]` of row ids is a
materialised population"*, `bbb-warden.md:32`), applied to the epistemic
column — and it lands exactly on the measured Valhalla cliff: **flattening stops
at an 8-byte payload** (VM-confirmed, `valhalla-lab/docs/three-truths.md`), so a
`TruthLaneId(u64)` flattens and a truth array could never. The JVM agrees with
the membrane about where the wall is. **Valhalla carries the noun; Panama
carries the verb; lance-graph owns the reality.**

### LE is the universal DTO layer — "typed syntax" means a versioned LE schema (operator, 2026-09-10)

**Frozen meaning.** *Little-endian is the universal DTO layer of the ABI.* This is
stronger than "LE is convenient serialization": LE is the canonical wire grammar that
guarantees every byte and bit position carries the SAME DTO label in Rust, Panama,
Java, storage, replay, and MUL interpretation. The ABI carries the value. The LE DTO
contract fixes the universal meaning of its positions. MUL may then assert what
epistemic KIND the value expresses. The particular `(frequency, confidence)` values
are content. LE adds no evidence and no confidence; it makes the labels and the
truth-kind interpretation universal. Without a versioned canonical LE DTO contract,
an ABI can transport bits but cannot guarantee every reader assigns them the same
epistemic meaning.

**The gap this closes.** The shape table above first let `TruthLiteral(192, 217)`
cross "as itself" while retiring `TruthU8` as "the wire form." A bare `(f, c)` pair
expresses a DEGREE but not what KIND of truth that degree belongs to. Degree without
kind is not typed syntax. So the ruling is sharpened, not reversed — `TruthU8` stays
the canonical T0 substrate representation, and what crosses is now defined:

> **Typed epistemic syntax crossing G11/Panama SHALL be bound to a versioned DTO
> schema with canonical little-endian field/bit interpretation. The LE contract is
> the universal ABI grammar that fixes the meaning of every wire position. Bare
> `(frequency, confidence)` fields or host-native layouts are not independently
> typed truth. An opaque handle is clean only when its substrate registry binds the
> same truth kind and schema.**

For two `u8` fields the operative contract is the canonical ordered byte sequence
`[frequency, confidence]` — individual bytes have no endianness, the ORDER is the
contract. For a packed multi-byte carrier such as `CausalEdge64`, the complete
integer-to-byte mapping must be explicitly little-endian at every crossing.

> **F-BBB-NARS-2 (LE).** Fail if identical typed wire bytes can acquire different
> DTO labels or epistemic kinds across implementations, host endianness, storage
> and replay; fail if truth kind depends on an unstated reader assumption rather
> than the DTO schema or typed-handle registry.

> **Evidence is not repetition.** Repetition of an identical canonical wire image is
> propagation of the same assertion, not automatically independent evidence. NARS
> revision still requires independent evidential provenance/stamps.

**This extends, and does not restate, the LE contract that already exists.**
`.claude/v3/soa_layout/le-contract.md` §3b (operator-locked 2026-07-02) is
two-level: every tenant carries its own facet LE contract, and the SoA envelope
carries the register-file descriptor (`ColumnDescriptor` offsets/widths,
`verify_layout()`, `ENVELOPE_LAYOUT_VERSION = 2` at `soa_envelope.rs:54`). lgj
already declares byte order as ABI shape: `LgjLaneDesc.endianness: u32 // 0 = little`
(`abi.rs:383`), `LGJ_MAGIC` doubles as an endianness probe (`abi.md:87-93`), and
`abi.md:1224` says it outright — *"Java can discover the ABI's SHAPE instead of
declaring it — sizes, alignments, pointer width, byte order. A wire encoding is
exactly such a shape."* The truth column now inherits that grammar; it did not have
it before.

**Measured 2026-09-10 — what is coded vs what this rules (no code changed):**

| question | answer | evidence |
|---|---|---|
| Any truth DTO with schema + version + canonical LE encode/decode? | **CODED for the envelope, ABSENT for truth.** No truth type rides the envelope contract | `ENVELOPE_LAYOUT_VERSION` exists; zero `to_le_bytes`/`from_le_bytes` in `translator.rs` or `causal-edge/src/edge.rs` |
| Is `TruthU8` a wire DTO? | **No — substrate value only.** Plain `#[derive(Copy)]` struct, no `repr(C)`, no version, no codec; without `repr(C)` Rust does not even guarantee field order | `translator.rs:34-40` |
| `CausalEdge64` byte order at crossings? | **Host-native.** `#[repr(transparent)] (u64)`; bit positions are register-defined (`FREQ_SHIFT=24`, `CONF_SHIFT=32`, `INFER_SHIFT=46`) and endianness-agnostic in-register, but the 8-byte image at any crossing is whatever the host writes — **0** endian conversions in the file. Its v1/v2 layouts are a compile-time feature, invisible in the bytes: exactly what a versioned schema exists to make visible. **⊘ Corrected same day, twice — the KIND is not absent from this carrier, and it is CODED, not ruled.** Operator, second pass: *"all bits are assigned, including 61..63"* — `layout.rs:94` `_LAYOUT_COVERAGE` const-asserts all 64 bits covered exactly once. Bits 59-60 = `CausalTopology` (`Direct` / `IndirectKnownIntermediates` / `IndirectUnknownIntermediates` / `Unknown`, `layout.rs:239-252` — *"indirect intermediate unknowns knowns"*), an additive view ordinal-identical with the older `TrustTexture` reading of the same bits (`bbab3541`, 2026-08-20, via #1154); bits 61-63 = `ReasoningBand` (`Surface` / `Association` / `Relation` / `Causal` / `Counterfactual` / `Perspective` / `Meta` / `Transcendent`, `layout.rs:353-373`; introduced as `TextureBand` in `bbab3541`, named `ReasoningBand` in `9891cca6`) — the **level of ASSERTION, Tarski permission** (`E-RUNG-BAND-AND-PLASTICITY-ARE-THREE-AXES-NEVER-ONE-LEVEL-FIELD-1`; `entropy-closure-causal-ground-v1` §4), `Relation` → `Causal` = relates-to → *causes* (`DISMECH_PREDICATES` `(0x90, "causes", "dismech:causes")`, `dismech_evidence.rs:511`). Writers `with_topology()` / `with_reasoning_band()` (`edge.rs:1009`, `:1057`), readers `topology()` / `reasoning_band()` (`:952`, `:979`), consumed by the W3 verdict (`dismech_counterfactual.rs:251-252`). `SPARE_SHIFT` is the legacy/raw accessor name, not unclaimed design space (`TD-SPARE-SHIFT-NAME-IS-STALE-1`). The first pass of this row read *"bits 61-63 are the reserved SPARE that the operator has now assigned as the NARS × Tarski rung (RULED 2026-09-10, unwritten in code)"* and called `TrustTexture` *"MUL's reading"* — both wrong. The `(f, c)` bytes have no kind *of their own*; the kind rides beside them in the same carrier, and which LENS a producer wrote is declared per class (`ClassView::band_reading`, `band_reading.rs`), never inferred from the bits | `edge.rs:160-176`, `layout.rs:52-77`, `:94`, `:239-252`, `:353-373`; `band_reading.rs` |
| Can `TruthLiteral`'s kind be inferred from its enclosing typed AST? | **No.** 0 code sites; the doctrine had it crossing as an untyped pair | this file, `bbb-warden.md` |
| Can MUL determine the same kind from the same bytes, host-independent? | **No — MUL never sees bytes.** `SituationInput` is typed `f64`s; `revise_fast(f1: u8, _c1: u8, f2: u8, _c2: u8)` takes bare degrees and ignores confidence. Kind is whatever the caller labelled | `mul.rs:12-30`, `nars_engine.rs:459` |

**Truth in this substrate is NARS × Tarski, and the carrier asserts its own kind
(operator, 2026-09-10).** A truth here is not a boolean. tesseract-rs is the worked
example: the *validity of a scanned property* — did the OCR read it right? — is a
statement ABOUT an observation, and the substrate carries that not as `true`/`false`
but as three coded coordinates on one carrier, **all 64 bits assigned**
(`layout.rs:94`, `_LAYOUT_COVERAGE`; operator, second pass: *"all bits are assigned,
including 61..63"*):

- **NARS `(f, c)`, bits 24-39** — the STRENGTH of the assertion.
- **`CausalTopology`, bits 59-60** — the SHAPE of the causal connection: `Direct` /
  `IndirectKnownIntermediates` / `IndirectUnknownIntermediates` / `Unknown`
  (*"indirect intermediate unknowns knowns"*; `layout.rs:239-252`; added in
  `bbab3541`, 2026-08-20, via #1154, ordinal-identical with the older `TrustTexture`
  view of the same bits). Under `entropy-closure-causal-ground-v1` §4b this is *WHAT
  kind of causal-topological hole* the edge is.
- **`ReasoningBand`, bits 61-63** — the LEVEL of assertion, Tarski PERMISSION:
  `Surface` / `Association` / `Relation` / `Causal` / `Counterfactual` / `Perspective`
  / `Meta` / `Transcendent` (`layout.rs:353-373`; introduced as `TextureBand` in
  `bbab3541`, definitively named in `9891cca6`). `Relation` → `Causal` is relates-to →
  **causes** — the authoritative predicate is `(0x90, "causes", "dismech:causes")`
  (`dismech_evidence::DISMECH_PREDICATES`, `:511`); the strongest older references say
  *"explains"* and mean this predicate. Under §4b this is *HOW a candidate may be
  admitted — the epistemic permission level, never a confidence float*.

**Tarski is adjacency, not identity.** Tarski DEPTH — derivational distance from
ground — is a separate quantity stored separately: `Belief.rung` / `Candidate.rung`
(`nars/belief.rs:96`, `nars/tactics.rs:79`, `max(premise rungs) + 1`). The fence
`E-RUNG-BAND-AND-PLASTICITY-ARE-THREE-AXES-NEVER-ONE-LEVEL-FIELD-1` (2026-09-07)
forbids folding the band with any rung: *"any struct, enum or lane that stores two
of the three in one field … is a LAYOUT-BREAK-class defect."* The causal-learning
account (§4b, the wider law): *"Entropy finds the holes. Causal topology gives the
holes shape. The reasoning band controls what kind of bridge may cross them.
Counterfactual + Revision tests whether the bridge actually carries explanatory
weight."* — 59-60 say what causal hole exists, 61-63 say what kind of candidate
assertion may bridge it, counterfactual removal plus revision tests whether it
carries causal weight. tesseract-rs today emits the strength (`sentence_nars_truth`)
and collapses the other two coordinates to a bool (`doc.v1` `low_confidence`) — the
boolean reports the polarity and discards the answer
(`E-THREE-KINDS-OF-MENGENLEHRE-AND-W2-SHIPPED-THE-NARROWEST-1`: *"'explains' and
'relates to' are different answers"*). Both fields have writers and readers
(`with_topology` `edge.rs:1009`, `with_reasoning_band` `:1057`; `topology()` `:952`,
`reasoning_band()` `:979`), and both are what the W3 verdict carries instead of a
bool (`dismech_counterfactual.rs:251-252`).

**One precision closes the loop with the LE ruling.** The bits cannot reveal which
lens a producer used — `TrustTexture` and `CausalTopology` are ordinal-identical on
the wire, and a band-free class reads the same three bits as a stamped one
(`band_reading.rs`: *"which reading a producer wrote is not recoverable from the
bits"*). That declaration is supplied by the schema — `ClassView::band_reading`, per
`(classid, rail)` — plus asserted provenance (`EdgeProvenance`; unstated origin
REFUSES). So the same principle that makes the S/P/O bytes typed — the carrier
asserts its own reference, via palette256 in FisherZ space — makes the truth typed:
**the carrier carries the complete coordinates; LE and the reading contract make
their interpretation universal.** That is what "typed syntax" means here, and it is
its strongest form: nothing about the kind lives in a reader's head. `ReasoningBand`'s
own contract (*"No auto-derivation … nothing derives this field from … NARS
frequency/confidence"*, `layout.rs`) is the in-code form of *LE adds no evidence; MUL
asserts the kind*.

**These dimensions are COORDINATES of truth, not annotations around it (operator,
2026-09-10, third pass).** Three grades a field can hold, and the LE ruling picks the
last:

| grade | what the bits do | MUL at that grade |
|---|---|---|
| **decorative** | can be displayed; nothing depends on them | observes a label |
| **permissive** | govern what the reasoner may accept or assert (the `entropy-closure-causal-ground-v1` §4b gate: *what kind of bridge may cross the hole*) | uses them as an admission gate |
| **defining [LE]** | part of the canonical identity of the assertion — omitting, changing, or reinterpreting them creates a DIFFERENT claim | knows WHICH epistemic claim propagated across storage, ABI and replay — meta-awareness, not a label |

Under the LE ruling the assertion is the product

```text
Assertion = proposition reference   (S, P, O — palette256 / FisherZ)
          × Pearl projection         (CausalMask, bits 40-42)
          × NARS valuation           ((f, c), bits 24-39)
          × causal topology          (CausalTopology, bits 59-60)
          × reasoning/assertion band (ReasoningBand, bits 61-63)
          × provenance               (EdgeProvenance / the class declaration)
```

so these two are NOT equivalent, even with identical S/P/O and identical `(f, c)`:

```text
(S,P,O, f,c, IndirectUnknownIntermediates, Relation)   "a relation is supported, but its mediation is unknown"
(S,P,O, f,c, IndirectKnownIntermediates,   Causal)     "a causal assertion is supported through known mediation"
```

The epistemic valence changed. Once defining, bits 59-60 and 61-63 may no longer be
silently ignored as optional metadata: **a decoder that drops
`IndirectUnknownIntermediates`, or reads `Relation` as `Causal`, has not produced a
lower-resolution view — it has changed what was asserted**, and that is exactly the
`F-BBB-NARS-2 (LE)` failure (*identical typed wire bytes acquiring a different
epistemic kind*). The Tarski adjacency is precisely that `ReasoningBand` controls
the level at which a claim may be asserted — `Relation` → `Causal` — while remaining
distinct from Tarski derivation depth. NARS says how strongly; topology says what
causal structure is known; the band says what assertion is licensed; LE ensures
nobody changes those questions while transporting the answer. A consumer that
carries only a perfume of Tarski (a `low_confidence` bool, a label with nothing
depending on it) has not carried the assertion.

**The smallest #1223 law (operator, 2026-09-10, verbatim — BINDING):**

> A field becomes **defining** when changing or omitting it changes the
> proposition, not merely its presentation. Every defining epistemic dimension
> SHALL participate in the versioned canonical LE DTO; a reader lacking its
> declared lens or provenance must **refuse**, never project a plausible default.

That is the movement: decoration becomes permission; permission becomes semantic
identity. **Coded vs ruled, by clause:** the refusal half is already CODED for the
reading contract — `band_reading.rs` (D-ACR-7, council-ratified): *"a lens mismatch,
an absent band, or untrusted provenance must FAIL, never return a plausible value"*,
`EdgeProvenance::Unknown` refuses, `BandPresence::Absent` refuses (G3′/G4′/G5b) —
and the participation half is RULED, defined by D-BBB-NARS-2 when it lands. By grade:
the W3 verdict carries both fields (`dismech_counterfactual.rs:251-252`) but
`ISS-REASONING-BAND-GATES-NOTHING` (2026-08-26) records that the band gates no
control loop yet, so today the code sits between decorative and permissive; the
§4b guard makes it permissive by design; the LE ruling makes it **defining**. The
versioned truth DTO that D-BBB-NARS-2 defines must therefore carry every defining
dimension — all six coordinates — never `(f, c)` alone, and its reader must refuse
where the lens or provenance is undeclared.

**The aliasing pair — the smallest and strongest falsifier for #1223 (operator,
2026-09-10, verbatim).**

```text
(S,P,O, f,c, IndirectUnknown, Relation)    "S and O are related; mediation is unknown."
(S,P,O, f,c, IndirectKnown,   Causal)      "P causally connects S to O; the mediation is known."
```

*The `(f,c)` values are identical, but the truths are not. `Causal` is not "Relation
with more confidence." It is a different licensed assertion. Likewise, `IndirectKnown`
is not a cosmetic refinement of `IndirectUnknown`. Therefore the complete truth
identity is `(S,P,O) × (f,c) × topology × assertion-band`. LE must preserve all four
components. Flattening either tuple to the same `(S,P,O,f,c)` is **epistemic
aliasing**: the DTO would transport identical confidence while silently changing what
is claimed.* That pair is `F-BBB-NARS-2 (LE)` in its smallest instance — encode, store,
replay, decode: if the two ever become the same thing, the DTO aliases — and it is
`F-CONSUMER-ASSERTION-1` in its smallest instance too: a consumer for which the pair is
one row (`supports = true`) has Tarski perfume, not Tarski semantics. (The four
components are the truth identity; Pearl projection and provenance complete the
six-coordinate assertion above — provenance is what declares the lens the four are
read through.)

**The consumer falsifier — Tarski perfume (operator, 2026-09-10, same pass).** A
consumer has a *perfume of Tarski* when it uses the words — truth, rung, causal — or
attaches `(f, c)`, and the result stays decorative. It becomes real only when the
consumer expresses a **satisfaction relation** — *this typed property about this
entity* →(witness + model)→ `(f, c)` — carried as the complete assertion:

```text
subject      an entity alias (never PII)
predicate    supports_diagnosis            (example shape, not a coded predicate)
object       a disease-ontology concept id
truth        NARS (f, c)
topology     IndirectKnownIntermediates
assertion    Causal
witness      source / provenance handle
```

Then LE makes that complete assertion invariant across storage, replay, Panama and
Java. **F-CONSUMER-ASSERTION-1 (Tarski perfume):** *if topology, assertion band,
proposition identity, or provenance can be removed or changed without altering
admission, interpretation, or replay, the consumer has only Tarski perfume.* Equally
decorative: `Relation` and `Causal` both collapsing to the same `supports = true`; an
unknown mediator becoming a known one without a new witness. The consumer does not
execute NARS or Tarski arithmetic — `D-BBB-NARS-1` forbids it — but it must carry the
typed proposition and preserve the substrate's distinctions; otherwise `(f, c)` is
confidence-flavoured metadata and LE only transports the perfume perfectly. This is
`F-BBB-NARS-2`'s twin at the consumer membrane and the acceptance gate for any
consumer's first typed assertion (the operator's worked case is the private clinical
consumer; nothing of it is quoted here — tesseract-rs's `low_confidence: bool` is the
perfume case measurable in the public tree). Consumer pre-flight: Q6 in
`ogar-consumer-preflight.md`.

**⊘ 2026-09-10, same day — the first cut of this paragraph was wrong on both fields,
and the operator corrected it within the hour.** It read: *"the `TrustTexture` lens at
bits 59-60 (coded) and the Tarski rung at bits 61-63 (ruled today; the field is
`SPARE_SHIFT`, 3 bits, rung 0..7) … What is CODED: the lens. What is RULED: the rung
assignment … Nothing here writes bits 61-63."* Four contradictions with the tree, all
named by the operator: (1) it called 61-63 newly assigned SPARE — they have been the
band since `bbab3541`/`9891cca6`, and `SPARE_SHIFT` is only the legacy/raw accessor
name (`TD-SPARE-SHIFT-NAME-IS-STALE-1`); (2) it equated them with `Belief.rung` — the
exact collapse the three-axes fence forbids, and `ReasoningBand`'s own doc says *"NOT
`RungLevel`, despite four shared variant names"*; (3) it said they were uncoded and
unwritten — `with_reasoning_band()` writes them at three call sites
(`dismech_counterfactual.rs:547`, two probes); (4) it called `TrustTexture` "MUL's
reading" — it is `causal_edge::layout::TrustTexture`, one of four homonyms, not
`contract::mul::TrustTexture` (`band_reading.rs`, `TYPE_DUPLICATION_MAP.md`). It also
omitted `CausalTopology` for 59-60 entirely. Losing text kept; corrected text above.
What stays CONJECTURE, unchanged and beside the point for these bits: that a Tarski
rung is *derivable* from any packed field (`probe_tarski_signed_witness.rs` withdrew
that claim as vacuous). This PR writes no bit of `CausalEdge64`.

**What this does NOT do.** No DTO struct, no opcode, no ABI symbol, no G11 import,
no Java, no conversion. D-BBB-NARS-2 (the syntax/vocabulary contract) is where the
versioned truth DTO schema will be DEFINED, and it stays Queued / *do not pre-build*.
D-BBB-NARS-3's `TruthLaneId` is clean under this ruling only because its registry
will bind kind + schema — that is now part of its gate.



Do **not** import `lance_graph_contract::nars` through the G11 fence merely
because it exists. If that module carries arithmetic semantics together with POD
types, **split out a tiny syntax/vocabulary contract first** and admit only that.
The fence widens by one deliberate module, in one commit, in all three places
its allowlist is spelled (`tests/g11_contract_import_fence.rs`'s `ALLOWED`,
lgj `CLAUDE.md § Enforcement`, `Cargo.toml`'s comment) — the shape lgj already
requires, and the reason its own history records the fence being prose until
2026-09-03 (`ISS-LGJ-G11-FENCE-WAS-PROSE`).

### What is ruled vs what is coded (measured 2026-09-10 by the 5+3 council)

The ruling above is binding. Most of what it rules is **not yet resident**, and this
section exists so no future session mistakes a decision for an accomplished fact.
The doctrine's own test is three sections down: *"A membrane without a gate is prose."*

**1. The epistemic column has NO structural gate — yet.** Each membrane is held by a
structural gate: T0/T1 by the simd-savant grep + the `ndarray::simd` re-export, T1/T2
by the G11 import fence + `kernels.rs` as sole ndarray importer, T2/T3 by
`ApiSurfaceTest`'s forbidden-type list + the array-return naming rule. **The epistemic
column adds none of these.** What it adds — `bbb-warden` step 4, `kernel-membrane-warden`
step 2b — are *review notes*, which property 1 below explicitly distinguishes from gates.
Both steps are real and discriminating (each catches a body that every signature-shaped
step passes, and each has a sanctioned silent case), but a review note is not a fence.
`F-BBB-NARS-1` likewise cannot be exercised today: grep across lgj@`8720d1d` `native/`
and `java/` for `TruthU8`/`revision`/`deduction`/`abduction`/`induction` returns **zero
hits**, so there is no Java surface to run it against. **The gate that will hold this
column is `ApiSurfaceTest`'s forbidden-type list plus a G11 allowlist entry, and it is
gated on D-BBB-NARS-2/-3** — which are Queued and marked *do not pre-build*. Until then
this half of the membrane is enforced by review, and saying otherwise would be the exact
defect this arc keeps finding.

**2. The named epistemic primitives are not at T1.** `revision`/`deduction`/`abduction`/
`induction` are CODED, but only inside `crates/lance-graph-planner/src/cache/nars_engine.rs:194-207`
(`Inference::{Deduction,Induction,Abduction,Revision}`) — a planner-internal dispatch,
**not** a T1 primitive callable from T2 — and are ABSENT at the lgj-abi membrane
entirely. `kernel-membrane-warden` step 2b already states the consequence correctly
("if it does not exist at T1, it lands at T1 first"); the T1 row above now carries the
same hedge, which it did not when first written.

**3. `TruthU8` is the RULED TARGET, and it has three incumbents.** The ruling makes
`TruthU8` canonical at T0. Measured, four truth types coexist today, each self-described
as canonical in some register:

| type | shape | site |
|---|---|---|
| `exploration::NarsTruth` | `f32 × 2` | `lance-graph-contract/src/exploration.rs:89` |
| `holograph::width_16k::schema::NarsTruth` | `u16 × 2`, packed | `holograph/src/width_16k/schema.rs:104` |
| `ndarray::hpc::nars::NarsTruth` | — | aliased `Truth` at `lance-graph-planner/src/cache/triple_model.rs:42` |
| `arm-discovery::TruthU8` | `u8 × 2` | `lance-graph-arm-discovery/src/translator.rs:28` |

**The engine that actually executes revision/deduction/abduction uses the third**, via
that alias. `TruthU8` occurs outside its own crate in exactly one file, a test. No
conversion path bridges them. So "T0 owns every resulting `TruthU8`" is the direction of
travel, not the current state — the convergence is tracked as **D-BBB-NARS-4**.

**4. The LE DTO contract is CODED for the envelope and ABSENT for truth.**
`ENVELOPE_LAYOUT_VERSION = 2` + `verify_layout()` exist and are operator-locked;
no truth type — not `TruthU8`, not `CausalEdge64`, not any `NarsTruth` — carries a
version, a `repr(C)` layout, or an LE codec (see § "LE is the universal DTO layer").
`CausalEdge64`'s byte image is host-native at every crossing today. Ruled 2026-09-10;
defined by D-BBB-NARS-2 when it lands; nothing built here.

### The ruling and its falsifier

> **D-BBB-NARS-1.** NARS truth arithmetic remains substrate-owned. G11/T3 may
> carry only typed NARS **syntax** and **opaque substrate handles**. NARS
> execution is lowered through the existing bulk plan-evaluation membrane; no
> Java-side arithmetic and no materialized truth population crosses Panama.
> `TruthU8` is the canonical substrate representation, while cross-membrane
> results are handles. Any required G11 expansion SHALL expose
> syntax/vocabulary only, never an arithmetic implementation surface.

> **F-BBB-NARS-1.** Fail if Java can implement, inspect, iterate, or reconstruct
> NARS truth arithmetic without invoking the substrate, or if a truth population
> crosses G11/Panama other than as an opaque handle.

The BBB does not move. It stays exactly where it is:

```
                BBB
T3  intent / names        ─────────────
T2  opaque bulk behavior handles
                              ↓
T1  algebra  (population ‖ epistemic)
T0  state
```

No VSA internals. No RoleKey. No NARS arithmetic. No byte positions. No truth
arrays. No Java compute path. **Only names and capabilities.**

## The compile-through rule (the Entropy half)

**Old code is not deleted; it is re-admitted only by compiling THROUGH the
membrane beneath it.** A T3 artifact containing T1 vocabulary (a stride, a
`[u8;12]`, a slot index) is a *cast leak* — rewritten as a call through T2, or
it does not compile. The `.claude/v3/ENTROPY-MILESTONES.md` N→1 ledger records
each old path that now compiles through a membrane instead of around it.

Three properties make this enforceable, not aspirational:

1. **Each membrane has a structural gate, not a review note.** T0/T1: the
   simd-savant grep + `ndarray::simd` re-export. T1/T2: the G11 import fence +
   `kernels.rs` as the sole ndarray importer. T2/T3: `ApiSurfaceTest`'s
   forbidden-type list + the array-return naming rule. A membrane without a
   gate is prose.
2. **Leaks are enumerated, dated, closed downward** (the ledger below).
3. **Named breaches, never unnamed ones.** `materializeRows()` / `importRows()`
   are the precedent: a crossing that must exist is allowed only under a name
   that says so at the call site.

## What the gate CANNOT catch (stated honestly)

Reflection cannot distinguish `int classid` (a T2 name, clean) from
`int facet` (a T1 slot index, a leak) — same type. So the T2/T3 gate catches
the *mechanical* subset (raw `byte[]` registers, unnamed array returns, FFM
types) and `bbb-warden` reviews the *semantic* subset (a raw `int` that is
really a slot). Do not claim the gate proves the membrane; it proves the
catchable half. The warden proves the rest.

## Agent → membrane map

| Membrane | Warden | Model | Verdicts |
|---|---|---|---|
| T0/T1 | `simd-savant` | sonnet | POLYFILL-CLEAN / RAW-INTRINSIC / SHADOW-KERNEL |
| T1/T2 | `kernel-membrane-warden` | opus | NAMED / HAND-COMPOSED / GEOMETRY-LEAK |
| T2/T3 | `bbb-warden` | opus | HANDLE-CLEAN / BYTE-POSITION / UNNAMED-BREACH / ARITHMETIC-SURFACE |
| T3/R2IL | `layer-boundary-warden` | opus | COMPILE-TIME-CLEAN / DOOR-KNOCKER / WRONG-SHELF |

All membrane wardens above T0/T1 are Opus: leak detection is accumulation
(read N files, verdict only holds them together). The pipeline: Sonnet writes
the preflight draft → Opus (preflight-drift-auditor) flips it, checking spec
vs main AND spec vs membrane → Sonnet fleet migrates call-sites (shared
checkout, edit-only) → Opus runs the wardens + the gates once. 5+3 rules only
on DOCTRINE changes (a new membrane, a moved line, a new named-breach class),
never on a call-site migration — that would be the recursion the 2026-08-04
ruling stopped.

---

## ENTROPY LEDGER — T2→T3 leaks (append-only; close downward)

Each row: the leak, the T2 name that replaces it, and the gate that will
reject the old spelling once closed. `[OPEN]` until the gate rejects it.

| # | Leak (T1/T0 vocab in a T3 surface) | Replace with (T2 name) | Gate | Status |
|---|---|---|---|---|
| L1 | `WideFieldMask.ofFacets(int... positions)` — slot indices cross | `classid` + `ClassView`-resolved field NAME; or a named `Reading` (RAILS/SPO) the ClassView selects | bbb-warden (semantic; reflection can't) | **[OPEN]** |
| L2 | 97 served `LgjLaneDesc` lanes — offset+stride cross to Java | field NAME; T2/`ClassView` owns geometry, Java never receives it to be "blind" about | bbb-warden | **[OPEN]** |
| L3 | `RowStore.classidAt / payloadLow64At / payloadHi32At` — per-row byte reads | fenced as inspection-only (javadoc line present); execution must not use them | ApiSurfaceTest note + bbb-warden | PARTIAL (fenced, not removed) |
| L4 | `FacetMatchView.matchesOf(row) -> int` — raw facet bitset | `WideFieldMask.ofMatchBits` bridges it; callers take the typed value | bbb-warden | PARTIAL |
| L5 | `Engine.LaneWindow.setU64` — raw word write | `importRows` (named breach) is the only sanctioned writer | ApiSurfaceTest (internal.ffm already fenced from public) | CLOSED |
| L6 | any future `byte[]` / `[u8;12]` rail array in a public signature | a named `Reading` value type OGAR emits per ClassView (Valhalla), read zero-copy | ApiSurfaceTest byte[]-fence (this PR) | CLOSED (forward guard) |
| L7 | any future array return not named `materialize*`/`import*` | a named terminal | ApiSurfaceTest array-return naming rule (this PR) | CLOSED (forward guard) |
| L8 | any future truth POPULATION in a public signature — `TruthU8[]`, a truth lane, a collection of them — or any T3 body that computes a truth FROM truths — **or any truth crossing as a bare `(u8, u8)` / host-order packed image with no versioned LE DTO schema binding its kind** (⊕ 2026-09-10) | the `TruthLaneId(u64)` opaque descriptor for the population; a named `Truth(…)` `plan_eval` operation for the arithmetic | **OPEN — review-note only** (`bbb-warden` step 4 + ARITHMETIC-SURFACE). The structural gate (ApiSurfaceTest forbidden-type entry + G11 allowlist) is gated on D-BBB-NARS-2/-3 | OPEN (forward guard, ungated) |

Provenance: the two fixes that produced this doctrine — the 7.5→1.1 ms
`lgj_hop` (T1 doing T0's job badly: gathered a contiguous lane; fixed inside
T1, T2/T3 unchanged) and the two-AND→ternlog conjunction (T2 hand-composing a
T1 op; fixed by naming the op at T1) — are the T0/T1 and T1/T2 membranes
working. Both are recorded in ndarray `.claude/blackboard.md` 2026-09-04 and
lance-graph-java `LATEST_STATE.md` 2026-09-04.
