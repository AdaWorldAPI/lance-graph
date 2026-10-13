# nal-polyfill-v1 — `NAL::narsese()` as a polyfill over the R2IL / mask-risc path

**Status:** PROPOSAL (D-NALP-0..11). Nothing is authorized beyond D-NALP-0,
this document. W1 and W2 are cheap and need no new opcode. Every later wave
is gated on the measurement of the one before it.
**Date:** 2026-10-12.
**Repos:** lance-graph (`lance-graph-nal`, `lance-graph-r2il`,
`lance-graph-mask-risc`, `lance-graph-quack`, contract), OGAR (`ogar-nal`,
`ogar-loco`), ndarray (any missing lane primitive).
**Template:** `.claude/plans/sql-polyfill-r2il-v1.md`. That plan already
settles most of the architecture for SQL. This plan applies the same rules to
Narsese and adds only what NAL needs and SQL does not.
**Parents:** D-HL-10 (inference atoms), D-NAL-AUDIT-0 (NAL ledger), D-RPF-0
(resident CE64 predicates), D-RPF-P5 / D-RPF-CONF-0 (truth surface).
**Plateau:** `.claude/knowledge/ce64-reasoning-substrate-plateau.md`.

## 1. Overview

```text
Rust  narsese!{ … }  ─ compile time ─┐
Rust  NAL::narsese("…") ─ runtime ───┤
Java  nal("…")  (later, one Panama call)
                                     ▼
              lance-graph-nal: parse → bind (ClassView reading + WideFieldMask)
                                     │  names → lanes, once
                                     ▼
              NAL procedures (Rust fns emitting loco calls, like r2il::procs)
                                     │  R2IL fold calls + the few ogar-nal bytes
                                     ▼
              FoldDialect (lance-graph-r2il) ──▶ mask_risc::Program ──▶ ndarray::simd
```

The ergonomics are `NAL::narsese("<bird --> animal>?")` for ad-hoc text and
`narsese!{ … }` for declared queries. Underneath, every NAL operation is
either a procedure over calls that already exist, or one of a very few new
execution bytes that a measurement has shown to be necessary. The "polyfill"
is the same arrangement ndarray uses for SIMD and the SQL plan uses for SQL:
procedure names are the API, R2IL calls implement them, and mask-risc is the
only engine.

## 2. The five components, side by side

The crate names are as they exist in the tree. The engine crate is
`lance-graph-mask-risc`, not "risc-mask".

| component | exists? | role | input → output | owns | never does |
|---|---|---|---|---|---|
| `lance-graph-mask-risc` | yes | **the engine.** The one bulk executor | `Program` (preds, mask ops, ternlog, gather, one terminal) → `Value` / caller out-buffer | `Pred` (`GtI32`…`MatchU64`, `EqU32Via`, `Range`, `MatchFacetStrided`, `MatchFacet16Strided`), `MaskOp` (`And/Or/Xor/AndNot/Not/Ternlog/Gather`), `Terminal` (`Count`, `MaskedSum/Min/MaxI32`, `MaskedStridedGroupSum`, `BlendI32`, `Keep`, `ScatterOrU32`, `CountKeyRunsU32`, `GroupSumI32`, `GroupReduce`, `GroupPowerSumsI32`, `GroupCrossPowerSumsI32`); `fuse` (ternlog fusion) | parse, bind, name anything, or know what NAL is |
| `lance-graph-quack` | yes | **the relational front end.** `Query`/`Filter`/`GroupBy` → `Program` | `Filter` tree (`Cmp`, `Plane`, `EqU32Via`, semijoin, `And/Or/Not`) → `Program` via `lower` / `lower_fused` | SQL three-valued `WHERE` (`sql_where`), group-by lowering, the two Boolean modes | execute (mask-risc does), or interpret a loco body |
| `lance-graph-r2il` | yes (D-SQLP-1) | **the procedure executor.** `FoldDialect` interprets R2IL fold-band calls over borrowed planes | loco calls (`LOAD`, `IntSLess`, `IntAnd`, `KEEP`, `SUM`, `GROUP_SUM`, …) → `Program`s, run at each scalar fold | `Val` (`Scalar` / `Slot` / `Address`; nothing row-sized), `procs` (`select_lt`, `filter_lt`, `sum`, `group_sum`, `margin`), `BoolMode` | parse text, or own an opcode table (that is `ogar-r2il`) |
| `ogar-nal` | **new (OGAR)** | **the NAL vocabulary.** An arity table, in the style of `ogar-r2il`, for the NAL operations that need a byte | none at runtime; a `Vocabulary` impl in the loco domain range | the few NAL bytes that §4 earns, plus the provisional concept id | execute anything, mint concepts, or hold a byte for an operation that a procedure over existing calls can express |
| `lance-graph-nal` | **new (lance-graph)** | **the Narsese front end.** Parse, bind, lower into procedures; the `NAL::narsese()` / `narsese!` surface | Narsese text or a declared form → bound statement plan → procedure calls | the parser, the binder (ClassView reading + WideFieldMask), the NAL procedures, refusals | execute (FoldDialect/mask-risc do), fall back to another engine, or build a `Program` directly (that would be a second spelling) |

Existing precedents for the parts that are new:
- `lance-graph-cypher-quack`: a text front end that lowers from the bound
  plan, never the raw AST, and refuses unsupported shapes by name.
- `ogar-ro`: relation predicates as a loco vocabulary.
- `ogar-loco/src/nars.rs`: the 34 recipes as a vocabulary with
  Datapath / Control / Gate tiers.

## 3. What exists that NAL needs (VERIFIED-IN-CODE, 2026-10-12)

- **Truth functions, scalar.** `causal_edge::isa::truth`: `deduction`,
  `induction`, `abduction`, `revision` (`isa.rs:226-261`). Planner
  `nars::truth::TruthValue`: `deduction`, `induction`, `abduction`,
  `analogy`, `expectation` (`truth.rs:36-107`). These are the per-row
  oracles for every population op below.
- **Field predicates in place.** D-RPF-0: Pearl3, inference nibble and Epi5
  predicates over `MaterializedEdges` via `Pred::MatchFacet16Strided`. A
  threshold `raw >= t` is a union of at most n + 1 ternary patterns. This is
  exactly a NAL truth filter ("c ≥ 0.8") over u8 fields.
- **Joins without materialising.** `Pred::EqU32Via` (fk resolved through a
  foreign lane, one facade pass) and `MaskOp::Gather` (semijoin over a
  resident foreign plane). These are the shared-middle-term pairing that a
  syllogism needs.
- **Grouped moments.** `Terminal::GroupCrossPowerSumsI32` folds
  `n, Σx, Σy, Σx², Σy², Σxy` per group in one pass, exact up to 2³² rows.
  With `x = w` (evidence weight) and `y = f`, it yields `Σw` and `Σw·f`,
  which is all of revision (`f = Σw·f / Σw`, `w_total = Σw`). See §4 B.
- **Value-producing terminal.** `Terminal::BlendI32` writes
  `out[i] = mask[i] ? then[i] : els[i]` into a caller buffer. It is the only
  terminal that produces a lane, and the precedent for §4 C.
- **ClassView / WideFieldMask.** `contract/src/class_view.rs`: `ClassView`
  (`fields`, `field_count`, `project(class, mask)`), `FieldMask(u64)`,
  `WideFieldMask` (`Small(u64)` until a class passes 64 fields, then
  `Wide(Box<[u64]>)`).
- **Byte permutes in ndarray.** `permute_bytes` / `shuffle_bytes` (64- or
  16-entry byte lookups per register) and `gather_u16`. These are candidate
  realizations for a 256-entry `c → w` table; not yet a `ndarray::simd`
  facade word for that purpose.

## 4. Decisions and the three things NAL needs that SQL did not

### A. NAL operations are procedures first (DECISION, carried from D-SQLP)

Each NAL operation is a procedure: a Rust function emitting R2IL fold calls
(the inlined form, `lance-graph-r2il::procs`), and a `PROC_DEF` body once
OGAR's executable `PROC_*` contract exists. A NAL byte in `ogar-nal` is
minted only for an operation that no procedure can express, with an
execution-level insufficiency proof (`isa-anti-lasagne-warden`). Expected
outcome of that rule, to be confirmed in W2–W5:

| NAL operation | lowering | new byte? |
|---|---|---|
| belief query `<S --> P>?`, copula / term match | `MatchFacet16Strided` or `EqU32` over the term lanes | no |
| truth filter `c ≥ t`, `f ≥ t`, Epi5 state | ≤ n+1 ternary patterns (D-RPF-0) | no |
| choice (highest confidence among candidates) | `GroupReduce` / `MaskedMaxI32` on the c lane per statement key | no |
| revision over a population | `GroupCrossPowerSumsI32` with `x = w`, `y = f` (§4 B) | no, if w is stored |
| count / support / evidence census | `Count`, `GroupSumI32` | no |
| deduction / induction / abduction / analogy producing new truth values | join on the middle term, then a per-row truth law into a new lane (§4 C) | **yes, one, if W4 proves it** |
| NAL-3 compounds, NAL-5 higher order, NAL-6 variables | blocked on compound terms (ledger R10) | out of scope |

### B. Revision without a new primitive: store the weight, not only the confidence

Coresearch gap G3 asked for a LUT-weighted strided group-sum because the row
stores `c` and revision needs `w = c/(1−c)`. The hypothesis behind this plan
(HYPOTHESIS, gated by W3) is to avoid the gap on the write path instead:
- each delta of a hypothesis layer (D-HL-7) carries its evidence as a
  fixed-point `i32` weight lane `w` (scale 2¹⁶) and a frequency lane `f`
  (0..255);
- then revision per statement is one `GroupCrossPowerSumsI32` with
  `key = statement`, `x = w`, `y = f`. It yields `Σw` in the `i64` `sum_x`
  field and `Σw·f` in the `i128` `sum_xy` field (`ndarray::simd::CrossPowerSums`);
- `c` is computed once, at commit, from `Σw`. That is the "round once" of
  D-HL-2.

**Overflow bounds.** The bounds are explicit, not assumed:
- **Per row:** `w` is an `I32` lane, so `|w| ≤ 2³¹ − 1`. At scale 2¹⁶ that is
  a weight of about 32,767, far above the 254 a single u8 confidence code can
  express.
- **A delta whose weight exceeds the lane is refused at write time,** never
  saturated. Saturating would silently undercount evidence.
- **`Σw`:** at most 2³² rows × 2³¹ ≤ 2⁶³, so the `i64` field is exact within
  the terminal's stated 2³²-row bound. The executor already refuses a wider
  plane.
- **`Σw·f`:** at most 2³² × 2³¹ × 255 < 2⁷¹, held exactly in the `i128`
  `sum_xy` field.
- **Commit:** reads both in their full width and rounds into the u8 code
  once.

This dissolves G3 for the delta path. G3 remains open for reading
already-baked rows that store only `c`.

### C. The one execution capability NAL needs: a per-row truth law into a lane

A syllogism produces new truth values: for each paired row, `f = f₁·f₂`,
`c = f₁·f₂·c₁·c₂` (deduction), with the other rules similar. That is
value-producing arithmetic, which mask algebra cannot express. It is the same
gap the SQL plan left OPEN for nullable arithmetic (`x + y > 5`). Two
candidate shapes, chosen by W4's measurement, not here:
1. **`MapPairU8` terminal:** in the style of `BlendI32`, writes
   `out[i] = law(a[i], b[i])` for u8 lanes into a caller buffer. A NAL rule is
   a composition of two-input u8 laws (`f = mul(f₁, f₂)`,
   `c = mul(mul(f₁, f₂), mul(c₁, c₂))`), each law a 256×256 u8 table, the
   same law-table shape as the `Palette256²` pair address. Rounding per stage
   is measured against the exact rule.
2. **Fixed-point arithmetic terminal:** `out[i] = (a[i]·b[i]) >> s` in u16/i32
   lanes. Fewer table bytes, no per-stage table rounding, and it needs an
   ndarray multiply-high word.

**MEASURED (2026-10-13, W4 probe):** neither candidate as written. Over all
2³² inputs per law, the table shape misses 10–18 % of rows by one code and
`f32` misses a few thousand; integer-valued `f64` (products of codes, one
division, one rounding) is exact on every row of deduction, analogy,
induction and abduction, and handles the variable division of `w / (w + k)`.
Entry `2026-10-13-d-nalp-4-truth-law-shape.md`.

Whichever wins, it is solved once for both NAL and SQL nullable arithmetic,
never twice. The path is: ndarray primitive with parity on every realization
row → mask-risc terminal with parity → `FoldDialect` wiring → one `ogar-nal`
byte → procedure.

### D. Binding goes through the ClassView, never through hard-coded offsets

- A Narsese statement population is a class. Its reading of the bytes
  (where S, copula, O, F, C, w, Epi5 sit) is declared by that class's
  `ClassView`, as plateau U1 proposes for CE64 bits 40..63. An undeclared
  reading is refused.
- The binder computes one `WideFieldMask` per query: exactly the fields the
  query reads. Only those lanes are bound and touched. This is where the
  cheapness comes from. The same mask is the RBAC projection: a field the
  caller may not read is absent, so the query refuses (fail-closed), never
  reads it silently.
- Names bind to lanes once, at bind time (the cypher-quack rule). Nothing
  after the binder sees a name.

### E. Two execution modes, never confused in claims (DECISION, carried from D-SQLP)

- **Declared** (`narsese!{ … }` or a schema-declared query): procedure
  bodies are generated into Rust at build time. "Compile-time polyfill" is
  true only here, and only once D-SQLP-4's generator exists.
- **Ad hoc** (`NAL::narsese(text)` at runtime): parsed, bound and run
  through the interpreter and mask-risc. The cost is one opcode dispatch per
  16,384-row tile.

### F. One spelling (DECISION)

Once `lance-graph-nal` exists, any other front end that filters or revises
on truth fields (quack, cypher-quack, lance-graph-java) does so through the
NAL procedures. A `Program` built directly for a NAL operation elsewhere is a
second spelling.

### G. The typed surface: `nal::{Operation}::(…)` (operator, 2026-10-12)

Each NAL operation is a zero-sized type in the `nal` module, used as
`nal::Deduction::run(&plan, …)`, `nal::Revision::run(…)`,
`nal::Choice::run(…)`. This is the shape ndarray already uses for SIMD,
where the immediate is a named constant chosen at compile time:
`a.ternlog::<{ ternlog::AND3 }>(b, c)`. Here the "immediate" is the NAL
operation.

```rust
pub trait NalOp {
    /// The loco byte when the op earned one (§4 A), else None: the op is a
    /// procedure over existing R2IL calls.
    const FN: Option<FnIndex>;
    /// Operands popped / pushed, read from the one arity table.
    const ARITY: (u8, u8);
    /// Emits the op's calls through the dialect: one inlined procedure.
    fn lower(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError>;
}
pub struct Deduction;   // impl NalOp: middle-term join + §4 C truth law
pub struct Revision;    // impl NalOp: GroupCrossPowerSumsI32 over w, f (§4 B)
pub struct Choice;      // impl NalOp: GroupReduce Max over c
```

- **Compile time, honestly stated.** A call through a concrete type
  (`nal::Deduction::run`) is monomorphized: no opcode dispatch, no table
  lookup, and the lowering is fixed when the crate compiles. What still runs
  at runtime is mask-risc over the bound lanes, which bottoms out in
  `ndarray::simd` (AVX-512 / AVX2 / NEON / wasm). "Hardware-accelerated
  algebra" is true of that bottom layer and is measured, not assumed (W10).
  Text through `NAL::narsese("…")` takes the interpreted path (§4 E), and
  the two must give identical answers.
- **One table, two spellings forbidden.** The op set lives once. Bytes come
  from `ogar-nal` (`pub const DEDUCTION: FnIndex`, like `ogar_r2il::SUM`); each
  `nal::` type reads its `FN` from there. A drift test compares the `nal::`
  type list with `ogar-nal`'s table (the `R2ILFn::MNEMONICS` pattern). An op
  without a byte has `FN = None` and is a procedure, so a loco program
  reaches it through `PROC_CALL` once that contract executes.
- **Loco palette mapping.** The `nal::` types are the Rust spelling of the
  `ogar-nal` vocabulary. A loco program, a `narsese!{}` block and a direct
  `nal::Deduction::run` call all lower to the same R2IL / mask-risc calls.
  W11's byte-identical `Program` gate holds across all three.
- **Not a second engine.** `NalOp::lower` emits calls; it never loops over
  rows.

### H. Text is a front end; strings are labels, never addresses (operator, 2026-10-12)

> "text" queries should follow the SQL / Gremlin over RISC ABI shaped
> substrate, never text serialization. Parsing <> label codebook. Strings are
> never stored as address, only as a label.

Rules that follow, each enforced in W6:

1. **Narsese text is a front end exactly like SQL, Cypher or Gremlin.** It is
   parsed and bound into the same structured plan the typed `nal::` calls
   build (§4 G). The text stops at the parser. Nothing after the binder sees a
   string: no text plan, no text IR, no text on any wire, no serialized
   Narsese between components.
2. **Parsing and the label codebook are separate steps.** The parser
   recognizes structure (copulas, brackets, truth values, punctuation) and
   produces term *tokens*. The binder resolves each token to an address
   through the label codebook. The parser never assigns an address, and the
   codebook never parses.
3. **Strings are stored only as labels.** A statement row holds addresses
   (classid / `NodeGuid` / term id) and truth codes, never a string.
   - A concept's names live in `ogar-vocab` as labels (D-LBL,
     `concept-labels-hotplug-v1`), pulled through `hotplug.rs` with the
     activation.
   - An app's or instance's names live in that consumer's own ClassView or
     overlay (D-LBL's rule for app-specific and private labels).
4. **Resolution is a lookup from label to address.** D-LBL builds the
   forward index (concept × language → label). Binding needs the reverse
   direction (label × language → address). That reverse index is a W6
   prerequisite, built from the same rows, never a second label store.
   - A label that resolves to more than one address is a typed refusal
     (`AmbiguousLabel`), never a guess.
   - A label that resolves to nothing is a refusal (`UnknownLabel`). Minting
     a new address from a string is not the parser's job.
5. **Output is addresses and codes.** A result is turned back into Narsese
   text only at the display edge, by looking labels up for the caller's
   language priority (`Activation::label(concept, langs)`). An absent label
   renders as the address, never as another language's label.

### OPEN

- **Label source for instances.** Concept labels come from `ogar-vocab`.
  Instance terms (`tweety`, a patient, a document) need a per-consumer label
  codebook. Which existing surface holds it (ClassView overlay, a deepnsm
  lexical lane, the consumer's own table) is decided in W6.
- **Term addressing width.** A CE64 edge carries 8-bit S/P/O bytes; the
  planner's arena uses u16 term ids; the address space is `NodeGuid`. Which
  one a statement row uses decides the join lanes (`EqU32Via` needs u32).
  Decide in W1. It must not be widened in place inside CE64.
- **Expectation as a sort key.** `e = c(f − ½) + ½` is not a single-field
  order. Choice by expectation needs either a stored lane or the §4 C law.
  Choice by confidence alone needs nothing new. Which rule is the default
  follows OpenNARS's choice rule, checked in W7.
- **Narsese parser.** A standalone crate, or reuse of an existing parser
  (`.claude/knowledge/nal-recipe-audit/oracles.md` lists what the pinned
  engines accept). Decided in W6.
- **`PROC_DEF` execution.** The same OPEN item as D-SQLP: until it exists,
  procedures are inlined call sequences.

### Status 2026-10-12 (D-NALP-1/2, `crates/lance-graph-nal`)

- **NAL operations are global core methods** (operator, 2026-10-12: "NARS
  ist global"). They run over any belief population, check no classid, and
  need no hot-plug and no minted concept; they read the one canonical reading
  `nal-statement-v1`. Classid-dependent methods are the other kind: declared at
  compile time through a consumer's `HotPlug` and checked against the row's
  classid (the D-CMH path, not this crate). This removes W8's concept-id mint
  request; W8 keeps only the bytes W4/W5 earn.
- **The NodeGuid is the address and the distinguished name** (operator,
  2026-10-12). Reading `nal-statement-v1`: semantic tiles `[classid hi,
  classid lo, S, copula, P, id, 0, 0]`. The key names the belief and nothing
  else; `f` and `c` are value lanes written in the key's sealed order, so key
  row `i` and truth row `i` are one belief by construction. Truth is never
  part of a name.
- **A selection is the 16-byte key minus an aperture** (operator,
  2026-10-12): `Aperture { key, bytes }` cares the leading `bytes` bytes and
  frees the rest. On the witnessed HHTL order (classid, then each tile coarse
  to fine) the matches are one range, found by the contract's
  `SealedFacetLane::bound_aperture`: 4 bytes = the class, 6 = a subject, 10 =
  a statement. This replaces the §4 A lowering column for the belief query:
  it is a range, not a lane predicate.
- **Term width (OPEN above): decided `u16` per tile.** A vocabulary past
  65,536 terms scales through the classid prefix, not by widening a tile.
- **Bind once, run the kept program.** `nal::{Question, TruthAtLeast, Choice}`
  run their R2IL procedure once at bind and keep the mask-risc `Program`; a
  call runs only `execute_compiled`. The aperture's range is the execution
  extent, so the confidence predicate and the `MAX` fold read only its rows
  (a range passed as an ANDed mask instead cost 195 µs at 1,048,576 rows;
  as the extent, 127 ns).
- **Measured** (`examples/nal_cost.rs`, release, AVX-512, median of 101,
  answers asserted first), n = 1,048,576: range counts 19–20 ns; confidence
  threshold over one statement 81 ns, over a 245-row subject 127 ns; choice
  (max confidence) over the subject 248 ns; bind 274 ns; the row loop
  495–559 µs.
- **OPEN, found here:** R2IL `RANGE` (arity 0) takes its bounds from two `u8`
  immediates, so through the byte it cannot name a range past row 255. The
  procedures emit the bound with `FoldDialect::emit_pred(Pred::Range)`. An
  `ogar-r2il` request, not minted here.
- **OPEN:** which belief holds the maximum (argmax) is not a mask-risc fold;
  `Choice` answers the maximum confidence only.
- Not yet: the ClassView `WideFieldMask` projection and the binder's mask
  (rest of D-NALP-1); W3 onwards.

### Status 2026-10-13 (D-NALP-3, revision)

- **Revision is one fold.** `nal::Revision::bind(bound, statement)` keeps one
  mask-risc program: the statement's range, then
  `GroupCrossPowerSumsI32 { key = evidence class, x = w, y = f }` into a
  four-group sink. The commit law reads `Σw` (`sum_x`) and `Σw·f` (`sum_xy`,
  `i128`) of group 0 and rounds once: `f = round(Σw·f / Σw)`,
  `c = round(255 · Σw / (Σw + k · 2¹⁶))`, capped at ONA's `MAX_CONFIDENCE`
  (0.99, code 252). `k = 1` (ONA `Config.h:168`). **Divergence from ONA,
  kept on purpose:** the cap applies to the result only. ONA caps after every
  pairwise step, so once its running confidence saturates its answer depends
  on arrival order (10 × `(255,230)` then 10 × `(0,230)`: f = 104; reversed:
  151). The fold is order-independent and keeps the exact `Σw` (f = 128);
  pinned by `saturated_evidence_is_the_exact_sum_where_onas_fold_depends_on_order`
  (codex P2 on #1538).
- **The key is the evidential base, read as a group.** `Beliefs::seal_evidence`
  takes `(belief, stamp)` pairs and classes each piece, per statement in sealed
  order: 0 independent; 1 exact duplicate (same stamp set and truth as an
  earlier independent piece, counted once); 2 overlapping (shares a root
  without being a duplicate, or one stamp with two truths); 3 no stamp. Groups
  2 and 3 must be empty or revision refuses (`DependentEvidence`,
  `MissingProvenance`). A stamp naming one root twice is refused at seal (ONA
  drops such events, `Memory.c:501`), which also refuses the stamp of ONA's
  reasoningbench answer #9 (`[10,2,11,2,1]`, c = 0.815686). A population
  sealed with `Beliefs::seal` (W2) has no stamps and never revises.
- **§4 B kept, with one change.** The weight is stored at seal, `w =
  round(2¹⁶ · c / (255 − c))` from the confidence code; code 255 is refused at
  write. The group key is the evidence class, not the statement: the
  statement is already the execution extent.
- **No R2IL byte.** The fold band names no cross-moment call, so
  `lance-graph-r2il` gains `FoldDialect::with_cross_sums` /
  `fold_cross_power_sums`, a dialect method with the standing of `emit_pred`.
- **Oracles.** ONA `Truth_Revision` transcribed from `Truth.c` at `dc4efd0`
  (folded pairwise) and the three OpenNARS points of `revision_points.nal`.
  Gate held: within one code of ONA on 60 random statements (at least 10 with
  three or more pieces, at least one empty) and on the three points; `Σw` and `Σw·f` equal the integer
  row sums exactly. Seven disable runs, each red on its named test: no overlap
  refusal, overlap classed independent, duplicate classed independent, `w`
  scale dropped (4 tests red), no provenance refusal, no repeated-root check,
  no confidence cap.
- **Differs from ONA by policy.** ONA's `Inference_RevisionAndChoice` keeps the
  stronger of two overlapping beliefs (choice); this fold refuses instead.
  Choice on overlap is not built.
- **OPEN.** Revising every statement of a subject or class in one pass needs a
  statement-ordinal key and a sink per statement; not built. The `w` lane is a
  second reading of `c`; deltas that carry their own weight (D-HL-7) would
  remove it. Not measured on the v4 arm yet.

## 5. Waves

| Wave | D-id | Repo | Work | Gate (all must pass) |
|---|---|---|---|---|
| W0 | D-NALP-0 | lance-graph | This plan, board rows | — |
| W1 | D-NALP-1 | lance-graph | Statement layout + ClassView reading: a declared `nal-statement-v1` reading (S, copula, O, f, c, w, Epi5, evidential base) over a probe table; the binder computes the query's `WideFieldMask`; decide the term width (OPEN). Prerequisite inside W1: `ClassView::project` takes a 64-bit `FieldMask` and `field_count` is bounded by `FieldMask::MAX_FIELDS` today (`class_view.rs`), so W1 adds additive wide variants of both (a `WideFieldMask` projection and a field count past 64), defaulting to the existing methods for the Small case; no existing implementation changes behaviour | a query touches exactly its mask's lanes (counted reads); an undeclared reading refuses; a masked-out field refuses, never reads; promotion past 64 fields keeps the same answer (Small vs Wide) |
| W2 | D-NALP-2 | lance-graph | Query procedures with no new byte: `select(copula, subject?, object?)`, `truth_at_least(field, t)`, `choice_by_confidence`, `support_count` in `lance-graph-r2il::procs` style | each equals a row-at-a-time oracle over `isa::truth` / accessor reads, on fixtures with empty and full selections; disable: drop the threshold pattern set to one pattern → red; no new opcode |
| W3 | D-NALP-3 | lance-graph | Revision as `GroupCrossPowerSumsI32` over `w`/`f` deltas (§4 B) | equals the exact integer revision (D-HL-2 arm c) on every statement; agrees with the OpenNARS revision points (`revision_points.nal`); a duplicate-evidence fixture under the D-HL-3 evidential base counts once; disable: drop the `w` scale → red |
| W4 | D-NALP-4 | ndarray, lance-graph | The per-row truth law (§4 C): measure `MapPairU8` (256² tables) vs fixed-point multiply-high on the four NAL-1 rules; ship the winner as ndarray word → mask-risc terminal → `FoldDialect` wiring | insufficiency proof recorded (no existing terminal expresses it); per-row parity with `isa::truth` on all 2³² `(f₁,c₁,f₂,c₂)` codes or a declared stratified sample, error stated in codes; parity on every SIMD realization row; one shape serves SQL nullable arithmetic too |
| W5 | D-NALP-5 | lance-graph | Syllogistic population ops: deduction, induction, abduction, analogy = middle-term join (`EqU32Via`/`Gather`) + W4 law | equals the scalar rule applied pair by pair (oracle enumerates pairs); abduction attaches to the right statement (ledger R1's defect must not reappear: a direction test that is red for the reversed statement); measured cost per derived row vs the scalar loop |
| W6 | D-NALP-6 | lance-graph | `lance-graph-nal` crate: the typed surface `nal::{Operation}` (`NalOp` trait, §4 G) over W2–W5 procedures; Narsese parser (NAL-1/2 statements, truth, questions) producing tokens only, a binder resolving tokens through the reverse label index (§4 H) onto the same types; `NAL::narsese(text)`; typed `Refusal` for every unsupported shape, including `AmbiguousLabel` and `UnknownLabel` | text and typed calls produce byte-identical `Program`s for the same query; no string reaches the plan, a row or any wire (a test scans the bound plan and the written rows for string data); an ambiguous or unknown label refuses; output renders through `Activation::label` and an absent label renders as the address; lowers from the bound plan, never the AST (`cypher-lowering-warden`); every refusal names its reason; no fallback to another engine; no `Program` built outside the procedures |
| W7 | D-NALP-7 | lance-graph | Oracle parity: compile the upstream NAL-1/2 fixtures (`nal1.*`, `nal2.*`) and `revision_points.nal`, run, compare with the pinned OpenNARS output; score with `jc::reliability` | every fixture's expected outputs reproduced within the D-HL-4 tolerance; ICC/Spearman reported per rule; feeds D-HL-4 and D-HL-10 |
| W8 | D-NALP-8 | OGAR | `ogar-nal` vocabulary, built after W4/W5 and before W6: arity table holding only the bytes W4/W5 earned, provisional concept id, drift test against the `nal::` type list (§4 G) | one arity table in the tree for NAL; no byte for an operation a procedure expresses; mint request for the concept id goes to OGAR (this plan does not mint) |
| W9 | D-NALP-9 | lance-graph | `narsese!{}` declared mode: build-time generation from procedure bodies (reuses D-SQLP-4's generator; blocked until it exists) | generated answers identical to interpreted; generated ≤ 1.05× hand-built `Program`; no interpreter dispatch in the hot loop's disassembly |
| W10 | D-NALP-10 | lance-graph | Benchmark, D-SQLP-3 protocol: a revision + one deduction query three ways (direct ndarray, hand-built `Program`, interpreted procedure); N = 65,536 and 1,048,576 | answers identical before timing; release, quiet machine, median of 101, avx512 and avx2 separately; pre-registered: interpreted ≤ 1.05× hand-built |
| W11 | D-NALP-11 | lance-graph, lance-graph-java | One spelling: quack / cypher-quack truth filters route through the procedures; later a Java `nal()` membrane like `sql()` | byte-identical `Program`s for the shapes those front ends already pin; Java: `ApiSurfaceTest` green, no row-proportional allocation |

**Kill conditions.**
- W3: the moment fold disagrees with exact revision by more than one code
  on any statement ⇒ §4 B is wrong; fall back to the G3 primitive.
- W4: neither shape stays within the D-HL-4 tolerance ⇒ no NAL byte ships;
  syllogisms stay scalar (planner `nars::tactics`).
- W10: interpreted > 1.5× hand-built ⇒ stop before W9/W11 and re-plan.
- A `Program` in W11 that is not byte-identical is a lowering bug, never a
  re-pin.

## 6. Order and ownership

```text
W1 → W2 → W3 ┐
        └──► W4 → W5 → W8 → W6 → W7
                              └──► W10 → (W9 once D-SQLP-4 exists) → W11
```

W8 comes before W6 because the `nal::` types read `FN` and `ARITY` from the
`ogar-nal` table (§4 G). W8 can only fix the table once W4 and W5 have decided
which operations earn a byte. An operation without a byte gets `FN = None`,
so W8 never blocks a procedure-only op.

```text
```

- **lance-graph:** W1–W7, W9–W11.
- **ndarray:** W4's primitive, if one is needed.
- **OGAR:** W8, a request; this plan mints nothing.
- **Shared with D-SQLP:** the `PROC_*` contract, the generator (D-SQLP-4),
  and §4 C's arithmetic terminal. Each is built once.

## 7. Wardens that apply

- `isa-anti-lasagne-warden` (W4, W8): a byte needs an execution-level
  insufficiency proof; no second IR.
- `cypher-lowering-warden` (W6): lower from the bound plan, no silent
  fallback.
- `kernel-membrane-warden` (W4): the truth law is T1 epistemic algebra; the
  procedure that composes it is T2.
- `bbb-warden` (W11): what crosses into Java is a name, never a lane or an
  opcode.
- `fold-carrier-scientist` (W3, W10): carrier and crossover claims are
  measured.
- `query-stage-profiler` (W10): timings are per stage and answer-checked.
