# SPEC v1 — D-ACR-7: the CE64 59..63 reading contract

> **Council:** 5+3, convened 2026-08-21. Orchestrator = main thread.
> **Qualified because:** `5plus3-council.md` §When to convene — *"a spec whose
> wrong resolution silently corrupts downstream sessions (canon entries,
> LE-layout adjacent decisions, classid/mask semantics)"*. §3b fence 3 states
> the failure verbatim: *"A consumer that reads one while a producer wrote the
> other gets a plausible wrong answer, silently."*
> **Deliverable:** `alpha-channel-rung-overlay-v1.md` §4 `D-ACR-7`.

## §1 FROZEN DECISIONS (cite-or-VIOLATES; never re-opened on taste)

| # | Frozen | Source |
|---|---|---|
| F1 | Bits 59-60 = `TRUTH_SHIFT` (2 bits), bits 61-63 = `SPARE_SHIFT` (3 bits). All 64 bits covered exactly once (`_LAYOUT_COVERAGE` const-assert). | `causal-edge/src/layout.rs:65,77,93-111` |
| F2 | **Nothing derives the band.** `ReasoningBand` is set ONLY by an explicit `with_reasoning_band()` — never from `CausalMask`, `InferenceType`, NARS, MUL, `ReasoningGap`, potholes or `ThinkingStyle`. | plan §3b fence 1; `layout.rs:70-72` |
| F3 | `ReasoningBand` is **never** `RungLevel`. Unrelated enums sharing variant names at different ordinals. | plan §3b fence 2; `contract/cognitive_shader.rs:157` |
| F4 | `TrustTexture` and `CausalTopology` are **the same 2 bits read differently**; the reading must be named per `(classid, rail)`, never assumed. | plan §3b fence 3; `layout.rs:58-63` |
| F5 | The band **grades**; the **witness reference discriminates** evidence-kind. No new bit is minted for the episodic-vs-epistemic axis. | plan §3b |
| F6 | Acceptance: any tactic-sampling pass filters on **`delta_conf` capability (14/34)**, never on `maturity().is_production()` (31/34). | plan §4 D-ACR-7 row; §3g; `contract/recipe_kernels.rs:319-322` |
| F7 | **No new CE64 bit, no `ENVELOPE_LAYOUT_VERSION` bump, no new address type.** | plan §5 Non-goals |
| F8 | `I-LEGACY-API-FEATURE-GATED`: a v1 path under the v2 feature must route to the canonical mapping OR be a documented feature-gated no-op. Field-isolation matrix mandatory when layout bits are touched. | `CLAUDE.md` iron rule |
| F9 | A guard needs BOTH a can-it-fire and a can-it-STAY-SILENT test on non-trivial input. | `CLAUDE.md` falsifiability rule |

## §2 INPUT INVENTORY (measured this session; exact file:line)

### 2.1 The bits and their accessors

| item | file:line | shape |
|---|---|---|
| `TRUTH_SHIFT = 59` | `causal-edge/src/layout.rs:65` | 2 bits, `TRUTH_MASK = BITS2_MASK << 59` (`:87`) |
| `SPARE_SHIFT = 61` | `causal-edge/src/layout.rs:77` | 3 bits, `SPARE_MASK = BITS3_MASK << 61` (`:88`) |
| `TrustTexture` (CE64) | `causal-edge/src/layout.rs:141` | **4** variants: `Crystalline=0 / Solid=1 / Fuzzy=2 / Murky=3` |
| `CausalTopology` | `causal-edge/src/layout.rs:239` | **4** variants |
| `ReasoningBand` | `causal-edge/src/layout.rs:353` | **8** variants (fills 3 bits exactly) |
| `trust_texture()` | `edge.rs:927-928` | reads `(self.0 >> TRUTH_SHIFT) & BITS2_MASK` |
| `truth()` raw ordinal | `edge.rs:935-936` | same bits, un-projected |
| `topology()` | `edge.rs:951-953` (v2) / `edge.rs:1144` (v1 arm) | **same bits**, other projection |
| `reasoning_band()` | `edge.rs:978` (v2) / `edge.rs:1154` (v1 arm) | bits 61-63 |
| `with_topology()` | `edge.rs:1008` (v2) / `edge.rs:1166` (v1 no-op) | |
| `with_reasoning_band()` | `edge.rs:1056` (v2) / `edge.rs:1178` (v1 no-op) | the ONLY writer (F2) |

### 2.2 The v1 provenance trap — measured, and NOT covered by the plan's §3b

`layout.rs:74-76` states it directly:

> *"**v1 provenance:** bits 61-63 were temporal bits 9-11, so a v1 edge with
> `temporal >= 512` reads a NON-ZERO band. Apply a version gate on edges of
> unknown provenance — the same rule `truth()` states for bits 59-60."*

So a reading contract that only answers *"which lens"* is **incomplete**: on an
edge of unknown provenance the bits may be **stale v1 temporal payload**, and
every lens then returns a confidently wrong value. §3b names the lens ambiguity
and is silent on this one.

### 2.3 Consumers of the BITS: **zero**

Grep across the workspace, both sibling repos, excluding `causal-edge` itself:
no call site of `reasoning_band()` / `topology()` / `with_topology()` /
`with_reasoning_band()` exists. The contract is being written **before its
first consumer** — prescriptive, not retrofitted.

`TrustTexture` as a **type** does have consumers (`contract/sensorium.rs:132`,
`contract/mul.rs:320-327`, `contract/benches/i4_batch.rs:71-88`) — **but none
of them reads it out of CE64 bits.** Type-consumed ≠ bit-consumed; the spec
must not conflate the two.

### 2.4 ⚠ `TrustTexture` is a FOUR-way homonym with THREE arities

`docs/TYPE_DUPLICATION_MAP.md:9` records *"TrustTexture (×2)"*. **Measured: ×4.**

| # | file:line | variants | arity |
|---|---|---|---|
| 1 | `causal-edge/src/layout.rs:141` | `Crystalline / Solid / Fuzzy / Murky` | **4** |
| 2 | `lance-graph-contract/src/mul.rs:82` | `Calibrated / Overconfident / Uncertain / Underconfident` | **4** |
| 3 | `lance-graph-planner/src/mul/trust.rs:30` | `Crystalline / Solid / Fuzzy / Murky / **Dissonant**` | **5** |
| 4 | `lance-graph/src/graph/arigraph/orchestrator.rs:114` | `Crystalline / **Fibrous** / Fuzzy` | **3** |

Two consequences the spec must carry:

- **#3 cannot fit the bits at all.** 2 bits hold 4 values; the planner enum has
  **5**. `Dissonant` is unrepresentable in CE64 — a producer holding a planner
  `TrustTexture` and writing bits 59-60 either truncates or is undefined.
- **#1 and #2 share a name AND an arity but have disjoint meanings.** A
  consumer resolving "TrustTexture" to the MUL one gets a plausible wrong
  answer with no type error — F4's failure mode, one layer deeper than F4
  describes it.

### 2.5 The `delta_conf` acceptance surface (F6)

`contract/recipe_kernels.rs:319-322` (the doc on the capability method):
*"no kernel declares `ThoughtField::Confidence` in `writes`, and only **14**
can move `delta_conf` — while 31 are `Operational`."*
`contract/recipe_dispatch.rs:188` — `dispatch_order() -> [u8; 34]`.

## §3 THE PROPOSED RESOLUTION (fully committed)

**One new zero-dep contract module: `lance_graph_contract::band_reading`.**
It mints no bit, no tenant, no layout version (F7). It is a *declaration table
plus a resolver*, not a new carrier.

### 3.1 The three things a reading must declare

A `(classid, rail)` pair resolves to a `BandReading`:

```rust
pub struct BandReading {
    /// WHICH projection of bits 59-60 this class wrote.
    pub truth_lens: TruthLens,        // Trust | Topology
    /// Whether bits 61-63 carry a ReasoningBand at all.
    pub band: BandPresence,           // Absent | Present
    /// WHICH evidence-kind carrier discriminates (F5) — a reference, not a bit.
    pub witness: WitnessKind,         // None | Table | CausalFacet | EpisodicBasin
}
```

- `TruthLens` resolves F4: the reading is **named**, never assumed.
- `BandPresence::Absent` is the zero-fallback default — an unstamped class
  declares no band, and reading one is a refusal (§3.3), not a `Surface(0)`.
- `WitnessKind` carries F5's evidence-kind axis as a *reference discriminator*;
  no bit is minted.

### 3.2 Provenance is part of the contract, not an afterthought (§2.2)

```rust
pub enum EdgeProvenance { V2Stamped, V1Legacy, Unknown }
```

The resolver's entry point takes provenance and **refuses** on
`V1Legacy | Unknown` for bits 61-63, because a v1 `temporal >= 512` aliases a
non-zero band. This is F8 applied to a *read* path: the v1 arm is a documented
refusal, not a plausible value.

### 3.3 Reads return `Result`, never a plausible fallback

```rust
pub enum BandReadError {
    LensMismatch { declared: TruthLens, requested: TruthLens },
    BandAbsent,
    UnknownProvenance,
    UndeclaredClass(u32),
}
```

A producer/consumer disagreement is an **error**, satisfying D-ACR-7's own
falsifier (*"must FAIL, not return a plausible value"*). No `unwrap_or(Surface)`
anywhere.

### 3.4 The `TrustTexture` homonym (§2.4)

**In scope:** the contract's `TruthLens::Trust` documents that it means
`causal_edge::layout::TrustTexture` (4 variants) and **explicitly not**
`contract::mul::TrustTexture`, `planner::mul::trust::TrustTexture` (5 —
unrepresentable) or the arigraph one (3). A `debug_assert`-style const check
pins the arity at 4.

**Out of scope:** renaming any of the four (that is a cross-crate refactor with
its own blast radius). The spec records the collision and updates
`TYPE_DUPLICATION_MAP.md`'s stale "×2" to "×4" — a doc correction, not a
rename.

### 3.5 The `delta_conf` acceptance condition (F6)

A `SamplingPolicy` helper the overlay must route through:
`fn admits(kernel) -> bool` returns `delta_conf`-capability, and the module doc
states why `maturity().is_production()` is the wrong filter (a watcher that
cannot dissent). Enforced by a test, not a comment.

## §4 NON-GOALS (each with its why)

| Out of scope | Why |
|---|---|
| Renaming any `TrustTexture` | cross-crate refactor, own blast radius; §3.4 records + documents instead |
| Writing any band anywhere | F2 — only `with_reasoning_band()` writes; this deliverable is the READING contract |
| A registry populated with real classes | D-ACR-2's rail mint is a separate, operator-gated decision the plan does not pre-empt |
| Touching `RungLevel` | F3 — unrelated enum; naming it here would invite the conflation the fence forbids |
| `WideFieldMask` composition | §6 Y2's parked basis collision (D-ACR-0 measured the cardinality mismatch) |
| Any `attention_mask*` reuse | D-ACR-0: rename register file, different contract, EXISTS-UNCALLED |

## §5 PRE-REGISTERED GATES (decided BEFORE any agent runs)

| # | Gate | Pass |
|---|---|---|
| G1 | `cargo test -p lance-graph-contract` | ≥ 1194 (current) + new, 0 failed |
| G2 | `cargo clippy -p lance-graph-contract --all-targets` | 0 warnings from this module |
| G3 | Lens-mismatch test | a `Topology`-declared class read as `Trust` returns `Err(LensMismatch)` — **not** a value |
| G4 | Provenance refusal | `V1Legacy` + `Unknown` both `Err(UnknownProvenance)` on bits 61-63 |
| G5 | Can-fire AND can-stay-silent (F9) | a declared class resolves; an undeclared one errors; both on non-trivial input |
| G6 | `delta_conf` filter (F6) | a mute kernel (one of the 20) is REJECTED and a capable one (of the 14) is ADMITTED — both asserted |
| G7 | Arity pin | a const/test asserts the CE64 `TrustTexture` arity is 4, so a 5-variant sibling cannot be silently substituted |
| G8 | No bit written | the module contains no `with_*` call and no `<<`/`&` against CE64 masks (it declares, it does not stamp) |

## §6 PER-SAVANT QUESTION SETS

Answer each `CONFIRMS / VIOLATES / GAP / PRIOR-ART-AT / RISK` + `file:line` +
≤2 sentences. ≤10 findings total. **Do not redesign** — a redesign urge is one
`RISK` finding, then stop.

### Savant 1 — prior art
1. Does a `(classid, rail) → reading` declaration table already exist anywhere (contract, planner, v3 docs)?
2. Is `BandReading`/`TruthLens`/`WitnessKind` (or a synonym) already a shipped type?
3. Does an E-id already record the 59..63 lens ambiguity, or the `TrustTexture` ×4 collision?
4. Is there prior art for a Result-returning reading resolver in this crate?
5. Does `ClassView` already carry a per-class lens selector this should extend instead?

### Savant 2 — iron rules
1. Does §3 VIOLATE `I-LEGACY-API-FEATURE-GATED` anywhere (esp. the v1 arm treatment)?
2. Does anything in §3 derive a band, violating F2?
3. Does §3 mint a bit / tenant / layout version, violating F7?
4. Does the module stay zero-dep (contract crate constraint)?
5. Do G3–G7 satisfy the falsifiability rule's fire/silent pair on NON-trivial input?
6. Does §3.4's arity pin conflict with any AP1–AP9 anti-pattern?

### Savant 3 — code truth (CODED / CLAIMED / ABSENT per claim)
1. Are all §2.1 file:line references real and correctly quoted?
2. Is §2.2's v1-provenance quote verbatim from `layout.rs`?
3. Is §2.3's "zero bit-consumers" claim reproducible by grep?
4. Is §2.4's ×4 / arity 3-4-5 table correct, and is `TYPE_DUPLICATION_MAP.md:9` really "×2"?
5. Is §2.5's 14/34 and 31/34 quote verbatim from `recipe_kernels.rs`?
6. Do `with_topology`/`with_reasoning_band` really have v1 no-op arms at the lines cited?

### Savant 4 — cascade impact
1. Every file/test/doc/board row that MUST change if §3 lands (mandatory vs follow-up)?
2. Does adding a contract module force any downstream crate change?
3. Which board files does hygiene require in the same commit?
4. Does `TYPE_DUPLICATION_MAP.md`'s correction pull in other stale rows?
5. Does this create work for D-ACR-2 (rail mint) that the plan says it must not pre-empt?

### Savant 5 — different views
1. What is the strongest alternative reading of "the reading contract" that §3 forecloses?
2. Is `Result`-on-mismatch the right severity, or does some call site need a total function?
3. Second-order: what does declaring `BandPresence::Absent` as default cost a future consumer?
4. Is there a reading of F5 under which `WitnessKind` belongs on the witness side, not here?
5. Does the provenance enum belong in `causal-edge` (where the bits live) rather than `contract`?

---
---

# DRAFT v2 — change ledger over SPEC v1

> Phase 2 consolidation, orchestrator only. 5 savants returned 39 findings.
> Raw output banked in the task transcripts; never forwarded to the reviewers.
> **Reviewers see this document ONLY.**

## L0 — SCOPE CORRECTION (operator input mid-Phase-2, NOT a savant finding)

**SPEC v1 addressed `CausalEdge64` only. That was a Phase-0 gap, and it is the
largest change in this ledger.** Operator, 2026-08-21: *"causaledge64 is the
muscle memory / causaledgev3 for granularity."* Verified against source:

| | file:line | shape |
|---|---|---|
| `CausalEdgeV3` | `causal-edge/src/edge_v3.rs:96-103` | `payload: [u8; 12]`, `const _: () = assert!(size_of == 12)`; *"`classid(4) \| payload(12)` = the canonical 16-byte facet, the payload half"* |
| the SAME two fields | `edge_v3.rs:49-50` | `[8] w_slot(6 low) \| truth/topology RAW(2 high)` · `[9] spare/ReasoningBand RAW(3 low) \| reserved(5 high)` |
| their accessors | `edge_v3.rs:199` `truth_raw()`, `:206` `spare_raw()` | RAW ordinals, un-projected |

**The V3 module doc already states D-ACR-7's problem, and leaves it open**
(`edge_v3.rs:86-90`):

> *"`w_slot` / truth / spare are preserved as **RAW ORDINALS**. Copying a CE64
> topology/truth ordinal `01` into V3 means "ordinal 01 preserved" — it is NOT
> an assertion that `IndirectKnown` (or `Solid`) is now source-authoritative for
> that row. **Which lens the ordinal was written through is the producer's
> knowledge, not the conversion's**."*

That sentence IS the deliverable's justification, written in shipped code by a
prior session. **The reading contract is what supplies the producer knowledge
the conversion cannot carry.** Consequence:

- **§3 now spans BOTH carriers.** `BandReading` is keyed by `(classid, rail)`
  and is carrier-agnostic; the projection function takes the raw ordinals, so
  it serves `CausalEdge64::{truth, reasoning_band}` and
  `CausalEdgeV3::{truth_raw, spare_raw}` identically. One contract, two
  carriers — never two contracts.
- **The v1-provenance trap (§2.2) does NOT apply to V3.** V3 has no v1 history;
  its bytes were never temporal. `EdgeProvenance` gains a `V3Register` arm that
  is always readable. This is a real asymmetry the spec must state, not smooth.
- **The muscle-memory/granularity split is now recorded**: CE64 reasons
  (`syllogize` reads SPO + freq/conf + causal_mask); V3 **rehydrates into CE64**
  to reason (`edge_v3.rs:16-26`). The reading contract sits above both and
  privileges neither.

**DEFERRED, with its tension stated rather than silently adopted.** The
operator also raised *"causaledgev3 can even use 6×2×8bit ↑n as BNN planning
equivalent."* This is NOT taken into v2, because the shipped module doc
explicitly forbids that reading today (`edge_v3.rs:29-36`):

> *"a packed EDGE REGISTER, **NOT** a slot-pure §3 facet … Do not read this as a
> content-blind facet: it is a typed edge register whose carving is its own
> contract."*

So "V3 as a `6×2×8bit` content-blind facet" contradicts a shipped doc-comment
**and** would be a sixth homonym against `attention_facet`'s `6×2×8bit` reading
(D-ACR-1, landed today). It is recorded as an open question for the operator —
a stacked (`↑n`) BNN-planning ladder over repeated 12-byte registers is a
plausible and interesting direction, but it needs its own deliverable and its
own resolution of the "typed register vs content-blind facet" contradiction.
**Not decided here.**

## L1 — the Result-vs-total fork (Savant 1 #8 + Savant 5 Q2, independently)

Two savants converged on the same defect from different lenses, which is the
strongest signal this council produced.

- Savant 1: *"`Result`-returning refusal is the **opposite** convention from
  every existing `ClassView` lens selector (`rail_carving`,
  `edge_codec_flavor`, `value_schema`) — all infallible with a documented
  zero-fallback default"* (`class_view.rs:1109-1111, 1127-1133`).
- Savant 5: a hot-path SIMD/batch scan needs a **total** function; per-row
  `Result` unwrapping is exactly what the infallible accessors exist to avoid.

**RESOLUTION — the spec conflated two different operations. They split:**

| operation | shape | why |
|---|---|---|
| **declaration lookup** — *what did class C declare?* | `ClassView::band_reading(class) -> BandReading`, **total**, zero-fallback default | sibling-consistent with `rail_carving` / `edge_codec_flavor`; hot-path safe |
| **projection** — *read THESE raw bits under that declaration* | `BandReading::project(raw, provenance) -> Result<Projected, BandReadError>` | this is where mismatch and stale-v1 bits actually live; D-ACR-7's falsifier demands a failure, not a plausible value |

The lookup can never fail (an undeclared class yields the zero-fallback
`BandReading`). The projection can, and must. **This is not a compromise
between the two savants — it is the recognition that they were describing
different functions.** Savant 1's finding is recorded as ACCEPTED-AND-SPLIT;
neither finding is discarded.

## L2 — writer-side is only half-closed (Savant 5 Q1)

Accepted. §3 policed readers only, so nothing stops a producer stamping bits
inconsistent with its own declaration — leaving F4's failure mode half open.
**Added:** `BandReading::admits(lens) -> bool`, the cheap pre-write check a
producer calls before `with_topology` / `with_reasoning_band`. It does not (and
cannot) *enforce* — enforcement at the write site is `causal-edge`'s to add and
is recorded as a follow-up, not smuggled in here.

## L3 — `Absent` vs never-declared (Savant 5 Q3)

Accepted. An audit that asks *"who opted out vs who never considered this"* was
unrepresentable. **Fixed by shape:** the registry returns
`Option<BandReading>` — `None` = never declared, `Some(BandReading { band:
Absent, .. })` = explicitly declared no band. The total lookup in L1 folds
`None` to the zero-fallback default for callers who do not care; the audit path
reads the `Option` directly.

## L4 — gate repairs (Savant 2 Q5 ×2)

- **G3, G4** gained their can-stay-silent halves explicitly (F9 requires BOTH
  stated): G3 now also asserts a correctly-matched lens returns `Ok`; G4 now
  also asserts `V2Stamped` **and** `V3Register` resolve without error.
- **G7** regime named: it is a **compile-time type-level assertion**, explicitly
  exempt from F9's runtime fire/silent duality. Stated rather than left
  ambiguous.
- Savant 2 Q6's AP1 watch accepted as a **gate**, not prose: **G9** — no
  `#[cfg(feature = ...)]` branch may change `BandReading` semantics under one
  name; a feature split must error, never re-mean.

## L5 — doc corrections (Savant 3, Savant 4 Q4)

- Savant 3: the `recipe_kernels` quote spans **319-322**, not 320-322. Fixed in
  §2.5. All other 5 code-truth questions returned CONFIRMS — **the v1 inventory
  is measured, not assumed**, which is the one thing this council most needed to
  verify.
- Savant 4 RISK accepted: correcting `TYPE_DUPLICATION_MAP.md:9` "×2"→"×4" also
  requires fixing **line 16** (its per-copy table lists 2 of 4) and **line 19**
  (*"rename one … in causal-edge"* is no longer a coherent single target across
  arities 4/4/5/3). All three lines change together or the doc stays internally
  inconsistent.

## L6 — accepted-as-stated (no change needed)

- Savant 2 Q1–Q4, Q6: no iron-rule violation. Zero-dep confirmed honest —
  `TruthLens::Trust` is a **doc-comment pointer**, not an import
  (`lance-graph-contract/Cargo.toml:10-17`).
- Savant 4 Q2: additive-only; no downstream crate is forced to change.
- Savant 4 Q5: D-ACR-2 is not pre-empted — `BandReading` is keyed by
  caller-supplied `(classid, rail)`, so no class-population work is created.
- Savant 5 Q5: `EdgeProvenance` stays in `contract`; moving it to `causal-edge`
  would force a dependency in one direction or the other. Cost recorded: it
  lives one crate from the bits it describes.

## L7 — recorded, deliberately NOT actioned (anti-collapse)

- **Savant 1 Q5 / #7** — *should `band_reading` be a third method on `ClassView`
  rather than a new module?* L1 adopts the `ClassView::band_reading` **method**
  for the lookup half, so this is partially taken. The **types** stay in their
  own module because `class_view.rs` is already 2000+ lines and the projection
  half has no `ClassView` analogue. The losing half of the finding is recorded,
  not deleted.
- **Savant 5 Q4** — *does `WitnessKind` belong on `Locus`/`WitnessEntry`?*
  (`causal_witness.rs:116-134` has `Locus::BasinAnchor = 8`.) Fork named and
  NOT taken: `WitnessKind` here is the *reference discriminator*, not the
  witness. If a later deliverable moves it, this line is the record that the
  alternative was seen and declined with a reason.

## §5′ — GATES, amended (supersedes §5)

G1, G2, G5, G6, G8 unchanged. G3, G4, G7 amended per L4. New: G9, G10.

| # | Gate | Pass |
|---|---|---|
| G3′ | Lens mismatch **and** match | `Topology`-declared read as `Trust` → `Err(LensMismatch)`; read as `Topology` → `Ok` |
| G4′ | Provenance, both directions | `V1Legacy`/`Unknown` → `Err(UnknownProvenance)`; `V2Stamped` **and** `V3Register` → `Ok` |
| G7′ | Arity pin (compile-time, F9-exempt — stated) | a const assertion pins CE64 `TrustTexture` arity at 4 |
| G9 | AP1 watch | no `#[cfg(feature)]` branch re-means `BandReading`; a split errors |
| G10 | **Carrier parity (L0)** | the same declaration projecting `CausalEdge64::truth()` and `CausalEdgeV3::truth_raw()` yields the identical result for the identical ordinal |

---
---

# v3 — PHASE 4 FIX LEDGER

> Reviewer verdicts in hand: **reviewer 2** (dilution-collapse) 8×PASS +
> 1×FIX(P2); **reviewer 3** (firewall) 6×PASS + **1×BLOCK(P0)** + 1×FIX(P1).
> **Reviewer 1 (overclaim) has not reported** — the first cast was lost with
> its sibling (no notification, `ListAgents` empty) and was recast.
> **v3 is NOT ratified until reviewer 1's verdict set is applied.**

## FIX-1 — L5 `BLOCK(P0)` (firewall): board hygiene was never committed to

**The BLOCK is correct, and it is a Phase-2 consolidation failure, not a spec
defect.** Verified: draft v2 contains **zero** occurrences of `LATEST_STATE`,
`STATUS_BOARD`, `EPIPHANIES`, `PR_ARC_INVENTORY` or `AGENT_LOG`. Savant 4
**answered** this in its Q3 with the full mandatory list; **my consolidation
dropped the finding entirely.** That is precisely the loss Phase 2 exists to
prevent, and it happened in Phase 2. Recorded as such rather than repaired
silently.

**Resolved by restoring the dropped answer as a committed section:**

### §6′ — BOARD HYGIENE COMMITMENT (same commit, non-negotiable)

| file | why it is required | content |
|---|---|---|
| `.claude/board/LATEST_STATE.md` | rule row *"a contract type / module"* | PREPEND a Contract Inventory entry for `band_reading` (types, what it mints: nothing) |
| `.claude/board/STATUS_BOARD.md` | rule row *"a new D-id / deliverable"* | flip the `D-ACR-7` row to Shipped with its falsifier outcome |
| `.claude/board/EPIPHANIES.md` | rule row *"a finding / correction"* — **three** qualify | PREPEND one entry covering: the `TrustTexture` ×4 homonym (arities 4/4/5/3, `Dissonant` unrepresentable in 2 bits), the v1-provenance trap and its **non**-application to V3, and the Result-vs-total split |
| `.claude/board/PR_ARC_INVENTORY.md` | rule row *"a merged PR"* | PREPEND on merge (post-merge commit, not at author time) |
| `docs/TYPE_DUPLICATION_MAP.md` | §3.4's own commitment | lines 9, 16 **and** 19 together (L5's cascade finding) |
| `.claude/board/AGENT_LOG.md` | rule row *"a completed agent run"* | ONE entry naming this council: which 5, which 3, verdict counts, v1→v2→v3 deltas — written by the orchestrator only (one-writer) |

## FIX-2 — §5′ G10 `FIX(P1)` (firewall): the gate was unhostable, and its own fix is unhostable too

The warden is right that G10 was not mechanically checkable, and its suggested
remedy — *"the test lives in `causal-edge`, importing the contract"* — **also
fails**, measured: `crates/causal-edge/Cargo.toml:20-23` reads *"No
dependencies — this crate is self-contained. `TrustTexture` is defined locally
(not imported from `lance-graph-contract`) to preserve the zero-dep
invariant."* **Both** crates are zero-dep by explicit design, so no single test
may hold both the contract and the carriers.

**Resolved by splitting the gate along the seam that already exists — the
projection takes RAW ORDINALS, so it never needed a carrier at all:**

| gate | host | asserts | zero-dep impact |
|---|---|---|---|
| **G10a** | `lance-graph-contract` | `project(raw, decl, provenance)` is a pure function of its arguments — identical `(raw, decl, provenance)` ⇒ identical result, for every `raw ∈ 0..4` (truth) and `0..8` (band). Carrier-agnosticism is then **by construction**: the function cannot see a carrier. | none — no carrier type is named |
| **G10b** | `causal-edge` | its OWN two accessors preserve the ordinal across `from_v1` → `rehydrate`: `CausalEdge64::truth()` == `CausalEdgeV3::truth_raw()` and the band likewise. | none — contract is not named |

**G10b is a genuinely MISSING test today, measured:** `edge_v3.rs`'s test module
has `v3_le_round_trip` (V3's own LE bytes), `mantissa_round_trips_raw_for_all_16_states`
and `inference_type_is_a_lossy_projection_of_the_mantissa` — **none asserts
truth/spare preservation across the lift/rehydrate pair**, despite the module
doc claiming *"every meaningful CE64-v2 field survives the round trip
byte-exact"* including *"the truth/topology 2-bit ordinal, and the spare/band
3-bit ordinal"*. So the FIX(P1) surfaced an unverified doc claim in shipped
code — recorded, and G10b is the test that closes it.

## FIX-3 — L6 `FIX(P2)` (dilution-collapse): "accepted" must read as verified, not asserted

Each accepted-as-stated bullet gains the `file:line` rigor L0/L5 use:

- **Savant 2 Q4 (zero-dep honest)** — `crates/lance-graph-contract/Cargo.toml:10-17`:
  `[dependencies]` has **zero entries** plus the explicit comment *"Zero
  dependencies by design … MUST stay dependency-free even of optional path
  deps"*. `TruthLens::Trust` is a doc-comment pointer; no edge is created.
  (Independently re-verified by reviewer 3 this pass.)
- **Savant 4 Q2 (additive-only)** — measured: **zero** glob imports of
  `lance_graph_contract::*` exist workspace-wide, so a new module cannot
  break any downstream build. Consumers adopt it or ignore it.
- **Savant 4 Q5 (D-ACR-2 not pre-empted)** —
  `alpha-channel-rung-overlay-v1.md:1190-1192`: *"D-ACR-2 and everything after
  it sit behind an operator mint decision that this plan does not pre-empt."*
  `BandReading` is keyed by **caller-supplied** `(classid, rail)`, so it
  populates no registry.
- **Savant 5 Q5 (`EdgeProvenance` placement)** — accepted WITH its cost named:
  it lives one crate from the bits it describes, and (per FIX-2's measurement)
  that is forced by BOTH crates' zero-dep posture, not a preference.

## FIX-4 — L0 addendum: the two-armed handover (operator, after the first cast)

`WitnessKind`'s target acquired a shape while the council ran
(`known-unknown-handover-network-v1.md` §9 ⊘⊘): handover is **two-armed by
substrate** — static ontology → an alpha-layer focus entry at the same address;
dynamic substrate → in place, with **Lance versioning** as the residue carrier
(`QueryReference::at(v, rung)`, a projection, zero copies). **No ownership ever
changes.**

Consequence for THIS spec, and it is small by design: `WitnessKind` remains a
**reference discriminator** and gains one doc sentence naming the two arms. No
type, field or gate changes — the reading contract is upstream of where the arms
diverge. Recorded so a later session does not read the two-armed design as
implying a `BandReading` variant per arm.

## Outstanding before ratification

1. **Reviewer 1 (overclaim-auditor)** — verdict set not yet received.
2. Any `BLOCK` it raises returns to Phase 0 rather than being argued away here.
3. Stricter verdict wins on any conflict with reviewers 2/3.

---

## FIX-5 — L0 `BLOCK(P0)` (overclaim): "V3's bytes were never temporal" is FALSE for populated instances

**The BLOCK is correct and is the council's most consequential finding.**
Verified at source this pass:

```rust
// edge_v3.rs:117 — no provenance parameter exists
pub fn from_v1(e: CausalEdge64, target: u16) -> Self
// edge_v3.rs:138-139 — a raw bit copy from a caller-supplied edge
p[8] = (e.w_slot() & 0x3F) | ((e.truth_raw() & 0b11) << 6);
p[9] = e.spare() & 0b111;
```

The reassuring comment above it (`:135-137`) — *"Under the v1 layout every one
of these accessors is a documented zero stub"* — is a **compile-time feature
condition**, not a runtime provenance guarantee. Under a v2-compiled build, a
CE64 of v1 or unknown provenance (whose bits 61-63 alias `temporal >= 512`)
lifted through `from_v1` carries those stale bits into V3 byte 9, and **the
resulting register is indistinguishable from a clean one.**

**Had this shipped, the reading contract would have been self-defeating at its
own core claim** — `V3Register` declared "always readable" is exactly the
plausible-wrong-answer §3.3 exists to refuse. It is the difference between a
statement about the type's *shape* (true: V3 has no v1 history) and about its
*populated instances* (false: they inherit whatever they were lifted from).

**Resolution — three changes, none touching the design:**

1. **`EdgeProvenance::V3Register` is redefined**: it means *"the caller asserts
   this register was minted clean"*, **never** *"V3 registers are clean"*. The
   contract **cannot infer** V3 provenance and must not try — the information
   was destroyed at the lift, one crate away.
2. **A V3 register of unstated origin is `Unknown`**, and `Unknown` already
   refuses (G4′). Zero-fallback applies: absent an assertion, refuse.
3. **Recorded as a real gap in `causal-edge`, filed as follow-up, NOT fixed
   here**: `from_v1` drops provenance. Fixing it means a signature change in a
   crate this deliverable does not own; the reading contract's job is to stop
   *trusting* what was lost, not to un-lose it.

L0's asymmetry claim is **struck** and replaced: the v1 trap applies to
**both** carriers — on CE64 directly, on V3 **transitively through the lift**.
That is a simpler contract, not a more complex one.

*(Citation corrected: the module-doc quote spans `edge_v3.rs:83-87`, not
86-90.)*

## FIX-6 — §5′ `BLOCK(P0)` (overclaim): a gate contradicted the fix, and another was a tautology

**G5 contradiction — correct, and it is FIX-1's own doing.** L1 adopted a
**total** lookup while §5′ carried G5 (*"an undeclared one errors"*) forward as
*unchanged*. Both cannot hold. Resolved by naming the surface each gate tests:

| gate | surface | asserts |
|---|---|---|
| **G5a** | `ClassView::band_reading` (total) | an undeclared class yields the **zero-fallback** `BandReading` and does NOT error — sibling-consistent |
| **G5b** | `BandReading::project` (fallible) | projecting under an undeclared class returns `Err(UndeclaredClass)` — the guard that must fire |

The audit distinction L3 bought (`Option` = never-declared vs `Some(Absent)` =
declared-no-band) is what makes G5a/G5b non-vacuous: they read different
returns of different functions, not the same thing twice.

**G10 tautology — correct, and it kills my own FIX-2 G10a.** G10a as written
(*"`project` is a pure function of its arguments"*) feeds the same input to the
same function twice and asserts the same output. That is the vacuous-assertion
pattern `CLAUDE.md`'s falsifiability rule names outright. **G10a is deleted.**

**Only G10b survives, and FIX-5 has now doubly motivated it:** compare
`CausalEdge64::truth()` against `CausalEdgeV3::truth_raw()` on the **same edge
after `from_v1`** — which tests the bit-copy fidelity at `edge_v3.rs:138`,
precisely the site where FIX-5 showed provenance is destroyed. It is hosted in
`causal-edge` (naming no contract type), so both zero-dep postures hold. And it
is a **missing test today**, measured: `edge_v3.rs`'s test module has
`v3_le_round_trip`, `mantissa_round_trips_raw_for_all_16_states` and
`inference_type_is_a_lossy_projection_of_the_mantissa` — none asserts
truth/spare preservation across lift/rehydrate, despite the module doc claiming
it.

## FIX-7 — L2 `FIX(P2)`: "cannot" is now shown, not asserted

`admits()` cannot enforce **by construction**, and the evidence is
`crates/causal-edge/Cargo.toml:20-23`: *"No dependencies — this crate is
self-contained. `TrustTexture` is defined locally (not imported from
`lance-graph-contract`) to preserve the zero-dep invariant."* A contract-side
helper structurally cannot hook `causal-edge`'s `with_topology` /
`with_reasoning_band` call sites, because the dependency edge that would let it
does not and must not exist. [G]-grade, now cited.

## RATIFICATION

| reviewer | verdicts | resolved by |
|---|---|---|
| 2 — dilution-collapse | 8 PASS, 1 FIX(P2) | FIX-3 |
| 3 — firewall | 6 PASS, **1 BLOCK(P0)**, 1 FIX(P1) | FIX-1 (board hygiene §6′), FIX-2 |
| 1 — overclaim | 4 PASS, **2 BLOCK(P0)**, 2 FIX(P1), 1 FIX(P2) | FIX-5, FIX-6, FIX-5 (L5 citation), FIX-7 |

**Three BLOCK(P0) raised, three resolved in Phase 4 — none argued away, none
requiring a Phase-0 re-spec** (each changed a claim, a gate, or a commitment;
the design — declaration table + fallible projection, two carriers, one
contract — survived all three unchanged).

**v3 is RATIFIED.** Implementation may proceed against §3 as amended by
FIX-1..7, with gates G1, G2, G3′, G4′, G5a, G5b, G6, G7′, G8, G9, G10b.

---

## Post-ratification operator addendum (2026-08-21): carrier wiring status + the temporal doctrine

Operator, after ratification: *"Alle V3-Varianten von CausalEdge,
EpisodicWitness und epistemic witness sind unwired oder planned. Temporal is
implicitly in the epistemic pothole; can be explicit in Rubikon revision,
CausalEdgeV3, and attention v3."*

**1. The unwired status is now operator-confirmed, not only measured.** §2.3
measured zero bit-consumers; the operator confirms the generation-level fact:
`CausalEdgeV3` is shipped-parallel-unwired, `EpisodicEdges64` is unmounted
(D-ACR-17 unbuilt), `EpisodicWitness64` never became code, `HoleV3` is blocked
(BoardAggregates = 15). **The reading contract is therefore prescriptive by
construction AND by ruling** — it defines how the first consumer reads, it
retrofits nothing.

**2. The temporal doctrine — the contract carries NO temporal field, and that
is doctrine, not omission.** Time is **implicit in the epistemic pothole**: a
pothole opens at `first_possible`, closes at its `Revision`, and the span lives
in the Lance version stream (`QueryReference::at(v, rung)` — the dynamic arm of
the two-armed trace). Explicit temporal exists in exactly **three sanctioned
homes**, none of them this contract:

| home | status | its own rule |
|---|---|---|
| Rubikon revision window | planned (§3f's open interval question, unresolved) | the window is cited, not recalled |
| `CausalEdgeV3` byte `[7]` TE | shipped, unwired | *"an INDEPENDENT signed relative chain offset the producer sets explicitly, never inherited"* (`edge_v3.rs`) |
| attention v3 | planned, not shipped | a future ClassView reading — not `attention_facet` (which is deliberately atemporal) |

**3. This is the v1 trap's lesson, generalized.** Bits 61-63 went stale
precisely because temporal lived *implicitly in a reclaimable field*.
Implicit-in-versions cannot go stale (versions are append-only); explicit
temporal is allowed only where a producer deliberately sets it. And
`EdgeProvenance` is about **layout epoch, never time** — conflating those two
axes is the category error the episodic-vs-epistemic honesty line already
fences one level up.
