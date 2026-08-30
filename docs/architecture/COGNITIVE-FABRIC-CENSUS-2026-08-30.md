# COGNITIVE-FABRIC-CENSUS — 2026-08-30

> **What this is.** A source-first, three-repo census (lance-graph + OGAR +
> the private domain consumer MedCare-rs) of the layers the reusable-cognition
> convergence (#1077 / #1078) will compose: the reasoning stack, the
> `BeliefArena` duplication vs #1078's W0 ownership census, the alpha/rung
> substrate, and the atom/masking execution membrane. Every claim carries
> `file:line` gathered at the pins below.
>
> **What this is NOT.** Not a plan (no D-ids, no waves — #1077 owns the atom
> census, #1078 owns the ownership waves; this document CONSUMES both and
> competes with neither). It exists to hand `W0` its `SOURCE FACT` rows and to
> satisfy `F-ARW-0` for them: any absence claim below rests on a three-repo
> read, never a single-repo grep.
>
> **The privacy fence, applied.** Per this crate's facade rule ("no domain
> catalogue of any kind … not even in doc examples") and the consumer's own
> commitments, this document carries **zero domain vocabulary**. The
> domain-detailed rendition lives in the private consumer
> (`MedCare-rs docs/REASONING_FABRIC_CROSSMAP.md`, PR #596 there). Ids that
> cross the seam are opaque `u16`s here, as the facade requires.

**Pins:** lance-graph `origin/main` `de1d0c2f` · OGAR `ddd51f4` · MedCare-rs
`origin/main` `e929179`. Prior arc treated as merged ground per #1077:
#1074 (MUL↔EWA), #1075 (the epistemic triptych — `revision.rs`, `fusion.rs`,
the Rubicon revision), #1076 (hygiene, chain terminated).

---

## §1 — The reasoning stack, as shipped

```
consumer domain model  (private repo — owns MEANING, mints opaque u16 ids)
        │  one stamped premise per independent evidence Axis
        ▼
lance-graph/src/reasoning.rs          ← the concept-blind facade (553 lines)
        │  PremiseBundle → resolve() / differential(&Throttle) -> Frontier
        ▼
lance-graph-planner/src/nars/         ← the mechanism home
        truth.rs    TruthValue: deduction/induction/abduction/analogy/revise
        belief.rs   BeliefArena: statement-keyed; S4 stamp-disjointness —
                    disjoint stamps POOL (confidence rises), overlap → CHOICE
        tactics.rs  RCR#4/TR#6/ASC#7/CAS#8/CR#11 · Throttle{c_min,budget,
                    hub_indegree} · Frontier{candidates,gaps} · GapKind
        ▲
        │  the concept-level belief/SPO stream
deepnsm-v2                            ← the INBOUND LEG (ruled scope:
                                        E-DEEPNSM-V2-IS-INBOUND-LEG-…-1)
```

Load-bearing details, measured:

- **`MAX_AXES = 64`** (facade): `Stamp::source(id)` folds `id % 64`; axis 0
  and axis 64 would collide, silently degrading POOL to CHOICE — `Axis::new`
  refuses out-of-range, so the fold is unreachable through the facade.
- **`GapKind`** (`tactics.rs:99-110`): `NoSharedMiddle / NoSibling /
  NoAbstraction / HubExcluded / BudgetExhausted` — a first-class reasoning
  FAILURE, "the reach-out hook". `Candidate{stmt, truth, premises:[u32;2],
  rung, tactic}` with `rung` = Tarski `max(premise rungs)+1`.
- **Counterfactual** re-exports (`worlds_differ`, `substitute_binding`,
  `Counterfactual`) carry the standing litmus: *terms compose statements;
  bits measure fingerprints — bits = reject.*
- **The consumer adapter is NOT a handroll.** Verified at its source: every
  mechanism symbol (`PremiseBundle, Axis, TruthValue, Throttle, CStmt,
  Copula, Frontier, GuardRule, detect_violations`) is imported from this
  facade; no local truth function exists in the consumer. What the consumer
  owns is meaning: id minting, its four independent evidence axes, its
  criteria tally, and its admission taxonomy (§3).

---

## §2 — The `BeliefArena` duplication: the W0 verdict

**Already ruled, already ledgered — W0 must cite it, not re-derive it.**

| SOURCE FACT | where |
|---|---|
| The ruling: reasoning lives in lance-graph; deepnsm-v2 is the inbound leg | `EPIPHANIES.md` `E-DEEPNSM-V2-IS-INBOUND-LEG-REASONING-LIVES-IN-LANCE-GRAPH-1` (2026-07-23) |
| The debt: V0 arena (`deepnsm-v2/src/belief.rs`, 625 lines, contract `NarsTruth`) superseded by the canonical arena (`lance-graph-planner/src/nars/belief.rs`, 476 lines, `TruthValue`) | `TECH_DEBT.md:752` `TD-DEEPNSM-V2-BELIEF-DUP` |
| Named blockers at filing time: (a) V0 tests = the D-DIA-V0 falsifying-slice record; (b) "check for external consumers first" | same TD entry |
| Guard provenance: the S4 stamp-disjointness guard was PIONEERED in the V0 copy | `TECH_DEBT.md:754` `TD-NARS-REVISION-UNGUARDED` |

**Two NEW measurements (2026-08-30, three-repo census — `F-ARW-0` clean):**

1. **Blocker (b) is CLEAR.** External consumers of
   `deepnsm_v2::{Belief, BeliefArena, CStmt, Copula, ReviseOutcome, Stamp}`
   across lance-graph + OGAR + the consumer repo: **zero**. Sole reference:
   deepnsm-v2's own re-export (`lib.rs:59`).
2. **The copies have DIVERGED.** Common surface: `new/entries/get/observe/
   revise_at/close_transitive/source/disjoint/union/transits`. **Planner-only:
   `admit_derived`** (`nars/belief.rs:226`). The dup compounds with every
   planner-side extension.

**The gap:** #1078's W0 census list names `reason.rs` + consumers, CE64/W-slot,
alpha primitives, revision/fusion/Rubicon, loco/r2il, the consumer's alpha and
session — **but not `nars/belief.rs` and not the TD id** (verified by grep
against both open plan heads). Since §7 of that plan lists "NARS
deduction/induction/abduction/revision primitives" as atom candidates, and an
atom needs exactly ONE backing arena, the dedup is upstream of the atom work.
With measurement 1 the TD's stated precondition is met; the payment path is
the TD's own: deepnsm-v2 EMITS the SPO/belief stream, consumes the planner
arena, deletes `belief.rs` + re-exports, citing the D-DIA-V0 tests.

---

## §3 — Consumer taxonomy ↔ `Frontier`/`GapKind`: the W2 seam, measured

The consumer's admission taxonomy — `Differential` (≥1 discriminating
observable met; the only class carrying a number), `UnderDetermined`
(supported only by observables shared with rivals; deliberately numberless,
reported as *what to measure next*), `NotRaised` — is a domain re-derivation
of a shape this repo already ships one level up:

| consumer (domain layer) | planner (`nars/tactics.rs`) | shared idea |
|---|---|---|
| `Differential` | `Frontier::candidates` | ranked, because evidence separated it |
| `UnderDetermined{shared, missing}` | `Frontier::gaps` (`ReasoningGap` naming the missing term) | blocked — and HERE is what is missing |
| `NotRaised` | throttle exclusion + `GapKind::HubExcluded` | never entered the frontier |
| "a marker shared by >1 rival discriminates nothing" | `hub_indegree` middle-term exclusion | **the same computation** — sharing-count as disqualifier |
| a domain prior gate: a candidate CLASS raised only on class-specific evidence | **no upstream equivalent** | the one genuinely novel consumer mechanism — the natural upstream candidate, moved as MECHANISM with its meaning left behind |

Three measured facts:

1. The facade's `differential()` already RETURNS a `Frontier` — the gap arm
   is not hidden.
2. The consumer EXPOSES it (`abductive_frontier() -> Frontier`, permissive
   throttle) — **and it has zero call sites.** Its UI derives "what to
   measure next" from a parallel domain-local tally instead.
3. So W2's ownership cut has a precise seam: generic halves
   (specificity-as-sharing-count, ranked/blocked/not-raised, name-the-missing-
   term) exist HERE and should be consumed, not re-derived; domain halves
   (identities, criteria, priors) stay private — exactly `F-ARW-2`'s test.

Per the consumer's surface-never-file rule, this is a census row for W2, not
an upstream change request.

---

## §4 — Alpha / rung / horizon: the orthogonality census

**Upstream primitives (shipped):** `RungLevel` 0..=9
(`contract/src/cognitive_shader.rs:157`, `from_u8` saturating — THE one
u8→rung mapping); `RowFocusMask` + `AttentionFocusFacet` + `FocusAxis`
(`contract/src/attention_facet.rs:370,185,128`); `FocusTrace`, `coverage()`,
`breadth_bits()`, `read_crossing(pre, post, ε)`, `RubiconVerdict`
(`contract/src/rubicon_witness.rs`; `PRE_RUBICON = Planning`, `POST_RUBICON =
CognitiveWork` — the Heckhausen crossing read from focus-of-attention).
Stale-line correction for #1077's §1 table: piece D (`RowFocusMask`) is no
longer "ABSENT from code" — it is shipped at the line above. Grading stands:
alpha integration is **REPRESENTATION-ONLY / HELD** until a producer→consumer
trace runs (`D-RLR-5`).

**Consumer side (verified, correcting an earlier same-session absence claim
that failed what is now `F-ARW-0`):** the private consumer ships a REAL
thin-provisioned same-address overlay over canonical `NodeRow` (1,190 lines):
stamp `{cycle: u32, seq: u32, rung: u8, visits: u16}` in the 16-byte value
slot; **allocate ≠ claim** (unclaimed reads `None` = "not attended", no
fallback to the base row); the key copied verbatim from the base row (never
re-minted — no tail decode, no V1/V3 exposure); the overlay borrows the spine
immutably, so *"the overlay reads the graph; the graph never reads the
overlay"* is a **compile-time** property. Live consumers exist (a
first-thought walk with a session-isolation falsifier). The earlier grep for
UPSTREAM type names concluded "zero wired" — the wrong test for a natively
implemented overlay; recorded here so the next census greps for the semantics,
not the names.

**⊘ METAPHOR REGRADE (operator, 2026-08-30): the Photoshop-alpha name
describes the STORAGE layer only — reserve-but-don't-claim, akin to the
split-tunnel trick.** The metaphor names the ALLOCATION GEOMETRY: allocate
the whole same-address space at zero rows, claim (materialize) only where
attention actually landed, discard whole. The same law already has ONE name
across the workspace — **RESERVE-DON'T-RECLAIM** (the consumer overlay's
`EdgeBlock` note; the A9 register's reserved slots 16..24; OGAR's
zero-fallback ladder: "a zero tier means *not consulted*, never *compacted
away*"). Three metaphors, one mechanism, three ASPECTS of one substrate:

| aspect | its name for the law |
|---|---|
| allocation geometry | Photoshop alpha / thin provisioning / split tunnel — reserve all, claim attended |
| exposure | photolithography — the mask exposes; unexposed surface costs nothing |
| operation | the Java receipts — mask-native ops on the resident substrate; materialization is a terminal escape |

> **⊘ CORRECTED IN PLACE (operator, 2026-08-30, same day — the first filing
> of this block over-read the regrade).** The first version concluded "a
> storage layer is supposed to be representation-only; the live-cognition
> question moves one layer up." **Absolutely wrong — in zero-copy, storage
> IS cognition.** There is no storage/compute split to retreat behind: the
> substrate's own canon says the memory organs are *"thinking tissue — not
> storage"*, and the zero-copy law says the array itself is a ClassView
> projection. The claimed alpha rows are LIVE cognitive state — reading and
> writing them is the thinking, not a record of thinking that some higher
> layer performs. Consequences, restored to their correct strength:
> **(a)** #1077's `REPRESENTATION-ONLY / HELD` grade STANDS for its stated
> reason — the producer→consumer trace (`D-RLR-5`) has not been RUN — not
> because the question was wrong; the metaphor regrade changes what the
> NAME claims, never what the trace must prove. **(b) ⊘ RE-CORRECTED
> (operator, same day): "alpha presence never mints evidence" is itself a
> BULLDOZER, not a fence** — *"a bullshit claim adding a bulldozing label
> where I just gave you an order to use tarski etc as orthogonal
> scalpels."* A blanket ban collapses "evidence" into one undifferentiated
> category on the very page that orders nine orthogonal axes — the exact
> one-blended-score failure. The scalpel-grade replacement, each row
> grounded in shipped code:
>
> - **object-level truth axis** — a visit does NOT raise a domain claim's
>   `(f,c)`; attention landing on X is not support for X. This is the one
>   true kernel the bulldozer was gesturing at, and it is already TYPED
>   (the Strict/Retro horizon machinery + the ddx hindsight lesson), so it
>   needs no extra label.
> - **meta/attention axis** — the residue IS the evidence: `read_crossing`
>   already consumes two `FocusTrace`s as the EVIDENCE of the Heckhausen
>   crossing (`RubiconVerdict` is derived from attention residue, shipped).
> - **Shannon axis** — claimed vs unclaimed is the searched/unsearched
>   partition: "examined N times, yielded nothing" is real information and
>   is exactly what a what-to-measure-next surface quantifies.
> - **witness/temporal axes** — the stamp (`cycle`, `seq`, `rung`,
>   `visits`) is provenance evidence of who looked, when, in what order;
>   replaying it is evidence about the trajectory of the thought.
> - **the transfer law** — evidence moves BETWEEN axes only through typed
>   inference (revision under disjoint stamps, the tactics), never by
>   leakage. The operator's own prior formulation was already exact:
>   *"Eye tracking ist kein Beweis, aber damit löst du das
>   needle-in-a-haystack Problem"* — no on one axis, load-bearing on the
>   others.

**The convergence this makes visible (the operator's prediction, checked):**
#1077, #1078 and the Java arc are ONE architecture seen from three angles —
#1077 the PROGRAM angle (byte-addressed atoms over the same reserved node
geometry; a new carrier before loco is proven insufficient = STOP), #1078 the
OWNERSHIP angle (who owns meaning vs mechanism over the same addresses), the
Java arc the EXECUTION angle (masks as the currency; R7's 960 B for 10⁹
projections is the measured price of "don't pay for the unattended"). Every
axis of the scalpel table then reads as an annotation ON the one resident
substrate, selected by masks, exercised by programs, recorded by receipts.

**The gap this census adds:** the consumer's REASONING path (its differential
surface and its step trace) contains zero references to that overlay — the
thinking runs beside a live attention channel it never claims into. The
lawful wiring is #1078's W6 BUY chain, not a local patch.

**The three coordinates, kept apart** (per #1078 §2.1 and its STOP list —
"No Tarski↔cognitive-rung collapse"):

| coordinate | carrier | meaning |
|---|---|---|
| Tarski depth | `Candidate.rung` = `max(premise)+1`; `reason.rs` derivation depth | distance from a leaf |
| cognitive rung | `RungLevel` / the overlay stamp's `rung: u8` | which attention layer landed |
| epistemic horizon | `planner/src/temporal.rs:87-97` `EpistemicMode::for_rung(0..=4 → Strict, 5..=8 → Aware, _ → Retro)` + `admits(status)` | what a rung may SEE — "low rungs reason strictly in the present; mid rungs admit hindsight; top rungs may spoiler-read" |

**The scalpel table (operator direction, 2026-08-30: the orthogonal axes are
what make causality reasoning a scalpel).** Each axis cuts a DIFFERENT causal
plane; each has a shipped carrier; each has a named collapse failure:

| axis | carrier | the cut it makes | collapse failure |
|---|---|---|---|
| Tarski depth | `Candidate.rung`; `reason.rs` premise pointers | how far a conclusion is from ground; which premises to re-open | depth read as confidence |
| Shannon `H` | `sensorium.rs:20` truth entropy; `nars/insight.rs:131,177` (normalized, 10-bin) | where uncertainty concentrates — which observation discriminates | "lower entropy = evidence gain" (constitutionally refused) |
| EWA `Σ` | `jc` `Spd2` sandwich `Σ' = MΣMᵀ` + the 3D log-normal-corrected KS concentration bound (`ewa_sandwich_3d.rs:36,476`) | how doubt propagates through a transform — the ANISOTROPY of trust | `Σ` squeezed to a scalar (banned: no new trust scalar) |
| NARS `(f,c)` | `TruthValue` | strength of the claim itself | conflated with band/authority |
| `CausalTopology` | CE64 2-bit field; `DismechTopology` `Direct / IndirectKnown / IndirectUnknown / Unknown` — **source-authoritative, never inferred** (`dismech_evidence.rs:52`) | WHAT KIND of link — incl. the epistemic-restraint control: "a reasoner that 'recovers' [an unknown mediator] is hallucinating closure" | topology inferred instead of asserted |
| Pearl 2³ | `exploration.rs` SEE/DO/IMAGINE; the SPO "2³ Projection Verbs" (`spo/store.rs:105`) | WHICH question — association vs intervention vs counterfactual | a rung-0 read answering with Retro material |
| rung 0–9 / meta | `RungLevel`; overlay stamp; `PhaseCensus`; `read_crossing` | who is thinking about the thinking — parallel, informing, never rewriting | Tarski↔rung collapse (banned) |
| temporal horizon | `EpistemicMode::for_rung`; `Spoiler` vs `Anachronistic` (`temporal.rs:845-856`: the SAME future row refuses for a `Strict` reader and admits as a deliberate, opted-in `Spoiler` for a rung-9+ `Retro` reader) | what the reasoner may SEE — hindsight opt-in, never leak | hindsight contamination |
| witness identity | `ReasoningWitness64` + `replay_ref` | who saw it — provenance, replay | cross-axis leakage: attention residue read as object-level support without typed inference |

The precision is the ORTHOGONALITY: nine assertable coordinates instead of one
blended score. The in-house proof of the failure mode is §8.0 of
`SYNERGY-MAP-S00-S07.md` — "numeric coincidence was treated as identity."

An instance of the banned collapse was proposed and WITHDRAWN this session
(deriving a trace's attention rung from its derivation depth) — recorded so it
is not re-proposed. #1077's finding 2 stands alongside: the "rung-4 physical
requirement" is a fossil (5 hits, 4 doc-only, 1 test); *"rung already differs
by temporal horizon, not by the right to think — what is missing is
demonstration, not permission."*

**The chained dependency: parallel rung scheduling → SPO 2³ → stockfish-rs
(measured 2026-08-30).**

```text
parallel rung scheduling  (ruled 2026-08-30)
  ├─ rung-aware dispatch loop ………………… MISSING — PhaseCensus/kanban is
  │     deliberately rung-blind (cycle_driver.rs:6: "no new semantic /
  │     temporal / rung / witness type"); the ruled work item is composing
  │     the census tick with per-rung readers, not adding a rung type there
  ├─ per-rung horizon reads ………………………… SHIPPED — QueryReference::at(v, rung)
  │     → EpistemicMode::for_rung; externally falsified by stockfish-rs
  │     examples/hindsight_stream.rs (D-SF-HINDSIGHT-1) re-running ladder
  │     reads under Strict discipline over real games (temporal.rs:856)
  ├─ multi-rung coexistence …………………………… SHIPPED representation-only —
  │     probe_parallel_rung (F-PARALLEL-RUNG-1): one arena holds rungs
  │     0/1/2 simultaneously (4/3/3, none demoted); the probe's own doc:
  │     "does NOT establish wall-clock or thread parallelism"
  ├─ read-cost amortization ……………………………… SPO 2³ verbs SHIPPED as the
  │     projection algebra (8 query directions over ONE resident store);
  │     the 8-cycle L1 amortization CONTRACT stays OPERATOR-RECOVERED
  │     INTENT (pass-1 archaeology) — MEASURE before pricing N rungs "free"
  └─ first-level cheapness ………………………………… the §5 membrane — gated by the
        BROKEN WIRE (top_k→window) and §5.6's CallMask/VAR_SET blockers
```

**stockfish-rs's role, measured — falsifier and oracle, NOT accumulator
template.** Real contributions: the hindsight falsifier above; the ratified
teacher stack (`DecisionEpisodeV1`, `TeacherTrace`, `TeacherLabel`,
`CandidatePolicy`, `search_with_order`, `PositionKey`, `GameEpisodeKey`) as
the expert-iteration oracle; and NNUE feature-transformer columns as a
demanding TEST CORPUS that certified the palette256 cosine replacement
(`nnue_palette_cosine.rs`: ρ_all ≥ 0.999 — "a dataset that validates a codec,
not a mechanism the substrate imitates"). RETRACTED and not to be rebuilt:
"NNUE-style accumulator over a 64×64 = 4096 tile" — `SYNERGY-MAP` §8.0's
measured retraction (domino is a 4×4/16-lane tile; no NNUE concept in
`domino.rs`; the 64×64 in `lane_j` is a cache-tier knob; attention is 256×256
palette-archetype). This census's session repeated the retracted framing once
before re-reading §8.0 — recorded as a second `F-ARW-0`-class lesson: a
downstream repo's pre-correction note (tesseract-rs) is not the canonical
board.

---

## §5 — The atom / masking execution membrane

The convergence thesis — first-level orchestration as cheap as MASKING, with
hardened native atoms made flexible through a byte-addressed program surface —
is largely SHIPPED design, not aspiration. Located:

### 5.1 `ogar-loco` (OGAR) — the low-code program surface

- **`Call = (function : value)`** — two bytes, each an index into a 256-entry
  codebook; every call is an address into a **256×256 table**.
- `DOMAIN_FLOOR = 0x90`, const-asserted stored-byte ABI — **144 shared-core
  slots** below, domain vocabulary above via the `Vocabulary` seam.
- **Body budget = 360 bytes** (`CONTENT_SLOTS 30 × PAYLOAD 12`,
  const-asserted; the 512-byte node's 480-byte slab minus 30 interleaved
  4-byte classids) → **180 / 120 / 90 calls** per body by `LaneShape`
  (mirroring `CascadeShape::{G6D2, G4D3, G3D4}`; local copy by design, to
  keep the plug-and-play posture). Refuse-don't-truncate: overflow demands a
  SPLIT. *(Corrects the informal "280+ bytes" figure: the ABI constant is
  360.)*
- Proving consumer: the `blockly-rs` block-editor arc (Blockly/Scratch opcode
  palette `0x1717` cited); template DSLs and flow frontends are the named
  siblings — "just different vocabulary".

### 5.2 `ogar-r2il` — the IR proxy underneath

r2sleigh's R2IL opcode set as an `ogar_loco::Vocabulary` — "it mints nothing:
no node layout, no lane carving, no call encoding, no second addressing
system." Two load-bearing choices: the **masked lane projection** ("re-reads
one already-written body under any `LaneShape` **without rebuilding it**" —
the photolithography-for-free property, shipped), and **no `r2sleigh` dep**
(opcode set consumed as an enumeration→arity table — the accepted, thin
cost of the IR layer). Siblings: `ogar-ro` (Relation Ontology predicates as a
callable `Vocabulary`), `ogar-elk`.

### 5.3 lance-graph-java — the mask-algebra receipts

`RowStore.hop(int edgeClassid, WideFieldMask, Mask) -> Mask` (mask-in /
mask-out) behind a fluent immutable `Graph.hop(...)` chain; 20 exported C
symbols incl. mask `and/or/andnot`, fused `plan_eval`, `hop`. Banked:
**R7** — 10⁹ group projections allocate exactly **960 B**, byte-identical
across runs; **R12** — the seam carries ordinals, never objects. Bounding
gap: `.at(version)`/`TemporalPov` — zero references in `java/` or
`consumers/`. The Java arc is the mask-algebra reference, NOT the temporal
one.

### 5.4 The atoms, located

| atom | home | state |
|---|---|---|
| Shannon `H` | entropy-closure plan lineage | §7-listed loco candidate (#1078) |
| proprioception | `contract/src/proprioception.rs` (+ qualia/world_map/world_model/crystal) | shipped |
| EWA sandwich | `crates/jc/src/ewa_sandwich_3d.rs` (+ splat bridge); MUL↔EWA coupling = merged #1074 | shipped + planned |
| counterfactual | `lance-graph-cognitive::world::counterfactual`, facade re-export | shipped (bits-vs-terms litmus) |
| revision / fusion / Rubicon revision | **merged #1075** (the epistemic triptych) | shipped |
| dialectic (thesis/antithesis/synthesis) | the 5 planner tactics incl. `cr_synthesize` | shipped V1 |
| Rubicon crossing (Heckhausen) | `rubicon_witness.rs` (`read_crossing`; kanban Planning→CognitiveWork→Evaluation→{Commit,Plan,Prune}) | shipped; `D-ACR-8` two-sided falsifier |
| the recipe ladder | `lance-graph-ogar/src/recipe_vocab.rs` — `op_of`/`recipe_of`/`ladder_program() -> Vec<FnIndex>` | a program already IS a byte-addressed atom sequence (#1077 finding 3; `F-RLR-2` STOP guards it) |

### 5.5 The carriers and the unified carving

- `CausalEdge64` — the reasoning hot path. `CausalEdgeV3` — band carve at
  bytes 8/9 (`band_reading.rs:26`), and it **rehydrates INTO CE64 to
  reason**; `from_v1`'s drops are assertions, never inferences (`:164`).
- `EpisodicWitness64` — **⊘ NOT YET A CODE SYMBOL** (`soa_view.rs:272`
  states it verbatim: "a queued design"); the deferred-accessor comment names
  a type that does not exist. Shipped witness names today:
  `ReasoningWitness64`, `EpisodicEdges64`, `WitnessTable`,
  `CausalWitnessFacet`. The documented `WitnessTable` wording drift
  (`{mailbox_ref, spo_fact_ref}` vs "witness corpus root handle") is #1078's
  W3, and the private consumer holds a LIVE encode/decode falsifier for it.
  *(Naming correction credited to a parallel 2026-08-30 audit; this doc's
  first filing carried the queued name in a carriers list unflagged.)*
- The 24×`i4` register — the SAME 12 content-blind bytes carry **three
  class-selected readings** (`causal_witness.rs`): frozen `12×u8` palette,
  orchestration `6×(8:8)` (`style_rails_at`, V3-replayable), and the A9
  `24×i4` Markov loci (`:182`; 16 operator-named — `temporal … antecedent,
  basin_anchor … quorum, contradiction` — slots 16..24 reserved-empty).
  **A reading, not a fourth `CascadeShape`** — the module's own correction:
  `G24N4` is a lane-shape name; `CascadeShape` gains no 24-group variant
  (`canonical_node.rs:893,1078`). Class id selects the interpretation; no
  bytes move — the lawful unification pattern in miniature.
- `CascadeShape::{G6D2, G4D3, G3D4}` over the ONE content-blind 12-byte V3
  payload (`facet.rs`): `6·2 = 4·3 = 3·4 = 12`; `ALIGNED = [G3D4, G6D2]`
  carve on tier boundaries so `group_of` is a **pure shift, never a branch**.
  The ClassView holds every sanctioned reading at once and picks per read —
  which is why re-reading a body under another shape (§5.2) and hopping the
  graph under a mask (§5.3) are the SAME cheap operation. `u8:u8` stays two
  bytes, never widened.
- Thinking styles — **⊘ the "rung 4 only" confinement is V1-HISTORICAL, not
  a live ruling** (operator, 2026-08-30: *"a stale inheritance … it was never
  a recent ruling"*). This document's first filing repeated
  `persona-vs-rung-ladder.md`'s row-4 shelf as if current; regraded in place.
  The ruled direction now: **(a)** rung 0–9 AWARENESS is scheduled **in
  parallel** as meta-aware thinking orchestration through
  `lance-graph-supervisor`'s kanban surface — `kanban_actor.rs`'s
  `PhaseCensus` is exactly the built primitive ("the read-only fleet
  visibility surface … one `&self` pass, not 64k RPCs"; `observe`/`record`/
  `at_rest`, plus `mul_target` as the pure gate-lowering) — the rung above
  watching the kanban of the layer below, informing it without rewriting it;
  **(b)** styles CONVERGE onto the execution membrane of §5 — mask-ALU
  cheapness for the first level, `ogar-loco`/`ogar-r2il` composition for
  depth — i.e. a style is a program over the atom/mask substrate, applicable
  at ANY rung, not a macro shelf at one. The ladder doc's content taxonomy
  (144 verbs at 2, 34 recipes at 3, `StyleFamily`(12)) remains a valid
  CONTENT census; what is retired is reading it as a *confinement* of styles
  to rung 4. O2 (style → recipe selection) survives, generalized: the edge
  runs from any rung's orchestration into the recipe layer.

### 5.6 Execution blockers, measured (W0/W5 SOURCE FACT rows)

Verified 2026-08-30, reconciling a parallel audit — these gate any
"nanosecond method composition" claim and precede any atom freeze:

1. **`VAR_SET`/`VAR_CHANGE` semantics mismatch (documented as genuine in the
   source itself).** `ogar-loco`'s `pushes_result` declares both as
   result-pushing (chainable assignment); no `DROP`/`POP` primitive exists;
   `statement_bounds` therefore correctly REFUSES ordinary "set a; set b"
   bodies (`DanglingOperands`) — while the shipped interpreter executes them
   as void (`ogar-loco/examples/interpret_probe.rs:36-50`: "this is not a
   probe bug; it is a genuine …"). Canonical choice (void, or explicit DROP)
   is stored-byte-ABI-adjacent → operator decision.
2. **The recipe ladder is not yet executable.** `ladder_program() ->
   Vec<FnIndex>`, not an `ogar_loco::Program`
   (`lance-graph-ogar/src/recipe_vocab.rs:126-128`); `domain_pushes_result`
   returns `Some(true)` for EVERY recipe (`:157-159`), so an unconsumed
   34-recipe ladder leaves 34 values and cannot statement-segment; no
   canonical dispatcher executes the 34.
3. **`ogar-r2il::CallMask` is `Box<[u64]>`** (`lib.rs:369-381`, heap per
   mask); the 180-call maximum fits `[u64; 3]` inline — the allocation-free
   form is the cheap prerequisite. Its shape guard (a mask carries its
   `LaneShape`; `project` takes both) is the SEED of the typed-mask-domain
   law: masks over different domains must never silently compose merely
   because both are bitsets.
4. **Atoms whose producers/consumers are not yet wired:** the contract
   counterfactual carries 4× `todo!` on its v3 poll/cancel path
   (`counterfactual.rs:256,268,275,353`); proprioception is an `[f32; 11]`
   observation classifier (`proprioception.rs:14`); `Spd2` + the 256-entry
   `SigmaCodebook` exist in-contract (`sigma_propagation.rs`) with consumers
   still planned. Per §7.2 of the convergence plan: callable-atom status
   FOLLOWS real wiring.
5. **Receipt provenance gap:** `RevisionDelta` keeps RESULT-side masks
   (preserved/introduced/withdrawn/revised/roots/resistance/contradictions/
   affected — `revision.rs:290-303`) but not the input-mask stack + operator
   lineage that produced them. A method receipt needs both: a collapsed
   effective mask for speed, the full mask-operation lineage for commit —
   Revision must not flatten provenance.

Out of local scope, stated rather than asserted: the Java-side lens
granularity and its measured columnar multiplier live in the separate Java
repo (not in this census's three-repo read); its mask/hop/reduction surface
is cited here only as far as `ARC-A-SOURCE-ARCHAEOLOGY.md` already banked it.

---

## §6 — The fences, restated with sources

**#1077** (`F-RLR-1..8` + constitutional): no second thinking DSL; no new
carrier before `ogar_loco` is proven insufficient; a scheduler may
prefetch/queue/wake but never decide truth, causality, independence, band
promotion or revision acceptance (*prefetch ≠ belief*); no new trust scalar;
no `ClassView`/`VocabularyRegistry` collapse; ambiguity/entropy/parallax never
terminate cognition — only the Rubicon boundary owns stop/commit/veto.

**#1078** (STOPs + §7.2): no new rung tenant/rail/classid before
alpha/view/session insufficiency is proven; no new provenance bits before
W-slot/receipt semantics are exhausted; no Tarski↔cognitive-rung collapse; no
Shannon/NARS/EWA/provenance/Band trust scalar; "alpha presence never mints
evidence" *(⊘ operator-regraded 2026-08-30 as a bulldozing simplification —
see the axis-typed replacement in §4; the kernel that survives: a visit
never raises object-level `(f,c)`, transfer between axes only through typed
inference)*; counterfactual elimination never becomes empirical observation;
domain vocabulary never leaks into this generic layer; Revision/Rubicon
unskippable. Rule of thumb: **"make the mechanism callable; make the law
unskippable."**

---

## §7 — What this census hands W0

1. `TD-DEEPNSM-V2-BELIEF-DUP` as a `SOURCE FACT` row (+ the two 2026-08-30
   measurements: external consumers = 0; `admit_derived` divergence).
2. The consumer-taxonomy ↔ `Frontier`/`GapKind` correspondence, incl. the
   exposed-but-unconsumed `abductive_frontier` and the domain prior gate as
   the one mechanism with no upstream equivalent (W2 raw material).
3. The alpha gap: a live same-address overlay on the consumer side whose
   REASONING path never claims into it (W6's motivation, W1's subject).
4. The stale-line correction for #1077 §1 piece D (`RowFocusMask` shipped).
5. A worked `F-ARW-0` warning: a name-based single-repo grep produced a false
   absence claim about the consumer overlay within THIS session — census by
   semantics, not by type name.
6. §5.6's five execution blockers as W5 rows — with the convergence direction
   they imply: unify at the EXECUTION-AND-RECEIPT layer (typed mask domains,
   one bulk native evaluation, one receipt carrying mask lineage +
   premises + the Fusion/Counterfactual/Revision verdict), never by packing
   orthogonal cognitive coordinates into one scalar — which is the same law
   the STOP list already carries, applied to the method surface.
7. A naming row: `EpisodicWitness64` is a queued design, not a symbol
   (`soa_view.rs:272`) — any W3 archaeology that greps for it finds comments,
   not code; the shipped names are `ReasoningWitness64` / `EpisodicEdges64` /
   `WitnessTable` / `CausalWitnessFacet`.
