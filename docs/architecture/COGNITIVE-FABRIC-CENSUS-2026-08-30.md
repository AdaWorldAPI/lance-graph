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

An instance of the banned collapse was proposed and WITHDRAWN this session
(deriving a trace's attention rung from its derivation depth) — recorded so it
is not re-proposed. #1077's finding 2 stands alongside: the "rung-4 physical
requirement" is a fossil (5 hits, 4 doc-only, 1 test); *"rung already differs
by temporal horizon, not by the right to think — what is missing is
demonstration, not permission."*

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
- `EpisodicWitness64` — column accessor deliberately deferred
  (`soa_view.rs:258-259`); the documented `WitnessTable` wording drift
  (`{mailbox_ref, spo_fact_ref}` vs "witness corpus root handle") is #1078's
  W3, and the private consumer holds a LIVE encode/decode falsifier for it.
- The 24×`i4` register — `causal_witness.rs:182` (the A9 reading of a
  12-byte register); `G24N4` is a **lane-shape name, never a second
  addressing system** (`canonical_node.rs:893,1078`).
- `CascadeShape::{G6D2, G4D3, G3D4}` over the ONE content-blind 12-byte V3
  payload (`facet.rs`): `6·2 = 4·3 = 3·4 = 12`; `ALIGNED = [G3D4, G6D2]`
  carve on tier boundaries so `group_of` is a **pure shift, never a branch**.
  The ClassView holds every sanctioned reading at once and picks per read —
  which is why re-reading a body under another shape (§5.2) and hopping the
  graph under a mask (§5.3) are the SAME cheap operation. `u8:u8` stays two
  bytes, never widened.
- Thinking styles: **rung 4 only**, per the ruled ladder
  (`.claude/v3/knowledge/persona-vs-rung-ladder.md` — 0-1 observation, 2 =
  144 verb atoms, 3 = the 34 recipes, 4 = `StyleFamily`(12) under
  *frozen × learned × exploration*). Open edge: O2 (rung-4 → rung-3
  selection), not a re-homing.

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
Shannon/NARS/EWA/provenance/Band trust scalar; **alpha presence never mints
evidence**; counterfactual elimination never becomes empirical observation;
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
