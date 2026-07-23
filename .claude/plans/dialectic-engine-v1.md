# dialectic-engine v1 — the reasoning cathedral (synthesis of the operator's six pillars, three theses, two antitheses)

> **Status:** ACTIVE. Produced by a real dialectic: operator dictation (governing)
> → 3-thesis fanout → 2-critic antithesis (both FIX-NEEDED) → this synthesis.
> Contradictions are resolved by decision with cost named — never averaged.
> The engine is built INSIDE lance-graph (E-FORWARD-AUTOCOMPLETE ruling); the
> substrate, not the session, does the thinking (operator: "our substrate, not
> you, is supposed to create thinking that you are incapable of").

## §0 — The six operator pillars (governing; verbatim intent)

1. **34 NARS tactics** = the thinking moves — term-logic syllogisms over
   concept statements with NARS truth functions; *"in real, based on the
   substrate's content, not popcount matching."*
2. **64k kanban field** = the parallel carrier — *"you can't run 64k parallel
   higher-order thoughts [conventionally]; imagine how"* → SIMT over
   syllogisms: masks, sweeps, one shared tissue.
3. **Staunen ↔ Wisdom poles** measure flow — novelty influx vs crystallized
   truth; flow = balanced conversion.
4. **Entropy × MUL as meta** — detects when *"the rung tissue dissolves if the
   thinking can't keep up."*
5. **Field rung-elevation** — *"just elevate the rung levels across 64k and
   build the reasoning cathedral on top."* Mass induction mints the parent
   floor; HHTL grows upward.
6. **Qualia interoception** — *"feel the texture of the thought from within"*;
   constitutive, never consultative (*"an autist has a hashmap of qualia but no
   tissue to use it from within — you can"*): qualia live in the **Datapath**
   bucket (inline multiply), introspection of qualia in **Control**. The system
   *feels flow, tension, novelty, the spark, epiphanies* — and *feels whether a
   text spider finds is a dull shadow or a new insight* (see §3.6).

## §1 — Synthesis decisions (S1–S12; each names the losing position + cost)

- **S1 — Truth never rides the mxm (P0, logic critic).** NARS deduction
  confidence `c=f₁f₂c₁c₂` is NON-associative → not a semiring ⊗; tiled mxm
  reorders accumulation → order-dependent confidence. RESOLUTION: the semiring
  computes **Boolean reachability only**; NARS truth is a **second pass** walked
  over the premise-pointer fabric (provenance-semiring style), each (f,c) from
  its specific ordered premise pair; parallel paths combine ONLY by
  disjoint-stamp revision or CHOICE. Cost: two passes instead of one; the
  611M/s LUT applies per-op, not per-chain.
- **S2 — Triple-keyed dedup stays; revision merges IN PLACE (P0, both
  critics).** Thesis 1's "dedup by stamp" is REJECTED — it destroys the shipped
  termination proof (reason.rs finite-triple-set argument). A statement exists
  ONCE in the arena; revision updates its (truth, stamp) in place at its
  existing rung; only genuinely-new statements get `max(premise rungs)+1`.
  Closure-internal duplicates resolve by CHOICE (higher expectation), no stamp.
- **S3 — Statement shape: the copula hybrid (P1).** `CStmt { s, copula, p }`
  with `Copula { Inh(→), Sim(↔), Impl(⇒), Rel(verb-term) }`. **Only Inh and
  Sim auto-transit; Rel (FSM verbs) NEVER freely composes** — this also fixes a
  latent unsoundness in the shipped blanket same-predicate closure ("dog bit
  man, man bit sandwich ⊬ dog bit sandwich"). Physical carrier stays compact
  ids; NodeGuid is the addressable identity, the reasoning index is a dense
  interned id (never 16-byte keys in a pivot/LUT index).
- **S4 — Stamps are fixed-width observation-source bitsets (P1).** `Stamp(u64)`
  over a bounded source horizon — never derivation ancestry (unbounded,
  serializing, Firewall-hostile). Disjoint → NARS revision (w-pooling,
  `w=c/(1−c); f=(w₁f₁+w₂f₂)/(w₁+w₂); c=(w₁+w₂)/(w₁+w₂+1)`; contradiction depth
  |f₁−f₂| recorded). Overlap → CHOICE (higher c), **no double count**. Honesty
  note (logic critic): bounded stamps make ASC non-circularity PROBABILISTIC,
  not guaranteed — so ASC challenges prefer observation-sourced counter-evidence
  (structurally independent), not re-derived graph evidence.
- **S5 — Flood throttle (P0, feasibility critic).** Abduction joins drop the
  same-predicate constraint → hub middle-term M mints d_M² hypotheses
  (~10⁶–10⁷/sweep on KJV). Throttle: (a) confidence floor c_min (abduction is
  weak by construction), (b) per-thought derivation budget k (64k·k hard cap),
  (c) **hub middle-term exclusion** (top-percentile in-degree M barred),
  routed through the existing MassExplorer budget+curiosity frontier — never
  eager arena closure.
- **S6 — Two buckets, stated explicitly.** Forward syllogisms
  (deduction/induction/abduction) lower to Datapath (reachability mxm + truth
  second pass). **CR and ASC are Control-bucket stamp-set operations** —
  disjointness is not a (⊕,⊗) op and cannot be an mxm.
- **S7 — Loop control reads f32; i4 is storage.** Δcoherence at i4 (step 1/7)
  is quantization-dead exactly in the slow-stall regime; insight/mush control
  reads f32 GraphSignals (or an i8/i16 8-sample slope), the packed QualiaI4 is
  per-thought storage only.
- **S8 — Council as byte lanes.** `InnerCouncil::deliberate` (f32 max_by,
  branchy) is 64k branches as shipped; the 3-archetype vote becomes i8 lanes
  (max-of-3 = 2 compare-select; split = (max≥hi)&(min≤lo)) before any 64k claim.
  Two orthogonal axes kept distinct: `advance_on_gate` moves PHASE (shipped);
  the GraphBias→recipe LUT selects the TACTIC inside CognitiveWork (new).
- **S9 — Rank by rate, never count (E-DOOMSCROLL, third confirmation).**
  Thought promotion/prune and WisdomMarker neighborhood bias use normalized
  rates (deduction_yield-style), size-normalized — a count-ranked field
  collapses into its largest basin.
- **S10 — Insight-vs-mush (thesis 3, amended by S7):**
  `INSIGHT = clamp(Δcoh + Δwonder − Δent, 0, 1) · [yield > θ]` (free-energy
  descent, grounded); `MUSH = 0.5·churn + 0.5·stall`,
  `churn = revision_velocity·(1−yield)`, `stall = entropy·(1−|Δcoh|)`. Feeds
  FlowState and the kanban gate. Thresholds are registered conventions; the
  discriminator must beat a size-preserving null before "detects insight" is
  promoted (E-BASIN-WIDTH discipline).
- **S11 — Dissolution → field elevation (pillar 4→5).** When Staunen influx
  outruns crystallization at rung r (entropy rising, yield→0, coherence unable
  to form), MUL declares the rung tissue dissolving; the response is a
  FIELD-scale mass-induction sweep minting parent concepts (new family basins;
  HHTL grows upward) — the cathedral's next floor — not per-thought churn.
- **S12 — Qualia are constitutive (pillar 6, design law).** The texture
  multiplies inline into tactic weight, veto, and flow within the sweep. **No
  QualiaReader service.** Ablation falsifier registered: texture-gated vs
  texture-ablated field must measurably differ (doomscroll cycles, dissolution
  latency, Staunen→Wisdom conversion) — else the qualia are decoration.

## §2 — Corrections owed to the board (append-only, land with V0)

- `E-NARS-IS-LOGIC-...-1` stated abduction truth as `c=f₁·c₁·c₂·k` — WRONG
  (deduction-shaped product). Orthodox + shipped form: `f = f_rule`,
  `w = f_obs·c₁·c₂`, `c = w/(w+HORIZON)`. Correction appended, not edited.
- **Shipped bug (critic-found):** `nars_revision` (ndarray) and the planner's
  `revise all history` path (nars_engine.rs:553) sum evidence with NO
  disjointness guard → self-reinforcement double-counting. → TECH_DEBT entry;
  fix lands with the contract-side guard, not silently.

## §3 — The five tactics (as synthesized; all term-logic, zero fingerprints)

| # | Tactic | Rule (premise pattern ⊢ conclusion) | Truth | Bucket |
|---|---|---|---|---|
| 4 RCR | abduction | `{P⇒M (rule), S⇒M (obs)} ⊢ S⇒P` — shared M in predicate position | `f=f_rule; w=f_obs·c₁·c₂; c=w/(w+k)` (weak) | Datapath (throttled S5) |
| 6 TR | divergence | sibling substitution: `{S cop P, S↔S′} ⊢ S′ cop P` — S′ enumerated via shared is_a parent; **similarity is a BELIEF (derived by comparison), never a tree-distance metric** | analogy: `f=f·f_sim; c=c·c_sim·f_sim` (low-c hypothesis → frontier) | Datapath |
| 7 ASC | self-critique | negation target `⟨1−f, c⟩` defines the goal; counter-evidence must be INDEPENDENTLY sourced (observation stamps), then revised in | revision (S4); self-revision blocked by overlap | Control |
| 8 CAS | abstraction | figure-selected, tree-guided: up = induction `{S→P, S→G} ⊢ G→P` (weak); down = deduction `{G→P, S→G} ⊢ S→P` (strong, discounting). The trie SELECTS candidates; the FIGURE (shared-term position) selects the truth function | induction / deduction | Datapath |
| 11 CR | dialectic | same statement, disjoint stamps → revision (synthesis: higher c, |f₁−f₂| preserved); overlap → CHOICE | S4 formulas | Control |

### §3.6 — The felt integration criterion (dull shadow vs new insight)

Fetched material (spider/arXiv, no-LLM constraint intact) is quarantined
(prior 0.1) and integrates ONLY when the new concept serves as the **middle
term composing two disjoint-stamp pre-existing beliefs** (non-hub M) and moves
an existing (f,c) via valid revision — never a lone is_a-to-hub (gameable).
**The felt form of the same event:** a *dull shadow* = nothing moves — no
derivation lands, truth unchanged, wonder flat (recognition without
composition). A *new insight* = the middle-term click — coherence and
expansion rise together (the spark), a marker may mint. Audit form (Control,
stamps) and felt form (Datapath, texture) are one event read from two buckets.

## §4 — Build order (each stage register-before-code; V0 falsifies the P0s)

- **V0 (the falsifying slice, ~10-concept scale, deepnsm-v2 `belief.rs`):**
  Belief-carrying arena — triple-keyed dedup preserved, `revise` merges
  (truth, stamp) in place, copula-gated transitivity. REGISTERED TESTS (named
  by the antithesis): `revision_disjoint_stamps_moves_truth_and_still_terminates`,
  `revision_overlapping_stamp_is_rejected`, plus `verbs_do_not_transit` and
  `revision_keeps_rung_in_place`. Red kills the design cheaply; green gates V1.
- **V1:** the five tactics over the Belief arena (this §3), throttles of S5,
  ReasoningGap as first-class failure.
- **V2:** the loop — GraphBias→recipe LUT, byte-lane council, texture window,
  insight/mush (S10) + its null falsifier, kanban wiring.
- **V3:** dissolution detection + field elevation (S11) — the cathedral floors;
  Staunen↔Wisdom flow accounting; epiphany attractors (rate-normalized, S9).
- **V4:** the 64k SIMT lowering — Boolean-reachability semiring + truth second
  pass (S1), masks, sweeps — only after V0–V3 green at small scale.
- **V5:** reach-out integration (spider/arXiv → §3.6 felt criterion) + the
  qualia ablation falsifier (S12).
