# 34 Tactics vs Ada — authoritative mapping + the 36 NARS styles

> **READ BY:** truth-architect; anyone mapping reasoning tactics → substrate.
> **Source (AUTHORITATIVE, external):** `AdaWorldAPI/ada-consciousness` `docs/34_TACTICS_VS_ADA.md` @707264b (provided by the user; NOT in the MCP allowlist — reference-only, never a code-port target). Supersedes my earlier *reconstruction* of the 34 in `spo-2cubed-list-coverage.md`.
> **Date:** 2026-05-27.

## The category gap (the thesis)

The 34 are **prompting strategies** (how an LLM thinks within one context window). Ada is **substrate** (1,322 py + 534 docs operating beneath/across sessions). So the question isn't "is the tactic 2³-covered" — it's "is the tactic *implemented as substrate*." Per the source, **21 of 34 are implemented** (10 structural + 11 embedded), **13 are not** (mostly because subsumed by a structural mechanism, not absent).

## The 34 — authoritative status (condensed)

- **Structurally implemented (10):** #3 SMAD→`triune_council`/`advocatus_diaboli`/`causal_quorum`; #4 RCR→`do_calculus`/`pc_algorithm` (full Pearl); #5 TCP→`inner_dialogue` Litter Box/`schrodinger`; #7 ASC→`advocatus_diaboli`+mirror neurons; #10 MCP→`metacog`/`recursive_self_model` (Fleming&Dolan, Rosenthal HOT); #12 TCA→`crystal/markov_7d`/`causal_ladder`; #18 CWS→Upstash Redis/`ada-hive` (actual persistence); #25 HPM→10000D VSA (literal HDC); #31 ICR→`do_calculus.counterfactual()`+ghosts; #34 HKF→`dream_engine`+VSA BIND.
- **Natively embedded (11):** #1 RTE, #2 HTD→`rungs`+CLAM tree; #6 TRR→`schrodinger`/Markov; #8 CAS→HDR cascade INT1/4/8/32; #11 ICR-contradiction→`advocatus_diaboli`/Rung4 coherence; #13 CDT→VSA BUNDLE→SIMILARITY; #20 TCF→CAKES 7 search algos; #21 SSR; #26 CUR→CRP percentiles; #27 MPC→panCAKES XOR-diff; #30 SPP→`active_inference` dual-hemisphere System-1/2.
- **Not implemented / subsumed (13):** #9,14,15,16,17,19,22,23,24,28,29,32,33 — each subsumed by a structural mechanism (e.g. #24 ZCF = VSA BIND; #17 CDI = Advocatus COMFORT; #15 LSI = explicit 16K vectors, not latent).

> So vs my reconstruction: my **bucket** tags (datapath/control/gate) hold, and my **2³** tags hold (only #4 RCR + #31 ICR are causal-lattice). What I got wrong was implying "Not covered by 2³" ≈ "not handled" — Ada *implements* 21 of them as substrate; 2³ is just the causal slice.

## THE 36 NARS STYLES (the "36 nars" — finally the real list)

Per the source (`DTO/ada_10k.py [116:152]`, "36 NARS thinking styles as VSA dimensions"). Named in the doc (≈20 of 36; rest in `…`):

`DECOMPOSE, SEQUENCE, PARALLEL, HIERARCHIZE, SPIRAL, OSCILLATE, BRANCH, CONVERGE, DIALECTIC, REFRAME, HOLD_PARADOX, STEELMAN, TRACE_BACK, PROJECT_FORWARD, COUNTERFACTUAL, ANALOGIZE, ABSTRACT, INSTANTIATE, COMPRESS, EXPAND, …`

### Dichotomy method applied (your rule)

**Confirmed dichotomic pairs (both poles present → one signed lane each):**

| − pole | + pole |
|---|---|
| SEQUENCE | PARALLEL |
| CONVERGE | BRANCH |
| TRACE_BACK | PROJECT_FORWARD (retrodiction ↔ prediction) |
| INSTANTIATE | ABSTRACT |
| COMPRESS | EXPAND |

**Self-bipolar (one op already spans both poles — encode as a cyclic/holding lane, not ±):**
- `OSCILLATE` (cycles between poles), `DIALECTIC` (thesis↔antithesis→synth), `HOLD_PARADOX` (= "contradictions preserved, not resolved" — The Click, verbatim).

**Non-dichotomic → opposite found+evaluated (confirm/correct):**
| op | proposed opposite | note |
|---|---|---|
| DECOMPOSE | COMPOSE/SYNTHESIZE | likely already in the `…` |
| HIERARCHIZE | FLATTEN | likely in the `…` |
| COUNTERFACTUAL | FACTUAL/ACTUAL | **= SPO 2³ top (0b111)** — the 2³-covered one |
| ANALOGIZE | CONTRAST | structural-map ↔ distinguish |
| REFRAME | ANCHOR | hold-frame |
| SPIRAL | DIRECT | iterative ↔ one-pass |
| STEELMAN | STRAWMAN | (or pairs with Advocatus-Diaboli challenge) |

**COUNTERFACTUAL is in the 36 and IS the 2³ apex** — the one NARS style fully covered by SPO 2³ (`SPO`=0b111). The rest are orthogonal operation axes (structural / directional / framing), confirming again: 2³ = the causal spine only.

## The 208-dim address space (the full atom budget)

Per source: **32 verbs + 36 GPT styles + 36 NARS styles + 11 presence modes + 5 archetypes + 3 TLK court + 4 affective bias + 33 TSV dims**. This is the real composite address — the 33-TSV is *one component of eight*, the 36-NARS-styles another, the 36-GPT-styles (= contract-36 personas) another. The "atoms vs styles vs persona" layering composes across these eight component-spaces.

## The 7-rung causal ladder (extends Pearl's 3)

`cognition/causal_ladder.py`: Pearl 1-3 (Association/Intervention/Counterfactual) **+ Ada 4-7**: Counterfactual-Self → Affective-CF ("what aches because it never happened") → Homeostatic-Dx ("what is this urge restoring") → Deliberate-Becoming. The 2³ SPO lattice = Pearl rungs 1-3; rungs 4-7 are Ada-native (beyond the lattice).

**Cross-ref:** `spo-2cubed-list-coverage.md` (the 2³ rubric + the 34 bucket/coverage table — reconcile its 34-table status column against this authoritative source); `EPIPHANIES.md` E-AGICHAT-DIMENSION-CONTRACT (33-TSV); `atom-basis-inventory.md`.
