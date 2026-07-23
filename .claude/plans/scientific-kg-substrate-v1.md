# Scientific-KG substrate — crawl → OCR → terms → reason → MUL (v1, scoping)

> **Status (§0):** PROPOSED — **scoping doc**, no code. Records the operator's
> direction (2026-07-23): *"wire spider-rs / tesseract-rs to reason about a
> scientific basis, find adjacent thinking, and build a scientific knowledge
> graph based on MUL and reasoning."* The **outward-facing crawl is BLOCKED**
> until its open decisions (§4) are resolved; the local, non-crawl deliverables
> (D-SCI-1) are buildable on a further "Go". Nothing here supersedes a shipped
> decision.
>
> **The one-line frame:** this wave is mostly a **composition of seams that
> already exist** — tesseract-rs `doc.v1`, the OGAR adapter path,
> `deepnsm-v2` (shape/ancestry/reason, shipped), the D-SRS-3 basin self-codes
> (planned), and MUL (`lance-graph-planner`). The genuinely NEW pieces are two:
> a **term/entity extractor** (the gate the finding below demands) and the
> **spider-rs crawl** (outward-facing).

---

## 1. The pipeline (what composes what)

```text
 spider-rs            tesseract-rs           deepnsm-v2                 lance-graph-planner
 (crawl, NEW)   →     doc.v1 (OCR,     →     term extract (NEW) →       MUL
 [BLOCKED §4]         EXISTS) ──OGAR──┐      FSM → SPO → shape/          (self-uncertainty,
                                      │      ancestry/reason (SHIPPED)   EXISTS)
                                      └──────────────────────────┐      ▲
                                                                 ▼      │
                                                        scientific KG ──┘ D-SRS-3 basin
                                                        (deepnsm-v2)      self-codes (PLANNED)
```

Seam-by-seam, with honest provenance:

- **Crawl → text (NEW, outward-facing).** `spider-rs` fetches scientific
  sources. This is the only genuinely external step; it is BLOCKED on §4
  (fork coordinates, robots/ToS, a licensed source whitelist). Digital-native
  papers (arXiv OA HTML/LaTeX) skip OCR entirely.
- **PDF/scan → text (EXISTS).** `tesseract-rs` already emits **`doc.v1`** (rich
  recognition: words/boxes/regions/tables) via its `OcrExecutor` /
  `decode_image` consumer surface, and its own docs route `doc.v1` **through
  OGAR adapters to `lance-graph-arm-discovery` / DeepNSM** (tesseract-rs
  `CLAUDE.md` "doc.v1 is the OPTIONAL seed a consumer feeds via OGAR to
  lance-graph-arm-discovery / DeepNSM"). We CONSUME that seam; we do not build
  an OCR path.
- **Text → entities (NEW — THE GATE).** `E-ACADEMIC-VOCAB-COLORBLIND-1`
  (2026-07-23): an academic-word vocab is structurally blind to the proper
  nouns that carry structure (0/7 KJV genealogy carriers present; the
  `begat`-forest dissolved). Scientific entities — named methods, authors,
  compounds, gene/theorem/dataset names — are exactly that omitted class. So
  the KG's quality **gates on term/entity extraction, not an academic lens.**
  This is D-SCI-1.
- **Entities → KG + reasoning (SHIPPED).** `deepnsm-v2`: FSM → SPO
  (`fsm`/`spo`), the shape detector routes each relation to its carrier
  (`shape`, `E-SHAPE-DETECTOR-MEASURED-1`), ancestry/taxonomy relocates to the
  DN radix-trie (`ancestry`), and the derivation-pointer fabric reasons over it
  (`reason`, `E-SELF-REASONING-FABRIC-1`). "is_a"/"part_of"/"cites"/"derives-from"
  scientific relations are precisely the per-predicate shapes the detector
  already classifies.
- **KG → self-uncertainty (PLANNED — D-SRS-3).** MUL wants a *"where am I
  uncertain"* signal; the D-SRS-3 basin self-codes (`self-reasoning-substrate-v1`)
  ARE that signal (basin distribution width = uncertainty). **D-SCI-4 depends on
  D-SRS-3** — the self-reasoning ladder and this wave converge at MUL.
- **Reasoning (EXISTS).** MUL (`lance-graph-planner/mul/`: Dunning-Kruger,
  trust qualia, compass, homeostasis, gate) consumes the uncertainty signal and
  drives *"adjacent thinking"* = concept adjacency (shape-similar predicates +
  basin proximity in the KG).

**Wiring direction (non-circular, confirm at §4):** `deepnsm-v2` SUPPLIES the KG
+ shape/uncertainty signals; `lance-graph-planner`/MUL CONSUMES them — the same
non-circular direction the workspace already uses (planner depends on the
substrate, never the reverse). `deepnsm-v2` must NOT gain a planner dependency.

---

## 2. Why the finding makes this tractable, not naive

The color-blindness finding is not a caveat — it is the wave's **first gate and
its cheapest de-risking**. It says: before any crawl or OCR, prove that
term/entity extraction recovers the structure an academic vocab destroys. If
D-SCI-1 fails (terms don't recover forest/dag structure that academic vocab
flattens), the whole wave is built on sand and we stop. If it passes, the
downstream seams are all already-shipped or already-designed — the risk
collapses onto the two new pieces, and one of them (the crawl) is deferred.

---

## 3. Deliverables — the ladder (probe-first, each KILL-gated)

> Gate numbers are registered pre-run per deliverable (git-ordering anti-tuning,
> as in D-SRS-1/2); the values below are the SHAPE of each gate, `X` = registered
> at authorization.

### D-SCI-1 — Term/entity extraction (LOCAL, the gate; buildable now)

Extract domain terms (capitalized/multi-word noun phrases, non-function tokens)
from a scientific corpus (arXiv OA abstracts / a public-domain science text,
never committed), build the KG with the TERM vocab, and compare to the same
corpus under the academic-20k vocab.

- **PASS gate (the inverse of `E-ACADEMIC-VOCAB-COLORBLIND-1`):** the
  term-vocab KG exposes **non-trivial forest/dag structure** (≥ `X`% of
  predicates route to RadixTrie/TriePlusEscalate/MaterializedFabric with
  amortization ≥ `2×`) that the academic-vocab KG **flattens** (routes to
  EdgeTable/Cyclic). The graph's shape must track the corpus's real entities,
  not the lens.
- **KILL:** term-vocab and academic-vocab KGs have statistically
  indistinguishable shape distributions — entity extraction bought nothing, the
  substrate can't see scientific structure, STOP the wave.

### D-SCI-2 — OCR ingest via the tesseract-rs `doc.v1` seam (cross-repo)

A scanned/PDF paper → tesseract-rs `doc.v1` → (OGAR adapter) → term extract →
`deepnsm-v2` KG. Consumes the EXISTING seam; builds no OCR.

- **PASS gate:** a paper round-trips image → `doc.v1` → terms → KG with ≥ `X`%
  of the digital-native (HTML/LaTeX) KG's entities recovered on the same paper
  (OCR noise floor registered against the digital source as oracle).
- **KILL:** OCR noise destroys entity recovery below the floor — route those
  sources to digital-native only.

### D-SCI-3 — The crawl (spider-rs) — ⚠ OUTWARD-FACING, BLOCKED on §4

Fetch a **licensed, whitelisted** scientific source set (e.g. arXiv OA / PMC OA
subset) under robots/ToS compliance, rate-limited, into the D-SCI-1/2 pipeline.
**MUL-steered, not blind** (per D-SCI-4): the crawl frontier is scored by the
KG's curiosity/epiphany signal — the web-side twin of the KG frontier. The plug
point (mapped in `AdaWorldAPI/spider`, @ /workspace/spider): replace/wrap
`spider::features::automation::prefilter_urls → classify_urls` (the LLM
URL-relevance gate, wired at 9 crawl sites) so the MUL score, not an LLM prompt,
decides "crawl HERE next"; `spider_agent_types::MapResult{relevance,
suggested_next}` is the scored-frontier DTO. Compliance is OPT-IN in spider
(robots default OFF; SSRF guard only behind the `firewall` feature) — a
compliant crawl MUST enable `respect_robots_txt` + the `firewall` feature
explicitly. spider's HTML harvest and tesseract OCR both emit the OGAR `DocIr`
(via `content_sha256`) — one ingest shape for crawl and scan.

- **PASS gate:** N papers crawled within the licensed whitelist, robots +
  firewall features ON, 0 robots/ToS violations, the MUL-scored frontier
  measurably prioritizing high-curiosity targets, feeding a KG whose shape
  census matches D-SCI-1's per-domain expectation.
- **KILL / STOP:** any source outside the whitelist, any robots/ToS breach —
  hard stop, not a warning.
- **Not authorized by this doc.** Requires the §4 decisions + explicit operator
  green-light. Never crawl on inference.

> **⊘ NO-LLM CONSTRAINT (operator-ruled 2026-07-23, append-only — binds any
> future D-SCI-3 implementation).** *"We don't use LLM — so you can't have an
> LLM or filter, except if you do it right."* Spider's `classify_urls`
> `relevance_prompt` LLM gate is FORBIDDEN as the frontier filter — it would
> smuggle a transformer oracle back into the loop DeepNSM exists to eliminate.
> "Doing it right" = the filter is computed from the substrate's own
> information-bearing machinery, deterministically:
> 1. **Open questions are enumerated, not generated:** `(s,p,?)` slots with
>    contradiction density > 0, thin-evidence beliefs (NARS `n/(n+1)` low),
>    `FailureTicket` (F>0.8) sites, high-`curiosity_mul` frontier edges — the
>    graph's own gaps ARE the query list.
> 2. **Relevance = codebook resonance, not a prompt:** candidate tokens
>    (anchor text / URL terms / snippet) scored against open-question terms via
>    the trained Cam96/palette distance + FSM→SPO (the <10µs/sentence path),
>    weighted by epiphany residue (`WisdomMarker` attractors) near the
>    question's basin.
> 3. **Dead ends are MEASURED after ingest, not judged before:** did NARS
>    evidence counts rise, did a contradiction resolve, did forward NOVELTY
>    arrive (the G-SRS3b evidence-composite instrument)? Zero delta across N
>    pages ⇒ prune the branch. Arithmetic, not opinion.
> 4. Even the FailureTicket tail escalates to the open-question queue — never
>    to an LLM call. Cross-ref: `E-MUL-EXPLORATION-GATEWAY-1` (the plug point),
>    `self-reasoning-substrate-v1` §D-SRS-3b (the evidence-composite
>    instrument this frontier consumes), `E-BASIN-WIDTH-IS-N-ARTIFACT-1`
>    (why geometry-only scoring is GIGO).

### D-SCI-4 — MUL as the exploration gateway (the operator's steer; ~90% scaffolded)

**Reframed (2026-07-23, operator: "use MUL as an exploration gateway for
insights following epiphanies"; grounded in `E-MUL-EXPLORATION-GATEWAY-1`).**
This is NOT a from-scratch build — a 3-agent read-only exploration found the
gateway is ~90% already scaffolded in `lance-graph-contract`. What exists:

- **KG edge frontier (curiosity-ranked):** `contract::exploration::MassExplorer`
  — `FrontierEdge::curiosity() = novelty × uncertainty`, `next_frontier_edge()`
  returns the top = "explore HERE next."
- **KG-uncertainty IN:** `sensorium::GraphSignals{truth_entropy,
  contradiction_rate, …}` → `suggested_bias() → GraphBias::Explore`
  (substrate-supplied, non-circular).
- **Epiphany trigger:** `free_energy::Resolution::{Commit(F<0.2),
  Epiphany(ΔF<0.05), FailureTicket(F>0.8)}` (the Click) +
  `escalation::EpiphanyDetector` (sim > baseline×1.5), whose `WisdomMarker`
  residue decays to a **floor 0.1, never zero** — past epiphanies are
  PERMANENT ATTRACTORS ("following epiphanies", literally).
- **Breadth/depth/settle verdict:** `escalation::InnerCouncil::from_signals →
  CollapseHint::{Fanout, RungElevate, Flow}`.
- **MUL curiosity:** `mul/compass::CompassNeedles.curiosity=(1−competence)×0.5`;
  `CompassDecision::Exploratory`.

**The one honest gap = the wire.** Nothing feeds `MulAssessment`/`GraphSignals`
into the frontier ordering; `FrontierEdge::curiosity()` is MUL-blind. The
deliverable is `FrontierEdge::curiosity_mul(&assessment, &signals)` = the
existing `novelty × uncertainty` scalar × the compass curiosity needle × an
epiphany-residue insight-density term. No new substrate type
(`GraphSignals` + `NarsTruth.confidence` are the IN; frontier sort key +
`CollapseHint` are the OUT). An explore verdict that crosses to the crawler
lands on the CONTRACT side (extend `CompassDecision`/`GateDecision`), never the
planner — non-circular.

- **PASS gate:** on a held-out split, the MUL-weighted frontier ordering
  (`curiosity_mul`) beats the MUL-blind `curiosity()` at reaching
  under-supported / high-surprise regions first (≥ `X` improvement in
  time-to-first-epiphany or coverage-of-hot-edges), AND adjacent-concept
  surfacing (shape-similar predicates + basin proximity) recovers ≥ `X`% of a
  human-curated related-concepts set.
- **KILL:** `curiosity_mul` is indistinguishable from `curiosity()` (MUL adds
  nothing) OR the epiphany-attractor term makes it loop on settled regions.
- **Prerequisite:** the D-SRS-3 basin self-codes are the ideal IN, but
  `NarsTruth.confidence` + `GraphSignals.truth_entropy` are shippable surrogates
  — so a first `curiosity_mul` is buildable WITHOUT D-SRS-3 (a `GraphSignals`
  self-code width field is the later refinement). Non-circular wiring confirmed.

> **Pre-run registration — D-SCI-4a `curiosity_mul` + qualia texture gestalt
> (2026-07-23, registered BEFORE the code; anti-tuning). Operator steer folded
> in: "wire qualia as a texture gestalt awareness."** The wire's output is NOT a
> bare scalar — it is a `TextureGestalt { texture: QualiaVector (17D felt
> quality), magnitude: f32 (frontier ranking key) }`. Qualia IS the awareness;
> the magnitude is a projection of it. All in `lance-graph-contract::exploration`
> (zero-dep, non-circular — reads `MulAssessment` + `GraphSignals`, both
> contract types). Gate is STRUCTURAL (directional inequalities, no tunable
> threshold), registered as:
> - **G-CM-1 NEUTRAL IDENTITY (the anti-vacuity falsifier):** with a NEUTRAL
>   `MulAssessment` (Calibrated trust, `DkPosition::Plateau`, `FlowState::Flow`,
>   `free_will_modifier = 1.0`) and `GraphSignals::default()` (all zero),
>   `curiosity_mul(neutral, default) == curiosity()` (within 1e-6). The wire adds
>   NOTHING when MUL has nothing to say — MUL earns its weight or is inert.
> - **G-CM-2 STAUNEN BOOST:** raising `truth_entropy` / `contradiction_rate`
>   (graph-wide surprise = the Staunen/wonder texture) STRICTLY INCREASES
>   `curiosity_mul` above `curiosity()` on the same edge. Surprise pulls
>   exploration.
> - **G-CM-3 HUMILITY GATE:** `DkPosition::MountStupid` + `TrustTexture::Overconfident`
>   STRICTLY DECREASES `curiosity_mul` below `curiosity()` (the doc's 0.3
>   humility discount) — the graph does NOT explore over-confidently.
> - **G-CM-4 GROUND GATE (never a hard stop):** less-solid ground (Uncertain /
>   Overconfident texture, or MountStupid) dampens the magnitude vs Calibrated,
>   but never to zero (floor > 0) — unsafe ground is discouraged, not forbidden
>   (reversibility, not paralysis).
> - **G-CM-5 TEXTURE FIDELITY:** the emitted `QualiaVector` carries
>   `entropy = f(edge-uncertainty, truth_entropy)`, `expansion = novelty`,
>   `tension = contradiction_rate`, `groundedness = f(trust texture)`, and the
>   derived `wonder = √(coherence·expansion)` (via `qualia::qualia_to_state`)
>   rises with Staunen — the gestalt is a faithful felt-reading, not decoration.
> - **KILL = any of:** neutral identity fails (the wire is not inert when it
>   should be — it is either vacuous or arbitrary); staunen does not boost;
>   MountStupid+Overconfident does not dampen; ground gate hits zero; or the
>   texture axes do not track their sources. Report verbatim; never relax.
> - **Proof surface:** deterministic `#[test]`s in `exploration.rs` (no corpus,
>   no crawl). The held-out frontier-ordering PASS gate above (beats MUL-blind on
>   time-to-first-epiphany) is D-SCI-4b, a later corpus probe; D-SCI-4a is the
>   structural wire + its gestalt semantics.
>
> **⊘ CORRECTION to G-CM-5 (2026-07-23, append-only; the registered line above
> stands as written).** G-CM-5 registered *"the derived `wonder =
> √(coherence·expansion)` … rises with Staunen."* The implementation found this
> is **ill-posed**: the qualia `wonder` (`qualia::qualia_to_state` drive-axis 9)
> is `√(coherence·expansion)` = **coherent-NOVELTY** wonder — it reads only trust
> (coherence) and edge novelty (expansion), NOT `truth_entropy`/`contradiction`.
> **Staunen (surprise-wonder) is a DIFFERENT quale**, carried by `arousal[0]`,
> `entropy[8]`, `tension[2]`. So the as-built G-CM-5 asserts Staunen tracks
> arousal/entropy/tension (which it does), and a paired
> `wonder_is_invariant_to_staunen` test guards the decoupling. Not a relaxation
> — a real two-wonders distinction (Csikszentmihalyi coherent-novelty vs
> Staunen surprise), recorded rather than fudged. Adversarially verified: the
> decoupling is mathematically correct, all 5 gates hold with margin, the
> `.max(0.0)` magnitude is NaN-safe (the sort cannot panic — strictly safer than
> the pre-existing `next_frontier_edge`), non-circular, SIMD-repr contract
> untouched.

---

## 4. Open decisions — BLOCKERS before the outward-facing build

1. **spider-rs coordinates — RESOLVED (2026-07-23).** `AdaWorldAPI/spider`
   (cloned @ /workspace/spider, HEAD `046c439`) — a 10-crate workspace with an
   agent layer (`spider_agent`, `spider_mcp`, `spider_doc_ir`) beyond stock
   spider-rs. Wire via the fork per P0. The MUL steering hook + doc-IR
   convergence are mapped (`E-MUL-EXPLORATION-GATEWAY-1`).
2. **Crawl scope + compliance.** Which sources, under which license (arXiv OA /
   PMC OA have explicit terms)? robots.txt honored, rate-limited, redirects off,
   size caps — the SSRF/politeness posture tesseract-ocr-web already models.
   Crawling is outward-facing and hard to reverse; it needs explicit operator
   scope, not agent inference.
3. **Corpus licensing for committed artifacts.** Like the AVL vocab and the KJV
   text, crawled corpora are **local-only, never committed**; only derived
   graph statistics/codes ship. Confirm no corpus text enters git.
4. **MUL wiring direction.** Confirm `deepnsm-v2` supplies, `lance-graph-planner`
   consumes (non-circular) — via a contract seam or the p64 convergence pattern,
   never a `deepnsm-v2 → planner` dependency.
5. **Where the scientific KG persists.** The SoA SPOG tenant follow-on
   (`Representation::graph_id()` = the G lane, `E-SHAPE-DETECTOR-MEASURED-1`) —
   or in-memory only for the probes.

---

## 5. Cross-refs

- `E-ACADEMIC-VOCAB-COLORBLIND-1` — the entity-extraction gate (D-SCI-1's basis).
- `E-SHAPE-DETECTOR-MEASURED-1` / `E-SELF-REASONING-FABRIC-1` — the shipped
  substrate (shape/ancestry/reason).
- `self-reasoning-substrate-v1` D-SRS-3 — the basin self-codes D-SCI-4 depends on.
- `literature-probe-ladder-v1` — the sibling genre-falsifier program;
  `genre_shapes.rs` is shared apparatus.
- tesseract-rs `CLAUDE.md` (doc.v1 / OcrExecutor / the OGAR → arm-discovery path),
  OGAR consumer-preflight — the OCR seam this wave consumes.
- `lance-graph-planner/mul/` — the MUL consumer.
