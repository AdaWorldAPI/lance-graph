# The English-Fact-Story Bifurcation — the world-spine capstone ("the holy grail")

**READ BY:** integration-lead, truth-architect, world-spine work; anyone wiring
DeepNSM → AriGraph → aerial/DOLCE → episodic.
**Status:** CONJECTURE (architecture synthesis, 2026-05-31). Assembles **shipped**
parts (each cited WIRED / CONJECTURE below) and **names the missing wires**.
End-to-end is unbuilt and unmeasured — this doc is the assembly map, not a claim
that the engine runs.
**References (does not duplicate):** `splat-codebook-aerial-wikidata-compression.md`,
`owl-dolce-hhtl-compartments-aerial-fed.md`, `agnostic-lazy-world-spine.md`,
`delta-card-addressing-integration-map.md`, `markov-triplet-query-quorum.md`;
EPIPHANIES `E-ENGLISH-BIFURCATES` (this doc's finding), `E-EPISODIC-CLOSURE`,
`E-ARM-JC-RESOLVES-BOTH-SEAMS`, the three-Markovs taxonomy.

---

## 0. The grail in one sentence

A deterministic, integer-addressed, **LLM-free** engine that reads English and
lands each clause where it belongs — **atemporal knowledge into the OWL/DOLCE
ontology (FACTS), temporal events into the episodic story-arc (STORIES)** — using
the SAME role-indexed deconstruction, on the SAME agnostic CAM-PQ substrate, with
bounded **±5..500** resolution. It is the concrete form of CLAUDE.md's AGI-as-glove
claim: *parsing a sentence and parsing a thought use the same algebraic slices.*

---

## 1. The keystone — English bifurcates (user, 2026-05-31)

The same SPO can resolve to **two destinations**, chosen by **temporality**:

```
                    DeepNSM  (English sensor — MUST stay English)
              COCA-4096 + PoS-FSM → SpoTriple{ s, p, o : 12-bit ranks }
              + SentenceStructure{ modifiers, negations, TEMPORALS }   ← parser.rs:57-66
                                     │
                          ┌──────────┴──────────┐   router reads `temporals`
               atemporal  │                     │  temporal      (WIRED field, today UNREAD)
                          ▼                     ▼
                   FACT-LANDING             STORY-ARC
            aerial 10000² splat          ±5 coreference (context_chain)
            resonance → DOLCE class   →   → EpisodicEdges64 basin (family==0)
            (similarity proposes,         → WitnessTable accumulate → prune
             CAM confirms)                    │
                          │                    │
                          ▼                    ▼
                  frozen identity      CLAM ±5  ──age──►  append-index ±500
                  (OGIT/CAM,           (within-session,   (cross-session,
                   never moves)         the only mover)    immutable pointer)
```

- **"a dog is a mammal"** → atemporal → FACT → DOLCE (frozen identity, never moves).
- **"the dog ran to the park"** → temporal → STORY → episodic arc (±5 → ±500).
- **The router signal already exists in the sensor.** DeepNSM emits
  `SentenceStructure{triples, modifiers, negations, **temporals**}`
  (`parser.rs:57-66`, WIRED). The `temporals` field *is* the fact/story switch —
  built today, read by nothing yet. The router is the smallest net-new piece.

---

## 2. The moving parts (honest inventory)

| part | role in the grail | status | home (file:line where known) |
|---|---|---|---|
| **DeepNSM** | English sensor: text → SPO + temporals | **WIRED** (102 tests) | `crates/deepnsm`; emit `parser.rs:395-413` → `spo.rs:38` |
| **`temporals` field** | the fact↔story **router signal** | **WIRED but UNREAD** | `parser.rs:57-66` |
| **10000² gaussian splat** | builds the codebook (float, OFFLINE) | **PARTIAL** — producer in ndarray; jc certifies ρ=0.9973 | `ndarray hpc::splat3d` + `jc::ewa_sandwich`, `sigma_codebook_probe` |
| **aerial** | splat→DOLCE proposer = the **literal→basin resolver** | **WIRED shape** (42 tests); end-to-end **CONJECTURE** | `lance-graph-arm-discovery`: `aerial::codebook::{TopKDistance,CodebookDistance}` |
| **OWL/DOLCE cache** | fact-landing target (frozen identity) | **WIRED** projector; #444 locality 98.6% | `aerial::ontology::{OntologyProjector,dolce_id}` |
| **`context_chain` ±5** | coreference / ambiguity resolver (replay) | **WIRED** (contract) | `contract::grammar::context_chain` (`MARKOV_RADIUS`, margin 0.1) |
| **EpisodicEdges64** | story-arc basin (`family==0` intra-basin spine) | **WIRED** (#446) | `contract::episodic_edges` |
| **WitnessTable** | accumulate-then-prune lifecycle | **WIRED** | `contract::witness_table` (`spo_fact_ref None→Some→tombstone`) |
| **±500 tier** | story-old cold tail | **CONJECTURE** (net-new) | (Lance append-index, per `E-EPISODIC-CLOSURE`) |

**Net:** ~5 tested shapes + 3 missing wires (§6) + 1 net-new router. Most of the
grail already exists and is tested in isolation; the grail is the **wiring**.

---

## 3. The three resolvers, three scales (corrects OQ-RESOLUTION-TREE)

The basin/literal grounding left "the resolution tree" open. It is **not one
mechanism** — it is three resolvers at three scales:

| scale | resolver | resolves | status |
|---|---|---|---|
| **local (±5)** | `context_chain` | coreference / pronoun / local ambiguity | **WIRED** |
| **semantic landing** | aerial 10000² splat → DOLCE | *which ontology basin a fragment belongs to* | **SHAPE wired** |
| **angle / story** | `head2head::select` | competing-arc arbitration | **WIRED** |

The **splat is the literal→basin resolver** — the piece the language↔meaning
duality was missing. literal-arc = many COCA pointers (surface, redundant);
basin-arc = the one DOLCE class (declared, exact). The splat is the resonance
field that lands a literal cluster on its basin: **similarity proposes (float,
offline, jc-certified), CAM confirms (integer, online).**

---

## 4. The bifurcation IS routing over the three lifecycle structures

It is not a new structure. `E-EPISODIC-CLOSURE` already established three
structures separated by **lifecycle**; the bifurcation is the rule that picks
**which one** an English SPO lands in, by temporality:

| English clause | destination | lifecycle structure (E-EPISODIC-CLOSURE) |
|---|---|---|
| atemporal **FACT** | DOLCE class | **frozen identity** (OGIT palette + CAM — never moves) |
| recent **STORY** | episodic arc, ±5 | **within-session CLAM** (the only thing that moves) |
| old **STORY** | episodic arc, ±500 | **cross-session append-index** (immutable pointer, pseudo-radix) |

So "±5..500" is not one window — it is the **hot CLAM (±5) aging into the cold
append-index (±500)**, exactly the two episodic structures already named.

---

## 5. The firewall (why this is safe) — the GoBD-with-Rumi guard, end to end

The whole engine holds the firewall the board already ratified
(`E-EPISODIC-CLOSURE`, the markov_soa SoC finding):

1. **Language stays UPSTREAM.** COCA / grammar templates live in DeepNSM only;
   core has **0 deepnsm dep** (the dep graph enforces it). DeepNSM scans English,
   emits SPO, stops.
2. **Both destinations are AGNOSTIC.** A DOLCE class and an `EpisodicEdges64`
   basin are **opaque handles** (`dolce_id:u8`, `EdgeRef{family,local}`,
   `spo_fact_ref:u64`) — never `rank:u16`. Storing a COCA rank as a basin witness
   would be the GoBD-with-Rumi error (a *language* lens over an *agnostic* graph).
3. **Float lives only offline.** The splat is float resonance = **discovery**; it
   runs once in jc (ρ=0.9973), emits a **frozen integer codebook**; aerial's
   online path is integer. Similarity proposes, identity addresses, **never
   swapped** (`I-VSA-IDENTITIES`).
4. **Two 4096s, kept apart.** The ~4096 story-arc basins are the independent 12-bit
   `local` space — **not** the COCA-4096 reused. Coupling basin-count to vocab
   would re-introduce language into addressing (OQ-BASIN-COUNT — confirmed distinct).

---

## 6. The missing wires (what is NOT built)

1. **DeepNSM SPO → `context_chain` ±5** (the user's "missing wire"). DeepNSM's own
   markov does **not** reach the contract-side ±5 resolver. Note the latent defect
   surfaced by grounding: DeepNSM has **two disconnected** mechanisms — a 512-bit
   `ContextWindow` (LIVE, used by `pipeline.rs:199`) and a 16384-dim `MarkovBundler`
   (**DEAD** — no production caller, `content_fp` constructed only in tests). They
   are dimensionally incompatible (OQ-ARC-PRODUCER).
2. **The temporal router** — read `temporals`, route fact-vs-story. Net-new; the
   signal is WIRED, the consumer is not.
3. **The ±5 → ±500 tier** — hot CLAM aging into the cold append-index. Net-new
   (likely the `EpisodicEdges64` cross-session column, not a bigger ring).

What is **already done for free**: the accumulate-then-prune lifecycle the
conjecture wanted ships verbatim in `WitnessTable`
(`spo_fact_ref None→Some→tombstone`); the ±5 replay-resolution ships in
`context_chain`. The grail does not need them invented — only connected.

---

## 7. First buildable slice + the promoting probe

**Slice (firewall-safe, verifiable offline):** `Trajectory::split_arcs →
(BasinArc, LiteralArc)` in deepnsm.

```rust
// crates/deepnsm/src/trajectory.rs (or arcs.rs) — zero new dep
pub struct BasinArc(pub Vec<f32>);   // the semantic spine: ONE role-superposed bundle
pub struct LiteralArc(pub Vec<u16>); // the language surface: COCA ranks (prunable later)
impl Trajectory { pub fn split_arcs(&self, literal_ranks: &[u16]) -> (BasinArc, LiteralArc); }
```

Proves: (a) "basin = one bundle, literal = many pointers" is realizable from the
existing `Trajectory.fingerprint` with no new substrate; (b) gives the **dead
`MarkovBundler` its first producer shape** (closes the no-producer gap);
(c) names the duality at the existing `disambiguator_glue` seam (today a bare
`&[f32]` + untyped candidate iterator). Stays entirely English-side — the
prune/tombstone lifecycle remains in contract `WitnessTable`.

**Probe that promotes this CONJECTURE → FINDING:** does **temporal-routed,
English-sourced** SPO landing reproduce the #444 locality result
(98.6% intra-basin, max fan-out 3) on the fact path? If yes, the bifurcation's
fact-leg addressing is real on language-derived data (not just curated ontologies
— the open #444 caveat). If no, the splat→DOLCE landing degrades to mostly-far
pointers and the fact-leg needs rework before wiring.

---

## 8. Open questions

- **OQ-ARC-PRODUCER** (blocks wire #1): dead 16384-dim `MarkovBundler` vs live
  512-bit `ContextWindow` — which is canonical? They cannot both feed the ±5 seam.
- **OQ-WINDOW-500**: tiered (±5 hot CLAM → ±500 cold append-index) vs a single
  grown radius. §4 argues tiered (it reuses the two existing episodic structures).
- **OQ-ROUTER-SIGNAL**: is `temporals` alone the router, or also FSM tense/aspect?
  A clause can be **both** (a fact asserted inside a narrative) — does it land
  twice (fact AND story), or does one win? The bifurcation may be a *fork*, not a
  *switch*.
- **OQ-BASIN-COUNT**: ~4096 story-basins = the independent 12-bit `local`, NOT the
  COCA-4096 (firewall). Confirmed distinct; keep them so.
- **OQ-GRAMMAR-TEMPLATES**: the 200–500 discoverable templates have **zero surface**
  today (one hardcoded 5-state FSM). Net-new, and orthogonal to the bifurcation —
  do not block the grail on it.

---

*This doc is the capstone assembly map. The four threads it ties —
splat/aerial/DOLCE (facts), DeepNSM (English), context_chain (±5), EpisodicEdges64
(stories) — each have their own doc above. The new claim is only the bifurcation
and its routing onto the three lifecycle structures.*

---

## Session update — 2026-05-31 (first wire shipped, commit 9af7f15)

Both gating OQs auto-resolved from source; the first slice is built, tested, pushed.

- **OQ-ARC-PRODUCER → RESOLVED: the 16384-dim role-indexed `Trajectory` is canonical** for the grail seam (not the 512-bit `ContextWindow`). It carries the `TEMPORAL` band `[9000..9200)` that IS the router, and already bridges to contract `context_chain` via `disambiguator_glue.rs:65`. The "dead" status is a *producer* gap (`MarkovBundler::push` uncalled by `pipeline.rs`), not wrong-substrate. The 512-bit ring stays DeepNSM's internal disambiguator.
- **OQ-ROUTER-SIGNAL → RESOLVED: FORK, not switch.** Every SPO relation is a fact-candidate; temporal content *adds* a story-arc ("the dog, which is a mammal, ran" → both). The temporal band is the discriminating signal; the fact leg is universal (commit-policy is downstream).
- **Shipped:** `crates/deepnsm/src/arcs.rs` — `Trajectory::{split_arcs, temporal_energy, threads_story, landing}` + `BasinArc`/`LiteralArc`/`Landing`. 5 tests; deepnsm 94+4+8+1 green; `arcs.rs` clippy-clean (pedantic+nursery). Firewall-safe (English-side, f32 upstream-only, no COCA rank reaches the agnostic graph).
- **Remaining wires (still net-new):** (1) `pipeline.rs` actually producing `Trajectory` (calling `MarkovBundler::push`); (2) the ±5→±500 tier; (3) committing routed landings into contract `EpisodicEdges64` (story) / DOLCE (fact). The promoting probe (English-SPO locality vs #444's 98.6%) is unrun.
- **New debt:** `TD-DEEPNSM-CLIPPY-195`.
