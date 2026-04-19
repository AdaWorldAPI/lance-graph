# Grammar Landscape — Three Stacks, One Target

> **READ BY:** agents working on DeepNSM extraction, grammar triangle
> integration, coreference resolution, Markov context chains, OSINT
> pipelines, or anything that touches `grammar/*` or `crystal/*` in
> `lance-graph-contract`.
>
> **Companion docs (load together):**
> - `grammar-tiered-routing.md` — 5-criterion coverage detector,
>   failure decomposition, morphology coverage baseline.
> - `linguistic-epiphanies-2026-04-19.md` — E13–E27 cross-repo
>   harvest (Chomsky isomorphism, Σ10 tiers, sigma_rosetta,
>   membrane, resonanzsiebe).
> - `cross-repo-harvest-2026-04-19.md` — H1–H14 VSA / CFG / Born-
>   rule foundation.
> - `integration-plan-grammar-crystal-arigraph.md` — E1–E12
>   shipping plan.
> - `crystal-quantum-blueprints.md` — mode duality.
> - `endgame-holographic-agi.md` — 5-layer north star.

---

## §1 The Three Grammar Stacks

The same mechanism — **text → Grammar Triangle (NSM × Causality ×
Qualia) → structured output** — exists in three independent
implementations. None of them is wired into DeepNSM. The shipping
work is not *building* grammar; it is *routing DeepNSM output
through the existing triangle* and *sharing the results across all
three consumers*.

### 1.1 Rust — `lance-graph-cognitive/src/grammar/` (1,929 LOC)

| File | LOC | Role |
|---|---|---|
| `nsm.rs` | 448 | 65 Wierzbicka semantic primes, text → prime activation vector (`NSMField`), fingerprint encoding |
| `causality.rs` | 396 | `CausalityFlow` = agent / action / patient / reason / temporality / agency / dependency_type |
| `qualia.rs` | 718 | 18-D felt-sense field |
| `triangle.rs` | 304 | `GrammarTriangle` composes all three → `Fingerprint` |
| `mod.rs` | 63 | Module root |

This is the most mature implementation and the one DeepNSM must
consume. **Shipped, untouched by this PR.**

### 1.2 Python — `bighorn/extension/agi_stack/universal_grammar/` (~5,000 LOC, 16 modules)

| File | Role |
|---|---|
| `core_types.py` | `Glyph5B` 5-byte archetype address, `Dimension` enum (18D / 64D / qHDR / HOT) |
| `verb_endpoints.py` | `VerbFamily` + `VerbMode` + `VerbRouter` + FastAPI routes + MCP tool generation |
| `calibrated_grammar.py` | Grammar with calibration data |
| `method_grammar.py` | `[method]payload` HTTP-as-ontology |
| `resonance.py` + `resonanzsiebe.py` | Resonance sieve (6-level SILENCE → SCREAM filter) |
| `scent_optimizer.py` | 1-byte scent-level grammar optimization |
| `situation_executor.py` + `_storage.py` | Situation-driven execution + persistence |
| `meta_uncertainty.py` | MUL integration |
| `exploration.py` + `invoke_router.py` | Dispatch routers |
| `markov_context.py` | Session-depth Markov (`SessionToken`, trajectory) |
| `awareness_blink.py` | Blink-unit awareness |
| `jina_integration.py` | Jina 1024-D bridge |
| `integration.py` | Cross-module glue |

The Python stack carries the architectural maturity. It's the
**reference spec** for the Rust and TypeScript implementations.

### 1.3 TypeScript — `agi-chat/src/grammar/` + `src/thinking/`

| File | LOC | Role |
|---|---|---|
| `grammar-awareness.ts` | 237 | Soft / strict awareness modes, response steering |
| `grammar-bridge.ts` | 212 | Bridge to thinking cycle |
| `extensions/langextract/grammar_triangle.py` | — | Parallel Python grammar triangle (duplicate of Rust) |

TS handles the user-facing awareness / bias steering. Consumes the
Python output where available.

### 1.4 Convergence verdict

All three implement the same Grammar Triangle — NSM × Causality ×
Qualia → fingerprint. The convergence target is **DeepNSM as the
shared extraction engine**, with the Grammar Triangle as its
structured output format, and all three language-specific stacks
downstream consumers of the same shape.

This PR's role: pipe DeepNSM FSM output through the existing Rust
`GrammarTriangle::analyze(text)` (D3 triangle bridge in the plan).

---

## §2 The Grammar Triangle (NSM × Causality × Qualia)

The Triangle composes three orthogonal linguistic signals per sentence
into a single fingerprint.

```
                NSM 65 primes
                     ▲
                     │
                 ┌───┴───┐
                 │       │
                 ▼       ▼
       Causality 2³    Qualia 18-D
```

- **NSM** (Wierzbicka 65 primes) — universal semantic atoms. I, YOU,
  THINK, WANT, BIG, BAD, BECAUSE, etc. Language-independent.
- **Causality** — `CausalityFlow` holds agent / action / patient /
  reason / temporality / agency / dependency_type. Currently 3/9
  slots; see §3 for extension.
- **Qualia** — 18-D felt-sense field (17-D experienced + 1 `classification_distance`,
  the RGB→CMYK qualia distinction from PR #205).

Grammar Triangle IS ContextCrystal at window=1 (harvest H4). Widening
the window to ±5 IS the Markov trajectory (D5).

---

## §3 TEKAMOLO Template + the 3/6 → 6/9 Slot Gap

TEKAMOLO (**T**emporal / **K**ausal / **M**odal / **L**okal, German
grammar-pedagogy mnemonic) is the adverbial-slot template that
extends SVO to cross-linguistic coverage.

### 3.1 What `CausalityFlow` currently has (3/9 TEKAMOLO slots)

```rust
pub struct CausalityFlow {
    pub agent: Option<String>,         // Subject
    pub action: Option<String>,        // Verb
    pub patient: Option<String>,       // Object
    pub reason: Option<String>,        // KAUSAL ✓
    pub temporality: f32,              // TEMPORAL (float, needs slot form)
    pub agency: f32,
    pub dependency_type: DependencyType,
}
```

### 3.2 The TEKAMOLO-completion extension (3 more slots, deferred)

```rust
    pub modal:      Option<String>,    // MODAL  (how — manner adverbials)
    pub local:      Option<String>,    // LOKAL  (where — spatial)
    pub instrument: Option<String>,    // WITH   (means / instrumental)
```

### 3.3 The thematic-role completion (3 more beyond TEKAMOLO)

Yesterday's discussion surfaced that modal / local / instrument alone
is insufficient. Full thematic-role theory adds:

```rust
    pub beneficiary: Option<String>,   // for whom (dative of benefit)
    pub goal:        Option<String>,   // to where (directional)
    pub source:      Option<String>,   // from where (ablative origin)
```

### 3.4 Optional language-specific additions

- **Path** — "through where". Finnish Prolative `-tse/-itse`;
  Turkish instrumental-path construction.
- **Purpose / Finale** — "in order to". German `zum + Inf`; Finnish
  Translative `-ksi` in purposive reading.
- **Result** — "leading to what". German `sodass`; Finnish
  consequence constructions.

### 3.5 Deferred from this PR (explicit)

The full 9-slot CausalityFlow is **deferred** (per user decision):
D0 documents it here; D2 `ticket_emit.rs` populates only the 3
existing slots; D3 triangle bridge maps only what's available.
Future PR lands the extension as a pure struct change.

### 3.6 YAML training angle (future target)

Author **200–500 TEKAMOLO templates per language** as YAML, fine-
tune a small LLM to emit slot-filled templates instead of free text.
The templates ARE the grammar constraint — output is slot-filling,
not generation, so the LLM cannot hallucinate new relations.

```yaml
- text: "The Pentagon contracted OpenAI in December because of ChatGPT's capabilities"
  template: tekamolo
  subject:  "Pentagon"
  verb:     "contracted"
  object:   "OpenAI"
  temporal: "December"
  kausal:   "ChatGPT's capabilities"
  modal:    null
  local:    null
```

---

## §4 Case Inventories Per Language (Native Terminology)

**Critical correction from yesterday's draft:** each language uses
its native case inventory, not a Latinate translation. Yesterday I
wrote Finnish "Accusative `-n/-t`" which is a Latinate transplant;
the actual Finnish object marking works differently. Every case
table below uses the language's own grammar tradition.

### 4.1 Finnish — 15 cases

**Object marking (correction from prior draft):**
- **Total object:** Nominative (plural) or Genitive `-n` (singular)
- **Partial / negated object:** Partitive `-a / -ä`
- **True Accusative:** only for personal pronouns (`minut`, `sinut`,
  `hänet`, `meidät`, `teidät`, `heidät`)

**Full 15-case inventory:**

| Case | Suffix | TEKAMOLO / role |
|---|---|---|
| Nominative | `-∅` | Subject (S) |
| Genitive | `-n` | Possessor / total-object singular |
| Partitive | `-a / -ä` | Partial / negated object |
| Accusative | `-n / -t` (personal pronouns only) | Object for pronouns only |
| Inessive | `-ssa / -ssä` | **LO** — in / inside |
| Elative | `-sta / -stä` | **LO** — from inside |
| Illative | `-Vn / -hVn / -seen` | **LO** — into |
| Adessive | `-lla / -llä` | **LO** / **MO** — at / on / by |
| Ablative | `-lta / -ltä` | **LO** — from surface |
| Allative | `-lle` | **LO** — onto / toward |
| Essive | `-na / -nä` | **MO** — as / in the state of |
| Translative | `-ksi` | **MO** — becoming |
| Abessive | `-tta / -ttä` | **MO** — without |
| Comitative | `-ine-` | **MO** — together with |
| Instructive | `-n` (pl) | **MO** — by means of |

### 4.2 Russian — 6 cases (full inventory including Instrumental)

| Case | Masc. sg | Fem. sg | Neut. sg | Role |
|---|---|---|---|---|
| Nominative | `-∅` | `-а / -я` | `-о / -е` | Subject |
| Genitive | `-а / -я` | `-ы / -и` | `-а / -я` | Possessor / negated-obj / partitive |
| Dative | `-у / -ю` | `-е / -и` | `-у / -ю` | Recipient — often Kausal indirect |
| Accusative | = Nom (inanimate) / = Gen (animate) | `-у / -ю` | `-о / -е` | Direct object |
| **Instrumental** | `-ом / -ем` | `-ой / -ей` | `-ом / -ем` | **Means / agent in passive = MODAL** |
| Prepositional | `-е` | `-е / -и` | `-е` | With в/на/о = LO or TE |

Russian Instrumental `-ом` ≈ Finnish Adessive `-lla/-llä`
(means/instrument) plus Finnish Essive `-na/-nä` (role/state)
folded together. One case ending commits TEKAMOLO Modal by
morphology alone.

### 4.3 German — 4 cases

| Case | Article (masc/fem/neut/pl) | Role |
|---|---|---|
| Nominativ | der / die / das / die | Subject |
| Genitiv | des / der / des / der | Possessor |
| Dativ | dem / der / dem / den | Recipient / Lokal with spatial prep |
| Akkusativ | den / die / das / die | Object |

Wechsel-prepositions (an, auf, hinter, in, neben, über, unter, vor,
zwischen) govern Dativ (static) or Akkusativ (directional). German
Dativ + `mit` = Modal; German Akkusativ + `durch` = Path.

### 4.4 Turkish — 6 cases + agglutinative chain

| Case | Suffix | Role |
|---|---|---|
| Nominative | `-∅` | Subject |
| Accusative | `-i / -ı / -u / -ü` | Object |
| Dative | `-e / -a` | Goal / direction |
| Locative | `-de / -da / -te / -ta` | LO |
| Ablative | `-den / -dan / -ten / -tan` | Source |
| Genitive | `-in / -ın / -un / -ün` | Possessor |

Agglutinative stacking: `ev-ler-imiz-de-y-diler` = "they were at our
houses" = 6 morphemes (house + plural + 1pl-possessor + locative +
buffer + past-3pl-copula).

### 4.5 Japanese — particles (no cases)

| Particle | Role |
|---|---|
| が (ga) | Subject marker |
| を (wo) | Object marker |
| に (ni) | Dative / Lokal / Temporal |
| で (de) | Lokal (locative-instrumental) / Modal |
| へ (he) | Directional (goal) |
| と (to) | Comitative / quotative |
| から (kara) | Source (ablative) |
| まで (made) | Terminative |

Particles attach post-positionally and commit grammatical role as
cleanly as case endings. No gender, no case paradigms — pronouns
usually dropped (verb-agreement-free language).

### 4.6 Hebrew / Arabic — no cases, root-pattern morphology

Semitic languages carry grammatical information in a trilateral
consonantal root (K-T-B = write) + vocalic template pattern
(katab = "he wrote", yaktub = "he writes", maktub = "written").
Role is determined by the **template**, not by suffix.

For OSINT the Semitic handling needs its own harvest pass (root +
pattern tables) — out of scope for this PR.

---

## §5 Pronoun-Feature Commitment (Second Orthogonal Axis)

Morphological slot-filling (§4) and pronoun-feature commitment are
**two orthogonal axes**. A language can be rich in one and poor in
the other. For coreference resolution both matter.

| Language | Morphology slot-filling | Pronoun feature commitment |
|---|---|---|
| English | **weak** (word order only) | moderate (he/she/it on 3sg) |
| German | moderate (4 cases) | **strong** (er/sie/es + full case paradigm) |
| Russian | heavy (6 cases) | **strong** (он/она/оно + full case paradigm) |
| Finnish | **very heavy** (15 cases) | **weak** (single `hän` for he/she — gender-neutral) |
| Japanese | agglutinative particles | **minimal** (usually dropped) |
| Turkish | agglutinative | weak (single `o` for he/she/it) |

**Counter-intuitive reversal:** Finnish is easiest on morphological
slot-filling (98%+ coverage per `grammar-tiered-routing.md` §Morph)
but NOT on pronoun feature resolution (`hän` is gender-neutral).
German and Russian are richest on pronoun features; they commit
gender AND case on every 3rd-person pronoun.

**Cross-lingual bundle strategy (EN + DE + RU + FI):** each language
contributes where its commitment is strongest. Bundle maximises both
axes simultaneously. This is the VSA-native coref superpower —
parse the same entity in four languages, XOR-bundle the trajectories,
let each language's morphology collapse the others' ambiguities.

---

## §6 Markov ±5 as the Context Upgrade

Pre-Markov reasoning unit = **sentence** (Grammar Triangle at
window=1). Isolated. Fragile on Wechsel.

Post-Markov reasoning unit = **trajectory** (ContextCrystal at
window=5). NARS reasons about "this sentence in this flow," not
about isolated sentences.

This is the **context dimension upgrade** to NARS + SPO 2³ + TEKAMOLO.
The 144-verb taxonomy (12 semantic families × 12 tense/aspect
variants) carries a TEKAMOLO slot prior per cell:

| Verb family | Expected TEKAMOLO profile |
|---|---|
| BECOMES | Temporal + Modal |
| CAUSES | Subject + Object + Kausal |
| TRANSFERS | Subject + Object + Goal + (optional Kausal) |
| GROUNDS | Object + Lokal |
| TRANSFORMS | Object + Temporal + Modal |

× 12 tense/aspect = **144 verb-role cells, each a slot-filling policy**.
Parse becomes cell-lookup + morphology-driven column fill + NARS
truth-merge.

---

## §7 Convergence Target — DeepNSM as Shared Extraction Engine

### 7.1 Today's state

- DeepNSM: 6-state PoS FSM, 4,096 COCA, SpoTriple packed u64,
  <10 µs/sentence. **85% English SVO.**
- Grammar Triangle: NSM × Causality × Qualia → fingerprint. Shipped,
  unwired from DeepNSM.
- Three language stacks: Rust (canonical), Python (reference),
  TypeScript (UI). Doing the same thing with different depth.

### 7.2 Target state (this PR)

- DeepNSM emits `FailureTicket` when coverage < 0.9 (D2).
- DeepNSM calls `GrammarTriangle::analyze(text)` via `triangle_bridge.rs`
  (D3). Output: `SpoWithGrammar { triples, causality, nsm_field,
  qualia_signature, classification_distance }`.
- Markov ±5 trajectory bundled via role-indexed VSA (D5).
- `ContextChain` supports coherence / replay / disambiguate with
  Mexican-hat kernel (D4).
- Role keys (SUBJECT / PREDICATE / OBJECT / TEKAMOLO / Finnish cases
  / tenses / NARS inferences) as contiguous `[start:stop]` slices
  in 10K VSA (D6).

### 7.3 Target state (future PRs, documented here for bootloading)

- **CausalityFlow 3→9 slots extension** (§3). `beneficiary`, `goal`,
  `source`, `modal`, `local`, `instrument` added as `Option<String>`.
- **200–500 YAML templates per language** (§3.6). Fine-tune small
  LLM to emit slot-filled templates instead of free text.
- **D3 triangle wired to TypeScript grammar-awareness.ts** — cross-
  stack unification.
- **D8 story-context bridge** — Markov ±5 trajectory feeds AriGraph
  `EpisodicMemory::add_with_trajectory`; `TripletGraph::story_vector`
  + graph direct lookup superpose for coref escalation.
- **D10 forward-validation harness** — Animal Farm benchmark,
  NARS-tested epiphanies against future arc.
- **Named Entity pre-pass** (the biggest OSINT blocker — flagged in
  `grammar-tiered-routing.md` §C8).

---

## §8 Cross-References (agents: load these together)

| Doc | Covers |
|---|---|
| `grammar-tiered-routing.md` | 5-criterion detector, failure decomposition, morphology coverage baseline (98% Finnish > 85% English), self-improving loop |
| `linguistic-epiphanies-2026-04-19.md` | E13–E27 cross-repo harvest: Chomsky isomorphism, Σ10 Rubicon tiers, method grammar, Markov living frame, resonanzsiebe, sigma_rosetta, 4D glyph coordinates, membrane |
| `cross-repo-harvest-2026-04-19.md` | H1–H14: Born rule, phase tag threshold, interference truth, Grammar Triangle ≡ ContextCrystal(w=1), NSM ≡ SPO axes, FP_WORDS=160, Mexican-hat, Int4State, Glyph5B, Crystal4K 41:1, teleport F=1, 144-verb, Three Mountains |
| `integration-plan-grammar-crystal-arigraph.md` | E1–E12 shipping plan for the contract grammar + crystal modules |
| `crystal-quantum-blueprints.md` | Crystal mode (bundled Markov SPO) vs Quantum mode (holographic residual) on the same 10K substrate |
| `endgame-holographic-agi.md` | 5-layer north-star stack, 12-step holographic memory loop, P0–P3 priorities |
| `fractal-codec-argmax-regime.md` | **Separate research thread** — MFDFA on Hadamard-rotated coefficients as a codec leaf. Not entangled with grammar work. |

---

## §9 How DeepNSM Must Change (Minimal-Diff Summary)

1. Read `SentenceStructure` from the FSM (existing).
2. Call `GrammarTriangle::analyze(text)` in parallel (new via D3).
3. Merge: SPO triples + CausalityFlow slots + NSM field + qualia
   signature → `SpoWithGrammar`.
4. Compute coverage from the 5-criterion detector.
5. If coverage < 0.9 → emit `FailureTicket` via `ticket_emit.rs` (D2).
6. Else → role-indexed VSA bundle into `Trajectory` (D5) using D6
   role keys.
7. `Trajectory.fingerprint IS SentenceCrystal.fingerprint`. No new
   crystal type. Feeds AriGraph episodic + triplet graph as before.

Total DeepNSM change: ~600 LOC new + ~30 LOC edits. Every other
layer of the stack reuses shipped infrastructure.
