# Grammar-Tiered Routing — DeepNSM × Grammar Triangle × LLM

> **READ BY:** agents working on DeepNSM, OSINT pipeline, AriGraph
> integration, grammar triangle, thinking-engine extraction.
>
> **Cross-references:**
> - `lance-graph-cognitive/src/grammar/` (1929 LOC: nsm.rs, causality.rs, qualia.rs, triangle.rs)
> - `lance-graph/crates/deepnsm/` (3335 LOC: vocabulary, parser, encoder, similarity, spo)
> - `agi-chat/src/grammar/grammar-awareness.ts` + `src/thinking/grammar-bridge.ts`
> - `bighorn/extension/agi_stack/universal_grammar/` (16 modules)
> - `causal-edge/src/edge.rs` (CausalEdge64: Pearl's 2³ causal mask)
> - `osint-pipeline-openclaw.md` (pipeline this feeds)

---

## The Principle

The Grammar Triangle is a **coverage detector**, not just an extraction
format. When all three vertices (NSM primes + CausalityFlow + Qualia)
align on a sentence, the fast path handles it locally. When they don't,
the failure is DECOMPOSED into a structured ticket and sent to the LLM
for targeted disambiguation — not "parse this sentence" but "disambiguate
these specific tokens given this partial parse."

```
Sentence
    │
    ▼
Grammar Triangle attempt
    │
    ├── ALL MATCH (90-99%): local extraction (<10μs)
    │     DeepNSM FSM → SPO triples → AriGraph
    │     deterministic, $0, bit-reproducible
    │
    └── PARTIAL MATCH (1-10%): structured failure ticket → LLM
          FailureTicket {
            partial_parse:  what DID parse (subject found, verb found)
            causal_ambiguity: which Pearl 2³ configs are plausible
            tekamolo_gaps:    which adverbial slots are unfilled/ambiguous
            wechsel_tokens:   which tokens have dual-role ambiguity
          }
          → LLM disambiguates ONLY the ambiguous parts
          → result merges with partial parse
          → NARS revision updates grammar templates
```

---

## Failure Decomposition: SPO × 2³ × TEKAMOLO × Wechsel

### Layer 1: SPO 2³ Causal Trajectory

CausalEdge64 already packs Pearl's 2³ causal mask:

```
bit 0: direct    (S directly causes O)
bit 1: enabling  (S enables O to happen)
bit 2: confounding (hidden C causes both S and O)
```

8 possible causal configurations per triple. When the grammar can parse
S, P, and O but can't determine the causal direction, it reports WHICH
of the 8 are plausible:

```rust
struct CausalAmbiguity {
    /// Bitmask: which of the 8 Pearl configurations the grammar considers possible.
    /// 0xFF = fully ambiguous; 0x01 = only direct; 0x05 = direct or confounding.
    plausible: u8,
    /// Confidence that the top candidate is correct (NARS-compatible).
    confidence: f32,
}
```

Example: "The Pentagon contracted OpenAI after meeting Thiel"
- S=Pentagon, P=contracted, O=OpenAI — clear SVO ✓
- Causal direction: direct? (Pentagon wanted OpenAI) or enabling? (Thiel enabled it) or confounding? (shared interest caused both meetings + contract)
- Grammar reports: `plausible = 0x07` (all three bits set)
- LLM resolves: "most likely direct (0x01), Thiel meeting is circumstantial"

### Layer 2: TEKAMOLO slot filling

TEKAMOLO (Temporal / Kausal / Modal / Lokal) — the four adverbial
dimensions that complete an SPO triple:

```rust
#[derive(Clone, Copy, Debug)]
enum SlotStatus {
    Filled(u16),     // token index that fills this slot
    Empty,           // slot not present in the sentence
    Ambiguous(u16),  // token present but unclear which slot it fills
}

struct TekamoloSlots {
    temporal: SlotStatus,  // TE: when? ("in December", "yesterday", "before the election")
    kausal: SlotStatus,    // KA: why? ("because of the contract", "due to pressure")
    modal: SlotStatus,     // MO: how? ("secretly", "reluctantly", "via intermediary")
    lokal: SlotStatus,     // LO: where? ("in Washington", "at the Pentagon")
}
```

The Grammar Triangle's CausalityFlow attempts to fill these slots.
When it can't determine whether "in Washington" is LO (where the
meeting happened) or is part of a compound NP ("Washington Post"),
it marks `lokal: Ambiguous(token_idx)`.

The LLM ticket says: "token 7 ('in Washington') — is this Lokal or
part of an NP? Rest of the parse is done."

### Layer 3: Wechsel (dual-role ambiguity)

"Wechselpräpositionen" generalized: tokens whose grammatical function
depends on context and can't be resolved by PoS alone.

English examples:
- **"over"**: spatial (above) vs temporal (finished) vs quantitative (more than)
- **"by"**: agent (passive voice) vs location (near) vs temporal (deadline)
- **"that"**: relative pronoun, conjunction, or demonstrative
- **"since"**: temporal (since 2020) vs causal (since he left)
- **"as"**: temporal, causal, comparison, or role ("as CEO")

```rust
struct WechselAmbiguity {
    /// Token position in the sentence.
    token_idx: u16,
    /// The token's surface form.
    token: u16,  // COCA index
    /// Possible grammatical roles this token could play.
    candidates: Vec<GrammaticalRole>,
}

enum GrammaticalRole {
    SubjectDeterminer,
    RelativePronoun,
    Conjunction,
    Preposition(TekamoloSlot),
    Particle,
    Complementizer,
}
```

### Combined failure ticket

```rust
struct FailureTicket {
    /// What the grammar DID successfully parse (partial SPO, partial slots).
    partial_parse: PartialParse,

    /// Which causal configurations are plausible (Pearl 2³).
    causal_ambiguity: Option<CausalAmbiguity>,

    /// Which TEKAMOLO slots are ambiguous or unfilled.
    tekamolo: TekamoloSlots,

    /// Which tokens have dual-role ambiguity.
    wechsel: Vec<WechselAmbiguity>,

    /// Overall coverage: what fraction of the sentence was parsed.
    coverage: f32,

    /// Recommended action.
    action: FailureAction,
}

enum FailureAction {
    /// Just disambiguate these specific tokens — cheapest LLM call.
    DisambiguateTokens(Vec<u16>),
    /// Resolve causal direction — medium LLM call.
    ResolveCausality,
    /// Fill TEKAMOLO slots — medium LLM call.
    FillSlots(Vec<TekamoloSlot>),
    /// Full re-parse needed — expensive, give LLM the whole sentence.
    FullReparse,
}
```

---

## The Tiered Pipeline

```
Document (N sentences)
    │
    ▼  per-sentence
┌──────────────────────────────────────────────────────────┐
│  Grammar Triangle check (coverage detector)               │
│                                                           │
│    1. Tokenize → COCA 4096 + 20K extension                │
│    2. PoS tag → 13-tag FSM                                │
│    3. Parse SVO → PartialParse                            │
│    4. Fill TEKAMOLO slots                                 │
│    5. Check NSM prime coverage                            │
│    6. Check qualia classification_distance                │
│    7. Check causal mask determinacy                       │
│    8. Flag Wechsel tokens                                 │
│                                                           │
│    IF all pass → Local extraction (DeepNSM, <10μs)        │
│    IF any fail → Build FailureTicket                      │
└─────────────┬──────────────────────┬──────────────────────┘
              │                      │
         LOCAL (90-99%)         TICKET (1-10%)
              │                      │
              ▼                      ▼
     DeepNSM full pipeline    ┌─────────────────────┐
     SPO + NARS truth         │  FailureAction dispatch│
     fingerprint              │                       │
     → AriGraph               │  DisambiguateTokens   │
                              │    → small LLM call   │
                              │    → merge with parse  │
                              │                       │
                              │  ResolveCausality     │
                              │    → medium LLM call  │
                              │    → set Pearl mask   │
                              │                       │
                              │  FillSlots            │
                              │    → medium LLM call  │
                              │    → fill TE/KA/MO/LO │
                              │                       │
                              │  FullReparse          │
                              │    → expensive LLM    │
                              │    → complete SPO      │
                              └────────┬──────────────┘
                                       │
                                       ▼
                              NARS revision feedback
                              (LLM extraction → grammar template update)
                              (next time, similar sentence matches locally)
```

---

## Existing Grammar Infrastructure

### lance-graph-cognitive/src/grammar/ (1929 LOC, SHIPPED)

| File | LOC | Role |
|------|-----|------|
| `nsm.rs` | 448 | 65 NSM semantic primitives + activation scoring |
| `causality.rs` | 396 | CausalityFlow: agency, temporality, dependency types |
| `qualia.rs` | 718 | 18D qualia field from grammar analysis |
| `triangle.rs` | 304 | GrammarTriangle: NSM × Causality × Qualia → fingerprint |
| `mod.rs` | 63 | Module root |

### bighorn/extension/agi_stack/universal_grammar/ (16 modules)

| File | Role |
|------|------|
| `core_types.py` | Base types for universal grammar (verbs, slots) |
| `verb_endpoints.py` | Verb-level HTTP endpoints for grammar testing |
| `calibrated_grammar.py` | Grammar with calibrated confidence |
| `method_grammar.py` | Method-level grammar decomposition |
| `resonance.py` | Grammar × resonance integration |
| `resonanzsiebe.py` | Resonance sieve (the 6-level SILENCE..SCREAM filter) |
| `scent_optimizer.py` | Scent-level (1-byte) grammar optimization |
| `situation_executor.py` | Situation-driven grammar execution |
| `meta_uncertainty.py` | MUL integration for grammar |
| `exploration.py` | Grammar exploration strategies |
| `integration.py` | Cross-module integration |
| `invoke_router.py` | Router for grammar dispatch |
| `awareness_blink.py` | Awareness-level grammar (blink = minimal unit) |
| `situation_storage.py` | Situation persistence |
| `jina_integration.py` | Jina embedding bridge |
| `__init__.py` | Package root |

### agi-chat/src/grammar/ + thinking/grammar-bridge.ts

| File | LOC | Role |
|------|-----|------|
| `grammar-awareness.ts` | 237 | Grammar awareness layer (detection) |
| `grammar-bridge.ts` | 212 | Bridge between grammar and thinking cycle |

### causal-edge (in lance-graph, SHIPPED)

CausalEdge64 already packs the Pearl 2³ mask:
```
bits 40-42: CausalMask (direct=1, enabling=2, confounding=4)
```

This IS the causal trajectory decomposition — it's one field read away
from the failure-ticket's `CausalAmbiguity` struct.

---

## What Needs Building (New Work)

| Component | LOC est. | Description |
|-----------|----------|-------------|
| `FailureTicket` struct + `FailureAction` enum | ~100 | The structured failure type |
| `TekamoloSlots` struct + slot-filling logic | ~200 | Parse adverbial positions from CausalityFlow |
| `WechselAmbiguity` detector | ~150 | Flag dual-role tokens (prepositions, conjunctions, relative pronouns) |
| Grammar → RoutingDecision function | ~100 | The 8-check coverage detector that returns Local or Ticket |
| LLM ticket dispatch (xai_client integration) | ~100 | Format ticket as LLM prompt, parse response, merge |
| NARS revision feedback (LLM → grammar template) | ~150 | When LLM succeeds where grammar failed, learn the pattern |
| **Total** | **~800** | |

Most of this composes from existing types. `CausalMask` is in causal-edge.
NSM coverage is in `grammar/nsm.rs`. Qualia classification_distance is
in `cognitive-shader-driver/engine_bridge.rs`. The routing decision
function orchestrates existing checks, not new analysis.

---

## The Self-Improving Loop

```
Cycle N:   grammar coverage = 85%  →  15% goes to LLM  →  LLM extracts
Cycle N+1: LLM extractions become NARS-revised grammar templates
           grammar coverage = 87%  →  13% goes to LLM
Cycle N+K: grammar coverage → 95%+ →  <5% goes to LLM
```

The grammar's coverage ratio improves monotonically because:
1. Every LLM extraction is a positive example for the grammar template bank.
2. NARS revision increases confidence on templates that succeed.
3. Low-confidence templates get pruned.
4. The TEKAMOLO slots self-calibrate (slot X is usually Temporal → raise prior).

The end state: the grammar handles ~99% locally, and the 1% that remains
is genuinely novel language that requires world knowledge, not parsing.
The grammar never needs to handle ambiguity it can't resolve — it just
needs to DETECT ambiguity precisely and delegate cleanly.

---

## Why This Matters Beyond NLP

This is the same pattern as the chess-NARS vertical's proprioception:

- Chess: the engine knows what it CAN evaluate (tactical) vs what it
  CAN'T (requires deeper search). The proprioception report says "I'm
  in Observer mode, searching at depth 12, uncertain about this line."
- Grammar: the parser knows what it CAN parse (SVO with clear roles)
  vs what it CAN'T (ambiguous Wechsel preposition). The failure ticket
  says "I parsed 80%, token 7 is ambiguous between Lokal and NP-part."
- Both: self-knowledge of capability boundaries, expressed as structured
  data, routed to the appropriate handler.

The Grammar Triangle's failure ticket IS proprioception applied to
parsing. Same contract, different domain.

---

## Morphologically Rich Languages — Higher Coverage, Not Lower

### The counterintuitive win

Grammar-heavy languages (Finnish, Hungarian, Estonian, Turkish, Basque,
Georgian, Russian, Japanese) are **easier** for local extraction than
English. The case system IS the slot filler — no inference needed.

English is one of the hardest languages for local extraction because
word ORDER is the only signal. "John saw Mary" vs "Mary saw John" —
only position tells you who's subject and who's object.

Finnish "Johni näki **Maryn**" — the accusative ending `-n` on Mary
tells you directly she's the object. Reorder to "Maryn näki Johni" —
still clear, morphology carries the grammar.

### Finnish 15 cases → TEKAMOLO mapping (brute-force, regular)

Finnish has ~15 cases × ~4 declension classes = ~60 morphological
patterns. Each case maps directly to a TEKAMOLO slot or an SPO role:

```
Case          Ending     TEKAMOLO slot / Role
─────         ──────     ────────────────────
Nominative    -∅         → Subject (S)
Genitive      -n         → Possessor / Object of some verbs
Accusative    -n/-t      → Object (O)
Partitive     -a/-ä      → Partial object / negated object

Inessive      -ssa/-ssä  → LO (in, inside)
Elative       -sta/-stä  → LO (from inside)
Illative      -Vn/-seen  → LO (into)
Adessive      -lla/-llä  → LO (on, at) / Instrument
Ablative      -lta/-ltä  → LO (from surface)
Allative      -lle       → LO (onto, toward)

Essive        -na/-nä    → MO (as, in the state of)
Translative   -ksi       → MO (becoming, changing into)

Abessive      -tta/-ttä  → MO (without — negated modal)
Comitative    -ine-      → MO (together with — social modal)
Instructive   -n (pl)    → MO (by means of — instrumental)
```

**This is a static lookup table, not a model.** Enumerate the endings,
map to slots, done. ~60 entries. Coverage: near-total for case-role
assignment. The grammar literally TELLS you the role.

### Hungarian (18 cases), Estonian (14), Turkish (6 + agglutination)

Same pattern, slightly different table sizes:

| Language | Cases | Agglutination | Est. lookup entries | Expected local coverage |
|----------|-------|---------------|--------------------|-----------------------|
| Finnish | 15 | High | ~60 | 98%+ |
| Hungarian | 18 | High | ~80 | 98%+ |
| Estonian | 14 | Medium | ~50 | 97%+ |
| Turkish | 6 | Very high | ~30 + suffix chains | 95%+ |
| Basque | 12 | High + ergative | ~50 | 95%+ |
| Georgian | 7 | Polypersonal verbs | ~40 + verb agreement tables | 93%+ |
| Russian | 6 | Moderate | ~40 + gender/number declensions | 92%+ |
| German | 4 | Low | ~16 per article/adjective table | 90%+ |
| Japanese | particles | Agglutinative | ~20 particles | 95%+ |
| English | 0 | None | N/A (word order only) | 85% (FSM baseline) |

**English is the WORST case for local extraction.** Every other language
with real morphology gives us free information.

### Implementation: one lookup table per language

```rust
/// Case-ending lookup table for a morphologically rich language.
struct CaseTable {
    language: &'static str,
    entries: Vec<CaseEntry>,
}

struct CaseEntry {
    /// The suffix to match (e.g. "-ssa", "-lle").
    suffix: &'static str,
    /// Which grammatical case this is.
    case: GrammaticalCase,
    /// Direct TEKAMOLO slot mapping (None if SPO role, not adverbial).
    tekamolo_slot: Option<TekamoloSlot>,
    /// SPO role (Subject / Object / None).
    spo_role: Option<SpoRole>,
}
```

Total per language: one const array of ~50-80 entries. Matching is
suffix-stripping: try longest suffix first, match to slot.

For agglutinative languages (Finnish, Turkish, Hungarian), process
suffixes right-to-left, peeling off one case/possessive/number
marker at a time. Each peeled suffix fills a slot. The stem is the
content word → COCA/20K lookup.

### The brute-force cost

Adding a new language:
1. Enumerate its case endings (~50-80 entries): **1 day of linguistic work**.
2. Map each case to TEKAMOLO or SPO role: **1 hour** (the mapping is
   standard cross-linguistic grammar).
3. Write the suffix-matching function: **~100 LOC** (shared across
   all agglutinative languages; only the table changes).
4. Test on a text corpus: **1 day**.

Total per language: **~2 days**. No model training, no data annotation,
no GPU. Pure reference-grammar transcription.

### Why this flips the "LLM covers more languages" argument

LLM-based extractors handle 100+ languages because pretraining saw
text in all of them. But they handle morphology **implicitly** (black
box) and **expensively** (full forward pass per sentence).

We handle morphology **explicitly** (lookup table) and **cheaply**
(suffix match). Our per-language cost is 2 days of one-time setup;
their per-language cost is billions of pretraining tokens.

For grammar-heavy languages specifically, our local coverage rate is
HIGHER than English (~98% vs ~85%), so even LESS goes to LLM. The
tiered-pipeline's cost equation gets better, not worse, as we add
morphologically rich languages.

### Priority languages for the OSINT use case

| Language | Why | Case complexity |
|----------|-----|-----------------|
| Russian | Intelligence domain (military, diplomatic) | 6 cases, gendered |
| Arabic | OSINT-critical (Middle East + North Africa) | 3 cases + root system (trilateral roots) |
| Turkish | Strategic (NATO, regional) | 6 cases + deep agglutination |
| Mandarin | Strategic (geopolitical) | No cases (like English — word order + particles) |
| Farsi | OSINT (Iran) | No cases, like English but SOV word order |
| German | European intelligence, financial | 4 cases, gendered |
| Hebrew | Intelligence (Israel) | No cases, root-pattern morphology |

Russian and Turkish are the highest-value additions: both grammar-heavy
(= high local coverage), both OSINT-critical (= high demand), both
achievable with lookup tables (= low effort).
