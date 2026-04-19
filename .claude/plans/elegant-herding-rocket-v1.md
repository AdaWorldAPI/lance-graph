# Plan — DeepNSM as Full Parser via Markov ±5 Context Upgrade

## Context

**Priority (user):** Making DeepNSM the language parser without LLM is
huge; it has priority. Markov ±5 SPO+TEKAMOLO bundling is **the
context upgrade to NARS + SPO 2³ + TEKAMOLO** — trajectory as
reasoning unit, not sentence. Crystal types are reused, not expanded.

**Already documented** (do not duplicate, read as prerequisites):

- `.claude/knowledge/grammar-tiered-routing.md` — the full 5-criterion
  coverage detector, SPO×2³×TEKAMOLO×Wechsel failure decomposition,
  morphology coverage table (Finnish 98 % > English 85 %), self-
  improving loop, OSINT language priorities.
- `.claude/knowledge/integration-plan-grammar-crystal-arigraph.md` —
  12 epiphanies E1–E12 (grammar-tiered, morphology-easier, FailureTicket,
  cross-lingual superposition, Markov ±5, NARS-about-grammar, crystal
  hierarchy, CrystalFingerprint, 5D quorum, episodic unbundle, AriGraph
  substrate, demo matrix).
- `.claude/knowledge/crystal-quantum-blueprints.md` — Crystal mode
  (bundled Markov SPO chain, Structured5x5 middle cells, bipolar) vs
  Quantum mode (holographic residual, Vsa10kF32 + PhaseTag, sandwich
  wings for phase keys).
- `.claude/knowledge/cross-repo-harvest-2026-04-19.md` — H1 Born rule,
  H2 phase-tag threshold, H3 interference truth, H4 Triangle ≡
  ContextCrystal(w=1), H5 NSM ≡ SPO axes, H6 FP_WORDS=160, H7 Mexican-
  hat, H8 Int4State, H9 Glyph5B, H10 Crystal4K 41:1, H11 teleport F=1,
  H12 144-verb taxonomy, H13 Three Mountains, H14 Triangle+Context
  hybrid.
- `.claude/knowledge/session-capstone-2026-04-18.md` — SB1–SB7, MB1–
  MB5, E1–E8, §7 AriGraph-already-shipped addendum.
- `.claude/knowledge/endgame-holographic-agi.md` — 5-layer stack, 12-
  step holographic memory loop, P0/P1/P2/P3 priorities.

## Shipped Today (merged to main)

- PR #208 — grammar/ + crystal/ contract modules, AriGraph
  unbundle hooks.
- PR #209 — sandwich layout, bipolar cells, Codex fixes.
- 112 contract tests + 11 episodic tests passing.

## Today's Lossiness Epiphanies (not yet in a knowledge doc)

| Form | Size | Lossy? | Why |
|---|---|---|---|
| 10 000-D f32 (Vsa10kF32) | 40 KB | **LOSSLESS** | 320 K bits of capacity ≫ any bundled signal; orthogonal role keys give exact unbundle |
| Signed 5^5 bipolar | 3 KB | **LOSSLESS** | Negative cancellation = VSA superposition; opposites cancel natively |
| Unsigned 5^5 | 3 KB | **LOSSY** | No cancellation; accumulation saturates |
| Binary bitpacked (10K or 16K) | 1.25 / 2 KB | **LOSSY** | Majority-vote bundle commits to 0/1, loses count |
| CAM-PQ projection (Binary16K → 10K) | 10 / 40 KB | **LOSSLESS** | Distance-preserving by construction |

**VSA convention:** role keys use contiguous `[start:stop]` slices
(e.g. SUBJECT → [0..156], OBJECT → [156..312]), not scattered bits.
This is the `vsa_permute`-friendly layout.

## First Concrete Target — Pronoun / Coreference Resolution

Before any abstract architecture claim, there's a bounded problem that
exercises the entire meta-inference stack: **resolving grammatical
references** — "it", "he", "she", "they", "the company", "that" —
back to their antecedents. Every downstream pipeline (OSINT, chess
commentary, Wikidata triples) depends on it. LLM-based extractors
solve it via attention over tokens; we solve it via **logical
reasoning over the ±5 Markov chain**.

**The problem:**

```
S_{-2}  "Pentagon contracted OpenAI in December."
S_{-1}  "Thiel was at the meeting."
S_{0}   "It announced the deal on Tuesday."       ← "it" = ?
S_{+1}  "The stock jumped 4%."
```

"It" could refer to Pentagon, OpenAI, or the meeting. English leaves
it ambiguous. LLMs use attention heads. We use the context chain as
the candidate enumerator and meta-inference as the decision rule.

**Grammatical roles ARE bundling — the VSA-native shortcut.**

Each grammatical role owns a contiguous `[start:stop]` slice of the
10K VSA space (D6). At parse time, content is XOR-bound into its
role's slice and accumulated:

```
for token in sentence:
    role_key   = role_keys[token.grammatical_role]      // per-slice key
    bound      = vsa_bind(content_fp(token), role_key)
    trajectory = vsa_bundle(trajectory, bound)          // accumulates
```

Because binding uses contiguous slices, bundling preserves
role-indexed content throughout. At retrieval:

```
subjects_bundle   = vsa_unbind(trajectory, SUBJECT_KEY)   // just subjects
objects_bundle    = vsa_unbind(trajectory, OBJECT_KEY)
locatives_bundle  = vsa_unbind(trajectory, LOKAL_KEY)
```

**The bundle IS the candidate store.** Candidate enumeration for
coreference becomes an O(1) slice unbind, not a search:

- "She announced…" → `vsa_unbind(±5_trajectory, SUBJECT_KEY)` recovers
  every subject played in the window.
- Then apply `vsa_clean` against a gender-feminine codebook to filter
  to feminine-animate survivors.
- Typically 1-2 candidates remain; Deduction commits.

**Why this matters:** without role-indexed bundling, candidate
enumeration scans the context chain and rebuilds the candidate set
every time. With role-indexed bundling, the bundle itself IS the
structured index — retrieval is a single XOR + clean.

**Cross-lingual bundling is literally role-slice addition.** Finnish
parse of the same entity lands Finnish content into the same
SUBJECT_KEY slice as English content. When the two trajectories are
bundled together, the SUBJECT slice carries both parses'
role-committed content. Unbinding recovers the aggregate — and
Finnish case morphology has already committed slots the English
parse left ambiguous.

**This is what D5 and D6 actually produce together.** D5's
`MarkovBundler` does the role-indexed bind-and-bundle. D6's role keys
are the slice addresses. Coreference resolution = D4
`ContextChain::disambiguate` running counterfactual substitution over
the role-unbind candidate set.

**How the stack resolves it — grounded in what we're shipping:**

1. **Enumerate candidates from ±5 Markov chain** — extract every noun
   in `ContextChain.preceding + focal + following` as a potential
   antecedent. Weight by recency × grammatical salience (subjects >
   objects > obliques) × Mexican-hat kernel.

2. **Hydrate each candidate's SPO role** — from D3's Grammar Triangle
   bridge, every prior sentence's SPO triple is available with Pearl
   2³ mask committed. Pentagon was Subject of "contracted"; OpenAI
   was Object; meeting was Locative modifier.

3. **Counterfactual axis test** — for each candidate, apply the
   substitution: "Pentagon announced the deal" / "OpenAI announced
   the deal" / "The meeting announced the deal". Score the SPO
   coherence (does the verb fit the subject? "meetings don't
   announce" → Pearl mask inconsistent → low score).

4. **Markov axis test** — bind each candidate as the Subject of S_{0},
   replay the chain via `ContextChain::replay_with_alternative`,
   measure `total_coherence`. The binding that maximises coherence
   with ±5 context wins.

5. **Joint reading (D7 meta-inference duality)** — counterfactual and
   Markov axes must agree. If both say "Pentagon" → Deduction commits.
   If they split (CF says Pentagon by subject-continuity; Markov says
   OpenAI because stock jumped = company signal) → the disagreement
   is diagnostic:
   - One of them is measuring a surface bias (subject continuity
     default) that the other correctly overrides.
   - The style's permanent dispatch picks Abduction to weigh which
     axis has grip on this signal profile.

6. **NARS revision on outcome** (D7 empirical layer) — when the
   ensemble agrees with a downstream confirmation (e.g. the LLM
   fallback would have picked the same answer, or a named-entity
   link confirms the binding), the `GrammarStyleAwareness` revises
   its prior over "how often the subject-continuity default holds
   on this content distribution."

**Two-tier resolution by inherent factual content.**

Pronouns split into two classes with different resolution costs:

**Fixed Pronomen — carry inherent factual commitments:**

| Pronoun | Inherent commits (axiomatic, permanent, language-specific) |
|---|---|
| I / we / me / us | Speaker-deictic (SPEAKER role) |
| you / thou | Addressee-deictic (ADDRESSEE role) |
| he / him / his | singular + animate + masculine |
| she / her | singular + animate + feminine |
| they (personal) | plural + animate |
| Proper names | rigid designators — committed entity |

For fixed pronouns, the **inherent-feature filter over ±5 candidates
IS the resolution**. "She announced…" with candidate set
`{Pentagon(neuter), OpenAI(neuter), Marietje(fem)}` — the filter
eliminates 2/3 axiomatically. No counterfactual replay needed. This
is Deduction on inherent facts. Cheap, permanent.

**Wechsel Pronomen — zero inherent commitment:**

| Pronoun | Ambiguous on |
|---|---|
| it | referent (anything non-animate, or expletive) |
| that | demonstrative / relative / complementizer / conjunction |
| this | demonstrative / intensifier ("this big") |
| they (singular / expletive) | "they say…", singular they |
| one | indefinite / ordinal / pro-form |
| which | relative / interrogative |

These need the **full meta-inference stack** (CF × Markov axes) from
the previous section — the expensive path. They're already on the
`WechselAmbiguity` list from D2.

**Cross-linguistic commitment profiles vary per pronoun class —
another reversal from the morphology story:**

| Language | Morphology (slot-filling) | Pronoun feature commitment |
|---|---|---|
| English | weak (word order) | moderate (he/she/it on 3sg) |
| German | moderate (4 cases) | **strong** (er/sie/es + case: ihn/ihm/seiner) |
| Russian | heavy (6 cases) | **strong** (он/она/оно + full case paradigm) |
| Finnish | **very heavy** (15 cases) | **weak** (single `hän` for he/she — gender-neutral) |
| Japanese | agglutinative particles | minimal (usually dropped) |
| Turkish | agglutinative | weak (single `o` for he/she/it) |

**Finnish is the easiest language on morphological slot-filling but
NOT on pronoun feature resolution** (because `hän` is gender-neutral).
German and Russian are richest on pronoun features.

A cross-lingual bundle (EN + DE + RU + FI) maximises both axes
simultaneously: each language contributes where its commitment is
strongest. Wechsel-heavy English + feature-weak Finnish + feature-
strong German/Russian = the complementary quartet for OSINT
resolution.

**Resolution dispatch:**

```
pronoun = lex_classify(token)
    │
    ├── FIXED → feature_filter(±5 candidates, pronoun.inherent_features)
    │              → 1-2 survivors → Deduction commits
    │              → no Markov / CF needed (usually)
    │
    └── WECHSEL → full D7 meta-inference
                   (CF axis × Markov axis × cross-lingual bundle)
                   → WechselAmbiguity on the FailureTicket if both weak
```

**Categories of reference resolution this unlocks** (all same
mechanism, different candidate generators):

| Reference type | Example | Candidate set |
|---|---|---|
| Anaphoric pronoun | "Pentagon … **it** announced …" | All preceding nouns, subject-weighted |
| Cataphoric pronoun | "If **he** calls, tell John …" | Next-mentioned proper noun |
| Dropped subject (Russian, Japanese, Italian) | "Пошёл домой." ("[he/she] went home") | Preceding subject, verb agreement narrows gender/number |
| Definite NP | "the company" | All prior company-mentioned entities |
| Demonstrative | "that", "this" | Most salient noun or fact by Markov coherence |
| Possessive | "his", "her", "its" | Possessor candidates filtered by animacy + agreement |
| Relative pronoun | "the man who …" | Immediate antecedent in syntactic scope |

**Cross-lingual bonus** — dropped subjects are cheap in
morphologically-rich languages because verb agreement narrows the
candidate set before the Markov axis fires. Russian
`Пошла домой` — feminine verb ending → candidate restricted to
feminine antecedents in ±5. Japanese verb form + particle commits
discourse role. English has no such morphological narrowing, which is
why English coreference is *the* hard problem and everyone else is
easier.

**What this means for shipping order.**

The D2/D3/D4/D5/D6/D7 deliverables already cover everything needed.
The coreference problem is **the first verification target** — the
end-to-end test that proves the stack works:

- D4 `ContextChain::disambiguate` runs the replay.
- D5 Markov bundle produces the trajectory fingerprints to score against.
- D6 role keys bind each candidate as the Subject in the counterfactual substitution.
- D7's meta-inference duality combines CF + Markov scores.
- D2 FailureTicket fires only when both axes are weak.

**Integration test corpus: Orwell, *Animal Farm*.**

Animal Farm is the canonical benchmark because:

- Orwell is famous for deliberately-clean textbook English grammar —
  plain, precise, unambiguous. Used in English-language teaching
  precisely because every sentence parses cleanly. If we cannot hit
  85 %+ local on Animal Farm, the tiered-routing claim fails.
- Heavy named-entity density: Napoleon, Snowball, Squealer, Boxer,
  Clover, Mollie, Benjamin, Old Major, Mr. Jones, Pilkington,
  Frederick, Whymper. Each has committed animacy / gender / species /
  social-role features — classic fixed-pronoun resolution targets.
- Gendered pronouns across species: "Mollie tossed her mane" /
  "Napoleon issued his decree" / "the horses lost their strength" —
  feature-filter resolution is unambiguous on 90 %+ of cases.
- Subject-continuity chains through chapters — Markov ±5 coherence is
  measurable across whole narrative sequences.
- Clear causal chains (Squealer explains why X; Napoleon decrees
  because Y) — SPO 2³ causal mask commits are testable.
- Public-domain (UK copyright expired 2020); ~40 K words; small
  enough to process end-to-end in a single run; large enough to give
  statistically meaningful coref accuracy.

**Benchmark targets:**

- Local (no LLM) coverage ≥ 85 % on full Animal Farm text.
- Fixed-pronoun coreference accuracy ≥ 95 % (he/she/his/her/their
  against gendered named animals).
- Wechsel-pronoun ("it", "that") accuracy ≥ 75 % via CF+Markov meta-
  inference.
- Cross-lingual bundle improvement: EN + DE (Orwell's German
  translation is widely-available, textbook-grade) shows measurable
  Wechsel-pronoun disambiguation improvement ≥ 5 %.
- End-to-end latency: full book parsed locally in ≤ 5 minutes
  (< 10 µs/sentence × 40 K sentences = 400 ms; the rest is ±5
  buffering and bundle construction).

---

## The Synthesis (what Markov ±5 actually buys)

Pre-Markov reasoning unit = **sentence**. Thinking style applies NARS
inference to one SPO triple + TEKAMOLO slots. Isolated. Fragile on
Wechsel.

Post-Markov reasoning unit = **trajectory**. Thinking style applies
NARS inference to a trajectory fingerprint that carries ±5 sentences
of context (position-permuted, Mexican-hat weighted). NARS doesn't
reason about "this sentence"; it reasons about "this sentence in
this flow." This is the context dimension upgrade to NARS+SPO2³+TEKAMOLO.

Cross-linguistic corollary (AGI-grade per user): XOR-bundle a trajectory
from EN and FI of the same entity. Finnish case morphology (adessive
`-llä`, elative `-sta`) commits Wechsel-ambiguous English roles into
specific TEKAMOLO slots. The superposed trajectory carries
disambiguation for free.

## To Build (one combined PR, ~1,555 LOC)

### D0 — Knowledge doc

**New:** `/home/user/lance-graph/.claude/knowledge/grammar-landscape.md`
(~320 lines). Three grammar stacks (Rust / Python / TypeScript) with
paths + LOCs; Triangle = NSM × Causality × Qualia; TEKAMOLO templates
with 3-slot gap flagged; YAML 200–500/language training pipeline as
future target; Markov ±5 as the context upgrade to NARS+SPO2³+TEKAMOLO.
Cross-refs to the five existing knowledge docs.

**Includes a §Case Inventories Per Language** section with native-
terminology tables for at least: Finnish (15 — with the Accusative
correction), Russian (6 — full Nom/Gen/Dat/Acc/**Ins**/Prep),
German (4 — Nom/Gen/Dat/Akk), Turkish (6 + agglutinative chain),
Japanese (particles: が/を/に/で/へ/と/から/まで). Each case entry
maps to its permanent TEKAMOLO / SPO role and notes when a Latinate
label (e.g. "Accusative") would be misleading for that language.

**Also updates (edit):** `grammar-tiered-routing.md` — patch the
Finnish case table with the correct Nominative/Genitive/Partitive
object marking and a note that true Accusative is personal-pronoun
only. Patch is ≤ 30 lines.

**CausalityFlow extension scope (deferred from this PR, documented
in D0).** The original "3 new fields" (modal / local / instrument)
is insufficient. Full thematic-role inventory adds 3 more:

- **Beneficiary** — "for whom" (dative of benefit). German
  `für ihn`, Finnish Allative `-lle` in benefactive reading, Russian
  Dative with `для` / `+dat`.
- **Goal / Direction** — "to where" (directional). Finnish Illative
  `-Vn/-hVn/-seen`, Russian Accusative with `в/на`, German `nach/
  zu` + Dat.
- **Source** — "from where" (ablative origin). Finnish Elative
  `-sta/-stä` or Ablative `-lta/-ltä`, Russian Genitive with `из/с`,
  German `aus/von` + Dat.

Optional later additions (language-specific):

- **Path** — "through where" (prolative). Finnish Prolative
  `-tse/-itse`, Turkish Instrumental-path construction.
- **Purpose / Finale** — "for what purpose". German `zum + Inf`,
  Finnish Translative `-ksi` in purposive reading.
- **Result** — "leading to what". German `sodass`, Finnish
  consequence constructions.

Full extended CausalityFlow (deferred follow-up PR, not this one):

```rust
pub struct CausalityFlow {
    // existing (3/9 slots):
    pub agent: Option<String>,
    pub action: Option<String>,
    pub patient: Option<String>,
    pub reason: Option<String>,        // Kausal
    pub temporality: f32,              // Temporal as float
    pub agency: f32,
    pub dependency_type: DependencyType,

    // TEKAMOLO completion (3 slots, the original "modal/local/instrument"):
    pub modal: Option<String>,
    pub local: Option<String>,
    pub instrument: Option<String>,

    // Thematic-role completion (3 more slots, classical theory):
    pub beneficiary: Option<String>,
    pub goal: Option<String>,
    pub source: Option<String>,
}
```

6 new `Option<String>` fields in total (not 3) to reach the full
thematic-role inventory. Still trivial struct change; still deferred
from the current PR's scope (user decision).

### D2 — DeepNSM emits FailureTicket

**Edit:** `crates/deepnsm/src/parser.rs` (+30 LOC) — end-of-parse
coverage branch. **New:** `crates/deepnsm/src/ticket_emit.rs` (+120).
**Feature gate:** `contract-ticket` on deepnsm; default stays zero-dep.

Ticket carries the SPO × 2³ × TEKAMOLO × Wechsel decomposition from
grammar-tiered-routing.md §"Combined failure ticket". Failure reason
itself is the routing signal (COCA miss → Wikidata; FSM incomplete →
LLM parse; slots unfillable → disambiguate; low primes → LLM; high
classification_distance → novel domain).

### D3 — Grammar Triangle wired into DeepNSM

**New:** `crates/deepnsm/src/triangle_bridge.rs` (+220 LOC).

Thin adapter: DeepNSM FSM produces `SentenceStructure`; call
`lance_graph_cognitive::grammar::GrammarTriangle::analyze(text)` in
parallel to produce `(NSMField, CausalityFlow, Qualia18D)`; merge
into `SpoWithGrammar { triples, causality, nsm_field,
qualia_signature, classification_distance }`.

**Feature gate:** `grammar-triangle` (additive).

**Explicit non-scope:** CausalityFlow TEKAMOLO extension (modal/local/
instrument) is **deferred** per user. Current 3/6 slots mapped.

### D4 — ContextChain reasoning ops (coherence / replay / disambiguate)

**Edit:** `crates/lance-graph-contract/src/grammar/context_chain.rs`
(+140 LOC). Upgrades ring-buffer to reasoning substrate:

```rust
impl ContextChain {
    pub fn coherence_at(&self, i: usize) -> f32;
    pub fn total_coherence(&self) -> f32;
    pub fn replay_with_alternative(&self, i: usize,
        alt: CrystalFingerprint) -> (Self, f32);
    pub fn disambiguate<I>(&self, i: usize, candidates: I)
        -> DisambiguationResult;
}
pub enum WeightingKernel { Uniform, MexicanHat, Gaussian }
```

Zero-dep preserved (internal Hamming on `Box<[u64; 256]>`; cosine for
VSA variants). Mexican-hat = harvest H7 landing.

### D5 — Markov ±5 SPO+TEKAMOLO bundler (role-indexed)

**New:** `crates/deepnsm/src/markov_bundle.rs` (+220 LOC).
**New:** `crates/deepnsm/src/trajectory.rs` (+80 LOC).

`MarkovBundler` — **role-indexed bundling** (the VSA-native shortcut
for coreference):

- Ring buffer of 11 `SentenceStructure`s.
- For each token, XOR-bind `content_fp(token)` with the appropriate
  role key from D6: SUBJECT_KEY, PREDICATE_KEY, OBJECT_KEY,
  MODIFIER_KEY, CONTEXT_KEY, TEMPORAL_KEY, KAUSAL_KEY, MODAL_KEY,
  LOKAL_KEY, INSTRUMENT_KEY, or a language-specific case key
  (Finnish Inessive, Russian Instrumental, etc.) when morphology
  commits the role.
- Because role keys own disjoint `[start:stop]` slices of the 10K
  vector, the bind lands content into its role's slice; no cross-
  role contamination.
- `vsa_permute(v, position_offset)` per sentence in the window
  (harvest H5 bit-chain).
- `vsa_bundle` with Mexican-hat weights (focal > ±1 > … > ±5).
- Produce `Trajectory { vsa_vector, context_chain, structured_5x5 }`.
  Trajectory's fingerprint IS `SentenceCrystal::fingerprint` — no new
  crystal type introduced.

**Companion retrieval API on `Trajectory`:**

```rust
impl Trajectory {
    /// O(1) unbind of a role slice — recovers the role-indexed
    /// content bundle without scanning the context chain.
    pub fn role_bundle(&self, role: GrammaticalRole) -> Box<[f32; 10_000]>;

    /// Filter the role bundle through a codebook (e.g. feminine-
    /// animate nouns for "she") to produce survivor candidates.
    pub fn role_candidates(
        &self,
        role: GrammaticalRole,
        filter_codebook: &[&[f32; 10_000]],
    ) -> Vec<Candidate>;
}
```

This is what makes coreference O(1) per query: candidate
enumeration = slice unbind + codebook clean, not a context-chain scan.

Target bundled form: **lossless** — `Vsa10kF32` sandwich (f32 wings
accumulate residuals; middle cells hold Structured5x5). Derived
`Binary16K` via sign-binarize for hot-path sweep.

### D6 — Role-key catalogue

**New:** `crates/lance-graph-contract/src/grammar/role_keys.rs`
(+160 LOC). Deterministic `LazyLock` `VsaVector`-shaped keys via FNV-64,
addressed as contiguous `[start:stop]` slices:

- 5 SPO-role keys: SUBJECT [0..2000], PREDICATE [2000..4000],
  OBJECT [4000..6000], MODIFIER [6000..8000], CONTEXT [8000..10000].
- 5 TEKAMOLO-slot keys (TEMPORAL, KAUSAL, MODAL, LOKAL, INSTRUMENT;
  last three future-ready).
- 15 Finnish case keys from `FinnishCase`.
- 12 tense keys (aligned with sigma_rosetta 144 = 12 × 12).
- 7 NARS inference keys.

(Exact slice boundaries set so role-domains don't overlap; Hamming
similarity within a role-domain stays meaningful.)

### D8 — Story Context = Episodic Memory + Triplet Graph (already shipped)

**The story IS a graph.** AriGraph's `triplet_graph.rs` (1064 LOC,
shipped) stores every SPO triple with NARS truth and temporal
metadata. `episodic.rs` (210 LOC + unbundle hooks from PR #208) stores
each sentence as an Episode. Together they already represent the
narrative: entities are nodes, verbs are edges, temporal ordering
makes a Markov chain through the graph. No new `StoryContext` struct
needed — just query methods that traverse what's there.

**Two-tier Markov chain via existing storage:**

| Tier | Where it already lives | Query method |
|---|---|---|
| Working memory (±5) | `MarkovBundler::Trajectory` (D5, new) | `Trajectory::role_bundle` |
| Paragraph (±50) | `EpisodicMemory` Hamming retrieval | existing `retrieve_similar(fp, k)` |
| Document (±500) | `TripletGraph` BFS association + spatial paths | existing `triplet_graph.rs` |
| Permanent (rigid designator) | AriGraph unbundled facts (PR #208) | `EpisodicMemory::unbundle_hardened` |

Each tier uses the **same role-indexed VSA bundling mechanism** and
the **same SPO store** — the scale is the query radius, not a new
data structure.

**On the grammar side (already covered in D5):**

`MarkovBundler` emits a `Trajectory` at each focal sentence. Feeds
into AriGraph via the existing `EpisodicMemory::add()`.

**On the AriGraph side — two thin additions, no new module:**

**(a) Graph direct lookup — objects as nodes, SPO as edges.**

The existing `TripletGraph` (1,064 LOC shipped) already stores
objects as nodes addressed by fingerprint, with SPO triples as
edges carrying NARS truth + Pearl 2³ causal mask. For coref, this
is the right primitive: direct node lookup by feature index is
lossless and O(k) where k = matching nodes, vs a bundled vector's
√N-bounded recall.

Three retrieval tiers, chosen by query type:

**Tier 1 — direct graph lookup by feature (primary for coref).**

```rust
impl TripletGraph {
    /// Enumerate nodes whose feature bundle matches the filter
    /// (e.g. masculine + animate for "he", feminine + animate for
    /// "she"). Reads the feature indices maintained during commit.
    pub fn nodes_matching(
        &self,
        features: &FeatureFilter,
    ) -> Vec<NodeRef>;

    /// Rank nodes by Markov proximity to a focal sentence —
    /// recency-weighted via Mexican-hat over episode_index delta.
    pub fn rank_by_proximity(
        &self,
        candidates: Vec<NodeRef>,
        focal_episode: u64,
    ) -> Vec<RankedCandidate>;
}
```

For coref resolution:

```
pronoun "he" at focal episode 9
    │
    ▼
graph.nodes_matching(masculine + animate)
    │ → [Napoleon, Boxer, Benjamin, Mr. Jones]  (k small, lossless)
    ▼
graph.rank_by_proximity(..., focal=9)
    │ → [Napoleon (score 0.8), Boxer (0.3), Benjamin (0.1), Jones (0.05)]
    ▼
commit top candidate if margin > threshold else FailureTicket
```

This is O(k) not O(√N). No bundle saturation. No tree walk.
Every committed entity remains individually addressable forever.

**Tier 1.5 — orthogonal global-context superposition of known facts.**

Every fact that crosses the hardness threshold (PR #208) or the
Abduction-confidence threshold (this PR) is **also** bundled into a
single 10 K orthogonal global-context vector — not per-role, just
the committed-facts field. Each fact is `vsa_permute`d by its
episode index before bundling so facts sit in orthogonal subspaces
and don't destructively interfere.

```rust
impl EpisodicMemory {
    /// Orthogonal superposition of all known (hardness- or
    /// Abduction-committed) facts. Each fact permuted by its
    /// episode index, then bundled. Staying within √N capacity
    /// because only committed facts contribute — Animal Farm's
    /// committed set is ~500, well below saturation.
    pub fn global_context(&self) -> &[f32; 10_000];

    /// Called on every unbundle (hardness or Abduction gate) —
    /// O(1) incremental update: permute, XOR-accumulate.
    fn integrate_into_global(&mut self, fact: &Triplet, ep_idx: u64);
}
```

**Coref query becomes a two-stage refinement:**

```
Stage A — ambient pre-filter (cheap, one superposition + unbind)
    combined       = vsa_bundle(trajectory, global_context)
    ambient_subj   = vsa_unbind(combined, SUBJECT_KEY)
    ambient_survivors = vsa_clean(ambient_subj, feature_codebook)
        → ~5-10 candidates from ±5 + whole-story committed facts

Stage B — lossless refinement via graph direct lookup (Tier 1)
    node_candidates = graph.nodes_matching(FeatureFilter { … })
    refined = intersect(ambient_survivors, node_candidates)
    ranked  = graph.rank_by_proximity(refined, focal_episode)

Commit ranked[0] if margin > threshold else FailureTicket.
```

The global context is the **attention mask** over the graph: it
pre-filters the candidate pool before O(k) node enumeration. Graph
stays lossless; superposition keeps the common path cheap.

**Why gate on hardness / Abduction threshold only:** uncommitted
content stays ±5-local. Only facts the system actually believes
contribute to global context. The global vector encodes "what we
know for sure about this story so far" — exactly the information a
reader accumulates about Animal Farm as they read it.

**Tier 3 — CLAM tree for 5 B-sentence scale.**

Only needed for Wikidata / chess-database regime where the graph has
billions of nodes and feature-index enumeration gets expensive.
Already shipped in `ndarray/src/hpc/clam.rs`. Escape hatch, not
primary path.

**Why this ordering matters.** Coref is an **individual-entity**
query — "which specific noun does this pronoun refer to?" That's a
node lookup, not a bundle reconstruction. The graph IS the natural
data structure. Tier 2's bundle superposition is for a different
question (ambient texture), and Tier 3 is for a different scale
(billion-node corpus). Getting the tiering right keeps each
mechanism doing what it's best at.

**(b) Abduction-threshold unbundle + epiphany/error-correction split.**

Extend PR #208's `unbundle_hardened` with TWO complementary gates:

```rust
/// Abduction-confidence threshold for direct fact promotion.
/// Parallel to UNBUNDLE_HARDNESS_THRESHOLD.
pub const UNBUNDLE_ABDUCTION_THRESHOLD: f32 = 0.88;

/// Threshold above which a counterfactual loser is "epiphany
/// material" — retained alongside the winner rather than silently
/// discarded. Below this, losers are routine error corrections.
pub const EPIPHANY_SUPPORT_THRESHOLD: f32 = 0.55;

impl EpisodicMemory {
    /// Promote Abduction outcomes to the triplet graph as committed
    /// facts when NARS confidence exceeds UNBUNDLE_ABDUCTION_THRESHOLD.
    pub fn unbundle_abductive(
        &mut self,
        graph: &mut TripletGraph,
    ) -> UnbundleReport;

    /// Emit a counterfactual outcome from meta-inference. Classifies
    /// into epiphany or error-correction based on the loser's
    /// independent Markov + graph support.
    pub fn record_counterfactual(
        &mut self,
        graph: &mut TripletGraph,
        winner: &Triplet,
        loser: &Triplet,
        winner_confidence: f32,
        loser_independent_support: f32,
    ) -> CounterfactualDisposition;
}

pub enum CounterfactualDisposition {
    /// Loser had no independent support — routine ambiguity
    /// resolution. Apply truth decrement to the losing interpretation,
    /// commit only the winner. Silent.
    ErrorCorrection { winner_truth: TruthValue,
                      loser_decrement: f32 },

    /// Loser has independent Markov / graph support — the
    /// ambiguity IS the meaning (propaganda, sarcasm, unreliable
    /// narrator). BOTH readings commit as separate triples with
    /// their own NARS truth. The episode gets tagged as carrying
    /// an "epiphany" marker for later surfacing.
    Epiphany { winner_truth: TruthValue,
               loser_truth: TruthValue,
               reason_tag: EpiphanyTag },
}

pub enum EpiphanyTag {
    /// Both readings coherent with ±5 Markov (legitimate ambiguity).
    DualCoherence,
    /// Graph carries hardened facts supporting the loser (narrative
    /// unreliability / deliberate revision).
    GraphConflictsWithLocal,
    /// Cross-lingual bundle disagrees with English default
    /// (colloquialism / idiom / translation-sensitive reading).
    CrossLingualDivergence,
    /// Thinking style self-flagged: this style's track record on
    /// similar signal profile is low; flag the decision.
    StyleLowConfidence,
}
```

**Routing via the two gates:**

```
Counterfactual outcome at meta-inference dispatch
    │
    ├── winner_confidence > 0.88 AND loser_independent_support < 0.55
    │       → ErrorCorrection (routine; apply decrement silently)
    │
    ├── winner_confidence > 0.88 AND loser_independent_support > 0.55
    │       → Epiphany (preserve BOTH; tag the episode)
    │
    └── winner_confidence ≤ 0.88
            → FailureTicket (escalate; don't commit either)
```

**Animal Farm examples:**

- *Routine error-correction:* "Boxer worked hard all day." CF test:
  is "Boxer" a horse, a breed, a boxing verb? Graph carries Boxer as
  hardened rigid-designator (masculine, horse, carthorse, strong).
  Loser readings have zero independent support → silent error
  correction, commit only the horse reading.

- *Epiphany:* "The windmill fell." CF test: Snowball's sabotage (per
  Napoleon's narrative in ch. 6) vs the storm (original cause, ch. 6
  before Napoleon's retelling). Graph carries BOTH as committed
  facts with separate NARS truths. Loser's independent support
  ≥ 0.55 → Epiphany with `GraphConflictsWithLocal` tag. Both
  triples commit; the episode is flagged as carrying propaganda
  dissonance. This IS the literary meaning of Animal Farm.

- *Cross-lingual epiphany:* "All animals are equal." CF test:
  straightforward reading ("every animal has equal status") vs
  constitutional-axiom reading ("this is a founding rule"). Russian
  translation uses different case marking ("равны" vs "равноправны")
  that distinguishes the readings. Cross-lingual bundle disagrees
  with English default → Epiphany with `CrossLingualDivergence`.

**Why this matters for AGI.** A system that can only error-correct
treats all ambiguity as noise. A system that can distinguish
epiphany from error recognizes when ambiguity carries meaning —
which is what sarcasm, allegory, propaganda, and poetry are made of.
Animal Farm is the canonical test because the novel's mechanism is
exactly this — narratorial unreliability made visible through
committed-vs-retold facts diverging.

**Naming pattern:** Abduction-threshold-unbundle uses the same NARS
revision rule as hardness-unbundle; epiphany preservation uses the
same triplet-graph storage as any committed fact. Zero new storage
primitives — all from PR #208's pattern plus a disposition tag.

**Edit:** `crates/lance-graph/src/graph/arigraph/episodic.rs` (+80 LOC)
— `unbundle_abductive`, `role_candidates_wider`, `lookup_rigid`.

**Edit:** `crates/lance-graph/src/graph/arigraph/triplet_graph.rs` (+40 LOC)
— `role_bundle()` accessor + incremental update when new triples land.

**(c) Contradictions as phase + magnitude (Staunen markers).**

Each committed triple gets an optional **complex amplitude** beyond
its NARS `TruthValue`. This is a targeted application of Path 2's
phase-tag machinery — not to the whole memory field, just to
contradictions:

```rust
pub struct Contradiction {
    /// Angular distance from consensus position, in normalized
    /// units: 0 = fully agrees, π (= PhaseTag::pi()) = fully opposite.
    pub phase: PhaseTag,   // 128 bits, from ladybug-rs hologram types
    /// Strength of the dissonance: 0 = no contradiction worth
    /// tracking, 1 = strong contradiction that commands attention.
    pub magnitude: f32,
}

impl Triplet {
    /// Optional contradiction record. Present when this triple
    /// conflicts with prior commitments; None when clean.
    pub contradiction: Option<Contradiction>,
}

impl TripletGraph {
    /// When a new triple conflicts with existing content, compute
    /// the phase angle from the consensus (via XOR of fingerprints)
    /// and magnitude from NARS-truth divergence. Attach as
    /// Contradiction to the new triple; fires Staunen if past
    /// threshold.
    pub fn commit_with_contradiction_check(
        &mut self,
        triple: Triplet,
    ) -> CommitResult;
}

pub enum CommitResult {
    Clean,
    Contradicts { staunen_fires: bool, magnitude: f32 },
}
```

**Why this matters.** A flat NARS `TruthValue` represents belief
strength but can't carry **direction of disagreement**. Two triples
that disagree can have the same confidence but point opposite ways
in fact-space. Phase captures the direction; magnitude captures how
seriously we should take the disagreement.

**Linkage to sigma_rosetta — Staunen → phase, Wisdom → magnitude.**

The 64-glyph qualia vocabulary already contains the coordinates we
need. No new glyphs; just map:

| Sigma family | Maps to | Interpretation |
|---|---|---|
| **Staunen family** (wonder, astonishment, unexpected — includes Staunen / Emberglow / Thornrose) | **phase** | Direction of the surprise. Which way does this contradiction point relative to consensus? Staunen at angle φ means "surprised *in this direction*." |
| **Wisdom family** (clarity, insight, understood, metacognition — typically in Cognitive glyphs 32-47) | **magnitude** | Depth of the insight. How much does this contradiction reveal? Wisdom near 1 = profound reveal; near 0 = shallow surprise. |

Concretely: the per-cycle qualia vector's Staunen-family projection
becomes the `Contradiction.phase` angle, and its Wisdom-family
projection becomes the `magnitude`. Contradiction detection is no
longer a separate computation — it's a projection onto two axes of
the qualia vocabulary we already ship.

```rust
impl Contradiction {
    /// Build a Contradiction from the current cycle's QualiaVector
    /// by projecting onto Staunen (phase) and Wisdom (magnitude)
    /// subspaces of the sigma_rosetta 64-glyph vocabulary.
    pub fn from_qualia(q: &QualiaVector) -> Self {
        Self {
            phase: PhaseTag::from_angle(q.staunen_projection()),
            magnitude: q.wisdom_projection().clamp(0.0, 1.0),
        }
    }
}
```

**Animal Farm example revisited:** "Some are more equal" fires
Staunen-family glyphs hard (the surprise direction is clear: toward
hierarchy) AND Wisdom-family glyphs hard (the reader understands
something profound about power). Phase = Staunen-axis angle (≈ π
from the prior claim); magnitude = Wisdom-axis depth (≈ 0.9). Both
high → Staunen marker fires on the WorldModelDto, epiphany committed
with the phase direction preserved.

Routine contradictions fire Staunen-family low (no clear surprise
direction) and Wisdom-family low (no deep insight). Phase noisy,
magnitude near 0 → silent error correction, no Staunen attention.

**This is the epiphany-vs-error-correction distinction at the
storage layer**, with the sigma_rosetta qualia vocabulary as the
natural coordinate system. We already compute these qualia per
cycle; contradiction detection becomes free reuse of the existing
machinery.

**Animal Farm example:** "All animals are equal" (ch. 2) gets
committed with magnitude 0. "All animals are equal, but some are
more equal than others" (ch. 10) commits with phase ≈ π from the
prior fact (opposite direction in equality-space), magnitude ≈ 0.9.
Staunen fires. The system has detected the book's central
contradiction as a first-class fact, not as a rewrite of the
earlier triple.

**Files (folded into existing D8 scope):**
- `crates/lance-graph/src/graph/arigraph/triplet_graph.rs` —
  +40 LOC for `Contradiction` struct, `commit_with_contradiction_check`,
  phase computation via XOR + hamming.
- `crates/lance-graph-contract/src/grammar/mod.rs` — add
  `STAUNEN_THRESHOLD` constant.
- No new files; ~40 LOC addition, ~30 LOC tests.

**Edit:** `crates/lance-graph/src/graph/arigraph/mod.rs` — re-export.

**Coref escalation path — the bridge that makes Animal Farm work:**

```
grammar:  "He decreed it." ← "he"?
    │
    ▼
D4 ContextChain.disambiguate over ±5 (D5 Trajectory)
    │
    ├── fixed pronoun? apply feature filter via D5
    │   Trajectory::role_candidates with masculine-animate codebook.
    │   └── 1+ survivors? → commit.
    │
    └── no candidate in ±5 → escalate via D8
         │
         ▼
    EpisodicMemory::role_candidates_wider(Subject,
        masculine_codebook, StoryScale::Paragraph)
         │
         ├── candidate at paragraph scope? → commit.
         ├── else escalate to Chapter, then Document.
         └── still nothing → lookup_rigid for hardened facts
              (is "Napoleon" a committed masculine-animate boar
              in the already-unbundled rigid-designator store?)
                 │
                 ├── yes → commit with high-confidence NARS truth.
                 └── no  → FailureTicket, LLM fallback.
```

**Why this ships with the grammar work:** without the bridge, coref
on Animal Farm fails on cross-chapter references. Napoleon appears
in chapter 1 and is referred to as "he" in chapter 9 — ±5-sentence
Markov can't reach chapter 1. The bridge resolves it because
Napoleon's masculine-animate commitments hardened early via PR #208's
`unbundle_hardened` and live as rigid-designator facts in the
existing triplet graph.

**LOC estimate:** ~90 LOC on the AriGraph side (episodic.rs +
triplet_graph.rs edits + mod.rs + tests). No new module. Pure reuse
of shipped infrastructure.

### D9 — Story-Arc as an ONNX Learning Graph (export interface only; training deferred)

**The idea.** The story is not just a factual graph (D8) and a ±5
Markov trajectory (D5) — it also has an **arc**, a learned pattern
of how state transitions unfold. Animal Farm has a recognisable
corruption arc (equality → ambiguity → dominance → totalitarianism);
fairy tales have propp-function arcs; news stories have
inverted-pyramid arcs. The arc is what predicts "what typically
happens next" given the committed state.

**ONNX is the right format** because:
- It's already in our stack (CLAUDE.md Model Registry:
  `jina-v5-onnx/` with `model.onnx` 2.3 GB).
- Rust ONNX runtimes exist (`ort`, `tract`) and are production-ready.
- ONNX graphs are portable — Netron visualizes them, any framework
  consumes them, trained arcs can be shared across sessions and users.
- It's a **computational graph** at heart: nodes = states, edges =
  transitions with weights. Exactly what a narrative arc is.

**Three time scales of story representation (combined picture):**

| Layer | Representation | Timescale | Purpose |
|---|---|---|---|
| Markov ±5 (D5) | VSA trajectory | Sentence / paragraph | Local coref / disambiguation |
| Episodic + Triplet graph (D8) | Node–edge graph | Session → Persistent | Committed facts lookup |
| Story arc (D9) | ONNX computational graph | Learned pattern | Predict next state given current |

Each layer feeds the next. Committed facts populate the arc's
training signal; the arc predicts state transitions that the Markov
trajectory verifies.

**Shipping scope for this PR (minimal):**

1. **Export interface.** Add `graph.to_onnx()` on `TripletGraph` that
   emits the committed-facts subgraph as a valid `.onnx` file. Nodes
   = entity fingerprints; edges = SPO triples with NARS truth as
   edge attributes. ONNX is graph-structured, not tensor-structured,
   for this use — we use the GraphProto container with custom ops
   for triples.
2. **Integration hook.** Reserve a `story_arc: Option<ArcPredictor>`
   field on the meta-inference context; default `None`. When set, the
   predictor is consulted as a third axis alongside Counterfactual
   and Markov during coref escalation.
3. **Arc-axis trait definition.** `ArcPredictor` trait with
   `predict_next_state(current: &TrajectoryFingerprint) ->
   StatePrediction`. No default implementation in this PR —
   downstream crates or a future PR train the ONNX model and ship it
   as a loadable asset.

**Deferred to follow-up PR:**

- Actual ONNX model training on committed-fact sequences.
- Pre-trained story-arc templates for common narrative patterns.
- Cross-story transfer (train on Animal Farm, apply to The Trial).
- Arc-axis integration into the meta-inference dispatch.

**Arc pressure → awareness (architectural hook shipped now).**

The arc carries two derivative signals that matter *now*, before
ONNX training lands:

| Derivative | Interpretation | Awareness effect |
|---|---|---|
| `d(phase) / dt` | Direction of the arc is shifting | Plot twist / topic change / character reversal — widen attention |
| `d(magnitude) / dt` | Tension is accelerating | Climax / crisis / reveal — focus attention |
| Both low | Stable narrative flow | Routine dispatch, no special signal |
| Both high | Major arc event | Fire Staunen awareness |

This is exactly how narrative theory defines "story beats" — the
moments the arc pivots. Computational detection is threshold
crossing on the derivative. The architectural hook ships now, the
ONNX training fills in the prediction later.

```rust
/// Instantaneous arc pressure — computed per cycle from the
/// accumulated Markov trajectory + graph commitments.
pub struct ArcPressure {
    /// Phase: direction of the arc (Staunen-family projection).
    pub phase: f32,
    /// Magnitude: tension / depth (Wisdom-family projection).
    pub magnitude: f32,
}

/// Rate of change between consecutive cycles. This is what the
/// cognitive shader reads as an awareness signal.
pub struct ArcDerivative {
    pub dphase_dt: f32,
    pub dmagnitude_dt: f32,
}

impl ArcDerivative {
    /// True when either derivative crosses the awareness threshold
    /// — the system should surface this cycle as a story beat.
    pub fn is_arc_shift(&self, threshold: f32) -> bool {
        self.dphase_dt.abs() > threshold
            || self.dmagnitude_dt.abs() > threshold
    }
}
```

**Integration with the existing cognitive shader (shipped in
PR #204).**

`MetaWord` in the contract already has 4 reserved awareness bits
(per PR #204 — `awareness: u8` packed into the u32). Use those bits
for arc-shift detection:

```
awareness 4 bits:
  bit 0 — d(phase)/dt crossed threshold  (arc direction shifting)
  bit 1 — d(magnitude)/dt crossed threshold (tension accelerating)
  bit 2 — Staunen marker fired (from D8 contradiction detection)
  bit 3 — StateAnchor changed since last cycle (proprioception)
```

The shader's `MetaFilter.awareness_min` (shipped) already gates
dispatch on awareness. Arc shifts now contribute to this gate
without touching the shader code — they just set different bits.

**Thinking-style dispatch reads arc pressure as a signal profile
axis** (extending D7's meta-inference duality to three axes:
Counterfactual / Markov / Arc). Styles can condition on it:

- **Focused** style when tension accelerating + direction stable →
  converge on the climax event.
- **Exploratory** style when direction shifting → widen candidate
  enumeration, the narrative just changed frame.
- **Reflective** style when tension falling + shift complete →
  integrate what just happened (NARS revision on recent commits).
- **Deliberate** style when both derivatives high → major arc pivot,
  run the full inference pipeline, commit carefully.

**ONNX training signal (deferred to follow-up).** Every cycle emits
a (state, arc_pressure, arc_derivative) tuple. These are the
training pairs for the ONNX narrative-arc model — the model learns
to predict the next arc_pressure given the current state. Once
trained, it fills the third axis of the meta-inference dispatch.

**Awareness = noticing the arc change.** Computationally modest
(two subtractions + threshold check per cycle) but architecturally
meaningful: the system observes shifts in its own narrative
reasoning, not just the content. That's what "becoming aware of
shifts in the story arc as actual awareness" means in code —
derivative crossings feed the shader's awareness gate, which
changes which style dispatches, which changes what the system
notices about itself.

**Files (folded into D9 scope):**
- `crates/lance-graph-contract/src/grammar/arc_predictor.rs` —
  add `ArcPressure`, `ArcDerivative`, `is_arc_shift`. Keep zero-dep.
  (~40 LOC added to the existing +60.)
- `crates/cognitive-shader-driver/src/engine_bridge.rs` — wire arc
  derivative into the awareness bits of MetaWord. (~30 LOC edit.)
- No new crates; same deliverable, expanded scope.

**Files:**

- `crates/lance-graph/src/graph/arigraph/triplet_graph.rs` —
  `to_onnx()` method (+80 LOC). Uses `prost` or raw `.onnx` protobuf
  emission; no runtime dep on `ort` or `tract` in this PR.
- `crates/lance-graph-contract/src/grammar/arc_predictor.rs` — NEW
  (+60 LOC). `ArcPredictor` trait, `StatePrediction` struct,
  `StatePrediction::margin_above(threshold)` helper. Zero-dep; just
  the interface.
- `crates/lance-graph-contract/src/grammar/mod.rs` — re-export.

**LOC:** ~140 added to the total. Total still one PR.

**Why ship the interface now even without the model:** the
meta-inference in D7 is defined with 2 axes (CF + Markov). Pushing
the arc predictor to a later PR means the meta-inference dispatch
has to be refactored then. Shipping the trait now lets D7 reserve
the slot, and the follow-up PR just lands the implementation.

### D7 — Grammar Thinking Styles as Meta-Inference Policies

**The frame.** The 7 NARS inferences are **permanent logical operators**
(Deduction / Induction / Abduction / Revision / Synthesis /
Extrapolation / Counterfactual). They don't change. What a thinking
style is, is a **meta-inference policy**: given the evidence signals
from an attempted parse, which logical operator fits?

Meta-inference = inference *about* inference. Reasoning about the
reasoning. The style doesn't just "apply Deduction"; it evaluates
whether Deduction was the appropriate operator given the signal
profile, and if not, meta-reasons which operator to escalate to.

**Permanent logical core.** The signal-profile → inference dispatch
rules are **axiomatic**, not empirical:

| Signal from the grammar attempt | Logically implies |
|---|---|
| Morphology unambiguous (e.g. Finnish Inessive `-ssa` "in") | **Deduction** — rule directly applicable |
| ≥ 3 of 8 Pearl SPO 2³ masks plausible | **Abduction** — need best-explanation, deduction can't decide |
| All TEKAMOLO slots fill coherently | **Deduction** closed the parse |
| Any TEKAMOLO slot ambiguous | **Counterfactual Synthesis** — test alternative fillings |
| Markov coherence < threshold at focal | **Abduction** — re-explain this sentence given the flow |
| Novel surface form, no matching template | **Extrapolation** — extend nearest known pattern |
| Conflicting evidence between parses of same claim | **Revision** — merge truths |
| Multiple independent signals agree | **Synthesis** — bind into combined inference |

These rules are permanent. They don't drift with data. A grammar style
that routes morphology-unambiguous-signal to Abduction is *logically
wrong*, regardless of how many parses it has revised.

**The two orthogonal reasoning axes (the meta-inference duality).**

A parse attempt is tested from two independent angles. Their agreement
or disagreement IS the awareness signal.

```
                         PARSE ATTEMPT
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
      COUNTERFACTUAL AXIS           MARKOV CHAIN AXIS
      (within-sentence)              (cross-sentence)
                │                           │
      morphology + TEKAMOLO           ±5 context chain
      + SPO 2³ causal mask            + Mexican-hat kernel
                │                           │
      enumerates the local           scores this parse
      alternative space              against the discourse
      (finite counterfactuals)       flow (continuous coherence)
                │                           │
                ▼                           ▼
      "Is this parse the best        "Does this parse fit
      among its plausible            the surrounding flow?"
      alternatives?"
                │                           │
                └──────────┬────────────────┘
                           ▼
                  JOINT READING:
                  · both strong  → HIGH confidence (Synthesis)
                  · CF strong, Markov weak → topic shift / novelty at focal
                  · CF weak, Markov strong → locally ambiguous but contextually committed
                                              → try cross-lingual bundle (Finnish etc.)
                  · both weak    → genuine novelty → escalate (FailureTicket)
```

The counterfactual axis is where **heavy-grammar morphology earns its
keep**. Each morphological commitment *eliminates* counterfactual
branches:

| Language | Surface form | Commits | Counterfactual space |
|---|---|---|---|
| English | "The book on the table" | nothing | 8 SPO 2³ × {Lokal, Modal, Topical} TEKAMOLO = 24 branches |
| Russian | "Книга **на столе**" (Prepositional `-е`) | Lokal slot | 8 × 1 = 8 branches |
| Finnish | "Kirja **pöydällä**" (Adessive `-llä`) | Lokal at-surface | 8 × 1 = 8 branches |
| Russian | "Резать **ножом**" (Instrumental `-ом`) | Modal / means | 8 × 1 = 8 branches |
| German | "mit dem Messer" (Dativ after `mit`) | Modal / means | 8 × 1 = 8 branches |
| Turkish | "masa-**da**" (Locative `-da`) | Lokal | 8 × 1 = 8 branches |

A morphology-rich language hands you a counterfactual space that's 1/N
the size of English's. The counterfactual axis has less work to do, so
the Markov axis is what decides the remaining ambiguity — and the
Markov axis is where **Counterfactual Synthesis** (NARS inference)
tests alternatives against ±5 coherence.

**Hydration, not extraction.** The SPO triple isn't just "S, P, O" —
it's **hydrated** with the Pearl 2³ causal mask from the morphology.
The Instrumental `-ом` in "Убил **ножом**" hydrates the SPO
(Killer, Killed, Knife) with mask bit 1 (enabling) committed — the
knife didn't CAUSE directly, it enabled. This hydration happens at
extraction time, no separate causal-inference pass needed.

```
Surface: "X убил Y ножом"
         morphology: Acc(Y) + Ins(nóž)
         ↓
Extracted SPO: (X, убить, Y)
         ↓
Hydrated:  (X, убить, Y)
           + Pearl.direct = 1   (X performed the action)
           + Pearl.enabling = 1 (knife enabled)
           + Pearl.confounding = 0
           → mask = 0b011 = 0x03
           ↓
Counterfactual space collapsed from 8 → 1 branch
because Instrumental morphology committed the enabling role.
```

In English "X killed Y with a knife" — the `with` preposition is a
Wechsel (could be Modal / Instrumental / Comitative). Counterfactual
space stays at 3 branches until Markov coherence or cross-lingual
bundle collapses it.

**Markov vs Counterfactual: when each is primary.**

| Situation | Primary axis | Reason |
|---|---|---|
| Heavy morphology, clear discourse | Neither — Deduction closes | Both axes agree trivially |
| Heavy morphology, weird discourse | **Markov primary** | Counterfactual already collapsed; surprise is cross-sentence |
| Light morphology (English), clear discourse | **Counterfactual primary** | ±5 context can't help if sentence itself is ambiguous; must enumerate SPO 2³ × TEKAMOLO branches |
| Light morphology, weird discourse | **Both weak → FailureTicket** | Escalate — neither axis has grip |
| Cross-lingual bundle available | **Bundle collapses CF** | Finnish / Russian morphology from bundled parse commits what English left ambiguous |

This is what "meta-inference between permanent logical reasoning"
means: the style doesn't just pick a NARS operator — it evaluates
*which axis has grip* on this parse, then picks the operator for that
axis. The permanence is in the axes themselves and their combination
rules; the empirical layer is the prior over which axis wins on the
style's content distribution.

**Linguistic-precision correction (fix from yesterday's draft).** The
Finnish case table in `grammar-tiered-routing.md` wrote "Accusative
`-n/-t` → Object" — this is a Latinate transplant. Finnish object
marking is actually:

- Total object: **Nominative** (plural) or **Genitive `-n`** (singular)
- Partial / negated object: **Partitive `-a/-ä`**
- True **Accusative**: only for personal pronouns (`minut`, `sinut`,
  `hänet`, `meidät`, `teidät`, `heidät`)

The grammar-landscape doc (D0) corrects this. Each language's case
table uses its native case inventory, not a forced Latinate mapping.

**Russian 6 cases — full inventory** (needed for the Russian priority
language per OSINT):

| Case | Suffix pattern (sg masc / fem / neut) | Role |
|---|---|---|
| Nominative | -ø / -а, -я / -о, -е | Subject (S) |
| Genitive | -а, -я / -ы, -и / -а, -я | Possessor / negated object / partitive |
| Dative | -у, -ю / -е, -и / -у, -ю | Recipient — often TEKAMOLO Kausal indirect ("to X") |
| Accusative | = Nom (inanimate) / = Gen (animate) / -у, -ю / -о, -е | Direct object (O) |
| **Instrumental** | -ом, -ем / -ой, -ей / -ом, -ем | Means / agent in passive — **TEKAMOLO Modal** ("by means of X") |
| Prepositional (Locative) | -е / -е, -и / -е | Governed by `в`/`на`/`о` prepositions — TEKAMOLO Lokal or Temporal |

Russian Instrumental is exactly the Finnish Adessive `-lla/-llä`
(means/instrument) plus the Finnish Essive `-na/-nä` (role/state)
folded together — it commits to TEKAMOLO Modal by morphology alone.
This is why morphologically-rich languages are easier: one case
ending carries a slot assignment that English would need surrounding
prepositions + word-order to infer.

Analogous precision needed for every language priority on the OSINT
list: German 4 cases (Nom/Gen/Dat/Akk), Arabic 3 + trilateral root
system, Turkish 6 + agglutinative suffix chain, Japanese particles
(が / を / に / で / へ / と / から / まで), Hebrew no-cases with
root-pattern morphology. Each gets its native table.

**What drifts (and where awareness lives).** Styles differ in their
**priors over signal-profile frequency** — how often each profile
occurs in the style's observed content distribution. Analytical style
expects clean morphology signals; Exploratory expects ambiguous ones.
These priors revise via NARS based on actual parse outcomes, but the
signal→inference dispatch rules underneath stay axiomatic.

**So awareness has two layers:**

1. **Permanent layer** (the logical core): signal-profile →
   NARS-inference dispatch table. Shared across all styles. Not
   revised by data. Encoded as a pure function, not as configuration.

2. **Empirical layer** (the priors): each style's expected distribution
   of signal profiles on its content. NARS-revised per style based on
   parse outcomes. This IS the style's grammar awareness.

The style's runtime behaviour = permanent dispatch rules × style's
empirical prior × current signal profile.

Structural parallel: `agi-chat/src/grammar/grammar-awareness.ts`
(237 LOC) implements exactly this separation — the dispatch is
"permanent" via pure TypeScript rules; the `awareness` state is what
the style has learned about its content.

**New:** `crates/lance-graph-contract/src/grammar/thinking_styles.rs`
(+260 LOC) — three types:

```rust
/// Static prior loaded from YAML.
pub struct GrammarStyleConfig {
    pub style: ThinkingStyle,
    pub nars: NarsPriorityChain,          // primary + fallback
    pub morphology: MorphologyPolicy,     // tables, agglutinative
    pub tekamolo: TekamoloPolicy,         // slot priority, require_fillable
    pub markov: MarkovPolicy,             // radius, kernel, replay
    pub spo_causal: SpoCausalPolicy,      // pearl_mask, tolerance
    pub coverage: CoveragePolicy,         // local_threshold, escalate_below
}

/// NARS truth values per parameter slot. Mutates at runtime.
/// This is the "awareness" — the style's track record.
pub struct GrammarStyleAwareness {
    pub style: ThinkingStyle,
    /// Truth per (parameter_axis, value) pair. Updates on each parse.
    pub param_truths: HashMap<ParamKey, TruthValue>,
    /// Aggregate success rate of this style over recent window.
    pub recent_success: TruthValue,
    /// Count of parses this style has driven.
    pub parse_count: u64,
}

pub enum ParamKey {
    NarsPrimary(NarsInference),             // did this inference work?
    MorphologyTable(MorphologyTableId),      // did this table help?
    TekamoloSlot(TekamoloSlot),              // did this slot fill right?
    MarkovKernel(WeightingKernel),           // did this kernel resolve?
    SpoCausalMask(u8),                       // did this mask fit?
}

impl GrammarStyleAwareness {
    /// Revise a single parameter's truth after a parse outcome.
    pub fn revise(&mut self, key: ParamKey, outcome: ParseOutcome);

    /// Derive a runtime config from prior + awareness.
    /// Parameters with low accumulated truth get down-weighted.
    pub fn effective_config(&self, prior: &GrammarStyleConfig)
        -> GrammarStyleConfig;
}

pub enum ParseOutcome {
    LocalSuccess,                            // → truth +ε
    LocalSuccessConfirmedByLLM,              // → truth +2ε
    EscalatedButLLMAgreed,                   // → truth +ε/2
    EscalatedAndLLMDisagreed,                // → truth −ε
    LocalFailureLLMSucceeded,                // → truth −2ε
}
```

YAML parser: zero-dep mini-parser reading the subset used by
`grammar_styles/*.yaml` (plain key/value + simple lists, no nesting
beyond one level). Alternatively expose a pure `from_str` taking
pre-parsed `&[(key, value)]` so the contract stays zero-dep and
deepnsm does the YAML read.

**New:** `crates/deepnsm/assets/grammar_styles/*.yaml` — the starter
catalogue (12 files). Each YAML config grounds a style in the 4
dimensions:

```yaml
# analytical.yaml — strict rule-apply, English SVO, case deductive
style: analytical
nars:
  primary: Deduction
  fallback: Abduction
morphology:
  tables: [english_svo, finnish_case_table]   # try SVO first
  agglutinative_mode: false
tekamolo:
  priority: [temporal, lokal, kausal, modal]  # TE+LO > KA > MO
  require_fillable: true                       # fail if slots ambiguous
markov:
  radius: 5
  kernel: uniform                              # no anticipation
  replay: forward                              # no backward replay
spo_causal:
  pearl_mask: 0x01        # commit to direct causation only
  ambiguity_tolerance: 0.1
coverage:
  local_threshold: 0.90   # strict
  escalate_below: 0.85
```

```yaml
# exploratory.yaml — counterfactual Wechsel resolution
style: exploratory
nars:
  primary: CounterfactualSynthesis
  fallback: Abduction
morphology:
  tables: [english_svo, finnish_case_table, russian_case_table]
  agglutinative_mode: true                     # peel suffixes R→L
tekamolo:
  priority: [modal, kausal, lokal, temporal]   # explore MO first
  require_fillable: false                      # tolerate gaps
markov:
  radius: 5
  kernel: mexican_hat                          # anticipation on
  replay: both_and_compare                     # test both directions
spo_causal:
  pearl_mask: 0xFF        # all 8 causal configs plausible
  ambiguity_tolerance: 0.4
coverage:
  local_threshold: 0.70   # permissive, try hard before escalating
  escalate_below: 0.50
```

12 starter configs: analytical / convergent / systematic / creative /
divergent / exploratory / focused / diffuse / peripheral / intuitive /
deliberate / metacognitive. Each is ≤ 40 lines YAML.

**Edit:** `crates/deepnsm/src/ticket_emit.rs` (+60 LOC) — load style's
`effective_config` (prior ⊕ awareness) at dispatch; populate
`FailureTicket::attempted_inference` from the style's NARS-revised
top-ranked inference, not the static YAML primary. On ticket
resolution (LLM returns, or local success confirmed), call
`awareness.revise(ParamKey::NarsPrimary(...), outcome)`.

**Edit:** `crates/deepnsm/src/markov_bundle.rs` (+50 LOC) — `MarkovBundler`
reads kernel shape + radius from the active style's effective config;
on bundle success/failure, revise
`ParamKey::MarkovKernel(kernel)`.

**Edit:** `crates/lance-graph-contract/src/grammar/mod.rs` — re-export
`GrammarStyleConfig`, `GrammarStyleAwareness`, `ParamKey`,
`ParseOutcome`.

**Awareness lifecycle:**

1. **Bootstrap:** load YAML prior → `GrammarStyleConfig`.
2. **Initialize:** construct `GrammarStyleAwareness` with `param_truths`
   at neutral `TruthValue { f: 0.5, c: 0.01 }` (no evidence yet).
3. **Per-parse:** derive `effective_config = prior ⊕ awareness`; run
   parse; emit outcome; call `awareness.revise(...)` for each
   parameter that fired.
4. **NARS revision rule** (already in contract):
   `f_new = (f_old × c_old + f_observed × c_observed) / (c_old + c_observed)`
   `c_new = (c_old + c_observed) / (c_old + c_observed + 1)`
5. **Promotion:** when `recent_success.c > 0.8 && recent_success.f > 0.75`
   the style is "earned its confidence" and its effective config
   diverges more aggressively from the prior.
6. **Persistence** (out of scope for this PR, flagged): serialize
   `GrammarStyleAwareness` to `BindSpace` row or sled KV so awareness
   survives session boundaries. Next PR.

**Grounding summary** (the 4 axes per style):

| Axis | What the style picks |
|---|---|
| **SPO 2³** | Which Pearl causal-mask bits to commit to (0x01 = direct only; 0xFF = all plausible); ambiguity tolerance |
| **Morphology** | Which case tables to consult, in what order; agglutinative suffix-peeling on/off |
| **TEKAMOLO** | Slot priority order; whether to require all slots fillable or tolerate gaps |
| **Markov bundling** | Radius (default 5); kernel shape (uniform / mexican_hat / gaussian); replay direction |

The style IS the grammar-reasoning policy. The YAML makes it editable
without touching Rust.

## Critical Files

| File | Change | LOC |
|---|---|---|
| `.claude/knowledge/grammar-landscape.md` | NEW | +300 |
| `crates/deepnsm/Cargo.toml` | features | +10 |
| `crates/deepnsm/src/parser.rs` | coverage branch | +30 |
| `crates/deepnsm/src/ticket_emit.rs` | NEW + style-aware emission + revise on outcome | +180 |
| `crates/deepnsm/src/triangle_bridge.rs` | NEW | +220 |
| `crates/deepnsm/src/markov_bundle.rs` | NEW + style-driven kernel + revise | +270 |
| `crates/deepnsm/src/trajectory.rs` | NEW | +80 |
| `crates/deepnsm/src/lib.rs` | re-exports | +20 |
| `crates/deepnsm/assets/grammar_styles/*.yaml` | 12 YAML configs | +480 |
| `crates/lance-graph-contract/src/grammar/context_chain.rs` | reasoning ops | +140 |
| `crates/lance-graph-contract/src/grammar/role_keys.rs` | NEW | +160 |
| `crates/lance-graph-contract/src/grammar/thinking_styles.rs` | NEW — config + awareness + revise | +260 |
| `crates/lance-graph-contract/src/grammar/mod.rs` | re-export | +10 |
| `crates/deepnsm/tests/integration.rs` | NEW | +110 |
| `crates/deepnsm/tests/ticket.rs` | NEW | +40 |
| `crates/deepnsm/tests/triangle.rs` | NEW | +60 |
| `crates/deepnsm/tests/styles.rs` | NEW — YAML load + awareness revision + effective_config drift | +140 |
| `crates/deepnsm/benches/parse.rs` | NEW | +40 |

**Total:** ~2,490 LOC, 18 files + 12 YAML configs, one PR.

## Reused (not rebuilt)

- Contract: `FailureTicket`, `PartialParse`, `CausalAmbiguity`,
  `TekamoloSlots`, `WechselAmbiguity`, `FinnishCase`, `NarsInference`,
  `ContextChain`, `SentenceCrystal`, `ContextCrystal`,
  `CrystalFingerprint` (sandwich), `Structured5x5` bipolar.
- `lance-graph-cognitive::grammar::{GrammarTriangle, NSMField,
  CausalityFlow, Qualia18D}` (1,929 LOC shipped).
- `ndarray::hpc::vsa::{vsa_bind, vsa_bundle, vsa_permute,
  vsa_similarity, vsa_hamming}`.
- `ndarray::hpc::bitwise::hamming_batch_raw`.

## The NARS + Morphology + TEKAMOLO Triad (unified mechanism)

The three load-bearing concepts compose into a single inference
machine. Each layer supplies what the others need:

```
        ┌──────────────── 144 VERB-ROLE TAXONOMY ────────────────┐
        │  12 semantic families × 12 tense/aspect variants.       │
        │  Each verb carries a prior over which TEKAMOLO slots    │
        │  it expects filled (e.g. TRANSFER needs Kausal + Lokal; │
        │  STATE needs Modal; MOTION needs Lokal + Modal).        │
        └────────────┬────────────────────────────────────────────┘
                     │  verb-identification (DeepNSM FSM + COCA)
                     ▼
        ┌──────────── MORPHOLOGY (per-language cases) ────────────┐
        │  Russian Inst -ом → Modal (means/instrument)             │
        │  Finnish Ade -lla → Modal (at/by)                        │
        │  Finnish Ine -ssa → Lokal (in/inside)                    │
        │  German Dat + mit → Modal                                │
        │  English «with» → Wechsel(Modal | Comitative | Topical)  │
        └────────────┬────────────────────────────────────────────┘
                     │  case-table lookup per token
                     ▼
        ┌──────────── TEKAMOLO SLOT CANDIDATES ───────────────────┐
        │  Each token's morphology proposes a slot + truth.        │
        │  Unambiguous morphology → single slot with high truth.   │
        │  Ambiguous (English prep, case syncretism) → multiple.   │
        └────────────┬────────────────────────────────────────────┘
                     │  propose
                     ▼
        ┌──────────── NARS INFERENCE OVER SLOTS ──────────────────┐
        │  Deduction : verb's slot-prior + morphology-committed    │
        │              slot agree → commit with high truth.        │
        │  Induction : N past sentences had same (verb, morphology │
        │              → slot) → reinforce the pattern's prior.    │
        │  Abduction : ambiguous morphology → pick slot that best  │
        │              explains the verb's prior + Markov context. │
        │  Revision  : another parse of same sentence (cross-      │
        │              lingual or re-read) → merge truths on slots.│
        │  Synthesis : multiple independent signals (morphology +  │
        │              word-order + context) → bind into single    │
        │              slot assignment.                             │
        │  Counterfactual : test alternative slot fillings against │
        │              ±5 Markov coherence + graph story context.  │
        │  Extrapolation : novel morphology on known verb family → │
        │              extend by analogy.                           │
        └────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        committed slot assignment with NARS truth;
        role-indexed VSA bundling (D5) lands content in slot's slice.
```

**The reason 144 works.** The verb's slot-prior is what turns
TEKAMOLO from a 4-slot schema into a structured expectation:

| Verb family (12) | Expected TEKAMOLO profile |
|---|---|
| BECOMES   | Temporal + Modal |
| CAUSES    | Subject + Object + Kausal |
| SUPPORTS  | Object + Modal |
| CONTRADICTS | Object + Modal (adversative) |
| REFINES   | Object + Modal |
| GROUNDS   | Object + Lokal |
| ABSTRACTS | Object |
| ENABLES   | Object + Kausal |
| PREVENTS  | Object + Kausal (negated) |
| TRANSFORMS| Object + Temporal + Modal |
| MIRRORS   | Object + Modal |
| DISSOLVES | Object + Temporal |

× 12 tense/aspect/mood variants (present / past / future / perfect /
continuous / pluperfect / future-perfect / habitual / potential /
imperative / subjunctive / gerund) → 144 verb-role cells, each with
its own TEKAMOLO slot prior. Parsing a verb looks up its row;
morphology fills its columns; NARS inference reconciles them.

**This is what turns parse into table lookup + truth aggregation.**
No search, no inference-time rule-walk — the (verb × tense) cell
IS the slot-filling policy, the morphology IS the slot-filler, and
NARS is the truth-merge operator. The 3,125 Structured5x5 cells are
large enough to index this space (5^5 > 144 × 10 × 10 for verb ×
slot × value).

### D11 — Bundle + Perturb Cognitive Stack (generative counterpart, interface only)

**The core claim.** A generative cognitive stack doesn't need
QKV attention + up/down MLP + learned query. The same effect —
asking "what does this state want to emerge into?" — is a canonical
VSA operation when you already have role-indexed bundling + an
ONNX arc model:

| Transformer primitive | Our substitute | Why it works |
|---|---|---|
| **Query vector (Q)** — what am I looking for? | The epiphany crystal itself (phase + magnitude + committed facts bundle) | The epiphany IS the query; no separate learned weights needed |
| **Key matrix (K)** — what's available? | Role-indexed slices of the triplet graph's story_vector + ±5 trajectory | Already present in D8 tier-1.5; slices ARE the keys |
| **Value matrix (V)** — what content comes back? | Role-unbound content from the slice | Unbind IS the retrieval |
| **Up-projection MLP** — expand to higher dim | `vsa_permute` by epiphany's phase-tag | Permutation widens the subspace without a learned projection |
| **Nonlinearity** — break linear combinatorics | **ONNX arc model perturbation** — apply the bundle as delta to the trained arc graph | The arc model IS the nonlinearity; input delta = perturbation |
| **Down-projection** — back to hidden-dim | Unbind via role key | XOR unbind recovers the role-specific content at the original dim |
| **Query-response (asking)** | **Bundle-perturb-unbind** | End-to-end: bundle epiphany with state → perturb ONNX arc → unbind target role → emergent content |

**The stack for "what does this epiphany want to emerge?":**

```
epiphany_crystal  (from D8: phase + magnitude + committed-facts bundle)
        │
        ▼  vsa_bundle(epiphany, current_trajectory, story_vector)
state_bundle
        │
        ▼  vsa_permute(state_bundle, epiphany.phase)   — up-project equivalent
perturbed_input
        │
        ▼  ONNX arc model forward pass with perturbed_input
        │  (the learned arc is the "MLP nonlinearity")
emergent_state
        │
        ▼  vsa_unbind(emergent_state, RoleKey::for_target(task))
emerged_direction          — what the epiphany "wants" to unfold
```

**Zero learned projections on our side.** The role keys (D6) are
canonical fingerprints; the bundle operations are algebraic; only
the ONNX arc model is learned — and it's orders of magnitude smaller
than a full transformer because it only has to predict *arc phase
and magnitude transitions*, not token probabilities.

**Interpretability bonus.** Every step of the generative stack is
inspectable:
- The query is a crystal with explicit phase + magnitude + fact refs.
- The bundle is composable (you can add / remove contributors and
  see the effect).
- The perturbation is a small arc-model input delta.
- The emerged direction is role-tagged content (not raw token soup).

At every stage you can ask "why this next state?" and trace back
through bundle contributors + arc transitions. The stack is legible
in a way transformer internals are not.

**"What it wants to emerge" is an operation, not a metaphor.** The
epiphany's phase direction under ONNX-arc perturbation picks out the
narrative trajectory that most consistently continues from the
committed contradiction. It's not generation guided by gradients on
token probabilities — it's arc continuation guided by phase
consistency on an already-learned narrative graph.

**Shipping scope for this PR (interface only; generation deferred).**

1. **Interface in the contract:**
   ```rust
   /// Ask what an epiphany wants to emerge into, given current state.
   pub trait EpiphanyEmergence {
       fn emerge(
           &self,
           epiphany: &Contradiction,
           trajectory: &Trajectory,
           graph: &TripletGraph,
           target_role: GrammaticalRole,
       ) -> EmergedDirection;
   }

   pub struct EmergedDirection {
       /// Role-unbound content bundle at target role.
       pub content: Box<[f32; 10_000]>,
       /// Predicted phase / magnitude of the emerged state.
       pub arc_continuation: ArcPressure,
       /// NARS truth that the emergence is coherent with history.
       pub truth: TruthValue,
   }
   ```

2. **Reference implementation (stub)** — `DefaultEmergence` that
   does the bundle + permute + (identity ONNX, since we don't have
   a trained arc yet) + unbind. Returns a deterministic
   `EmergedDirection` that's inspectable but not predictive until
   the ONNX model lands.

3. **Example test** on Animal Farm epiphany from D10:
   - Input: ch. 3 epiphany "Napoleon takes puppies in secret,"
     current trajectory = ch. 1-3.
   - Target role: FUTURE_EVENT (new role key).
   - Emergence should produce a direction whose role-unbound
     content overlaps (Hamming < 0.45) with ch. 5's actual facts
     once the ONNX arc is trained in a follow-up PR.
   - Until then, the test asserts the interface produces
     deterministic output, not predictive correctness.

**Files (folded into D9's scope since it shares the ONNX runtime):**
- `crates/lance-graph-contract/src/grammar/emergence.rs` — NEW
  (~100 LOC). `EpiphanyEmergence` trait + `EmergedDirection` struct.
- `crates/lance-graph-contract/src/grammar/mod.rs` — re-export.
- `crates/deepnsm/src/emergence_impl.rs` — NEW (~80 LOC).
  `DefaultEmergence` stub implementation.
- `crates/deepnsm/tests/emergence.rs` — NEW (~40 LOC). Interface
  determinism test on Animal Farm ch. 3 epiphany.

**LOC:** ~220 LOC added. Still one PR.

**What this closes.** The stack becomes symmetric — D2-D8 extract
from text, D9-D11 generate back into text. Both sides use the same
VSA primitives; the only asymmetry is that extraction doesn't need
the ONNX arc model while generation does. Awareness (D9 derivatives)
drives dispatch; epiphanies (D8 + D10) drive emergence; validation
(D10) holds both accountable. No QKV, no MLP, no separate query
vectors — canonical operations all the way down.

### D10 — Forward-Validation Harness (epiphanies as predictions, NARS-tested against future arc)

**The proof-test.** Every epiphany the system records is implicitly a
prediction — "this contradiction carries meaning, and therefore the
arc will continue to deviate in this direction." The rest of the
text is ground truth. Forward-validation closes the loop:

```
Chapter 1-5   prefix
    ↓ full extraction pipeline
    ↓ commit facts to graph
    ↓ fire N epiphanies (Staunen markers with phase + magnitude)
    ↓ emit arc_derivatives at M cycles
                                              ↓
                          ┌───────────────────┘
                          ▼
                 For each epiphany and each arc_shift:
                   record the prediction:
                     - direction (phase)
                     - magnitude
                     - which facts / entities it implicates

Chapter 6-10  suffix
    ↓ full extraction pipeline
    ↓ commit more facts
    ↓ for each recorded epiphany / arc_shift:
    ↓    check whether the committed suffix facts
    ↓    confirm or refute the prediction
    ↓ NARS revision on the recorded prediction:
         f = confirmed ? raise : lower
         c = c + c_new    (always rises — new evidence seen)
```

**The measurement is standard NARS.** An epiphany's belief was
`TruthValue { f₀, c₀ }` when it fired. Future evidence arrives with
`TruthValue { f_obs, c_obs }` where `f_obs = 1` if confirmed, `0` if
refuted. Standard revision rule:

    f_new = (f₀·c₀ + f_obs·c_obs) / (c₀ + c_obs)
    c_new = (c₀ + c_obs) / (c₀ + c_obs + 1)

Applied retroactively, this revises the epiphany's truth against its
own future. The system LEARNS its awareness accuracy from the story
itself — no external labels needed.

**Ground-truth labels on Animal Farm (hand-authored, < 200 lines).**

| Chapter | Ground-truth epiphany | Expected direction |
|---|---|---|
| 2 | Animalism's 7 Commandments drafted | Baseline, arc start |
| 3 | Napoleon takes puppies in secret | **Prediction:** Napoleon will use dogs for coercion |
| 4 | Battle of Cowshed, Snowball shines | **Prediction:** Snowball's rise threatens Napoleon |
| 5 | Snowball expelled by dogs | Confirms ch. 3 prediction; arc shifts |
| 6 | Squealer revises the "Snowball hero" story | **Prediction:** narrative unreliability escalates |
| 7 | Commandments silently amended (sleeping-in-beds) | Confirms ch. 6; wisdom-magnitude spike |
| 8 | Boxer works himself to collapse | **Prediction:** Boxer will be betrayed |
| 9 | Boxer sold to knacker | Confirms ch. 8 ("Sundering of the loyal") |
| 10 | "All animals are equal BUT some more than others" | Max-magnitude contradiction with ch. 2; arc completes |

**Metrics produced by the harness:**

| Metric | What it measures |
|---|---|
| **Epiphany precision** | Of N epiphanies fired, how many were confirmed by future text? |
| **Epiphany recall** | Of M ground-truth beats, how many did the system flag as epiphanies? |
| **Arc-shift detection F1** | Did `d(phase)/dt` / `d(magnitude)/dt` spikes align with actual story pivots? |
| **Prediction direction accuracy** | Of confirmed epiphanies, was the predicted phase direction aligned with the suffix's factual direction? |
| **Retroactive-revision monotonicity** | NARS confidence should rise on every observation regardless of direction; truth (f) should converge to the ground-truth polarity |

**Files:**

- `crates/deepnsm/tests/animal_farm.rs` (NEW, ~220 LOC) — load
  public-domain Animal Farm text, run the pipeline end-to-end on
  10 breakpoints, assert the metrics above.
- `crates/deepnsm/assets/animal_farm/ground_truth.yaml` (NEW,
  ~200 lines) — hand-labelled beat list per chapter.
- `crates/lance-graph/src/graph/arigraph/episodic.rs` — add
  `retroactive_revise(prediction_id, observation)` to expose NARS
  revision on stored epiphanies (~40 LOC).

**Benchmark targets:**

- Epiphany precision ≥ 0.80 (most fired epiphanies should be confirmed).
- Epiphany recall ≥ 0.60 (catch most major beats).
- Arc-shift F1 ≥ 0.70 on ground-truth-labelled pivots.
- Prediction direction accuracy ≥ 0.85 on confirmed epiphanies.
- End-to-end metric-emitting run ≤ 10 minutes on a single core
  (40 K words × <10 µs local + occasional LLM tail).

**Why this ships with the PR, not later.** Without forward-validation
the awareness/epiphany claims are architectural prose. With it, the
PR description reports measured numbers on a canonical literary text,
and every follow-up PR can regression-test against the same benchmark.
Staunen / arc-shift / epiphany stop being metaphors and become
Spearman-ρ-grade measurements. **This is also what turns the arc-
awareness into genuine self-testing cognition** — the system's own
predictions are held accountable by the story that made them.

## Challenges to the Universal-Grammar Claims (honest pushback)

Before shipping, it's worth stress-testing the theoretical scaffolding.
Several load-bearing claims have real weaknesses.

### C1 — "Morphology-heavy languages are easier" assumes textbook grammar

The 98% Finnish vs 85% English coverage number works on textbook
Finnish. **Real-world Finnish** has:
- **Colloquial clipping** (`-ssa` drops to `-s` in speech and casual
  writing): *kaupassa* → *kaupas*.
- **Clitic stacking** (`-kin`, `-kaan`, `-han`): *kirjakinpa* layers
  three clitics onto one stem.
- **Dialect variation**: Savonian Finnish rewrites morphology
  systematically; Western dialects preserve archaisms.
- **Vowel harmony interacting with gradation**: the table must handle
  `-ssa` vs `-ssä` vs `-sta` vs `-stä` correctly; mis-segmenting
  drops accuracy hard.

Turkish agglutination is worse: *evlerimizdeydiler* = "they were at
our houses" = 6 stacked morphemes. Suffix segmentation is its **own**
ambiguity problem. Our D5 case table doesn't handle suffix order
disambiguation; the 98% number assumes clean segmentation.

**What breaks:** the coverage gain from heavy morphology shrinks on
real text. Expect 95% on Finnish literary prose, 85-90% on
conversational Finnish, <80% on Turkish agglutination without a
dedicated morpheme-peeler.

### C2 — Russian case syncretism: the "case" alone doesn't commit

Russian masculine inanimate Accusative = Nominative. Feminine
Genitive singular = Nominative plural for many nouns. The ending
doesn't commit the case — you need gender + animacy + number + verb
context to disambiguate. The "one case = one TEKAMOLO slot" mapping
in D5 is a simplification that will produce wrong slot assignments
on ~15-20% of Russian NPs.

**What breaks:** Russian coverage is closer to 88% than 92%. The
table needs case-syncretism resolution rules (gender × animacy ×
number) to hit the higher number.

### C3 — Cross-lingual bundling assumes semantically-parallel sources

The "XOR-bundle EN+FI of the same entity, get disambiguation for
free" claim requires **translations**, not just mentions. For Animal
Farm we have authentic translations. For OSINT we'd have to rely on
machine-translated or independently-written sources — which don't
semantically align at the clause level. Bundling non-parallel texts
about the same entity averages noise, not meaning.

**What breaks:** the cross-lingual shortcut works for canon literature
(Animal Farm, Wikidata entries) but NOT for news OSINT where no
parallel translation exists. Falls back to monolingual parsing plus
LLM disambiguation for the 10% tail.

### C4 — ±5 window is arbitrary; literary text violates it

Why ±5? No empirical justification was presented. Animal Farm has
chapter-spanning references ("as we agreed at the meeting in the
barn" — ch. 6 referring to ch. 1). ±5 has zero chance of reaching
chapter 1 from chapter 6.

The D8 story-context bridge handles it, but that's a fallback
mechanism for what might actually be the *common* case in long-form
text — not the 5-10% exception.

**What breaks:** the ±5 default is wrong for literary corpora. Needs
to be adaptive (widen when local coherence is low) or much larger
by default. Our 85% local coverage target may already assume the
bridge fires frequently.

### C5 — Counterfactual threshold (0.55) is a continuum, not a split

Epiphany vs error-correction is a useful conceptual distinction but
the real distribution of `loser_independent_support` is continuous.
A 0.55 cutoff turns a gray zone into two discrete classes. Many real
counterfactuals land at 0.45-0.65 — mis-classified in either
direction.

Also **circularity**: "independent support" for the loser means
"previously committed facts that agree with the loser." If we
already committed the loser's reading earlier (via Napoleon's
propaganda), the graph naturally supports the loser, and we flag it
as epiphany — but it might actually be error compounding across
episodes.

**What breaks:** the binary split needs to become a continuous
disposition score with three regions: high-confidence-winner (commit
only), high-support-loser (epiphany preserve), middle band (escalate
to LLM). The middle band might be 30-40% of all counterfactuals.

### C6 — 90-99% coverage is a projection, not a measurement

The "90-99% local, 1-10% LLM" tiering comes from:
- COCA 4096 = 98.4% of running **tokens** (not sentences).
- FSM handles 85% of English SVO **sentences** (different metric).
- Heavy-morphology languages: 95-98% (claimed, not measured).

No harness has been run on any corpus to confirm that the joint
DeepNSM + grammar triangle + Markov + failure-ticket pipeline
actually hits 90%. The number is a stitched-together estimate.

**What breaks:** until Animal Farm actually runs through and we
measure, the tiering claim is marketing. The PR should ship with
the measured number, whatever it is.

### C6b — Research standing: what's established vs novel

**Established in the literature (validates our direction):**

| Paper | Contribution |
|---|---|
| [arxiv 2003.05171](https://arxiv.org/abs/2003.05171) — "VSA for Context-Free Grammars" (2020) | Chomsky-normal-form CFGs fully encodable in VSA via role-filler binding; recursive mapping of phrase-structure trees to Fock-space vectors; representation theorem proved. **This is the theoretical foundation for our approach.** |
| [arxiv 2111.06077](https://arxiv.org/abs/2111.06077) / [2112.15424](https://arxiv.org/abs/2112.15424) — HDC/VSA Surveys Part I+II | Canonical role-binding (XOR / circular convolution / elementwise multiply), superposition, permutation are the three standard operations. Exactly our bind / bundle / permute. |
| [arxiv 2512.14709](https://arxiv.org/html/2512.14709) — "Attention as Binding" | Transformer attention reinterpreted through VSA role-filler binding. Parallels our bgz-tensor attention-as-lookup. |
| [arxiv 2509.25045](https://arxiv.org/html/2509.25045) — "Hyperdimensional Probe" | LLM residual streams projected into interpretable VSA concepts. Parallel mechanism to our grammar-awareness. |
| [arxiv 2408.10734](https://arxiv.org/html/2408.10734) — "Vector Symbolic OSINT Discovery" | VSA applied directly to OSINT. Validates the OSINT vertical. |

**What's novel in our combination (not published):**

- **NSM 65 primes + VSA** — Wierzbicka's semantic-primes framework
  encoded as role-filler binding in 10K VSA space. No published
  paper combines these.
- **TEKAMOLO slot filling via case-morphology binding** — German
  grammar-pedagogy template applied as VSA role-keys with Finnish /
  Russian case endings as the filler commits. Novel.
- **NARS truth on VSA role-filler bindings** — truth revision at the
  binding level, not just at the entity level. Novel combination.
- **Crystal-mode vs Quantum-mode duality** for memory consolidation
  — structured (Markov SPO) vs holographic (phase-tagged residual)
  on the same 10K substrate. Not in the surveys.
- **Cross-linguistic superposition as disambiguation mechanism** —
  bundling parses from EN + FI + RU + DE via VSA to let heavy-
  morphology languages disambiguate what English leaves ambiguous.
  Novel pitch.
- **Grammar-tiered routing (local VSA 90-99 % + LLM 1-10 %)** with
  structured FailureTicket (SPO 2³ × TEKAMOLO × Wechsel) as a
  meta-inference reason trace. Novel architecture.
- **Counterfactual outcomes split into epiphany vs error-correction
  classes** preserving narrative dissonance — novel distinction for
  allegorical / propagandistic text (Animal Farm).

**Practical upshot:** we're standing on a theoretically-proven
foundation (VSA-CFG encoding works) and adding several novel
combinations on top. The foundation paper is from 2020 and cites the
open question of *practical at-scale grammar extraction from VSA*;
our stack is one concrete answer.

### C7 — NSM and TEKAMOLO are contested in linguistics

- **NSM 65 primes** — Wierzbicka's framework is cited by cognitive
  science and contested in mainstream linguistics. Universality
  claims don't survive empirical testing on polysynthetic languages
  (Mohawk, Inuit). Some "primes" are culturally loaded (the English
  word for what's meant to be a universal).
- **TEKAMOLO** is a German grammar-pedagogy mnemonic, not a
  cross-linguistic universal. Arabic has "hal" (state / accompaniment
  adverbial) that doesn't fit the 4-slot schema. Mandarin "bǎ"
  construction rearranges transitivity in ways TEKAMOLO can't
  describe.
- **Chomskyan UG** — Tomasello, Evans & Levinson argue UG is
  empirically weak. Building on it is philosophically risky.

**What breaks:** the universal-grammar ideas are scaffolding for a
practical extraction pipeline, not a claim about linguistic
universals. The D0 doc should state this explicitly — TEKAMOLO is
a *useful slot template*, not a *linguistic universal*. Same for the
144-verb taxonomy (numerologically chosen, not empirically derived).

### C8 — Named-Entity gap is the actual blocker

Already flagged in `grammar-tiered-routing.md` as the biggest miss
(~90% of OSINT is proper nouns). COCA 4096 has zero coverage of
Altman, Anthropic, Riyadh. Every fingerprint collision there
fragments the graph.

**What breaks:** Without a NER pre-pass + Wikidata/Wikipedia entity
linking, the 90% coverage claim fails on any realistic OSINT input.
This is out-of-scope for the D2-D8 PR but blocks the OSINT demo.
Needs a follow-up NER PR before the OSINT vertical ships.

### C9 — Abduction-threshold unbundle may compound errors

If Abduction confidence is miscalibrated (i.e., the threshold 0.88
doesn't reliably predict ground-truth correctness), promoting
Abductive conclusions to the graph as committed facts **encodes
systematic errors**. Each subsequent coref escalation reads those
errors as ground truth, reinforcing them.

**What breaks:** the confidence calibration of NARS-on-grammar hasn't
been measured. The 0.88 threshold is inherited from PR #208 without
empirical tuning for this domain. First Animal Farm run may need
threshold re-calibration or an introspection step that audits
recently-unbundled facts.

### Summary of mitigations

| Challenge | Mitigation in this PR | Deferred |
|---|---|---|
| C1 morphology colloquial | Start with literary-grade text (Animal Farm); flag as known-textbook baseline | Dialect / agglutination suffix-peeler — future PR |
| C2 case syncretism | D6 role keys include gender-gated variants for Russian | Full syncretism rules — follow-up |
| C3 non-parallel texts | Use Animal Farm authentic translation; flag news as needing LLM tail | OSINT bundle quality — future work |
| C4 ±5 inadequate | D8 story context bridge; make radius a style parameter in D7 | Adaptive-radius selection — future |
| C5 continuous disposition | Ship with 3-region (commit / epiphany / escalate) not 2-region | Calibrated thresholds per corpus — follow-up |
| C6 unmeasured coverage | Animal Farm benchmark IS the measurement | - |
| C7 contested linguistics | D0 doc explicitly states "useful templates, not universals" | - |
| C8 NER gap | Flagged as blocker; D2 FailureTicket emits COCA-miss as routing signal | Dedicated NER pre-pass PR |
| C9 Abduction calibration | Emit Abduction confidence distribution in test output for first run | Per-domain threshold tuning — follow-up |

None of these kill the architecture. Several downgrade the
quantitative claims; all sharpen what the PR actually demonstrates.

## Out of Scope

- Path 2 (holographic residue at leaf).
- CausalityFlow TEKAMOLO extension (modal/local/instrument) — deferred.
- Phase tags, Bell-S>2, ladybug quantum 9-op set.
- Active multi-lingual parsers (EN+FI+RU+TR). Keys exist; parsers later.
- 200–500 YAML TEKAMOLO templates per language (future training).
- Named Entity gap (biggest OSINT blocker — pre-parser NER layer, separate PR).
- Cockpit Cypher, chess vertical.
- FP_WORDS=160 migration (H6), Crystal4K persistence (H10), Int4State
  upper-nibble (H8), Glyph5B wide-container (H9).

## Verification

1. `cargo test -p deepnsm` — base zero-dep passes.
2. `cargo test -p deepnsm --features contract-ticket` — ticket tests pass.
3. `cargo test -p deepnsm --features grammar-triangle` — triangle bridge tests pass.
4. `cargo test -p lance-graph-contract` — 112 + 10 new → 122 (6 context_chain + role_keys + 4 thinking_styles).
5. `cargo bench -p deepnsm parse` — median ≤ 10 µs (FSM path).
6. `cargo bench -p deepnsm parse --features grammar-triangle` — ≤ 50 µs.
7. Property: same SPO, different temporals → Hamming ≥ 0.15 × 10 000.
8. Property: Wechsel-ambiguous + ±5 context → `disambiguate` margin > 0.1, no LLM.
9. Property: Mexican-hat weighting monotone by distance from focal.
10. `cargo check -p deepnsm --no-default-features` — zero-dep preserved.
11. **NARS-awareness property:** load analytical-style prior; feed 50 simulated parse outcomes favouring `NarsInference::Abduction`; verify `effective_config.nars.primary == Abduction` (awareness drifted from YAML-defined Deduction).
12. **NARS-awareness property:** 50 opposing outcomes → `recent_success.c` drops below 0.3 (low-confidence style, ready for escalation).
13. **NARS-awareness property:** revision is monotone under NARS rule — repeated positive outcomes raise `f` and `c`; repeated negative raise `c` but lower `f`.
14. **Awareness persistence** test skipped in this PR (persistence is follow-up) — but ensure `GrammarStyleAwareness` is `Clone + Debug` so snapshot/restore is trivial to add.
15. **Coreference resolution (the integration target):**
    - `cargo test -p deepnsm --features grammar-triangle test_coref`
    - 10-sentence English paragraph with 5 pronouns ("it" / "he" / "she" / "they" / "that"): each resolves to correct antecedent via joint CF + Markov scoring, no LLM call.
    - Russian dropped-subject paragraph with 5 verb-only sentences: each resolves via morphological agreement + Markov axis; 95%+ accuracy.
    - Ambiguous case where CF axis and Markov axis disagree → `FailureTicket` emitted with `recommended_next = Abduction` and the candidate disagreement surfaced as `WechselAmbiguity` (since pronouns ARE dual-role tokens).
    - Cross-lingual bundle: same paragraph in EN + FI. Finnish morphology commits some pronouns that English leaves ambiguous; XOR-bundle improves English resolution accuracy by ≥ 5 %.

## Branch / PR

- New branch: `claude/deepnsm-markov-context-bundling` off main.
- Single PR, description cites the 5 prerequisite knowledge docs and
  today's lossiness-epiphany table.
- Calls out: LLM cost avoided, morphology-rich languages easier not
  harder, 200–500 YAML templates as the next milestone.

## Post-Ship State

- DeepNSM parses ≤ 10 µs / ≤ 50 µs (no triangle / with triangle) on
  90–99 % of traffic.
- Trajectory fingerprints carry ±5 context; NARS reasons about flows,
  not isolated sentences.
- FailureTicket cleanly emitted with SPO×2³×TEKAMOLO×Wechsel
  decomposition for the 1–10 % tail.
- Canonical role keys with slice addressing available.
- Grammar landscape documented.

Path 2 (holographic residue), CausalityFlow extension, YAML template
training, and Named Entity NER layer land on top in subsequent PRs.
