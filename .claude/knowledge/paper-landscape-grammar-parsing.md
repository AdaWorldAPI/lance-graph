# Paper Landscape — Grammar Parsing × VSA × Active Inference

> **READ BY:** integration-lead, truth-architect, family-codec-smith,
> any agent touching deepnsm, grammar, AriGraph, or the free-energy
> resolution pipeline.
>
> **Created:** 2026-04-21
> **Scope:** Maps 14 recent papers onto the lance-graph grammar stack
> (DeepNSM + RoleKey VSA + FreeEnergy active inference + AriGraph).
> Each entry: citation, one-line finding, what it validates/challenges
> in our architecture, and the specific code cross-reference.

---

## Tier 1 — Foundational (directly validates our algebraic substrate)

### Shaw, Furlong, Anderson & Orchard (2501.05368) — VSA Category Theory Foundation

**Finding:** Right Kan extensions prove that dimension-preserving
binding/bundling MUST be element-wise operations. Division ring
structure required for full reversibility. Co-presheaf generalization
decouples index category (dimensional compression) from value category
(ring structure).

**Validates:**
- `RoleKey::bind` (element-wise XOR on contiguous slices) is
  categorically optimal — not a design choice, a theorem consequence.
- XOR on GF(2)^d IS a division ring → full reversibility holds.
- Our slice-addressing scheme (ℐ = disjoint intervals [0..2000),
  [2000..4000), ...) is an instance of their index category with
  monoidal product = disjoint union.

**Key equations:**
- Kan extension: `(Ran_e v⊗̄w)_i = ∫_{jk} ℐ(i,e(j,k)) ⋔ (v_j·w_k)`
- Simplifies to element-wise: `v⊗w = ∫_i v_i · w_i`
- Role-filler: `w = (first ⊗ v_1) ⊕ (second ⊗ v_2)` with
  recovery `v_1 ∼ first ⊘ w` — our RoleKey::bind + unbind.
- Braiding ρ for sequences: `list(x_1,...,x_n) = x_1 ⊕ ρx_2 ⊕ ρρx_3 ⊕ ...`
  — this IS `vsa_permute` per position in the Markov bundler (D5).
- Non-commutative binding needed for hierarchical structure — validates
  why we use DIFFERENT role keys for S/P/O.

**Cross-ref:** `contract::grammar::role_keys::{RoleKey::bind, unbind, vsa_xor}`.

---

### Kleyko, Davies, Frady, Kanerva et al. (2106.05268) — VSA/HDC Survey Part II

**Finding:** VSA's algebraic structure enables "computing in
superposition" — efficient solutions to combinatorial search via
high-dimensional distributed representations. Computational
universality established.

**Validates:** Our XOR-superposition of N role bindings (tested at
5 simultaneous roles recovering at margin 1.0) IS computing in
superposition. The combinatorial search problem they describe =
our counterfactual hypothesis enumeration in `Resolution::from_ranked`.

**Cross-ref:** `contract::grammar::role_keys::vsa_xor`, `free_energy::Resolution`.

---

### Gallant & Okaywe (1501.07627) — MBAT: Objects, Relations, Sequences

**Finding:** Matrix binding (MBAT) satisfies machine-learning
constraints for VSA: similar structures → similar vectors. Phrases
should be binding-sums. Three-stage learning: representation →
association → inference.

**Validates:** Our three-stage pipeline mirrors theirs:
1. Representation = RoleKey::bind (content → role-indexed VSA)
2. Association = Markov ±5 bundling (context accumulation)
3. Inference = FreeEnergy resolution (hypothesis selection)

Their "phrases as binding-sums" = our SPO triple as
`SUBJECT_KEY.bind(s) ⊕ PREDICATE_KEY.bind(p) ⊕ OBJECT_KEY.bind(o)`.

**Cross-ref:** Plan D5 `MarkovBundler`, `Trajectory`.

---

## Tier 2 — Empirical validation of the grammar tier

### Graichen, de-Dios-Flores & Boleda (2601.19926) — "Grammar of Transformers" (337-article systematic review)

**Finding:** TLMs handle formal syntax well (agreement >85% BLiMP)
but show weak, variable performance on syntax-semantics interface
(<75% on binding, coreference, quantifier scope, island effects).
Severe English dominance (69%). Mechanistic methods underutilized.

**Validates:** Our tiered routing — DeepNSM handles the >85% formal
syntax locally; FreeEnergy + counterfactual resolves the <75%
syntax-semantics interface. Their call for "syntax-semantics interface
investigation + mechanistic methods" = exactly what our active-
inference stack provides.

**Cross-ref:** `contract::grammar::ticket::FailureTicket` (escalation
for the <75% tail), `free_energy::Resolution`.

---

### Jian & Manning (2603.17475 / EACL 2026) — Abstraction-First Language Learning

**Finding:** GPT-2 learns class-level verb behavior BEFORE item-
specific behavior. Sequential emergence: syntactic subcategorization
(t<100) → semantic argument structure (t>100) → non-local
dependencies (t>1000). Count-based exemplar baseline is strictly
worse.

**Validates:**
- `GrammarStyleConfig::nars.primary = Deduction` (class-level rules
  first) IS the abstraction-first policy.
- Sequential emergence maps to Markov radius scaling: ±1 captures
  subcategorization, ±3 captures argument structure, ±5 captures
  non-local. WeightingKernel::MexicanHat emphasizes local first.
- Their 4 verb classes (to-dative / motion / reciprocal / spray-load)
  = rows in our 144-verb taxonomy with characteristic TEKAMOLO priors.
- Exemplar-first baseline fails = Markov bundling without role-key
  structure is class-blind. Role keys ARE the abstraction mechanism.

**Cross-ref:** `contract::grammar::thinking_styles::NarsPriorityChain`,
`context_chain::WeightingKernel::MexicanHat`.

---

### Schulz, Mitropolsky & Poggio (2510.02524) — How LMs Learn CFGs

**Finding:** KL divergence over PCFG decomposes as sum over
subgrammar contributions (Theorem 4.3). Transformers learn all
subgrammar levels in PARALLEL. Models FAIL on deep recursion
despite handling long shallow contexts.

**Validates:**
- Our `FreeEnergy { likelihood, kl_divergence, total }` decomposition
  mirrors their KL-over-subgrammars. Each role-key slice IS a
  "subgrammar" in the VSA decomposition.
- Recursion failure = why we use Markov ±5 contextual coherence
  instead of recursive parsing. Deep recursion becomes "does this
  nested structure cohere with ±5 context?" — a flat comparison.
- Parallel subgrammar learning = our FSM handles all PoS categories
  simultaneously.

**Cross-ref:** `contract::grammar::free_energy::FreeEnergy`.

---

### Alpay & Senturk (2603.05540) — Grammar-Constrained LLM Decoding

**Finding:** Doob h-transform: `p(v|y<t) = p(v|y<t) · h(y<tv)/h(y<t)`.
Grammar survival probability modulates base LLM distribution.
Structural Ambiguity Cost (SAC): right-recursive O(1)/token,
concatenative Θ(t²)/token. Lower bound: Ω(t²) for parse-preserving
engines.

**Validates:**
- Their grammar-conditional is the dual of our free-energy: both
  are multiplicative modulations of a base distribution by structural
  constraint.
- SAC = our counterfactual branch count. Pearl 2³ mask reduces SAC
  by committing causal bits from morphology.
- Their Ω(t²) lower bound does NOT apply to us: we don't preserve
  the full parse forest. Active inference commits to argmin_F and
  discards (or marks epiphany). We trade parse-preservation for
  decision speed.

**Cross-ref:** `contract::grammar::free_energy::Resolution` (commit
discards losers), `EPIPHANY_MARGIN` (preserves runner-up only when
margin is tight).

---

## Tier 3 — Supporting evidence for specific design choices

### Starace et al. (2310.18696, EMNLP 2023) — Joint Encoding of Linguistic Categories

**Finding:** Related grammatical categories share overlapping
encodings in LLMs; pattern holds cross-lingually.

**Validates:** Role-key slice adjacency for morphologically-related
cases (Finnish Adessive and LOKAL_KEY map to overlapping TEKAMOLO
slots). Cross-lingual bundling works because categories are shared
at the representational level.

**Cross-ref:** `contract::grammar::role_keys::FINNISH_SLICES`,
`contract::grammar::role_keys::LOKAL_KEY`.

---

### Tjuatja, Liu, Levin & Neubig (2305.18185) — Agentivity Probe

**Finding:** Optionally transitive verbs test agent-vs-patient role
assignment. GPT-3 outperforms corpus statistics.

**Validates:** Pearl 2³ bit 0 = agency. Optionally transitive verbs
= exact Wechsel case ("The door opened" vs "John opened the door").
Their dataset = potential eval benchmark for `Resolution::resolve`.

**Cross-ref:** `contract::grammar::ticket::CausalAmbiguity::plausible_mask`,
`contract::grammar::free_energy::Hypothesis::causal_mask`.

---

### Petit, Corro & Yvon (2310.14124) — Supertagging + ILP

**Finding:** Supertagging (per-token category) + integer linear
program for structural consistency = compositional generalization.

**Validates:** Our PoS tagging (supertag) + `TekamoloPolicy::require_fillable`
(structural consistency). ILP = our Markov ±5 coherence (both prevent
locally-plausible but globally-inconsistent parses).

**Cross-ref:** `contract::grammar::tekamolo::TekamoloSlots`,
`thinking_styles::TekamoloPolicy`.

---

### Sultana & Ahmed (2602.20749) — Grammar–Semantic Feature Fusion

**Finding:** 11 explicit grammar features + frozen BERT = 2-15%
improvement. Grammar as explicit inductive bias, not learnable module.

**Validates:** Grammar-as-inductive-bias is the right framing. Their
11 features are a shallow version of our TEKAMOLO slot-filling +
SPO extraction. Full role-indexed VSA bundling should exceed their
2-15% improvement substantially.

**Cross-ref:** `contract::grammar::tekamolo`, `role_keys`.

---

### Shaikh, Ziems et al. (2306.02475, ACL 2023) — Cultural Codes

**Finding:** Sociocultural background characteristics significantly
improve pragmatic reference resolution.

**Validates:** `GrammarStyleAwareness` as per-style empirical prior.
Different thinking styles resolve the same ambiguity differently
because their priors over signal-profile frequency differ — exactly
the cultural-prior effect they measure.

**Cross-ref:** `contract::grammar::thinking_styles::GrammarStyleConfig`.

---

### Perez-Beltrachini et al. (2301.12217) — Conversational Semantic Parsing

**Finding:** Multi-turn QA grounded to SPARQL over large-vocab KGs.
Challenges: entity grounding, conversation context, generalization.

**Validates:** AriGraph triplet-graph + ContextChain = our equivalent.
Their "conversation context" = our ±5 Markov chain. We don't need
SPARQL because SPO triples are queried directly via
`TripletGraph::nodes_matching`.

**Cross-ref:** `arigraph::triplet_graph`, `grammar::context_chain`.

---

### Hussein (2602.14238) — CFG/GPSG Parser

**Finding:** CFG+GPSG parser producing dependency + constituency
trees; handles noise; UAS 54.5%.

**Validates:** Our baseline to beat. Their noise tolerance =
our `PartialParse` + `FailureTicket`. UAS 54.5% should be
significantly exceeded by adding Markov coherence + role-key binding.

**Cross-ref:** `contract::grammar::ticket::PartialParse`.

---

## The unclaimed intersection

**No paper in this landscape combines:**

1. Structural parsing (rule-based, not neural)
2. Active-inference ambiguity resolution (free-energy, not attention)
3. Role-indexed distributed representation (VSA with Kan-extension-
   justified element-wise ops)
4. NARS-revised epistemic awareness (per-parse revision, not gradient)

Shaw et al. provide the algebraic foundation (Tier 1). Graichen
et al. identify the target (syntax-semantics interface, Tier 2).
Jian & Manning validate the dispatch order (abstraction-first, Tier 2).
Alpay & Senturk formalize the grammar-conditional dual (Tier 2).

Our stack sits at the intersection. The closest prior art is
Shaw's category-theoretic VSA + Petit's supertagging+ILP, but
neither has the active-inference free-energy loop or the NARS-
revised epistemic awareness layer.

---

## Papers not yet fully retrieved

- **biorxiv 2022.02.22.481380v3** — PDF too large for WebFetch.
  Likely a neuroscience paper on VSA / neural binding.
- **ResearchGate VSA-for-CFGs (Mitropolsky?)** — 403 forbidden.
  This is likely the 2003.05171 paper already cited in the plan
  (VSA encoding of Chomsky-normal-form CFGs via Fock space).
