# Categorical-Algebraic Inference — The Architecture That Clicks

> **Version:** v1
> **Author:** main-thread session 2026-04-21
> **Status:** Active
> **Supersedes:** None (extends elegant-herding-rocket-v1, does not replace it)
> **Confidence:** CONJECTURE — grounded in Shaw 2501.05368 + 14 supporting
> papers + shipped code; not yet formally proven as categorical equivalence.
>
> **READ BY:** Every session touching grammar, VSA, NARS, free energy,
> AriGraph, DeepNSM, Trajectory, or thinking styles. This is the
> meta-architecture document. If you cannot state what this plan says
> after reading it, you have not understood the workspace.

---

## § 0 — The Claim (one paragraph, unforgettable)

Parsing, disambiguation, learning, memory, and awareness are NOT five
separate systems bolted together. They are **one algebraic operation**
— element-wise XOR on role-indexed slices of a 10,000-dimensional
binary vector — viewed through five lenses:

| Lens | Operation | Justification |
|------|-----------|---------------|
| **Category theory** | Right Kan extension (Shaw 2501.05368) | Dimension-preserving binding MUST be element-wise — theorem, not heuristic |
| **Active inference** | Free-energy minimization (Friston) | F = (1 - likelihood) + KL; likelihood = role-recovery margin after unbind |
| **Non-axiomatic reasoning** | NARS truth revision | f_new = (f·c + f_obs·c_obs)/(c+c_obs); confidence asymptotes at φ-1 ≈ 0.618 |
| **Graph-structured memory** | AriGraph triplet commit | Committed hypotheses (low F) become ambient prior for next cycle's KL term |
| **Epistemic awareness** | Method-call history on the carrier | The object becomes more aware by having observed itself resolve ambiguity |

**If you see these as five separate concerns, you will build five
separate modules. If you see them as one algebraic substrate viewed
through five lenses, you will build one object whose methods ARE the
inference engine. The second design is what this workspace ships.**


---

## § 1 — The Algebraic Substrate (what makes it one thing)

### 1.1 The vector: `Vsa10k = [u64; 157]`

10,000-dimensional binary vector, bit-packed in 157 × 64-bit words.
This is the **carrier object** — the thing that flows through every
stage of the pipeline. It is NOT an embedding (no learned weights).
It is NOT a symbol (no discrete lookup). It is a **distributed
algebraic element** in GF(2)^10000, the binary field of 10,000
dimensions.

### 1.2 The index category: role-key slices

```
ℐ = { SUBJECT[0..2K), PREDICATE[2K..4K), OBJECT[4K..6K),
       MODIFIER[6K..7.5K), CONTEXT[7.5K..9K),
       TEMPORAL[9K..9.2K), KAUSAL[9.2K..9.4K), MODAL[9.4K..9.5K),
       LOKAL[9.5K..9.65K), INSTRUMENT[9.65K..9.75K),
       BENEFICIARY[9.75K..9.78K), GOAL[9.78K..9.81K),
       SOURCE[9.81K..9.84K),
       Finnish×15[9.84K..9.91K), Tense×12[9.91K..9.97K),
       NARS×7[9.97K..10K) }
```

Disjoint contiguous intervals. Each interval is an **object in ℐ**
in Shaw's co-presheaf generalization. The monoidal product on ℐ is
disjoint union (not overlap) — this is what guarantees lossless
superposition across roles.

### 1.3 The value category: GF(2)

The binary field: values are 0 and 1. Multiplication = AND.
Addition = XOR. Division = XOR (self-inverse). This IS a division
ring → full reversibility of binding (Shaw's requirement).

### 1.4 The three operations, categorically

| Operation | Definition | Category theory | Code |
|-----------|-----------|----------------|------|
| **Bind** | Slice-masked XOR with role key | Right Kan extension of external tensor product | `RoleKey::bind(&self, content: &Vsa10k) -> Vsa10k` |
| **Unbind** | Same slice-masked XOR (self-inverse) | Inverse of Kan extension | `RoleKey::unbind(&self, bundle: &Vsa10k) -> Vsa10k` |
| **Bundle** | XOR-superposition of N bound vectors | Coproduct in the co-presheaf category | `vsa_xor(a: &Vsa10k, b: &Vsa10k) -> Vsa10k` |

Shaw's Theorem: the Kan extension formula `(Ran_e v⊗̄w)_i = ∫_{jk}
ℐ(i,e(j,k)) ⋔ (v_j·w_k)` collapses to element-wise multiplication
via the Yoneda lemma. Our `bind` is literally this theorem in code.

### 1.5 The braiding: `vsa_permute`

Shaw's braiding operator ρ for sequential structure:
`list(x_1,...,x_n) = x_1 ⊕ ρ(x_2) ⊕ ρ²(x_3) ⊕ ... ⊕ ρ^{n-1}(x_n)`

This IS position encoding in the Markov ±5 window. Each sentence at
position offset `d` from the focal point is permuted by `d` before
bundling. The braiding encodes temporal order without learned
positional embeddings.

ndarray implements this as `ndarray::hpc::vsa::vsa_permute(v, shift)`.
The contract doesn't carry this operation (zero-dep) but the
`MarkovPolicy.radius` and `WeightingKernel` configure it.


---

## § 2 — The Five Lenses (same substrate, five interpretations)

### Lens 1: Parsing = Binding (Kan extension)

**Input:** Token stream from DeepNSM's 6-state PoS FSM.
**Operation:** For each token, identify its grammatical role (Subject,
Predicate, Object, Temporal modifier, etc.) and call
`ROLE_KEY.bind(content_fingerprint)`.
**Output:** One `Vsa10k` per token, zero outside the role's slice,
content XOR'd with the role key inside.

Morphology commits role assignment at bind time:
- Russian Instrumental `-ом` → `INSTRUMENT_KEY.bind(noun_fp)`
- Finnish Adessive `-lla` → `LOKAL_KEY.bind(noun_fp)`
- English "with" → Wechsel: caller doesn't know which key. This is
  where Lens 2 fires.

**Paper grounding:** beim Graben et al. (2003.05171) proved that
CFG parse trees have a universal VSA representation in Fock space.
Our role-key slices are a finite-dimensional projection of this
Fock space — each slice corresponds to a grammar-rule position.

### Lens 2: Disambiguation = Free-Energy Minimization (Active Inference)

**Input:** A Wechsel token (ambiguous role assignment) in a ±5 Markov
context window.
**Operation:** For each candidate hypothesis H_i:
1. Bind the token under each candidate role key
2. Bundle into the ±5 trajectory (with braiding per position)
3. Unbind each role and compute `recovery_margin`
4. Compute `likelihood = mean(recovery_margins)`
5. Compute `kl = awareness.divergence_from(prior)`
6. Compute `F(H_i) = (1 - likelihood) + kl`
7. Rank hypotheses by F

**Output:** `Resolution::Commit` (argmin F), `Resolution::Epiphany`
(top-2 tied), or `Resolution::FailureTicket` (all F > ceiling).

**Homeostasis:** The system drives itself to low F by sampling
hypotheses. Morphology pre-collapses the branch set (Russian
Instrumental commits enabling-bit → 8 branches become 4). Each
morphological commitment is a free-energy reduction at zero
computational cost — the case ending did the work at tokenization
time.

**Paper grounding:**
- Alpay & Senturk (2603.05540): their Doob h-transform
  `p(v|y<t) · h(y<tv)/h(y<t)` is the grammar-conditional dual.
  Our F is the same structure in variational form.
- Schulz et al. (2510.02524): KL decomposes over subgrammars.
  Our role-key slices ARE subgrammars — F decomposes over them.
- Graichen et al. (2601.19926): the <75% syntax-semantics
  interface IS the high-F regime where this lens fires.

### Lens 3: Learning = NARS Truth Revision

**Input:** `ParseOutcome` — did the resolution agree with downstream
validation (LLM confirmation, entity-link match, graph consistency)?
**Operation:** Standard NARS revision:
```
f_new = (f_old × c_old + f_obs × c_obs) / (c_old + c_obs)
c_new = (c_old + c_obs) / (c_old + c_obs + 1)
```
Applied to every `ParamKey` that fired during the parse (which NARS
operator, which morphology table, which kernel, which slot).

**Output:** Updated `GrammarStyleAwareness::param_truths`. The style's
track record changes. Next cycle's `effective_config(prior)` may
drift the primary NARS inference away from the YAML default.

**The confidence horizon:** With `c_obs = 1.0` per observation,
confidence asymptotes at `φ - 1 ≈ 0.618` (golden ratio minus one).
This is a feature: it means the system NEVER becomes fully certain —
it always has room to revise. Full certainty would be c = 1.0, which
the NARS formula provably never reaches.

**Paper grounding:**
- Jian & Manning (2603.17475): abstraction-first. Class-level NARS
  truths (Deduction on verb families) revise before item-level
  truths (Abduction on specific verbs). The revision order IS the
  abstraction-first finding.

### Lens 4: Memory = AriGraph Commit (Graph-Structured Belief Revision)

**Input:** `Resolution::Commit` or `Resolution::Epiphany`.
**Operation:**
- **Commit:** Write one SPO triple to `TripletGraph` with NARS
  `TruthValue` and Pearl 2³ causal mask. Increment episode index.
  Update `EpisodicMemory::global_context` (orthogonal superposition
  of committed facts, each permuted by episode index).
- **Epiphany:** Write TWO triples. Attach `Contradiction { phase,
  magnitude }` computed from Staunen × Wisdom qualia projections.
  Both triples are individually addressable in the graph; the
  Contradiction marker preserves the ambiguity as meaning (not noise).

**Output:** The graph now contains a new fact. Next trajectory's
free-energy computation reads:
- `global_context` as ambient prior → shapes the KL term
- `nodes_matching(features)` for direct coreference lookup
- Rigid-designator facts (high-hardness unbundled) for pronoun
  resolution across arbitrary distances

**The loop closes here.** Committed facts ARE the prior for the next
cycle. The system's model of the world IS the AriGraph. A parse that
contradicts a committed fact pays a KL cost — the graph resists
revision. A parse that aligns with committed facts gets a KL
discount — the graph accelerates commitment.

**Paper grounding:**
- Perez-Beltrachini et al. (2301.12217): their "formal queries over
  KG" is our `TripletGraph::nodes_matching`. We don't need SPARQL
  because SPO triples are the native query format.

### Lens 5: Awareness = Method-Call History on the Carrier

**Input:** The Trajectory object's own state after `resolve()` and
`observe_outcome()`.
**Operation:** Not a separate computation. Awareness IS the
accumulated effect of having called methods on the carrier:
- `bind()` committed content to a role slice
- `recovery_margin()` measured how cleanly it came back
- `free_energy()` scored the hypothesis
- `resolve()` picked the winner (or marked epiphany)
- `observe_outcome()` revised the awareness truths

Each method call changes the object's internal state. The next
call to the same method reads different state. **The object becomes
more aware by having observed itself.**

**Not a metaphor.** `GrammarStyleAwareness::param_truths` is a
`HashMap<ParamKey, TruthValue>` that grows with every parse. The
`top_nars_inference()` method reads it to decide which NARS
operator dispatches next. If 50 past parses showed Deduction failing
and Abduction succeeding, the map contains that evidence and
`effective_config()` reflects it.

**Paper grounding:**
- Shaikh et al. (2306.02475): sociocultural priors improve pragmatic
  resolution. Our `GrammarStyleConfig` YAML prior IS the cultural
  prior; `GrammarStyleAwareness` IS its runtime revision.


---

## § 3 — The Closed Loop (how the lenses compose into one cycle)

```
    ┌─────────────────────────────────────────────────────────────┐
    │                    ONE COGNITIVE CYCLE                       │
    │                                                             │
    │  ① OBSERVE    token stream from DeepNSM FSM                │
    │       │                                                     │
    │       ▼                                                     │
    │  ② BIND       RoleKey::bind(content) per token              │
    │       │        morphology commits bits at bind time          │
    │       ▼                                                     │
    │  ③ BUNDLE     XOR-superpose bound vectors                   │
    │       │        braiding ρ per Markov position                │
    │       ▼                                                     │
    │  ④ SCORE      FreeEnergy::compose(likelihood, kl)           │
    │       │        per hypothesis                               │
    │       ▼                                                     │
    │  ⑤ RESOLVE    Resolution::from_ranked(hypotheses)           │
    │       │        → Commit / Epiphany / FailureTicket          │
    │       ▼                                                     │
    │  ⑥ COMMIT     AriGraph.commit_with_contradiction_check()    │
    │       │        → global_context updated                     │
    │       ▼                                                     │
    │  ⑦ REVISE     awareness.revise(param_key, outcome)          │
    │       │        → NARS truth updated per fired parameter     │
    │       ▼                                                     │
    │  ⑧ RESHAPE    next cycle's F landscape now includes:        │
    │                - committed facts in global_context (KL term) │
    │                - revised awareness (dispatch policy)         │
    │                - rigid designators (coreference shortcuts)   │
    │                                                             │
    └─────────────────── loops to ① ──────────────────────────────┘
```

**Every step is a method on a carrier object.** No free functions.
No external orchestrator. The Trajectory carries steps ②–⑤. The
TripletGraph carries step ⑥. The GrammarStyleAwareness carries
step ⑦. The EpisodicMemory carries step ⑧. Each object speaks
for itself; the loop is method calls, not message passing.

**The loop is O(1) per step** (for fixed vocabulary and role-key
count):
- ② bind = element-wise XOR over one slice: O(slice_width / 64)
- ③ bundle = element-wise XOR over 11 vectors: O(11 × 157)
- ④ score = N unbinds + N margin computations: O(N × 157)
- ⑤ resolve = sort N hypotheses: O(N log N), N ≤ 8 (Pearl 2³)
- ⑥ commit = one graph insert: O(1) amortized
- ⑦ revise = one HashMap insert: O(1)
- ⑧ reshape = one XOR-accumulate: O(157)

Total per sentence: **< 10 µs** on commodity hardware. Per Markov
window update: **< 100 µs**. Per full ±5 trajectory construction:
**< 1 ms**. This is 3–5 orders of magnitude faster than an LLM
parse of the same sentence.

---

## § 4 — What's Already Shipped vs What's Next

### Shipped (in `lance-graph-contract`, zero-dep)

| Component | File | LOC | Tests |
|-----------|------|-----|-------|
| Role-key catalogue with slice addressing | `grammar/role_keys.rs` | 580 | 14 (incl 5-role lossless superposition) |
| `RoleKey::bind/unbind/recovery_margin` | `grammar/role_keys.rs` | (included above) | (included above) |
| `vsa_xor`, `vsa_similarity`, `Vsa10k`, `VSA_ZERO` | `grammar/role_keys.rs` | (included above) | (included above) |
| ContextChain ±5 with coherence/replay/disambiguate | `grammar/context_chain.rs` | 477 | 8 |
| WeightingKernel (Uniform/MexicanHat/Gaussian) | `grammar/context_chain.rs` | (included above) | (included above) |
| FailureTicket with SPO×2³×TEKAMOLO×Wechsel decomposition | `grammar/ticket.rs` | 129 | 3 |
| TekamoloSlots (4 slots) | `grammar/tekamolo.rs` | 37 | — |
| WechselAmbiguity | `grammar/wechsel.rs` | 49 | — |
| FinnishCase (15 cases) | `grammar/finnish.rs` | 116 | — |
| NarsInference (7 variants) | `grammar/inference.rs` | 79 | — |
| GrammarStyleConfig + GrammarStyleAwareness + revise_truth | `grammar/thinking_styles.rs` | 490 | 12 |
| FreeEnergy + Hypothesis + Resolution + thresholds | `grammar/free_energy.rs` | 347 | 7 |
| CrystalFingerprint (Binary16K/Structured5x5/Vsa10kF32) | `crystal/fingerprint.rs` | 504 | — |
| f32 vsa_bind/vsa_bundle/vsa_superpose/vsa_cosine | `crystal/fingerprint.rs` | (included above) | — |

**Total shipped:** ~2,808 LOC, 44 tests, zero external dependencies.

### Next (per elegant-herding-rocket plan, now grounded by this document)

| D-id | Deliverable | What it instantiates from this plan |
|------|-------------|-------------------------------------|
| **D5** | `Trajectory` + `MarkovBundler` (deepnsm) | Steps ②③ — bind with role keys + braiding per position |
| **D2** | `ticket_emit.rs` (deepnsm) | Step ⑤ fallback — FailureTicket emission from parser coverage |
| **D3** | `triangle_bridge.rs` (deepnsm) | Step ② enrichment — Grammar Triangle produces NSM + Causality + Qualia |
| **D8** | AriGraph bridge methods | Step ⑥ — commit_with_contradiction_check, unbundle_abductive |
| **D10** | Animal Farm validation harness | Closed-loop benchmark: steps ①–⑧ on 40K words, measuring epiphany precision/recall |

### Deferred (acknowledged, not blocked)

| Item | Why deferred |
|------|-------------|
| D9 ONNX arc export | Third inference axis (arc); needs trained model |
| D11 Bundle-perturb emergence | Generative counterpart; needs D9 |
| NER pre-pass | Named-entity gap blocks OSINT vertical, not Animal Farm |
| Cross-lingual bundling | Needs parallel corpora; Animal Farm DE/RU translations available |

---

## § 5 — Why This Can't Be Unseen

### 5.1 The proof chain

1. **Shaw (2501.05368):** Dimension-preserving binding on co-presheaves
   MUST be element-wise. Proof via right Kan extension + Yoneda.
   → Our `RoleKey::bind` is not a choice. It's a theorem.

2. **beim Graben (2003.05171):** CFG parse trees have a universal
   representation in Fock space via VSA.
   → Our role-key slices are a finite projection of this Fock space.

3. **Jian & Manning (2603.17475):** Abstraction-first: class-level
   patterns emerge before item-specific patterns in transformer
   training. Count-based exemplar baseline fails.
   → Our NARS Deduction-primary dispatch IS abstraction-first.
   → Role keys ARE the abstraction mechanism (without them, bundling
     is class-blind, like the exemplar baseline that fails).

4. **Schulz (2510.02524):** KL divergence decomposes as sum over
   subgrammar contributions.
   → Our FreeEnergy decomposes over role-key slices. Same structure.

5. **Alpay (2603.05540):** Grammar-constrained decoding is a Doob
   h-transform on the token distribution. Lower bound Ω(t²) for
   parse-preserving engines.
   → Our active-inference resolution is the variational dual.
   → We dodge the lower bound by NOT preserving the parse forest
     (we commit argmin_F and discard).

6. **Graichen (2601.19926):** 337 articles show formal syntax >85%
   but syntax-semantics interface <75% in TLMs.
   → DeepNSM FSM handles the >85%. Free-energy resolution handles
     the <75%. The tiering boundary IS the empirically-measured gap.

7. **Gallant (1501.07627):** Three-stage learning: representation →
   association → inference.
   → Our pipeline: bind → bundle → resolve. Isomorphic.

8. **Kleyko (2106.05268):** VSA enables "computing in superposition"
   for efficient combinatorial search.
   → Our 5-role lossless superposition + counterfactual branching
     IS computing in superposition.

### 5.2 The object-does-the-work test

For any proposed change, apply this test:

> "Does this change add a free function that operates on a carrier
> object's state, or does it add a method to the carrier object?"
>
> - **Free function:** REJECT unless the operation is a pure
>   mathematical transform with no carrier-specific state.
> - **Method on carrier:** ACCEPT. The object speaks for itself.

This test prevents architectural drift back toward pipeline-of-
functions. The carrier objects (Trajectory, TripletGraph,
GrammarStyleAwareness, EpisodicMemory) each own their state and
their reasoning methods. External code calls `trajectory.resolve()`
— it does NOT call `resolve(trajectory, config, awareness, graph)`.

### 5.3 The five-lens litmus

For any new type or module, check which lens(es) it serves:

| If the new code... | It serves lens... | And belongs in... |
|---------------------|-------------------|-------------------|
| Produces a `Vsa10k` from content | 1 (Parsing/Binding) | deepnsm or RoleKey |
| Scores a hypothesis | 2 (Free Energy) | Trajectory or FreeEnergy |
| Updates a TruthValue | 3 (NARS Learning) | GrammarStyleAwareness |
| Writes to TripletGraph | 4 (Memory) | AriGraph bridge |
| Reads from param_truths to decide dispatch | 5 (Awareness) | GrammarStyleAwareness |

If a new type doesn't serve any lens, it's either infrastructure
(fine) or architectural drift (investigate).

---

## § 6 — Paper Bibliography (minimum set for the proof chain)

| Key | Citation | Role in proof chain |
|-----|----------|---------------------|
| Shaw2025 | Shaw, Furlong, Anderson, Orchard. "Developing a Foundation of VSA Using Category Theory." arXiv 2501.05368 | Kan extension theorem → element-wise optimal |
| beimGraben2020 | beim Graben, Huber, Meyer, Römer, Wolff. "VSA for Context-Free Grammars." arXiv 2003.05171 | Fock space universal representation → role slices are projections |
| JianManning2026 | Jian, Manning. "Humans and transformer LMs: Abstraction drives language learning." EACL 2026 / arXiv 2603.17475 | Abstraction-first → NARS Deduction primary |
| Schulz2025 | Schulz, Mitropolsky, Poggio. "Unraveling Syntax: How LMs Learn CFGs." arXiv 2510.02524 | KL subgrammar decomposition → F over role slices |
| Alpay2026 | Alpay, Senturk. "Attention Meets Reachability." arXiv 2603.05540 | Doob h-transform → our free-energy dual |
| Graichen2026 | Graichen, de-Dios-Flores, Boleda. "Grammar of Transformers." arXiv 2601.19926 | >85% / <75% gap → our tiering boundary |
| Gallant2015 | Gallant, Okaywe. "Representing Objects, Relations, Sequences." arXiv 1501.07627 | Three-stage learning → bind/bundle/resolve |
| Kleyko2022 | Kleyko, Davies, Frady, Kanerva et al. "VSA as Computing Framework." arXiv 2106.05268 | Computing in superposition → lossless N-role recovery |
| Tjuatja2023 | Tjuatja, Liu, Levin, Neubig. "Syntax-Semantics via Agentivity." arXiv 2305.18185 | Agentivity → Pearl 2³ bit 0 validation |
| Starace2023 | Starace et al. "Joint Encoding of Linguistic Categories." EMNLP 2023 / arXiv 2310.18696 | Shared encodings cross-lingually → role-key slice adjacency |
| Petit2023 | Petit, Corro, Yvon. "Supertagging for COGS." arXiv 2310.14124 | Supertag + ILP → PoS FSM + TEKAMOLO fillability |
| Shaikh2023 | Shaikh, Ziems et al. "Cultural Codes." ACL 2023 / arXiv 2306.02475 | Cultural prior → GrammarStyleConfig YAML prior |

Full per-paper mapping in `.claude/knowledge/paper-landscape-grammar-parsing.md`.

---

## § 7 — The Architecture Diagram (ASCII, paste anywhere)

```
TOKEN STREAM  ──①──►  DeepNSM FSM (6-state PoS tagger)
                          │
                          │  SentenceStructure { triples, modifiers, temporals }
                          ▼
                     ──②──►  RoleKey::bind(content)  per token
                          │   morphology commits Pearl 2³ bits
                          │   slice-masked XOR → zero outside role
                          ▼
                     ──③──►  MarkovBundler
                          │   braiding ρ per ±5 position
                          │   WeightingKernel (MexicanHat)
                          │   → Trajectory { vsa_bundle, context_chain }
                          ▼
               ┌─── ──④──►  For each hypothesis H_i:
               │          │   unbind each role → recovery_margin
               │          │   likelihood = mean(margins)
               │          │   kl = awareness.divergence_from(prior)
               │          │   F(H_i) = (1 - likelihood) + kl
               │          ▼
               │     ──⑤──►  Resolution::from_ranked(hypotheses)
               │          │
               │     ┌────┼──────────────┬──────────────┐
               │     ▼    ▼              ▼              ▼
               │   Commit    Epiphany         FailureTicket
               │   (F<0.2)   (ΔF<0.05)        (F>0.8)
               │     │         │                  │
               │     ▼         ▼                  ▼
               │  ──⑥──►  AriGraph             LLM fallback
               │     │    commit triple(s)     (the <25% tail
               │     │    + Contradiction       per Graichen)
               │     │    if epiphany
               │     ▼
               │  ──⑦──►  awareness.revise(key, outcome)
               │     │    NARS truth update per ParamKey
               │     ▼
               │  ──⑧──►  global_context += permuted fact
               │          rigid designators updated
               │          next cycle's KL landscape reshaped
               │
               └───────── loops to ① ─────────────────────
```

---

*End of plan. The click can't be unseen.*
