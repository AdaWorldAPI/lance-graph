# Learning Layer Curriculum — Neurosymbolic + RLVR + Causal (v1)

> **READ BY:** `truth-architect`, `integration-lead`, `nars-engineer`, `bus-compiler`,
> anyone proposing a `style_synthesize` / `intervene` / `counterfactual` capability,
> anyone considering RLVR or fine-tuning of stack-local models (Qwen3 / Jina v5 /
> Reranker v3 / ModernBERT / Reader-LM), anyone wiring Σ9-Σ10 EPIPHANY tier dispatch.
>
> **PAIRED WITH:** `causal-edge-64-spo-variant.md` (the SPO `CausalEdge64` —
> grouped-mechanism data substrate); `causal-edge-64-thinking-engine-variant.md`
> (the dispatch-payload `CausalEdge64` — emission point); `cognitive-shader-driver-thinking-engine-reunification.md`
> (the consumer of synthesized styles); the `THINKING_ORCHESTRATION_WIRING.md` gap list.
>
> **Status:** PROPOSAL (8-paper synthesis, no code yet; ratify §6 before fan-out)

---

## 1. Identity

This doc is the **curriculum for the stack's learning layer** — the missing self-
improvement loop that turns the existing `Think` struct (substrate live as of
PR #372) into a system that *trains itself* on its own failures.

The substrate is in place. Eight papers — read in the order below — explain how
to wire it. The doc maps each paper to the specific stack component it
operationalizes, and provides a four-PR implementation roadmap.

**Why a curriculum and not a plan.** The pieces span four research lines that
typically read in isolation: Schölkopf-style structural causal models, MIT-style
Bayesian program learning, code-LLM neurosymbolic reasoning, and RLVR/GRPO
training. The stack already has the *substrate* for all four (CausalEdge64,
StyleVectors, AriGraph SPO, Σ-tier router, MUL gate). What's missing is the
*joint reading* — knowing which paper supplies which verb. This doc is that
joint reading.

---

## 2. The one-paragraph synthesis

The eight papers collapse to one shape: **probabilistic programs over
structural causal models, learned from grouped/multi-environment data,
executed with explicit conditional structure, with calibrated uncertainty
over outcomes, trained via reinforcement learning with deterministic
verifiable rewards.** The stack supplies (in order): grouped causal data
via SPO-G quads, programs-as-code via JIT-compiled ThinkingStyles,
continuous program latents via `StyleVectors`, Bayesian belief over both via
NARS truth + MUL, explicit conditional dispatch via `MUL::GateDecision` +
`Σ-tier router`. The missing pieces are: explicit `intervene`/`counterfactual`
operators on `NarsInferenceType`, an ICM-invariance column in BindSpace, a
TextGrad-style closed-loop style optimizer, a GRPO trainer over AriGraph
trajectories, and an LINC-style classical prover for Σ9-Σ10 EPIPHANY
escalation.

---

## 3. The eight papers

### 3.1 Causal de Finetti (Guo, Tóth, Schölkopf, Huszár — 2022)

- **arXiv:** [2203.15756](https://arxiv.org/abs/2203.15756)
- **Central claim:** Independent Causal Mechanisms (ICM) is *statistically*
  formalized by a causal de Finetti theorem; causal structure is identifiable
  from grouped/exchangeable data even when i.i.d. data alone cannot
  identify it.
- **Stack mapping:** **AriGraph SPO-G quads** (PR #372 — landed). The G slot
  in the quad is the grouping/environment label. Different G values expose
  different mechanisms holding stable across environments — exactly the
  multi-environment data Causal de Finetti requires.
- **What to extract for the stack:** the **G-grouping doctrine** — when the
  ingestion path tags a triple with its G (mechanism/regime), causal
  inference downstream gets a free identifiability boost. Operationally: any
  ingestion bridge that drops `G` is throwing away identifiability signal.

### 3.2 Latent Program Networks (Bonnet, Macfarlane — 2024)

- **arXiv:** [2411.08706](https://arxiv.org/abs/2411.08706)
- **Central claim:** Program induction works better in a **continuous latent
  space** than in discrete symbolic search; **test-time gradient adaptation**
  on the latent generalizes to OOD tasks. State-of-the-art on ARC-AGI without
  retraining the network.
- **Stack mapping:** **`StyleVectors` (4096-head per-style projection)** in
  `lance-graph-planner::cache::triple_model`. The stack already has a
  continuous latent program space — LPN's contribution is *what to do with
  it at inference time*.
- **What to extract:** **gradient-on-`StyleVector` at inference time**.
  Differentiate trajectory loss w.r.t. `StyleVector` ∈ ℝ⁴⁰⁹⁶ when the MUL
  gate detects elevated free-energy. Cheaper than retraining; closes the
  `THINKING_ORCHESTRATION_WIRING.md` Gap 1 (12 vs 36 ThinkingStyle) by
  *learning the missing 24* at runtime instead of authoring them.

### 3.3 LINC (Olausson, Gu, Lipkin, Zhang, Solar-Lezama, Tenenbaum, Levy — 2023)

- **arXiv:** [2310.15164](https://arxiv.org/abs/2310.15164)
- **Central claim:** LLM-as-semantic-parser (NL → first-order logic) + external
  theorem prover **beats Chain-of-Thought** across model scales. **A 15.5B
  open-source model with LINC outperforms GPT-4 + CoT by 10% absolute** on
  ProofWriter.
- **Stack mapping:** **`Σ9-Σ10` EPIPHANY tier → L4 planner** dispatch path in
  the `SigmaTierRouter` (PR #371/#372). LINC is what L4 actually *does* when
  invoked: parse the AriGraph context into FOL, hand off to a deterministic
  prover (Z3 recommended over Prover9 — SMT theories match stack queries
  better; Rust bindings exist).
- **What to extract:** the **complementary-failure-mode result (§5)**. LINC
  and CoT fail on different problems. The MUL gate is already the
  "ensemble dispatcher" — when NARS confidence is high, commit; when
  free-energy is high, escalate to LINC. The stack's dispatch substrate is
  already this shape; LINC fills the "what does Σ9-Σ10 compute?" slot.

### 3.4 Adaptive Problem Generation via Symbolic Representations / Opt-Sym (Yeo, Jeon, Weerakoon, Qiao, Prakash, Solar-Lezama, Misra — 2026)

- **arXiv:** TBD (paper provided by user; ID pending in HF index)
- **Central claim:** Closed-loop training-data generation in **symbolic space**
  (SymPy / SMT-LIB) with **TextGrad-style prompt optimization** against
  student-model performance. 3-8% gains across GSM8K/MATH-500/GSM-Symbolic
  on Qwen2.5-1.5B/3B; **5× more diverse problems** by cosine distance.
- **Stack mapping:** the **data-generation half** of the BPL story (LPN is
  induction; Opt-Sym is generation). The closed loop maps to:
  AriGraph trajectories (= symbolic seed) → symbolic modification (= NARS
  rotation / SPO constraint injection) → exact solve (= NARS revise
  deterministic) → render (= `thinking-engine` lens stack) → verify (= MUL
  gate + I-SUBSTRATE-MARKOV check) → student rollout (= local Qwen3 RLVR) →
  gradient signal (= MUL `free_energy` band).
- **What to extract:** the **Goldilocks calibration mechanism**. Generate
  problems targeting 40-60% student pass rate (max learning signal). The
  Σ-tier router *already* discretizes difficulty (Σ1-Σ5 STATIC → easy;
  Σ6-Σ8 EMERGENT → Goldilocks; Σ9-Σ10 EPIPHANY → too hard, expensive). Use
  Σ-tier as the difficulty axis.

### 3.5 Executable Counterfactuals (Vashishtha, Dai, Mei, Sharma, Tan, Peng — 2025)

- **arXiv:** [2510.01539](https://arxiv.org/abs/2510.01539)
- **Central claim:** Counterfactual reasoning has three steps — **abduction
  → intervention → prediction** (Pearl's full counterfactual chain). LLMs
  drop 25-40% from interventional to counterfactual reasoning. **RL induces
  the core cognitive behaviors; SFT does not generalize OOD.**
- **Stack mapping:** the **Pearl 2³ rung** in `nars_engine.rs`. The stack
  acknowledges all three rungs by name; this paper says how to *train* on
  them. The three steps map to existing stack capabilities:
  - abduction = `NarsInferenceType::Abduction` (live)
  - intervention = MISSING from `NarsInferenceType` — proposed in §6.1
  - prediction = `NarsInferenceType::Deduction` forward (live)
- **What to extract:** the **RL > SFT for OOD generalization** result. When
  PR-3 below trains a local model on stack data, use **GRPO** (next paper)
  rather than supervised fine-tuning. This is a load-bearing methodological
  commitment, not a detail.

### 3.6 Conformal Counterfactual Generation for LLM Control (Farzaneh, D'Oro, Simeone — 2026)

- **arXiv:** [2601.20090](https://arxiv.org/abs/2601.20090)
- **Central claim:** Wrap an LLM agent in an **SCM model of the
  (user, LLM, environment) triple**, use probabilistic abduction + test-time
  scaling, yields **conformal counterfactual sets that contain the true
  outcome with bounded probability** (calibrated uncertainty quantification).
- **Stack mapping:** the **MedCare-rs / q2 cockpit safety layer**. When an
  LLM proposes a treatment plan (or a cockpit action), the patient's
  CausalEdge64 history + AriGraph SPO-G **is** the SCM. Generate N
  counterfactual replays via §6.4 below; conformal calibration bounds the
  true-outcome probability.
- **What to extract:** the **calibration recipe**. This is what makes the
  L4 planner output *trustable*: not "the LLM said X" but "with probability
  ≥ 95%, the true outcome is in this conformal set of K candidates".
  Critical for any consumer where the cost of a wrong commit is high
  (medical, financial, OSINT).

### 3.7 TextGrad (Yuksekgonul et al. — 2024)

- **arXiv:** ~[2406.07496](https://arxiv.org/abs/2406.07496) (the field's
  reference impl — verify exact ID at adoption)
- **Central claim:** Treat **text as differentiable** via LLM-as-optimizer.
  A natural-language "gradient" describes how to improve a prompt; an LLM
  optimizer applies it. Enables prompt optimization without numerical
  gradients on the underlying model.
- **Stack mapping:** the **closed-loop refinement engine** for
  `style_synthesize` (§6.3). The `StyleVector` is a real numerical
  vector — autograd works directly. TextGrad's contribution is the
  *auxiliary signal*: when the MUL gate fires "student fails on >3-hop
  predicates", TextGrad translates that into a textual gradient that
  updates the style's prompt prior.
- **What to extract:** the **hybrid gradient pattern**. Numerical gradients
  on `StyleVector` directly; textual gradients on the rendering prompt
  via TextGrad. Two signals, one optimizer loop. Stronger than either
  alone.

### 3.8 GRPO / DeepSeekMath (Shao et al. — 2024)

- **arXiv:** [2402.03300](https://arxiv.org/abs/2402.03300)
- **Central claim:** **Group Relative Policy Optimization** for RLVR. Instead
  of a learned reward model, use **group-relative advantages** over multiple
  rollouts of the same problem. Eliminates the value-function critic, reduces
  variance, works with deterministic verifiers.
- **Stack mapping:** the **trainer algorithm** for the local model. NARS is
  already the deterministic verifier (does the trajectory commit? what's the
  truth confidence?). GRPO consumes that verifier directly — no need for a
  learned reward model.
- **What to extract:** the **algorithm spec**. n rollouts per problem,
  group-relative advantage normalization, KL penalty against reference
  policy. ~800 LOC of Rust + `candle` / `burn` for the policy gradient
  update. PR-3 below.

---

## 4. The curriculum — reading order

Read in this order. Each paper's prerequisites are the prior entries.

### Tier 0 — Doctrinal Frame (read first, ≤ 1 hour total)

1. **Causal de Finetti** (3.1) — establishes ICM as statistical principle, justifies SPO-G quads.
2. **Executable Counterfactuals** (3.5) — establishes Pearl 2³ rungs as
   trainable verbs, surfaces the RL > SFT result.

### Tier 1 — Method Substrate (read second, ≤ 2 hours)

3. **LINC** (3.3) — establishes neurosymbolic dispatch (Σ9-Σ10 escalation).
4. **GRPO / DeepSeekMath** (3.8) — establishes the RLVR algorithm; supplies the
   train-loop primitive.

### Tier 2 — Closed-Loop Generation (read third, ≤ 2 hours)

5. **Latent Program Networks** (3.2) — supplies `StyleVector` test-time
   adaptation.
6. **TextGrad** (3.7) — supplies the closed-loop prompt optimizer for
   `style_synthesize`.
7. **Opt-Sym / Adaptive Problem Generation** (3.4) — synthesizes 5+6 into
   the data-generation pipeline.

### Tier 3 — Safety / Calibration (read fourth, ≤ 1 hour)

8. **Conformal Counterfactual Generation** (3.6) — calibrated bounds for
   L4 planner outputs; required before MedCare-rs / q2 high-stakes consumers.

**Total reading load:** ~6 hours for the full curriculum. Sprint workers
should be assigned ONE tier each, with tier 0 mandatory for all.

---

## 5. Stack-substrate alignment table

| Paper | Stack component | State |
|---|---|---|
| 3.1 Causal de Finetti | AriGraph SPO-G quads | Live (PR #372) |
| 3.2 LPN | `StyleVectors` (`lance-graph-planner::cache::triple_model`) | Live, underused (no test-time gradient) |
| 3.3 LINC | `Σ9-Σ10 → L4 planner` dispatch path | Live, prover not wired (gap §6.5) |
| 3.4 Opt-Sym | (nothing yet) | MISSING — proposed §6.2 |
| 3.5 Executable CFG | `Pearl 2³` in `nars_engine` | Live in name, missing intervene/counterfactual verbs |
| 3.6 Conformal CFG | (nothing yet) | MISSING — proposed §6.5 |
| 3.7 TextGrad | (nothing yet) | MISSING — proposed §6.3 |
| 3.8 GRPO | (nothing yet) | MISSING — proposed §6.4 |

Five live components; four missing modules. The missing modules compose
into the four-PR roadmap below.

---

## 6. Implementation roadmap — 5 PRs after PR #372's substrate

Each PR is independently mergeable, sized for one CCA2A 12-worker fleet, and
verifiable via deterministic NARS commits (no LLM verifier needed).

### 6.1 PR-LL-1: NARS intervene/counterfactual verbs (~200 LOC)

Add two variants to `NarsInferenceType` (currently 5: Deduction, Induction,
Abduction, Revision, Choice):

```rust
pub enum NarsInferenceType {
    Deduction, Induction, Abduction, Revision, Choice,
    Intervention,    // NEW — Pearl rung 2 (do-calculus)
    Counterfactual,  // NEW — Pearl rung 3 (3-step: abduce, intervene, predict)
}
```

Thread through `Style → NarsInferenceType` selector + add
`AriGraph::intervene_on(subject, predicate, value)` that produces a
counterfactual SPO-G with G tagged `G::Intervention`.

**Closes:** Pearl 2³ named-but-not-dispatched gap. Required precondition
for PR-LL-4.

### 6.2 PR-LL-2: ICM-invariance column + Opt-Sym data generator (~800 LOC)

Two coupled additions:

(a) **`IcmInvarianceColumn`** in BindSpace (1 bit per row) marking which
mechanisms are *believed stable under intervention*. Becomes the
regularizer at NARS revision time — only revise non-ICM-constrained facts.

(b) **`lance-graph-planner::data_gen`** — symbolic AriGraph problem
generator:

```rust
pub struct SymbolicProblemModifier {
    seed_corpus: AriGraphTrajectoryCorpus,
    student_caps: MulSnapshot,         // free-energy / plasticity / DK
    diversity_threshold: f32,          // target cosine distance (Opt-Sym: 5x baseline)
    sigma_target: SigmaTier,           // Σ6-Σ8 = Goldilocks band
}

impl SymbolicProblemModifier {
    pub fn generate_batch(&self, n: usize) -> Vec<SyntheticTrajectory> {
        // 1. Sample seed from corpus
        // 2. Apply symbolic mods:
        //    - SPO constraint injection
        //    - NARS predicate rotation via Abduction
        //    - Truth-band gradient adjustment
        //    - Σ-tier jump
        // 3. Solve via NARS forward chain (deterministic)
        // 4. Render via thinking-engine lens stack
        // 5. Verify I-SUBSTRATE-MARKOV (Chapman-Kolmogorov)
        // 6. Reject if diversity < threshold OR Σ-tier ≠ sigma_target
    }
}
```

**Closes:** Causal de Finetti ICM-invariance gap; Opt-Sym generator gap.

### 6.3 PR-LL-3: TextGrad-style style optimizer (~400 LOC)

Hybrid gradient pattern over `StyleVector`:

```rust
pub fn optimize_style_via_textgrad(
    initial: StyleVector,
    target_band: TruthBand,
    iterations: u32,
) -> StyleVector {
    let mut style = initial;
    for _ in 0..iterations {
        let batch = generate_batch_with(&style);
        let mul = student_caps_on(&batch);
        // Numerical gradient: free_energy vs StyleVector (LPN-style test-time)
        let numerical_grad = autograd_free_energy(&style, &batch);
        // Textual gradient: natural-language feedback via LLM optimizer
        let text_grad = compose_textual_gradient(&mul, &target_band);
        // Hybrid update: numerical step + LLM-mediated textual nudge
        style = update_style(style, numerical_grad, text_grad);
    }
    style
}
```

**Closes:** LPN test-time adaptation gap; TextGrad gap;
`THINKING_ORCHESTRATION_WIRING.md` Gap 1.

### 6.4 PR-LL-4: GRPO trainer for local models (~800 LOC, new crate)

`crates/lance-graph-trainer/`:

```rust
pub struct GrpoTrainer<S: StudentModel> {
    student: S,                        // Qwen3 fine-tunable head (via candle or burn)
    generator: SymbolicProblemModifier,// PR-LL-2
    verifier: NarsEngine,              // existing, deterministic
}

impl<S: StudentModel> GrpoTrainer<S> {
    pub fn step(&mut self) -> TrainStats {
        let problems = self.generator.generate_batch(128);
        let rollouts = self.student.roll_out(&problems, /* n_per */ 4);
        let rewards: Vec<f32> = rollouts.iter()
            .map(|r| self.verifier.verifies(r))  // graded truth, not just 0/1
            .collect();
        let advantages = grpo_group_relative(rewards);
        self.student.policy_gradient_update(rollouts, advantages)
    }
}
```

**Closes:** GRPO trainer gap. Critical: use **NARS truth confidence**
(graded ∈ [0,1]) as the reward, not binary commit/reject. Stack's reward
signal is strictly stronger than Opt-Sym's binary pass/fail.

### 6.5 PR-LL-5: LINC bridge + conformal counterfactual sets (~600 LOC)

`crates/linc-bridge/`:

```rust
pub enum ReasoningMode {
    Nars,                          // Σ1-Σ8 default
    Linc { prover: Z3Prover },     // Σ9-Σ10 deductive
    Conformal { n: usize, alpha: f32 }, // wrap LINC in conformal calibration
}

pub fn dispatch(query: AriGraphQuery, mode: ReasoningMode) -> Resolution {
    match mode {
        ReasoningMode::Linc { prover } => {
            let fol = arigraph_to_fol(&query.context);
            let goal = parse_conclusion(&query.goal);
            match prover.entails(fol, goal) {
                Entails::True => Resolution::Commit(Truth::CERTAIN),
                Entails::False => Resolution::Reject,
                Entails::Unknown => Resolution::Escalate,  // fallback to NARS
            }
        }
        ReasoningMode::Conformal { n, alpha } => {
            // Farzaneh 2026: N counterfactual replays + conformal calibration
            let candidates = (0..n).map(|_| counterfactual_replay(&query)).collect();
            conformal_set(candidates, alpha)
        }
        _ => narsengine.resolve(query),
    }
}
```

**Prover choice:** Z3 via `z3-rs` bindings. SMT theories (arithmetic,
bitvectors, arrays) match stack queries better than pure FOL.

**Closes:** LINC bridge gap; Conformal CFG safety layer for MedCare-rs / q2.

---

## 7. Open questions (ratify before sprint fan-out)

- **OQ-LL-1:** Reward shape for GRPO — graded NARS confidence ∈ [0,1] vs
  binary commit/reject. *Curriculum recommendation: graded* (strictly more
  information).
- **OQ-LL-2:** TextGrad LLM optimizer — local Qwen3 vs frontier API. Local
  keeps the no-API-dependency story but costs ~10× more compute.
  *Curriculum recommendation: local Qwen3 with frontier fallback when
  free-energy stays high after 8 textual iterations*.
- **OQ-LL-3:** Prover choice (Z3 vs Prover9 vs Vampire vs HOL Light). Z3 has
  Rust bindings + SMT support; Prover9 is LINC's reference; HOL Light is
  for proof-bearing code (the s2n-bignum-bench direction). *Curriculum
  recommendation: Z3 for Σ9-Σ10 default; HOL Light reserved for the
  verified-code consumer path*.
- **OQ-LL-4:** Should `style_synthesize` write back to the contract's 36
  fixed styles, or maintain a separate `learned_styles` pool? Writing
  back closes the duplication gap; separate pool preserves the contract's
  zero-dep invariant. *Curriculum recommendation: separate pool, contract
  exposes a `StylePoolProvider` trait.*
- **OQ-LL-5:** ICM-invariance column update protocol — when does a
  mechanism *lose* its invariance bit? Counterfactual replay that
  observably changes behavior on a known-invariant mechanism is a signal
  the invariance bit should clear. *Curriculum recommendation: clear the
  bit on counterfactual-truth contradiction; reset on N consecutive
  invariance-confirming observations*.
- **OQ-LL-6:** Σ-tier as difficulty axis for Opt-Sym calibration — does
  the existing `SigmaTierRouter` produce Σ-tier classifications fast
  enough to be a hot-path filter? *Probe required before PR-LL-2 fan-out*.

---

## 8. Iron rule compliance

| Rule | Compliance check |
|---|---|
| I-SUBSTRATE-MARKOV | All synthesized trajectories MUST pass Chapman-Kolmogorov test in PR-LL-2 verify step. Reject if associativity breaks. |
| I-NOISE-FLOOR-JIRAK | Conformal CFG (PR-LL-5) calibration MUST use Jirak-derived bounds, not classical Berry-Esseen. The CounterFactual rollouts are correlated via shared latent abduction; classical bounds underestimate variance. |
| I-VSA-IDENTITIES | `style_synthesize` (PR-LL-3) produces a `StyleVector` (identity fingerprint), not new content. Content stays in the existing YAML registries and contract enum. |
| I1 (BindSpace read-only) | `IcmInvarianceColumn` (PR-LL-2) is a NEW BindSpace column; writes go through `CollapseGate::bundle`, never raw assignment. |
| Method-on-carrier | All four new capabilities (intervene, synthesize, train, prove) are methods on existing carriers (AriGraph, StyleVector, Student, Query), not free functions. |
| AGI-as-glove (SoA) | Synthesized styles land in `StyleColumn` (extension of existing SoA); no new layer; no AGI service. |

All six iron rules are satisfied by the proposed PRs. The curriculum is
doctrinally consistent.

---

## 9. Cross-references

- **`causal-edge-64-spo-variant.md`** — the AriGraph SPO `CausalEdge64`
  bit layout, post-PR #372 with G:5 generators / W:6 witnesses / truth:2 bands.
  Direct substrate for Causal de Finetti §3.1.
- **`causal-edge-64-thinking-engine-variant.md`** — the 8-channel cascade
  variant. Emission point for synthesized styles from PR-LL-3.
- **`causal-edge-64-synergies-and-pr-trajectory.md`** — the cross-comparison
  of the two CausalEdge64 variants and their joint trajectory.
- **`cognitive-shader-driver-thinking-engine-reunification.md`** — the
  consumer surface for synthesized styles; reunification means
  `style_synthesize` writes one place, reads everywhere.
- **`encoding-ecosystem.md`** — MANDATORY pre-read for any codec touch
  during PR-LL-2 rendering.
- **`bf16-hhtl-terrain.md`** — probe queue; OQ-LL-6 belongs here as a
  CONJECTURE pending probe.
- **`lab-vs-canonical-surface.md`** — MANDATORY pre-read before wiring
  the GRPO trainer's REST/gRPC interfaces. Canonical surface is
  `UnifiedStep`; LAB scaffolding is the shader-lab binary.
- **`THINKING_ORCHESTRATION_WIRING.md` Gap 1** — closed by PR-LL-3.
- **`THINKING_ORCHESTRATION_WIRING.md` Gap 4** (Elevation not connected)
  — closed by the existing `SigmaTierRouter` consuming PR-LL-3's free-energy
  gradient as the elevation signal.

---

## 10. Sprint readiness checklist

Before spawning the CCA2A sprint fleet for any of PR-LL-1 through PR-LL-5:

- [ ] All 6 OQs in §7 ratified or explicitly deferred with rationale
- [ ] Worker capacity check: each PR is one 12-worker fleet sized; spawn
      sequentially (LL-1 → LL-2 → LL-3 → LL-4 → LL-5) since each is a
      precondition for the next
- [ ] Verifier corpus prepared for PR-LL-2: at minimum 500 AriGraph
      trajectories with known-correct NARS commits as the seed
- [ ] Local Qwen3 fine-tunable head wired via `candle` or `burn` (PR-LL-4
      precondition; ~2 weeks of separate prep work)
- [ ] Z3 Rust binding (`z3-rs`) vendored or path-deped (PR-LL-5 precondition)
- [ ] Σ-tier router probe (OQ-LL-6) run before PR-LL-2

---

## 11. What this curriculum does NOT cover

- **Pre-training of foundation models** — outside scope. Stack uses Jina v5,
  Qwen3, ModernBERT, etc. as frozen encoders. PR-LL-4 fine-tunes a *head*,
  not the foundation.
- **Multi-agent debate / consensus** — outside scope. The CCA2A pattern
  covers session-time agent coordination; runtime multi-agent reasoning
  lives in the existing `a2a_blackboard` substrate.
- **Curriculum learning across sessions** — single-session focus. Cross-
  session learning curriculum is a follow-up doc once PR-LL-4 produces
  trained heads worth versioning.
- **Adversarial / robustness training** — the curriculum is about
  *capability* learning, not robustness. Adversarial extension is a
  separate research line (PromptBench, etc.) that compose with this work
  but aren't required for the substrate.

---

## 12. Versioning

- **v1 (this doc):** 8-paper synthesis, 5-PR roadmap, ratification pending
- **v2 (after PR-LL-1):** update §6.1 with shipped interface; clear OQ-LL-1
- **v3 (after PR-LL-2):** update §6.2 with shipped generator; clear OQ-LL-5/6
- **v4 (after all 5 PRs):** consolidate as `learning-layer-shipped-v1.md`,
  this doc archived to `.claude/handovers/`

Per CLAUDE.md APPEND-ONLY governance: status updates land here in §12;
the rest of the doc is immutable below the version line.
