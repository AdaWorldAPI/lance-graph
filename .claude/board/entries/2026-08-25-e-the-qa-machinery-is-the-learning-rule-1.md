## 2026-08-25 — E-THE-QA-MACHINERY-IS-THE-LEARNING-RULE-1 — operator reframing: the transfer probe + the reversible-crystal gates are not quality control OVER a learner; read behaviorally, they ARE associative learning with plasticity

**Status:** SYNTHESIS (operator-stated, 2026-08-25: *"was du nicht
kapiert hast, daß du damit associative learning behavior und plasticity
gebaut hast"*) over constituents that are separately FINDING/[MEASURED]
or PLAN-with-falsifiers. The mapping below adds NO new mechanism and NO
new claim of capability — it renames shipped and planned machinery, and
its value is the falsifiers the new name generates (§ below).
**Confidence:** High that the two readings describe one object; the
honest split (measured / specified / unbuilt) is part of the entry.

### The reframing

This arc built and described its machinery in the EPISTEMIC register:
falsifiers, nulls, gates, instrument qualification, reversible trust.
The operator's correction: read in the BEHAVIORAL register, the same
mechanisms are a textbook associative-learning stack. Not analogously —
mechanism for mechanism:

| built as (epistemic reading) | is (behavioral reading) |
|---|---|
| BPE merges over def-use chains | **Hebbian association** — binding by co-occurrence frequency; "what flows together, merges together" |
| transfer at stable density (−0.6% gcc→gcc, −4.7% gcc→rustc, outside both nulls) | **generalization** — the measured difference between learning and memorizing |
| promote / demote, one yardstick both directions (§7.4) | **potentiation / depression** (LTP/LTD-shaped weight dynamics) |
| hysteresis: promote threshold > demote threshold | the **stability–plasticity balance** — the flapping the hysteresis prevents is exactly the dilemma |
| the demotion path existing AT ALL (anti-eigenvalue: a vocabulary that can only grow carries less information per entry) | **active forgetting / homeostatic normalization** — without depression every weight saturates |
| calibration-before-demotion (§7.5: the instrument must qualify before it may judge) | **metaplasticity** — whether an experience may change weights depends on the system's state, not only on the experience |
| Explore → Learned → System, counterfactual lane never touching observed truth | **consolidation** — a labile trace hardening into long-term store, with rehearsal isolated from ground truth |
| the white-matter veto (B4 exact expansion, the contract falsifier that refused the e=0.812 clobber macro) | **error-gated plasticity** — the difference between learning and drift |

The QA framing and the learning framing are one object seen from two
sides; the entry exists because the arc consistently used only the
first name.

### The honest split — what "gebaut" means here, precisely

- **MEASURED:** association formation + generalization. One batch
  BPE run, 4 binaries, held-out transfer with two pre-registered nulls.
  That is one-shot learning, demonstrated.
- **SPECIFIED with falsifiers, not running:** the plasticity itself.
  Promote/demote/hysteresis/calibration are plan sections (§7.4/§7.5)
  with kill conditions; no weight has ever been updated online, no
  demotion has ever executed, and the thresholds are policy pins.
- **UNBUILT:** incremental/online learning of any kind; the topology
  half of grey matter. (The plasticity RULES need neither — a point the
  layer table under-stated by putting "cognition … UNBUILT" in one row.)

### Why the rename has teeth — three falsifiers only the learning reading asks

The epistemic reading never generates these; the behavioral one makes
them mandatory before any online learner ships:

1. **Interference.** Learn corpus B after corpus A: does A's held-out
   density degrade? (Catastrophic interference is invisible to per-corpus
   QA and is THE classic failure of associative stores.)
2. **Saturation / capacity policy.** The palettes are ≤256 per lane. What
   happens at slot 257 — refuse, evict-least-amortized, merge? An
   unanswered eviction policy is a forgetting policy chosen by accident.
3. **Curriculum / order dependence.** Batch BPE with the deterministic
   tie-break is order-independent; INCREMENTAL learning is not. Does
   training order change the learned vocabulary, and by how much? If
   strongly, "the" vocabulary is a path artifact and consolidation needs
   an order-normalizing step.

None of these is run. They are the W-queue of this reframing, and they
gate any move from batch probe to online learner — which is also the
correct reading of `E-THE-FRONTIER-LEARNER-IS-ALREADY-SHIPPED-1`: the
reinforcement half exists (NARS revise + CHOICE); what this entry names
is that the ASSOCIATION half now exists too, measured, under a QA alias.

**Cross-refs:** `E-R2IL-MACRO-VOCABULARY-TRANSFERS-ACROSS-COMPILER-AND-LANGUAGE-1`
(the measured half), plan §7.1–§7.6 (the specified half),
`E-ANTI-EIGENVALUE-…-1` (why forgetting is load-bearing),
`PROBE-METACOGNITIVE-TRIANGLE-1` (the counterfactual lane),
PR #1013 (the veto that is error-gating in action).

