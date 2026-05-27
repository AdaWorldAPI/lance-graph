# Atom Basis Inventory — D-ATOM-0 (the 32 harvested cognitive atoms)

> **READ BY:** D-ATOM-1 (`contract::atoms`), D-ATOM-2 (`recipe.rs`), D-ATOM-3 (`quorum.rs`); `truth-architect`.
> **Status:** D-ATOM-0 resolution. Source = **harvested ladybug-rs / PersonaHub**, `callcenter-membrane-v1.md` §16 ("Persona as Function: 32 Atoms × 16 Weightings"). NOT qualia, NOT the 36 styles.
> **Date:** 2026-05-27.

## The layering (from §16 — one identity, three representations)

- **Atoms** = the **32 named cognitive atoms** (semantic *operations*). These are the i4-32D dims / 64 poles. **Dichotomy-bounded is an atom property — checked here.**
- **Style** = an **i4-32D vector** — a weighting of the 32 atoms (16 levels each = `16^32 = 2^128`). The `ThinkingStyle` 6-bit id (36→64 slots) *resolves to* such a vector; it is not itself an atom.
- **Persona** = `PersonaSignature { atom_bitset: u32, palette_weight, template_id }` + YAML runbook.
- **Business / FIBU** = sidecar in the **style/persona palette** (Layer 1/3), NOT an atom dim.

## The 32 atoms + dichotomy classification

The user's rule: *atoms whose opposite is already among the 32 form a signed bipolar dim; atoms whose opposite is NOT listed get the true opposite evaluated for value — if valuable it materializes the − pole, else the atom is unipolar (intensity 0–15, opposite pole spare).*

**A. Clean ± pairs already in the 32 (both poles present → one signed dim each):**

| − pole | + pole |
|---|---|
| decomposition | synthesis |
| compression | expansion |
| uncertainty | confidence |
| induction | deduction |

**B. Opposite not listed but VALUABLE → materialize the − pole (new atom on that dim):**

| atom (listed) | true opposite (materialized) | why valuable |
|---|---|---|
| contradiction | coherence/consistency | the §3 contested↔settled axis |
| counterfactual | factual/actual | Pearl rung-1 vs rung-3 |
| desire | aversion | conative ± (approach/avoid) |
| empathy | detachment | analytical vs relational mode |
| negation | affirmation | polarity operator |

**C. Unipolar — no valuable opposite (intensity 0–15, − pole spare):**

- *Reasoning domains* (selectors, not opposites): causal, temporal, spatial, modal, deontic.
- *Operations*: analogy, metaphor, narrative, hypothesis, retrieval, clarification, revision, quantification, comparison, classification, perspective, intention, belief.

**D. Needs your call — the inference triad.** You hinted `abduction–induction` and `deduction–induction`. deduction↔induction is the clean pair (A). **abduction** is the odd one: it pairs cleanly as **retrieval ↔ abduction** (recall-known − / hypothesize-new +), or you intend a single 3-pole inference-direction structure. Flag, not guess.

## Count

- 4 clean pairs (A) → 4 signed dims, 8 atoms
- 5 materialized pairs (B) → 5 signed dims (5 listed atoms + 5 new − poles)
- ~18 unipolar (C) → intensity-only poles, − side spare
- 1 (abduction) pending your call (D)

32 listed atoms occupy 32 of the 64 poles; the 9 dichotomy-bounded ones (A+B) load their opposite pole; the unipolar ~18 leave their − pole spare (or available for future opposites under the evaluate-for-value rule).

## OPEN

1. Resolve the inference triad (D) — abduction pairing.
2. Confirm the 5 materialized opposites (B) are the right ones / right − poles.
3. i4 weight encoding: unsigned 0–15 (harvest, intensity) vs signed −8..+7 (bipolar dichotomy). The dichotomy-bounded atoms (A+B) want signed; the unipolar (C) want unsigned-intensity — likely a **mixed encoding** to settle in D-ATOM-1.
