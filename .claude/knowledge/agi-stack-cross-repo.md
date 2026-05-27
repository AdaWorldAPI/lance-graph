# AGI stack — cross-repo ground truth (ada-consciousness · ladybug-rs · neo4j-rs)

> **READ BY:** truth-architect; integration-lead; anyone grounding the 34 tactics / SPO(Q) / container architecture.
> **Sources (AUTHORITATIVE, external — user-provided, NOT in MCP allowlist, reference-only never port-target):**
> - `ada-consciousness` `docs/34_TACTICS_VS_ADA.md` @707264 → see `34-tactics-vs-ada.md`.
> - `ladybug-rs` `docs/34_TACTICS_x_REASONING_LADDER.md` @177a321 (this doc, §1).
> - `neo4j-rs` `docs/SPOQ_AUDIT.md` @c3c2fde (this doc, §2).
> **Date:** 2026-05-27.

## §1 — ladybug-rs: 34 tactics × the Sun et al. (2025) reasoning ladder

**The paper (Sun et al. 2504.11741, NeurIPS 2025):** LLMs fail on hard reasoning from 3 *structural* deficiencies — (a) multiplicative error `P=0.9^n→48%@n=7`, (b) convergent strategy lock-in, (c) no self-correction. 4 tiers: Easy / Medium (~90% post-SFT) / **Hard (plateau ~65%)** / **Extremely-Hard (<10%)**.

**ladybug-rs implements all 34 as structural primitives**, each tagged to the tier it breaks + a peer-reviewed mechanism + an exact Rust module. They reduce to **three mechanisms** (this is the spine):

1. **PARALLEL INDEPENDENCE** (vs sequential dependency) → breaks Tier-2 error propagation. 7-layer stack reads a shared fingerprint core, 7 CAKES search algos, shadow-parallel verify. Tactics #1,2,5,20,26,30. (Berry-Esseen, Avizienis N-version, Wolpert NFL.)
2. **TRUTH-AWARE INFERENCE** (vs next-token prob) → Tier-2 detection + Tier-3 insight. Every step carries a **NARS TruthValue (freq, conf)**; revision detects conflict; abduction generates hypotheses; CollapseGate HOLD keeps superposition; Brier calibration. Tactics #3,7,10,11,17,21,28. (Wang NARS, Peirce abduction, Brier, Festinger.)
3. **STRUCTURAL DIVERGENCE** (vs convergent optimization) → Tier-3 creativity wall. 12 ThinkingStyles parameterically distinct (Analytical↔Creative dist >0.6, can't converge); **counterfactual world via XOR `world⊗factual⊗counterfactual`** (Pearl Rung 3); Granger temporal; reversible cross-domain bind; TD-learning on style Q-values. Tactics #4,6,9,13,23,28,31,34. (Guilford, Pearl, Granger, Gentner.)

Maps directly onto the hardware partition (datapath/control/gate) and the markers we already established. Key module anchors: `cognitive/{recursive,metacog,collapse_gate,style}.rs`, `nars/{contradiction,adversarial,inference}.rs`, `search/{causal,temporal,hdr_cascade,distribution}.rs`, `world/counterfactual.rs`, `orchestration/{debate,persona}.rs`, `fabric/shadow.rs`, `core/vsa.rs`.

## §2 — neo4j-rs SPOQ_AUDIT: the container architecture (and the 4th role Q)

**SPOQ = S, P, O, + Q (Qualia) — FOUR roles, not three.** Crystal role seeds `ROLE_S/P/O/Q`; trace = `S⊕ROLE_S ⊕ P⊕ROLE_P ⊕ O⊕ROLE_O ⊕ Q⊕ROLE_Q` (`spo.rs:768-783`). **This reframes the "SPO 2³" question:** the *causal* lattice is SPO 2³ = 8 projections; **Q (qualia/affect) is the 4th role bound alongside** — phenomenal, orthogonal to the causal powerset. Full powerset SPOQ = 2⁴ = 16, but Q is the affective overlay, not a causal projection. (Matches the earlier finding: the 18D Qualia is a *separate* vector.)

**Container substrate:**
- `Container = [u64;128]` = 8,192 bits = 1 KB (16 AVX-512 loads). `CogRecord = meta(1KB) + content(1KB) = 2KB`.
- **MetaView word layout** W0-W127: W0=DN, W1=type, W2=time(created/modified ms), W4-7=NARS truth, W8-11=CollapseGate, W12-15=layer markers, W16-31=edges, W32-39=RL/Q-values, W56-63=qualia, **W112-125 reserved**, W126-127=checksum.
- `belichtungsmesser()` 7 sample points `[0,19,41,59,79,101,127]` → (mean, sd) = the **SD entropy gate** (FLOW<0.15 / HOLD / BLOCK>0.35) — the implicit-gating marker, container-native.
- **144-verb codebook**; **10-layer cognitive stack** (L1 Recognition→L10 Crystallization; L7 Contingency=XOR-bind counterfactual, L8 Integration=bundle, L10 promote Fluid→Node).
- **One-binary blackboard:** agents are threads reading `&BindSpace` (zero-copy); debate/shadow/roleplay become in-process. neo4j-rs collapses to a ~2,100 LOC **Cypher compiler** (`CypherEngine::query(&BindSpace)`) — the external query bridge; the DB functionality moves into ladybug-rs at the Container level.

**Audit verdict:** 26/26 factual claims verified, 7 minor discrepancies (most urgent D5: 10 layers × 5-byte markers = 50B in 32B → marker overflow, data-corruption risk for L8-10). Top expansion: counterfactual BindSpace snapshots (copy-on-write `Arc<[CogRecord]>` fork → explore counterfactual world read-only → merge/discard).

## §3 — reconciliation with lance-graph

- The 34 tactics' three mechanisms = the workspace's datapath(shader) / control(planner+escalation) / gate(elevation+SD/F markers) partition. NARS-truth-per-step = `contract::nars` + spo truth semiring. CollapseGate SD = the entropy gate (Invariant #2). Counterfactual = `CausalEdge64` −6 mantissa / SPO=0b111. ✓ consistent.
- **SPOQ's Q** = the qualia role → lance-graph's `QualiaColumn` (18→16D) / `QualiaI4_16D`. So the AGI-as-SoA four columns map: Fingerprint≈S/O content, Edge≈P+causal, Meta≈layer/RL, **Qualia≈Q**.
- **Caveat:** Container `[u64;128]` (8K-bit) vs lance-graph's 16K `Vsa16kF32`. ladybug ran 10K-D and *failed* (E-AGICHAT-DIMENSION-CONTRACT: "empty cathedral"); the workspace restores the *contract* on the i4 SoA floor, not the 10K carrier. The SPOQ container layout is a *design reference* for the MetaView word budget, not a port.

## §4 — open / for the inventory

- The full 36 NARS styles (now ~20 named in `34-tactics-vs-ada.md`) + the 208-dim address space (32 verbs + 36 GPT + 36 NARS + 11 presence + 5 archetype + 3 TLK + 4 affect + 33 TSV) are the real composite atom budget.
- Reconcile `spo-2cubed-list-coverage.md`'s 34-table *status* column against §1 here (ladybug implements all 34; ada-consciousness implements 21 — both authoritative for their repo).
- Decide: does the workspace track **SPO 2³ (8, causal)** or **SPOQ 2⁴ (16, +qualia)** as the lattice? The audit says Q is bound but qualia is a separate affective role — lean SPO 2³ for causality, Q as overlay.

**Cross-ref:** `34-tactics-vs-ada.md`, `spo-2cubed-list-coverage.md`, `atom-basis-inventory.md`, `EPIPHANIES.md` E-AGICHAT-DIMENSION-CONTRACT (the 10K "empty cathedral" → restore-on-SoA lineage), THINKING_RECONCILIATION.md (5 taxonomies).
