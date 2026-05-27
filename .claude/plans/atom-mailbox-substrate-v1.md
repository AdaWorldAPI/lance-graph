# atom-mailbox-substrate-v1

> **Status:** PROPOSAL (implements `EPIPHANIES.md` E-LADDER-SERVES-MAILBOX, 2026-05-27).
> **Confidence:** HIGH on the mechanism-to-existing-machinery mapping (every piece anchors to a shipped type or an iron rule); **CONJECTURE on the atom basis itself** (D-ATOM-0, the load-bearing unsolved decision) and on the I4-32D SIMD layout until probed.
> **Plan file:** `.claude/plans/atom-mailbox-substrate-v1.md`
> **Predecessors:** `rung-persona-orchestration-v1` (D-PERSONA-1 shipped the escalation/checklist/ghost types this extends), `rung-mul-grounding-v1` (MUL experience curve + wisdom marker), `cognitive-substrate-convergence-v1` (CausalEdge64 v2, Pearl 2³).
> **Anchored iron rules (FINDING):** `I-VSA-IDENTITIES` (persona=Layer-2, Test 0 register-laziness, bipolar ±1, Test 2/3), `E-BATON-1` (mailbox-as-owner, no persisted singleton), `I-LEGACY-API-FEATURE-GATED` (`CausalEdge64` 4-bit signed mantissa @46-49, Counterfactual=−6), The Click (Staunen×Wisdom, Resolution thresholds, `awareness.revise`).

---

## The one-line thesis

The escalation ladder serves the **mailbox**, not the persona. Under it sits a three-layer cognitive basis — **atoms (bipolar I4-32D) → thinking styles (compositions) → persona recipes (compositions + thresholds)** — where each atom is *measured by a quorum*, split quorums are *preserved as a counterfactual mantissa*, and memory is *ephemeral-hot in the mailbox, calcified-cold in SPO + a Lance tombstone-witness*.

## Scope (six pillars — see the epiphany for the full derivation)

1. **Ladder→mailbox reframe.** Persona = Layer-2 dispatch policy (β + fan-out pattern), not a container. The D-PERSONA-1 types are already mailbox-shaped; this is a reframe, not a rebuild.
2. **3-layer basis.** **atom** = one lane of the **LOCKED 33-dim TSV** (E-AGICHAT-DIMENSION-CONTRACT; 3 Pearl + 9 Rung + 5 Σ + 8 Ops + 4 Presence + 4 Meta) — bare-metal, not human-legible; **style** = one i4 vector over the atoms (the molecule); **persona** = composition of styles + thresholds. Cranelift templates compile the *recipe* (the object), not the atom lanes. Atoms dispatch through `cognitive-shader-driver` (which owns SIMD) — **no SIMD in the atom layer**. Business is an OGIT-inherited sidecar, not an atom.
3. **Quorum projection.** A dichotomy needs a quorum to place a measurement between its poles; each atom value = `(I4 position, quorum-confidence)` = NARS truth per axis. Splits are Contradictions, never averaged.
4. **Temperature axis.** wisdom↔Staunen = sampling temperature, self-regulated by free energy; the `WisdomMarker` 0.1 floor = minimum temperature (φ-1 humility). Distinct from plasticity (update-rate).
5. **Counterfactual mantissa.** On `is_split`: commit the majority pole, fork the minority into a counterfactual mailbox retained as a `CausalEdge64` −6 nibble; ghost-tier test on β headroom; minority win → `awareness.revise`.
6. **AriGraph hot/cold/tombstone.** Ephemeral-hot in mailbox → calcify to cold SPO → tombstone-witness in *versioned* Lance (= GoBD audit by construction). One compression hierarchy down the codec atlas.

---

## Decision gates

- **D-ATOM-0 — the atom basis. ✅ RESOLVED — the basis is LOCKED, not derived.** It is agichat's 33-dim TSV (`E-AGICHAT-DIMENSION-CONTRACT` / `CANONICAL_DIMENSION_ALLOCATION.md`): **3 Pearl + 9 Rung + 5 Σ + 8 Operations + 4 Presence + 4 Meta** = 33, restored on the shipped i4-32 floor. No ICA/PCA, no "demote the 36 styles" (the 36 `ThinkingStyle` ids are *styles* — vectors over the atoms — not the atoms). Catalogue committed in `contract::atoms::CANONICAL_ATOMS` + `.claude/knowledge/atom-basis-inventory.md`. Earlier "ICA/PCA over 36" framing was wrong and is retracted.
- **Remaining sub-gates (layout, not basis):** (i) 32-vs-33 carrier reconciliation (i4-32 floor holds 32 lanes; TSV is 32+1); (ii) "8 spare" (STYLE_ENCODING) vs "4 Presence + 4 Meta" (contract body); (iii) per-group i4 sign/scale (ordinal ladders = magnitude, the few ± lanes signed). NARS is **not** ~24 atom dims — NARS-inference is 3 of the 8 Operations; the rest of the families are orthogonal (supersedes the old "24 NARS" budget line).

---

## Deliverables + dependency DAG

| D-id | Scope | Crate / files | Depends on | Basis-dependent? |
|---|---|---|---|---|
| **D-ATOM-1** | LOCKED 33-TSV catalogue + `I4x32` bare-metal carrier (pack/unpack only; **no SIMD** — dispatch via `cognitive-shader-driver`) | `contract::atoms` (scaffolded ✓) | D-ATOM-0 ✅ | unblocked |
| **D-ATOM-2** | style/persona = I4-32D compositions; Cranelift recipe templates | `contract::jit` (`StyleRegistry`), `contract::thinking` (back enum with composition) | D-ATOM-1 | YES |
| **D-ATOM-3** | quorum-projection `(position, confidence)` per axis | `contract::escalation` (`InnerCouncil`→per-axis), `contract::a2a_blackboard` | D-ATOM-1 (axis shape); mechanism semi-independent | partial |
| **D-ATOM-4** | counterfactual mantissa: **v2** deposit `−6` on split, **v3** mailbox + `awareness.revise` | `contract::escalation`, `CausalEdge64` mantissa path | `is_split` (shipped) + mantissa (shipped) | **NO** |
| **D-ATOM-5** | AriGraph hot→calcify→Lance tombstone-witness + link integrity | `lance-graph` core (AriGraph), Lance versioned store | — | **NO** |

**Critical path:** D-ATOM-0 ✅ (locked TSV) → D-ATOM-1 (catalogue done; pack/unpack + carrier wiring remain) → D-ATOM-2 (the OO layer — the part that matters). D-ATOM-4 v2 and D-ATOM-5 are basis-independent. D-ATOM-4 v3 needs the ractor outer-swarm from `rung-persona` D-PERSONA-5. **SIMD is the `cognitive-shader-driver`'s, never the atom layer's.**

---

## Per-agent split (the `///` scaffold wave)

One **Sonnet** agent per deliverable, **disjoint file scopes**, **edit/write only** (no cross-file refactors). Each agent: (1) reads `.claude/board/AGENT_LOG.md` + `LATEST_STATE.md` + `EPIPHANIES.md` E-LADDER-SERVES-MAILBOX first; (2) writes **`///`-doc scaffolding only** — public type + signature surface with rustdoc specs, `todo!()` / `unimplemented!()` bodies, and `// BLOCKED: <question>` markers wherever a decision is missing (do **not** invent a basis, a version, or an API); (3) prepends its own AGENT_LOG entry. Iron rule for workers: **leave a `BLOCKED` marker rather than guess** (the surreal-poc Wave-A precedent).

- **Agent-A → D-ATOM-4 v2** (basis-independent, smallest, safest first slice).
- **Agent-B → D-ATOM-5** (basis-independent).
- **Agent-C → D-ATOM-3 trait surface** (mechanism only; axis wiring `BLOCKED` on D-ATOM-1).
- D-ATOM-1 catalogue is done (locked TSV); next is the `cognitive-shader-driver` carrier wiring + pack/unpack. **D-ATOM-2 (the OO style/persona object layer) is the deliverable that matters** — that *is* the metacognition; D-ATOM-1 just has to be bare-metal-correct and out of the way.

## Execution loop (per deliverable — "scaffold → review → implement → PR → green → merge → repeat")

1. **Scaffold** — Sonnet agent writes the `///` surface (above).
2. **Review (P2 gate)** — see *Review mapping* below; produces a findings list (correctness + reuse/simplification + iron-rule compliance).
3. **Implement** — fix the findings, **replace `todo!()`/`///`-stubs with real bodies**, resolve `BLOCKED` markers (escalate any that need a decision). Run `cargo test -p <crate>`.
4. **PR** — sub-branch `claude/atom-<d-id>` off `claude/splat3d-cpu-simd-renderer-MAOO0`; PR **into** the working branch (not main). Board hygiene in the same commit (STATUS_BOARD row, LATEST_STATE inventory, AGENT_LOG).
5. **Subscribe** — `subscribe_pr_activity` on the PR; autofix CI failures + review comments per the PR-activity protocol.
6. **Merge** — only when CI is green **and** the merge policy (below) permits.
7. **Repeat** for the next deliverable in DAG order.

## Review mapping ("P2 codex savant team" → what we actually have)

There is **no literal `codex` binary** in this environment. The P2 gate maps to one of: **(i)** the `/code-review` skill at `high`/`ultra` effort (ultra = multi-agent cloud review of the diff — closest to a "savant team"); **(ii)** spawning 2-3 **Opus** review agents with disjoint lenses (correctness / iron-rule-compliance / reuse-simplification) and synthesizing on the main thread. Default proposal: **`/code-review high` per PR, escalating to `ultra` for D-ATOM-1/2** (the basis-bearing ones). Confirm in the gating question.

---

## Invariants (inherited + new)

- Restore-on-substrate, not port · persona = Layer-2 catalogue (no container struct) · atoms ≤ the I4-32D bipolar basis; NARS *type* in a register (Test 0) · markers ≤ 32 identities (I-VSA-IDENTITIES) · splits = Contradiction, never averaged · counterfactual stays in a separate lane (Counterfactual-tagged, never observed SPO truth) · AriGraph = the one graph (no second store) · ephemeral bundle, no persisted singleton (E-BATON-1) · ractor async only at the swarm boundary (no double-mailbox) · respawn bounded (N retries → FailureTicket) · `latency_budget` time arbiter, no wall-clock in the hot Pod · i4 precision tradeoff cited to `FormatBestPractices.md`; SIMD path gated on MANDATORY `ndarray-vertical-simd-alien-magic.md`.

## Honest gaps / open questions

- **D-ATOM-0 is genuinely unsolved** — the basis derivation is asserted nowhere yet; this plan cannot proceed past D-ATOM-2 scaffolding without it.
- The 36↔64 arithmetic (36 named atoms inside 32 dims / 64 poles, ~28 spare?) was never closed in dialogue — D-ATOM-0 must resolve it.
- Temperature as flat peer dim vs **meta-atom read first** (one-pass vs two-stage I4 sweep) — layout-level open.
- `WitnessCorpus` and `SigmaTierRouter` Σ-tier D-ids (homes for the tombstone-witness and the Rubikon admission gate) were cited from dialogue — **verify against STATUS_BOARD before wiring** D-ATOM-5 / D-ATOM-4 v3.
- substrate-Markov re-scope (unsolicited-materialization-only) is **out of scope here** — it awaits the `[FORMAL-SCAFFOLD]` dependency check.
