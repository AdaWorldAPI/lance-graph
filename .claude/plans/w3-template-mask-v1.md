# W3 — the grammar template as StepMask: no finetuning, three tables — v1

> **Status:** PROPOSED (doc-only). Gated on **D-W3M-1** — no further W3 work
> lands before that probe is green. Realizes the item parked out-of-scope in
> `deepnsm-v3-convergence-v1` §4 ("Grammar **templates** as compiled thinking
> templates (StepMask) — that is W3, a separate wave; `StepMask` does not exist
> in source yet"). Extends the four-paper FINDING in
> `.claude/knowledge/left-corner-grammar-tree-pointer-fabric.md`. Does NOT
> supersede any shipped decision.
>
> **Organizing frame:** the operator's focal question — *"do we need
> finetuning for the left-corner parsing tree (e.g. template mask)?"* — is
> answered **NO**. A template mask is not a workaround for the absence of
> training; it is **exactly the mechanism the left-corner literature already
> uses.** The whole "learning" in LC parsing at this level is **counting into
> tables** — never gradients, never an optimizer. This plan lands three
> table-shaped artifacts and zero training loops.

---

## 0. The ruling in one line (do not re-derive)

**No finetuning. Three tables, zero optimizers.** The left-corner tree runs on:

1. an **8 KB left-corner relation mask** (can category X begin goal A?),
   compiled offline from the rule inventory — **never learned**;
2. a **64 KB attach-vs-project pair table** (the ONE nondeterministic LC
   choice), **count-derived** from a treebank slice — same class as
   `freq_is_cosine` (counting, no training loop);
3. the **existing Escalate → global-graph channel** for non-local context
   (annotation, not weights).

Total budget claim (**CONJECTURE until D-W3M-1/2/3**): the whole LC tree runs on
**8 KB mask + 64 KB pair table + escalation** — everything table-lookup,
consistent with the 611M-lookups/sec doctrine. Nothing on this path touches a
gradient.

**This is a doc.** No `StepMask` type exists in source yet; §5 names where it
lands. The gate (D-W3M-1) is buildable now against a treebank slice with zero
new contract types.

---

## 1. Why the answer is NO FINETUNING — the mechanism, paper-grounded

Status: **FINDING** for the mechanism mapping (grounded in the four-paper review,
`left-corner-grammar-tree-pointer-fabric.md`, 2026-07-22); **CONJECTURE** for
every accuracy number until its named probe runs.

The left-corner tradition never had gradient learning at this layer. Its three
decision surfaces are all counting-or-compilation:

| LC decision surface | Paper anchor | What "learning" means | Our substrate |
|---|---|---|---|
| **Left-corner relation** — "can category X begin goal A?" | Moore 2000 (precompiled constant-time pair check; his single biggest lever, **+67%** from cheap-check-first) | **Compilation** from the rule inventory. Never learned. | `standing_mask::fires` (`dirty ∩ interest ≠ ∅`) / `WideFieldMask` — the existing mask algebra |
| **Attach-vs-project** — the ONE nondeterministic LC choice | Manning & Carpenter 1997 (pair-conditioning alone cut errors **~20%** over mother-only) | **Counting** into a table conditioned on the `(left-corner, goal)` PAIR. No training loop. | one `palette256:palette256` u8 count table — structurally ONE 256:256 rail |
| **Non-local context** when local underdetermines | Roark & Johnson 2000 (ancestor annotation captures nearly all the gain AND makes search cheaper) | **Annotation**, not weights | the existing `WaveGrounding::Escalate` → global SPO/AriGraph channel |

Liu 2025's arc-eager result licenses resolving the attach/project choice
**greedily** (deterministic argmax over the pair-table row) — **no beam, no
k-best.** Determinism is a feature at this scope (consistent with the "no
PCFG/beam/FOM machinery" KEEP-invariant already logged in the knowledge doc,
and with the scarcity-inversion: beams were a RAM artifact, not a principle).

**The load-bearing consequence:** a "template mask" is not a substitute for the
model we didn't train. It is the compiled form of the grammar itself. Asking
"do we need to finetune the mask?" is asking "do we need to train a lookup
table?" — the LC literature's answer, for thirty years, is *no, you compile it
and you count into it.*

---

## 2. Artifact 1 — the left-corner relation = TEMPLATE MASK (8 KB)

**FINDING (mechanism):** the LC relation IS the existing mask algebra. Per goal
category, a bitmask of admissible left-corner categories: bit `x` of row `A` set
⟺ category `x` can begin a constituent of goal category `A`.

- **Shape:** a 256-category inventory → `256 × 256` bits = **8 KB** — the
  recurring 256×256 LUT shape (bgz17 palette distance/compose tables, helix
  `DistanceLut`, bgz-tensor attention-as-lookup, OGAR key-tier centroid tile).
- **Source:** compiled **offline from the rule inventory** — a transitive
  closure over "A → X …" productions (the standard LC-relation precompile).
  Never learned, never counted; it is a deterministic function of the grammar.
- **Substrate:** this is `standing_mask::fires` (`dirty ∩ interest ≠ ∅`) at
  category granularity, or `WideFieldMask` for inventories past 64 categories.
  The mask check is **one O(1) bitmask AND**.
- **Consumer role:** the cheap check. It prunes most continuations before the
  pair table is ever read (Moore's +67% lever).

**WARNING (Moore Table 3, adopted verbatim from the knowledge doc):** do **NOT**
left-factor the rule set when compiling this relation — left factoring injected
empty categories and degraded LC parsing, sometimes catastrophically. Use
**bottom-up prefix merging (BUPM)**, the right transform for a bottom-up
streaming recognizer.

---

## 3. Artifact 2 — attach-vs-project = a COUNT TABLE (64 KB)

**FINDING (mechanism):** the single nondeterministic LC choice — at each step,
**attach** the recognized left-corner to the current goal, or **project** a new
constituent above it — is resolved by ONE table conditioned on the
`(left-corner category, goal category)` PAIR.

- **Shape (resolved — not two contracts):** the RUNTIME table is
  **decision-only**: `palette256:palette256` → **one u8 per cell** → `256 × 256`
  = **64 KB**, each cell holding the pre-resolved argmax outcome (a small enum:
  `0 = attach`, `1 = project`, `2 = unseen/fallback`). The attach/project COUNT
  PAIR that the argmax is derived from lives in a **build-time side artifact**
  (the tally, kept out of the hot path), NOT in the 64 KB cell — a u8 cannot
  hold two counts, so it holds only their resolved decision. This keeps the
  runtime read a single O(1) byte lookup and matches the compiled-offline
  philosophy (Artifact 1's mask is likewise compiled, not carried as raw rule
  text). Manning & Carpenter's core innovation (pair conditioning, not
  mother-only) is **structurally ONE 256:256 rail** — the same `X:Y` rail shape
  as the LE contract's `6×(u8:u8)` carving.
- **Unseen `(lc, goal)` cells:** a cell for a pair never observed in the
  training slice is stamped `2 = fallback` at compile time and, at runtime,
  defers to Artifact 3's Escalate channel (the global graph) rather than
  guessing attach/project. Unseen is an explicit, addressable state — never a
  silent default to `attach` — so coverage gaps are visible, not hidden.
- **Source:** **count-derived** from a treebank / corpus slice — for each
  observed `(lc, goal)` context, tally how often the gold parse attached vs
  projected. **Same class as `freq_is_cosine`** (`E-FREQ-IS-COSINE-REPLACEMENT-1`,
  AGENT_LOG 2026-07-19): distance/decision FROM counting, **no gradient, no
  training loop.**
- **Resolution:** the argmax is computed **at build time** (over the side
  artifact's counts) and baked into the cell; at runtime the resolution is a
  **single deterministic byte read** — `attach`/`project`/`fallback` — with no
  beam and no per-step search (Liu 2025 arc-eager license). "Greedy
  deterministic" describes the decision the byte encodes, not a runtime scan.
- **Consumer role:** the second, more expensive read — gated by the Artifact-1
  mask check. Never consulted when the mask already pruned the continuation.

---

## 4. Artifact 3 — non-local context = the existing Escalate channel

**FINDING (mechanism):** when the local `(lc, goal)` pair underdetermines the
decision, escalate to the global graph. This is **already built** — no new
artifact.

- **Substrate:** `WaveGrounding::Escalate` → global SPO / AriGraph. This is
  deepnsm's ancestor-annotation channel (Roark & Johnson §3.3: parent/ancestor
  annotation is non-local context that improves accuracy AND cuts
  states-considered — *"the non-local information not only improves the final
  product of the parse, but it guides the parser more quickly"*).
- **Nature:** **annotation, not weights.** The resident book IS the annotation,
  O(1) addressable (the scarcity-inversion default posture — R&J smuggled
  non-local context into category labels because their parser could not see
  back; here the whole context is resident).
- **CONJECTURE (testable):** escalation makes resolution **cheaper**, not just
  more correct — expect fewer total pair-table reads on ambiguous inputs when
  escalation is available. Registered as a side-observation on D-W3M-2, not its
  gate.

---

## 5. The ORDERING INVARIANT — cheap → expensive (Moore, adopted)

The three surfaces run in strict cost order; this is the +67% lever, not a
micro-optimization:

```text
1. mask check      — one O(1) bitmask AND (Artifact 1). Prunes most continuations.
2. pair-table read — one 256:256 u8 lookup (Artifact 2). Only on mask-survivors.
3. escalate        — global-graph read (Artifact 3). Only when local underdetermines.
```

**INVARIANT (Moore, +67%):** the cheap O(1) local check gates the expensive
global check. Never read the pair table before the mask has pruned; never
escalate before the pair table has been consulted. (Same shape as the existing
±8-local-vs-graph-global two-stage filter — local first, escalate on failure.)

---

## 6. Home — the compiled-thinking-template stack, NOT the FSM

**FINDING (placement):** the grammar template = `(goal category → StepMask of
admissible parser moves)`. It lives in the **compiled-thinking-template stack**
(elixir-template × StepMask, per the V3 rulings —
`.claude/v3/README.md`, "compiled thinking templates"), **not in the FSM.**

- The FSM (deepnsm-v2 `fsm.rs`, 6-state PoS) stays flat and deterministic — its
  scope is right (Moore Table 1: a full CFG averages 7.2×10²⁷ parses/sentence;
  flatness is the control experiment). The grammar template is the *compiled
  move-admissibility layer above it*, keyed by goal category.
- `StepMask` does **not** exist in source yet (parked in
  `deepnsm-v3-convergence-v1` §4). D-W3M-3 introduces it as `(goal → StepMask)`
  consuming the compiled-thinking-template stack; the `v3-template-smith` agent
  carries this surface.
- **INVARIANT (Moore §7): keep reference pointers minimal** — identity +
  position; the rest is reconstructible (argues against widening the move
  encoding). The template maps goal → admissible moves; it does not carry parse
  state.

---

## 7. Staged deliverables (gates before code)

### D-W3M-1 — THE GATE FOR EVERYTHING ELSE: count-vs-oracle probe

The attach/project pair table (Artifact 2), derived by **pure counting** from a
small treebank slice, resolves the attach-vs-project choice measurably better
than the unconditioned (mother-only / prior) baseline.

- **Method:** take a small treebank slice; for each `(lc, goal)` context tally
  gold attach vs project; build the 256:256 count table; measure resolution
  accuracy of greedy-argmax vs the treebank's OWN parses (held-out).
- **Literature anchor:** Manning & Carpenter's **~20% error reduction** at POS
  level from pair-conditioned counts over mother-only.
- **Registered pass floor (set BEFORE running):** pair-conditioned resolution
  accuracy ≥ unconditioned baseline **+ 0.05 absolute** (conservative floor;
  M&C observed ~20% relative error reduction — this asks for a fraction of it as
  a green signal, not the full effect).
- **KILL:** if the counts do **not** beat the unconditioned baseline (Δ ≤ 0),
  the count-table mechanism is falsified for our data — STOP; no further W3 work.
- **Buildable now.** Zero new contract types; a probe example over a treebank
  slice (same shape as `freq_is_cosine.rs` / the P-series probes). Pure
  counting, no optimizer.
- **Status:** CONJECTURE (the ~20% is M&C's number on WSJ, not yet our data).

### D-W3M-2 — the 8 KB LC relation mask + cheap-check-first wiring

Compile the `256×256` bit LC relation (Artifact 1) from the rule inventory
(BUPM, **not** left-factored) and wire the mask-first ordering (§5).

- **Gate:** mask-first pruning **measurably reduces pair-table reads with zero
  accuracy change** vs pair-table-only (byte-identical decisions, strictly fewer
  Artifact-2 reads). Side-observation: record whether escalation count drops too
  (the D-W3M-1-free CONJECTURE from §4).
- **Depends on:** D-W3M-1 green (the pair table it gates must be validated first).
- **Status:** CONJECTURE (pruning ratio unmeasured); FINDING that the mask is
  compiled-not-learned.

### D-W3M-3 — StepMask integration: the grammar template as `(goal → StepMask)`

Introduce `StepMask` in the compiled-thinking-template stack; express the grammar
template as `(goal category → StepMask of admissible moves)` consuming Artifacts
1+2 through that surface.

- **Gate:** **identical resolutions through the template path vs the direct
  path** — byte-parity of decisions (the template is a re-expression, not a new
  policy). Same falsifier shape as the P-series integer-exact probes.
- **Depends on:** D-W3M-1 + D-W3M-2 green.
- **Home:** compiled-thinking-template stack (elixir-template × StepMask), NOT
  `fsm.rs`. Carried by `v3-template-smith`.
- **Status:** CONJECTURE until the byte-parity test is green.

---

## 8. Adopted invariants (from the four-paper FINDING)

- **INVARIANT (Moore, +67%):** cheap O(1) local check gates the expensive global
  check (§5).
- **INVARIANT (Moore §7):** keep the move encoding minimal (identity + position);
  the rest is reconstructible — do not widen it.
- **WARNING (Moore Table 3):** do NOT left-factor the rule set; use BUPM.
- **LICENSE (Liu 2025 arc-eager):** resolve attach-vs-project **greedily**
  (deterministic argmax) — no beam, no k-best.
- **KEEP (no PCFG/beam/FOM):** determinism is a feature at this scope; the mask +
  count table + escalation is the whole machine.
- **NO FINETUNING (operator ruling, this doc):** all three surfaces are
  compilation or counting. A gradient on this path is a design smell — flag it.

---

## 9. Cross-refs

`.claude/knowledge/left-corner-grammar-tree-pointer-fabric.md` (the four-paper
FINDING this plan compiles into deliverables); `deepnsm-v3-convergence-v1` §4
(where W3 was parked); `E-FREQ-IS-COSINE-REPLACEMENT-1` (AGENT_LOG 2026-07-19 —
the counting-not-training precedent D-W3M-1 mirrors); `E-MARKOV-TEMPORAL-STREAM-1`
/ `E-HORIZON-NOT-BOUND-1` (the temporal-stream + escalation substrate Artifact 3
rides); `.claude/v3/README.md` "compiled thinking templates (elixir-template ×
StepMask)" (the home for D-W3M-3); `.claude/v3/agents/BOOT.md` `v3-template-smith`
(the carrier agent); `standing_mask::fires` / `WideFieldMask` (Artifact 1
algebra); `crates/deepnsm-v2/src/{fsm,wave}.rs` (the FSM the template sits above,
and the Escalate channel). Papers: Manning & Carpenter 1997 (IWPT-97, PLCG);
Roark & Johnson 2000 (arXiv cs/0008017); Moore 2000 (IWPT-2000); Liu 2025
(JLM 13(2)).
