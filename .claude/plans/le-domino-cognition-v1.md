# LE-Domino Cognition — converged north-star + shared kanban-door contract (v1, 2026-06-02)

> Capstone of the 2026-06-02 co-design session (jan + Opus), grounded by two 3-way fan-outs.
> **Thesis:** the most *expressive* substrate for the most *efficient* ops — one substrate, one
> operation, a local gradient, a self-propagating cascade — **grounded (never wishful) by construction.**
> **Status: DESIGN over mostly-scaffold.** The operators exist as types+tests; the wiring IS the build.
> This doc is also the **shared contract** where this work meets a parallel session's
> SurrealDB-actions→kanban push (see § Shared door).

## The one-paragraph architecture
**One substrate** — the mailbox SoA (`MailboxSoA`), owned by Rust: hot = `&mut` (write), witness =
`&` (read-only reference to another mailbox), cold = permanent → **Fact** (a calcified `SpoRecord`).
**One operation** — SPO-2³ NARS `syllogize` over the **full** `CausalEdge64` (the `CausalMask`
8-subset lattice + s/p/o planes + NARS truth), *not* XOR/scalar. **One driver** — the local pairwise
NARS `(frequency, confidence)` **diff** between two mailboxes, and the sign of its change. **One
cascade** — the LE contract self-propagates as a **domino** through the witness-chain, forward and
backward; whatever it can't resolve locally, it **triggers** (never fabricates).

## The driver — the (f,c) diff is the local free-energy gradient
- **diff drops (converge)** → damp → **commit / calcify → Fact** (the Rubicon `FreeEnergy<0.2` gate,
  computed locally/cheaply: a pairwise subtraction, not a global sweep).
- **diff gains (diverge):** frequency diverging under high confidence = contradiction → **fork →
  counterfactual** (`deposit_counterfactual`, separate lane, never-as-SPO-truth); low confidence → **escalate**.
- **won't settle** = the *resistance* signal that kicks **elevation** up a tier.
- Two sinks only — **settle (commit)** or **fork (counterfactual, preserved not amplified)** — so the
  cascade can't run away. The diff *is* the surprise term, made local and integer-cheap.

## The cascade — the LE-contract domino
A thinking update tips the first domino; it propagates through `witness → mailbox → witness → mailbox`:
- **backward** (toward what I was built on) = **metacognition** (witness-of-witness = chain depth; re-`syllogize` the premises).
- **forward** (toward what was built on me) = **belief revision**.
No orchestrator — local message-passing. Backward rides the **existing** W-chain (the `witness_table`
"Markov belief-update arc"); **forward needs one new field** (a "who-witnessed-me" link — the single
real missing primitive).

## Escalation — what LE can't do, it triggers (never fabricates)
Tier-0 = the local domino. On local failure (diff won't settle, premise missing) it triggers, in
ascending cost: **NARS revision** (pull evidence) · **thinking-style** (different OGIT-class lens) ·
**kanban-ractor** (re-dispatch/re-plan) · **mailbox** (heavier op). This is the elevation cost-model.
Closure: every gap routes to something *real*.

## Shared door (COORDINATION — read before touching the kanban surface)
The kanban is a **multi-producer door**; producers meet at the **contract**, never a private path:
- **Producer A — version-poll / surreal:** `Scheduler::on_version<V: MailboxSoaView>(view, DatasetVersion, ExecTarget) -> Option<KanbanMove>` (`lance-graph-contract/src/scheduler.rs:51`). Reactive seam = **Lance-update = witness-pointer = SurrealDB-kanban-subscription** (AGENT_LOG); SurrealDB = a **view over leading LanceDB** (handover #422), not the cold store.
- **Producer B — a parallel session** is building **SurrealDB DDL-AST `actions`** that push into this same door (not yet on this board → coordinate via the contract).
- **Producer C — LE-escalation** (this work): emits the same `KanbanMove` on local failure.
- **Meeting point:** `KanbanMove` (`kanban.rs:112`, ≤16 B owned, Rubicon-gated) + `ConsumerEnvelope::Plan`. Do NOT add a private push path.

## Anti-wishful guarantee (by construction)
1. NARS confidence decays per hop → cascade self-terminates below the commit floor.
2. Witness-grounding — only real *written* edges are reasoned over ("what is not written is not traceable").
3. Firewall — similarity *proposes* (float, upstream); `syllogize`/CAM *addresses* (integer, exact). No crossing.
4. Smart constructors — ill-formed inference returns `None` at construction, not a runtime fantasy.
5. The diff's two sinks (settle/fork) + escalate-don't-fabricate close it.

## Everything / nothing — the seam-map (live vs scaffold/absent)
| Layer | Exists as | Status |
|---|---|---|
| SPO-2³ op | `CausalEdge64::syllogize` + `causal_edge::pearl::CausalMask` | built+tested, **0 callers** |
| NARS (f,c) + revision | `NarsTables` (`tables.deduction[f1*256+f2]`), edge `confidence()` | **live** (planner) |
| mailbox uses full edge | `apply_edges` scalarizes (mantissa·conf) | **dumb-merge** (the gap) |
| shared door | `Scheduler::on_version → KanbanMove` / `ConsumerEnvelope::Plan` | types live, **test-only impls** |
| backward W-chain | `WitnessTable` / `WitnessEntry` arc | type live; **W-slot unpopulated** (`w_slot()`→0) |
| forward link | — | **absent** (the one new field) |
| cold = Fact | `calcify` → `SpoRecord` + `Tombstone` | **`todo!()`, uncompiled orphan** |
| ephemeral radix trie | vart `Tree::clone()` COW | **not a dependency** (doc-prose only) |

## Ordered seams (build path)
1. **Backward domino** *(collision-free, internal)*: on edge write, populate the W-slot, walk the W-chain, `syllogize`, compute the pairwise (f,c) diff per hop, route on drop/gain. First place doctrine becomes running, grounded behavior.
   - **ATOM SHIPPED (2026-06-02):** `CausalEdge64::route_against(self, prior) -> DominoStep{Settle,Fork,Escalate,Terminal}` (`causal-edge/src/syllogism.rs`) — the one-hop NARS-grounded router on the pairwise (f,c) diff; 6 tests green offline; the **first live caller of `syllogize`** (needed one new type, `DominoStep`, the decision sink).
   - **Remainder:** the multi-hop W-chain walk + the cross-mailbox resolver (`mailbox_ref` → witnessed edge) + acting on the decision (write conclusion / `deposit_counterfactual` / emit `KanbanMove`) + populating the W-slot on emit. The live driving loop awaits the ractor seam.
2. **Escalation emit** *(shared door — coordinate with the surreal session)*: on won't-settle, emit `KanbanMove` via `on_version` / `ConsumerEnvelope::Plan`.
3. **Forward link** on `WitnessEntry` (the one new field) → forward propagation.
4. **Cold / calcify + radix-trie** *(big; blocked on D-PERSONA-5 / ractor)*: compile the orphan, amortize to Fact, vart COW for ephemeral forks.
