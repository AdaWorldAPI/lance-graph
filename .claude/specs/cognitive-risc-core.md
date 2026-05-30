# Cognitive RISC — Load-Bearing Core (v0.1)

> Session-boot context. Paste at the top of any new session (Claude Code or chat).
> Contains ONLY invariants. If a line here is violated, the architecture breaks.
> Everything not here is downstream and reconstructible from here.

## The one-sentence thesis

**Dumb uniform substrate (SPO), smart operations above it.** The intelligence is never *in* the triple — it's in what operates over uniform triples. SPO's refusal to be smart is the precondition for everything above it being smart. This is RISC: dumb uniform instructions, cleverness pushed up into the compiler.

## The five-layer stack (the "RISC" frame)

| Layer | Role | Cadence | RISC analogue |
|---|---|---|---|
| **Substrate** | SoA, LE byte contract, surrealkv WAL/ACID. Policy-free state. | persistent | register file |
| **Compilation** | planning-phase JIT × AST -> compiled candidate sets | plan-speed (slow) | compiler |
| **Schedule** | kanban = the precipitated plan; hands candidates to shader | per-plan | instruction issue |
| **Execution** | cognitive shader runs precompiled candidates over SoA | sub-us (fast) | execution unit |
| **Producer** | Rubicon now / agents later. Drives compilation. | — | the program |

**Only the Producer layer changes under the AGI-inversion.** Everything below is producer-agnostic. That is the whole reason the inversion is a swap, not a rewrite.

## The invariants (violating any one = re-federation / CISC slide)

1. **Nothing semantic in the register file.** Substrate stores bytes; the layer above assigns meaning. Meaning in the byte layout = welded to today's design = inversion costs a schema migration. This is the master rule; the rest are corollaries.
2. **<f,c> + discovery_origin travel as opaque payload.** The substrate never interprets a candidate's truth-value or origin; only Producer/planner reads them. Lets candidate-combination logic be arbitrary without the substrate caring.
3. **Uniform = uniform logical schema + partitioned physical ownership.** Not one god-array. Single-writer per mailbox; cross-mailbox refs are **witness pointers**, never shared writes.
4. **Load/store discipline.** Only the **commit gate** touches the cold path (Lance/Surreal). Everything else is SoA->SoA in-arena. Hot ops never reach durable store.
5. **Witness materialization at commit.** hot->hot witnesses stay pointers (same arena clock). hot->cold and cold->hot witnesses must be **copied/snapshotted** — a cold fact must never point into an arena about to epoch-reset. This is the single rule that keeps "no compaction, just epoch reset" from corrupting provenance.
6. **Epoch reset, not compaction.** SoA is 2–6KB; per-slot fragmentation is noise. Reclaim = drop the arena when the epoch retires. Tombstones are **minimal forwarding records** (generation counter + witness back-pointer), never payload.
7. **Two-clock decoupling everywhere.** Hot path at shader speed; commit/plan at cold-store speed. Coupling them backpropagates cold latency into the shader and stalls everything. The 64k–512k SoA range is the **shock-absorber buffer** between the clocks, not a thought-count target.
8. **Backpressure is correct, not a failure.** At sub-us production it's guaranteed. Bounded mailboxes (ractor defaults to unbounded -> OOM trap). Shed under pressure by **<f,c>**: prune low-c plan-state first, protect near-commit. Degrades toward the most-believed set.
9. **Candidate generation is plan-phase, before the kanban exists.** Closes the homunculus regress: deliberation is a *phase*, the kanban is its *precipitate*. Proposers are **bounded, non-recursive** (emit k candidates). Only Rubicon does EFE arbitration. Proposers dumb, arbiter smart.
10. **Two-tier free energy.** Real Friston/EFE only at the cold/commit tier (small N). Hot tier gets a scalar proxy (<f,c> x goal-alignment, one FMA), never a planning loop. Per-thought active inference at 512k x sub-us is impossible; don't pretend otherwise.
11. **WAL persists the substrate line ONLY.** No compiled candidates, no planning artifacts in the durable set. They're reconstructible from plan + AST. If JIT'd candidates leak into the WAL, the inversion becomes a WAL-format migration.

## AST is the hub (the unification)

- **One canonical AST.** Elixir surface syntax -> AST. OWL/DOLCE/OGIT/Odoo -> *same* AST (implicit-logic extraction). Both lower to SurrealQL **and** to planner candidates.
- **Build AST as hub, not translators.** Elixir = one parser in. SurrealQL = one codegen out. Ontology extractors = other parsers in. Never Elixir->SurrealQL directly.
- **Business logic is just one proposer's candidates.** A business rule, a mined association (Aerial+), an LLM conjecture, and an AST-walk step are the *same candidate object*, differing only by `discovery_origin`. There is no business-rules subsystem — there's an AstWalker proposer that reads OWL/Odoo.
- **A move/rule/inference = a guarded rewrite over SPO state.** Same AST node shape across all domains. This is the agnosticism claim.

## Odoo extraction boundary (honest coverage line)

Declarative strata lift cleanly to AST/triples: ORM **domains** (`[('field','op',val)]` — already SPO-shaped), **ir.rule** record rules, **@api.constrains / @api.depends** field-dependency graphs. Imperative Python **method bodies do NOT** — that's the "dynamic behavior not in the static definition" wall (cf. IST/BPMN paper's 5.81% failures). AstWalker harvests declarative strata as high-confidence Curated/Extracted candidates; flags method bodies as low-c, defer-to-runtime-trace. Don't try to static-AST-walk arbitrary Python into business logic — it won't generalize.

## Foundry relationship (resolves the recurring category error)

**Foundry is an interpreter over a live ontology. This stack is a compiler over a frozen one.** Every Foundry component maps by *semantics* but **inverts by binding time**: Workshop (runtime widget binding) vs A2UI (compile-time ontology->UI projection). "dynamic" / "low-code" / "LangGraph-like" all smuggle in Foundry's *runtime-interpretation* model — reject that import. LangGraph maps to the execution semantics correctly and to the binding time incorrectly: this stack is *compiled* LangGraph-over-triplets, not interpreted.

Semantic map (correct), binding inverts (the catch):
Ontology -> Ash-resource-shaped / SPO ; Action -> governed AST rewrite ; Function -> Rust semantic fn / ractor handler ; Workshop -> A2UI projection ; Automate -> ractor mailbox + PubSub. All real by meaning, all early-bound where Foundry is late-bound.

## discovery_origin (u8) — !! ISA WIDTH AT RISK

```
bits 0-1 : ProvenanceTier (4)  -- Curated/Extracted/ArmDiscovered/Ratified   [stable]
bits 2-4 : proposer id    (8)  -- AstWalker/PairStats/Aerial+/LLM/dIPC-A/dIPC-B/... [GROWS]
bits 5-7 : reserved       (3)
```
**Proposer-id at 3 bits caps at 8; already 6 named.** "Business logic is just another proposer" + dual-IPC dialectic both imply proposers proliferate. Widen proposer field (steal reserved -> 6 bits/64, or go u16) **before surrealkv WAL hardens the LE wire format.** Once it's in the byte grammar, widening = migration across every component. This is the RISC ISA-ossification trap, live, now.

## Open forks (load-bearing but deferrable — NOT yet decided)

- **F1 — UI binding time.** Does an ontology change update running UIs *without rebuild*? YES -> runtime templating (minijinja/interpreter), Workshop is a real mechanical model. NO -> compiler, askama right, Foundry = inspiration only. *Everything in the UI layer falls out of this one answer.* Default lean: NO (compile-time, to stay coherent with plan-time-everywhere).
- **F2 — SurrealDB integration.** Read Lance storage directly (heavy, fragile vs both release cadences) **or** federate via shared DataFusion catalog (Arrow TableProviders, tractable). Default lean: federate.
- **F3 — jinja+AST vs minijinja/JIT.** Downstream of F1. Detail, but load-bearing once F1 lands.
- **F4 — proposer-id width.** Decide before WAL hardens (see above). Not really optional.

## The bring-up test (the falsifiable slice)

**Chess into OWL.** Encode openings/methods/verbs as OWL/ttl (meaning in the *content*, never the substrate — no chess-special field, ever). Run proposers (Aerial+ as proposer-not-oracle) over it. See if GM-flavored *legal* candidates fall out of the same proposer->candidate->AST-rewrite->commit loop that will handle Odoo. Ground truth is a stockfish call away. Exercises every layer on a checkable board.

**Smallest possible first slice:** substrate WAL round-trip — write a SoA thought through surrealkv, commit with materialized witness, read back after a simulated schema bump. If the LE-contract + versioning survives that, the floor holds and the rest is licensed.

## Pin versions (single coupling point)
lance 6.0.1 / lancedb 0.29 / datafusion 53. SurrealDB versioning aligns here.

---
*Pin numbers and "default leans" are as-of-this-doc; update as forks resolve. Everything above the forks section is invariant — change it only with a reason you can state.*
