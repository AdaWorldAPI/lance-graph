# D-MBX-* Completion Map — the whole arc to "ONE SoA, end-to-end"

> Live tracker for unified-soa-convergence-v1 (`E-SOA-IS-THE-ONLY`). Reverse of the
> plan's bottom-up sequence: this is the **dependency-ordered path to completion**.
> Status mirrors STATUS_BOARD; this doc adds the critical-path + gating-OQ view.
> Maintained via `tee` (board-hygiene, newest facts win; supersede by re-`tee`).

## Where we are (2026-06-18 update; base 2026-05-30)
- **D-MBX-A1** — migrated thoughtspace columns on `MailboxSoA<N>` — **SHIPPED** (between #418/#433).
- **D-MBX-A2** — close BindSpace expressivity gaps — **SHIPPED (carrier), 2026-06-18 reconcile.** Columns landed post-2026-06-13: W1 `22f5120a` (temporal/expert/sigma) + W1b `707360dc` (dense content/topic/angle) + W1c + W4a `BackingStore`/`BackingStoreWrite` shim, with accessors + parity + field-isolation tests. OQ-1 resolved (dense planes hot). S/P/O role slices = NON-GAP (VSA-unbind vs `grammar/role_keys`). Residual: OQ-2 fold decision (landed standalone). The §6 S1 gate (`D-CE64-MB-1-impl` par-tile apex / `PR-NDARRAY-MIRI-COMPLETE`) was **sidestepped** (enum-over-trait shim, OQ-C), not satisfied. See `E-DMBXA2-SHIPPED-RECONCILE`.
- **D-MBX-A6-P1** — contract seam (`kanban`/`soa_view`/`StepDomain::Kanban` + `class_id` N1 hook) — **SHIPPED #437**.
- **D-MBX-A6-P2** — Rubicon lifecycle DAG enforcement + `ExecTarget` — **IN PR #439**.
- Everything else below: **Queued**, gated as shown.

## The completion DAG (what blocks what)
```text
 SHIPPED ───────────────────────────────────────────────────────────────────
  A1 (SoA columns)        A6-P1 (#437 contract seam)    A6-P2 (#439 lifecycle+ExecTarget)
        │                        │                              │
 FOUNDATIONS (should land EARLY — everything downstream needs them) ─────────
  D-MBX-10  SoA version byte (MailboxSoAHeader; I-LEGACY-API-FEATURE-GATED)   [gates OQ-11.5]
  D-MBX-11  lance bump DONE via #445 (=6.0.0 -> =7.0.0, NOT =6.0.1)          [SUPERSEDED]
        │
 HOT-PATH SoA EXPRESSIVITY ──────────────────────────────────────────────────
  D-MBX-A2  close BindSpace gaps (content_ref, S/P/O role slices, fold)  [gates OQ-1/OQ-2]
        │
  D-MBX-A3  witness_arc:[u32;W] per-row column (R4 belief-state arc handle) [gates OQ-11.2]
        │           │
  D-MBX-A4  Staunen×Wisdom plasticity spreader (Planning-gated, Hebbian)  [gates OQ-11.1 + `phase` field]
  D-MBX-A5  SPO-W witness dual-residency; SoA decides commit modality     [gates D-MBX-4]
        │
 CONSUMER WIRING (the A6 spine -> real cycle) ────────────────────────────────
  D-MBX-A6-P3  impl MailboxSoaOwner for MailboxSoA<N> (ractor owns; try_advance_phase
               in the real cycle) + planner candidate-gen emits KanbanMove{exec}   <-- NEXT
        │
  D-MBX-8   Σ10 commit stamps t=-550ms (Libet) in SigmaTierRouter -> ractor START
        │
 SIMD + COLD ALIGNMENT ───────────────────────────────────────────────────────
  D-MBX-7   lance-graph containers ≡ MailboxSoA ≡ ndarray::simd_soa (1.4-4.2x;
            HARD PREREQ for the SurrealDB transparent view)  [gates D-MBX-A2 + 10 + 11 + ndarray-miri]
        │
 SUBSTRATE VIEW (the payoff — now a BIDIRECTIONAL SUBSCRIPTION, not a build) ─────
  D-MBX-9   Rubicon kanban = the mailbox Lance version arc, BOTH directions (substrate-free):
            OUT  mailbox advance_phase commit = version = kanban move  (E-VERSION-ARC-IS-THE-KANBAN)
            IN   surreal LIVE/scheduled event over versions() = planner->execution scheduler
                 firing the next advance_phase  (E-SUBSTRATE-IS-THE-SCHEDULER)
            => collapses from "build a view" to "LIVE-subscribe + schedule" (like GitHub CI/PR sub).
            surreal #31 Timeline over Dataset::versions() IS the surface; MailboxSoaView (#437) = read lens.
            [still gated by surreal_container BLOCKED(B/C/D) OQ-11.6 for the surreal side; design substrate-free]
        │
 WORKSPACE CONVERGENCE (the "nine consumers" all read ONE SoA) ─────────────────
  D-MBX-12  8-PR alignment, sequenced OQ-11.8:
            12.4 lance-graph -> 12.5 planner -> 12.6 shader-driver -> 12.7 callcenter
            -> 12.1 AriGraph -> 12.9 thinking-styles -> 12.2 Vsa16k audit -> 12.8 ontology audit
```

## Critical path to "done" (longest chain)
`A6-P2(#439)` → **A6-P3** → ~~A2~~ (SHIPPED carrier 2026-06-18) → A3 → A5 → D-MBX-7 → **D-MBX-9** (transparent view) → D-MBX-12 (nine-consumer convergence). D-MBX-10/11 are off-critical-path foundations that should land in parallel NOW (11 is mechanical, 10 is the version-gate everything reads). **Note (2026-06-18):** the engine_bridge per-row re-home (plan §6 "S2") is ~80% pre-absorbed by the W4a `BackingStore`/`BackingStoreWrite` shim; its residual (re-point the `#[cfg(with-engine)]` lab callers through the shim) folds into S3 (driver flips off the singleton), which is the true next substantive node and is gated on OQ-2 + the pre-existing `--features with-engine` compile break (`QUALIA_DIMS` unimported, engine_bridge.rs:259).

## Gating Open Questions (these block, not the code)
- **OQ-11.6** — surreal_container fork coords (URL/branch/kv-lance flag); BLOCKED(B/C/D). **Blocks D-MBX-9** (the whole payoff). Highest-leverage unblock.
- **OQ-1 / OQ-2** — content-ref shape + temporal/expert fold. Block **D-MBX-A2** (hot-path expressivity).
- **OQ-11.1** — plasticity spread radius/decay. Blocks **D-MBX-A4**.
- **OQ-11.2** — witness-arc width W + handle encoding. Blocks **D-MBX-A3**.
- **OQ-11.5** — version-byte scheme. Blocks **D-MBX-10**.
- **OQ-11.7** — planner 5-phase cutover feature-gating. Blocks **D-MBX-A6** full (P3+).
- **OQ-11.8** — the 8-PR sequence (resolved order above). Sequences **D-MBX-12**.

## Loose ends folded in (from EPIPHANIES LE-1..4; not D-MBX-numbered yet)
- **LE-1** EW64 as DeepNSM>Markov>grammar coref witness pointer (no bundling) — relates A3/A5 + the EW64 type.
- **LE-2** unify cold SPO + cold AriGraph (EW64 = cheap witness pointer) — relates 12.1 + A5.
- **LE-3** mailbox-cycle-end Rubicon commit → cold SPO-W + SLA/goalstate — hooks `is_absorbing` (A6-P2 ✓) → D-MBX-A5 + D-MBX-9.
- **LE-4** Odoo/OWL business-logic action substrate — explicitly OTHER SESSION.
- **EW64 type** (AriGraph episodic edge; shares CE64 low-40 SPO bits; payload = witness-arc pointer + aerial-fed prefetch confidence) — a contract type that realizes A3's arc handle; candidate next-after-P3.

## Recommended execution order (autonomous, this session)
1. **D-MBX-11** (mechanical lance bump) — unblocks the stack alignment, near-zero risk.
2. **D-MBX-A6-P3** (consumer wiring) — turns the shipped contract spine into a live cycle; highest architectural value, on critical path.
3. **D-MBX-10** (version byte) — the foundation every cold/view consumer reads; pairs with OQ-11.5.
4. Then A2→A3 (hot-path expressivity + witness arc), surfacing OQ-1/OQ-2/OQ-11.2 for ratification as reached.
5. **D-MBX-9** stays BLOCKED on OQ-11.6 (surreal fork) — flag for user unblock; the `MailboxSoaView` borrow trait already lets it land with zero contract change once unblocked.

## 2026-07-17 update — D-MBX-A6-P3b (output-overhaul carrier) SHIPPED
- **D-MBX-A6-P3b** — `StrategyOutcome { reliability, intended_move: Option<KanbanMove> }` carrier landed on `PlanInput.outcome` (planner-internal, additive). `StyleStrategy` retires the dead-store `_reliability` and SURFACES reliability + a **bootstrap** intended move (`Planning→CognitiveWork`, owner 0 / cycle 0, warden-BOOTSTRAP-OK) — no mutation, no commit, no advance. 216 lib + 4 probe tests green. **UNBLOCKED** (no mint, not the OQ-11.7 cutover). Next nodes still open: thread the outcome OUT of `compose_and_execute` (`compose_and_collect()` sibling), promote `StrategyOutcome` to `contract::kanban` on first cross-bridge read, and the owner-consume/advance edge (OQ-11.7). See EPIPHANIES `E-STRATEGY-OUTCOME-CARRIER-1`.
