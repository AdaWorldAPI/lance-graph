# 2026-10-06 — D-CE64-TIME-0: `CausalEdge64` as the register that survives a cycle

## VERIFIED-IN-CODE — inventory (main @ 7a2b0ef1)

1. **Where cycle state survives.** `MailboxSoA` columns (`edges`, energy,
   plasticity counters, …) across `tick()`; writes are cycle-gated by
   `write_row(row, cycle, cell)` (`Accepted` / `Stale` / `Future`). Durable
   persistence is Lance versions; the Alpha overlay is discardable per cycle.
2. **`StreamDto`** (`thinking-engine/src/dto.rs`) is sensor input:
   `source`, `codebook_indices`, `timestamp`. It has no `now` field and carries
   no `CausalEdge64`. The nearest "current surviving register image" is a
   `MailboxSoA` `edges` row.
3. **Types carrying CE64 across cycles:** `MailboxSoA::edges` (`edge` /
   `set_edge` / `write_row`); `bindspace::EdgeColumn` (`Box<[u64]>`).
4. **Created** everywhere via `pack` / `pack_v2`; **revised** by
   `CausalEdge64::learn` and `forward` (`Revision` arm), and as u8 truths by
   `NarsTables::revise`; **copied** through `edge` / `set_edge`;
   **discarded:** every probe-local edge (`reasoning_band_probe`,
   `relational_certification_probe` write local edges, not rows).
5. **Transient fold outputs:** #1370 `EligibleRecipes` (transient by design),
   `SpogTenants` merges, `apply_edges` energy (reset by `consume_firing`).
6. **Folds that could feed F/C or EpistemicState5:** `learn` (F/C, plasticity),
   `NarsTables::revise` / `deduce`, `ontology_warrant::Quorum`,
   `causal_audit::SupportLedger` (distinct sources).
7. **Transition seam:** `edge(row)` → `learn` → `write_row(row, cycle, edge)`
   → `tick()` → `edge(row)`.
8. **Edge-as-message vs register:** `apply_edges` reads an incoming CE64's
   mantissa and confidence into energy and never revises the stored edge; the
   `edges` column is register-shaped.
9. **Replay:** yes, from the seed register, the folds in order, and the law
   generation (f32 revision is deterministic on one platform).
10. **Smallest real path:** one `MailboxSoA` row through the seam in 7.
    `CausalEdge64::learn` has no production caller, and no production code
    writes bits 59..63 from evidence.

## MEASURED

`ce64_cycle_survival_probe.rs`: 10 tests, 9 disable runs red (no write-back,
no `learn`, distinct-source condition removed, confidence bar removed,
downgrade removed, an irrelevant field leaking into revision, a ledger that
survives the cycle, no `set_populated`, no `n_rows` guard on the read). The
first ledger disable came back green because it recorded into a clone; redone
so the ledger really persists, it fails. After review (#1373) the register row
is declared (`set_populated`) and read through the production `MailboxSoaView`
lens, so the survival is of a live mailbox row, not of backing storage.
Trajectory: `f/c` 128/0 → 230/191 (code 3) → 230/213 (code 7) → 172/223
(code 3); eligibility `OBS`, `OBS`, `OBS|STRATIFY`, `OBS`.

The #1370 law moved unchanged to `examples/shared/affordance_law.rs`; its own
probe still passes 9 tests, and breaking the shared `measure` turns 2 of them
red.

## OPEN

`ISS-NO-EVIDENCE-WRITER-FOR-EPISTEMIC-STATE`: no production writer of bits
59..63 from evidence; cross-cycle source identity is not carried.
