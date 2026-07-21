# TAKEOVER ROADMAP — single-driver consolidation (2026-07-21)

> Operator directive (2026-07-21): **one session takes over everything** — the
> x265-x266 session's open arcs + this session's arcs. This doc is the
> consolidated, deduped, prioritized ledger. Authoritative source rows:
> `STATUS_BOARD.md` (D-ids), `ISSUES.md` (blockers), the active `.claude/plans/*`.
> Board hygiene: this roadmap is the driver's hand; per-D-id status stays on
> `STATUS_BOARD.md`, findings on `EPIPHANIES.md`.

## What just shipped (both sessions, merged — the baseline)

- **Recipe dispatch, organ-gated (converged):** `recipe_loci` (Door C, mine,
  #784) × `witness_fabric` (self-computing A9 loci, #783) × `dispatch_guard`
  (the two-axis Sudoku gate: single-pass binding × multipass standing wave,
  `GateOutcome::Escalate`). Doctrine locked: **±8 = reference horizon, not a
  bound; causality past it → absolute address (256×256 tile)** (`E-HORIZON-NOT-BOUND-1`,
  `E-SUDOKU-TISSUE-WEAVE-1`). This thread is COMPLETE from both sides.
- **Jina/DocuScope calibration (#785/#787):** palette256²/CAM-PQ ADC as the real
  `contract::distance` scalar reference; whole-work-is-one-tile confirmed on KJV.
- **dispatch_mode** (#782, the Cynefin pre-router), **D-TRI-6 ascent loop** (P3).

## Priority tiers (unblocked-first, value-weighted)

### P0 — KEYSTONE (unblocks the most; do first, carefully)

- **K1 · Batched mint (D-TRI-1 / D-V3-W2a).** ONE layout-gated mint landing:
  triangle-tenant classids + **chess classids** + Tasks-SoA task classid +
  BoardAggregates (10th ValueTenant @152, T1–T6). Operator-deferred 2026-07-10;
  the takeover un-defers it. **Gate: `v3-envelope-auditor` + field-isolation
  matrix MANDATORY** (RESERVE-DON'T-RECLAIM, fit, version-stability, slot-purity).
  **Unblocks:** D-AW-3 (awareness facets), D-TTV-1 (thinking tenants), the chess
  classids (D-TRI-4/5), D-CSW-3. This is the single highest-leverage item.

### P1 — HIGH-VALUE, UNBLOCKED NOW (drive these in parallel to K1's audit)

- **P1a · D-CSW-1 leg-2 — standing wave on REAL data.** Core claim GREEN on
  synthetic (auc_wave .997 vs single .878, reverse-control .000); leg-2 = real
  `temporal.rs`/Lance version stream + wild corpora (COCA + KJV both shipped).
  Grounds the just-shipped `witness_fabric::standing_wave_grounded` on real text.
  **Unblocks D-CSW-2, D-CSW-3.** Probe-shaped, low risk. → START HERE.
- **P1b · Chess teacher measurement (D-TRI-4/5).** `stockfish-rs` (in-scope) +
  `temporal.rs`: game = version-stream, Stockfish NNUE eval = non-drifting Truth.
  D-TRI-4 = thinking-transfer validity+reliability (jc battery); D-TRI-5 =
  emulation≠resonance on opponent move prediction. The measurement harness can
  START before K1's chess classids (uses shipped stockfish-rs + temporal).
- ~~**P1c · D-MTS-1 — Markov-as-stream parity probe.**~~ **STRUCK (operator
  2026-07-21, `E-NO-BUNDLE-STANDING-WAVE-1`).** There is no VSA ±5 braid to reach
  parity with — the bundle is a hallucination + a single-ownership violation. The
  standing wave over `temporal.rs` (`witness_fabric::standing_wave_grounded`,
  shipped) IS the substrate. Replaced by **P1c′ · TD-BUNDLE-RESIDUE-1** — the
  scoped excision of `vsa_bundle`/markov-bundle call sites in deepnsm, each →
  a single-owner temporal-stream read (tests-green-gated).
- **P1d · Brick 3 (D-V3-W4b) — L4 learning-loop end-to-end probe.** residue →
  owner-stamped lane → next-cycle template read. Closes the P4 self-reasoning
  carrier. My earlier flagged thrust; unblocked.

### P2 — UNBLOCKED, SECONDARY

- **D-TRI-2** 12-family vs 12-step reading agreement (jc over real shader cycles).
- **D-TRI-3** nail→hammer dispatch probe (object resonance → atom vs inverted).
- **D-TRI-6 tail** real-cycle settle-rung distribution + jc threshold calibration.
- **D-MTS-2** L4 palette256² shader fidelity certification (certification-officer).
- **D-MTS-3** hierarchical-4⁴ vs flat-256 codebook fidelity (ndarray/bgz17).
- **D-V3-W3b/c/d** ElixirTemplate→graph-flow adapter, Rig oracle, template catalogue.
- **D-DNV-2** deepnsm SpoTriple→CausalEdge64 2³ end-to-end (IN PR — land it).

### P2-GATED — behind K1 (the batched mint)

- **D-AW-3** A2–A7 awareness facets · **D-AW-4** redundant sibling lanes ·
  **D-AW-5** jc collapse gate · **D-TTV-1** thinking tenants → V3 · **D-CSW-3**
  full-width ladder vs CE64 cram. All need lanes minted + real data.

### P3 — CROSS-REPO / BLOCKED (coordinate or wait)

- **D-EPI-*** ruff SPO-harvest filing (predicate registry, C# golden, convergence
  gate); A2b Blocked (council), A7 Blocked (Q-A7). Different domain (ruff).
- **D-DNV-3** arm-discovery 2nd proposer — Blocked (`ARM-JIRAK-FLOOR`).
- **D-DNV-4** episodic-witness tenant + basin=family wake — Blocked (calcify
  chain is `todo!()`; needs own wave).
- **D-V3-W5a/b/c/e** consumer adoption (q2/fleet/ladybug/smb) · **W6b/c** legacy
  retirement — Blocked (corpus proof / P4 checkpoint).
- **ISSUES open:** `ISS-OGAR-GENETICS-MIRROR-PENDING`, `ISS-CONTRACT-APP-PREFIX-MIRROR`,
  `ISS-Q2-CASCADE3-NIBBLE-ANCESTRY`, `[E-MEMB-1]`, `[W-F9-X3]` disk-quota — cross-repo/infra.

## Execution order (this driver's plan)

1. **P1a (D-CSW-1 leg-2)** — start now; grounds the shipped convergence on real data.
2. ~~P1c (D-MTS-1)~~ → **P1c′ (TD-BUNDLE-RESIDUE-1)** — scoped excision of the
   bundle violation (deepnsm call sites → single-owner temporal-stream reads).
3. **P0 K1 (batched mint)** — with the envelope-auditor gate; unblocks the P2-GATED cluster + chess classids.
4. **P1b (chess teacher)** — once chess classids land (or start the harness on shipped stockfish-rs).
5. **P1d (Brick 3)** — L4 learning loop.
6. Sweep P2 as capacity allows; escalate P3 blockers to the operator.

Each item lands as its own PR (draft), board-hygiene in-commit, probe-first
(no synthesis without a measured gate). This doc updates as items move.
