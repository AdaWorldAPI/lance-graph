# Baton/Collapse De-reification — spec + plan v1 (2026-06-02)

> **Authorized by jan** ("go ahead, write the specs and your plan, run the 5+3 council"),
> correcting a doctrine reification he flagged: *"the collapse gate emission deeply disturbs
> me … there is no collapse … baton was a figure of speech."* Branch `claude/jolly-cori-clnf9`.

## Verdict

The reification is **doctrinal, not structural.** The code is sound; ~2 doctrine sentences were
wrong. **Scope: 2 doc edits, ZERO code, ZERO machinery, ZERO deletions, NO new ticket.** The
corrected doctrine states **live vs scaffold** explicitly — it must not replace one overclaim with
another (the trap B1/B3 guarded against).

## The corrected model (honest — live vs scaffold)

- **SoA columns = owned dendrites** holding thinking atoms (`CausalEdge64` / `QualiaI4_16D` /
  `MetaWord`), in `MailboxSoA<N>`.
- **Fire→wire→witness (structural mechanism):** an interaction writes one `CausalEdge64` into its
  `EdgeColumn` (`EdgeColumn::set`); **that edge — and only that edge — is the witness.** What is
  not written is not traceable.
- **Scaffold today (NOT yet wired — stated as such):** the W-slot→`WitnessTable` *which-mailbox*
  provenance is contract-declared (`w_slot()`→0 in v1, zero non-test writers); the edge write is
  **test-exercised only — no production write path yet** (`MailboxSoA::set_edge` has zero callers;
  `persist_cycle`'s sole caller is under `#[cfg(test)]`); the driver→owner write seam awaits the
  ractor wire (`MailboxSoaOwner` has only the `FakeSoa` test impl). Matches board "witness_arc MISSING."
- **Owner-only write (STRUCTURALLY ENFORCED):** write capability is only `&mut self` on the owner;
  **no bypass `&mut`-columns accessor exists** → Rust proves no-alias / no-race at compile time
  (E-CE64-MB-4).
- **No collapse:** `emit()` = threshold scan→write→reset; "collapse" is a misnomer, not an op.
- **No baton-as-mechanism:** "baton" = retired folk name for the LE `(u16, CausalEdge64)` wire contract.
- **Bundle/superpose machinery is REAL, Markov-load-bearing — KEEP, untouched.** `apply_edges` does
  additive superpose (= the wave). **Markov-safety: CLEARED** (R5; all 3 brutal confirm `MergeMode`
  is untouched).

## DO NOW — 2 doctrine edits, zero code (APPLIED)

**EDIT 1 — `CLAUDE.md` P-1, the `2026-05-26` carrier-scoping blockquote** (`CLAUDE.md:24-40`, was
159 words) → replaced with ~155 words: kills "inter-mailbox state IS the Baton"; no
`CollapseGateEmission` token, no byte formula in scripture; slogan unbolded; framed **structural**
+ an explicit **"Scaffold today:"** label (B1's anti-"live"-overclaim fix; B3's anti-monument cuts).

**EDIT 2 — `.claude/plans/north-star-integration-v1.md` WD-5:**
- L27 (open framing): "emissions ride as `CollapseGateEmission` batons" → "emissions are
  `CollapseGateEmission` wire writes" (direct edit).
- L53 (inside the **RATIFIED** resolution table): **annotated, not rewritten** — original cell text
  retained, dated terminology-correction appended (append-only on a ratified cell, per B2).

## FILE — one ISSUE only (no TD-id minted)

- **ISSUE `MERGEMODE-BUNDLE-SPLIT-BRAIN`** (filed in `ISSUES.md`): enum doc (`collapse_gate.rs:24`)
  says "majority vote"; the only receiver `apply_edges` does additive superpose and ignores
  `merge_mode()`; no dispatcher consumes the tag. jan's "which is the truth" call. Markov is anchored
  on the additive-superpose algebra regardless.

## DEFERRED — noted here, NO ticket (B3: don't mint ceremony for a rename that may never run)

- **Rename de-reify (parked):** `CollapseGateEmission`→`WireWrite`, `push_baton`→`push_wire`,
  `baton_count`→`wire_count`, term "baton"→"wire". ~18 `.rs` `CollapseGateEmission` hits + ~24
  `push_baton`/`baton_count` hits + ~361 doc hits (re-verify before executing). **Carve-out hazard:**
  the planner `CollapseGate`/`CollapseOp` and ndarray's *own* local `CollapseGate` enum are SEPARATE
  same-named types — do NOT migrate them. No TD-id; this bullet is the whole record unless scheduled.

## NOT touched

`MergeMode` semantics + ordinals; the `emit`→`CausalEdge64` write path; planner
`CollapseGate`/`CollapseOp`; ndarray's local enum; `I-SUBSTRATE-MARKOV`. `EPIPHANIES.md`
E-BATON-1 / E-CE64-MB-4 left intact (append-only; already record the folk-term origin).

## Council outcome (5 research + 3 brutal, all SHIP-WITH-FIXES — fixes folded above)

R1 separability · R2 owner-only-write YES + Bundle split-brain · R3 witness = edge write +
provenance scaffold · R4 the 2-edit list · R5 Markov CLEARED. B1 correctness: killed the
"live"-as-running overclaim (the edge write is test-only today) → structural + "scaffold today".
B2 governance: real 159-word baseline (not 172), ratified-cell append, same-commit board hygiene.
B3 anti-monument: cut the `CollapseGateEmission` token + byte formula from doctrine, unbold the
slogan, no TD-id.
