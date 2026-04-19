# Technical Debt Log — Open + Paid (double-entry, append-only)

> **Append-only ledger** for knowingly-deferred work: TODOs, shortcuts,
> workarounds, unsafe assumptions, missing probes, hardcoded
> thresholds, stubs, and anything else we shipped with intentional
> debt. Debt moves Open → Paid by status-flip; rows are NEVER
> deleted.
>
> **Purpose:** separate from `ISSUES.md` (bugs) and `IDEAS.md`
> (speculation). Tech debt is **code that works but we know
> something better is owed**. This file is where we admit it.

---

## Double-entry discipline

Same pattern as `ISSUES.md`:

1. **Open Debt** — known shortcut or deferral, captured at shipping
   time.
2. **Paid Debt** — when the shortcut is replaced with the proper
   implementation, append here with PR anchor + Status flip on the
   original Open entry to `Paid YYYY-MM-DD`.

The Open entry stays in its section forever (chronology). The Paid
section accumulates retirements for audit.

---

## Governance

- **Append-only.** Never delete a row.
- **Mutable fields:** `**Status:**` and `**Payoff:**` lines only.
- **`permissions.ask` on Edit** (same rule as other bookkeeping
  files).

## Cross-references

- `ISSUES.md` — an issue may become tech debt when knowingly
  deferred rather than fixed.
- `IDEAS.md` — a rejected idea that was "the better way" often
  leaves behind a debt entry documenting the compromise shipped
  instead.
- `PR_ARC_INVENTORY.md` — which PR introduced the debt + which PR
  paid it.
- `STATUS_BOARD.md` — debt items that block deliverables are
  cross-referenced from the D-id row.
- `EPIPHANIES.md` — an epiphany often retroactively turns something
  into tech debt (the old approach is now known to be suboptimal).

---

## Kanban Format (priority + scope on every entry)

Every debt item carries:
- **Priority** — `P0` must-pay-before-next-phase / `P1` pay-soon /
  `P2` eventual / `P3` keep-tracked-but-low.
- **Scope** — which agent / deliverable / domain owes it:
  `@<agent-name>`, `D<N>`, `domain:<tag>`.

Ticket tag: `[P2 @truth-architect D10 domain:grammar]`. Same
filter discipline — agents pull their own debt by `@`-mention.

## Open Debt

(Seeded with known deferrals from recent PRs. New items PREPEND
with today's date.)

```
## YYYY-MM-DD — <short title>
**Status:** Open
**Priority:** P0 | P1 | P2 | P3
**Scope:** @<agent> D<N> domain:<tag>
**Introduced by:** PR #NNN
**Payoff estimate:** <rough LOC / time>

<one paragraph: what shortcut was taken, why, what the proper fix
looks like, any blocking dependencies>

Cross-ref: <file:line / deliverable D-id / epiphany entry>
```

### Seeded from PRs #204–#211

## 2026-04-19 — Contract `ContextChain::coherence_at` returns 0 for non-Binary16K variants
**Status:** Open
**Priority:** P2
**Scope:** @resonance-cartographer @container-architect D4 domain:grammar
**Introduced by:** PR #210
**Payoff estimate:** ~80 LOC + tests

D4 shipped with Hamming-based coherence on the `Binary16K` variant
only; other `CrystalFingerprint` variants (`Structured5x5`,
`Vsa10kI8`, `Vsa10kF32`) return 0 as a zero-dep fallback. Cosine
coherence on the f32 variants would unlock richer disambiguation
but requires adding a minimal linear-algebra shim without breaking
the zero-dep invariant of the contract.

Cross-ref: `crates/lance-graph-contract/src/grammar/context_chain.rs`.

## 2026-04-19 — CausalityFlow has 3/9 TEKAMOLO slots; modal/local/instrument + beneficiary/goal/source deferred
**Status:** Open
**Priority:** P1
**Scope:** @integration-lead @truth-architect D1 domain:grammar
**Introduced by:** PR #208 + #210 (deliberate deferral)
**Payoff estimate:** 6 new `Option<String>` fields in
`lance-graph-cognitive/src/grammar/causality.rs` + tests

Full thematic-role inventory needs 6 more slots on `CausalityFlow`.
Deferred per user decision; D2 ticket emission and D3 triangle
bridge map only the 3 existing slots for now. Phase 2 work is
consistent with 3/9; Phase 3 may benefit from the extension.

Cross-ref: `grammar-landscape.md` §3, `STATUS_BOARD.md` D1 row.

## 2026-04-19 — Named-Entity pre-pass (NER) is the biggest OSINT blocker; stubbed out
**Status:** Open
**Introduced by:** architectural choice (all PRs)
**Payoff estimate:** dedicated PR, ~800 LOC new crate / subsystem

COCA 4096 has zero coverage of proper nouns (Altman, Anthropic,
Riyadh). Every unknown entity falls through to hash-bucket
collisions in the SPO graph. Grammar work proceeds without NER;
OSINT pipeline is blocked on this.

Cross-ref: `grammar-tiered-routing.md` §C8,
`STATUS_BOARD.md` Research threads section.

## 2026-04-19 — FP_WORDS = 157 (not 160); SIMD remainder loops remain
**Status:** Open
**Introduced by:** architectural choice (ndarray::hpc::vsa)
**Payoff estimate:** coordinated ndarray + lance-graph change,
~30 LOC in ndarray + 0 in lance-graph if field naming is stable

160 u64 = 10,240 bits (SIMD-clean for AVX-512 / AVX2 / NEON), zero
remainder loop in every SIMD pass. Current 157 u64 has 5-word
scalar tail. Performance delta is measurable but not critical yet.

Cross-ref: `cross-repo-harvest-2026-04-19.md` H6.

## 2026-04-19 — Abduction-threshold for unbundle-to-graph is hand-picked (0.88)
**Status:** Open
**Introduced by:** #208 (inherits PR design)
**Payoff estimate:** empirical calibration run on real corpus +
~30 LOC threshold parameterization

NARS Abduction confidence threshold for promoting facts into the
triplet graph is hand-picked at 0.88. Miscalibration on a specific
corpus (e.g. Animal Farm) would compound errors. Calibration is
pending D10 validation harness.

Cross-ref: `integration-plan-grammar-crystal-arigraph.md` E8,
`STATUS_BOARD.md` D10 row.

---

## Paid Debt

(No debt paid at initial commit. When an Open entry is retired,
APPEND here with same title + PR anchor.)

```
## YYYY-MM-DD — <same title as Open entry> (from YYYY-MM-DD)
**Status:** Paid YYYY-MM-DD
**Payoff:** PR #NNN (commit SHA) — <one-line description>

<verbatim original Open paragraph>

Cross-ref: <same + PR link>
```

---

## How to use this file

**When shipping with a known shortcut** — prepend to **Open Debt**
with `**Status:** Open` + `**Introduced by:** PR #NNN` +
`**Payoff estimate:**`. One paragraph describing what's owed.

**When paying debt** — append to **Paid Debt** with the same title
+ date anchor + `**Status:** Paid YYYY-MM-DD` + `**Payoff:** PR
#NNN`. Flip the Open entry's Status to `Paid YYYY-MM-DD`.

**When debt becomes irrelevant** (e.g. the feature it blocked got
abandoned) — flip Open Status to `Moot YYYY-MM-DD`. Keep the row.

Nothing is lost. Every shortcut has a trail from introduction to
payoff (or abandonment).
