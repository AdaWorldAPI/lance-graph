# Architecture Entropy Ledger

> **APPEND-ONLY** — same governance as `PR_ARC_INVENTORY.md` /
> `EPIPHANIES.md` / `TECH_DEBT.md` / `ISSUES.md`. New rows append
> below. The `Entropy` and `Plan-Status` columns are the only
> mutable per-row fields; structural claims (Region / Component /
> File / DupCount / Seam) are immutable history.
>
> **Companion to** `.claude/knowledge/soa-dto-fma-map.md`. The
> map describes the architecture. This ledger scores each
> component on **integration state**, **loose-end count**, and
> **duplicate potential** — the three dimensions the user named:
> "what SoA has which integration state and how many lose ends and
> unconnected vs to be refactored, dead ends, vs integration plan
> exists, active or abandoned or stalled".
>
> **Why this exists:** today the workspace has DTO spaghetti —
> 6-copy NARS, 4-copy ThinkingStyle, 3-copy VSA stacks, 2-copy
> SentenceCrystal, 2-copy GateDecision (different shapes). The
> map names them; this ledger scores them so the next session can
> sort by entropy and pick the highest-leverage fix without
> re-discovery.

---

## Scoring rubrics

**Integration state:**
- `Wired` — implementation present, consumers reading/writing it, tests cover it.
- `Stub` — types exist, body is `unimplemented!()`/regex/empty `Ok(())`/single mock.
- `Aspirational` — referenced in docs/plans but **zero source hits in `crates/`**.
- `Dead` — code exists but no consumer / workspace-excluded / superseded.

**Duplicate potential:**
- `None` — single canonical, no parallel.
- `Low` — one shared def + adapters.
- `Med` — 2 copies, no formal bridge.
- `High` — 3+ copies OR same-name-different-shape collision.
- `Spaghetti` — 4+ copies with subtly different semantics.

**Entropy (1-5):**
1. Clean: canonical, wired, doc + tests + plan agree.
2. Mostly clean: small drift (e.g. doc lag), no behavioural risk.
3. Partial: working but loose end (one missing wiring, one stale doc).
4. High: 2-3 unconnected duplicates OR major seam broken.
5. Spaghetti: 4+ duplicates / cross-crate name collision / dead pointer in plan.

**Plan-status:**
- `Shipped` — plan deliverable shipped (cite PR#).
- `Active` — plan v1 active per `INTEGRATION_PLANS.md`.
- `Stalled` — plan exists but D-id has been "In PR" without a merge for >14 days.
- `Abandoned` — plan superseded or explicitly retired.
- `Missing` — no plan in `INTEGRATION_PLANS.md` or `STATUS_BOARD.md`.

---

(See full rebased content in /tmp/ledger_main.md — too large to inline)
