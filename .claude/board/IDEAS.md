# Ideas Log — Open + Implemented + Integration (triple-entry, append-only)

> **Append-only ledger** for every architectural idea, speculative
> design, "what if we tried X" moment. Ideas accumulate here
> whether or not they're ready to ship. When one gets implemented,
> it moves from Open → Implemented → Integration (a row linking
> the idea to the plan entry that scheduled it + the PR that
> shipped it).
>
> **Purpose:** a speculation has nowhere else to live until it's
> scoped into a plan. This file is the speculation surface. Ideas
> die or graduate here; nothing is lost.

---

## Triple-entry discipline

Every idea moves through three ledger sections in this file:

1. **Open Ideas** — speculative; captured when proposed.
2. **Implemented Ideas** — idea became real; row appended with PR
   anchor + integration-plan D-id reference.
3. **Integration Plan Update Log** — the paired "what the plan
   changed when this idea landed" row, citing the specific
   `INTEGRATION_PLANS.md` version bump or `STATUS_BOARD.md` row
   flip triggered by the idea.

The row in Open is NEVER moved; its Status flips. The Implemented
row is a NEW append that cites the Open anchor. The Integration
row is a THIRD append that cites both.

This is **triple-entry bookkeeping** — three sections, same idea,
cross-linked. The cost is a bit more writing; the benefit is that
every shipped idea has an audit trail from speculation → code →
plan consequence.

---

## Rejected / Deferred

Ideas that don't graduate go into a fourth section:

4. **Rejected / Deferred Ideas** — with `**Rationale:**` and cross-
   ref to the original Open entry. The Open row's Status flips to
   `Rejected YYYY-MM-DD` or `Deferred to <when>`.

Deferred ideas can later reactivate — append a new Open entry
citing the Deferred one; Deferred row's Status flips to `Reactivated
YYYY-MM-DD <new-entry>`.

---

## Governance

- **Append-only.** Never delete a row.
- **Mutable fields:** `**Status:**` line, `**Rationale:**` line (if
  added later with more context).
- **`permissions.ask` on Edit** (same rule as other bookkeeping
  files). Write for appends stays unprompted.

## Cross-references

- `EPIPHANIES.md` — if an idea came from an epiphany, both entries
  cross-reference each other.
- `INTEGRATION_PLANS.md` — the plan version that incorporated the
  idea.
- `STATUS_BOARD.md` — the D-id status row that reflects the idea's
  shipping status.
- `PR_ARC_INVENTORY.md` — the PR that landed the code.
- `ISSUES.md` — if implementing an idea surfaced a bug, both rows
  link.

---

## Kanban Format (priority + scope on every entry)

Every idea carries:
- **Priority** — `P0` must-ship-this-phase / `P1` next-phase / `P2`
  eventual / `P3` speculative.
- **Scope** — which agent / deliverable / domain: `@<agent-name>`,
  `D<N>` (plan D-id), `domain:<grammar|codec|arigraph|infra|...>`.

Ticket tag on each entry: `[P2 @family-codec-smith D7 domain:grammar]`.
Agents filter by `@`-mention or domain to see what's theirs.

## Open Ideas

(Prepend new ideas here with today's date. Format:)

```
## YYYY-MM-DD — <short title>
**Status:** Open
**Priority:** P0 | P1 | P2 | P3
**Scope:** @<agent> D<N> domain:<tag>

<one paragraph: what the idea is, rough scope, why it matters>

Cross-ref: <epiphany entry / plan D-id / related knowledge doc>
```

---

## Implemented Ideas

(When an Open idea ships, APPEND here with same title + PR anchor.)

```
## YYYY-MM-DD — <same title as Open entry> (from YYYY-MM-DD)
**Status:** Implemented YYYY-MM-DD via PR #NNN
**Shipped as:** D<N> in integration plan v<K>
**PR:** #NNN (commit SHA)

<verbatim original Open paragraph>

Cross-ref: <same + PR link + plan D-id>
```

The original Open entry's Status flips to `Implemented YYYY-MM-DD`.

---

## Integration Plan Update Log

(When an idea triggers a plan change — version bump, D-id status
move, new deliverable — APPEND here. This is the third-entry row.)

```
## YYYY-MM-DD — Plan consequence of <idea title> (from YYYY-MM-DD)
**Trigger idea:** <idea title> (YYYY-MM-DD)
**Plan change:** <version bump / D-id flip / deliverable added>
**Plan entry:** `INTEGRATION_PLANS.md` v<K> entry or new v<K+1> entry
**Status board update:** <D-id> → <new Status>

<one paragraph: what the plan documented differently after this idea>
```

---

## Rejected / Deferred Ideas

(Ideas that don't graduate go here.)

```
## YYYY-MM-DD — <same title as Open entry> (from YYYY-MM-DD)
**Status:** Rejected YYYY-MM-DD  |  Deferred to <when / trigger>
**Rationale:** <short explanation>

<original Open paragraph>

Cross-ref: <original + any related>
```

---

## How to use this file

**When a new architectural idea surfaces** — prepend to **Open
Ideas** with today's date. One paragraph. If it needs more, create
a knowledge doc and link.

**When an Open idea ships** — APPEND to **Implemented Ideas**; flip
Open Status to `Implemented YYYY-MM-DD`. Then APPEND to
**Integration Plan Update Log** with the plan consequence.

**When an Open idea is rejected** — APPEND to **Rejected /
Deferred Ideas** with Rationale; flip Open Status.

**When a deferred idea reactivates** — prepend a NEW Open entry
citing the deferred one; flip the deferred entry's Status to
`Reactivated YYYY-MM-DD <new-entry>`.

Nothing is lost. Every idea has a trail from speculation to
disposition.

## 2026-04-19 — FP_WORDS = 256 (supersede the 160 plan)
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @truth-architect ndarray domain:vsa domain:codec

Current: `FP_WORDS = 157`  (10,048 bits, 5-word remainder on AVX-512)
Planned (H6 harvest): `FP_WORDS = 160`  (10,240 bits, SIMD-clean)
**Proposed: `FP_WORDS = 256`**  (16,384 bits, cache-line-perfect, matches `Container<[u64; 256]>`)

**Why 256 over 160:**

- LanceDB `FixedSizeList<UInt8, 2048>` = 2 KB per row = 16,384 bits already.
  Padding 157 → 256 in Container currently wastes 99 u64 per fingerprint (62%).
- Container primitive is already `[u64; 256]`; unifying `FP_WORDS` with it
  means zero padding, zero remainder loops at any SIMD level, cache-line
  alignment guaranteed (2 KB / 64 B = 32 cache lines, every level clean).
- VSA capacity: Plate's bound rises ~1.6× (bundled-items-per-fingerprint
  capacity ~1,500 → ~2,400 at error < 1%).
- No rebake of stored fingerprints needed — Container was already 256 wide.

**Cost:** ~30 LOC in `ndarray::hpc::vsa` constants + test updates;
docs shift "10k VSA" language → "16k VSA". Plate's capacity math re-tune.

**Supersedes:** TECH_DEBT entry "FP_WORDS = 157 (not 160); SIMD remainder
loops remain" — the 160 plan was the right direction, 256 is the correct
destination.

**Cross-ref:** `.claude/knowledge/cross-repo-harvest-2026-04-19.md` H6,
`.claude/board/TECH_DEBT.md` FP_WORDS entry. Container layout in
`lance-graph-contract::cam::Container`.
