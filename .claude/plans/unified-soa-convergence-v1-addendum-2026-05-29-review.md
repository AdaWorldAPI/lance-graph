# unified-soa-convergence-v1 — review addendum (post-merge, 2026-05-29)

> **Status:** REVIEW NOTES — review of the merged plan after PR #434 landed on `main` (`1186dfd3`).
> **Reviewed:** 2026-05-29, session `claude/lance-graph-ontology-review-Pyry3`.
> **Scope:** does not touch the §1 / §11 user-stated rulings (`E-SOA-IS-THE-ONLY` + refinements). Adds clarifications, fills one stack-table gap, records a verification result, and flags one cross-doc drift that belongs to a separate PR.
> **Anchor file:** `.claude/plans/unified-soa-convergence-v1.md` (685 lines, shipped PR #434).

---

## 1. Status flip (the one update that propagates back to the plan itself)

§9 P0 ("design ratification") and §15 "PRs (recent context)" both refer to the plan as `in PR (this one)` / `PR (this one)`. The plan merged in **PR #434** on 2026-05-29 (`1186dfd3` merge commit). The in-place plan edits in this branch flip those two references.

The five-ruling content in §1 / §11 is **untouched** (author-stated, council-bypassed per §16; correcting that would violate the council-bypass discipline).

---

## 2. §3.2 per-row total — clarification (no semantic change)

§3.2 lists per-row column types totalling `4 + 1 + 4 + 8 + 8 + 4 + 2 + 6 + 4·W = 37 + 4·W` bytes, then narrates "Per-row hot total: ~30 B (bare migrated SoA columns) + ~6 KB Hamming identity planes ... = ~6 KB/thought."

Both numbers are right but they answer **different questions**:

- The `~30 B` count includes only the **D-MBX-A1-shipped** columns (`energy 4 + plasticity 1 + last_emission 4 + edges 8 + qualia 8 + meta 4 + entity 2 = 31 B`) — i.e. *the columns that exist on `mailbox_soa.rs` today*. The shipped subset.
- The `~6 KB/thought` figure includes the per-row footprint **plus** the 3 × `[u64; 256]` identity planes (3 × 2048 B = 6144 B per **thought**, not per row).
- After D-MBX-A2 / A3 land, the per-row bare total grows by `content_ref(6) + witness_arc(4·W)`; at the default `W = 16` (OQ-11.2) that's `+70 B`, taking the per-row bare total to **≈101 B**. The hot per-thought ceiling math (64k–256k thoughts at 300–600 MB / 1.2–2.4 GB) is **still dominated by the identity planes** (6 KB ≫ 100 B per row), so the §3.2 ceiling numbers do not change materially even after the Queued columns land.

**Recommendation:** when §3.2 is next revised (a separate v2 PR, not this addendum), split the per-row table into "shipped today" vs "shipped + queued" columns to remove the ambiguity. The ceiling math stays correct either way.

---

## 3. §4.2 stack alignment — surrealdb fork row (the one true gap)

The §4.2 stack-pin table covers `arrow / datafusion / lance / lancedb / ndarray` but is silent on **`surrealdb`** — even though §4.3 / D-MBX-9 / OQ-11.6 all hinge on a SurrealDB fork pin (with the `kv-lance` backend feature) to materialize the Rubicon kanban as a transparent view over LanceDB. The pin is OQ-11.6 prose only; a future reader skimming §4.2 will conclude the stack alignment is already complete when in fact one pin is still unknown.

**Addendum row** (proposed for §4.2 in a future v2):

| Layer | Current | Target | Delta |
|---|---|---|---|
| `surrealdb` (fork) | BLOCKED — fork URL + branch unknown | TBD; must carry the `kv-lance` backend feature flag | BLOCKED — see OQ-11.6 |

This makes the "one transparent container view" prereq visible at the same altitude as the other stack pins, instead of being buried in §4.3.

---

## 4. §4.2 verification — independently re-checked 2026-05-29

I re-checked the §4.2 claims against the workspace `Cargo.toml`s in this branch (`claude/lance-graph-ontology-review-Pyry3`, HEAD `1186dfd3`):

| Pin claim | File:line | Verified | Notes |
|---|---|---|---|
| `arrow = "58"` | `crates/lance-graph/Cargo.toml:16` | ✓ | also at callcenter:29, ontology:49, holograph:29 |
| `datafusion = "53"` | `crates/lance-graph/Cargo.toml:21` | ✓ | also at callcenter:37, holograph:35 |
| `lance = "=6.0.0"` | `crates/lance-graph/Cargo.toml:38` | ✓ | also at benches:10, callcenter:30, ontology:46, holograph:38 — **5 files, matches §4.2 D-MBX-11 scope** |
| `lancedb = "=0.29.0"` | `crates/lance-graph/Cargo.toml:41` | ✓ | only declared in `lance-graph` |

**D-MBX-11 readiness:** confirmed mechanical 5-file edit (`=6.0.0` → `=6.0.1`). When cargo prohibition lifts, one `cargo check` per crate gates the bump.

---

## 5. CLAUDE.md "Key Dependencies" drift — flagged, NOT fixed by this addendum

The lance-graph `CLAUDE.md` "Key Dependencies" block (under `## Key Dependencies`) still lists:

```
arrow = "57"
datafusion = "51"
lance = "2"
```

The workspace pins shown in §4 above are **arrow 58 / datafusion 53 / lance =6.0.0**. The CLAUDE.md block is months stale (it dates to the 2026-04-21 categorical-algebraic-inference update).

**Why this addendum does not fix it:** CLAUDE.md is workspace-wide doctrine that drives every session boot via the SessionStart hook. A drive-by edit hidden inside an unrelated PR is the wrong altitude. The right move is a focused 1-PR drift-fix that updates the Key Dependencies block + the Cross-Repo Dependencies table + verifies no other CLAUDE.md section quotes stale pins.

**Tracking:** add to `.claude/board/TECH_DEBT.md` under a new `TD-CLAUDE-MD-DEPS-DRIFT` entry (separate change).

---

## 6. Cross-check vs `.claude/board/STATUS_BOARD.md`

STATUS_BOARD.md § unified-soa lines 549-566 already lists D-MBX-A1 (Shipped) + A2/A3/A4/A5/A6/7/8/9/10/11/12 (Queued). The dependencies / gates listed in those rows are consistent with §10 of the plan. The post-merge action is the status flip on the section's own header line — done in this PR.

---

## 7. Cross-check vs `EPIPHANIES.md`

`E-SOA-IS-THE-ONLY` (eb5c4a5, 2026-05-29) carries the five rulings verbatim, and the §11.3 / §11.4 / §11.6 refinements are integrated into the same epiphany block (per the plan's §0 anchoring). Council bypass is correctly documented (§16). **No epiphany-side correction is needed.**

---

## 8. Cross-check vs `PR_ARC_INVENTORY.md`

PR_ARC's 2026-05-29 unified-soa entry was written *pre-merge* and still records `**Status:** PROPOSAL`. Per the APPEND-ONLY rule (only Status / Confidence lines are updatable), the post-merge flip lands on the existing entry. Done in this PR. A new PR_ARC entry is also PREPENDED for #434 itself (the merge event).

---

## 9. What this addendum does NOT do

- Does **not** edit the rulings text in §1 / §11 (council-bypass discipline).
- Does **not** touch §10 deliverable specs (those are the work product, edited only by their own implementation PRs).
- Does **not** patch CLAUDE.md (out of altitude — separate focused PR).
- Does **not** invoke `cargo` (session-stability prohibition still active).

---

## 10. Summary of in-place edits accompanying this addendum

| File | Change | Rule cited |
|---|---|---|
| `.claude/plans/unified-soa-convergence-v1.md` §0 / §9 / §15 | "in PR (this one)" → "SHIPPED in PR #434 (merged 2026-05-29)" — 3 spots | Plans are append-only; status references are updatable |
| `.claude/board/INTEGRATION_PLANS.md` (2026-05-29 unified-soa entry) | `**Status:** PROPOSAL` → `**Status:** SHIPPED (PR #434, merged 2026-05-29)` | CLAUDE.md "Status / Confidence lines are updatable" |
| `.claude/board/STATUS_BOARD.md` (unified-soa section) | Add `> **Status:** SHIPPED (PR #434, merged 2026-05-29).` line under the section header | same |
| `.claude/board/PR_ARC_INVENTORY.md` | PREPEND a new entry for the #434 merge with Added / Locked / Deferred / Docs / Confidence | rule 1: new PRs PREPEND |
| `.claude/board/LATEST_STATE.md` | Update the "Last updated" header line to lead with #434 | post-merge gov rule |
| `.claude/board/TECH_DEBT.md` | PREPEND `TD-CLAUDE-MD-DEPS-DRIFT` | flag for follow-up PR |

---

_End of addendum._
