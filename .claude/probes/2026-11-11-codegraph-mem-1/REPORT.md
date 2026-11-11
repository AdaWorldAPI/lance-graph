# D-CODEGRAPH-MEM-1 — grep-native Codegraph memory (reduced slice)

**Status:** reduced Cargo / syntactic / curated slice, as EXO-230 gate 3 allows only by explicit
approval. Draft; awaiting that approval. Board: EXO-230 (navigation), EXO-229 (Charon intake).

## 1. The original plan

The original ~15 grep/sed/head/tail commands were **not recovered**. Searched without a hit:
- every checked-out repo's working tree;
- lance-graph history on all branches, including deleted files and the #1921 branch;
- Linear issues and documents;
- local session transcripts.

The closest committed sources:
- `.claude/board/entries/2026-11-02-codegraph-dr3-dt64-ruff-architecture-review.md` §5.3–5.6
  (`.index.md` + Mermaid graded MISSING, one index per crate, Mermaid at crate/module level);
- the operator's three design references of 2026-11-11 (grep-native `.codegraph/` of line-oriented
  facts; SQL/Gremlin only as later front ends over Codegraph; no SQLite, a Python bridge plus
  generated Mermaid).

The recipes in `.codegraph/index.md` are therefore labelled **NEW / DERIVED 2026-11-11**.

## 2. Crosswalk

| Existing capability | Location | Reusable as-is | Gap |
|---|---|---|---|
| callers / callees / references / impact, graded | `crates/lance-graph-codegraph` (#1921) | yes | facts only for 4 fixtures; no Charon artifact for any real crate |
| Charon rows reader (`#schema`, byte spans, `files.tsv` fnv64) | `codegraph/src/artifact.rs` | format reused by the generator | — |
| crate dependencies | Cargo manifests | yes | Codegraph has no crate-dependency fact |
| imports | not in Codegraph (LLBC has no `use`) | — | syntactic only |
| generator convention (`--write` / `--check` / `--self-test`) | `entries_index.py`, `supersession_index.py` | yes | — |
| curated diagrams + legend | `.mermaid/` (#1997, merged) | yes, classDefs reused | not generated |
| decisions (ADMISSION B, D4, G2) | board entries, EXO issues | yes | not machine-readable before this slice |

## 3. What landed

- `.claude/tools/codegraph_index.py`: stdlib only, deterministic; `--write`, `--check`, `--self-test`.
- `.codegraph/` (108 generated files, ~4.9 MB):
  - `crates.tsv`, `dependencies.tsv`, `reverse_dependencies.tsv`, `symbols.tsv`, `imports.tsv`,
    `calls.tsv`, `decisions.tsv`, `gaps.tsv`, `freshness.tsv`, `coverage.tsv`;
  - 90 `crates/<crate>.index.md`;
  - 7 `views/<symbol>.md`;
  - `index.md`.
- `.codegraph/architecture.tsv`: the only hand-maintained input. It holds reviewed architectural
  relationships, each with `path:line` and an anchor token that `--check` re-validates.
- Two categories, kept apart:
  - **Extracted:** Cargo edges, declarations, `use` leaves with cfg, Charon calls.
  - **Architectural:** `decisions.tsv`, from `architecture.tsv` only, never inferred from imports.
- `.github/workflows/codegraph-index.yml`: runs self-test plus `--check` on PRs touching the
  generator or `.codegraph/`.

Every crate's calls read `CALL_GRAPH=ARTIFACT_MISSING`, because no Charon store exists for real
crates. The Charon path is exercised against the Codegraph fixture
(`cg_app::run → cg_lib::area_of`, CONFIRMED, correct line).

## 4. Negative tests (all in `--self-test`)

| Required negative | Test | Disable run |
|---|---|---|
| stale index cannot claim current code | 01, plus real-repo probe (appended line to `dt64/src/lane.rs` → `STALE crate lance-graph-dt64`) | red |
| conditional import stays conditional (and Cargo `target` cfg kept) | 02 | red (both halves) |
| possible dynamic call never CONFIRMED | 03 | red |
| plan never becomes shipped | 04 | red |
| same-name symbols not conflated | 05 | red |
| byte-identical output | 06 (two runs; member order reversed) | not disable-run |
| missing Charon artifact reported, not empty | 07 | red (row and `CALL_GRAPH` line) |
| Charon ingestion + fnv64 STALE | 08 | red (hash and line offset) |
| use-tree expansion | 09 | not disable-run |
| curated anchor rot detected | 10 | red |

The rebase onto main (#1996 merged) made `--check` report two views stale before regeneration, the
guard firing on real drift.

## 5. Measurement (10 fresh agents, same model, same questions)

Arm A explored source only. Arm B searched `.codegraph/` first, then verified in source. Grades
come from an answer key built from source before the runs.

`tokens` is each agent's total context consumed. All runs share ~425k of base context
(system prompt and CLAUDE.md files), so compare the differences, not the totals.

| Q | Arm | Correct | Tokens | Tool calls | Time | Files opened |
|---|---|---|---|---|---|---|
| 1 ClassView → DT64 → LaneBinding → Quack | A | correct, more detail (test-only `AbiLanes` ctor, reorder gap) | 442,249 | 10 | 39.1 s | 6 |
| 1 | B | correct (same hops, lab-only Quack hop, gaps) | 437,363 | 6 | 27.6 s | 12 |
| 2 consumers of WideFieldMask | A | 17/17 | 425,940 | 2 | 10.4 s | 0 |
| 2 | B | 17/17 (grepped src, not the index) | 425,857 | 2 | 10.4 s | 1 |
| 3 effects of changing `LaneBinding::check` | A | complete | 436,031 | 5 | 18.9 s | 1 |
| 3 | B | **partial**: missed direct calls in `dt64/tests/lane_socket.rs` and `shape.rs` lines; two lines off by 7 | 435,023 | 5 | 20.0 s | 5 |
| 4 Quack → Lance allow/block, DR→Lance handoff | A | correct, more breadth (ALLOW/BLOCK gates) | 447,044 | 13 | 35.2 s | 4 |
| 4 | B | correct (#1950 probe, BLOCKED_D4, S-INGRESS) | 438,741 | 9 | 37.2 s | 6 |
| 5 ADMISSION B decision and refusals | A | correct + extra (ISSUES row, attestation gap) | 434,853 | 5 | 18.1 s | 2 |
| 5 | B | correct | 430,268 | 6 | 16.2 s | 5 |

**Reading:**
- The index saved tokens and tool calls on the architecture questions (Q1, Q4, Q5: −4.6k to −8.3k
  tokens, up to −4 calls).
- It made no difference for a single-symbol census (Q2).
- It did **worse** on the call-impact question (Q3). Calls are `ARTIFACT_MISSING`, and the index
  arm leaned on it instead of grepping exhaustively.
- Source-only answers were broader on Q1, Q4 and Q5.
- n = 1 per cell: this is an indication, not a result. Accuracy did not improve; it fell on Q3.

## 6. Gap report

| Gap | Owner |
|---|---|
| No Charon/ruff artifact for any real crate, so callers, callees and impact stay `ARTIFACT_MISSING`; Q3 shows the cost | EXO-229 gate A (Charon → ruff_charon → Codegraph intake) |
| Codegraph has no crate-dependency or import facts; the generator uses Cargo and syntax with provenance | EXO-229 (only if a schema extension is justified) |
| Imports are unresolved (re-export chains, globs → POSSIBLE); macro-generated items and function-local `use` invisible; tests, examples and benches not scanned | this tool, next slice |
| Whether `--check` should gate every PR touching `crates/` (today it gates only the generator and `.codegraph/`) | EXO-230 decision |
| A2UI / Askama class-bound projections: no reviewed in-repo evidence, so no row | a2ui-rs owner |
| Architectural rows (24) are hand-curated; coverage is what was reviewed, not complete | EXO-230 |
| ADMISSION B call-site classification has no owner | EXO-140 |

## 7. Suggested CLAUDE.md addition (not applied)

> **Repository memory.** Before broad source exploration, search `.codegraph/` (`rg -n NAME
> .codegraph/`) for established dependencies, declarations and reviewed architectural facts. Run
> `python3 .claude/tools/codegraph_index.py --check` first: a STALE crate means its facts may be
> out of date. The index is a projection; open the cited `path:line` before relying on a fact or
> editing. `CALL_GRAPH=ARTIFACT_MISSING` means calls are unknown, not absent; for call impact,
> grep the source exhaustively.
