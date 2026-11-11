# 2026-11-11 — Codegraph over real Charon stores: false unaffected for public symbols

**Status:** MEASURED (probe). `.claude/probes/2026-11-11-codegraph-recon-1/`. Board: EXO-229, EXO-230.

Charon → ruff_charon → `lance-graph-codegraph` ran on dt64, contract, quack and r2il (library targets,
9,138 positions). A sweep of `impact` + `callers` over every position took 1.2 s with 0 refusals.
`ClosureWidth(64)` and the 256-id tile are single-tile `Cube::closure` limits; Codegraph uses
`closure_wide` (`MAX_POSITIONS = 2^20 − 256`).

**Finding:** `impact` of a `PUBLIC` symbol whose callers sit in crates outside the store returns an
empty set, not UNKNOWN. On dt64, 261 of 370 public symbols do this (`BindStore::admit`, `method_ref`:
~140 external call sites). The `cg_query` example now prints `UNKNOWN external-callers-unobserved`;
the library is unchanged. OPEN (EXO-230): should `Graded.unknown` carry it.

rust-analyzer (`sag-ra-probe GRAPH=1`) misses calls inside macro arguments (it marks the macro region),
does not index `#[cfg(test)]` unit modules at all, and records 6,280 constructions as calls. Recall of
the curated Mermaid call edges, over observed files: Charon 61/87, RA 91/99. Most of the gap is
variant construction, which Charon does not model as a call.
