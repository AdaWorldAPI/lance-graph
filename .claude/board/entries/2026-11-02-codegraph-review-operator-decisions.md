# Codegraph review: operator decisions and P1 result (2026-11-02)

**Status:** P1 DONE (in PR #1923, not merged); P2, P3 DECIDED, prepared as plans.
**Source:** [architecture review](2026-11-02-codegraph-dr3-dt64-ruff-architecture-review.md) §7.

## Principle (DECISION, operator 2026-11-02)
One canonical normalised codegraph over DR³; one method authority, DT64.
Ruff/Charon, Rust/MIR, MLIR and other frontends may keep versioned IR
artifacts and caches, never a second independent semantic store. Lazy
hydration only outside the hot path. MCP, `.index.md`, Mermaid and external
call queries are later deterministic projections of the same graph, never a
new source of truth. No automatic merge.

## P1 — atomic refresh (DONE, measured)
- `refresh_crate` is prepare → validate → commit
  (`lance-graph-codegraph/src/graph.rs`). Staged: interner (`SymbolStage`,
  `artifact.rs`), positions incl. `last_owner` (`PosStage`), the grown tile
  map (not handed to the table), the edge batch (`Cube::stage`, `&self`), the
  closure refresh (`Cube::stage_closure`). Commit: `Symbols::absorb`,
  `Positions::absorb`, `MapTable::advance` + `seal`, `Cube::commit`, closure
  cells, lanes, masks; no error path. `Cube::rebase` also checks before it
  changes anything.
- Codex P2 on #1923 closed: `Node::Bind` keeps its global id; an extending
  rebase or commit places a bind that was unmapped. A mapped bind keeps its
  coordinate (positions never renumber), so a view keeps its meaning.
- Falsifiers: NeedsReopen, injected stage refusal, injected closure refusal
  each leave a full-state snapshot unchanged (interner, positions,
  `last_owner`, lanes, digests, statuses, table version/revisions/tile
  history, cube frame and every log, masks, cached closures); no version is
  issued; retry == the refresh that never failed, and `m::b` keeps 4 call
  sites as in a clean open; the snapshot can fire. xlog: bind placed by
  rebase and by commit; mapped bind unchanged; staging issues no version and
  refuses a non-extending map.
- 7 disable runs, each red on its own test only: absorb before the
  reservation check; advance before staging; absorb before the closure
  block; dedup keys before the multiset diff; drop rebind in `rebase`; drop
  rebind in `commit`; drop the `extends` check in `stage`.
- CI: `rust-test.yml` now runs `crates/lance-graph-codegraph` (workspace
  excluded; ran in no workflow before).

## P2 — header plus per-row FunDeclId (DECIDED, plan amended)
`d-codegraph-0-binary-lanes-v1.md` §8: PROV section (charon version and
commit, LLBC schema version, LLBC digest, toolchain, target + build config,
source revision; UNKNOWN never equal to a value), FUNDECL + ARTREF lanes
(identity = `(artifact, FunDeclId)`), typed objects, `INVOKES_WORD` to a
registered DT64 reference (refused when undeclared), F9–F12.
OPEN: where "function realises word W" comes from (lens-declared table or a
Charon-read attribute; never a name match).

## P3 — DT64 the method authority (DECIDED, plan written)
`d-dt64-method-authority-census-v1.md`: adapters may stay as listed `LAB`
exceptions with a removal path; production words must be declared; the
census fails on an undeclared word and on a stale exception. egglog pinned
to `5deaa5637a306c48c998f88d5d08792d2dbfd037` in `egglog-r2il-probe`
(was `branch = "main"`); its 7 suites pass at the pin.
