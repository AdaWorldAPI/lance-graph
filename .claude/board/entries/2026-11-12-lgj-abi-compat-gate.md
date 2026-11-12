# 2026-11-12 — a PR check that compiles lance-graph-java's `lgj-abi` against the change

**Kind:** CI only (`.github/workflows/lgj-abi-compat.yml`,
`.github/scripts/lgj-abi-relevant.sh`). No crate changed.
**Grade:** MEASURED locally. With a variant added to `mask_risc::ExecError`
(not committed), the workflow's command exits 101 with `E0004` in `lgj-abi`;
restored, it exits 0. The relevance script answers `true` for the merge of
PR 2039 (7 files under `lance-graph-mask-risc`) and `false` for the board-only
merge of PR 2043.

- `lgj-abi` path-depends on `lance-graph-contract`, `lance-graph-mask-risc`
  and, for its tests, `lance-graph-quack`. Five fixes there share one shape: a
  new `ExecError` variant meeting an exhaustive match. Each was found only at
  the next push to lance-graph-java.
- The check runs `cargo check --all-targets` on `lgj-abi` against the pull
  request's merge commit, with lance-graph-java, ndarray and OGAR at their
  default branch heads. No Java and no benchmark.
- It has no `paths:` filter. It always starts, decides relevance from the
  diff against the first parent, and concludes success when the change cannot
  affect `lgj-abi`, so it can be made a required check without blocking
  unrelated PRs.
- `contents: read`, `persist-credentials: false`, no secret.

**Open:** the workflow has not run on GitHub yet; its first run is on the PR
that adds it. lance-graph-java has the matching daily check in its own PR.
