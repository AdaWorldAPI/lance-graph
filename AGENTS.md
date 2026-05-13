# Repository Guidelines

## Project Structure & Module Organization
- `crates/lance-graph/` hosts the Rust Cypher engine; keep new modules under `src/` and co-locate helpers inside `query/` or feature-specific submodules.
- `crates/lance-graph-python/src/` contains the PyO3 bridge; `python/python/lance_graph/` holds the pure-Python facade and packaging metadata.
- `python/python/tests/` stores functional tests; mirror new features with targeted cases here and in the corresponding Rust module.
- `examples/` demonstrates Cypher usage; update or add examples when introducing new public APIs.

## Build, Test, and Development Commands
- `cargo check` / `cargo test --all` (run inside `crates/lance-graph`) validate Rust code paths.
- `cargo bench --bench graph_execution` measures performance-critical changes; include shortened runs with `--warm-up-time 1`.
- `uv venv --python 3.11 .venv` and `uv pip install -e '.[tests]'` bootstrap the Python workspace.
- `maturin develop` rebuilds the extension after Rust edits; `pytest python/python/tests/ -v` exercises Python bindings.
- `make lint` (in `python/`) runs `ruff`, formatting checks, and `pyright`.

## Coding Style & Naming Conventions
- Format Rust with `cargo fmt --all`; keep modules and functions snake_case, types PascalCase, and reuse `snafu` error patterns.
- Run `cargo clippy --all-targets --all-features` to catch lint regressions.
- Use 4-space indentation in Python; maintain snake_case modules, CamelCase classes, and type-annotated public APIs.
- Apply `ruff format python/` before committing; `ruff check` and `pyright` enforce import hygiene and typing.

## Testing Guidelines
- Add Rust unit tests alongside implementations via `#[cfg(test)]`; prefer focused scenarios over broad integration.
- Python tests belong in `python/python/tests/`; name files `test_*.py` and use markers (`gpu`, `cuda`, `integration`, `slow`) consistently.
- When touching performance-sensitive code, capture representative `cargo bench` or large-table pytest timing notes in the PR.

## Commit & Pull Request Guidelines
- Follow the existing history style (`feat(graph):`, `docs:`, `refactor(query):`), using imperative, ≤72-character subjects.
- Reference issues or discussions when relevant and include brief context in the body.
- PRs should describe scope, list test commands run, mention benchmark deltas when applicable, and highlight impacts on bindings or examples.

## Cursor Cloud specific instructions

### Prerequisites (handled by update script)

The update script installs `protobuf-compiler` and `libssl-dev` via apt and
clones `AdaWorldAPI/ndarray` to `/ndarray` (the path `crates/*/Cargo.toml`
resolve `../../../ndarray` to from inside the workspace). No other external
services or databases are required — all storage uses Lance datasets and
in-memory Arrow.

### Running Rust services

Standard build/test/lint commands are documented above and in `CLAUDE.md`.
Key non-obvious notes:

- **Workspace members vs excluded crates:** `Cargo.toml` lists 10 workspace
  members and 15 excluded crates. `cargo test` (no args) only runs workspace
  member tests. To test excluded crates use `cargo test --manifest-path
  crates/<name>/Cargo.toml`.
- **CI-gated checks (what must pass before merge):**
  1. `cargo fmt -- --check` (workspace members)
  2. `cargo fmt --manifest-path crates/lance-graph/Cargo.toml -- --check`
  3. `cargo clippy --manifest-path crates/lance-graph-contract/Cargo.toml
     --lib --tests -- -D warnings` (mandatory, zero tolerance)
  4. `cargo clippy --manifest-path crates/lance-graph/Cargo.toml --lib
     --tests -- -D warnings` (advisory, `continue-on-error: true`)
  5. `cargo test --manifest-path crates/lance-graph/Cargo.toml` (unit + doc)
  6. `cargo test --manifest-path crates/lance-graph-contract/Cargo.toml --lib`
- **Excluded crates have extensive `cargo fmt` drift** (especially
  `cognitive-shader-driver`, `thinking-engine`, `holograph`, `bgz17`). This
  is known and not checked by CI. Each crate was formatted by its original
  author independently; applying `cargo fmt` to one of these is a deliberate
  decision, not a CI requirement.
- **bgz-tensor has 5 pre-existing test failures** (size assertions that are
  platform-sensitive). These are not in the CI gate.
- **ndarray path:** All `path = "../../../ndarray"` entries resolve to
  `/ndarray` when the workspace is at `/workspace`. The update script clones
  it there on first run and does `git pull` on subsequent runs.
