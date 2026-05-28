# D-RPYDTO-2a — class-body recursion patch (POC, in-session)

**Status.** Empirically validated 2026-05-28 in-session. Builds clean, produces
3555 bundles across 388 families from `odoo/addons/` (vs. 0 bundles before the
patch). See `.claude/board/ruff-py-dto-poc/REPORT.md` for the full numbers and
artifact paths.

**Authoritative home.** This is a temporary vendor of the patched crate
sources from `https://github.com/astral-sh/ruff` master @ 2025-05-19 (zipball).
The upstream PR lives at `D-RPYDTO-2a` in `.claude/plans/D-RPYDTO-2a.md` (or
will, once that document is rewritten with the empirical evidence). This
directory exists **only** so the next session can rebuild the binary without
re-downloading the zipball, and so reviewers can read the diff without
hunting for the working tree.

## What the patch changes

Three files in `crates/ruff_python_dto_check/src/`:

1. **`lib.rs::harvest_module`** — replaces the top-level-only loop over
   `parsed.syntax().body` with a recursive walker (`walk_module_fns`) that
   descends into `Stmt::ClassDef.body` and nested `Stmt::FunctionDef.body`.
   Makes class methods on `MethodView` / `models.Model` reachable.

2. **`matcher/function_with_decorator.rs::harvest_module_with_config`** — same
   fix applied to the config-driven matcher path (`walk_function_defs`
   helper).

3. **`preflight/scanner.rs::scan_file`** — splits the per-function logic into
   `process_function_def` (returns `bool` for `file_has_matched`) and a new
   `process_class_body` that recursively walks class bodies + nested classes.
   Imports + class counting remain at module level (no change).

Everything else is identical to upstream.

## Why this is critical-path

The Odoo retarget (and any framework that defines decorated handlers inside
class bodies — Odoo `models.Model`, Flask `MethodView`, Django CBV, FastAPI
class-based routing) returns zero bundles without this fix. With the fix,
3555/3555 `@api.depends|constrains|onchange|model` methods across 388
business-object families flow into the bundle output and downstream TTL.

## How to rebuild

```bash
# 1. Get the zipball (or use the working tree at /tmp/ruff-fork/ruff-main).
curl -L -o ruff.zip https://github.com/astral-sh/ruff/archive/refs/heads/main.zip
unzip ruff.zip

# 2. Apply the three files from this directory:
cp lib.rs.patched ruff-main/crates/ruff_python_dto_check/src/lib.rs
cp matcher_function_with_decorator.rs.patched \
   ruff-main/crates/ruff_python_dto_check/src/matcher/function_with_decorator.rs
cp preflight_scanner.rs.patched \
   ruff-main/crates/ruff_python_dto_check/src/preflight/scanner.rs

# 3. Build.
cd ruff-main
cargo build -p ruff_python_dto_check --release --bin ruff-py-dto

# 4. Use against Odoo (see ../tools/ruff-py-dto-odoo-bin/).
./target/release/ruff-py-dto preflight /path/to/odoo/addons/account \
  --out /tmp/preflight-out
./target/release/ruff-py-dto harvest \
  --config /path/to/lance-graph/tools/ruff-py-dto-odoo-bin/odoo.config.json \
  --root /path/to/odoo/addons \
  --out /tmp/harvest-out
```

## Upstream-PR scope

The upstream PR proper (target: astral-sh/ruff PR queue) should land in this
order:

1. **Walk-helper extraction** (`walk_function_defs` etc.) as a refactor with
   *no behavior change* — current call sites still touch only top-level
   functions because the walker stays at depth 0 for them.

2. **Class-body recursion opt-in** behind a new config flag
   (`match_rules[].search_scope = "class_methods"`) so Flask-route-only users
   keep the old behavior bit-for-bit. Default OFF.

3. **`ClassWithBase` matcher kind** (separate PR) — the *typed* path that lets
   a rule say "match any class with base `models.Model`, then walk its methods
   looking for decorator X." Adds two `MatchKind` variants:
   `ClassWithBase { base_qualname, search_methods: MatchRule }` and the
   `class_field_metadata` emit-path helpers.

4. **First-class `Axis` + `DelegationTuple` on `RouteContract`** — moves the
   dual-axis classification (per `E-BUSINESS-LOGIC-IS-GRAMMAR-1`) out of the
   downstream TTL emitter and into the bundle schema itself, so every
   consumer reads the same axis/delegation values.

This vendored directory ships only step 1 + step 2's "walker is always on"
shortcut, because that's what gets bundles flowing for the in-session
empirical proof. Steps 3–4 are upstream-PR work in `D-RPYDTO-2a.md`.
