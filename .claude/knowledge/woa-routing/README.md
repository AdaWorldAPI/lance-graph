# WoA Flask Routing Table (Harvested Reference)

> **Source:** `/home/user/WoA/app.py` (Flask backend, mandantenfähig, Acme Services GmbH)
> **Harvested:** 2026-05-16 via Python `ast.parse` from lance-graph session
> **Routes:** 96 endpoints (60 with GET, 59 with POST; many serve both)
> **Cross-repo reference only** — checked into lance-graph `.claude/` to make the WoA surface area greppable from this workspace without a zipball roundtrip.

## Files

- `routing-table.json` — full detail per route: `path`, `methods`, `handler`, `decorator`, `line`, `context` (enclosing function — usually `create_app` since WoA uses the app-factory pattern), `docstring`, `other_decorators` (auth/csrf/limiter chain).
- `routing-table.csv` — flat columns; `methods` pipe-delimited, `other_decorators` semicolon-delimited; suitable for spreadsheet import.

## Harvester contract

The harvester recognizes the following decorator shapes:

- `@app.route('/path', methods=[...])` (Flask's main pattern; the WoA standard)
- `@bp.route(...)` / any `<expr>.route(...)` (blueprint variant)
- `@app.get(...)` / `.post(...)` / `.put(...)` / `.delete(...)` / `.patch(...)` / `.options(...)` / `.head(...)` (Flask 2.x shortcuts; method inferred from decorator name when `methods` kwarg is absent)

Routes are sorted by `(path, methods)` for deterministic diffs.

## What this is NOT

- Not a live runtime extraction (no `app.url_map` introspection) — pure static AST. This means decorator expressions that are computed at runtime (e.g. `route(some_var, ...)`) are skipped silently.
- Not a security or auth model — the `other_decorators` column surfaces what's stacked on each endpoint (`@login_required`, `@csrf.exempt`, `@limiter.limit(...)`, etc.) but does not interpret them.

## When to refresh

When `WoA/app.py` changes meaningfully — re-run the harvester. The script lives inline in the lance-graph commit that introduced this directory; copy it from there or re-run the heredoc from session history. Drop a dated note here when you do.

## Use cases

- **Cross-repo audit:** "Does the WoA API expose route X for tenant Y?" — grep here, then jump to `WoA/app.py:<line>`.
- **Membrane wiring (callcenter / sharepoint / spear):** identifying which WoA endpoints feed which downstream consumer is faster against this table than against the 2,904-line app.py.
- **Sprint planning:** when planning lance-graph integration with WoA (e.g. via `q2` cockpit or `sharepoint` connector), this table is the contract surface.
