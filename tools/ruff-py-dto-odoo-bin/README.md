# ruff-py-dto Odoo extraction toolchain

POC sidecar — the **Rust binary** lives in the patched `ruff` fork (see
`../../vendor/ruff-patches/README.md`); this directory holds the **config**
and the **post-processing TTL emitter** so an Odoo retarget runs end-to-end.

## Files

- `odoo.config.json` — matches `@api.depends`, `@api.constrains`,
  `@api.onchange`, `@api.model` decorated methods. Used with
  `ruff-py-dto harvest --config`.
- `bundles_to_ttl.py` — reads the NDJSON bundles, emits one OGIT-conformant
  Turtle file per family. Heuristic axis-classifier per
  `E-BUSINESS-LOGIC-IS-GRAMMAR-1` (Deterministic / Heuristic / Hybrid +
  Transitive / Intransitive + first-pass German fiscal causal-marker
  extraction).
- `sample-bundles.account_move.ndjson` — 93 bundles from `account_move`
  for shape inspection.
- `sample-output.account_move.ttl` — what the emitter produces from those
  bundles.
- `sample-preflight.account.json` — preflight report on
  `odoo/addons/account/`, top-level decorator histogram.
- `sample-manifest.full-odoo.json` — manifest from a full-tree harvest
  (3555 bundles, 388 families across 622 addons).

## How to run end-to-end

```bash
# 1. Build the patched binary (see ../../vendor/ruff-patches/README.md).

# 2. Preflight a single addon (sanity check the decorator histogram).
./ruff-py-dto preflight /path/to/odoo/addons/account --out /tmp/preflight-out
cat /tmp/preflight-out/preflight.report.json | jq .decorator_histogram

# 3. Harvest bundles across whatever scope you want.
./ruff-py-dto harvest \
  --config tools/ruff-py-dto-odoo-bin/odoo.config.json \
  --root /path/to/odoo/addons \
  --out /tmp/harvest-out

# 4. Emit TTL.
python3 tools/ruff-py-dto-odoo-bin/bundles_to_ttl.py \
  /tmp/harvest-out/bundles \
  /tmp/ttl-out
```

## Provenance

Empirically validated 2026-05-28: full pipeline produces 3555 OGIT-method
TTL resources across 388 Odoo families in ~30 s wall-clock. See
`.claude/board/ruff-py-dto-poc/REPORT.md` for the empirical numbers,
axis-distribution stats, and the upstream-PR scope.
