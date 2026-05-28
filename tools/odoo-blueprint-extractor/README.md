# odoo-blueprint-extractor

Stdlib-only Python 3 package that parses Odoo ORM classes via the `ast`
module and emits candidate `OdooEntity` consts as Rust source text with
`OdooConfidence::Extracted` provenance. Part of Stage-1 extraction for
the `lance-graph-ontology` crate.

## Usage

```bash
python -m odoo_blueprint_extractor \
    --addons /home/user/odoo/addons \
    --addon uom \
    --out -              # '-' = stdout; a directory path for file output (EXT-2)
    [--audit /tmp/fallback.json]
```

Stdout produces a complete Rust source file ready for
`crates/lance-graph-ontology/src/odoo_blueprint/extracted/<addon>.rs`.

## Design rationale

- **Python `ast` over tree-sitter**: `ast` is stdlib, ships with every
  CPython 3.10+ install, has zero runtime deps, and handles every ORM
  class/field shape we observe. Tree-sitter requires a native extension
  and is not present on the build host (`/home/user/odoo/` is offline).

- **Offline extraction**: Odoo source lives at `/home/user/odoo/addons/`
  — no network calls, no Odoo runtime, pure static analysis. Fields and
  methods are read from AST nodes, not reflected from a running registry.

- **`OdooConfidence::Extracted`**: marks every emitted const as
  auto-extracted (not human-curated). The curated L-doc consts in
  `odoo_blueprint::l{1..15}` keep `OdooConfidence::Curated` and remain
  canonical on merge conflicts (per the plan's merge ordering rule).

- **`EXT_` prefix**: extracted consts are prefixed `EXT_` (e.g.
  `EXT_ACCOUNT_MOVE`) so they never share a symbol with curated lane
  consts (e.g. `ACCOUNT_MOVE_L1`). Cross-reference lives in
  `D-ODOO-EXT-5`'s `pairing.rs`.

- **Regulation IRIs from `parsers/regulation.py`**: a 30-entry keyword
  table maps docstring/field-help text to OGIT-inherited codebook IRIs
  (e.g. `"ogit:regulation/de/ustg/15"` for Vorsteuer). Per
  `E-CODEBOOK-INHERITS-FROM-OGIT`, the IRI is the identity handle;
  content lives in the OGIT registry, not here.

- **Fallback log**: unrecognized field kinds become `OdooFieldKind::Other`;
  unclassified methods become `OdooMethodKind::Helper`. Both are logged to
  `--audit <path>` (JSON). Stage-1 success bar: < 5 % `::Other` rate per
  TIER-1 addon.

## Cross-references

- Plan: `.claude/plans/odoo-source-extraction-v1.md`
- Rust shapes: `crates/lance-graph-ontology/src/odoo_blueprint/mod.rs`
- L-doc curated consts: `crates/lance-graph-ontology/src/odoo_blueprint/l{1..15}.rs`
- Epiphany: `E-CODEBOOK-INHERITS-FROM-OGIT` (`.claude/board/EPIPHANIES.md`)
