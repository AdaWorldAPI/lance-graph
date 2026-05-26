# `vendor/` — sibling-clone softlinks

This directory follows the workspace convention from
`MedCare-rs/vendor/` (and the pattern documented in
`MedCare-rs/Cargo.toml`'s exclude block): vendored dependencies are
softlinks to sibling clones at `/home/user/<name>/`. The link is
stable from inside the lance-graph repo; the actual source tree
lives alongside this repo on the filesystem.

## Current contents

| Softlink | Target | Source SHA | Imported | Reason |
|---|---|---|---|---|
| `ractor` | `../../ractor` (i.e. `/home/user/ractor`) | `a50c675` | 2026-05-21 | **Temporary vendor import** — `AdaWorldAPI/ractor` is not yet in the session's MCP github scope. Imported via anonymous public zipball from `https://api.github.com/repos/AdaWorldAPI/ractor/zipball/HEAD`. Replace with proper MCP-scoped fetch once `adaworldapi/ractor` is added to the scope. |

The SHA + timestamp for each import is also stored at
`/home/user/<name>/.vendor-import-sha.txt` for cross-session
visibility.

## Why softlinks (not in-tree copies)

The workspace convention is sibling-clone softlinks for three reasons:

1. **Avoids `multiple workspace roots found in the same workspace`**
   — every vendored dep here (`ractor`, `lance-graph` itself when
   vendored elsewhere, `ndarray` when vendored) declares its own
   `[workspace]` block. Excluding the softlink path from this repo's
   workspace via `[workspace] exclude = ["vendor/<name>"]` (in the
   consumer's `Cargo.toml`, added when a path dep is wired) keeps
   cargo happy.
2. **Repo size stays small** — a 3.4 MB source tree doesn't bloat
   `lance-graph`'s commit history.
3. **Sibling clone is the single source of truth** — updates land
   in `/home/user/<name>/` once and every consumer's softlink picks
   them up.

## Replacing this temporary import with the canonical path

When `adaworldapi/ractor` is added to the MCP github scope:

1. Confirm the sibling clone at `/home/user/ractor` is current
   (re-zipball if needed, or `git clone` via the local proxy once
   the proxy honors the new scope).
2. Optionally `git init` + add the `adaworldapi/ractor` remote and
   `git fetch` to attach commit history.
3. The softlink + path-dep wiring in any consumer's `Cargo.toml`
   stays unchanged.

## Wiring a path dep (when needed)

This vendor import is **source-only**; no `Cargo.toml` yet declares
a path dep on `vendor/ractor`. When a consumer crate inside this
workspace needs to depend on ractor, add (in the consumer's
`Cargo.toml`):

```toml
[dependencies]
ractor = { path = "../../vendor/ractor/ractor" }
```

…and (in the root `Cargo.toml`):

```toml
[workspace]
exclude = [
    # existing entries...
    "vendor/ractor",
]
```

The `exclude` is mandatory because `vendor/ractor` (via softlink to
`/home/user/ractor`) declares its own `[workspace]` block with
members `ractor`, `ractor_cluster`, `ractor_cluster_derive`,
`ractor_cluster_integration_tests`, `ractor_example_entry_proc`,
`ractor_playground`, `xtask`.
