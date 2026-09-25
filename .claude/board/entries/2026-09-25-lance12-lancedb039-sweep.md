# 2026-09-25 — lance 12 / lancedb 0.39 / Rust 1.98.1 sweep

**Status:** MEASURED · OPEN (merge order across repos)

## What moved
- `lance*` `11.*` → `12.*` and `lancedb` `0.38.*` → `0.39.*`, measured against crates.io. lancedb 0.39.0 pins `lance = "=12.0.0"`. arrow ^58 and datafusion ^54 are unmoved.
- **`object_store` 0.13 → 0.14 for our direct dependency.** lance 12 requires ^0.14.1, while datafusion 54 stays on ^0.13.2, so both majors are in the graph (upstream's split). Every direct call site here hands the store or a `Path` to lance or to lance-graph-hydrate. On 0.13, `memwal_atomicity_probe` and `hydration_probe` failed with E0308 ("multiple different versions of crate `object_store`").
- **lancedb `remote` on the consumers behind `lancedb-sdk`** (lance-graph, surreal_container). 0.39.0 does not compile without it: `pub mod job;` is ungated while `job.rs` uses the `remote`-gated `Error::Http`. It is set at crate level, so it is active only with the optional dependency.
- Sub-crate toolchains (`reader-lm`, `bge-m3`, `python`) 1.95.0 → 1.98.1, matching the root. Builder images for `cognitive-stack`, `symbiont` and `thinking-engine` (was 1.82) → `rust:1.98`.

## Gates
- `cargo check --all-targets` on lance-graph, -catalog, -hydrate, -callcenter and -ontology: green. The check first went red on the `object_store` split, which the pin change above fixed.
- `cargo check -p lance-graph --features lancedb-sdk --lib`: green.
- `cargo test -p lance-graph-hydrate -p lance-graph-catalog`: green.
- The lance-graph crate's own full test run was NOT run locally (disk); CI is the gate.

## Open — merge order
MedCare-rs and q2 consume lance-graph from `branch = "main"`. MedCare-rs on its own bump branch resolves lance 11 AND 12 side by side until this PR merges. The consumer bumps should land right after this one; if a consumer merges first, its graph carries two lance majors.
