//! D-CASCADE-V1-1 — Zone 1 / Zone 2 `serde::Serialize` static check.
//!
//! Per `.claude/plans/ogit-cascade-supabase-callcenter-v1.md` Pillar 2 +
//! `.claude/knowledge/soa-dto-dependency-ledger.md` Zone classifications:
//!
//! - **Zone 1** — `cognitive-shader-driver::BindSpace` columns +
//!   `thinking-engine::dto.rs` types. May NOT carry `serde::Serialize`.
//! - **Zone 2** — `lance-graph-callcenter::lance_membrane`,
//!   BBB-scalar-only Arrow projection (`CognitiveEventRow`,
//!   `ExternalIntent`, `CommitFilter`). May NOT carry `serde::Serialize`.
//! - **Zone 3** — `transcode/`, `phoenix`, `postgrest`, `drain`,
//!   `supabase`. Serialize ALLOWED.
//!
//! This build script parses the four Zone 1/2 source files with `syn` and
//! emits `cargo::error::` if any `pub struct` / `pub enum` carries
//! `#[derive(... Serialize ...)]`.
//!
//! The poison-pill test (`tests/zone_serialize_check_compile_fail.rs`) is
//! gated on `--features _internal_test_serialize_poison`. Default builds
//! must NOT activate that feature; the check reads only the four canonical
//! files listed below.

use std::path::{Path, PathBuf};

/// Files this check parses. Ordered (Zone 2, Zone 1) → (Zone 2 trait, Zone 1
/// shader DTOs). Paths are resolved relative to `CARGO_MANIFEST_DIR` /
/// workspace root so the build script works whether run from the crate or
/// from the workspace root.
const ZONE_FILES: &[(&str, &str, &str)] = &[
    // (zone label, path-relative-to-callcenter-crate, fallback path-relative-to-workspace)
    (
        "Zone 2",
        "src/external_intent.rs",
        "crates/lance-graph-callcenter/src/external_intent.rs",
    ),
    (
        "Zone 2",
        "src/lance_membrane.rs",
        "crates/lance-graph-callcenter/src/lance_membrane.rs",
    ),
    (
        "Zone 2",
        "../lance-graph-contract/src/external_membrane.rs",
        "crates/lance-graph-contract/src/external_membrane.rs",
    ),
    (
        "Zone 1",
        "../lance-graph-contract/src/cognitive_shader.rs",
        "crates/lance-graph-contract/src/cognitive_shader.rs",
    ),
];

fn resolve(rel: &str, fallback: &str) -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p1 = Path::new(&manifest).join(rel);
    if p1.is_file() {
        return Some(p1);
    }
    // Try workspace-root fallback (manifest_dir is callcenter; ../../ is workspace root).
    let p2 = Path::new(&manifest).join("..").join("..").join(fallback);
    if p2.is_file() {
        return Some(p2);
    }
    None
}

/// Inspect a `#[derive(...)]` attribute for any path ending in `Serialize`.
/// Returns the offending derive name on hit (e.g. `Serialize` or
/// `serde::Serialize`).
fn derive_has_serialize(attr: &syn::Attribute) -> Option<String> {
    if !attr.path().is_ident("derive") {
        return None;
    }
    let mut hit: Option<String> = None;
    let _ = attr.parse_nested_meta(|meta| {
        // Last segment of the derive path: `Serialize` matches both
        // `Serialize` and `serde::Serialize`.
        if let Some(last) = meta.path.segments.last() {
            if last.ident == "Serialize" {
                let full = meta
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                hit = Some(full);
            }
        }
        Ok(())
    });
    hit
}

/// Returns `(name, derive_name)` violations found in a parsed file.
fn scan_file(file: &syn::File) -> Vec<(String, String)> {
    let mut hits = Vec::new();
    for item in &file.items {
        let (ident, attrs, vis) = match item {
            syn::Item::Struct(s) => (s.ident.to_string(), &s.attrs, &s.vis),
            syn::Item::Enum(e) => (e.ident.to_string(), &e.attrs, &e.vis),
            _ => continue,
        };
        if !matches!(vis, syn::Visibility::Public(_)) {
            continue;
        }
        for attr in attrs {
            if let Some(derive_name) = derive_has_serialize(attr) {
                hits.push((ident.clone(), derive_name));
            }
        }
    }
    hits
}

fn main() {
    let mut violations: Vec<(String, String, String, String)> = Vec::new();
    let mut scanned: Vec<String> = Vec::new();

    for (zone, rel, fallback) in ZONE_FILES {
        let path = match resolve(rel, fallback) {
            Some(p) => p,
            None => {
                println!(
                    "cargo:warning=zone_serialize_check: could not locate {} (fallback {}); skipping",
                    rel, fallback
                );
                continue;
            }
        };
        println!("cargo:rerun-if-changed={}", path.display());
        let src = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                println!(
                    "cargo:warning=zone_serialize_check: failed to read {}: {}",
                    path.display(),
                    e
                );
                continue;
            }
        };
        let file = match syn::parse_file(&src) {
            Ok(f) => f,
            Err(e) => {
                println!(
                    "cargo:warning=zone_serialize_check: failed to parse {}: {}",
                    path.display(),
                    e
                );
                continue;
            }
        };
        scanned.push(path.display().to_string());
        for (ident, derive_name) in scan_file(&file) {
            violations.push((
                zone.to_string(),
                path.display().to_string(),
                ident,
                derive_name,
            ));
        }
    }

    println!(
        "cargo:warning=zone_serialize_check: scanned {} file(s) for Zone 1/2 Serialize violations",
        scanned.len()
    );

    if !violations.is_empty() {
        for (zone, path, ident, derive_name) in &violations {
            // cargo:warning makes the violation visible in the build output.
            println!(
                "cargo:warning=ZONE-SERIALIZE-VIOLATION [{}] {} :: pub struct/enum `{}` carries `#[derive({})]` — Zone 1/2 types may NOT serialize (see soa-dto-dependency-ledger.md)",
                zone, path, ident, derive_name
            );
        }
        // FIX-2 (meta-1 review, 2026-05-07) + codex P2 (2026-05-07):
        // Scope the hard abort to direct builds of this crate. CARGO_PKG_NAME
        // is the package whose build script is running (always
        // "lance-graph-callcenter" here), so the previous CARGO_PKG_NAME
        // check was tautologically true and aborted ALL invocations —
        // including transitive cargo check of unrelated crates that pull
        // callcenter as a dep. The right env var is CARGO_PRIMARY_PACKAGE,
        // set by cargo to "1" only when the package is being built
        // directly (e.g. `cargo build -p lance-graph-callcenter` or a
        // top-level test in this crate). For all transitive consumers,
        // CARGO_PRIMARY_PACKAGE is unset, so the abort path stays
        // dormant and only the cargo:warning lines fire.
        let direct_build = std::env::var("CARGO_PRIMARY_PACKAGE").is_ok();
        let strict = std::env::var("CARGO_FEATURE_ZONE_CHECK_STRICT").is_ok();
        if direct_build || strict {
            // cargo::error:: aborts the build with the message attached to the
            // first violation.
            let first = &violations[0];
            println!(
                "cargo::error=D-CASCADE-V1-1 zone_serialize_check: `{}` in {} (Zone {}) carries `#[derive({})]` — Zone 1/2 types may NOT serialize. Move to Zone 3 (transcode/phoenix/postgrest/drain/supabase) or remove the derive.",
                first.2, first.1, first.0, first.3
            );
            std::process::exit(1);
        }
    }
}
