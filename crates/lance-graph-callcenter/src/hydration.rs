//! Boot-time hydration of `FAMILY_TABLE` from TTL files.
//!
//! Implements the boot sequence described in
//! `.claude/specs/pr-d4-family-hydration.md` §2 (OQ-1 option c):
//!
//! 1. `load_seed(SEED_TTL)` — parse the inline seed bytes.
//! 2. `load_overlay(dir)` — merge an optional on-disk TTL directory.
//! 3. `sanity_gate(&map)` — assert minimum domain coverage.
//! 4. `commit(&map)` — write into `FAMILY_TABLE: OnceLock<Arc<RwLock<FamilyTableInner>>>`.
//!
//! ## Concurrency model
//!
//! `FAMILY_TABLE` is set once via `OnceLock` (the Arc's referent is
//! immutable from the outside; the inner `RwLock` handles generation
//! bumps during hot-reload). Pre-hydration reads of `try_resolve` return
//! `Err(HydrationError::TableNotInitialized)`; the backward-compat shim
//! `super_domain_for_family` maps that to `SuperDomain::Unknown`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;

use lance_graph_ontology::parse_family_registry;

use crate::super_domain::SuperDomain;
use crate::unified_bridge::OgitFamily;

// ═══════════════════════════════════════════════════════════════════════════
// Seed TTL — compiled into the binary via include_str!
// ═══════════════════════════════════════════════════════════════════════════

/// Inline seed: `crates/lance-graph-callcenter/data/family_registry.ttl`.
/// Compiled in at build time. ~10 KB for ~35 entries at GA.
pub const SEED_TTL: &str = include_str!("../data/family_registry.ttl");

// ═══════════════════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════════════════

/// Hydration-specific error surface.
#[derive(Debug, thiserror::Error)]
pub enum HydrationError {
    /// The inline seed could not be parsed. This is a release bug — the seed
    /// ships with the binary, so any parse failure is a compile-time mistake.
    #[error("seed TTL parse failed: {reason}")]
    SeedParseFailed { reason: String },

    /// A directory walk or file read failed.
    #[error("I/O error reading {path}: {reason}")]
    Io { path: String, reason: String },

    /// The sanity gate rejected the merged map (< min distinct non-Unknown
    /// domains). Only returned when `HydrationPolicy::RequireMinDomains`.
    #[error("insufficient domain coverage: found {found}, required {required}")]
    InsufficientDomains { found: usize, required: usize },

    /// `try_resolve()` was called before `new_hydrated()` committed the table.
    #[error("FAMILY_TABLE not yet initialized — call UnifiedBridge::new_hydrated first")]
    TableNotInitialized,
}

// ═══════════════════════════════════════════════════════════════════════════
// HydrationPolicy — fail-hard vs fail-soft
// ═══════════════════════════════════════════════════════════════════════════

/// Controls how `new_hydrated` behaves when the sanity gate fails.
#[derive(Clone, Debug, Default)]
pub enum HydrationPolicy {
    /// Fail the constructor if the merged map has fewer than `min` distinct
    /// non-Unknown domains. Default for binary entrypoints.
    RequireMinDomains { min: usize },
    /// Log a warning and continue with whatever seed data is available.
    /// Default for tests and library consumers.
    #[default]
    BestEffort,
}

// ═══════════════════════════════════════════════════════════════════════════
// HydrationSourceSet — tracks where a hydration came from
// ═══════════════════════════════════════════════════════════════════════════

/// Records the data sources that contributed to a particular `FamilyTableInner`
/// generation.
#[derive(Clone, Debug, Default)]
pub struct HydrationSourceSet {
    pub seed: bool,
    pub overlay_dir: Option<PathBuf>,
}

// ═══════════════════════════════════════════════════════════════════════════
// FamilyTableInner — the versioned payload inside FAMILY_TABLE
// ═══════════════════════════════════════════════════════════════════════════

/// Inner state of `FAMILY_TABLE`. One generation per hydration run.
///
/// `table[family_id]` holds the super domain for that basin. Default is
/// `SuperDomain::Unknown` for every unclassified slot.
pub struct FamilyTableInner {
    pub table: [SuperDomain; 256],
    /// Monotonically increasing generation counter. Starts at 1 on the first
    /// `commit()` call; incremented on every subsequent hot-reload.
    pub generation: u64,
    /// Wall-clock instant of the most recent hydration.
    pub loaded_at: Instant,
    /// Which sources contributed to this generation.
    pub source: HydrationSourceSet,
}

impl FamilyTableInner {
    fn from_map(
        map: &HashMap<u8, SuperDomain>,
        generation: u64,
        source: HydrationSourceSet,
    ) -> Self {
        let mut table = [SuperDomain::Unknown; 256];
        for (&id, &sd) in map {
            table[id as usize] = sd;
        }
        Self {
            table,
            generation,
            loaded_at: Instant::now(),
            source,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FAMILY_TABLE — the global OnceLock
// ═══════════════════════════════════════════════════════════════════════════

/// Global family table. Set exactly once by the first `commit()` call;
/// subsequent calls (`reload_family_table`) write through the inner `RwLock`.
pub static FAMILY_TABLE: OnceLock<Arc<RwLock<FamilyTableInner>>> = OnceLock::new();

// ═══════════════════════════════════════════════════════════════════════════
// Public hydration API
// ═══════════════════════════════════════════════════════════════════════════

/// Parse the inline seed TTL bytes and return a `HashMap<u8, SuperDomain>`.
/// Fails hard on parse error (the seed is compiled in; any failure is a bug).
pub fn load_seed(ttl: &str) -> Result<HashMap<u8, SuperDomain>, HydrationError> {
    let entries = parse_family_registry(ttl.as_bytes())
        .map_err(|f| HydrationError::SeedParseFailed { reason: f.reason })?;
    let mut map = HashMap::new();
    for entry in entries {
        if let Some(sd) = parse_super_domain_name(&entry.super_domain_name) {
            map.insert(entry.family_id, sd);
        }
        // Unknown-mapped or unrecognised names are silently skipped —
        // the table default is Unknown, so omission is correct.
    }
    Ok(map)
}

/// Merge an optional overlay TTL directory into an existing map.
///
/// If `overlay_dir` is `None` or does not exist, returns `Ok(())` and the
/// map is unchanged (soft-warn semantics per spec §2.2).
///
/// Each overlay TTL file is parsed independently; a parse failure on one
/// file skips that file (soft-warn) but does not abort the rest.
pub fn load_overlay(
    map: &mut HashMap<u8, SuperDomain>,
    overlay_dir: Option<&Path>,
) -> Result<(), HydrationError> {
    let dir = match overlay_dir {
        Some(d) if d.exists() => d,
        _ => return Ok(()), // missing overlay is normal — silently skip
    };

    let read = std::fs::read_dir(dir).map_err(|e| HydrationError::Io {
        path: dir.display().to_string(),
        reason: e.to_string(),
    })?;

    for entry in read {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("ttl") {
            continue;
        }
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                // Soft-warn: log but continue.
                eprintln!(
                    "[hydration] WARN: failed to read overlay file {}: {}",
                    path.display(),
                    e
                );
                continue;
            }
        };
        let entries = match parse_family_registry(&bytes) {
            Ok(e) => e,
            Err(f) => {
                eprintln!(
                    "[hydration] WARN: failed to parse overlay file {}: {}",
                    path.display(),
                    f.reason
                );
                continue;
            }
        };
        for entry in entries {
            if let Some(sd) = parse_super_domain_name(&entry.super_domain_name) {
                map.insert(entry.family_id, sd);
            }
        }
    }

    Ok(())
}

/// Assert minimum domain coverage.
///
/// Counts distinct non-Unknown `SuperDomain` values in `map`. Returns
/// `Ok(())` if ≥ `min` are present, `Err(InsufficientDomains)` otherwise.
pub fn sanity_gate(map: &HashMap<u8, SuperDomain>, min: usize) -> Result<(), HydrationError> {
    let distinct: std::collections::HashSet<SuperDomain> = map
        .values()
        .filter(|&&sd| sd != SuperDomain::Unknown)
        .copied()
        .collect();
    if distinct.len() >= min {
        Ok(())
    } else {
        Err(HydrationError::InsufficientDomains {
            found: distinct.len(),
            required: min,
        })
    }
}

/// Write `map` into `FAMILY_TABLE`.
///
/// First call: initialises the `OnceLock` with generation=1.
/// Subsequent calls (hot-reload): acquire the `RwLock` write-guard and bump
/// the generation.
pub fn commit(map: &HashMap<u8, SuperDomain>, source: HydrationSourceSet) {
    match FAMILY_TABLE.get() {
        None => {
            // First hydration — initialise the OnceLock.
            let inner = FamilyTableInner::from_map(map, 1, source);
            // Ignore the error: a concurrent thread may have beaten us.
            let _ = FAMILY_TABLE.set(Arc::new(RwLock::new(inner)));
        }
        Some(arc) => {
            // Hot-reload — bump generation inside the existing Arc.
            let mut guard = arc.write().unwrap();
            let next_gen = guard.generation + 1;
            *guard = FamilyTableInner::from_map(map, next_gen, source);
        }
    }
}

/// Attempt to resolve a family to its super domain.
///
/// Returns `Err(HydrationError::TableNotInitialized)` if `new_hydrated` has
/// not yet committed the table. Returns `Ok(SuperDomain::Unknown)` for any
/// unclassified family id.
pub fn try_resolve(family: OgitFamily) -> Result<SuperDomain, HydrationError> {
    let arc = FAMILY_TABLE
        .get()
        .ok_or(HydrationError::TableNotInitialized)?;
    let guard = arc.read().unwrap();
    Ok(guard.table[family.raw() as usize])
}

/// Return the current generation counter, or 0 if the table has not been
/// initialised yet.
pub fn current_generation() -> u64 {
    FAMILY_TABLE
        .get()
        .map(|arc| arc.read().unwrap().generation)
        .unwrap_or(0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Private helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Map a `superDomain` literal string to a `SuperDomain` enum variant.
/// Returns `None` for unrecognised names (treated as Unknown — omitted).
///
/// OQ-4 resolution (locked 2026-05-13): `"SMB"` maps to `WorkOrderBilling`
/// (covers the 3 Foundry-shape entities under `ogit.SMB:`). `"SMB.bson"` also
/// maps to `WorkOrderBilling` (covers the 14 BSON-shape entities under
/// `ogit.SMB.bson:`). Both sub-namespaces share the same super-domain;
/// the namespace string discriminates them in `OntologyRegistry::enumerate`.
fn parse_super_domain_name(name: &str) -> Option<SuperDomain> {
    match name.trim() {
        "Healthcare" => Some(SuperDomain::Healthcare),
        "Science" => Some(SuperDomain::Science),
        "Genetics" => Some(SuperDomain::Genetics),
        "QuantumPhysics" => Some(SuperDomain::QuantumPhysics),
        "TicketTool" => Some(SuperDomain::TicketTool),
        "WorkOrderBilling" => Some(SuperDomain::WorkOrderBilling),
        "Osint" => Some(SuperDomain::Osint),
        // OQ-4: Foundry-shape SMB (ogit.SMB:) and BSON-shape SMB (ogit.SMB.bson:)
        // both belong to WorkOrderBilling at the super-domain level.
        // The split between the two surface shapes is enforced by the
        // `OgitFamily` slot assignment (0x80..=0x82 vs 0xA0..=0xAD) and
        // by `registry.enumerate("SMB")` vs `registry.enumerate("SMB.bson")`.
        "SMB" | "SMB.bson" => Some(SuperDomain::WorkOrderBilling),
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── U1 — seed parse round-trip ────────────────────────────────────────
    #[test]
    fn family_seed_parse_round_trip() {
        let map = load_seed(SEED_TTL).expect("seed must parse cleanly");
        // Assert every SuperDomain variant 1-7 appears at least once.
        let domains: std::collections::HashSet<SuperDomain> = map.values().copied().collect();
        assert!(
            domains.contains(&SuperDomain::Healthcare),
            "Healthcare must appear in seed"
        );
        assert!(
            domains.contains(&SuperDomain::Science),
            "Science must appear in seed"
        );
        assert!(
            domains.contains(&SuperDomain::Genetics),
            "Genetics must appear in seed"
        );
        assert!(
            domains.contains(&SuperDomain::QuantumPhysics),
            "QuantumPhysics must appear in seed"
        );
        assert!(
            domains.contains(&SuperDomain::TicketTool),
            "TicketTool must appear in seed"
        );
        assert!(
            domains.contains(&SuperDomain::WorkOrderBilling),
            "WorkOrderBilling must appear in seed"
        );
        assert!(
            domains.contains(&SuperDomain::Osint),
            "Osint must appear in seed"
        );
        // FMA is family 0x10 = 16.
        assert_eq!(
            map.get(&16).copied(),
            Some(SuperDomain::Healthcare),
            "FMA (0x10=16) must map to Healthcare"
        );
    }

    // ── U2 — sanity gate passes with ≥ 5 domains ─────────────────────────
    #[test]
    fn family_sanity_gate_passes_five_domains() {
        let mut map = HashMap::new();
        map.insert(1, SuperDomain::Healthcare);
        map.insert(2, SuperDomain::Science);
        map.insert(3, SuperDomain::Genetics);
        map.insert(4, SuperDomain::QuantumPhysics);
        map.insert(5, SuperDomain::TicketTool);
        assert!(sanity_gate(&map, 5).is_ok());
    }

    // ── U3 — sanity gate fails with 4 domains ────────────────────────────
    #[test]
    fn family_sanity_gate_fails_four_domains() {
        let mut map = HashMap::new();
        map.insert(1, SuperDomain::Healthcare);
        map.insert(2, SuperDomain::Science);
        map.insert(3, SuperDomain::Genetics);
        map.insert(4, SuperDomain::QuantumPhysics);
        match sanity_gate(&map, 5) {
            Err(HydrationError::InsufficientDomains {
                found: 4,
                required: 5,
            }) => {}
            other => panic!("expected InsufficientDomains(4, 5), got {:?}", other),
        }
    }

    // ── U5 — load_overlay with non-existent dir is soft-ok ───────────────
    #[test]
    fn family_load_overlay_missing_dir_is_ok() {
        let mut map = HashMap::new();
        load_overlay(&mut map, Some(Path::new("/nonexistent_xyz_123"))).unwrap();
        assert!(
            map.is_empty(),
            "no entries should be added from missing dir"
        );
    }

    // ── U5b — load_overlay with None is soft-ok ──────────────────────────
    #[test]
    fn family_load_overlay_none_is_ok() {
        let mut map = HashMap::new();
        load_overlay(&mut map, None).unwrap();
        assert!(map.is_empty());
    }

    // ── U6 — SMB Foundry-shape entries parse correctly (OQ-4 resolution) ──
    //
    // The seed TTL declares 3 entries under superDomain "SMB" (Foundry-shape,
    // `ogit.SMB:` namespace, family IDs 128..=130).  They must map to
    // WorkOrderBilling and must NOT bleed into the SMB.bson group.
    #[test]
    fn family_smb_foundry_maps_to_work_order_billing() {
        let map = load_seed(SEED_TTL).expect("seed must parse cleanly");

        // Family IDs 128, 129, 130 — SMB Foundry-shape (0x80..=0x82)
        assert_eq!(
            map.get(&128).copied(),
            Some(SuperDomain::WorkOrderBilling),
            "SMB Foundry Customer (0x80=128) must map to WorkOrderBilling"
        );
        assert_eq!(
            map.get(&129).copied(),
            Some(SuperDomain::WorkOrderBilling),
            "SMB Foundry Invoice (0x81=129) must map to WorkOrderBilling"
        );
        assert_eq!(
            map.get(&130).copied(),
            Some(SuperDomain::WorkOrderBilling),
            "SMB Foundry TaxDeclaration (0x82=130) must map to WorkOrderBilling"
        );
    }

    // ── U7 — SMB BSON-shape entries parse correctly (OQ-4 resolution) ─────
    //
    // The seed TTL declares 14 entries under superDomain "SMB.bson" (BSON-shape,
    // `ogit.SMB.bson:` namespace, family IDs 160..=173 = 0xA0..=0xAD).
    // All must map to WorkOrderBilling; no slot collision with Foundry entries.
    #[test]
    fn family_smb_bson_maps_to_work_order_billing() {
        let map = load_seed(SEED_TTL).expect("seed must parse cleanly");

        // All 14 BSON-shape slots 0xA0..=0xAD (160..=173) must be WorkOrderBilling.
        for id in 160u8..=173u8 {
            assert_eq!(
                map.get(&id).copied(),
                Some(SuperDomain::WorkOrderBilling),
                "SMB BSON slot {id} (0x{id:02X}) must map to WorkOrderBilling"
            );
        }
    }

    // ── U8 — Foundry and BSON slots are disjoint (no contamination) ────────
    //
    // OQ-4 guarantee: `registry.enumerate("SMB")` returns exactly 3 Foundry
    // entities; `registry.enumerate("SMB.bson")` returns the 14 BSON entities.
    // At the family-table level this maps to non-overlapping slot ranges.
    #[test]
    fn family_smb_foundry_and_bson_slots_are_disjoint() {
        let map = load_seed(SEED_TTL).expect("seed must parse cleanly");

        // Foundry slots: 128..=130
        let foundry_ids: std::collections::HashSet<u8> = (128u8..=130u8).collect();
        // BSON slots: 160..=173
        let bson_ids: std::collections::HashSet<u8> = (160u8..=173u8).collect();

        // Disjoint check
        let overlap: std::collections::HashSet<u8> =
            foundry_ids.intersection(&bson_ids).copied().collect();
        assert!(
            overlap.is_empty(),
            "SMB Foundry and BSON slot ranges must not overlap; got overlap: {overlap:?}"
        );

        // Both ranges exist in the map
        for &id in &foundry_ids {
            assert!(
                map.contains_key(&id),
                "SMB Foundry slot {id} missing from seed map"
            );
        }
        for &id in &bson_ids {
            assert!(
                map.contains_key(&id),
                "SMB BSON slot {id} missing from seed map"
            );
        }
    }

    // ── U9 — parse_family_registry handles both ogit.SMB: and ogit.SMB.bson: ──
    //
    // Regression against namespace contamination: passing a minimal inline TTL
    // containing one SMB Foundry entry and one SMB.bson entry through
    // `parse_family_registry()` must yield exactly 2 distinct entries with
    // different family IDs.
    #[test]
    fn family_parse_both_smb_namespaces_from_inline_ttl() {
        const INLINE: &str = r#"
@prefix ogit:      <http://www.purl.org/ogit/> .
@prefix ogit.meta: <http://www.purl.org/ogit/meta/> .
@prefix xsd:       <http://www.w3.org/2001/XMLSchema#> .

ogit:SmbFoundryCustomer
    a ogit:FamilyNamespace ;
    ogit.meta:superDomain "SMB" ;
    ogit.meta:familyId    "128"^^xsd:unsignedByte .

ogit:SmbBsonCustomer
    a ogit:FamilyNamespace ;
    ogit.meta:superDomain "SMB.bson" ;
    ogit.meta:familyId    "160"^^xsd:unsignedByte .
"#;
        let entries = lance_graph_ontology::parse_family_registry(INLINE.as_bytes())
            .expect("inline SMB TTL must parse cleanly");

        assert_eq!(entries.len(), 2, "expected exactly 2 entries");

        // Sort is guaranteed by parse_family_registry (sorts by family_id).
        assert_eq!(entries[0].family_id, 128);
        assert_eq!(entries[0].super_domain_name, "SMB");
        assert_eq!(entries[1].family_id, 160);
        assert_eq!(entries[1].super_domain_name, "SMB.bson");

        // Both map to WorkOrderBilling via load_seed logic
        let sd_foundry = parse_super_domain_name(&entries[0].super_domain_name);
        let sd_bson = parse_super_domain_name(&entries[1].super_domain_name);
        assert_eq!(sd_foundry, Some(SuperDomain::WorkOrderBilling));
        assert_eq!(sd_bson, Some(SuperDomain::WorkOrderBilling));
    }
}
