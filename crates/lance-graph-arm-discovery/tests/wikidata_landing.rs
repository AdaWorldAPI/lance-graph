// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! D-ARM-14 Phase 2 — the worked end-to-end: **splat → aerial discovery →
//! `dolce_id` + ndjson → land on the canonical hub types**, on a Wikidata-shaped
//! fixture. Gated behind `--features landing` so aerial's default build stays
//! zero-dep.
//!
//! It lands on the **real merged** hub types — `lance_graph_contract::class_view::FieldMask`
//! (presence) and `hash::fnv1a_str` (the structural-signature value). The 16ⁿ
//! `NiblePath` router is #442 (not yet on `main`), so its bit-shift addressing is
//! reproduced inline here, annotated `= contract::hhtl::NiblePath`; swap the
//! `np_*` helpers for `NiblePath::{root,child,basin,is_ancestor_of}` once #442
//! merges. Firewall intact: aerial (the lib) never imports the hub; this *test*
//! consumes both sides to prove the seam — the `(ClassId, signature, FieldMask)`
//! triple + the `dolce_id` u8 are the only things that cross.
//!
//! Run: `cargo test --manifest-path crates/lance-graph-arm-discovery/Cargo.toml \
//!        --features landing --test wikidata_landing -- --nocapture`

#![cfg(feature = "landing")]

use lance_graph_arm_discovery::ndjson::to_ndjson;
use lance_graph_arm_discovery::{
    extract_rules, CandidateTriple, Dataset, ExtractParams, FeatureSpec, Item, OntologyProjector,
    TopKDistance, NARS_PERSONALITY_K,
};
use lance_graph_contract::class_view::{ClassId, FieldMask};
use lance_graph_contract::hash::fnv1a_str;

// ── the Wikidata-shaped fixture ──────────────────────────────────────────────
// Real QIDs; `etype` = the discovery-side entity-type category; `dolce` = the
// cache-resolved DOLCE id (0=Endurant, 1=Perdurant); `path` = the P279 nibble
// descent below the basin; `props` = the representative property-id set (its
// IDENTITIES drive the shape-family, its COUNT drives the presence mask).
struct WClass {
    id: ClassId,
    qid: &'static str,
    etype: u32,
    dolce: u8,
    path: &'static [u8],
    props: &'static [&'static str],
}

fn corpus() -> Vec<WClass> {
    vec![
        WClass { id: 1, qid: "Q215627", etype: 0, dolce: 0, path: &[0x1], props: &["P21", "P569", "P570"] }, // person
        WClass { id: 2, qid: "Q5", etype: 1, dolce: 0, path: &[0x1, 0x2], props: &["P21", "P569", "P570", "P106"] }, // human ⊂ person (+occupation)
        WClass { id: 3, qid: "Q515", etype: 2, dolce: 0, path: &[0x3], props: &["P1082", "P625", "P17"] }, // city
        WClass { id: 4, qid: "Q11424", etype: 3, dolce: 1, path: &[0x0], props: &["P57", "P577", "P161"] }, // film
        WClass { id: 5, qid: "Q5398426", etype: 4, dolce: 1, path: &[0x1], props: &["P57", "P577", "P161"] }, // tv-series ≡ film
        WClass { id: 6, qid: "Q1656682", etype: 5, dolce: 1, path: &[0x2], props: &["P585", "P276"] }, // event
    ]
}

// ── contract::hhtl::NiblePath (#442) — inlined as (path, depth) until it merges ──
fn np_root(basin: u8) -> (u64, u32) {
    ((basin & 0x0F) as u64, 1)
}
fn np_child((p, d): (u64, u32), n: u8) -> (u64, u32) {
    if d >= 16 || n >= 16 {
        (p, d)
    } else {
        ((p << 4) | (n & 0x0F) as u64, d + 1)
    }
}
fn np_basin((p, d): (u64, u32)) -> u8 {
    ((p >> (4 * (d - 1))) & 0x0F) as u8
}
fn np_is_ancestor((pa, da): (u64, u32), (pb, db): (u64, u32)) -> bool {
    da > 0 && da <= db && (pb >> (4 * (db - da))) == pa
}
fn nibble_path(basin: u8, children: &[u8]) -> (u64, u32) {
    children.iter().fold(np_root(basin), |acc, &c| np_child(acc, c))
}

// ── hub landing primitives (REAL merged contract types) ──────────────────────
/// Presence bitmask — one bit per present property (`= contract::class_view::FieldMask`).
fn presence(props: &[&str]) -> FieldMask {
    let positions: Vec<u8> = (0..props.len() as u8).collect();
    FieldMask::from_positions(&positions)
}
/// `FieldMask::inherit` (#442) — bitwise union; inlined via the public `.0`.
fn inherit(parent: FieldMask, delta: FieldMask) -> FieldMask {
    FieldMask(parent.0 | delta.0)
}
/// Structural signature over the canonical (sorted, deduped) property-ID set +
/// basin — `= class_signature::StructuralSignature(fnv1a(...) as u32)`. Property
/// IDENTITIES (not positions) so person {P21,…} ≠ film {P57,…} though both fill
/// positions {0,1,2}.
fn signature(dolce: u8, props: &[&str]) -> u32 {
    let mut ps: Vec<&str> = props.to_vec();
    ps.sort_unstable();
    ps.dedup();
    fnv1a_str(&format!("{dolce}|{}", ps.join(","))) as u32
}

/// Discover `entity-type → DOLCE-facet` from a splat-shaped codebook + data —
/// the aerial proposer half. Returns the per-etype discovered `dolce_id`.
fn discover_dolce_ids(classes: &[WClass]) -> std::collections::HashMap<u32, u8> {
    // f0 = entity-type (6), f1 = DOLCE facet (4). Plant each etype ⇒ its facet.
    let spec = FeatureSpec::new(vec![6, 4]);
    let mut rows = Vec::new();
    for c in classes {
        for _ in 0..50 {
            rows.push(vec![c.etype, c.dolce as u32]);
        }
    }
    let data = Dataset::new(spec.clone(), rows);
    // Splat top-k: each entity-type sits next to its DOLCE facet (distance 1).
    let edges: Vec<(Item, Item, u32)> = classes
        .iter()
        .map(|c| (Item::new(0, c.etype), Item::new(1, c.dolce as u32), 1))
        .collect();
    let oracle = TopKDistance::new(spec, u32::MAX, &edges);

    let rules = extract_rules(
        &oracle,
        &data,
        &ExtractParams { theta: 2, max_antecedent: 1, min_support_ppm: 100_000, min_confidence_ppm: 700_000 },
    );

    // The proposer emits the stable dolce_id (NOT a hardcoded IRI) — Phase-2 fix.
    let proj = OntologyProjector::dolce_subclass("wd:");
    let mut out = std::collections::HashMap::new();
    for r in &rules {
        if r.antecedent.len() == 1 && r.antecedent[0].feature == 0 && r.consequent[0].feature == 1 {
            if let Some(dolce) = proj.dolce_id(&r.consequent) {
                out.insert(r.antecedent[0].category, dolce);
            }
        }
    }
    out
}

#[test]
fn end_to_end_splat_to_wikidata_hhtl() {
    let classes = corpus();

    // ── Stage A+B: aerial discovers the basin (dolce_id) from splat + data ──
    let discovered = discover_dolce_ids(&classes);
    for c in &classes {
        assert_eq!(
            discovered.get(&c.etype).copied(),
            Some(c.dolce),
            "aerial must recover {}'s DOLCE basin from the splat",
            c.qid
        );
    }
    eprintln!("✓ Stage A/B: aerial recovered all {} DOLCE basins from the splat", classes.len());

    // ── Stage C: land each discovered class on the canonical hub types ──
    let mut sigs = std::collections::HashSet::new();
    let mut landed = Vec::new();
    for c in &classes {
        let basin = discovered[&c.etype]; // from DISCOVERY, not hardcoded
        let path = nibble_path(basin, c.path); // = NiblePath::root(basin).child(..)
        let mask = presence(c.props); // = FieldMask
        let sig = signature(basin, c.props); // = StructuralSignature
        let triple: (ClassId, u32, FieldMask) = (c.id, sig, mask); // the D-CLS triple
        sigs.insert(sig);
        landed.push((c.qid, basin, path, triple));
        assert_eq!(np_basin(path), basin, "basin survives down the nibble path");
        assert_eq!(mask.count(), c.props.len() as u32, "one present bit per property");
        eprintln!("  landed {:>9} → basin {basin} path {:#x} sig {sig:#010x} mask {:#06b}", c.qid, path.0, mask.0);
    }

    // N4 falsification #1 — the corpus COLLAPSES to fewer shape-families.
    assert!(sigs.len() < classes.len(), "shapes must collapse (6 classes → {} families)", sigs.len());
    assert_eq!(sigs.len(), 5, "film ≡ tv-series is the one twin");
    let film = classes.iter().find(|c| c.qid == "Q11424").unwrap();
    let tv = classes.iter().find(|c| c.qid == "Q5398426").unwrap();
    assert_eq!(signature(1, film.props), signature(1, tv.props), "film and tv-series share one signature");
    let person = classes.iter().find(|c| c.qid == "Q215627").unwrap();
    assert_ne!(signature(0, person.props), signature(1, film.props), "person ≠ film");

    // N4 falsification #2 — subclass inherits PATH + MASK-as-delta, basin preserved.
    let human = classes.iter().find(|c| c.qid == "Q5").unwrap();
    let person_path = nibble_path(0, person.path);
    let human_path = nibble_path(0, human.path);
    assert!(np_is_ancestor(person_path, human_path), "person ⊃ human in the nibble tree");
    assert_eq!(np_basin(person_path), np_basin(human_path), "subclassing preserves the DOLCE basin");
    let delta = FieldMask::from_positions(&[3]); // human's extra property (P106) bit
    assert_eq!(inherit(presence(person.props), delta), presence(human.props), "human mask = person mask ⊕ P106");

    // ── the ndjson the hub reads (the firewall-crossing wire) ──
    let proj = OntologyProjector::dolce_subclass("wd:");
    // one illustrative rule: person ⇒ Endurant
    let rule = lance_graph_arm_discovery::CandidateRule {
        antecedent: vec![Item::new(0, person.etype)],
        consequent: vec![Item::new(1, person.dolce as u32)],
        cooccur: 50,
        antecedent_count: 50,
        window: 300,
    };
    let triples = vec![CandidateTriple::from_rule(&rule, &proj, NARS_PERSONALITY_K)];
    let nd = to_ndjson(&triples);
    assert!(nd.contains("\"p\":\"rdfs:subClassOf\""));
    eprintln!("✓ Stage C: corpus collapsed 6 → {} families; ndjson seam:\n{nd}", sigs.len());
}
