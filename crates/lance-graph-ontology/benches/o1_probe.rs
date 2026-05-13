//! D-CASCADE-V1-11 — O(1) probe.
//! classification: bare-metal
//!
//! `name -> registry-row` HashMap p99 vs SPARQL-equivalent linear-scan p99.
//! Target: >= 100x speedup. Sibling agent-cascade-cols (Wave 3, parallel) ships
//! the `cam_pq_code: [u8; 6]` column on `MappingRow`; until that lands we
//! measure the registry HashMap baseline (which IS O(1)).

use lance_graph_contract::property::{Marking, Schema};
use lance_graph_ontology::{
    namespace::OgitUri,
    proposal::{MappingProposal, MappingProposalKind},
    OntologyRegistry,
};
use std::time::Instant;

const N_ROWS: usize = 1024;
const N_ITERS: usize = 5_000;

fn make_registry(n: usize) -> (OntologyRegistry, Vec<String>) {
    let reg = OntologyRegistry::new_in_memory();
    let mut names = Vec::with_capacity(n);
    for i in 0..n {
        let uri = format!("ogit.Bench:Entity{i}");
        let parsed = OgitUri::parse(&uri).unwrap();
        let ns = parsed.namespace().unwrap().to_string();
        let name = parsed.name().unwrap().to_string();
        reg.append_mapping(MappingProposal {
            public_name: uri.clone(),
            bridge_id: "ogit".into(),
            ogit_uri: parsed,
            namespace: ns,
            kind: MappingProposalKind::Entity {
                schema: Schema::builder(Box::leak(name.into_boxed_str()))
                    .required("id")
                    .build(),
            },
            marking: Marking::Internal,
            confidence: 1.0,
            source_uri: format!("bench://{uri}"),
            checksum: format!("ck-{i}"),
            created_by: "bench".into(),
        })
        .unwrap();
        names.push(uri);
    }
    (reg, names)
}

fn p99(samples: &mut [u128]) -> u128 {
    samples.sort_unstable();
    let idx = ((samples.len() as f64) * 0.99).round() as usize;
    samples[idx.min(samples.len() - 1)]
}

fn main() {
    let (reg, names) = make_registry(N_ROWS);
    for _ in 0..1024 {
        let _ = reg.resolve_uri(&names[0]);
    }

    let mut reg_samples = Vec::with_capacity(N_ITERS);
    for i in 0..N_ITERS {
        let start = Instant::now();
        let _ = reg.resolve_uri(&names[i % names.len()]);
        reg_samples.push(start.elapsed().as_nanos());
    }
    // SPARQL-equivalent proxy: linear scan over enumerate(namespace) — same
    // shape `SELECT ?o WHERE { :name ogit:hasCamPqCode ?o }` would walk.
    let mut sparql_samples = Vec::with_capacity(N_ITERS);
    for i in 0..N_ITERS {
        let key = &names[i % names.len()];
        let start = Instant::now();
        let rows = reg.enumerate("Bench");
        let _ = rows.iter().find(|r| r.ogit_uri.as_str() == key);
        sparql_samples.push(start.elapsed().as_nanos());
    }
    let p_reg = p99(&mut reg_samples);
    let p_sparql = p99(&mut sparql_samples);
    let ratio = p_sparql as f64 / p_reg.max(1) as f64;
    println!("rows={N_ROWS} iters={N_ITERS}");
    println!("registry  p99 = {p_reg} ns");
    println!("sparql_px p99 = {p_sparql} ns");
    println!("ratio (sparql/registry) = {ratio:.1}x");
    println!(
        "target >= 100x: {}",
        if ratio >= 100.0 { "PASS" } else { "FAIL" }
    );
}
