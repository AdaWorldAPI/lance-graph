//! Run the aiwar OSINT family-node POC on the real graph.
//!
//! ```text
//! cargo run -p lance-graph-contract --example aiwar_family_poc -- [path]
//! ```
//! Default path: `/tmp/aiwar_graph.json` — download from
//! <https://raw.githubusercontent.com/AdaWorldAPI/aiwar-neo4j-harvest/main/data/aiwar_graph.json>.
//!
//! Prints the OSINT family **class view** + the projected Gotham `GraphSnapshot`
//! (family nodes = entity categories, members hang off them). q2's cockpit wires
//! the same snapshot to the Quadro-2 visual.

use lance_graph_contract::aiwar::{aiwar_node_rows, AiwarClassView};
use lance_graph_contract::literal_graph::ingest_aiwar_json;
use lance_graph_contract::soa_graph::{project_snapshot, OSINT_GOTHAM};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/aiwar_graph.json".to_string());
    let json = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("could not read {path}: {e}");
            eprintln!(
                "download: https://raw.githubusercontent.com/AdaWorldAPI/aiwar-neo4j-harvest/main/data/aiwar_graph.json"
            );
            std::process::exit(1);
        }
    };
    let g = ingest_aiwar_json(&json).expect("parse aiwar_graph.json");
    let view = AiwarClassView::from_graph(&g);
    let rows = aiwar_node_rows(&g);
    let snap = project_snapshot(&rows, &OSINT_GOTHAM);

    let family_nodes = snap
        .nodes
        .iter()
        .filter(|n| n.kind == "Family" || n.kind == "Anchor")
        .count();
    let members = snap.nodes.len() - family_nodes;

    println!("aiwar OSINT family-node POC ({path}):");
    println!(
        "  ingested {} entities / {} edges; {} categories (the class view)",
        g.node_count(),
        g.edge_count(),
        view.len()
    );
    println!(
        "  projected GraphSnapshot: {} nodes ({members} members + {family_nodes} family hubs), {} edges",
        snap.nodes.len(),
        snap.edges.len()
    );
    println!("  family nodes (category ⇒ id, members):");
    for (cat, fam) in view.categories() {
        let count = snap
            .nodes
            .iter()
            .find(|n| n.id == format!("family:{fam:06x}"))
            .and_then(|n| {
                n.props
                    .iter()
                    .find(|(k, _)| k == "members")
                    .map(|(_, v)| v.clone())
            })
            .unwrap_or_default();
        println!("    {fam:>3}  {cat:<30} {count} members");
    }
}
