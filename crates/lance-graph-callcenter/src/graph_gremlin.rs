//! `graph_gremlin` — a minimal Gremlin-style traversal over a [`GraphSnapshot`].
//!
//! The pure-Rust contrast to the DataFusion node/edge tables ([`crate::graph_table`],
//! `query-lite`): `g(&snap).v(&["id"]).out().to_vec()` walks the adjacency in the
//! snapshot with **zero SQL, zero DataFusion** — just the `source → target` edge
//! list. A "very basic Gremlin POC": `V` / `out` / `in_` / `out_e(label)` /
//! `in_e(label)` / `values_kind` / `to_vec` / `count`.
//!
//! SurrealQL graph traversal lowers to the SAME steps — `->edge->` ≈ [`out`],
//! `<-edge<-` ≈ [`in_`], `->edge(WHERE ...)->` ≈ [`out_e`] — so this doubles as
//! the SurrealQL traversal kernel over the family-adapter graph. Both consume the
//! `GraphSnapshot` the SoA projector ([`lance_graph_contract::soa_graph`])
//! produces from the 32-byte node head; the family nodes are the stable hubs the
//! traversal hops through.
//!
//! [`out`]: Traversal::out
//! [`in_`]: Traversal::in_
//! [`out_e`]: Traversal::out_e

use lance_graph_contract::graph_render::GraphSnapshot;
use std::collections::HashSet;

/// The Gremlin `g` — a traversal source bound to one graph snapshot.
pub struct GraphTraversalSource<'a> {
    snap: &'a GraphSnapshot,
}

/// `g(&snap)` — open a traversal over the snapshot (the Gremlin `g`).
pub fn g(snap: &GraphSnapshot) -> GraphTraversalSource<'_> {
    GraphTraversalSource { snap }
}

impl<'a> GraphTraversalSource<'a> {
    /// `g.V(ids)` — seed the traversal at the given vertex ids. An empty slice is
    /// `g.V()` (all vertices).
    pub fn v(&self, ids: &[&str]) -> Traversal<'a> {
        let current: Vec<String> = if ids.is_empty() {
            self.snap.nodes.iter().map(|n| n.id.clone()).collect()
        } else {
            ids.iter().map(|s| s.to_string()).collect()
        };
        Traversal {
            snap: self.snap,
            current,
        }
    }
}

/// An in-flight traversal: the multiset of vertex ids currently held, plus the
/// snapshot to hop over. Steps consume `self` and return `Self` (Gremlin fluent).
pub struct Traversal<'a> {
    snap: &'a GraphSnapshot,
    current: Vec<String>,
}

impl<'a> Traversal<'a> {
    /// `out()` — follow outgoing edges (`source ∈ current → target`), any label.
    #[must_use]
    pub fn out(mut self) -> Self {
        self.step(None, true);
        self
    }

    /// `out(label)` — outgoing edges whose label equals `label`.
    #[must_use]
    pub fn out_e(mut self, label: &str) -> Self {
        self.step(Some(label), true);
        self
    }

    /// `in()` — follow incoming edges (`target ∈ current → source`), any label.
    #[must_use]
    pub fn in_(mut self) -> Self {
        self.step(None, false);
        self
    }

    /// `in(label)` — incoming edges whose label equals `label`.
    #[must_use]
    pub fn in_e(mut self, label: &str) -> Self {
        self.step(Some(label), false);
        self
    }

    fn step(&mut self, label: Option<&str>, outgoing: bool) {
        let cur: HashSet<&str> = self.current.iter().map(String::as_str).collect();
        let mut next: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for e in &self.snap.edges {
            if let Some(l) = label {
                if e.label != l {
                    continue;
                }
            }
            let (from, to) = if outgoing {
                (&e.source, &e.target)
            } else {
                (&e.target, &e.source)
            };
            if cur.contains(from.as_str()) && seen.insert(to.clone()) {
                next.push(to.clone());
            }
        }
        self.current = next;
    }

    /// Terminal `values("kind")` — project the `kind` of each reached vertex
    /// (skips ids that are not present as nodes, e.g. a dangling adapter target).
    #[must_use]
    pub fn values_kind(&self) -> Vec<String> {
        self.current
            .iter()
            .filter_map(|id| {
                self.snap
                    .nodes
                    .iter()
                    .find(|n| &n.id == id)
                    .map(|n| n.kind.clone())
            })
            .collect()
    }

    /// Terminal `toList()` — the vertex ids currently reached.
    #[must_use]
    pub fn to_vec(self) -> Vec<String> {
        self.current
    }

    /// Terminal `count()` — number of vertices reached.
    #[must_use]
    pub fn count(self) -> usize {
        self.current.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::graph_render::{RenderEdge, RenderNode};

    fn node(id: &str, kind: &str) -> RenderNode {
        RenderNode {
            id: id.to_string(),
            label: id.to_string(),
            kind: kind.to_string(),
            confidence: 1.0,
            props: vec![],
        }
    }
    fn edge(source: &str, target: &str, label: &str) -> RenderEdge {
        RenderEdge {
            source: source.to_string(),
            target: target.to_string(),
            label: label.to_string(),
            frequency: 1.0,
            confidence: 1.0,
            inferred: false,
        }
    }

    fn sample() -> GraphSnapshot {
        // A -knows-> B -knows-> C ; A -member-of-> family:00000a
        GraphSnapshot {
            nodes: vec![
                node("A", "Person"),
                node("B", "Person"),
                node("C", "Person"),
                node("family:00000a", "Family"),
            ],
            edges: vec![
                edge("A", "B", "knows"),
                edge("B", "C", "knows"),
                edge("A", "family:00000a", "member-of"),
            ],
            inferences: vec![],
            contradictions: vec![],
            timestamp: 0,
        }
    }

    #[test]
    fn out_follows_outgoing_edges() {
        let s = sample();
        assert_eq!(g(&s).v(&["A"]).out().to_vec(), vec!["B", "family:00000a"]);
    }

    #[test]
    fn in_follows_incoming_edges() {
        let s = sample();
        assert_eq!(g(&s).v(&["B"]).in_().to_vec(), vec!["A".to_string()]);
    }

    #[test]
    fn out_e_filters_by_label() {
        let s = sample();
        // g.V("A").out("knows") = [B]; out("member-of") = [family:00000a]
        assert_eq!(g(&s).v(&["A"]).out_e("knows").to_vec(), vec!["B"]);
        assert_eq!(
            g(&s).v(&["A"]).out_e("member-of").to_vec(),
            vec!["family:00000a"]
        );
    }

    #[test]
    fn two_hop_traversal() {
        let s = sample();
        // g.V("A").out("knows").out("knows") = [C]
        assert_eq!(
            g(&s).v(&["A"]).out_e("knows").out_e("knows").to_vec(),
            vec!["C"]
        );
    }

    #[test]
    fn values_kind_projects_node_property() {
        let s = sample();
        // A's "member-of" neighbour is the family hub → kind "Family".
        assert_eq!(g(&s).v(&["A"]).out_e("member-of").values_kind(), vec!["Family"]);
    }

    #[test]
    fn unknown_label_yields_empty() {
        let s = sample();
        assert_eq!(g(&s).v(&["A"]).out_e("nope").count(), 0);
    }

    #[test]
    fn v_with_no_seed_is_all_vertices() {
        let s = sample();
        assert_eq!(g(&s).v(&[]).count(), 4);
    }
}
