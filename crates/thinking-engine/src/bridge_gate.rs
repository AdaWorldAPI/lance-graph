//! `CognitiveBridgeGate` — re-exported from `lance_graph_contract::bridge_gate`.
//!
//! D-TEH-1 (`thinking-engine-harvest-closure-v1` W1, 2026-09-02): the
//! zero-dep gate contract now lives in the contract crate, all seven items
//! together, so `lance-graph-callcenter` no longer needs a path dependency on
//! this crate to implement it. This re-export keeps every existing
//! `thinking_engine::bridge_gate::*` path valid during the migration wave; the
//! gate's own unit tests moved with the items.
//!
//! What stays here is the engine-side proof that PURE ops (codebook lookup,
//! distance) never touch the gate — it calls this crate's lens modules, so it
//! belongs next to them, not in the contract.

pub use lance_graph_contract::bridge_gate::{
    auth_to_result, CognitiveAuthResult, CognitiveBridgeError, CognitiveBridgeGate,
    CognitiveOpKind, DenyAllGate, PassthroughGate,
};

#[cfg(test)]
mod tests {
    /// Pure-op test: confirm that pure ops (encode, distance lookup) are
    /// NOT routed through the gate. This is a documentation/design test —
    /// there is no gate call in those paths, so they succeed unconditionally.
    #[test]
    fn pure_ops_dont_touch_gate() {
        // Pure codebook lookup — gate is not called.
        let centroid = crate::jina_lens::jina_lookup(42);
        assert!(centroid < 256);

        let dist = crate::jina_lens::jina_distance(0, 1);
        let _ = dist; // pure math, no gate

        // BGE-M3 pure lookup
        let centroid2 = crate::bge_m3_lens::bge_m3_lookup(100);
        assert!(centroid2 < 256);

        // Reranker pure lookup
        let centroid3 = crate::reranker_lens::reranker_lookup(500);
        assert!(centroid3 < 256);
    }
}
