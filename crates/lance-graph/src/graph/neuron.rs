//! NeuronPrint: 6D holographic representation of a single neuron's behavior.
//!
//! Each neuron (layer i, feature j) has 6 roles in the transformer:
//!   Q = how it queries        (34 bytes)
//!   K = what it matches       (34 bytes)
//!   V = what it retrieves     (34 bytes)
//!   Gate = whether it fires   (34 bytes)
//!   Up = how it amplifies     (34 bytes)
//!   Down = how it compresses  (34 bytes)
//!
//! Total: 204 bytes per neuron. Holographic: bundle all 6 → 34 bytes.
//! The CAM index (row_idx) aligns all 6 tables — same row = same feature.
//!
//! Three constructs:
//!   NeuronPrint  — what a neuron IS (the object, 204 bytes)
//!   NeuronQuery  — how you ASK it (the query, selective role probing)
//!   NeuronTrace  — how it REASONS (the thinking, NARS truth from role ratios)

use ndarray::hpc::bgz17_bridge::Base17;

// ─── Object: what a neuron IS ───────────────────────────────────────────────

/// Complete 6D representation of a single neuron at (layer, feature).
/// 204 bytes. Each field is a 34-byte Base17 vector.
#[derive(Clone, Debug)]
pub struct NeuronPrint {
    /// Layer index in the model.
    pub layer: u16,
    /// Feature/row index within the layer.
    pub feature: u32,
    /// Query projection: how this neuron queries.
    pub q: Base17,
    /// Key projection: what this neuron matches.
    pub k: Base17,
    /// Value projection: what this neuron retrieves.
    pub v: Base17,
    /// Gate projection: whether this neuron fires (SwiGLU gate).
    pub gate: Base17,
    /// Up projection: how this neuron amplifies.
    pub up: Base17,
    /// Down projection: how this neuron compresses.
    pub down: Base17,
}

impl NeuronPrint {
    /// Bundle all 6 roles into a single 34-byte holographic fingerprint.
    /// The gestalt contains all roles in superposition.
    pub fn bundle(&self) -> Base17 {
        let mut dims = [0i32; 17];
        for d in 0..17 {
            dims[d] = self.q.dims[d] as i32
                + self.k.dims[d] as i32
                + self.v.dims[d] as i32
                + self.gate.dims[d] as i32
                + self.up.dims[d] as i32
                + self.down.dims[d] as i32;
        }
        let mut out = [0i16; 17];
        for d in 0..17 {
            out[d] = (dims[d] / 6).clamp(-32768, 32767) as i16;
        }
        Base17 { dims: out }
    }

    /// Attention fingerprint: Q ⊕ K (what this neuron attends to).
    pub fn attention(&self) -> Base17 {
        self.q.xor_bind(&self.k)
    }

    /// Retrieval fingerprint: K ⊕ V (what this neuron retrieves when matched).
    pub fn retrieval(&self) -> Base17 {
        self.k.xor_bind(&self.v)
    }

    /// MLP fingerprint: Gate ⊕ Up ⊕ Down (the nonlinear transform).
    pub fn mlp(&self) -> Base17 {
        self.gate.xor_bind(&self.up).xor_bind(&self.down)
    }

    /// Byte size of the 6 Base17 role vectors (payload only, excludes layer/feature metadata).
    pub const PAYLOAD_SIZE: usize = 6 * 34; // 204
}

// ─── Query: how you ASK a neuron ────────────────────────────────────────────

/// Selective probe into neuron roles. Set the roles you want to query.
/// None = wildcard (don't constrain this role).
#[derive(Clone, Debug, Default)]
pub struct NeuronQuery {
    /// Constrain layer (None = any layer).
    pub layer: Option<u16>,
    /// Constrain feature (None = any feature).
    pub feature: Option<u32>,
    /// Query vector for Q-role (None = don't probe Q).
    pub q: Option<Base17>,
    /// Query vector for K-role (None = don't probe K).
    pub k: Option<Base17>,
    /// Query vector for V-role (None = don't probe V).
    pub v: Option<Base17>,
    /// Query vector for Gate-role (None = don't probe Gate).
    pub gate: Option<Base17>,
    /// Query vector for Up-role (None = don't probe Up).
    pub up: Option<Base17>,
    /// Query vector for Down-role (None = don't probe Down).
    pub down: Option<Base17>,
}

impl NeuronQuery {
    /// "What does this query attend to?" — probe Q against K store.
    pub fn attention(q: Base17) -> Self {
        NeuronQuery { q: Some(q), ..Default::default() }
    }

    /// "What is retrieved for this key?" — probe K against V store.
    pub fn retrieval(k: Base17) -> Self {
        NeuronQuery { k: Some(k), ..Default::default() }
    }

    /// "Does this feature fire?" — probe Gate.
    pub fn gating(gate: Base17) -> Self {
        NeuronQuery { gate: Some(gate), ..Default::default() }
    }

    /// "What does layer N do?" — constrain to a specific layer.
    pub fn at_layer(mut self, layer: u16) -> Self {
        self.layer = Some(layer);
        self
    }

    /// Score a NeuronPrint against this query. Lower = better match.
    /// Only active (Some) roles contribute to the score.
    pub fn score(&self, neuron: &NeuronPrint) -> u32 {
        let mut total = 0u32;
        let mut count = 0u32;
        if let Some(ref q) = self.q { total += q.l1(&neuron.q); count += 1; }
        if let Some(ref k) = self.k { total += k.l1(&neuron.k); count += 1; }
        if let Some(ref v) = self.v { total += v.l1(&neuron.v); count += 1; }
        if let Some(ref g) = self.gate { total += g.l1(&neuron.gate); count += 1; }
        if let Some(ref u) = self.up { total += u.l1(&neuron.up); count += 1; }
        if let Some(ref d) = self.down { total += d.l1(&neuron.down); count += 1; }
        if count > 0 { total / count } else { u32::MAX }
    }

    /// How many roles are active in this query.
    pub fn active_roles(&self) -> u8 {
        [&self.q, &self.k, &self.v, &self.gate, &self.up, &self.down]
            .iter()
            .filter(|r| r.is_some())
            .count() as u8
    }

    /// Pearl-like mask: which roles are active (6-bit).
    /// Bit 0=Q, 1=K, 2=V, 3=Gate, 4=Up, 5=Down.
    pub fn role_mask(&self) -> u8 {
        let mut mask = 0u8;
        if self.q.is_some() { mask |= 1 << 0; }
        if self.k.is_some() { mask |= 1 << 1; }
        if self.v.is_some() { mask |= 1 << 2; }
        if self.gate.is_some() { mask |= 1 << 3; }
        if self.up.is_some() { mask |= 1 << 4; }
        if self.down.is_some() { mask |= 1 << 5; }
        mask
    }
}

// ─── Thinking: how a neuron REASONS ─────────────────────────────────────────

/// NARS truth values derived from the 6 role ratios.
/// The MLP roles (Gate/Up/Down) encode causal structure.
#[derive(Clone, Debug)]
pub struct NeuronTrace {
    /// NARS frequency: P(fires) derived from Gate activation.
    /// gate_magnitude / max_magnitude → [0, 1].
    pub frequency: f32,
    /// NARS confidence: Up/Down ratio → evidence strength.
    /// High Up + low Down = strong positive evidence.
    /// Low Up + high Down = strong compression (less evidence).
    pub confidence: f32,
    /// Attention strength: Q·K alignment (L1 distance, inverted).
    /// Low distance = strong attention = this neuron activates.
    pub attention: f32,
    /// Retrieval coherence: K·V alignment.
    /// Low distance = coherent retrieval (what's stored matches what's keyed).
    pub coherence: f32,
    /// NARS expectation: c * (f - 0.5) + 0.5.
    pub expectation: f32,
}

impl NeuronTrace {
    /// Derive NARS truth from a NeuronPrint.
    pub fn from_neuron(n: &NeuronPrint) -> Self {
        // Gate magnitude → frequency (how often this neuron fires)
        let gate_mag = n.gate.dims.iter().map(|d| (*d as f32).abs()).sum::<f32>();
        let max_mag = 17.0 * 32768.0;
        let frequency = (gate_mag / max_mag).clamp(0.0, 1.0);

        // Up/Down ratio → confidence
        let up_mag = n.up.dims.iter().map(|d| (*d as f32).abs()).sum::<f32>();
        let down_mag = n.down.dims.iter().map(|d| (*d as f32).abs()).sum::<f32>().max(1.0);
        let confidence = (up_mag / (up_mag + down_mag)).clamp(0.0, 0.99);

        // Q·K alignment → attention strength
        let qk_dist = n.q.l1(&n.k) as f32;
        let attention = 1.0 - (qk_dist / max_mag).clamp(0.0, 1.0);

        // K·V alignment → retrieval coherence
        let kv_dist = n.k.l1(&n.v) as f32;
        let coherence = 1.0 - (kv_dist / max_mag).clamp(0.0, 1.0);

        let expectation = confidence * (frequency - 0.5) + 0.5;

        NeuronTrace { frequency, confidence, attention, coherence, expectation }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_neuron(layer: u16, feature: u32, base_val: i16) -> NeuronPrint {
        NeuronPrint {
            layer,
            feature,
            q: Base17 { dims: [base_val; 17] },
            k: Base17 { dims: [base_val + 10; 17] },
            v: Base17 { dims: [base_val + 20; 17] },
            gate: Base17 { dims: [base_val + 100; 17] },
            up: Base17 { dims: [base_val + 50; 17] },
            down: Base17 { dims: [base_val + 30; 17] },
        }
    }

    #[test]
    fn test_neuron_bundle() {
        let n = make_neuron(0, 0, 100);
        let b = n.bundle();
        // Average of 100, 110, 120, 200, 150, 130 = 135
        assert_eq!(b.dims[0], 135);
    }

    #[test]
    fn test_neuron_payload_size() {
        assert_eq!(NeuronPrint::PAYLOAD_SIZE, 204);
    }

    #[test]
    fn test_query_attention() {
        let q = NeuronQuery::attention(Base17 { dims: [100; 17] });
        assert_eq!(q.active_roles(), 1);
        assert_eq!(q.role_mask(), 0b000001); // Q only
    }

    #[test]
    fn test_query_score() {
        let n = make_neuron(0, 0, 100);
        // Query that matches Q exactly
        let q_exact = NeuronQuery::attention(Base17 { dims: [100; 17] });
        let score_exact = q_exact.score(&n);
        // Query that's far from Q
        let q_far = NeuronQuery::attention(Base17 { dims: [10000; 17] });
        let score_far = q_far.score(&n);
        assert!(score_exact < score_far, "exact match should score lower (closer)");
    }

    #[test]
    fn test_query_multi_role() {
        let q = NeuronQuery {
            q: Some(Base17 { dims: [100; 17] }),
            k: Some(Base17 { dims: [200; 17] }),
            ..Default::default()
        };
        assert_eq!(q.active_roles(), 2);
        assert_eq!(q.role_mask(), 0b000011); // Q + K
    }

    #[test]
    fn test_trace_from_neuron() {
        let n = make_neuron(5, 42, 100);
        let t = NeuronTrace::from_neuron(&n);
        assert!(t.frequency > 0.0);
        assert!(t.confidence > 0.0 && t.confidence < 1.0);
        assert!(t.attention > 0.0); // Q and K are close (only differ by 10)
        assert!(t.expectation > 0.0 && t.expectation < 1.0);
    }

    #[test]
    fn test_high_gate_high_frequency() {
        let mut n = make_neuron(0, 0, 0);
        n.gate = Base17 { dims: [30000; 17] }; // high gate
        let t = NeuronTrace::from_neuron(&n);
        assert!(t.frequency > 0.8, "high gate should mean high frequency: {}", t.frequency);
    }

    #[test]
    fn test_attention_xor_bind() {
        let n = make_neuron(0, 0, 100);
        let attn = n.attention(); // Q ⊕ K
        // Should be non-zero (Q ≠ K)
        assert!(attn.dims.iter().any(|d| *d != 0));
    }

    #[test]
    fn test_query_at_layer() {
        let q = NeuronQuery::attention(Base17 { dims: [100; 17] }).at_layer(15);
        assert_eq!(q.layer, Some(15));
    }
}
