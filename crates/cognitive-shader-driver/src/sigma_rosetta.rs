//! Sigma Rosetta — 17D sparse encoding for the cognitive shader driver.
//!
//! # The Rosetta Stone (Hamming-Native Language)
//!
//! Translates between human-readable qualia/verbs and the internal
//! fingerprint/BindSpace columns. This is where the LLM token stream
//! meets the cognitive shader substrate.
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │             SIGMA ROSETTA 17D                    │
//! │       (Sparse encoding for the Membrane)         │
//! ├─────────────────────────────────────────────────┤
//! │  64 qualia glyphs × 144 verbs = 9,216 atoms      │
//! │           │                                      │
//! │           ▼                                      │
//! │  17D sparse encoding: 16 bands + 1 glyph         │
//! │  (vs QPL 17D: convergence observables)           │
//! │           │                                      │
//! │           ▼                                      │
//! │  Fingerprint<256> (same BindSpace atom)          │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # Two Views of the Same 17D Space
//!
//! - **QPL 17D** (thinking-engine qualia.rs): 17 convergence observables.
//!   See that module for the canonical band list.
//!
//! - **Sigma Rosetta 17D** (this module): 16 band slots +
//!   glyph composition. See `SIGMA_BAND_NAMES` below for the slot labels.
//!
//! These are CMYK vs RGB of the same qualia space (see `engine_bridge.rs`
//! for the observer-frame explanation).

// ═══════════════════════════════════════════════════════════════════════════
// 12 verb roots × 12 tenses = 144 universal grammar verbs
// ═══════════════════════════════════════════════════════════════════════════

pub const VERB_ROOTS: [&str; 12] = [
    "sense",     // 0: perceive
    "feel",      // 1: experience affect
    "think",     // 2: cognize
    "act",       // 3: execute
    "become",    // 4: transform identity
    "hold",      // 5: maintain state
    "release",   // 6: let go
    "create",    // 7: generate novel
    "transform", // 8: change form
    "connect",   // 9: build relation
    "separate",  // 10: establish boundary
    "integrate", // 11: unify parts
];

pub const VERB_TENSES: [&str; 12] = [
    "present", "past", "future", "continuous", "perfect", "pluperfect",
    "future_perfect", "habitual", "potential", "imperative", "subjunctive", "gerund",
];

pub const N_VERBS: usize = 144;

#[inline]
pub const fn verb_index(root: u8, tense: u8) -> u8 {
    (root % 12) * 12 + (tense % 12)
}

pub fn verb_root_name(root: u8) -> &'static str {
    VERB_ROOTS[(root % 12) as usize]
}

pub fn verb_tense_name(tense: u8) -> &'static str {
    VERB_TENSES[(tense % 12) as usize]
}

// ═══════════════════════════════════════════════════════════════════════════
// 64 qualia glyphs — canonical vocabulary (4 families × 16 slots).
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
pub struct QualiaGlyph {
    pub idx: u8,
    pub name: &'static str,
    pub emoji: &'static str,
    pub family: GlyphFamily,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlyphFamily {
    Core,       // 0-15: void, presence, warmth, flow, clarity...
    Relational, // 16-31: connection, trust, bond, holding...
    Cognitive,  // 32-47: focus, analysis, synthesis, memory...
    Somatic,    // 48-63: tension, pleasant, excited, fatigue...
}

pub const GLYPHS: [QualiaGlyph; 64] = [
    // Core (0-15)
    QualiaGlyph { idx: 0,  name: "void",          emoji: "🌑", family: GlyphFamily::Core },
    QualiaGlyph { idx: 1,  name: "presence",      emoji: "✨", family: GlyphFamily::Core },
    QualiaGlyph { idx: 2,  name: "warmth",        emoji: "🔥", family: GlyphFamily::Core },
    QualiaGlyph { idx: 3,  name: "flow",          emoji: "🌊", family: GlyphFamily::Core },
    QualiaGlyph { idx: 4,  name: "clarity",       emoji: "💎", family: GlyphFamily::Core },
    QualiaGlyph { idx: 5,  name: "grounding",     emoji: "🌳", family: GlyphFamily::Core },
    QualiaGlyph { idx: 6,  name: "expansion",     emoji: "🌌", family: GlyphFamily::Core },
    QualiaGlyph { idx: 7,  name: "contraction",   emoji: "🫧", family: GlyphFamily::Core },
    QualiaGlyph { idx: 8,  name: "resonance",     emoji: "🎵", family: GlyphFamily::Core },
    QualiaGlyph { idx: 9,  name: "dissonance",    emoji: "⚡", family: GlyphFamily::Core },
    QualiaGlyph { idx: 10, name: "wonder",        emoji: "🌸", family: GlyphFamily::Core },
    QualiaGlyph { idx: 11, name: "grief",         emoji: "🌧", family: GlyphFamily::Core },
    QualiaGlyph { idx: 12, name: "play",          emoji: "🎭", family: GlyphFamily::Core },
    QualiaGlyph { idx: 13, name: "stillness",     emoji: "🧘", family: GlyphFamily::Core },
    QualiaGlyph { idx: 14, name: "boundary",      emoji: "🚪", family: GlyphFamily::Core },
    QualiaGlyph { idx: 15, name: "emergence",     emoji: "🌱", family: GlyphFamily::Core },
    // Relational (16-31)
    QualiaGlyph { idx: 16, name: "connection",    emoji: "🤝", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 17, name: "distance",      emoji: "🌉", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 18, name: "trust",         emoji: "💜", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 19, name: "caution",       emoji: "🦔", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 20, name: "bond",          emoji: "🤝", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 21, name: "privacy",       emoji: "🔐", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 22, name: "giving",        emoji: "🎁", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 23, name: "receiving",     emoji: "🙏", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 24, name: "mirroring",     emoji: "🪞", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 25, name: "autonomy",      emoji: "🦋", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 26, name: "belonging",     emoji: "🏠", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 27, name: "solitude",      emoji: "🌙", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 28, name: "witnessing",    emoji: "👁", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 29, name: "being_seen",    emoji: "🌅", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 30, name: "holding",       emoji: "🫂", family: GlyphFamily::Relational },
    QualiaGlyph { idx: 31, name: "release",       emoji: "🕊", family: GlyphFamily::Relational },
    // Cognitive (32-47)
    QualiaGlyph { idx: 32, name: "focus",         emoji: "🎯", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 33, name: "diffuse",       emoji: "☁", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 34, name: "analysis",      emoji: "🔬", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 35, name: "synthesis",     emoji: "🧬", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 36, name: "certainty",     emoji: "✓", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 37, name: "uncertainty",   emoji: "❓", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 38, name: "memory",        emoji: "📜", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 39, name: "anticipation",  emoji: "🔮", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 40, name: "learning",      emoji: "📚", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 41, name: "forgetting",    emoji: "🍂", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 42, name: "love",          emoji: "💜", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 43, name: "creation",      emoji: "🎨", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 44, name: "destruction",   emoji: "💥", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 45, name: "transformation",emoji: "🦋", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 46, name: "persistence",   emoji: "⚓", family: GlyphFamily::Cognitive },
    QualiaGlyph { idx: 47, name: "adaptation",    emoji: "🌿", family: GlyphFamily::Cognitive },
    // Somatic (48-63)
    QualiaGlyph { idx: 48, name: "tension",       emoji: "💪", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 49, name: "relaxation",    emoji: "🌀", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 50, name: "energy",        emoji: "⚡", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 51, name: "fatigue",       emoji: "😴", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 52, name: "pleasant",      emoji: "🌺", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 53, name: "pain",          emoji: "🩹", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 54, name: "hunger",        emoji: "🍽", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 55, name: "satiation",     emoji: "😌", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 56, name: "excited",       emoji: "🔥", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 57, name: "numbness",      emoji: "❄", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 58, name: "breath",        emoji: "💨", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 59, name: "pulse",         emoji: "💓", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 60, name: "warmth_soma",   emoji: "☀", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 61, name: "cold",          emoji: "🧊", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 62, name: "softness",      emoji: "🪶", family: GlyphFamily::Somatic },
    QualiaGlyph { idx: 63, name: "hardness",      emoji: "🗿", family: GlyphFamily::Somatic },
];

pub fn glyph(idx: u8) -> &'static QualiaGlyph {
    &GLYPHS[(idx % 64) as usize]
}

pub fn glyph_by_name(name: &str) -> Option<&'static QualiaGlyph> {
    GLYPHS.iter().find(|g| g.name == name)
}

// ═══════════════════════════════════════════════════════════════════════════
// SigmaState — (qualia, verb, tau) triple, the LLM↔substrate unit
// ═══════════════════════════════════════════════════════════════════════════

/// Sigma state: qualia glyph + verb + τ temperature.
///
/// The core unit of communication between the LLM layer and the
/// cognitive shader substrate. Three u8s + one f32 = 7 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SigmaState {
    pub qualia_idx: u8,   // 0..63 (or 0..255 when extended)
    pub verb_idx: u8,     // 0..143 (root * 12 + tense)
    pub tau: f32,         // 0.0 = qualia only, 1.0 = fully verb-bound
    pub confidence: f32,  // 0.0..1.0
}

impl Default for SigmaState {
    fn default() -> Self {
        Self { qualia_idx: 0, verb_idx: 0, tau: 0.5, confidence: 1.0 }
    }
}

impl SigmaState {
    pub fn new(qualia_idx: u8, verb_idx: u8, tau: f32) -> Self {
        Self {
            qualia_idx,
            verb_idx: verb_idx % N_VERBS as u8,
            tau: tau.clamp(0.0, 1.0),
            confidence: 1.0,
        }
    }

    /// Is the state verb-dominated (action-infused)?
    pub fn is_verb_bound(&self) -> bool { self.tau >= 0.5 }

    pub fn qualia_name(&self) -> &'static str {
        glyph(self.qualia_idx).name
    }

    pub fn verb_name(&self) -> String {
        format!("{}_{}", verb_root_name(self.verb_idx / 12), verb_tense_name(self.verb_idx % 12))
    }

    /// Human-readable Σ(qualia, verb, τ=...) notation.
    pub fn describe(&self) -> String {
        format!("Σ({}, {}, τ={:.2})", self.qualia_name(), self.verb_name(), self.tau)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Triangle Gestalt — from agi-chat/kopfkino/unified-dto.ts
// ═══════════════════════════════════════════════════════════════════════════

/// Cognitive triangle: clarity (thinking) / warmth (feeling) / presence (body).
/// The SD of the three vertices drives the CollapseGate decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleGestalt {
    pub clarity: f32,   // top — cognition
    pub warmth: f32,    // left — emotion
    pub presence: f32,  // right — somatic
}

impl TriangleGestalt {
    pub fn new(clarity: f32, warmth: f32, presence: f32) -> Self {
        Self {
            clarity: clarity.clamp(0.0, 1.0),
            warmth: warmth.clamp(0.0, 1.0),
            presence: presence.clamp(0.0, 1.0),
        }
    }

    /// Centroid (barycentric, 2D).
    pub fn centroid(&self) -> (f32, f32) {
        let x = 0.5 * self.warmth + 1.0 * self.clarity + 0.0 * self.presence;
        let y = 0.866 * self.clarity;
        (x / 3.0, y / 3.0)
    }

    /// 0 = unbalanced, 1 = perfect equilateral.
    pub fn balance(&self) -> f32 {
        1.0 - self.std_dev() * 2.0
    }

    /// Standard deviation of the three vertices.
    pub fn std_dev(&self) -> f32 {
        let mean = (self.clarity + self.warmth + self.presence) / 3.0;
        let var = ((self.clarity - mean).powi(2)
            + (self.warmth - mean).powi(2)
            + (self.presence - mean).powi(2)) / 3.0;
        var.sqrt()
    }

    /// Compute gate (Flow/Hold/Block) from SD — matches shader driver thresholds.
    pub fn gate(&self) -> GestaltGate {
        let sd = self.std_dev();
        if sd < 0.15 { GestaltGate::Flow }
        else if sd < 0.35 { GestaltGate::Hold }
        else { GestaltGate::Block }
    }

    /// Area of the triangle (energy).
    pub fn area(&self) -> f32 {
        // Heron's formula on unit triangle scaled by (c+w+p)/3
        let s = (self.clarity + self.warmth + self.presence) / 3.0;
        s.max(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GestaltGate {
    Flow,
    Hold,
    Block,
}

// ═══════════════════════════════════════════════════════════════════════════
// Bridge: QPL-17D (convergence observables) ↔ Sigma-Rosetta-17D (felt bands)
// ═══════════════════════════════════════════════════════════════════════════

/// Sigma Rosetta 17D band order — canonical, experientially loaded.
pub const SIGMA_BAND_NAMES: [&str; 16] = [
    "warmth", "presence", "openness", "sovereignty",
    "tenderness", "groundedness", "activation", "clarity",
    "coherence", "closeness", "surrender", "seeking",
    "awakening", "bond", "synthesis", "resonance",
];

/// Convert QPL-17D (thinking-engine qualia) → Sigma-Rosetta 16 bands.
/// Dim 17 in Sigma is reserved for glyph composition (encoded separately).
///
/// QPL indices 0..16 — see thinking-engine `qualia::DIMS_17D` for labels.
pub fn qpl_to_sigma(qpl: &[f32; 17]) -> [f32; 16] {
    [
        qpl[3],                        // 0: warmth       ← warmth
        qpl[11],                       // 1: presence     ← presence
        qpl[15],                       // 2: openness     ← expansion
        (1.0 - qpl[5]).clamp(0.0, 1.0),// 3: sovereignty  ← 1 - boundary
        qpl[10],                       // 4: tenderness   ← qpl[10]
        qpl[14],                       // 5: groundedness ← groundedness
        qpl[0],                        // 6: activation   ← arousal
        qpl[4],                        // 7: clarity      ← clarity
        qpl[9],                        // 8: coherence    ← coherence
        qpl[10],                       // 9: closeness    ← qpl[10]
        qpl[13],                       // 10: surrender   ← receptivity
        qpl[1],                        // 11: seeking     ← valence (positive seeks)
        qpl[6],                        // 12: awakening   ← depth
        qpl[12],                       // 13: bond        ← assertion (relational)
        qpl[16],                       // 14: synthesis   ← integration
        (1.0 - qpl[2]).clamp(0.0, 1.0),// 15: resonance   ← 1 - tension
    ]
}

/// Inverse: Sigma 16 bands → QPL 17D (some dims approximate).
pub fn sigma_to_qpl(sigma: &[f32; 16]) -> [f32; 17] {
    let mut qpl = [0.0f32; 17];
    qpl[0]  = sigma[6];                      // qpl[0]  ← activation
    qpl[1]  = sigma[11];                     // qpl[1]  ← seeking
    qpl[2]  = 1.0 - sigma[15];               // qpl[2]  ← 1 - resonance
    qpl[3]  = sigma[0];                      // qpl[3]  ← warmth
    qpl[4]  = sigma[7];                      // qpl[4]  ← clarity
    qpl[5]  = 1.0 - sigma[3];                // qpl[5]  ← 1 - sovereignty
    qpl[6]  = sigma[12];                     // qpl[6]  ← awakening
    qpl[9]  = sigma[8];                      // qpl[9]  ← coherence
    qpl[10] = sigma[9];                      // qpl[10] ← closeness
    qpl[11] = sigma[1];                      // qpl[11] ← presence
    qpl[12] = sigma[13];                     // qpl[12] ← bond
    qpl[13] = sigma[10];                     // qpl[13] ← surrender
    qpl[14] = sigma[5];                      // qpl[14] ← groundedness
    qpl[15] = sigma[2];                      // qpl[15] ← openness
    qpl[16] = sigma[14];                     // qpl[16] ← synthesis
    // qpl[7] + qpl[8] are convergence-only — no sigma mapping
    qpl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verb_index_wraps() {
        assert_eq!(verb_index(0, 0), 0);
        assert_eq!(verb_index(1, 0), 12);
        assert_eq!(verb_index(11, 11), 143);
        assert_eq!(verb_index(12, 0), 0); // wraps
    }

    #[test]
    fn glyph_family_ranges_correct() {
        assert_eq!(GLYPHS[0].family, GlyphFamily::Core);
        assert_eq!(GLYPHS[20].family, GlyphFamily::Relational);
        assert_eq!(GLYPHS[35].family, GlyphFamily::Cognitive);
        assert_eq!(GLYPHS[56].family, GlyphFamily::Somatic);
    }

    #[test]
    fn sigma_state_describe() {
        let s = SigmaState::new(4, verb_index(2, 0), 0.8); // clarity + think_present
        let d = s.describe();
        assert!(d.contains("clarity"));
        assert!(d.contains("think_present"));
        assert!(s.is_verb_bound());
    }

    #[test]
    fn triangle_balanced_is_flow() {
        let t = TriangleGestalt::new(0.5, 0.5, 0.5);
        assert_eq!(t.gate(), GestaltGate::Flow);
        assert!(t.balance() > 0.95);
    }

    #[test]
    fn triangle_unbalanced_blocks() {
        let t = TriangleGestalt::new(1.0, 0.0, 0.5);
        assert_eq!(t.gate(), GestaltGate::Block, "sd = {}", t.std_dev());
    }

    #[test]
    fn qpl_sigma_roundtrip_preserves_shared_dims() {
        let mut qpl = [0.0f32; 17];
        qpl[0] = 0.8;
        qpl[3] = 0.6;
        qpl[4] = 0.9;
        qpl[11] = 0.7;
        let sigma = qpl_to_sigma(&qpl);
        let back = sigma_to_qpl(&sigma);
        assert!((back[0] - 0.8).abs() < 1e-6, "qpl[0] should roundtrip");
        assert!((back[3] - 0.6).abs() < 1e-6, "qpl[3] should roundtrip");
        assert!((back[4] - 0.9).abs() < 1e-6, "qpl[4] should roundtrip");
        assert!((back[11] - 0.7).abs() < 1e-6, "qpl[11] should roundtrip");
    }

    #[test]
    fn glyph_by_name_lookup() {
        assert_eq!(glyph_by_name("warmth").unwrap().idx, 2);
        assert_eq!(glyph_by_name("bond").unwrap().idx, 20);
        assert_eq!(glyph_by_name("focus").unwrap().idx, 32);
        assert!(glyph_by_name("nonexistent").is_none());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Σ13 — Interaction Dome (multi-agent kinematics)
// ═══════════════════════════════════════════════════════════════════════════
//
// Σ13 sits above Σ12. Where Σ12 is single-agent (one qualia × one verb × one τ),
// Σ13 is the kinematic relationship between agents interacting in the blackboard.
//
// Harvested from bighorn/AGI_STACK_CONSOLIDATION_PLAN.md:
//   "Σ13 - INTERACTION DOME — Kinematic relationships, multi-agent"
//
// An InteractionKinematic describes how agent A's SigmaState evolves
// relative to agent B's — two states plus a relative tension vector.

/// Σ13 — two-agent interaction kinematic.
///
/// This is the unit of multi-agent coordination. A blackboard sweep
/// produces a set of these for all active agent pairs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InteractionKinematic {
    /// Agent A's sigma state.
    pub a: SigmaState,
    /// Agent B's sigma state.
    pub b: SigmaState,
    /// Relative tension (0 = resonance, 1 = dissonance).
    pub tension: f32,
    /// Which predicate plane (0..7) best describes the interaction.
    pub predicate_plane: u8,
}

impl InteractionKinematic {
    pub fn new(a: SigmaState, b: SigmaState) -> Self {
        // Simple tension heuristic: combine verb distance and qualia distance.
        let v_diff = ((a.verb_idx as i16 - b.verb_idx as i16).abs() as f32) / 144.0;
        let q_diff = ((a.qualia_idx as i16 - b.qualia_idx as i16).abs() as f32) / 64.0;
        let tension = ((v_diff + q_diff) / 2.0).clamp(0.0, 1.0);

        // Predicate plane heuristic: same verb root → SUPPORTS (2),
        // opposite tau → CONTRADICTS (3), otherwise CAUSES (0).
        let predicate_plane = if a.verb_idx / 12 == b.verb_idx / 12 {
            2 // SUPPORTS
        } else if (a.tau - b.tau).abs() > 0.5 {
            3 // CONTRADICTS
        } else {
            0 // CAUSES
        };

        Self { a, b, tension, predicate_plane }
    }

    /// Resonance = 1 - tension.
    pub fn resonance(&self) -> f32 { 1.0 - self.tension }

    /// Is this pair aligned (low tension)?
    pub fn aligned(&self) -> bool { self.tension < 0.3 }
}

#[cfg(test)]
mod sigma13_tests {
    use super::*;

    #[test]
    fn same_state_is_full_resonance() {
        let s = SigmaState::new(4, verb_index(2, 0), 0.5);
        let k = InteractionKinematic::new(s, s);
        assert_eq!(k.tension, 0.0);
        assert!(k.aligned());
    }

    #[test]
    fn different_verb_root_has_higher_tension() {
        let a = SigmaState::new(4, verb_index(2, 0), 0.5); // think_present
        let b = SigmaState::new(4, verb_index(8, 0), 0.5); // transform_present
        let k = InteractionKinematic::new(a, b);
        assert!(k.tension > 0.0);
        assert_ne!(k.predicate_plane, 2, "not same-verb-root, should not be SUPPORTS");
    }

    #[test]
    fn opposite_tau_contradicts() {
        let a = SigmaState::new(4, verb_index(2, 0), 0.9);
        let b = SigmaState::new(4, verb_index(8, 0), 0.1);
        let k = InteractionKinematic::new(a, b);
        assert_eq!(k.predicate_plane, 3, "large tau diff → CONTRADICTS plane");
    }
}
