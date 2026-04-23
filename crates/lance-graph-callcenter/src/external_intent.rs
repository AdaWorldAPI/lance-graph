//! External intent and cognitive event row — the two BBB crossing types.
//!
//! - `ExternalIntent`: inbound — what an external consumer deposits at the gate.
//! - `CognitiveEventRow`: outbound — scalar-only projection of a committed cycle.
//!
//! Both types carry ONLY Arrow-scalar-compatible primitives (`u8`, `u16`, `u64`,
//! `bool`, `Vec<u8>`). VSA types, semiring types, and `[u64; 256]` fingerprints
//! never appear here — that is the BBB invariant, enforced at compile time.
//!
//! Plan: `.claude/plans/callcenter-membrane-v1.md` § 4 + § 10.9 + § 10.11

use lance_graph_contract::{
    external_membrane::{ExternalEventKind, ExternalRole},
    persona::RoutingHint,
};

use crate::dn_path::DnPath;

/// External consumer intent deposited at the callcenter gate.
///
/// All fields are Arrow-scalar-compatible. No VSA types, no semiring types.
/// The `body` bytes are the raw seed payload — they cross the gate as opaque
/// bytes and are interpreted as a seed INSIDE the substrate after the membrane
/// translates them into a `UnifiedStep`.
///
/// # BBB Invariant (§ 10.9 iron rule)
///
/// 1. Pass membrane — this struct IS the membrane crossing shape.
/// 2. Get a role — `role: ExternalRole` stamped at construction.
/// 3. Get a place — `dn: DnPath` is the deterministic address.
/// 4. Translate — `LanceMembrane::ingest()` converts this to `UnifiedStep`.
#[derive(Clone, Debug)]
pub struct ExternalIntent {
    /// Which external family is sending this event.
    pub role: ExternalRole,
    /// DN-addressed location (URL path parsed to 6 u64 hashes).
    pub dn: DnPath,
    /// Raw seed payload — opaque bytes, interpreted inside the substrate.
    pub body: Vec<u8>,
    /// Optional explicit routing to a specific card or family.
    pub routing: RoutingHint,
    /// Whether this triggers a new reasoning cycle, passive context, or outbound commit.
    pub kind: ExternalEventKind,
}

impl ExternalIntent {
    pub fn seed(role: ExternalRole, dn: DnPath, body: Vec<u8>) -> Self {
        Self {
            role,
            dn,
            body,
            routing: RoutingHint::default(),
            kind: ExternalEventKind::Seed,
        }
    }

    pub fn context(role: ExternalRole, dn: DnPath, body: Vec<u8>) -> Self {
        Self {
            role,
            dn,
            body,
            routing: RoutingHint::default(),
            kind: ExternalEventKind::Context,
        }
    }
}

/// Scalar-only projection of one committed `ShaderBus` cycle.
///
/// This is `LanceMembrane::Commit` — the type that crosses the BBB outbound.
/// Every field is an Arrow-scalar primitive (`u8`, `u16`, `u64`, `bool`).
///
/// # BBB Invariant
///
/// The cycle fingerprint (`[u64; 256]`, 2 KB) is NOT carried here. Its full
/// VSA state lives stack-side. Only two scalar identity words cross the gate:
/// `cycle_fp_hi = fingerprint[0]` and `cycle_fp_lo = fingerprint[255]`.
/// Together they identify the cycle without carrying the full binding.
///
/// Why this is compile-time safe: `CognitiveEventRow` contains no type that
/// fails `Send` or that would require `unsafe impl`. Arrow's `Array` trait
/// requires concrete primitive types — `Vsa10k`, `RoleKey`, and `[u64; 256]`
/// cannot be Arrow array elements without wrapping as `FixedSizeBinary`, which
/// is a deliberate re-encoding step, not an accidental leak.
///
/// # Schema (§ 4 of plan)
///
/// | Column        | Arrow type | Source                          |
/// |---------------|------------|---------------------------------|
/// | external_role | UInt8      | ExternalRole discriminant       |
/// | faculty_role  | UInt8      | FacultyRole discriminant        |
/// | expert_id     | UInt16     | stable_hash_u16(card_yaml)      |
/// | dialect       | UInt8      | query dialect (Phase B)         |
/// | scent         | UInt8      | CAM-PQ 1-byte key (Phase C)     |
/// | thinking      | UInt8      | MetaWord::thinking()            |
/// | awareness     | UInt8      | MetaWord::awareness()           |
/// | nars_f        | UInt8      | MetaWord::nars_f()              |
/// | nars_c        | UInt8      | MetaWord::nars_c()              |
/// | free_e        | UInt8      | MetaWord::free_e()              |
/// | cycle_fp_hi   | UInt64     | fingerprint[0]                  |
/// | cycle_fp_lo   | UInt64     | fingerprint[255]                |
/// | gate_commit    | Boolean    | GateDecision::is_flow()                                |
/// | gate_f         | UInt8      | free_e (redundant, for queries)                        |
/// | rationale_phase| Boolean    | true = Stage 1 rationale; false = Stage 2 answer       |
///
/// `rationale_phase` surfaces the MM-CoT stage split:
/// `FacultyDescriptor.inbound_style` (Stage 1) vs `outbound_style` (Stage 2).
/// Phase A: always false (single-stage emission). Phase B: wired from the
/// faculty dispatcher when `FacultyDescriptor::is_asymmetric()` is true.
#[derive(Clone, Debug, Default)]
pub struct CognitiveEventRow {
    // ── Identity columns (§ 4 schema, § 10.11 metadata address bus) ──
    pub external_role: u8,
    pub faculty_role:  u8,
    pub expert_id:     u16,
    pub dialect:       u8, // Phase B: set by polyglot front-end parser
    pub scent:         u8, // Phase C: full ZeckBF17→Base17→CAM-PQ chain
    // ── MetaWord fields ──
    pub thinking:  u8,
    pub awareness: u8,
    pub nars_f:    u8,
    pub nars_c:    u8,
    pub free_e:    u8,
    // ── Cycle identity (NOT the full 2 KB fingerprint) ──
    pub cycle_fp_hi: u64, // fingerprint[0]   — first word of cycle signature
    pub cycle_fp_lo: u64, // fingerprint[255]  — last word of cycle signature
    // ── Gate outcome ──
    pub gate_commit: bool, // true = F < 0.2 (Flow); false = Hold or Block
    pub gate_f:      u8,   // free_e at gate time (for SQL filter `WHERE gate_f < 50`)
    // ── MM-CoT stage (DU-4, plan unified-integration-v1.md § DU-4) ──
    pub rationale_phase: bool, // Phase B: true = inbound_style stage (rationale)
}
