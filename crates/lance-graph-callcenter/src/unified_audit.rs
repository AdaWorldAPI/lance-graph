//! Merkle-chained audit log for `UnifiedBridge::authorize()` decisions
//! (per `.claude/plans/super-domain-rbac-tenancy-v1.md` §13.3 / D-SDR-4).
//!
//! Each `UnifiedBridge::authorize()` call that materially gates access
//! (Deny / Escalate / Audit-required Allow) emits one `UnifiedAuditEvent`
//! through an `AuditSink` (see `crate::audit_sink`). Events form a chain: the merkle root of
//! event N includes the merkle root of event N-1 plus a per-super-domain
//! `merkle_salt` (§13.4 hard-lock — cross-domain audit logs are
//! unlinkable). Tampering with any past event is detectable by chain
//! re-verification.
//!
//! ## Hot path
//!
//! ```text
//! UnifiedBridge::authorize() decision
//!     │
//!     ▼  build UnifiedAuditEvent  (tenant, owl, op, role, decision, ts)
//! event.canonical_bytes()
//!     │
//!     ▼  chain.advance(salt, &event)
//! new_root = AuditMerkleRoot::chain(prev_root, salt, canonical_bytes)
//!     │
//!     ▼  sink.emit(event { merkle_root: new_root })
//! durable record   (JSON Lines / Lance dataset / no-op)
//! ```
//!
//! D-SDR-4 scope: type system + chain mechanics + sink trait
//! reference impl + tamper detection helper. Production sinks
//! (`JsonlAuditSink`, `LanceAuditSink` in `crate::audit_sink`) are D-SDR-4b/sprint-7.
//! Wiring into `UnifiedBridge::authorize()` is D-SDR-5.
//!
//! ## Separate from `crate::audit`
//!
//! `crate::audit::AuditEntry` (feature-gated `audit-log`) records
//! **RLS-rewritten plan executions** — different event type, different
//! schema, different sink contract. The two coexist: a typical request
//! emits one `UnifiedAuditEvent` (auth decision) and zero or one
//! `audit::AuditEntry` (the rewritten plan, if execution proceeded).

use crate::super_domain::SuperDomain;
use crate::unified_bridge::{OwlIdentity, TenantId};

// ═══════════════════════════════════════════════════════════════════════════
// AuthOp — what the caller asked for
// ═══════════════════════════════════════════════════════════════════════════

/// Operation classifier for an audit event. Captures `Operation`'s shape
/// without dragging the `lance_graph_rbac::policy::Operation<'_>` lifetime
/// into the audit record (which must be `'static` / owned for sinking).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AuthOp {
    Read = 0,
    Write = 1,
    Act = 2,
}

impl AuthOp {
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AuthDecision — what the bridge returned
// ═══════════════════════════════════════════════════════════════════════════

/// Decision tag preserved in the audit record. Mirrors
/// `AccessDecision` shape but lifetime-free for durable storage.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AuthDecision {
    Allow = 0,
    Deny = 1,
    Escalate = 2,
    BridgeError = 3,
}

impl AuthDecision {
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AuditMerkleRoot — chained hash with per-super-domain salt (§13.3)
// ═══════════════════════════════════════════════════════════════════════════

/// 8 bytes. Merkle root of one event in the audit chain. Includes the
/// prior root + per-super-domain salt (§13.4 — cross-domain audit logs
/// are unlinkable; an OSINT auditor seeing a Healthcare-side merkle can't
/// correlate it with the Healthcare-side chain).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AuditMerkleRoot(pub u64);

impl AuditMerkleRoot {
    /// The chain's seed root — used when no prior event exists.
    pub const GENESIS: Self = Self(0xa5a5_a5a5_a5a5_a5a5);

    /// Chain operator: advance the merkle root by hashing `prev_root` +
    /// `salt` + `entry_bytes`. Uses FNV-1a 64-bit (deterministic across
    /// platforms / Rust versions — safe to persist + cross-binary verify).
    pub fn chain(prev_root: Self, salt: u64, entry_bytes: &[u8]) -> Self {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a 64 offset basis
        h = fnv_step(h, &prev_root.0.to_le_bytes());
        h = fnv_step(h, &salt.to_le_bytes());
        h = fnv_step(h, entry_bytes);
        Self(h)
    }

    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

#[inline]
fn fnv_step(mut h: u64, bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01B3;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

// ═══════════════════════════════════════════════════════════════════════════
// UnifiedAuditEvent — one row in the chain
// ═══════════════════════════════════════════════════════════════════════════

/// One audit-chain entry. `merkle_root` is computed by
/// `AuditChain::advance(...)` at emission time; it carries the
/// chain-integrity bind to the prior event + the per-super-domain salt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnifiedAuditEvent {
    /// Wall-clock timestamp in milliseconds since UNIX epoch. Quantized to
    /// ms so the canonical-bytes representation is byte-deterministic
    /// across Rust / C# emitters (per §15.2 cross-language determinism
    /// rules).
    pub ts_unix_ms: u64,
    /// Multi-tenant Chinese wall tag (§3.8).
    pub tenant: TenantId,
    /// Super domain the OGIT basin belongs to (looked up via
    /// `FAMILY_TO_SUPER_DOMAIN` at emit time).
    pub super_domain: SuperDomain,
    /// 3-byte OwlIdentity = `(family u8, slot u16)` — the per-row
    /// identity that was authorized. Slot is full-width since PR #364
    /// review surfaced that registry IDs are globally u16 and would
    /// alias under the prior u8 truncation.
    pub owl: OwlIdentity,
    /// Read / Write / Act.
    pub op: AuthOp,
    /// Allow / Deny / Escalate / BridgeError.
    pub decision: AuthDecision,
    /// Actor role name's FNV-1a 64-bit digest. We hash the role rather
    /// than store the `&'static str` so the event is `Copy` + the audit
    /// record is fixed-size (durable storage friendly).
    pub actor_role_hash: u64,
    /// Computed by [`AuditChain::advance`] at emission; equals
    /// `AuditMerkleRoot::chain(prev_root, salt, self.canonical_bytes())`.
    pub merkle_root: AuditMerkleRoot,
    /// Merkle root of the immediately preceding event in this chain.
    /// `AuditMerkleRoot::GENESIS` for the first event.
    /// Excluded from `canonical_bytes()` — it is the prior chain output,
    /// not an input; including it would create a circular dependency.
    /// D-SDR-4b field; populated by `AuditChain::advance()`.
    pub prev_merkle: AuditMerkleRoot,
}

impl UnifiedAuditEvent {
    /// Canonical byte representation used as input to `AuditMerkleRoot::chain`.
    /// Field order is fixed; little-endian for all integers; no padding.
    /// **`merkle_root` is excluded** — it's the OUTPUT of the chain, not
    /// an input.
    pub fn canonical_bytes(&self) -> [u8; 8 + 4 + 1 + 3 + 1 + 1 + 8] {
        let mut out = [0u8; 26];
        out[0..8].copy_from_slice(&self.ts_unix_ms.to_le_bytes());
        out[8..12].copy_from_slice(&self.tenant.raw().to_le_bytes());
        out[12] = self.super_domain.raw();
        out[13..16].copy_from_slice(&self.owl.to_canonical_bytes());
        out[16] = self.op.as_u8();
        out[17] = self.decision.as_u8();
        out[18..26].copy_from_slice(&self.actor_role_hash.to_le_bytes());
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AuditChain — stateful chain advancer
// ═══════════════════════════════════════════════════════════════════════════

/// Tracks the prior merkle root for one super domain so each new event
/// can chain off it. Construct one per super-domain context the
/// `UnifiedBridge` operates against.
#[derive(Clone, Copy, Debug)]
pub struct AuditChain {
    pub super_domain: SuperDomain,
    /// Per-super-domain salt — looked up from the super-domain registry
    /// (`SuperDomainEntry::merkle_salt` once D-SDR-4 wires it in via the
    /// authorize() path). For D-SDR-4 minimum, callers pass it explicitly.
    pub salt: u64,
    /// Root of the last-emitted event. Initialized to `GENESIS`.
    pub last_root: AuditMerkleRoot,
}

impl AuditChain {
    /// Construct a fresh chain seeded at `GENESIS` for the given super
    /// domain + salt.
    pub fn new(super_domain: SuperDomain, salt: u64) -> Self {
        Self {
            super_domain,
            salt,
            last_root: AuditMerkleRoot::GENESIS,
        }
    }

    /// Resume a chain from a known prior root (e.g. on process restart
    /// after reading the last persisted event's root).
    pub fn resume(super_domain: SuperDomain, salt: u64, last_root: AuditMerkleRoot) -> Self {
        Self {
            super_domain,
            salt,
            last_root,
        }
    }

    /// Stamp `event.merkle_root` with the chained hash and update
    /// `self.last_root`. Returns the freshly-stamped event for emission.
    ///
    /// D-SDR-4b: captures `self.last_root` into `event.prev_merkle` BEFORE
    /// chaining so the prior root is available for single-event spot-checks
    /// in `verify-jsonl` / `verify-lance` without scanning from genesis.
    /// `prev_merkle` is NOT included in `canonical_bytes()` — that would
    /// create a circular dependency.
    pub fn advance(&mut self, mut event: UnifiedAuditEvent) -> UnifiedAuditEvent {
        event.prev_merkle = self.last_root; // capture BEFORE chaining
        let new_root = AuditMerkleRoot::chain(self.last_root, self.salt, &event.canonical_bytes());
        event.merkle_root = new_root;
        self.last_root = new_root;
        event
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HydrationRefreshAudit — emitted on every FAMILY_TABLE reload
// ═══════════════════════════════════════════════════════════════════════════

/// Audit event emitted whenever `FAMILY_TABLE` is reloaded (on first boot or
/// on a hot-reload triggered by `BridgeHandle::reload_family_table()`).
///
/// Per spec §3.2 — consumers can observe `generation` to change-detect
/// without racing on the table lock.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HydrationRefreshAudit {
    /// Wall-clock timestamp in milliseconds since UNIX epoch of the reload.
    pub ts_unix_ms: u64,
    /// Generation counter after the reload. Starts at 1; incremented on each
    /// subsequent reload.
    pub generation: u64,
    /// Number of family_id → SuperDomain mappings that changed or were newly
    /// set relative to the prior generation. Zero on the first boot (no prior
    /// to compare against).
    pub updated_count: u32,
    /// Human-readable description of the source(s) that contributed to this
    /// reload (e.g. `"seed"`, `"seed+overlay:/etc/ogit/family"`).
    pub source: String,
}

impl HydrationRefreshAudit {
    /// Build a `HydrationRefreshAudit` from the current wall clock.
    pub fn now(generation: u64, updated_count: u32, source: impl Into<String>) -> Self {
        let ts_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            ts_unix_ms,
            generation,
            updated_count,
            source: source.into(),
        }
    }
}

// The audit sink trait + NoopAuditSink moved to crate::audit_sink in
// sprint-7 (OQ-7-2 locked 2026-05-13). UnifiedAuditSink and
// NoopUnifiedAuditSink were the D-SDR-4 placeholders; production
// consumers use `AuditSink` from `crate::audit_sink` directly.

// ═══════════════════════════════════════════════════════════════════════════
// Chain verification — tamper detection
// ═══════════════════════════════════════════════════════════════════════════

/// Verify a slice of consecutive audit events against the chain contract.
///
/// Given `start_root` (the merkle root prior to `events[0]`) and `salt`,
/// re-derive each event's merkle root and confirm it matches what's
/// stored. Returns `Ok(final_root)` if the chain is intact, or
/// `Err(broken_index)` pointing at the first event that fails to match
/// the recomputed root.
///
/// This is the tamper-detection helper: any change to a past event's
/// fields (or its computed merkle_root) — or insertion / deletion of an
/// event — breaks the chain at that point and is detected here.
pub fn verify_chain(
    start_root: AuditMerkleRoot,
    salt: u64,
    events: &[UnifiedAuditEvent],
) -> Result<AuditMerkleRoot, usize> {
    let mut prev = start_root;
    for (i, ev) in events.iter().enumerate() {
        let expected = AuditMerkleRoot::chain(prev, salt, &ev.canonical_bytes());
        if expected != ev.merkle_root {
            return Err(i);
        }
        prev = expected;
    }
    Ok(prev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_bridge::OgitFamily;

    fn fresh_event() -> UnifiedAuditEvent {
        UnifiedAuditEvent {
            ts_unix_ms: 1_700_000_000_000,
            tenant: TenantId(42),
            super_domain: SuperDomain::Healthcare,
            owl: OwlIdentity::new(OgitFamily(7), 5),
            op: AuthOp::Read,
            decision: AuthDecision::Allow,
            actor_role_hash: 0xCAFE_BABE_DEAD_BEEF,
            merkle_root: AuditMerkleRoot::GENESIS, // overwritten by chain.advance
            prev_merkle: AuditMerkleRoot::GENESIS, // overwritten by chain.advance
        }
    }

    #[test]
    fn merkle_root_chain_is_deterministic() {
        let bytes = [1u8, 2, 3, 4];
        let a = AuditMerkleRoot::chain(AuditMerkleRoot::GENESIS, 0x1234, &bytes);
        let b = AuditMerkleRoot::chain(AuditMerkleRoot::GENESIS, 0x1234, &bytes);
        assert_eq!(a, b);
    }

    #[test]
    fn merkle_root_chain_depends_on_salt() {
        let bytes = [1u8, 2, 3, 4];
        let a = AuditMerkleRoot::chain(AuditMerkleRoot::GENESIS, 0x1234, &bytes);
        let b = AuditMerkleRoot::chain(AuditMerkleRoot::GENESIS, 0x5678, &bytes);
        assert_ne!(a, b, "different salts must produce different roots");
    }

    #[test]
    fn merkle_root_chain_depends_on_prior() {
        let bytes = [1u8, 2, 3, 4];
        let a = AuditMerkleRoot::chain(AuditMerkleRoot::GENESIS, 0x1234, &bytes);
        let b = AuditMerkleRoot::chain(AuditMerkleRoot(0xDEADBEEF), 0x1234, &bytes);
        assert_ne!(
            a, b,
            "different prior roots must produce different new roots"
        );
    }

    #[test]
    fn audit_chain_advance_stamps_event_and_updates_state() {
        let mut chain = AuditChain::new(SuperDomain::Healthcare, 0xC0FFEE);
        let ev = chain.advance(fresh_event());

        assert_ne!(ev.merkle_root, AuditMerkleRoot::GENESIS);
        assert_eq!(chain.last_root, ev.merkle_root);
    }

    #[test]
    fn audit_chain_two_events_chain_through() {
        let mut chain = AuditChain::new(SuperDomain::Healthcare, 0xC0FFEE);
        let e1 = chain.advance(fresh_event());
        let e2 = chain.advance(UnifiedAuditEvent {
            ts_unix_ms: 1_700_000_000_001,
            ..fresh_event()
        });
        assert_ne!(e1.merkle_root, e2.merkle_root);
        assert_eq!(chain.last_root, e2.merkle_root);
    }

    #[test]
    fn verify_chain_accepts_genuine_chain() {
        let mut chain = AuditChain::new(SuperDomain::Healthcare, 0xC0FFEE);
        let events: Vec<UnifiedAuditEvent> = (0..5)
            .map(|i| {
                chain.advance(UnifiedAuditEvent {
                    ts_unix_ms: 1_700_000_000_000 + i,
                    ..fresh_event()
                })
            })
            .collect();

        let final_root = verify_chain(AuditMerkleRoot::GENESIS, 0xC0FFEE, &events)
            .expect("genuine chain should verify");
        assert_eq!(final_root, chain.last_root);
    }

    #[test]
    fn verify_chain_detects_tampered_event_in_middle() {
        let mut chain = AuditChain::new(SuperDomain::Healthcare, 0xC0FFEE);
        let mut events: Vec<UnifiedAuditEvent> = (0..5)
            .map(|i| {
                chain.advance(UnifiedAuditEvent {
                    ts_unix_ms: 1_700_000_000_000 + i,
                    ..fresh_event()
                })
            })
            .collect();

        // Tamper with event[2] — flip the decision from Allow to Deny.
        // Leave the merkle_root unchanged (the attacker would have to
        // recompute it but doesn't know the salt).
        events[2].decision = AuthDecision::Deny;

        let err = verify_chain(AuditMerkleRoot::GENESIS, 0xC0FFEE, &events)
            .expect_err("tampered chain should fail verification");
        assert_eq!(
            err, 2,
            "verification should fail at index 2 (the tampered event)"
        );
    }

    #[test]
    fn verify_chain_detects_wrong_salt() {
        let mut chain = AuditChain::new(SuperDomain::Healthcare, 0xC0FFEE);
        let events: Vec<UnifiedAuditEvent> = (0..3)
            .map(|i| {
                chain.advance(UnifiedAuditEvent {
                    ts_unix_ms: 1_700_000_000_000 + i,
                    ..fresh_event()
                })
            })
            .collect();

        // Verify with the wrong salt — should fail at index 0.
        let err = verify_chain(AuditMerkleRoot::GENESIS, 0xBADBEEF, &events)
            .expect_err("wrong salt should break verification");
        assert_eq!(err, 0);
    }

    #[test]
    fn noop_sink_swallows_events() {
        use crate::audit_sink::{AuditSink, NoopAuditSink};
        let sink = NoopAuditSink;
        let mut chain = AuditChain::new(SuperDomain::Healthcare, 0xC0FFEE);
        let ev = chain.advance(fresh_event());
        sink.emit(ev).expect("noop never errors"); // doesn't observe — by design
    }

    #[test]
    fn canonical_bytes_round_trips_field_order() {
        let ev = fresh_event();
        let bytes = ev.canonical_bytes();

        // Verify field-by-field by spot-checking offsets.
        assert_eq!(&bytes[0..8], &ev.ts_unix_ms.to_le_bytes());
        assert_eq!(&bytes[8..12], &ev.tenant.raw().to_le_bytes());
        assert_eq!(bytes[12], ev.super_domain.raw());
        assert_eq!(&bytes[13..16], &ev.owl.to_canonical_bytes());
        assert_eq!(bytes[16], ev.op.as_u8());
        assert_eq!(bytes[17], ev.decision.as_u8());
        assert_eq!(&bytes[18..26], &ev.actor_role_hash.to_le_bytes());
    }
}
