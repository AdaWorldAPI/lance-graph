//! `facet_fold` — ENTROPY-MILESTONES **M26**: the `Belief ⟷ SpoFacet`
//! round-trip.
//!
//! A [`Belief`](super::belief::Belief)'s statement PERSISTS as a reading of
//! the M20 awareness register ([`SpoFacet`](lance_graph_contract::awareness_facet::SpoFacet)),
//! not as a new store. This is a **lossless, content-blind byte relabel** —
//! NOT a lossy codebook encode. The `SpoFacet` is a reading of a 12-byte
//! content-blind register; a belief's statement (`s`, `cop`, `p`) is written
//! into those same bytes and read back byte-for-byte.
//!
//! The SEMANTIC placement (mapping a term to a `palette256²` centroid via a
//! trained codebook, e.g. [`bgz17::HierarchicalPalette`]) is SEPARATE future
//! work — it needs term embeddings the planner does not have. This fold is
//! the lossless statement-CARRIER only: the register bytes hold the raw
//! `CStmt` fields relabeled as `Palette256Pair`s, round-trippable exactly.
//!
//! Byte layout (six rails, matching [`SpoFacet::from_register`]'s convention
//! `rail k = (bytes[2k], bytes[2k+1])`):
//!
//! | Rail | Field                 | Contents                                            |
//! |------|-----------------------|------------------------------------------------------|
//! | 0    | `subject`             | `s` low byte, `s` high byte (u16 LE)                  |
//! | 1    | `predicate`           | copula tag (0=Inh,1=Sim,2=Impl,3=Rel), `Rel` low byte |
//! | 2    | `object`              | `p` low byte, `p` high byte (u16 LE)                  |
//! | 3    | `ew_subject`          | `Rel` high byte, 0 (spare)                            |
//! | 4    | `ew_predicate`        | provenance summary (may be lossy — NOT gated)         |
//! | 5    | `ew_object`           | provenance summary (may be lossy — NOT gated)         |
//!
//! The round-trip GATE (see the tests below) covers rails 0–3 exactly — the
//! full `CStmt` (`s`, `cop` incl. `Rel(u16)` payloads > 255, `p`) — because
//! that is the belief's STATEMENT. Rails 4–5 carry a documented, lossy
//! provenance summary (`rung`, `premises` length) that is never asserted
//! round-trip-exact; provenance need not survive the register the way the
//! statement must.

use super::belief::{CStmt, Copula};
use lance_graph_contract::awareness_facet::SpoFacet;

/// Copula tag values written into rail 1's low byte (`predicate.0`).
const TAG_INH: u8 = 0;
const TAG_SIM: u8 = 1;
const TAG_IMPL: u8 = 2;
const TAG_REL: u8 = 3;

/// Fold a [`CStmt`] into an [`SpoFacet`] reading — a lossless, content-blind
/// byte relabel of the statement's own fields, not a codebook encode.
///
/// - rail 0 (`subject`) = `s` as `(lo, hi)` LE bytes.
/// - rail 1 (`predicate`) = `(tag, rel_lo)` — the copula's discriminant tag,
///   plus `Rel(v)`'s low byte (`0` for every other copula).
/// - rail 2 (`object`) = `p` as `(lo, hi)` LE bytes.
/// - rail 3 (`ew_subject`) = `(rel_hi, 0)` — `Rel(v)`'s high byte (the u16
///   payload overflow), `0` spare. Together, rails 1+3 carry `Rel`'s full
///   u16 losslessly: `rel_lo | (rel_hi << 8) == v`.
/// - rails 4–5 (`ew_predicate`, `ew_object`) = a lossy provenance summary
///   (`rung` LE + a saturating premise count) — documented as NOT
///   round-trip-gated; see the module docs.
#[must_use]
pub fn to_spo_facet(stmt: &CStmt, rung: u32, premises_len: usize) -> SpoFacet {
    let (tag, rel_lo, rel_hi) = match stmt.cop {
        Copula::Inh => (TAG_INH, 0u8, 0u8),
        Copula::Sim => (TAG_SIM, 0u8, 0u8),
        Copula::Impl => (TAG_IMPL, 0u8, 0u8),
        Copula::Rel(v) => (TAG_REL, (v & 0xff) as u8, (v >> 8) as u8),
    };
    let s = stmt.s;
    let p = stmt.p;
    let rung_bytes = rung.to_le_bytes();
    let premises_sat = premises_len.min(u8::MAX as usize) as u8;
    SpoFacet::from_rails([
        ((s & 0xff) as u8, (s >> 8) as u8),
        (tag, rel_lo),
        ((p & 0xff) as u8, (p >> 8) as u8),
        (rel_hi, 0),
        (rung_bytes[0], rung_bytes[1]),
        (rung_bytes[2].wrapping_add(rung_bytes[3]), premises_sat),
    ])
}

/// Recover the exact [`CStmt`] from an [`SpoFacet`] built by
/// [`to_spo_facet`] — the inverse of the statement-carrying rails (0–3).
/// Provenance (rails 4–5) is NOT recovered here (it is a lossy summary, not
/// part of the statement — see the module docs).
#[must_use]
pub fn cstmt_from_spo_facet(f: &SpoFacet) -> CStmt {
    let s = u16::from(f.subject.0) | (u16::from(f.subject.1) << 8);
    let p = u16::from(f.object.0) | (u16::from(f.object.1) << 8);
    let (tag, rel_lo) = f.predicate;
    let rel_hi = f.ew_subject.0;
    let cop = match tag {
        TAG_INH => Copula::Inh,
        TAG_SIM => Copula::Sim,
        TAG_IMPL => Copula::Impl,
        _ => Copula::Rel(u16::from(rel_lo) | (u16::from(rel_hi) << 8)),
    };
    CStmt { s, cop, p }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stmts() -> Vec<CStmt> {
        let mut v = Vec::new();
        // Boundary term-ids, spanning every copula.
        let terms: [u16; 4] = [0, 255, 256, 65535];
        let copulas: [Copula; 7] = [
            Copula::Inh,
            Copula::Sim,
            Copula::Impl,
            Copula::Rel(0),
            Copula::Rel(255),
            Copula::Rel(256),   // exercises rel_hi != 0
            Copula::Rel(65535), // max u16, exercises rel_hi == 0xff
        ];
        for &s in &terms {
            for &p in &terms {
                for &cop in &copulas {
                    v.push(CStmt { s, cop, p });
                }
            }
        }
        v
    }

    /// M26 GATE: the belief's STATEMENT round-trips exactly through the
    /// SpoFacet register, for every copula (incl. `Rel` payloads > 255) and
    /// boundary term-ids. Proves a Belief PERSISTS as a reading of the
    /// register — no new store, no information loss on the statement.
    #[test]
    fn cstmt_round_trips_through_spo_facet_for_all_copulas_and_boundaries() {
        for stmt in stmts() {
            let f = to_spo_facet(&stmt, 3, 2);
            let back = cstmt_from_spo_facet(&f);
            assert_eq!(back, stmt, "round-trip failed for {stmt:?}");
        }
    }

    /// The fold is a pure byte relabel: reading `to_spo_facet(..).to_register()`
    /// back through `SpoFacet::from_register` reproduces the identical facet —
    /// no hidden state, just the 12 bytes.
    #[test]
    fn to_spo_facet_is_a_pure_byte_relabel_of_the_register() {
        for stmt in stmts() {
            let f = to_spo_facet(&stmt, 7, 1);
            let reg = f.to_register();
            let reread = SpoFacet::from_register(reg);
            assert_eq!(
                f, reread,
                "register relabel must be self-consistent for {stmt:?}"
            );
        }
    }

    /// Rel's u16 payload spans both bytes losslessly (rail 1 lo + rail 3 hi).
    #[test]
    fn rel_payload_spans_two_rails_losslessly() {
        let stmt = CStmt {
            s: 10,
            cop: Copula::Rel(0xBEEF),
            p: 20,
        };
        let f = to_spo_facet(&stmt, 0, 0);
        assert_eq!(f.predicate.1, 0xEF, "rel low byte in rail 1");
        assert_eq!(f.ew_subject.0, 0xBE, "rel high byte in rail 3");
        assert_eq!(cstmt_from_spo_facet(&f), stmt);
    }
}
