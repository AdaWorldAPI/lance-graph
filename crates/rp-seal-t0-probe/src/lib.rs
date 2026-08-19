//! X-C2-1 — the RP-SEAL fault-injection harness (Tier 0 prerequisite).
//!
//! A synthetic cycle (default 65,536 × 512 B = 32 MiB) under controlled fault
//! injection; each candidate scheme audits the cycle and the **injection
//! record is the ground truth**. Metric: false-accept rate per
//! (scheme, injection, multiplicity); a false accept is an affected chunk the
//! auditor did not flag.
//!
//! Landed in this arc: the harness, injections I1–I9, and the two HASH
//! schemes — S1U (locus-UNbound content digest, modelling the shipped
//! `ndarray::hpc::seal` shape) and S6 (locus+version-bound digest, the C2
//! recommendation's hash half). The RS-family scheme arms (S2–S5: flat RS,
//! row/col P+Q, product, cascade) belong to X-C2-3 and plug into the same
//! [`Scheme`] trait.
//!
//! Anti-vacuity gate (charter X-C2-1): `tests/controls.rs` first reproduces
//! the known in-tree false accepts as positive controls against the REAL
//! implementations — if those controls fail, this harness measures nothing
//! and no other result from it may be reported.
//!
//! **Wall-clock discipline (T0.3):** nothing here measures time. Every
//! quantity is a count over deterministic trials.

use std::collections::BTreeSet;

/// Bytes per chunk — the witness-row stride of the substrate.
pub const CHUNK_BYTES: usize = 512;
/// Chunks in the full synthetic cycle (65,536 × 512 B = 32 MiB).
pub const FULL_CHUNKS: usize = 65_536;

/// The durable coordinates a bound seal embeds. `slot` is the canonical
/// position; `version` is the cycle's publication version (W_write in the
/// T0.3 vocabulary — the harness binds it so stale substitution is
/// positively diagnosable, not merely detectable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Locus {
    pub slot: u64,
    pub version: u64,
}

/// Deterministic chunk content for a locus — splitmix64 stream keyed on
/// (slot, version), so any (slot, version) pair regenerates its exact bytes
/// and trials are reproducible without storing fixtures.
pub fn chunk_content(slot: u64, version: u64) -> Vec<u8> {
    let mut x = slot
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(version.wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add(0x632B_E59B_D9B4_E019);
    let mut out = Vec::with_capacity(CHUNK_BYTES);
    while out.len() < CHUNK_BYTES {
        // splitmix64
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.extend_from_slice(&z.to_le_bytes());
    }
    out
}

/// One sealed cycle under audit: per-slot content plus the stored seal that
/// travels WITH the chunk (that is the unbound-seal reality the harness
/// models: a wrong-slot / stale / duplicate substitution moves both the
/// bytes and their seal, exactly as a fragment move would).
pub struct Cycle {
    pub version: u64,
    pub chunks: Vec<Option<Vec<u8>>>, // None = erased
    pub seals: Vec<Option<Vec<u8>>>,
}

impl Cycle {
    /// Build + seal a clean synthetic cycle of `n` chunks at `version`.
    pub fn sealed<S: Scheme + ?Sized>(scheme: &S, n: usize, version: u64) -> Self {
        let mut chunks = Vec::with_capacity(n);
        let mut seals = Vec::with_capacity(n);
        for slot in 0..n as u64 {
            let content = chunk_content(slot, version);
            let seal = scheme.seal_chunk(Locus { slot, version }, &content);
            chunks.push(Some(content));
            seals.push(Some(seal));
        }
        Self {
            version,
            chunks,
            seals,
        }
    }
}

/// The nine charter injections (C2 §MECHANISM), each returning the ground
/// truth: the set of slots whose durable content is wrong or missing after
/// the fault. Faults operate on (chunk, seal) PAIRS where the physical
/// analogue moves both.
#[derive(Debug, Clone)]
pub enum Injection {
    /// I1 — single erasure, position known.
    I1 { pos: usize },
    /// I2 — double erasure.
    I2 { a: usize, b: usize },
    /// I3 — silent corruption: one bit flipped in one chunk's content, seal
    /// left as stored.
    I3 { pos: usize, bit: usize },
    /// I4 — wrong-slot substitution: the (chunk, seal) pair from `src`
    /// placed at `dst`.
    I4 { src: usize, dst: usize },
    /// I5 — stale chunk: the (chunk, seal) pair regenerated from an OLD
    /// version placed at the same slot.
    I5 { pos: usize, old_version: u64 },
    /// I6 — duplicate chunk: `src`'s pair copied over `dst` (src keeps its
    /// own too).
    I6 { src: usize, dst: usize },
    /// I7 — correlated failure domain: every chunk in `[start, start+len)`
    /// silently corrupted (one failure domain).
    I7 {
        start: usize,
        len: usize,
        bit: usize,
    },
    /// I8 — boundary-correlated: corruption at the two edge slots of each
    /// group of `group` chunks.
    I8 { group: usize, bit: usize },
    /// I9 — phase-aligned: corruption at every slot ≡ `phase` (mod
    /// `stride`), strides congruent to the cascade levels
    /// (4/16/64/256/1024/4096).
    I9 {
        stride: usize,
        phase: usize,
        bit: usize,
    },
}

impl Injection {
    /// Apply to a sealed cycle; return the affected-slot ground truth.
    pub fn apply(&self, cy: &mut Cycle) -> BTreeSet<usize> {
        let mut affected = BTreeSet::new();
        let n = cy.chunks.len();
        let flip = |c: &mut Vec<u8>, bit: usize| {
            let b = bit % (c.len() * 8);
            c[b / 8] ^= 1 << (b % 8);
        };
        match *self {
            Injection::I1 { pos } => {
                cy.chunks[pos] = None;
                cy.seals[pos] = None;
                affected.insert(pos);
            }
            Injection::I2 { a, b } => {
                for p in [a, b] {
                    cy.chunks[p] = None;
                    cy.seals[p] = None;
                    affected.insert(p);
                }
            }
            Injection::I3 { pos, bit } => {
                flip(cy.chunks[pos].as_mut().expect("present"), bit);
                affected.insert(pos);
            }
            Injection::I4 { src, dst } => {
                cy.chunks[dst] = cy.chunks[src].clone();
                cy.seals[dst] = cy.seals[src].clone();
                affected.insert(dst);
            }
            Injection::I5 { pos, old_version } => {
                let old = chunk_content(pos as u64, old_version);
                // The stale pair carries the OLD seal — sealed when it was
                // current, at the same slot.
                cy.seals[pos] = Some(vec![]); // placeholder; caller reseals below
                cy.chunks[pos] = Some(old);
                affected.insert(pos);
            }
            Injection::I6 { src, dst } => {
                cy.chunks[dst] = cy.chunks[src].clone();
                cy.seals[dst] = cy.seals[src].clone();
                affected.insert(dst);
            }
            Injection::I7 { start, len, bit } => {
                for p in start..(start + len).min(n) {
                    flip(cy.chunks[p].as_mut().expect("present"), bit);
                    affected.insert(p);
                }
            }
            Injection::I8 { group, bit } => {
                let mut p = 0;
                while p < n {
                    let last = (p + group - 1).min(n - 1);
                    for q in [p, last] {
                        if affected.insert(q) {
                            flip(cy.chunks[q].as_mut().expect("present"), bit);
                        }
                    }
                    p += group;
                }
            }
            Injection::I9 { stride, phase, bit } => {
                let mut p = phase % stride;
                while p < n {
                    flip(cy.chunks[p].as_mut().expect("present"), bit);
                    affected.insert(p);
                    p += stride;
                }
            }
        }
        affected
    }

    /// I5 needs the OLD seal (sealed when the stale pair was current) —
    /// apply it after [`Injection::apply`] using the auditing scheme.
    pub fn reseal_stale<S: Scheme + ?Sized>(&self, scheme: &S, cy: &mut Cycle) {
        if let Injection::I5 { pos, old_version } = *self {
            let old = chunk_content(pos as u64, old_version);
            cy.seals[pos] = Some(scheme.seal_chunk(
                Locus {
                    slot: pos as u64,
                    version: old_version,
                },
                &old,
            ));
        }
    }
}

/// A candidate seal scheme. X-C2-3's coding-family arms (S2–S5) implement
/// the same trait with a repair verdict layered on.
pub trait Scheme {
    fn name(&self) -> &'static str;
    fn seal_chunk(&self, locus: Locus, content: &[u8]) -> Vec<u8>;
    /// `true` = ACCEPT (the chunk audits clean at this locus).
    fn accept(&self, locus: Locus, content: &[u8], stored: &[u8]) -> bool;
}

/// S1U — the locus-UNbound content digest: 48-bit truncated blake3 of the
/// content alone. This is the shape of the shipped `ndarray::hpc::seal`
/// (`MerkleRoot`), modelled at chunk granularity.
pub struct S1Unbound;
impl Scheme for S1Unbound {
    fn name(&self) -> &'static str {
        "S1U hash-only, locus-unbound (shipped shape)"
    }
    fn seal_chunk(&self, _locus: Locus, content: &[u8]) -> Vec<u8> {
        blake3::hash(content).as_bytes()[..6].to_vec()
    }
    fn accept(&self, _locus: Locus, content: &[u8], stored: &[u8]) -> bool {
        self.seal_chunk(_locus, content) == stored
    }
}

/// S6 — the locus+version-bound digest (the C2 recommendation's hash half):
/// 64-bit blake3 over (slot ‖ version ‖ content). Wrong-slot, stale, and
/// duplicate substitutions become positively diagnosable.
pub struct S6Bound;
impl Scheme for S6Bound {
    fn name(&self) -> &'static str {
        "S6 hash, locus+version-bound"
    }
    fn seal_chunk(&self, locus: Locus, content: &[u8]) -> Vec<u8> {
        let mut h = blake3::Hasher::new();
        h.update(&locus.slot.to_le_bytes());
        h.update(&locus.version.to_le_bytes());
        h.update(content);
        h.finalize().as_bytes()[..8].to_vec()
    }
    fn accept(&self, locus: Locus, content: &[u8], stored: &[u8]) -> bool {
        self.seal_chunk(locus, content) == stored
    }
}

/// Audit every present chunk at its canonical locus; a missing chunk or
/// seal is a DETECT by presence. Returns the flagged-slot set.
pub fn audit<S: Scheme + ?Sized>(scheme: &S, cy: &Cycle) -> BTreeSet<usize> {
    let mut flagged = BTreeSet::new();
    for slot in 0..cy.chunks.len() {
        match (&cy.chunks[slot], &cy.seals[slot]) {
            (Some(content), Some(stored)) => {
                let locus = Locus {
                    slot: slot as u64,
                    version: cy.version,
                };
                if !scheme.accept(locus, content, stored) {
                    flagged.insert(slot);
                }
            }
            _ => {
                flagged.insert(slot);
            }
        }
    }
    flagged
}

/// The false accepts: affected slots the audit did NOT flag.
pub fn false_accepts(affected: &BTreeSet<usize>, flagged: &BTreeSet<usize>) -> BTreeSet<usize> {
    affected.difference(flagged).copied().collect()
}

/// The false alarms: flagged slots the injection did NOT touch (the null
/// side of the metric — a scheme that flags clean chunks is as broken as
/// one that accepts dirty ones).
pub fn false_alarms(affected: &BTreeSet<usize>, flagged: &BTreeSet<usize>) -> BTreeSet<usize> {
    flagged.difference(affected).copied().collect()
}

/// The charter null control: `trials` audits of clean chunks; returns the
/// number of spurious flags (must be 0). Content varies per trial via the
/// deterministic stream, so this is 10⁶ DISTINCT clean chunks, not one
/// chunk hashed 10⁶ times.
pub fn null_control<S: Scheme + ?Sized>(scheme: &S, trials: u64) -> u64 {
    let mut spurious = 0;
    for t in 0..trials {
        let locus = Locus {
            slot: t,
            version: 1 + (t % 7),
        };
        let content = chunk_content(locus.slot, locus.version);
        let stored = scheme.seal_chunk(locus, &content);
        if !scheme.accept(locus, &content, &stored) {
            spurious += 1;
        }
    }
    spurious
}

/// One (scheme × injection) run at the given cycle size: returns
/// (affected, flagged, false-accepts, false-alarms).
pub fn run_one<S: Scheme + ?Sized>(
    scheme: &S,
    n: usize,
    inj: &Injection,
) -> (
    BTreeSet<usize>,
    BTreeSet<usize>,
    BTreeSet<usize>,
    BTreeSet<usize>,
) {
    let mut cy = Cycle::sealed(scheme, n, 3);
    let affected = inj.apply(&mut cy);
    inj.reseal_stale(scheme, &mut cy);
    let flagged = audit(scheme, &cy);
    let fa = false_accepts(&affected, &flagged);
    let alarms = false_alarms(&affected, &flagged);
    (affected, flagged, fa, alarms)
}
