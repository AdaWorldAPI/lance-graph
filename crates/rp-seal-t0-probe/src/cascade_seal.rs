//! ⚠ HELD — NOT WIRED, NOT RATIFIED (operator STOP, 2026-08-19).
//!
//! ⊘ REGISTER-GRID CORRECTION (same day): the digest-register tree below
//! survives (tiny digests, reduce-children-only, online bubble-up), but
//! `resolve_petal(slot, content)` is fed FROM the one flush-time
//! dereference (`NodeRowPacket::as_le_bytes` → Lance serializer), never
//! from cast time or a freeze fold — and a "petal" is 16 REGISTER
//! positions (pointers + resolved mask + digest state), never an 8-KiB
//! buffer. See the spec doc's ARCHITECTURAL CORRECTION.
//!
//! Pre-STOP scaffold committed only so the working tree stays clean; this
//! file is deliberately NOT declared in `lib.rs` and its digest deps are
//! NOT in `Cargo.toml` — it compiles into nothing. It becomes the req-11
//! in-architecture measurement harness ONLY after the operator ratifies
//! `docs/lotus/SEAL-FINALIZATION-MAP.md`, and will be reshaped by the
//! map's answers (leaf granularity row-512B vs petal-8KiB; Seam A vs B).
//!
//! The cascade-accumulated seal (operator STORNO spec,
//! `docs/lotus/CASCADE-ACCUMULATED-SEAL-SPEC.md`).
//!
//! Integrity is accumulated as the Morton-cascade image resolves in memory:
//! a petal's digest is computed the moment its bytes land (hot); interior
//! nodes reduce CHILD DIGESTS only, bubbling up the instant their last
//! child resolves; after the final petal only that petal's root path runs,
//! so the root exists before publication with no finalize pass, no payload
//! rescan, no storage reread, no physical-order dependency, and no
//! encryption. The petal digest binds canonical locus + version +
//! resolved/present state + content; an unresolved petal digests as
//! explicitly UNRESOLVED, so absence is part of content identity.
//!
//! The digest primitive is generic ([`PetalDigest`]) because the choice is
//! made ONLY inside this architecture (`examples/cascade_seal_bench.rs`),
//! never from isolated hash throughput.

/// A digest primitive slotted into the accumulator. Implementations must be
/// pure functions of their arguments (the accumulator supplies canonical
/// coordinates; the primitive never sees arrival order).
pub trait PetalDigest {
    fn name(&self) -> &'static str;
    /// Digest width in bytes (fixed per primitive).
    fn width(&self) -> usize;
    /// Leaf: binds locus + version + resolved-flag + content.
    fn petal(&self, slot: u64, version: u64, resolved: bool, content: &[u8]) -> Vec<u8>;
    /// Interior: binds tree position + the children's digests, in canonical
    /// child order. Payload bytes never reach this.
    fn reduce(&self, level: u8, index: u64, children: &[Vec<u8>]) -> Vec<u8>;
}

/// BLAKE3, truncated to 16 bytes.
pub struct Blake3Digest;
impl PetalDigest for Blake3Digest {
    fn name(&self) -> &'static str {
        "BLAKE3/16"
    }
    fn width(&self) -> usize {
        16
    }
    fn petal(&self, slot: u64, version: u64, resolved: bool, content: &[u8]) -> Vec<u8> {
        let mut h = blake3::Hasher::new();
        h.update(&slot.to_le_bytes());
        h.update(&version.to_le_bytes());
        h.update(&[u8::from(resolved)]);
        if resolved {
            h.update(content);
        }
        h.finalize().as_bytes()[..16].to_vec()
    }
    fn reduce(&self, level: u8, index: u64, children: &[Vec<u8>]) -> Vec<u8> {
        let mut h = blake3::Hasher::new();
        h.update(&[level]);
        h.update(&index.to_le_bytes());
        for c in children {
            h.update(c);
        }
        h.finalize().as_bytes()[..16].to_vec()
    }
}

/// CRC32C (Castagnoli, hardware-accelerated via the `crc32c` crate). 4-byte
/// width — the random false-accept floor is 2^-32 and the code is LINEAR
/// (constructed collisions are trivial); admissible here because the threat
/// model is fault detection, not an adversary (spec req 3: no encryption /
/// no adversarial claim). The benchmark reports it; the width caveat rides
/// every mention.
pub struct Crc32cDigest;
impl PetalDigest for Crc32cDigest {
    fn name(&self) -> &'static str {
        "CRC32C/4 (linear; 2^-32 floor)"
    }
    fn width(&self) -> usize {
        4
    }
    fn petal(&self, slot: u64, version: u64, resolved: bool, content: &[u8]) -> Vec<u8> {
        let mut c = crc32c::crc32c(&slot.to_le_bytes());
        c = crc32c::crc32c_append(c, &version.to_le_bytes());
        c = crc32c::crc32c_append(c, &[u8::from(resolved)]);
        if resolved {
            c = crc32c::crc32c_append(c, content);
        }
        c.to_le_bytes().to_vec()
    }
    fn reduce(&self, level: u8, index: u64, children: &[Vec<u8>]) -> Vec<u8> {
        let mut c = crc32c::crc32c(&[level]);
        c = crc32c::crc32c_append(c, &index.to_le_bytes());
        for ch in children {
            c = crc32c::crc32c_append(c, ch);
        }
        c.to_le_bytes().to_vec()
    }
}

/// xxHash3-64. 8-byte width, non-cryptographic, excellent short-input speed.
pub struct Xxh3Digest;
impl PetalDigest for Xxh3Digest {
    fn name(&self) -> &'static str {
        "XXH3/8"
    }
    fn width(&self) -> usize {
        8
    }
    fn petal(&self, slot: u64, version: u64, resolved: bool, content: &[u8]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(17 + content.len());
        buf.extend_from_slice(&slot.to_le_bytes());
        buf.extend_from_slice(&version.to_le_bytes());
        buf.push(u8::from(resolved));
        if resolved {
            buf.extend_from_slice(content);
        }
        xxhash_rust::xxh3::xxh3_64(&buf).to_le_bytes().to_vec()
    }
    fn reduce(&self, level: u8, index: u64, children: &[Vec<u8>]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(9 + children.len() * 8);
        buf.push(level);
        buf.extend_from_slice(&index.to_le_bytes());
        for c in children {
            buf.extend_from_slice(c);
        }
        xxhash_rust::xxh3::xxh3_64(&buf).to_le_bytes().to_vec()
    }
}

/// The accumulator: a fanout-F digest tree over `n` petal slots, resolved
/// in ANY arrival order, bubbling interior reductions online.
pub struct CascadeSeal<'d, D: PetalDigest + ?Sized> {
    digest: &'d D,
    fanout: usize,
    version: u64,
    /// levels[0] = petal digests … levels[last] = the single root.
    levels: Vec<Vec<Option<Vec<u8>>>>,
    /// Per interior node: how many children have resolved.
    resolved_children: Vec<Vec<u32>>,
    /// Accounting for F-SEAL-NORESCAN: payload bytes fed to petal().
    pub payload_bytes_digested: u64,
    /// Accounting for F-SEAL-ROOT-LATENCY: interior reduce() invocations.
    pub reduces: u64,
}

impl<'d, D: PetalDigest + ?Sized> CascadeSeal<'d, D> {
    /// `n` must be a power of `fanout` (the cascade is a full tree).
    pub fn new(digest: &'d D, n: usize, fanout: usize, version: u64) -> Self {
        assert!(fanout >= 2);
        let mut levels = vec![vec![None; n]];
        let mut resolved_children = Vec::new();
        let mut w = n;
        while w > 1 {
            assert_eq!(w % fanout, 0, "n must be a power of fanout");
            w /= fanout;
            levels.push(vec![None; w]);
            resolved_children.push(vec![0u32; w]);
        }
        Self {
            digest,
            fanout,
            version,
            levels,
            resolved_children,
            payload_bytes_digested: 0,
            reduces: 0,
        }
    }

    /// Resolve one petal — called at the moment the petal's bytes land, so
    /// the content is digested hot. Idempotence is the caller's business
    /// (the substrate resolves each slot once per cycle); re-resolving
    /// replaces the digest and re-bubbles the path.
    pub fn resolve_petal(&mut self, slot: usize, content: &[u8]) {
        let d = self
            .digest
            .petal(slot as u64, self.version, true, content);
        self.payload_bytes_digested += content.len() as u64;
        let fresh = self.levels[0][slot].is_none();
        self.levels[0][slot] = Some(d);
        self.bubble(slot, fresh);
    }

    /// Finalize: any still-unresolved petals digest as explicitly
    /// UNRESOLVED (absence is content identity). Touches ZERO payload
    /// bytes — an all-resolved cycle's finalize is a no-op.
    pub fn finalize(&mut self) {
        for slot in 0..self.levels[0].len() {
            if self.levels[0][slot].is_none() {
                let d = self.digest.petal(slot as u64, self.version, false, &[]);
                self.levels[0][slot] = Some(d);
                self.bubble(slot, true);
            }
        }
    }

    fn bubble(&mut self, mut idx: usize, mut fresh: bool) {
        for lvl in 0..self.resolved_children.len() {
            let parent = idx / self.fanout;
            if fresh {
                self.resolved_children[lvl][parent] += 1;
            }
            let full = self.resolved_children[lvl][parent] == self.fanout as u32;
            // Reduce when the node just became full, or when a re-resolve
            // ripples through an already-full node.
            if !full {
                return;
            }
            let base = parent * self.fanout;
            let children: Vec<Vec<u8>> = (base..base + self.fanout)
                .map(|i| self.levels[lvl][i].clone().expect("full node"))
                .collect();
            let d = self
                .digest
                .reduce((lvl + 1) as u8, parent as u64, &children);
            self.reduces += 1;
            fresh = self.levels[lvl + 1][parent].is_none();
            self.levels[lvl + 1][parent] = Some(d);
            idx = parent;
        }
    }

    /// The content identity. `Some` once every petal has resolved (or
    /// [`CascadeSeal::finalize`] stamped the absences).
    pub fn root(&self) -> Option<&[u8]> {
        self.levels.last().unwrap()[0].as_deref()
    }
}
