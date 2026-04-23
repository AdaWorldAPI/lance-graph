//! DN-addressed path — the URL hierarchy that becomes a scent.
//!
//! `tree/{ns}/heel/{h}/hip/{h}/branch/{b}/twig/{t}/leaf/{l}`
//!
//! Each segment is a u64 FNV-1a hash of the path component. The 6-tuple
//! compresses via ZeckBF17→Base17→CAM-PQ→scent (1B, ρ=0.937).
//! Phase C wires the full compression chain. Phase A carries the
//! parsed address and a stub scent derived by XOR-folding the 6 hashes.
//!
//! Plan: `.claude/plans/callcenter-membrane-v1.md` § 10.12 – § 10.13

/// Parsed DN address — 6 u64 segment hashes.
///
/// The address IS the context key: the same 6-tuple routes, retrieves
/// similar entries (CAM-PQ), and pulls the AriGraph subgraph (scent).
/// See plan § 10.13 for the four uses of one compressed object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DnPath {
    pub ns:     u64,
    pub heel:   u64,
    pub hip:    u64,
    pub branch: u64,
    pub twig:   u64,
    pub leaf:   u64,
}

impl DnPath {
    /// Parse from a slash-separated path string.
    ///
    /// Expected form: `tree/{ns}/heel/{h}/hip/{h}/branch/{b}/twig/{t}/leaf/{l}`
    /// Returns `None` if the path does not match the 12-segment pattern.
    pub fn parse(path: &str) -> Option<Self> {
        let segs: Vec<&str> = path.trim_start_matches('/').split('/').collect();
        if segs.len() < 12 { return None; }
        if segs[0] != "tree" || segs[2] != "heel" || segs[4] != "hip"
            || segs[6] != "branch" || segs[8] != "twig" || segs[10] != "leaf"
        { return None; }
        Some(Self {
            ns:     fnv1a(segs[1]),
            heel:   fnv1a(segs[3]),
            hip:    fnv1a(segs[5]),
            branch: fnv1a(segs[7]),
            twig:   fnv1a(segs[9]),
            leaf:   fnv1a(segs[11]),
        })
    }

    /// Phase-A scent stub: XOR-fold of the 6 segment hashes into 1 byte.
    ///
    /// Phase C replaces this with the full ZeckBF17→Base17→CAM-PQ chain
    /// (16Kbit → 48B → 34B → 6B → 1B, ρ=0.937). Until then, this gives
    /// a deterministic, stable placeholder that exercises the scent field.
    pub fn scent_stub(&self) -> u8 {
        let fold = self.ns ^ self.heel ^ self.hip ^ self.branch ^ self.twig ^ self.leaf;
        (fold
            ^ (fold >> 8)
            ^ (fold >> 16)
            ^ (fold >> 24)
            ^ (fold >> 32)
            ^ (fold >> 40)
            ^ (fold >> 48)
            ^ (fold >> 56)) as u8
    }
}

/// FNV-1a 64-bit — zero-dep, stable for the same input string.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_full_dn_path() {
        let p = DnPath::parse(
            "/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/abc",
        )
        .unwrap();
        assert_ne!(p.ns, 0);
        assert_ne!(p.leaf, 0);
    }

    #[test]
    fn parse_rejects_short_path() {
        assert!(DnPath::parse("/tree/ada/heel/callcenter").is_none());
    }

    #[test]
    fn parse_rejects_wrong_keywords() {
        // "hip" replaced by "HIP" — case-sensitive
        assert!(DnPath::parse(
            "/tree/ada/heel/cc/HIP/v1/branch/agents/twig/card/leaf/abc"
        )
        .is_none());
    }

    #[test]
    fn scent_stub_is_deterministic() {
        let p1 = DnPath::parse(
            "/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/abc",
        )
        .unwrap();
        let p2 = DnPath::parse(
            "/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/abc",
        )
        .unwrap();
        assert_eq!(p1.scent_stub(), p2.scent_stub());
    }

    #[test]
    fn different_paths_typically_differ() {
        let p1 = DnPath::parse(
            "/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/abc",
        )
        .unwrap();
        let p2 = DnPath::parse(
            "/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/xyz",
        )
        .unwrap();
        // leaf differs → scent should differ (not guaranteed but very likely for FNV-1a)
        assert_ne!(p1.leaf, p2.leaf);
    }
}
