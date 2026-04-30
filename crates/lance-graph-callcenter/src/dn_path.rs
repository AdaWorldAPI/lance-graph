//! DN-addressed path — the URL hierarchy that becomes a scent.
//!
//! `tree/{ns}/heel/{h}/hip/{h}/branch/{b}/twig/{t}/leaf/{l}`
//!
//! Each segment is a u64 FNV-1a hash of the path component. The 6-tuple
//! compresses via ZeckBF17→Base17→CAM-PQ→scent (1B, ρ=0.937).
//! Phase C wires the full compression chain. The scent is computed by
//! FNV-1a hashing the canonical hex representation of the 6 segment
//! hashes and XOR-folding the 64-bit digest to 1 byte.
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

    /// 64-bit FNV-1a digest over the canonical hex path.
    ///
    /// The canonical form is the 6 segment hashes rendered as hex and
    /// concatenated with `/` separators (deterministic, stable, zero-dep).
    /// CAM-PQ stages downstream (HHTL Phase C) keep the full bits;
    /// [`scent()`](Self::scent) folds this to u8 for HHTL Phase A bucket
    /// dispatch.
    pub fn scent_u64(&self) -> u64 {
        use core::fmt::Write;
        let mut buf = String::with_capacity(6 * 17);
        let segments = [self.ns, self.heel, self.hip, self.branch, self.twig, self.leaf];
        for (i, seg) in segments.iter().enumerate() {
            if i > 0 { buf.push('/'); }
            let _ = write!(buf, "{:016x}", seg);
        }
        fnv1a(&buf)
    }

    /// Compute the scent of this DN path: FNV-1a hash of the canonical
    /// path string, folded to a single `u8`.
    ///
    /// XOR-folds [`scent_u64()`](Self::scent_u64) (64 → 8 bits), preserving
    /// avalanche properties much better than the old XOR-fold of individual
    /// segment hashes.
    ///
    /// Future phases may replace this with ZeckBF17→Base17→CAM-PQ
    /// (16Kbit → 48B → 34B → 6B → 1B, ρ=0.937) once bgz-tensor
    /// enters the callcenter dep tree.
    pub fn scent(&self) -> u8 {
        let h = self.scent_u64();
        let folded = h
            ^ (h >> 8)
            ^ (h >> 16)
            ^ (h >> 24)
            ^ (h >> 32)
            ^ (h >> 40)
            ^ (h >> 48)
            ^ (h >> 56);
        folded as u8
    }

    /// Deprecated alias — use [`scent()`](Self::scent) instead.
    #[deprecated(since = "0.1.1", note = "renamed to `scent()`; the XOR-fold stub has been replaced with FNV-1a")]
    pub fn scent_stub(&self) -> u8 {
        self.scent()
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
    fn scent_is_deterministic() {
        let p1 = DnPath::parse(
            "/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/abc",
        )
        .unwrap();
        let p2 = DnPath::parse(
            "/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/abc",
        )
        .unwrap();
        assert_eq!(p1.scent(), p2.scent());
    }

    #[test]
    fn different_paths_different_scents() {
        let p1 = DnPath::parse(
            "/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/abc",
        )
        .unwrap();
        let p2 = DnPath::parse(
            "/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/xyz",
        )
        .unwrap();
        assert_ne!(p1.scent(), p2.scent());
    }

    #[test]
    fn empty_path_scent() {
        let p = DnPath::default();
        let s1 = p.scent();
        let s2 = p.scent();
        assert_eq!(s1, s2);
    }

    #[allow(deprecated)]
    #[test]
    fn scent_stub_alias_matches_scent() {
        let p = DnPath::parse(
            "/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/abc",
        )
        .unwrap();
        assert_eq!(p.scent_stub(), p.scent());
    }

    #[test]
    fn scent_u64_fold_matches_scent() {
        let p = DnPath::parse(
            "/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/abc",
        )
        .unwrap();
        let h = p.scent_u64();
        let folded = (h
            ^ (h >> 8)
            ^ (h >> 16)
            ^ (h >> 24)
            ^ (h >> 32)
            ^ (h >> 40)
            ^ (h >> 48)
            ^ (h >> 56)) as u8;
        assert_eq!(folded, p.scent());
    }

    #[test]
    fn scent_distribution_100_paths_low_collision() {
        let paths: Vec<DnPath> = (0..100)
            .map(|i| {
                DnPath::parse(&format!(
                    "/tree/tenant/heel/agent_{i}/hip/session_{i}/branch/leaf_{i}/twig/t_{i}/leaf/l_{i}"
                ))
                .unwrap()
            })
            .collect();
        let scents: Vec<u8> = paths.iter().map(|p| p.scent()).collect();
        let unique: std::collections::HashSet<_> = scents.iter().copied().collect();
        assert!(
            unique.len() >= 50,
            "FNV-1a XOR-fold should distribute >=50 unique buckets across 100 distinct paths, got {}",
            unique.len()
        );
    }

    #[test]
    fn scent_u64_distribution_100_paths_all_unique() {
        let paths: Vec<DnPath> = (0..100)
            .map(|i| {
                DnPath::parse(&format!(
                    "/tree/tenant/heel/agent_{i}/hip/session_{i}/branch/leaf_{i}/twig/t_{i}/leaf/l_{i}"
                ))
                .unwrap()
            })
            .collect();
        let scents: std::collections::HashSet<u64> =
            paths.iter().map(|p| p.scent_u64()).collect();
        assert_eq!(
            scents.len(),
            100,
            "scent_u64 in 64-bit codomain should have zero collisions in 100 paths"
        );
    }
}
