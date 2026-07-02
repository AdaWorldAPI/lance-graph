//! Deterministic 1BRC-format corpus generator.
//!
//! Produces `station;temp\n` lines with ~400 procedurally-generated station
//! names (invented syllables, NOT the upstream Java-1BRC city list — see
//! README "Reference inventory" for why: the archival-recipe contract wants
//! the corpus fully reconstructible from `(rows, seed)` alone, with no
//! external dataset dependency). Streams a SHA-256 digest while writing so
//! the recipe line (`rows=<N> seed=<S> sha256=<hash>`) never needs a second
//! read pass over the file.

use crate::sha256::Sha256;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Number of distinct stations in every generated corpus.
pub const STATION_COUNT: usize = 400;

/// Result of a `gen()` call — the archival recipe: everything needed to
/// reproduce the exact same corpus bytes again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenResult {
    pub rows: u64,
    pub seed: u64,
    pub sha256_hex: String,
}

/// SplitMix64 — a small, fast, well-distributed deterministic PRNG. Chosen
/// over `std`'s (nonexistent) RNG or a crates.io dependency because this
/// crate is dependency-free by design; SplitMix64 is a widely published
/// public-domain algorithm (Vigna 2015), reimplemented here in ~10 lines.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// Invented syllable pool used to build station names — purely synthetic,
/// no relation to any real-world gazetteer or the upstream 1BRC city list.
const SYLLABLES: &[&str] = &[
    "ka", "ri", "mo", "ta", "lu", "ven", "dor", "shi", "zan", "qui", "bel", "fen", "gor", "hu",
    "ith", "jol", "kra", "myn", "non", "oru", "pex", "ryn", "sae", "tuv", "uli", "vex", "wren",
    "xan", "yol", "zeph",
];

/// Deterministic, seed-derived station names. Same `(seed, count)` always
/// produces the same `Vec<String>` in the same order (the RNG stream and
/// insertion order are both deterministic; the `HashSet` below is used only
/// for a membership check, never iterated, so its unordered nature never
/// leaks into the output).
pub fn station_names(seed: u64, count: usize) -> Vec<String> {
    let mut rng = SplitMix64::new(seed ^ 0xA_11CE_5EED);
    let mut seen = std::collections::HashSet::with_capacity(count * 2);
    let mut names = Vec::with_capacity(count);
    while names.len() < count {
        let syl_count = 2 + (rng.next_u64() % 3) as usize; // 2..=4 syllables
        let mut raw = String::new();
        for _ in 0..syl_count {
            let idx = (rng.next_u64() as usize) % SYLLABLES.len();
            raw.push_str(SYLLABLES[idx]);
        }
        let mut chars = raw.chars();
        let name = match chars.next() {
            Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
            None => continue,
        };
        if seen.insert(name.clone()) {
            names.push(name);
        }
    }
    names
}

/// Format tenths-of-a-degree as a `[-]D+.D` string (one decimal place),
/// matching the exact 1BRC wire format the parser in `lib.rs` expects.
fn format_tenths(t: i32) -> String {
    let neg = t < 0;
    let abs = t.unsigned_abs();
    let whole = abs / 10;
    let frac = abs % 10;
    if neg {
        format!("-{whole}.{frac}")
    } else {
        format!("{whole}.{frac}")
    }
}

/// Generate a deterministic 1BRC-format corpus at `path`: `rows` lines of
/// `station;temp\n`, fully reproducible from `(rows, seed)`.
///
/// Per-station means are spread across `[-20.0, 20.0)` degrees, derived
/// from station INDEX (not the RNG stream) so the mean table is stable
/// regardless of how many random draws the row-generation loop performs.
/// Each row's temperature is `mean + variate` (variate uniform in
/// `[-10.0, 10.0]`), clamped to `[-99.9, 99.9]`.
pub fn gen(path: &Path, rows: u64, seed: u64) -> io::Result<GenResult> {
    let stations = station_names(seed, STATION_COUNT);
    let means: Vec<i32> = (0..stations.len())
        .map(|i| ((i * 131 + 37) % 400) as i32 - 200)
        .collect();

    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(1 << 20, file);
    let mut hasher = Sha256::new();
    let mut rng = SplitMix64::new(seed);
    let mut line_buf = Vec::with_capacity(32);

    for _ in 0..rows {
        let idx = (rng.next_u64() as usize) % stations.len();
        let variate = (rng.next_u64() % 201) as i32 - 100; // +-10.0 degrees, in tenths
        let tenths = (means[idx] + variate).clamp(-999, 999);

        line_buf.clear();
        line_buf.extend_from_slice(stations[idx].as_bytes());
        line_buf.push(b';');
        line_buf.extend_from_slice(format_tenths(tenths).as_bytes());
        line_buf.push(b'\n');

        writer.write_all(&line_buf)?;
        hasher.update(&line_buf);
    }
    writer.flush()?;

    let digest = hasher.finalize();
    Ok(GenResult {
        rows,
        seed,
        sha256_hex: Sha256::hex(&digest),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn station_names_are_deterministic_and_unique() {
        let a = station_names(42, STATION_COUNT);
        let b = station_names(42, STATION_COUNT);
        assert_eq!(a, b);
        assert_eq!(a.len(), STATION_COUNT);
        let unique: std::collections::HashSet<_> = a.iter().collect();
        assert_eq!(unique.len(), STATION_COUNT, "station names must be unique");
    }

    #[test]
    fn different_seeds_diverge() {
        let a = station_names(1, 50);
        let b = station_names(2, 50);
        assert_ne!(a, b);
    }

    #[test]
    fn format_tenths_examples() {
        assert_eq!(format_tenths(0), "0.0");
        assert_eq!(format_tenths(53), "5.3");
        assert_eq!(format_tenths(-53), "-5.3");
        assert_eq!(format_tenths(999), "99.9");
        assert_eq!(format_tenths(-999), "-99.9");
    }
}
