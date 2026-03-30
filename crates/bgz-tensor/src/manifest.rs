//! Manifest + hydration helpers (feature-gated behind `hydrate`).
//!
//! The library itself is zero-dep. This module only compiles when
//! `--features hydrate` is active (for the `hydrate` binary).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

/// Where bgz-tensor data lives relative to crate root.
pub const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data");
pub const PALETTES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/palettes");

#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub models: HashMap<String, ModelEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelEntry {
    pub source: String,
    pub format: String,
    pub shards: usize,
    pub total_bytes_bgz7: u64,
    pub release_tag: String,
    pub sha256: HashMap<String, String>,
}

/// Runtime path to a bgz7 shard. Compiles without the file existing.
pub fn bgz7_path(model: &str, shard: usize) -> PathBuf {
    Path::new(DATA_DIR)
        .join(model)
        .join(format!("shard-{shard:02}.bgz7"))
}

/// Check if a model's data is hydrated (all shards present and non-empty).
pub fn is_hydrated(model: &str, shard_count: usize) -> bool {
    (0..shard_count).all(|i| {
        let p = bgz7_path(model, i);
        p.exists() && std::fs::metadata(&p).map(|m| m.len() > 0).unwrap_or(false)
    })
}

/// Load manifest from data/manifest.json.
pub fn load_manifest() -> io::Result<Manifest> {
    let path = Path::new(DATA_DIR).join("manifest.json");
    let data = std::fs::read_to_string(&path)?;
    serde_json::from_str(&data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Read a palette file (always present, committed to git).
pub fn read_palette(name: &str) -> io::Result<Vec<u8>> {
    let path = Path::new(PALETTES_DIR).join(name);
    std::fs::read(&path)
}

/// Verify SHA256 of a file against expected hash.
pub fn verify_sha256(path: &Path, expected: &str) -> io::Result<bool> {
    use sha2::{Digest, Sha256};
    let data = std::fs::read(path)?;
    let hash = format!("{:x}", Sha256::digest(&data));
    Ok(hash == expected)
}
