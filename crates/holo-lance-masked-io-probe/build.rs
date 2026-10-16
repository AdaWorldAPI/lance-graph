//! Records the `lance` version this build resolved, so the probe prints what
//! it was built against. The repository tracks no `Cargo.lock`; cargo writes
//! one next to this excluded crate's manifest before running this script.

use std::path::Path;

fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let lock = Path::new(&dir).join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let text = std::fs::read_to_string(&lock).unwrap_or_default();
    let mut version = "unknown".to_string();
    let mut in_lance = false;
    for line in text.lines() {
        if line == "[[package]]" {
            in_lance = false;
        } else if line == "name = \"lance\"" {
            in_lance = true;
        } else if in_lance {
            if let Some(v) = line.strip_prefix("version = \"") {
                version = v.trim_end_matches('"').to_string();
                break;
            }
        }
    }
    println!("cargo:rustc-env=PROBE_LANCE_VERSION={version}");
}
