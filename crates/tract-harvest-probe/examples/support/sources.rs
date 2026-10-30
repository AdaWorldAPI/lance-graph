#![allow(dead_code)]
//! Locating the Tract and ndarray sources the probe builds with, from the
//! git checkouts this crate's `Cargo.lock` names.
use std::path::{Path, PathBuf};

pub fn lock_rev(lock: &str, url_part: &str) -> Option<String> {
    lock.lines()
        .filter(|l| l.starts_with("source = \"git+") && l.contains(url_part))
        .find_map(|l| {
            l.rsplit('#')
                .next()
                .map(|r| r.trim_end_matches('"').to_string())
        })
}

pub fn checkout(prefix: &str, rev: &str) -> Option<PathBuf> {
    let home = std::env::var("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join(".cargo"));
    let base = home.join("git/checkouts");
    std::fs::read_dir(&base).ok()?.flatten().find_map(|d| {
        let name = d.file_name().to_string_lossy().to_string();
        if !name.starts_with(prefix) {
            return None;
        }
        let p = d.path().join(&rev[..7]);
        p.exists().then_some(p)
    })
}

pub fn rs_files(dir: &Path, out: &mut Vec<(String, String)>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            if let Ok(t) = std::fs::read_to_string(&p) {
                out.push((p.to_string_lossy().to_string(), t));
            }
        }
    }
}

/// `(tract onnx/src files, ndarray src files)` as `(path, text)`.
/// `(path, text)` per source file.
pub type Files = Vec<(String, String)>;

pub fn load(crate_dir: &Path) -> (Files, Files) {
    let lock = std::fs::read_to_string(crate_dir.join("Cargo.lock")).unwrap_or_default();
    let mut t = Vec::new();
    let mut n = Vec::new();
    if let Some(d) = lock_rev(&lock, "/tract").and_then(|r| checkout("tract-", &r)) {
        rs_files(&d.join("onnx/src"), &mut t);
    }
    if let Some(d) = lock_rev(&lock, "/ndarray").and_then(|r| checkout("ndarray-", &r)) {
        rs_files(&d.join("src"), &mut n);
    }
    (t, n)
}
