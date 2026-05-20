//! preflight_drift — board-claim vs. cargo-reality reconciliation.
//!
//! Compares numeric claims in `.claude/board/LATEST_STATE.md` and `CLAUDE.md`
//! against actual workspace state (Cargo.toml members/exclude, PR table rows).
//! Refuses sprint spawn when drift exceeds the threshold.
//!
//! Port of `WoA/.claude/v0.2/tools/preflight_drift.py` (Python AST) to the
//! Rust ecosystem. Zero dependencies — stdlib only, single-file. Build with
//!     rustc preflight_drift.rs -o preflight_drift
//! Run from repo root:
//!     ./preflight_drift                        # default threshold = 2
//!     ./preflight_drift --threshold 0          # zero-drift mode
//!     ./preflight_drift --metric crates        # only check one metric
//!
//! Exit codes:
//!     0  no drift above threshold — sprint spawn OK
//!     1  drift above threshold    — refuse to spawn
//!     2  parse error              — warn, do not block

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let mut threshold: i64 = 2;
    let mut only: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threshold" => {
                threshold = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2);
                i += 2;
            }
            "--metric" => {
                only = args.get(i + 1).cloned();
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!("{}", HELP);
                return ExitCode::from(0);
            }
            _ => i += 1,
        }
    }

    let repo = match find_repo_root() {
        Some(p) => p,
        None => {
            eprintln!("ERR: no Cargo.toml found walking up from cwd");
            return ExitCode::from(2);
        }
    };

    let cargo_toml = repo.join("Cargo.toml");
    let claude_md = repo.join("CLAUDE.md");
    let latest_state = repo.join(".claude/board/LATEST_STATE.md");

    let real = match real_metrics(&cargo_toml) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERR: cannot read {}: {e}", cargo_toml.display());
            return ExitCode::from(2);
        }
    };

    let claimed = claimed_metrics(&claude_md, &latest_state);

    let mut drift_rows: Vec<DriftRow> = Vec::new();
    for (key, real_val) in &real {
        if let Some(only_key) = &only {
            if only_key != key {
                continue;
            }
        }
        if let Some(claim_val) = claimed.get(key) {
            let diff = (*real_val - *claim_val).abs();
            drift_rows.push(DriftRow {
                key: key.clone(),
                claimed: *claim_val,
                real: *real_val,
                drift: diff,
            });
        } else {
            drift_rows.push(DriftRow {
                key: key.clone(),
                claimed: -1,
                real: *real_val,
                drift: -1, // unknown — claim missing
            });
        }
    }

    print_report(&drift_rows, threshold);
    if drift_rows.iter().any(|r| r.drift > threshold) {
        ExitCode::from(1)
    } else {
        ExitCode::from(0)
    }
}

const HELP: &str = "preflight_drift — board-claim vs. cargo-reality
  --threshold N   max acceptable drift per metric (default 2)
  --metric KEY    check only this metric (workspace_members|excluded|prs_in_table)
  -h, --help      this help";

struct DriftRow {
    key: String,
    claimed: i64,
    real: i64,
    drift: i64,
}

fn find_repo_root() -> Option<PathBuf> {
    let mut p = env::current_dir().ok()?;
    loop {
        if p.join("Cargo.toml").exists() {
            return Some(p);
        }
        if !p.pop() {
            return None;
        }
    }
}

fn real_metrics(cargo_toml: &Path) -> std::io::Result<Vec<(String, i64)>> {
    let text = fs::read_to_string(cargo_toml)?;
    let members = count_array_entries(&text, "members");
    let excluded = count_array_entries(&text, "exclude");
    Ok(vec![
        ("workspace_members".into(), members),
        ("excluded".into(), excluded),
        ("workspace_total".into(), members + excluded),
    ])
}

// Counts non-comment quoted strings inside the `<name> = [ ... ]` block.
// Tolerant of trailing commas, mixed indentation, and `#`-prefixed comment
// lines. Stops at the matching ']'.
fn count_array_entries(toml: &str, name: &str) -> i64 {
    let mut count: i64 = 0;
    let mut in_block = false;
    let mut depth: i32 = 0;
    for raw in toml.lines() {
        let line = raw.trim_start();
        if !in_block {
            if line.starts_with(&format!("{name} =")) || line.starts_with(&format!("{name}=")) {
                in_block = true;
                depth = bracket_delta(line);
                count += count_quoted(strip_comment(line));
                if depth == 0 && line.contains(']') {
                    in_block = false;
                }
            }
            continue;
        }
        let body = strip_comment(line);
        count += count_quoted(body);
        depth += bracket_delta(body);
        if depth <= 0 {
            in_block = false;
        }
    }
    count
}

fn strip_comment(line: &str) -> &str {
    // Naive: split on first '#' outside quotes. Sufficient for our Cargo.toml
    // shape (no `#` characters inside the quoted crate paths).
    let mut in_quote = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_quote = !in_quote,
            '#' if !in_quote => return &line[..i],
            _ => {}
        }
    }
    line
}

fn count_quoted(line: &str) -> i64 {
    // Even-numbered quote-count → quoted-string count is quotes/2.
    let q = line.bytes().filter(|&b| b == b'"').count() as i64;
    q / 2
}

fn bracket_delta(line: &str) -> i32 {
    let mut d = 0;
    let mut in_quote = false;
    for c in line.chars() {
        match c {
            '"' => in_quote = !in_quote,
            '[' if !in_quote => d += 1,
            ']' if !in_quote => d -= 1,
            _ => {}
        }
    }
    d
}

fn claimed_metrics(
    claude_md: &Path,
    latest_state: &Path,
) -> std::collections::HashMap<String, i64> {
    let mut out = std::collections::HashMap::new();
    if let Ok(text) = fs::read_to_string(claude_md) {
        // CLAUDE.md canonical phrase: "N crates, M in workspace, K excluded".
        // Match the first occurrence.
        if let Some(caps) = find_crates_phrase(&text) {
            out.insert("workspace_total".into(), caps.0);
            out.insert("workspace_members".into(), caps.1);
            out.insert("excluded".into(), caps.2);
        }
    }
    if let Ok(text) = fs::read_to_string(latest_state) {
        let prs = count_pr_rows(&text);
        if prs > 0 {
            out.insert("prs_in_table".into(), prs);
        }
    }
    out
}

fn find_crates_phrase(text: &str) -> Option<(i64, i64, i64)> {
    // Look for: "N crates, M in workspace, K excluded" in any line.
    // Tolerant of whitespace and surrounding markdown.
    for line in text.lines() {
        let l = line.to_ascii_lowercase();
        if !l.contains("crates") || !l.contains("in workspace") || !l.contains("excluded") {
            continue;
        }
        let nums: Vec<i64> = l
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|s| s.parse().ok())
            .collect();
        if nums.len() >= 3 {
            return Some((nums[0], nums[1], nums[2]));
        }
    }
    None
}

fn count_pr_rows(text: &str) -> i64 {
    // PR table rows in LATEST_STATE.md look like:
    //     | **#389** | 2026-05-16 | ...
    // Count lines starting with `| **#` and a digit.
    text.lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("| **#")
                && t.bytes()
                    .nth(5)
                    .map(|b| b.is_ascii_digit())
                    .unwrap_or(false)
        })
        .count() as i64
}

fn print_report(rows: &[DriftRow], threshold: i64) {
    let any_over = rows.iter().any(|r| r.drift > threshold);
    let banner = if any_over { "DRIFT — refuse to spawn" } else { "OK — within threshold" };
    println!("preflight_drift: {banner} (threshold={threshold})");
    println!("  {:<22} {:>10} {:>10} {:>10}", "metric", "claimed", "real", "drift");
    for r in rows {
        let claimed = if r.claimed < 0 { "—".to_string() } else { r.claimed.to_string() };
        let drift = if r.drift < 0 { "?".to_string() } else { r.drift.to_string() };
        let marker = if r.drift > threshold { "  <— OVER" } else { "" };
        println!(
            "  {:<22} {:>10} {:>10} {:>10}{}",
            r.key, claimed, r.real, drift, marker
        );
    }
}
