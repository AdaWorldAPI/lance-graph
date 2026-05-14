//! D-PARITY-V2-10 smoke test: assert the bin parses and runs to completion.
//! classification: bare-metal

use std::process::Command;

#[test]
fn check_runs_and_scans_workspace() {
    let bin = env!("CARGO_BIN_EXE_dto-class-check");
    let out = Command::new(bin).output().expect("bin runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Must produce a summary line over scanned types and find >= 22 ledger types.
    assert!(stdout.contains("scanned:"), "stdout: {stdout}");
    let n: usize = stdout
        .lines()
        .find_map(|l| {
            l.strip_prefix("scanned: ")
                .and_then(|s| s.split(' ').next())
                .and_then(|s| s.trim_end_matches(';').parse().ok())
        })
        .unwrap_or(0);
    assert!(
        n >= 22,
        "expected >= 22 scanned types, got {n}; stdout: {stdout}"
    );
}
