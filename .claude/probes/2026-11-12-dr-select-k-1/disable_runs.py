#!/usr/bin/env python3
"""Disable runs for crates/dr-select-k-lab: break one guard, expect red.

Run from the repository root on a clean, committed tree. Each case asserts
that its anchor occurs exactly once, applies the patch, runs the tests,
records whether they failed, and restores the file from git.
"""
import subprocess, sys, os
CRATE = "crates/dr-select-k-lab"
CASES = [
    ("admissible ignores the order witness", "src/ops.rs",
     "if order != Order::Source && bound.order() == order {", "if true {"),
    ("mask residual uses > instead of >=", "src/ops.rs",
     "bits |= u64::from(v >= t) << j;", "bits |= u64::from(v > t) << j;"),
    ("positions residual uses > instead of >=", "src/ops.rs",
     ".filter(|&i| a[i as usize] >= t)", ".filter(|&i| a[i as usize] > t)"),
    ("masked count ignores the mask", "src/ops.rs",
     "under: if gated { Some(Operand::Plane(0)) } else { None },", "under: None,"),
    ("top-T over positions takes one row too many", "src/ops.rs",
     "View::Positions(p) => out.extend(p.iter().take(limit)", "View::Positions(p) => out.extend(p.iter().take(limit + 1)"),
    ("order index breaks ties by row, not guid", "src/source.rs",
     "by_key_guid.sort_unstable_by_key(|&r| (key[r as usize], guid[r as usize]));",
     "by_key_guid.sort_unstable_by_key(|&r| (key[r as usize], r));"),
    ("the row-tie index breaks ties by guid", "src/source.rs",
     "by_key_row.sort_unstable_by_key(|&r| (key[r as usize], r));",
     "by_key_row.sort_unstable_by_key(|&r| (key[r as usize], guid[r as usize]));"),
    ("bind accepts a stale selection", "src/bind.rs",
     "if selection.version != source.version() {", "if false {"),
    ("staging is not counted as a storage access", "src/source.rs",
     "    pub(crate) fn take(&self, rows: &[u32], counters: &Counters) -> Staged {\n        Counters::bump(&counters.storage, 1);",
     "    pub(crate) fn take(&self, rows: &[u32], counters: &Counters) -> Staged {"),
    ("a fold bumps the storage counter", "src/ops.rs",
     "Counters::bump(&counters.operand_rows, out.len() as u64);",
     "Counters::bump(&counters.operand_rows, out.len() as u64); Counters::bump(&counters.storage, 1);"),
    ("ordered selection drops the last row of the range", "src/source.rs",
     "let rows = index[lo..hi.max(lo)].to_vec();",
     "let rows = index[lo..hi.max(lo).saturating_sub(1).max(lo)].to_vec();"),
    ("filter does not count its position copy", "src/source.rs",
     ".collect();\n        Counters::bump(&counters.bytes_copied, (rows.len() * 4) as u64);",
     ".collect();"),
    ("ordered does not count its position copy", "src/source.rs",
     "let rows = index[lo..hi.max(lo)].to_vec();\n        Counters::bump(&counters.bytes_copied, (rows.len() * 4) as u64);",
     "let rows = index[lo..hi.max(lo)].to_vec();"),
]
env = dict(os.environ, CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0", CARGO_INCREMENTAL="0")
env.pop("RUSTFLAGS", None)
red = 0
for name, rel, old, new in CASES:
    path = os.path.join(CRATE, rel)
    src = open(path).read()
    assert src.count(old) == 1, f"anchor moved: {name}"
    open(path, "w").write(src.replace(old, new))
    r = subprocess.run(["cargo", "test", "--manifest-path", f"{CRATE}/Cargo.toml", "--tests"],
                       env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    built = "test result:" in out
    failed = [l.split()[1] for l in out.splitlines() if l.startswith("test ") and l.endswith("FAILED")]
    subprocess.run(["git", "checkout", "--", path], check=True)
    verdict = "RED" if (built and r.returncode != 0 and failed) else ("DID-NOT-BUILD" if not built else "GREEN")
    red += verdict == "RED"
    print(f"{verdict}\t{name}\t{len(failed)} failed\t{','.join(failed[:3])}")
print(f"# {red} of {len(CASES)} red")
sys.exit(0 if red == len(CASES) else 1)
