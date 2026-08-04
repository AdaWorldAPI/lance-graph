## Silent-defaults IDENTITY-MERGE sweep (report-only, Sonnet)

- Task: sweep crates/**/*.rs for the `Copula::Rel(pid.parse().unwrap_or(0))`-shaped
  defect (a default applied to what is actually an IDENTITY, silently merging
  distinct inputs into one shared key/index/discriminant).
- Scoped to the brief's stated antecedents: `.parse()`/`.try_into()`/
  `TryFrom::try_from()`/map-dictionary-lookup (`.get(&key)`)/index-resolution
  (`.position`/`.find`/`.binary_search`) feeding `.unwrap_or*`. Exact uncapped
  counts via Grep count-mode: parse=33, try_into=5, TryFrom=0, get(&..)=55,
  position/find/binary_search=38 → 131 total, every one read in context.
- Unscoped raw `.unwrap_or(`/`.unwrap_or_default()`/`.unwrap_or_else(` totals
  (NOT individually read, out of the brief's antecedent scope): 1037/108/207
  = 1352 across crates/.
- Class counts (sum to 131): IDENTITY-MERGE=12, MEASURE-DEFAULT=71,
  PROVEN-SAFE=48. `crates/jc` hits (4) folded into MEASURE/PROVEN-SAFE,
  excluded from ranking per instructions.
- Top-ranked finding: `lance-graph-callcenter/src/bin/audit_verify.rs:745`
  — `event_merkle_str.parse().unwrap_or(0)` in the audit tamper-evidence
  cross-verify tool; a malformed merkle string silently becomes identity 0
  and feeds the published "OK (both, matching merkle)" / "JSONL-only" /
  "Lance-only" counts — same defect shape as the confirmed
  `blw_lens_twin.rs` bug, but in a security-relevant tool. Second-ranked:
  `lance-graph-planner/examples/reason_whole_book.rs:88` — the almost
  byte-identical recurrence of the fixed defect
  (`Copula::Rel(_pid.parse::<u16>().unwrap_or(0))`), not previously caught.
- Full ranked table (12 rows) + full per-hit classification table (131 rows)
  + fixes: `/tmp/audit_silent_defaults.md`.
- Discipline: report-only, no `.rs` edits, no cargo run. Did not read all
  1352 unscoped raw hits (see report's "What I did NOT do").
