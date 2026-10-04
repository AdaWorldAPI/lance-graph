# D-HPS-2 — resolve once, then run the population (2026-10-04)

**STATUS:** measured (branch `ccr-f6094d67-h6ulb3`, unmerged)

Follows `D-HPS-1`. The metadata is resolved once per population, and the row
loops do no registry / ClassView / read-mode lookup.

- `soa_graph`: `project_snapshot` and `nearest_anchor` resolve the domain's
  `TailVariant` once (`domain_tail`) and pass it to `hhtl_path` / `family_of` /
  `identity_of`. Before: one `classid_read_mode` per helper call per row.
- `nan_projection`: new `project_energy_nonfinite_resolved(rows, schema)` and
  `energy_all_finite_resolved(rows, schema)` take the population's
  `ValueSchema` (e.g. `ResolvedReading::read_mode.value_schema`) and do no
  lookup. The mixed API resolves once per run of equal classid and delegates
  to them; its per-row residue is one classid compare. The schema gate and the
  exponent-mask test are unchanged.
- Falsifiers: a test-only counter in `classid_read_mode` pins lookups at 1 for
  both 1 and 1000 rows (soa_graph, mixed wrapper) and at 0 for the resolved
  path. Disable runs, all red: per-row lookup in `hhtl_path`; per-row lookup in
  the resolved loop; schema gate dropped; mixed wrapper reduced to runs of 1.
- Measured (release, `target-cpu=native`, 100k homogeneous rows):
  per-row lookup 23.7–24.6 ns/row, resolved 5.0 ns/row, mixed wrapper
  10.0–10.3 ns/row. soa_graph was not timed.
- The 128-bit matcher lives in ndarray (`ternary_match_strided16_to_mask`,
  ndarray #340). 12 B vs 16 B at stride 512, 65,536 rows: 5.1–5.3 ns/row for
  both; no measurable cost.

**OPEN:**
- mask-risc does not expose the 16 B matcher until ndarray #340 merges.
- The bake paths were not changed: q2 `osint-bake/src/bin/fma.rs` does a
  homogeneous per-node `classid_read_mode(CLASSID_FMA)` (another repo), and
  deepnsm-v2 `promote.rs` `key_at` does one per call.
- symbiont `domino.rs` is not migrated: its boards are classid 0, so the mixed
  wrapper already costs it one lookup.
