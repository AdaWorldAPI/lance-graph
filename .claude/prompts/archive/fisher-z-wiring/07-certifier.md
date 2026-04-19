# Step 7: Update Certification Probe for Fisher z

> Agent: certification-officer + truth-architect
> READ FIRST: certify_hhtld.rs, certification/hhtld_cosine_levels_17b.json
> Depends on: Steps 5, 6

## What to Do

Update `certify_hhtld.rs` to add a Fisher z cascade level.
The probe measures: for each pair (a, b), does
`fisher_z_table.lookup_f32(centroid_a, centroid_b)` preserve
the true f32 cosine at ρ≥0.999?

### New measurement level:

```
Level 1: Base17 fold (17 dims) — reconstruction cosine
Level 2: Palette centroid — reconstruction cosine  
Level 3: HHTL-D cascade — routing accuracy
Level 4: Fisher z table — pairwise cosine fidelity  ← NEW
```

### Metrics for Level 4:

- Spearman ρ vs true f32 cosine (target: ≥0.999)
- Pearson r vs true f32 cosine (target: ≥0.995)
- Mean absolute restore error (target: ≤0.002)
- Max restore error (report, no hard target)
- i8 value distribution entropy (target: >6.0 bits — full range used)

### Output:

Update the JSON report at `.claude/knowledge/certification/hhtld_cosine_levels_17b.json`
with the measured Fisher z values per role.

## Pass Criteria (from the probe session)

| Role | Spearman ρ | Restore err |
|------|-----------|-------------|
| All 21 | ≥0.995 | ≤0.002 |

These thresholds are based on the Python probe results.
The Rust probe should match within float precision.
