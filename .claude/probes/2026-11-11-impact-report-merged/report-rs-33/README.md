# PR 2002's report-rs outputs, regenerated with the merged API

Store: report-rs `81705d9` (the PR 33 head), re-extracted in this session with Charon `c69e231`
(nightly-2026-09-17, library target, default features, `--opaque crate::app`) and ruff_charon.
The import reported `harvested=29542 represented=29541 residual=1`, the numbers PR 2002 recorded.
The store is not committed (6 MB of rows); these six files are its output.

Command, for each symbol on the first line of the matching file in
`../../2026-11-11-codegraph-recon-2/store-output/`:

```
cg_query <store> <report-rs checkout> impact-report SYMBOL --targets report_rs=lib \
  --exclude report_rs=opaque:crate::app \
  --exclude report_rs=features-off:lance,iam,flow,oidc,projects,projects-pg
```

The flags were not recorded by PR 2002. They are read back from its `GAP` lines, and the result
below is the check that they are the right ones.

## Difference from PR 2002's recorded files

Line by line, after removing the new `observed` line:

| file | difference |
|---|---|
| `pull.report` | one line renamed: `UnresolvedDispatch { dynamic_calls: 17 }` is now `DynamicCalls { unknown: 101, store_dynamic_calls: 17 }`. All 102 `impact` lines and the five other gaps are identical. |
| `poll.report` | none |
| `srcpull.report` | none |
| `missing.authorize_memberships.report` | none |
| `missing.maplog.report` | none |
| `missing.report_rs_iam_authorize.report` | one line added: `HarvestIncomplete` for `report_rs` (1 residual). PR 2002's own later test `a_missing_symbol_carries_its_crate_harvest_gap` requires it; its recorded file predates that. |

None of the six symbols has a referrer outside its answer, so the `ReferrersOutsideAnswer` gap
added after the acceptance review does not appear in them. It does fire on this store for
`report_rs::prefs::{report_rs::prefs::Layout}::label` (`referrers: 1`).

Every file gains one `observed report_rs Complete` line (the stored-crate list from PR 2001). A
missing symbol now exits 4.

The `callers`, `callees` and `refs` files of PR 2002 come from queries this merge did not change
and are not regenerated.
