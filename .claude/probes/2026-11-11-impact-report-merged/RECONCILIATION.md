# Impact report: one API from PR 2001 and PR 2002

Ruling: D-CODEGRAPH-BOARD-RULING-2, Wave 0 (Linear EXO-232). Both PRs added
`Codegraph::impact_report` in `crates/lance-graph-codegraph/src/impact_report.rs` with different
types. This branch carries both PRs' commits and one implementation. `Codegraph::impact` is
unchanged in both and here.

The two reports (`../2026-11-11-crai-report-1/REPORT.md`, `../2026-11-11-codegraph-recon-2/REPORT.md`)
keep the names their PR used. This table maps them.

## Types

| PR 2001 | PR 2002 | Merged |
|---|---|---|
| `ExtractionScope<'a>` (borrowed) | `Scope` (owned) | `Scope` (owned) |
| `Targets::{LibraryOnly, LibraryAndTests}`, one value for the whole store | `Targets { lib, tests, examples, benches }` per crate | PR 2002's, per crate |
| `Dependents::{NotDeclared, Declared(&[..])}` | `consumers_closed: bool` | `Consumers::{NotDeclared, Declared(Vec), AllInStore}` |
| — | `excluded: Vec<(crate, regions)>` | kept |
| `ImpactAnswer::{Found(report), SymbolNotInStore}` | `ImpactReport.symbol: SymbolState` | `SymbolState`, plus PR 2001's `observed` crate list on the report |
| `no_known_gap()` | `closed_in_scope()` | `closed_in_scope()` |

## Gaps

| PR 2001 `CoverageGap` | PR 2002 `Gap` | Merged `Gap` |
|---|---|---|
| `ConsumersNotDeclared` (symbol public) | `ExternalConsumersUnobserved` (symbol or a caller public) | `ExternalConsumersUnobserved` |
| `ConsumersNotInStore { crates }` | — | kept |
| `TestTargetsNotExtracted` | `TargetsNotExtracted`, `TargetsUndeclared` | PR 2002's |
| — | `RegionsExcluded` | kept |
| `UnresolvedDispatch { possible }` | — | `TraitDispatch { possible }` |
| `DynamicCalls { unknown }` | `UnresolvedDispatch { dynamic_calls }` | `DynamicCalls { unknown, store_dynamic_calls }` |
| `PartialCrate`, `Residuals`, `ResidualsNotRead` | `HarvestIncomplete` | `HarvestIncomplete` |
| `StaleSource`, `MissingSource` (whole store) | `StaleSource` (per crate; a vanished file counted as stale) | both, per crate |
| — | `SourceFilesUnread` | kept |
| `NoFileList` | `FilesUnrecorded` | `FilesUnrecorded` |
| `InvalidFileList` | — (a short row counted as stale) | `InvalidFileList`, also for a table without `#schema` |
| `FreshnessNotChecked` | `FreshnessUnchecked` | `FreshnessUnchecked` |

## Behaviour taken from each

From PR 2001:
- A symbol whose rows a refresh removed is `Missing` (`has_rows`), not found with an empty answer.
- A crate read without a CRATE row is listed and counts as partial.
- Declared consumers are named, so a consumer missing from the store is reported by name.
- `files.tsv` is read by its `#schema` columns; the digest is `lance_graph_contract::hash::fnv1a`.

From PR 2002:
- Targets are declared per crate; an undeclared crate is a gap.
- Excluded regions and source files the extractor never read are gaps.
- The external-consumer gap also fires when a caller of the symbol is public.
- A missing symbol in a stored crate carries that crate's target, region, harvest and file gaps.
- Harvest and file gaps are reported for the crates the answer depends on, not for every crate
  in the store. Under `Consumers::AllInStore` with a public frontier that is every stored crate.

Removed duplicates: PR 2002's `stored_crates`, `residuals_of_crate` and `artifact::fnv64_hex`
(PR 2001's `crate_list`, `crate_residuals` and the contract `fnv1a` do the same jobs).

## CLI

`cg_query <store> <src> impact-report SYMBOL [--deps a,b | --consumers-closed]
[--targets CRATE=lib,tests,examples,benches]... [--exclude CRATE=REGION]... [--no-freshness]`

- `report` is kept as an alias (PR 2002's name).
- `--targets '*=LIST'` declares every stored crate. PR 2001's `--with-tests` is gone; a crate
  without `--targets` is reported as undeclared instead of assumed library-only.
- Output lines: `symbol`, `impact`, `observed`, `GAP`, `closed_in_scope`. A missing symbol exits 4.
- PR 2002's recorded `store-output/*.report` files were produced by its own `report`; the one
  line that changes name is `GAP UnresolvedDispatch { dynamic_calls: N }`, now
  `GAP DynamicCalls { unknown: M, store_dynamic_calls: N }`. They are not regenerated here: the
  report-rs store they came from is not in this container.

## Tests

`tests/impact_coverage.rs` holds both suites on the merged types (12 from PR 2001, 14 from
PR 2002) and three narrowing tests: adding a declared consumer, declaring test targets, and
removing an excluded region each remove their gap.
