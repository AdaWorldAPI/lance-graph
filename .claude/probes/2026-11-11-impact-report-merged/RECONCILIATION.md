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
  in the store.

Decided in the merge, in neither PR as such:
- The crates the answer depends on are the owner, every stored crate with a function in the
  answer, and the stored crates that can be consumers. Those are the declared ones
  (`Declared`), or every stored crate when the consumers are not named (`NotDeclared`,
  `AllInStore`) and the symbol or one of its callers is public. PR 2002 checked every stored
  crate only under `consumers_closed`; with undeclared consumers a stale stored crate that could
  call the symbol was not reported.
- PR 2002's dynamic-call gap for an empty `graded.unknown` is KEPT (symbol is an impl or is
  referenced, and the store holds a dynamic call). The first merged commit dropped it on the
  argument that `impact` then already lists every function holding a dynamic call. That argument
  is about positions only: a caller that is confirmed through a direct call says nothing about
  where its dynamic call goes. Restored after the operator's review, with a fixture that isolates
  the case (`a_dynamic_call_from_an_already_confirmed_caller_is_still_a_gap`): the gap is
  `DynamicCalls { unknown: 0, store_dynamic_calls: 1 }`. The earlier note "no fixture store
  produced the case" covered only the five committed stores; a four-row synthetic store does.
- `ReferrersOutsideAnswer { referrers }` is new, from the PR 2002 session's acceptance review
  (PR 2004, D-CODEGRAPH-RECON-ACCEPT, finding 2b). A function that takes the address of the
  symbol, or of a function in the answer, without calling it is in no mask, because a
  `REFERENCES` row is not a call edge. Both PRs then reported "closed"; PR 2002's dynamic-call
  half hid the hole only when the store happened to hold a dynamic call. The gap counts the
  distinct referrers outside the answer and needs no dynamic call. On the report-rs store it
  fires for `prefs::Layout::label` (`frame::tools_html` references it and never calls it).
- `TraitDispatch` (`graded.possible`) and `DynamicCalls` (`graded.unknown` or a reachable
  dynamic call) stay two gaps. Neither PR had both.

## CLI

`cg_query <store> <src> impact-report SYMBOL [--deps a,b | --consumers-closed]
[--targets CRATE=lib,tests,examples,benches]... [--exclude CRATE=REGION]... [--no-freshness]`

- `report` is kept as an alias (PR 2002's name).
- `--targets '*=LIST'` declares every stored crate. PR 2001's `--with-tests` is gone; a crate
  without `--targets` is reported as undeclared instead of assumed library-only.
- Output lines: `symbol`, `impact`, `observed`, `GAP`, `closed_in_scope`. A missing symbol exits 4.
- PR 2002's recorded `store-output/*.report` files are regenerated in `report-rs-33/`; see
  its README for the line-by-line difference.

## Tests

`tests/impact_coverage.rs` holds 36 tests on the merged types:

- 12 from PR 2001 and 14 from PR 2002, each with its original intent.
- 3 narrowing tests: adding a declared consumer, declaring test targets, and removing an
  excluded region each remove their gap.
- 3 added after the disable runs showed a guard without a test: a deleted source file is
  `MissingSource` and not `StaleSource`; a declared consumer with no caller in the answer still
  needs its own target declaration; a dynamic call from an already confirmed caller is still a
  gap.
- 4 from the acceptance review: a function that only takes the address is a gap, with and
  without a dynamic call in the store; a referrer that also calls is in the answer and no gap;
  only references into the answer are counted; undeclared consumers inspect every stored crate
  and `graded` stays equal to `impact`.

One PR 2001 test changed what it queries. `a_changed_source_file_reports_stale_coverage` edited
a `cg_lib` file and asked about `cg_app::run` with "nothing depends on `cg_app`" declared. Under
the scoped rule that edit cannot move the answer, so the test now asks about a `cg_lib` symbol,
asserts the declared-scope case reports nothing for `cg_app::run`, and asserts the gap returns
when the consumers are undeclared.

Disable runs are reproducible: `python3 .claude/probes/2026-11-11-impact-report-merged/disable_runs.py`
from the repository root, on a clean tree. It patches each of 27 guards out through an anchor
that must occur exactly once, runs the test file, restores the source, and exits 0 only when
every guard is red. The last run is `disable_runs.receipt.tsv` (27 guards, 0 not red), which
names the tests each guard breaks.

## Acceptance state (D-CODEGRAPH-WAVE0-ACCEPTANCE)

- PR 2002's report-rs outputs against the merged API: regenerated from a fresh extraction of
  report-rs `81705d9`, in `report-rs-33/`. Four of six files are identical to PR 2002's; one has
  the renamed dynamic-call line; one gains the harvest gap PR 2002's own later test requires.
- `member-tests` timeout: not caused by this change. On `main` the job took 19.7, 20.5, 21.1,
  26.4 and 29.8 minutes over five runs against `timeout-minutes: 30`. PR 2001's head hit the
  limit twice (30.3 and 30.2 minutes) with every step slower than on `main`, not one outlier; an
  earlier head of the same PR passed in 24.7 and PR 2002 passed in 25.6. The job runs about
  twenty `cargo test --manifest-path` steps, and the hydrate and surreal_container steps each
  compile the arrow / datafusion / lance stack again. Raising the limit or splitting those steps
  is a CI change for its own PR.
