# Kythe readings from rust-analyzer facts

This is the bridge from witnessed semantic facts to Kythe-compatible readings. The
facts are the occurrences rust-analyzer's `StaticIndex` records. Nothing is
written by hand except the pin file, which is the oracle.

```
XREFS=1 target/release/sag-ra-probe fixtures/sag_fixture > occ.tsv       # witnessed facts
python3 -I tools/kythe_entries.py --root fixtures/sag_fixture occ.tsv > e.json  # Kythe entries
python3 -I tools/kythe_xrefs_check.py --tools KYTHE_TOOLS --root fixtures/sag_fixture \
    --occ occ.tsv --pin fixtures/sag_fixture_xrefs.pin.tsv               # /xrefs, compared back
python3 -I tools/test_kythe_entries.py                                   # producer unit tests
```

`KYTHE_TOOLS` holds `entrystream`, `write_tables` and `http_server` from the Kythe
v0.0.76 release. The check pipes the entries through
`entrystream --read_format=json --sort --write_format=delimited`, builds serving
tables with `write_tables`, starts `http_server`, and asks `/xrefs` for all
definitions and all references of every symbol.

## Result on `fixtures/sag_fixture` (2026-10-31)

- 138 occurrence rows, 591 Kythe entries. Four `Attribute` symbols are
  skipped: they have no Kythe node kind in the table.
- **Kythe agrees with rust-analyzer on all 24 symbols.** For each one,
  `/xrefs` returns exactly the occurrence set rust-analyzer recorded, for
  definitions and references separately.
- **8 of the 11 hand-read pins pass. The 3 that fail are gaps in rust-analyzer's
  facts, not in the conversion.** They are the references to `count_below`,
  `popcount_mask` and `mask_below` inside `assert_eq!(...)` in `tests/law.rs`,
  lines 21 and 31. `StaticIndex` records none of them, so no producer working
  from its output can emit them. The check therefore exits 1, and that is the
  correct outcome.
- **Control:** the calls on line 21 were moved out of the macro
  (`let l = count_below(..); let r = ..; assert_eq!(l, r);`). Rust-analyzer then
  records all three; the calls left inside `assert_eq!` on line 31 stay missing.
  On the same file, calls outside a macro (`count_with_tally`, `count_div`) and
  `assert_eq` itself are recorded. `StaticIndex::add_file` does call
  `descend_into_macros_exact` on each token (ra_ap_ide 0.0.358
  `static_index.rs`), so why the descent fails for these arguments is not yet
  known.

## Not produced

- **No `ref/call` edges.** A reference is not a call. Call edges need a source
  of call facts (Charon, which is not installed here) and are not inferred from
  references.
- No hover, documentation or type facts.
