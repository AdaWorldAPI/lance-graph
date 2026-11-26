# D-DIRSIM-DT64-PATHS-2 — one path parser, location to names, the lane allocations traced

**Status:** In PR. **Date:** 2026-11-26. Follows D-DIRSIM-DT64-PATHS (#2188).

## What

- `dt64::path_level(iri) -> Option<u8>` is public: the one parser of the
  `dir:ouPath/<k>` predicates, for the lanes and for every consumer. It
  allocates nothing.
- `ad::ou_names(&Dn128, &OuDictionary) -> Option<Vec<String>>`: a location
  read as its OU names straight from the dictionary, no DN rendered or
  parsed. `ad::project` uses it, so there is one implementation.

## Measured

The 3 allocations per `LaneBinding::lane()` call recorded open in #2188 were
dir-sim's, not DT64's: a lane check calls `lane_of` for every bound field,
and the old `path_level` compared the digits with `k.to_string()`. With
three path fields bound and three borrows, the old parser allocates 9 times
(the disable run reproduces exactly 9); the new one 0. The allocation gate in
`tests/dt64_paths.rs` now covers the borrow as well as the cell reads.

## Evidence

`tests/dt64_paths.rs` 5 tests, `tests/ad_projection.rs` 8. Disable runs, all
red: the old allocating parser (9 allocations), the leading-zero guard
removed (`/01` accepted), the code/segment offset in `ou_names` off by one
(the names test plus five projection tests).

## Open

- A consumer (report-rs `user_table`) still reads its values outside the
  lanes; that is report-rs H1.
