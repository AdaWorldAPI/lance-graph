# D-RPR-1 handoff: escape-driven lookup in preparation, legal fold (2026-10-20)

**Status:** MEASURED, lab only. **Basis:** `crates/mork-pathmap-probe` at `2aaf6e7`: `tests/chain.rs` 8/8, `examples/chain_cost.rs`. Details are in `recursive-perspective-rebinding-v1.md` §11.10.

- **Operation chain:**
  - the escapes of the `TileExec::reach` fold (values from the CSR escape list) are the next addresses;
  - `rebind` consumes them as `SortedRun::neighbours` seeks, one per changed slot and predicate;
  - `Focus::slot_of` translates them inside preparation.
- **The fold:** it contains no Gather, no scatter and no foreign key. Its co-addressing is checked on the program itself.
- **Oracle:** every answer equals the full-scan oracle. Incremental and full preparation give equal frames.
- **Costs:** 360 incremental seeks against 3 840 for a full rebuild per admission, so 3 480 avoided. An incremental admission takes about 90–290 µs; the tile part of that is 2–4 µs, and the rest is mostly `rebind`'s CSR copy, not seeks. Folds take 2–6 µs.
- **Direct arm:** 66 seeks and 23.6 µs per pass, against 1 509 µs for the first framed pass; framing is repaid after about 85 re-asks. On this small chain it is BOUNDED, not a win.
- **Disable runs:** 3 of 4 red. Disabling `keep` makes `ask` livelock, so it hangs instead of failing.

OPEN: A1 / D-RMAP-1 §6 (is a counted preparation lookup admitted?); D-RPR-1 authorization; the `rebind` copy floor; the KJV fixture.
