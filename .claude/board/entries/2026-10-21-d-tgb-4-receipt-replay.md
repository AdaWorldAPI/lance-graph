# 2026-10-21 — D-TGB-4: a receipt instead of history (lab)

**Status:** TEST-PINNED (lab probe `crates/thought-git-probe`, `src/receipt.rs`, `tests/receipt.rs`; 10 tests, 10 disable runs red). Not wired into `cycle_driver`, `LanceCycleWriter`, the supervisor or CE64. Claim: EXO-50 `session_019fbZ92dFKQ5BJHrA3YcH9A/003`; issue EXO-58.

## What landed

- `Receipt`: base ref (dataset, branch, version), program id with its template version, program inputs as two lanes (sources, targets) read at one position, seed, execution config, the live slot and generation (provenance only), a fold over the values read, provenance roots, a fold over the writes, and the outcome.
- `replay(receipt, store, config)` recomputes from the receipt alone; it takes no slot pool. It refuses on a missing seed, a changed config, an unknown base, changed sources (the source fold) or a different output (the output fold). Comparison is by folds; the writes are not kept in the receipt.
- `replay_live` and `MetadataTable::stage` check that the slot still holds the thought under the same generation.
- `MetadataTable`: landed receipts keyed by (dataset, data version, thought id). `stage` refuses before the data version is sealed (data first, then metadata) and on a key naming another thought. `seal` writes a fold over every field last. `landed` / `replay_landed` accept only a sealed row whose seal still matches and whose data version exists.
- `Replayed` carries the receipt's roots as inherited and 0 independent roots.
- CI: `.github/workflows/thought-git-probe.yml`. Before this, no workflow ran the crate.

## Falsifiers (each with its disable, all verified red 2026-10-21)

| test | disable |
|---|---|
| a released thought (Commit, Plan, Prune), its slot freed and reused, replays cell for cell | — (replay takes no pool, by signature) |
| a receipt without its seed refuses | default the seed to 0 |
| a different seed is an output mismatch (seeds shown to give different writes first) | skip the output check |
| a different execution config refuses | skip the config check |
| a changed source refuses; a change to a non-source cell does not | skip the source fold |
| a freed, reused or ABA-reused slot (same thought id, new generation) fails closed | ignore the slot generation |
| contradictions and roots keep their meaning under replay (equal `next_version`, one contradiction) | report receipt roots as independent |
| a replay earns no new certification; a genuinely new root does (Gadamer) | report receipt roots as independent |
| an unsealed, torn, missing or data-less metadata row is not a landing | accept unsealed rows; skip the seal comparison; stage before the data version |
| the same plan in two slots has one identity; a different seed changes it | put the slot into the identity |

## Measured vs not measured

- Measured: the 10 tests and the 10 disable runs above, on the fixture address space (12 rows × 4 lanes), one program (`DERIVE`).
- Not measured: timing (none taken); Lance-backed metadata (the store and table are in-memory stand-ins); atomicity of the two Lance versions (still OPEN in thought-git-basics item 8); programs other than `DERIVE`; receipts across a rebase.
- Fold collisions: the folds are 64-bit witnesses, not proofs.

OPEN: `BaseRef` stands in for D-CPG's `DurableRef`, which has no contract type; the plan places receipt, slot and `DurableRef` in `lance-graph-contract` at D-TGB-7. D-RFT-5 can now reference this receipt shape.
