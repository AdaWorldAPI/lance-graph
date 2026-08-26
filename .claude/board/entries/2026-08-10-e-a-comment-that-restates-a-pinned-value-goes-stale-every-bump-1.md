## 2026-08-10 — E-A-COMMENT-THAT-RESTATES-A-PINNED-VALUE-GOES-STALE-EVERY-BUMP-1

**Status:** FINDING `[G]` (two recurrences in the same file, both in git history).

`rust-toolchain.toml`'s comment restated the pinned version in prose. It has
therefore been **wrong twice**:

1. said `1.94.1` after a bump → fixed by `10f87fb6`
   (*"docs(toolchain): fix stale 1.94.1 comment"*);
2. said `1.95.0` after the bump to `1.97.1` (`b2b08b07`) — **the identical failure,
   ~3 months later**, because the earlier fix corrected the *value* and left the
   *structure* that guarantees the value goes stale.

A bump edits `channel = "…"` and nobody re-reads the paragraph below it. The
one-off correction resets the clock; it does not stop it.

**Structural fix (this PR):** the comment **no longer restates the version at all**
— it points at the `channel` line as authoritative — and carries an **append-only
bump log** instead, one line per bump with its commit/PR and reason. Appending
cannot contradict; re-narrating always can.

**Generalizes beyond this file:** any prose that duplicates a machine-readable
value has a half-life. Either derive it, or make the duplicate append-only. (Same
shape as `E-CLAUDE-MD-KEY-DEPENDENCIES-WENT-STALE-…-1` above, found the same hour —
one is a config comment, the other a doc table, both duplicated a pin and both went
wrong.)

