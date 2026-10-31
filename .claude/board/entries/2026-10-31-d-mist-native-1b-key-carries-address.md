# 2026-10-31 — D-MIST-NATIVE-1b: the row key carries the address

**Status:** LAB, measured by tests. Plan: `.claude/plans/d-mistaber-native-v1.md` §10. DECISION (operator): "Key carries the address, classid beside the declaration."

- Each Mistaber node row's key is a V3 key with its `Morton8x8` address in the identity tier (bytes 14..16).
- The classid bytes stay zero. The classid is per population on `RegisterLanes`, beside the slab declaration.
- `Kb::assemble` refuses a row whose key disagrees with column 0 (`KeyMismatch`).
- 19 tests + 2 doctests; all 11 goldens exact; disable runs K1–K3 red.

STATUS: measured | OUTCOME: rows are self-addressed by key and column 0, with no classid in row bytes | OPEN: real concept ids from the OGAR codebook (lab stand-ins 0x7F11..13 today).
