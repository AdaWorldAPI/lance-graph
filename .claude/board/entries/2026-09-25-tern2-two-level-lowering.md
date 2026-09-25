# 2026-09-25 — `Lowering::Tern2`: 4–5 plane Boolean chains run as two ternlog passes

**Status:** MEASURED · DONE (production lowering) · OPEN (the remaining gap to the inlined ceiling is the ndarray kernel loop shape, HYPOTHESIS, unchanged)
**Supersedes:** the "Open — a `Tern2` lowering" section of `entries/2026-09-25-window-scheduling-and-two-level-ternlog.md`.

## What landed (`crates/lance-graph-mask-risc`)
- `Program::fused_tern2` reads the chain as one 32-bit truth table over its (at most five) leaves. It then searches for a simple disjoint decomposition `f = h(g(x,y,z), u, v)`. For each of the ten inner triples, all four `(u,v)` restrictions must lie in `{0, g, !g, 1}` for one common `g`.
- It is tried after the one-level folds, so it only claims chains over 4–5 planes. It covers `Count`, `Any`, and `Keep` into `Out::Mask`.
- Execution (`run_fused_tern2`) works in chunks of `TILE_WORDS`, with two ternlog calls per chunk:
  - `Count`/`Any` fold `h` over an on-stack chunk of `g`. No scratch slot is carved and nothing is allocated (pinned by the poison-arena and counting-allocator test).
  - `Keep` writes `g` into the chunk of `Out::Mask`, then applies `h` in place.
  - Cut words use the one-level folds' edge rules.
- A chain with no such split stays tiled. The can-stay-silent case is 5-input majority, which has three distinct non-constant restrictions for every triple.

## Numbers (`examples/tern2_probe.rs`, n = 2^20, `target-cpu=native`, median of 41, 3 runs)

| chain | Count tiled → Tern2 (µs) | ×    | Keep tiled → Tern2 (µs) | ×       |
|---|---|---|---|---|
| 4p `(a&b)\|(c&d)` | 26.0 → 12.3 | 2.1 | 25.0 → 13.6 | 1.8–1.9 |
| 5p `((a\|b)&c)^(d&!e)` | 30.8 → 14.2 | 2.1 | 24.9 → 15.2 | 1.6 |

These agree with the window probe's `tern2@256` arm (12.5 / 14.4 µs), now reached through `execute_compiled`.

## Pins moved
Three tests pinned "a fourth plane stays tiled" (`fused_ternlog.rs`, `keep_fold.rs`, `program_collapse.rs`). They are re-pinned to `Tern2`, with the reason written in each. The one-level refusals they also assert are unchanged. Disable runs, each red then restored green:
- mapping `!g` to `g`;
- dropping `Keep`'s tail clear;
- dropping `Keep`'s edge merge.
