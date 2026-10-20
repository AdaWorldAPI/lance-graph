# D-RMAP-DIR-1: compiled membership closure vs a 256-slot focus over dir-sim (2026-10-20)

**Status:** MEASURED, lab only. **Basis:** `crates/dir-rmap-probe` (`tests/equivalence.rs` 5/5, `examples/bench.rs` at 8,000 and 65,000 users, 3 runs at 8,000). Plan `d-rmap-dir-1-membership-closure-focus-v1.md`.

- **Arms:** A dir-sim `View` walks; D crepe Datalog over fixture ids; B membership and nesting CSR; B+ closure rows per group and chain plus roles by ordinal; F B+ projected onto a 256-slot focus (and F+users, one focus row per user).
- **Equivalence:** B, B+, D equal A on every user, group and chain of the small fixture; B+ equals D on all 79,229 (8,000) and 611,965 (65,000) memberships before timing. 7 guards disable-verified red; the closure's parent-admit check stayed green and is marked an optimisation.
- **A's cost is a full membership scan per call:** 0.4 ms at 35,000 rows, 0.7 ms at 263,000.
- **B:** W2 user → groups 92–124 ns vs 408–466 µs; all-user reconciliation 7.1 s → 0.9 ms (8,000 users).
- **B+:** 32–49 ns per user's groups, 11–15 ns per user's roles; compile 1.6 ms independent of user count; one nesting pair inserted incrementally in 2–2.5 µs.
- **F:** ≤ 2× faster evaluation than B+ with ≤ 64 focus groups, break-even ~1,000 evaluations per prepared focus; 1.3× at 128; none near 256 and none cold. The focus saves the OR stage only; question tests dominate past a few hundred questions.

OPEN: hot-path read pattern of B/B+/F (reads at a user's direct-group values) under `ISS-ADMISSION-PREP-BINDING`; membership row patching; ordinal remap after a snapshot rebuild; real directory data; D-RMAP-DIR-2 not authorized.
