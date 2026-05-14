
## 2026-05-14T00:00 — initial draft (sonnet)

**Spec path:** `.claude/specs/pr-ce64-mb-2-pal8-nars-regression.md`
**Size:** 32 KB (32399 bytes)
**Plans cited:** `causaledge64-mailbox-rename-soa-v1.md` §3 compat invariants (C1, C2, C3)
**Sources read:** `crates/causal-edge/src/edge.rs` (full layout + all tests), `crates/causal-edge/src/tables.rs` (NarsTables), `crates/lance-graph-planner/src/cache/nars_engine.rs` (NarsEngine, SpoHead, from/to_causal_edge), `crates/cognitive-shader-driver/src/bindspace.rs` (EdgeColumn, BindSpace), `.github/workflows/rust-test.yml` (existing CI shape)
**Key delta vs parent plan §3:** Discovered that the actual `edge.rs` layout has NO unused bits (bits 49-51 = plasticity, 52-63 = temporal) — the plan's "13 reserved bits 51-63" does not match the codebase. Flagged as OQ-1 (highest priority). Tests are written against functional properties of v2 accessors (not raw bit positions) so they remain correct once W2 resolves the reclaim strategy.
**Open questions surfaced for meta-review:**
- OQ-1: What bits does W2 actually reclaim? edge.rs has no unused bits in the 51-63 range the plan describes. Options: shorten temporal, compress direction+inference+plasticity triad, or other. This is blocking.
- OQ-2: TrustTexture import path — causal_edge re-export vs contract canonical (contract preferred per CLAUDE.md doctrine).
- OQ-3: pack_v2() vs setter-only API surface — setter-only is simpler; confirm with W2.
**Test count:** 7 total (6 gating + 1 ignored property test)
**PAL8 note:** No `PAL8` symbol exists in causal-edge or planner source — term comes from session knowledge doc. "PAL8" in spec refers to the CausalEdge64 u64 serialization form, not a named Rust type.
