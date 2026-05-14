
## W8 scratchpad — 2026-05-14 sprint-log-10

**Role:** ndarray-miri-complete spec (PR-NDARRAY-MIRI-COMPLETE)
**Output:** `.claude/specs/pr-ndarray-miri-complete.md` (23,080 bytes)

### Files read (mandatory order)
1. `.claude/board/sprint-log-10/MANIFEST.md` — confirmed W8 row + output target
2. `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §8 — primary scope (OPEN items confirmed)
3. `.claude/board/LATEST_STATE.md` — current state read
4. `.claude/board/AGENT_LOG.md` — does not exist yet (first sprint-10 agent to write)
5. `/home/user/ndarray/src/simd_nightly/u_word_types.rs` — confirmed U16x32/U32x16/U64x8 gap
6. `/home/user/ndarray/src/simd_nightly/i_word_types.rs` — confirmed I-word gap (same family)
7. `/home/user/ndarray/src/simd_nightly/f32_types.rs` — model for full method family
8. `/home/user/ndarray/src/simd.rs` lines 200-291 — cfg blocks + Miri comment confirmed
9. `/home/user/ndarray/scripts/miri-tests.sh` — 3 exclusion clauses + root cause comment confirmed

### Key findings from audit
- **u_word_types.rs**: U16x32/U32x16/U64x8 missing: simd_eq/ne/lt/le/gt/ge, simd_clamp, select, to_bitmask, zero. U16x32 additionally missing: from_u8x64_lo/hi, pack_saturate_u8, mullo, shl, shr.
- **i_word_types.rs**: I16x16/I16x32/I32x16/I64x8 missing same comparison+mask+clamp+select family.
- **simd.rs**: Lines 205-220 already document the Miri gap exactly; existing cfg blocks have NO `not(miri)` guard.
- **miri-tests.sh**: The script's own comment (lines 44-58) says "The missing piece is a cfg(miri) switch in src/simd.rs." Part B delivers that switch.
- **_original_draft.rs**: present in simd_nightly/, not in mod.rs, confirmed deletable.
- **simd_nightly/mod.rs**: exports 24 uppercase types but NO lowercase aliases; cfg(miri) block needs them added.

### Spec decisions
- Part A uses Option B (raw `Mask<..>` return) not typed UMask wrappers — lower surface, OQ-1 is escalation path.
- U16x32 widening helpers use scalar loops (correct under Miri; production uses intrinsics).
- Part B cfg(miri) block excludes BF16 from primary re-export — OQ-2 governs feature gating.
- miri-tests.sh update: remove clauses 1+2, keep clause 3 (pyramid_tests runtime cost).

### Open questions surfaced (for meta-review)
- OQ-1: select() signature on simd_avx512::U32x16 — typed or raw? HIGH risk if typed.
- OQ-2: BF16x16/BF16x8 feature gating under cfg(miri) — requires Cargo.toml check.
- OQ-3: F16x16 exclusion from cfg(miri) block — LOW, confirm F16x16 absent from simd_avx512.

### Status: COMPLETE — spec at `.claude/specs/pr-ndarray-miri-complete.md`
