# D-NFR-6 v2 — native operations over the SoaEnvelope borrow

**Date:** 2026-10-23 · **Status:** MEASURED, lab only · **Plan:** `.claude/plans/d-nfr-6-soa-borrow-v2.md` · **Crate:** `crates/soa-view-probe` · supersedes `2026-10-23-d-nfr-6-validate-once.md`

- **Correction:** v1 put a `Table` with its own `FrameStore` beside the SoA and checked a frame per operation. That re-derived ownership `SoaEnvelope` already carries. Mutation safety is the borrow (`&self` views, the owner's `&mut`); frame provenance and authorization are separate questions and do not justify a per-operation check.
- **Design:** view = `NodeRowPacket` (512-byte rows, not `Clone`); address = `FacetSlot` (one of 32 × 16 bytes, in range by construction); operations = `FacetFold`, one ndarray kernel each (`eq_u32_strided_to_mask`, `ternary_match_strided16_to_mask`, `masked_strided_group_sum` over 12 × `u8`).
- **Pinned:** E0502 (no write during a live view), E0597 (no view outliving the owner), E0599 (no second holder); packet = direct ndarray = row loop at 8 sizes; 3 disable runs red.
- **Cost:** packet equals direct within noise at 64 / 4096 / 65536 rows, 0 allocations in both; asm: same kernel calls, B +4–7 instructions (`n_rows × 512` and register moves), no `lock`/`cpuid`.
- **Withdrawn:** D-NFR-6a (faster `FrameStore::check`).
