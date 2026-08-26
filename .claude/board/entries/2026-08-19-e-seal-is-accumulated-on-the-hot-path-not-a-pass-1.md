## 2026-08-19 — E-SEAL-IS-ACCUMULATED-ON-THE-HOT-PATH-NOT-A-PASS-1

**Status:** RULING `[operator]` (STORNO of "replace FNV with a faster
whole-cycle seal hash") + source-anchored map delivered, implementation
HELD per the STOP.

**The ruling:** ZERO dedicated post-finalization payload pass. The
Morton/cascade working image is resident while being resolved; integrity
is accumulated ON THAT HOT PATH: petal digest while bytes are hot →
parent reduction over child digests only → cycle root ready before the
ONE Lance append → root associated with the RETURNED DatasetVersion in
the durable receipt (never a pre-write hash input). Content/cascade root
= prepared-content identity; DatasetVersion = durable publication
coordinate — two concepts, kept distinct. NO whole-cycle FNV, NO
replacement whole-cycle hash, NO reread/reload, NO encryption, NO second
32 MiB image, NO Morton-order reconstruction at persist. Leaf granularity
(row 512 B vs petal 8 KiB) NOT frozen — measured at the seam. Digest
primitive benchmarked ONLY inside the architecture. X-C2-1's
locus/version findings remain constraints; X-C2-3 ECC never dictates the
hot-path checksum geometry. Full text:
`docs/lotus/CASCADE-ACCUMULATED-SEAL-SPEC.md`.

**The map (delivered, `docs/lotus/SEAL-FINALIZATION-MAP.md`):** row
finality = freeze's per-row fold (persist_sink.rs:381-383), nothing
earlier; FNV at :402-428 called from freeze :384; FNV uniquely binds the
frame identity (cycle + base_version = A_last), the FULL landing sequence
incl. superseded intermediates, and the kanban control plane — none of
which an image-only root carries; candidate seams = cast-time (bytes
hottest; cast order IS canonical order, stream_position = base + CastId)
or fused into freeze's existing fold walk; **idempotency audit: YES — the
(cycle, batch_hash) contract can consume
H(cycle ‖ base_version ‖ landing_chain_digest ‖ image_root) directly, all
parts accumulated hot, FNV deleted entirely, no second hash, no second
pass (falsifier: payload bytes digested == produced, exactly once)**.
Honest boundary: NO 4096→…→1 cascade, NO 16-row petal, NO 64K grid exists
in the writer source today — the seal INTRODUCES the cascade over the
cycle's witness set; that is why granularity must be measured, not
assumed. OLD path touches payload ~5×(incl. the FNV rescan); TARGET ~3×
with zero hashing passes. Pre-STOP probe scaffold held uncommitted until
this map is ratified.

