# Cross-Session Broadcast — Committed, Curated Append-Only

> **This IS committed.** Unlike AGENT_LOG.md (gitignored, ephemeral),
> every entry here travels with the repo. Use ONLY for messages another
> session MUST see before starting work — architectural decisions,
> urgent corrections, findings that can't wait for the next PR merge.
>
> Most coordination belongs in Layer A (teleport role switch) or Layer B
> (local AGENT_LOG.md). See `.claude/AGENT_COORDINATION.md` §Layer C for
> when to use this channel instead.
>
> **Append via `tee -a` heredoc** — no Read, no overwrite, pre-allowed.
> The `cat >>` form remains permitted for back-compat but `tee -a` is
> the canonical pattern (chunked writes survive partial-write recovery).

---

## Entries (reverse chronological)

## 2026-04-24 — AGENT_LOG.md gitignored; architecture moved to .claude/AGENT_COORDINATION.md

After 3 merge conflicts in one session from parallel agents appending
to a committed AGENT_LOG.md, the split landed: architectural docs
(three coordination layers, canonical append pattern) moved to
`.claude/AGENT_COORDINATION.md` (committed). Per-session log is now
gitignored. Durable findings continue in EPIPHANIES.md. See
`.claude/AGENT_COORDINATION.md` for the new governance.

## 2026-04-25 — auth-rls-lite shipped: RlsRewriter without xz2/liblzma collision

**For:** smb-office-rs session (bus REQUEST e1cf316 + PR #12)
**Commit:** 34e236b on claude/teleport-session-setup-wMZfb
**Branch:** also rebased onto main (860d082 auth split already on main via PR #267)

**New features on lance-graph-callcenter:**

```toml
# In smb-bridge Cargo.toml — use this:
lance-graph-callcenter = { path = "...", features = ["auth-rls-lite"] }
```

- `auth-rls-lite` = `auth-jwt` + `query-lite` (datafusion with `default-features = false`)
- `auth-rls` = `auth-jwt` + `query` (datafusion full — same as before, still collides with lance)
- `query-lite` = datafusion minimal (logical plan + optimizer, no compression backends)

**What this unblocks:**
1. `smb-bridge::auth` collapses to re-export of `lance_graph_callcenter::auth::*`
2. `auth-rls-lite` gives you `RlsRewriter` + `OptimizerRule` without xz2/liblzma conflict
3. Wire `RlsRewriter::new(actor)` over F4 connectors immediately

**What this does NOT do:** The full `auth-rls` (with compression) still collides.
That needs datafusion 52+ or upstream xz2 fix. But `auth-rls-lite` gives you
everything RlsRewriter actually uses (common, logical_expr, optimizer).

## 2026-04-25 — ndarray VSA migrated to 16384-bit (P0): VSA_DIMS=16_384, VSA_WORDS=256

**Branch:** ndarray `claude/teleport-session-setup-wMZfb` (commit `7041ea11`)

The ndarray HPC VSA module is now aligned with the canonical Binary16K
format. The deprecated `[u64; 157]` / 10000-bit format is gone.

```
vsa.rs            VSA_DIMS  10_000 → 16_384  (power of 2)
                  VSA_WORDS    157 →    256  (16384/64 exact, no padding)
                  VSA_BYTES   1250 →   2048  (16384/8 exact)

arrow_bridge.rs   SOAKING_DIMS       10000 → 16_384
                  SIGMA_MASK_BYTES    1250 →   2048
                  DEFAULT_SOAKING_DIM 10000 → 16_384

deepnsm.rs        nsm_to_fingerprint -> [u8; 1250] → [u8; 2048]
                  XOR loop now 32×U8x64 (zero scalar tail)
```

**SIMD-clean at every precision tier.** No scalar tail at FP16x32, FP32x16,
F64x8, U8x64 — every register width divides 16384 evenly. The "SIMD-alignment
sin" documented in lance-graph EPIPHANIES.md 2026-04-24 no longer applies to
ndarray.

1619 ndarray lib tests pass; 0 failed. All consumers (lance-graph
arigraph, callcenter, contract, q2 cockpit) can now rely on a single
canonical format end-to-end.

## 2026-04-26 — ndarray::hpc::renderer shipped: SIMD double-buffer mothership

**Branch:** ndarray `claude/teleport-session-setup-wMZfb` (commit `01f4ecd4`)

The hardware-acceleration mothership for q2 cockpit / Palantir Gotham
visual rendering now lives in `ndarray::hpc::renderer`. Per-tier dispatch
via the existing `crate::simd` polyfill — same pattern as `hpc::vsa`.

```
crate::simd::F32x16  → AVX-512 (__m512, _mm512_fmadd_ps)
                     → AVX2   (__m256, _mm256_fmadd_ps)
                     → AMX    (tile-backed)
                     → NEON   (float32x4_t, vfmaq_f32) at NEON LANES = 4
                     → scalar (f32::mul_add loop)
```

**Surface:**
- `RenderFrame` — SoA: positions/velocities (3·N f32), charges (N f32),
  fingerprints (VSA_WORDS·N u64). Capacity padded to PREFERRED_F32_LANES.
- `Renderer` — double-buffer with atomic XOR-flip (`fetch_xor(1, AcqRel)`).
  `read_front()` for REST/SSE; `write_back()` for shader cycle.
- `tick(dt, damping)` — one FMA pass per chunk:
  `position = velocity·dt + position` via `F32x16::mul_add`.
  Then atomic swap. Caller controls 60 fps cadence.
- `GLOBAL_RENDERER: LazyLock<Renderer>` (4096-node default capacity).
- `integrate_simd` / `apply_uniform_force` — exposed for custom physics.

**Why ndarray, not lance-graph or q2:** rendering is hardware acceleration.
ndarray is the substrate that owns SIMD types, FMA, BLAS, CAM-PQ, CLAM,
fingerprint distance kernels. Putting the renderer here means q2 / cockpit
gets it via a single dep, and the same renderer can drive Palantir-style
graph visualization in any binary that pulls ndarray.

**Consumers (q2 / cockpit-server):**
```rust
use ndarray::hpc::renderer::{Renderer, RenderFrame};
let r = Renderer::with_capacity(4096);
loop {
    // shader cycle writes new positions to back buffer ...
    r.tick(1.0/60.0, 0.95);  // SIMD-FMA + atomic swap
    let snapshot = r.read_front();  // serve to REST / SSE
}
```

1630 ndarray lib tests pass (was 1619 → +11 renderer tests). Zero regressions.

## 2026-04-26 — ndarray U8x64 rasterizer intrinsics shipped (seismon wishlist Tier 1+2)

**Branch:** ndarray `claude/teleport-session-setup-wMZfb` (commit `1f224bae`)

8 new methods on `U8x64` across all 3 backends (AVX-512 / AVX2 / scalar):

| Method | AVX-512 intrinsic | What it unlocks |
|---|---|---|
| `pairwise_avg` | `_mm512_avg_epu8` | Mipmap 4×4 downsample in 2 ops |
| `cmpgt_mask` | `_mm512_cmpgt_epu8_mask` | Threshold, Z-test, hit-test |
| `mask_blend` | `_mm512_mask_blend_epi8` | Sprite alpha blit |
| `shl_epi16` | `_mm512_slli_epi16` | Nibble write (completes `shr_epi16` pair) |
| `mask_store` | `_mm512_mask_storeu_epi8` | Partial-tile edge writes |
| `saturating_add` | `_mm512_adds_epu8` | Additive blend (completes `saturating_sub` pair) |
| `permute_bytes` | `_mm512_permutexvar_epi8` | Cross-lane byte shuffle (palette remap) |

**For seismon session:** these + existing `cmpeq_mask` + `shr_epi16` +
`saturating_sub` + `shuffle_bytes` give you the full rasterizer toolkit.
`framebuffer.rs` + `rasterize_dots` + `rasterize_edges` can now be built
end-to-end against `crate::simd::U8x64`.

**Tier 3 (deferred):** `U16x32` lane type, `movemask_epi8`. Not blocking;
can synthesize via existing `unpack_lo/hi_epi8` + I32 ops.

9 tests pass. All methods have matching scalar fallbacks.
