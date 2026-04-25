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
