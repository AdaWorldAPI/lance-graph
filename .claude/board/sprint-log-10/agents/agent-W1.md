# Agent W1 — par-tile-crate — Sprint-10 Scratchpad

**Worker:** W1 (par-tile-crate)
**Timestamp:** 2026-05-14T12:35:38Z
**Model:** claude-sonnet-4-6
**Status:** COMPLETE

## Deliverable

Spec written to: `.claude/specs/pr-ce64-mb-1-par-tile-crate.md`
File size: 37,134 bytes (~36 KB, 13 sections)

## Plans Cited

1. `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` — §0 (zone framing), §4 (AttentionMask struct), §5 (MailboxSoA<N> struct + footprint + lifecycle), §6 (par-tile NEW substrate crate), §7 (PR-CE64-MB-1 scope ~1500 LOC), §11 OQ-5 (defer vendored-rayon)
2. `.claude/board/LATEST_STATE.md` — confirmed ractor="0.14" from PR #366 S7-W3 (CallcenterSupervisor)
3. `.claude/board/sprint-log-10/MANIFEST.md` — fleet coordination, W1 role
4. `crates/lance-graph-supervisor/Cargo.toml` — exact ractor version match ("0.14", optional, features=["tokio_runtime"])
5. `.claude/knowledge/A2Aworkarounds.md` — Workaround 1 (File Blackboard) coordination protocol

## Key Delta vs Parent Plan

Parent plan §6 listed par-tile as "NEW crate (to be specced)" with ~1500 LOC estimate. This spec materializes the full Cargo.toml, 7-module layout, all three Mailbox<T> backings (InMemoryMailbox/TokioMailbox/SupabaseSubMailbox), AttentionMask SoA with LRU eviction + wrap-around renormalization, MailboxSoA<N> with push_row/dispatch_cycle/emit_one/drop_row lifecycle, BindSpaceView via NonNull<u8> raw pointer (dep-isolation without importing cognitive-shader-driver), and dep-guard build.rs that panics on forbidden transitive deps.

## Key Design Decisions (DELTA vs parent plan)

1. **BindSpaceView dep isolation**: cognitive-shader-driver CANNOT be a par-tile dep (diamond invariant). Solved via `NonNull<u8>` + lifetime `'a` — callers (lance-graph-supervisor, bevy) hold Arc<BindSpace> and pass raw pointers.
2. **EvictionSender enum**: `Std(std::sync::mpsc::Sender<EvictionEvent>)` + `#[cfg(feature = "tokio-backing")] Tokio(tokio::sync::broadcast::Sender<EvictionEvent>)` — keeps AttentionMask pure std by default.
3. **LRU wrap at u32::MAX**: renormalize all LRU arrays (subtract u32::MAX/2, preserve relative ordering). Triggered at ~860 seconds of continuous operation at 1 cycle/5ms.
4. **dep-guard build.rs**: `FORBIDDEN_DEPS = ["lance", "arrow", "datafusion", "blas", "mkl", "cognitive-shader-driver", "lance-graph-callcenter"]` — panics at compile time if any appear in transitive deps.
5. **vendored-rayon as no-op stub**: feature exists but panics if enabled (OQ-5 deferred per §11).
6. **Source LOC**: ~1,425 (within §7 1500 estimate); tests ~540 LOC separate.

## Open Questions Surfaced for Meta-Review

- **OQ-A**: Should causal-edge appear as a direct `[dependencies]` entry or should par-tile newtype CausalEdge64 rather than re-export? (Affects whether downstream crates get the type from par-tile or must also depend on causal-edge.)
- **OQ-B**: Should vendored-rayon feature be a no-op stub (current spec) or omitted entirely until OQ-5 profiling shows the throughput cliff? (Stub risks confusion; omitting risks forgetting to re-add.)
- **OQ-C**: BindSpaceView's `unsafe fn from_raw_parts` requires callers to prove the Arc<BindSpace> outlives the view. Should par-tile expose a safe constructor that takes `&Arc<BindSpace>` (requiring cognitive-shader-driver import, breaking diamond) or keep the unsafe API and document the contract?

## Files-to-Touch Summary (from spec §11)

| File | Action |
|---|---|
| `Cargo.toml` (workspace root) | Add `crates/par-tile` to members |
| `crates/par-tile/Cargo.toml` | New |
| `crates/par-tile/build.rs` | New (dep-guard) |
| `crates/par-tile/src/lib.rs` | New |
| `crates/par-tile/src/mailbox.rs` | New (Mailbox<T> trait + 3 backings) |
| `crates/par-tile/src/attention.rs` | New (AttentionMask SoA) |
| `crates/par-tile/src/soa.rs` | New (MailboxSoA<N>) |
| `crates/par-tile/src/bind_space_view.rs` | New (BindSpaceView) |
| `crates/par-tile/src/sigma_tier.rs` | New (SigmaTier enum + routing) |
| `crates/par-tile/tests/` | New (5 test files) |
| `crates/lance-graph-supervisor/Cargo.toml` | Add par-tile dep |
| `crates/cognitive-shader-driver/Cargo.toml` | Add par-tile dep |

