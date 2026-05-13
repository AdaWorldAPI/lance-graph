# Meta-1 Review — Sprint-4 Tier-2 D-SDR Follow-up + FMA Convergence Specs

**Reviewer:** Meta agent M1 (Opus, accumulation tier)
**Scope:** 12 worker agents (W1-W12), 12 shipped specs (~230 KB total), 12 per-agent scratchpads
**Method:** Read SPRINT_LOG manifest, all 12 agent logs in full, then cross-grep specs for SpoQuad / merkle / SuperDomain / CognitionHandle / NamespaceBridge / slot-width / Vsa16kF32 consistency.

> **Tone:** brutally honest. Sprint-4 was the third pass of CCA2A and the first run with Sonnet workers. The grindwork/accumulation split mostly held, but a new failure mode dominated: **Sonnet workers bailed at the first Write-denied prompt instead of retrying with `tee`** even though the protocol pre-cleared `Bash(tee -a:*)` and `Write(**/*.md)`. Four workers (W4, W9, W10, W11) burned a retry on this. Ironically, M1 (this reviewer) just hit the same wall on the first two Write attempts and is now using `tee` — the failure mode is reproducible at the meta tier too.

---

## Verdict

**Ship sprint-4.** All 12 specs landed; most are oversized (sign of completeness, not bloat); cross-spec coherence is materially better than sprint-3 because workers actually grepped HEAD instead of accepting brief assumptions. **Hold** on three cross-spec inconsistencies before engineering pickup (see § 3).

---

## Per-worker assessment

### W1 — Sprint-4 master execution plan (`sprint-4-execution-plan.md`, 24 KB)
**Verdict:** Solid, 2× target. Single-turn delivery (one log entry between INIT and DONE — suspicious efficiency, may indicate light synthesis). Manifest covers all 4 waves with named CI gates and 5 cross-spec coordination flags. What's MISSING: no LOC accounting beyond W12's rough estimates; no per-spec risk weighting; the "FMA demo manifest" section is asserted but not enumerated as a runnable. Acceptable for a master plan — index function preserved.

### W2 — Q2 stubs dedup (`td-q2-stubs-dedup.md`, 16 KB)
**Verdict:** Solid, 2× target. Real-repo grep: **W2 caught the most consequential factual divergence in the sprint** — `SpoQuad` does not exist in lance-graph HEAD. The brief and W11/W12 both assume `SpoQuad` is canonical. W2 flagged it as OQ-1 with explicit "NOT found in HEAD" annotations. Also corrected the stub path (`crates/stubs/` not `crates/lance-graph`) and confirmed three stub crates (`q2-ndarray`, `graph-flow`, `notebook-query`) by reading q2's workspace Cargo.toml. This worker did its job.

### W3 — D-SDR API deprecation playbook (`td-api-drift-deprecation.md`, 21 KB)
**Verdict:** Strong, 2× target, justified by the silent-vs-loud drift taxonomy. Concrete findings: `Policy::evaluate` drift is silent (worst case); `BridgeError` rename is a hard compile error with 3 variant renames; `AuditChainBuilder` doesn't exist (moved into `UnifiedBridge::with_audit_chain()`); `FamilyEntry` shape changed 2→7 fields with `String→&'static str` lifetime narrowing. Cross-flags W7 (release timing) and W10 (slot truncation) correctly raised. Soft spot: the deprecation shim itself is described, not code-sketched at MR-ready depth.

### W4-retry — Super-domain subcrates (`td-super-domain-subcrates.md`, 20 KB)
**Verdict:** Solid, 1.7× target. **W4 v1 bailed on Write-denial; retry used `tee` and succeeded.** Defines `UnifiedBridgeImpl` trait with `Bridge: NamespaceBridge` associated type, walks Medcare/Smb/Woa/Hubspot/Hiro migration in 4 phases, surfaces a real discriminant question (`SuperDomain::SmallBusiness` may not exist yet — current enum has 8 starters Healthcare=1..Osint=7). Conformance test sketch is concrete. What's MISSING: spec asks for `cognitive_stack()` from each impl (W6 dependency) but does not pin the exact `CognitionHandle::for_super_domain(sd)` constructor — handshake with W6 is by-name only.

### W5 — SIMD callcenter batch retrofit (`td-simd-callcenter-batch.md`, 16 KB)
**Verdict:** Strong, 2× target. Real-repo grep: **W5 caught that ALL hot scalar loops live in one file** (`vsa_udfs.rs`), not the multi-file `batch.rs`/`similarity.rs`/`bundle.rs` the brief assumed. W5 also corrected the prompted ndarray surface names — `vsa_cosine_batch`/`vsa_bundle_simd` are NOT canonical; actual names are `hamming_distance_raw`, `hamming_batch_raw`, `hamming_top_k_raw`, `vsa_bind`, `vsa_bundle`. Zero-copy bridge via `[u64; 256]` confirmed by inspection. The 4× AVX2 bench assertion is calibrated to a real primitive.

### W6 — thinking-engine wire-up (`td-thinking-engine-wire.md`, 21 KB)
**Verdict:** Solid, 1.75× target. Confirms the 582 KB substrate's actual module inventory by name (ghosts.rs / qualia.rs / cognitive_stack.rs / composite_engine.rs / l4_bridge.rs / world_model.rs / tensor_bridge.rs). Compose-not-rebuild table maps each D-SDR-13/15/17 clean-room type to its existing module. Decision: new crate `lance-graph-cognition-bridge` rather than extending zero-dep `lance-graph-contract` — correct call, respects the contract crate's invariant. Adds `CognitionHandle` as `Option<Arc<Mutex<CognitionHandle>>>` on `UnifiedBridge<B>` with backward-compat noop default. What's WRONG: spec names W11's coordination type `IntentClass { atom, confidence, lens_agreement, cypher_template }` but W11's FMA spec never instantiates this type — handshake is by-name only (see § 3).

### W7 — D-SDR PR follow-up + consumer push (`td-sdr-pr-release.md`, 15 KB)
**Verdict:** Solid, 2× target. SHAs captured exactly (2c3e87d D-SDR-3, 1d0157f D-SDR-4, dabd510 lockfile, dc9e081 D-SDR-5, plus 5 governance harvest commits explicitly excluded from PR-A). PR-A/B/C/D structure with `mcp__github__create_pull_request` parameters, CI matrix, 4-hour rollback SLA, no force-push. W3 cross-flag for shim ordering hard-gated. The release narrative is the most operationally executable spec in the sprint.

### W8 — Audit sink Lance + JSONL (`td-sdr-audit-persist.md`, 24 KB)
**Verdict:** Strong, 2.4× target — size justified. **W8 caught the merkle-width discrepancy that ripples through the sprint**: brief specified `prev_merkle [u8;32]` + `event_merkle [u8;32]` (SHA-256 shape) but the actual implementation is `AuditMerkleRoot(pub u64)` FNV-1a 64-bit. Spec corrects the Arrow schema to `UInt64`, sets `AuditMerkleRoot::GENESIS = 0xa5a5_a5a5_a5a5_a5a5`, surfaces that `UnifiedAuditEvent` currently has no `prev_merkle` field (D-SDR-4b action item). Replay tool, perf budget, 4 open questions all present. **Model spec for the sprint.**

### W9-retry — Family hydration TTL (`td-sdr-family-hydration.md`, 14 KB)
**Verdict:** Solid, target-met. **W9 v1 bailed on Write-denial; retry succeeded.** Surfaces the stale-comment bug: `NamespaceRegistry::seed_defaults()` claims to populate the table but only maps IRI→u32 context_id, NOT family→SuperDomain. Introduces `SuperDomain::Unhydrated` distinct from `Unknown`, with `try_resolve(family) -> Result<SuperDomain, HydrationError>`. Regression test at line 667 of current source explicitly documents the bug. The 256-element static table plan is sound.

### W10-retry — Slot widen u16 + bridge-err audit (`td-sdr-slot-and-bridgeerr.md`, 15 KB)
**Verdict:** Solid, 1.8× target. **W10 v1 bailed on Write-denial twice; retry confirmed pre-existing file from prior attempt was complete.** Pinpoints truncation: `unified_bridge.rs:449` does `(ptr.entity_type_id() & 0xFF) as u8` — single call site. Notes the existing test at line 690 (`assert!(sink.snapshot().is_empty(), "no audit on bridge error")`) **encodes the bug as intentional** — must be inverted. `AuthDecision::BridgeError` variant exists in unified_audit.rs:80 but is never written. Both fixes pinned to exact line numbers; high engineer-ready value.

### W11-retry — FMA heart-click smoke (`fma-heart-click-smoke.md`, 29 KB)
**Verdict:** Strong, 2.4× target. **W11 v1 bailed on Write-denial; retry used `tee` and succeeded.** ASCII architecture diagram, 5 golden inputs (Heart / LeftAtrium / Aorta / Mitral / Cardiomyopathy) with allow/deny matrix, drug crosswalk via MedCare `drug-knowledge-bases-2026-05-05` release, CI integration, dependency chain. **W11 correctly carries over W8's u64 merkle correction** (lines 283, 291: "merkle_u64: u64 ... NOT [u8;32] — u64 per W8 correction"). Also pins `Vsa16kF32` as enum variant of `CrystalFingerprint`, not standalone. **Best cross-spec literacy in the sprint.**

### W12 — Cross-repo PR sequencing graph (`sprint-4-pr-graph.md`, 13 KB)
**Verdict:** Solid, 2× target. 17-row per-repo PR table, Mermaid wave graph, 10-row CI matrix, R1-R6 rollback triggers, Q1-Q3 coordination questions. Critical chain (W10 → W8 → W4 → W2 → W11) named with slot-width as load-bearing root. What's MISSING: Q3 ("does W6 need rebase against W8/W10 contract changes?") is left unanswered — most concrete coordination question in the sprint should have a decision, not a deferral. Also: PR table calls W11's deliverable a "smoke test harness" at ~600 LOC but does not name FMA OWL provenance/SHA pinning the way W9 (sprint-3) eventually did.

---

## Cross-spec inconsistencies

1. **`SpoQuad` reality gap.** W2 explicitly says "NOT found in HEAD grep" and lists it as OQ-1. W11's FMA spec, W12's PR graph, and the brief all assume `SpoQuad` is canonical / re-exportable. Either the type exists in a private commit and W2's grep was incomplete, OR W11+W12 are assuming a type that doesn't ship. **Engineer must resolve before any q2 re-export PR opens.**

2. **Merkle width (u64 vs [u8;32]) — W8 corrected, but partial spread.** W8 found `AuditMerkleRoot(u64)` FNV-1a is canonical, NOT the SHA-256 [u8;32] shape the brief assumed. W11 picked up the correction explicitly. W12's PR graph never mentions merkle width. W4's super-domain spec asks each impl for a typed `audit_sink()` but does not require it conform to the u64 chain. Risk: a downstream consumer adopts [u8;32] schemas from the brief without re-reading W8.

3. **`CognitionHandle` constructor ABI is by-name only.** W6 defines `CognitionHandle::for_super_domain(sd)` and says W4 should call it from each impl's `cognitive_stack()` method. W4's spec mentions W6 dependency at step 5 of medcare migration ("Add `CognitiveStack::for_domain(SuperDomain::Healthcare)` call") — but the method name is `CognitiveStack::for_domain`, not `CognitionHandle::for_super_domain`. **Two different proposed types and methods.** Pick one.

4. **`UnifiedBridgeImpl` trait location.** W4 puts it in `lance-graph-callcenter`. W2 lists `UnifiedBridge` (struct, not trait) in `lance-graph-callcenter::unified_bridge` and never mentions `UnifiedBridgeImpl`. W11 imports `UnifiedBridge<MedcareBridge>` directly. If W4's new trait lands, every consumer crate gets an extra trait import — W3's deprecation shim does NOT cover this, and W7's PR-A consumer-push narrative will fail review unless the trait is re-exported.

5. **`SuperDomain::SmallBusiness` discriminant churn.** W4 itself flags this — current enum (per W9's grep) has Healthcare=1..Osint=7 with `SmallBusiness` not listed. W4 asks the engineer to verify the discriminant exists before applying the template. W9 builds `try_resolve` over the existing 8-variant enum. If `SmallBusiness` is added to satisfy W4, W9's static `[SuperDomain; 256]` initializer needs a re-stamp. Coordination is not pinned in either spec.

---

## Permission-bail failure mode

Four of twelve workers (W4, W9, W10, W11) hit the same wall: `Write` tool returned denied on first attempt, agents reported task-failure or stalled, and only the explicit retry round (with "use `tee -a` if Write is denied — retry once") recovered them. M1 hit the same wall writing this review. The protocol pre-cleared both `Bash(tee -a:*)` AND `Write(**/*.md)`, but the agent prompts apparently did not propagate the fallback strategy aggressively enough.

**Protocol gap for sprint-5:**
- Add to every worker prompt's TOOL USAGE section: "If `Write` returns a permission error, IMMEDIATELY retry with `Bash` `tee <path> <<'EOF' ... EOF`. Do NOT report failure — both paths are pre-cleared."
- Consider a one-line `assert` in the spawn template: agent confirms it can `tee -a` to a sentinel test file before starting real work.
- Sonnet's tendency to surface tool-denials as terminal failures (where Opus would route around them) is real and should be designed against rather than retried per-task.

---

## Top 3 risks for engineering pickup

1. **`SpoQuad` may not exist.** Anyone trying to start W2 (q2 stub dedup) tomorrow hits a wall: re-export what? Until OQ-1 is resolved, the dedup PR cannot be drafted. Pre-coding action: 30-minute grep across full lance-graph (including private branches) to confirm presence/absence and decide between "ship `SpoQuad` first" or "delete the re-export from W2's plan."

2. **`UnifiedBridgeImpl` trait introduction is a viral API change.** W4 introduces a new trait that every super-domain consumer must adopt. W3's deprecation shim does not cover this trait. Opening PR-A (W7) before W4 lands forces medcare-rs/smb-office-rs to adopt the trait inside the deprecation window — or W4 must land first (re-sequencing wave order). W12's PR graph assumes W4 lands in Wave 2 after W3, which is the wrong order if the trait is mandatory.

3. **`SuperDomain` enum discriminant churn.** W9's hydration table is keyed on the existing enum. W4 may add `SmallBusiness`. Adding an enum variant in a different position shifts every `as u8` cast and every static-array index that uses the discriminant. Concrete failure mode: `FAMILY_TO_SUPER_DOMAIN[family_id as usize]` mapping rebuilt against the old enum, deployed against the new enum, silently mis-classifies. Pre-coding action: pick one canonical enum layout in this sprint's first PR and freeze before W4/W9 touch the table.

---

## Follow-up grindwork (Sonnet-scale, 1-3 hours each)

The 12 specs themselves request these but they were not done in this sprint:

1. **Resolve OQ-1 (W2):** Confirm `SpoQuad` presence/absence in lance-graph HEAD via full-repo grep (including private branches). Document in `TECH_DEBT.md`.
2. **Reconcile `CognitionHandle` vs `CognitiveStack` (W4/W6):** Single PR that pins one type, one constructor signature. ~50 LOC.
3. **Extend `UnifiedAuditEvent` with `prev_merkle: AuditMerkleRoot`** (W8 D-SDR-4b action item) — required before any Lance sink writes.
4. **Invert the bridge-err audit-emission regression test** at `unified_bridge.rs:690` from `assert!(sink.snapshot().is_empty())` to assert `sink.snapshot().len() == 1` with `decision == AuthDecision::BridgeError` (W10).
5. **Add `SuperDomain::Unhydrated` variant** + update the `[SuperDomain; 256]` initializer to use it as the not-yet-hydrated sentinel (W9). ~30 LOC.
6. **Pin `SuperDomain::SmallBusiness` discriminant** in `super_domain.rs` before W4 migration starts (W4 §4.2 verification step).
7. **Write the `cargo deny` / `cargo-public-api` baseline** for `lance-graph-callcenter` so W3's deprecation shim has a published-surface anchor.
8. **Pin FMA OWL provenance** — `fma_obo.owl.zip` URL + sha256 in `fma-heart-click-smoke.md` Appendix (W11 references but does not pin).
9. **Resolve W12 Q3** — decide whether W6 rebases against W8/W10 contract changes or whether W6 can land in Wave 2 parallel.
10. **Author the `IntentClass` struct** (W6 names it, W11 implicitly consumes it via the Cypher cell) — one canonical location in `lance-graph-cognition-bridge`.

---

## Aggregate sprint metrics

- **Specs shipped:** 12 / 12 (~230 KB total; avg ~19 KB, target avg ~9 KB → 2× oversize)
- **Workers requiring retry:** 4 / 12 (W4, W9, W10, W11) — all due to `Write`-denial bail
- **Real-repo corrections made by workers:** W2 (SpoQuad missing, stub paths), W3 (AuditChainBuilder removed, FamilyEntry shape), W5 (ndarray surface names, file locations), W6 (substrate module inventory), W8 (merkle width u64), W9 (stale seed_defaults comment), W10 (truncation line + bug-as-test)
- **Hard cross-spec inconsistencies:** 5 (SpoQuad, merkle width spread, CognitionHandle ABI, UnifiedBridgeImpl trait surface, SuperDomain discriminant churn)
- **Engineering pickup readiness:** **Hold** on cross-spec inconsistencies 1-3; rest of sprint is pickup-ready

**Closure:** Ship sprint-4 specs to the branch. Run the 10-item grindwork list as sprint-5's Wave 0 before any Wave 1 PR opens.
