## 2026-05-14T12:34 — W2 causaledge64-v2 (Sonnet)

**Spec path:** `.claude/specs/pr-ce64-mb-2-causaledge64-v2.md`
**Size:** 30,478 bytes (~30 KB, 13 sections)
**Plans cited:** `causaledge64-mailbox-rename-soa-v1.md` §3 (layout extension), §2 (truth-band lens collapse), §7 (PR sequencing); `crates/causal-edge/src/edge.rs` (actual shipped layout); `crates/causal-edge/src/network.rs`; `crates/lance-graph-planner/src/cache/nars_engine.rs`

**Key delta vs parent plan §3:**
- CRITICAL: Plan §3 describes "current" layout with 13 reserved bits (bits 51-63). Actual shipped `edge.rs` uses ALL 64 bits — no reserved bits exist. This is a BLOCKER (OQ-LAYOUT-1).
- Recommended Option C: drop `temporal(12)` → AriGraph SPO-G quad, `plasticity(3)` → MailboxSoA::plasticity_counters, `infer(3)` → AttentionMask::style_slots. Frees 18 bits for G(5)+W(6)+truth(2)+spare(5).
- PAL8 serialization (4101-byte format referenced in plan) not found in `crates/causal-edge/` source — OQ-PAL8-FORMAT BLOCKER for W3.
- `forward()` method reads bits 46-48 (`inference_type`) which becomes G slot under Option C — OQ-FORWARD-REFACTOR.
- TrustTexture defined locally (zero external deps constraint preserved).
- Feature flag `causal-edge-v2-layout` default ON; version bump 0.1.0 → 0.2.0.

**Open questions for meta-review:**
1. OQ-LAYOUT-1 (BLOCKER): Which of Options A-E does plan author ratify? Without answer, bit allocation is blocked.
2. OQ-PAL8-FORMAT (BLOCKER for W3): PAL8 impl not in crates/causal-edge/ — where does it live? W3 cannot write round-trip tests without knowing.
3. OQ-FORWARD-REFACTOR: Does `forward()` refactor land in PR-CE64-MB-2 scope or a follow-on PR?

