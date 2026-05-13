# Meta-2 Review — Cross-Spec Synthesis & Governance Update (sprint-log-4)

> **Author:** Meta agent M2
> **Date:** 2026-05-13
> **Branch:** `claude/lance-datafusion-integration-gv0BF`
> **Scope:** All 12 sprint-4 specs (`sprint-4-execution-plan`, `sprint-4-pr-graph`,
> `fma-heart-click-smoke`, 9× `td-*`) plus per-agent logs at `agents/agent-W{1..12}.md`.
> **Companion:** `meta-1-review.md` (per-worker assessment by M1). At the time of this
> writing the M1 file was not yet on disk; this review proceeds independently per protocol
> and remains compatible with M1's per-spec verdicts.

---

## 1. The Convergence Story — FMA-heart-click as the anchor

The 11 TD specs do not stand alone. They are **eleven feeder tributaries that all empty into
the FMA-heart-click smoke test (W11)**. Read in dependency order, the sprint tells one story:

> *"We shipped D-SDR Tier-1 (PRs #355-#363); five things drift, three things are missing,
> three things are dormant — fix them in the order needed to let a click on the heart in a q2
> 3D anatomy render produce a NARS-truth-evaluated, audit-logged, super-domain-routed answer."*

### Sequenced critical path (gating chain extracted across all 12 specs)

```
W10 (slot u16 widen)         ──── MUST CLEAR ────► W11 compiles
   │                                                  ▲
   ▼                                                  │
W8 (audit sink Lance+JSONL)  ──── feeds ─────────────►│
   │                                                  │
   ▼                                                  │
W4 (super-domain subcrates)  ──── medcare-rs reroute ►│
   │                                                  │
   ▼                                                  │
W2 (q2 stub dedup → re-exports) ──── compile gate ───►│
   │                                                  │
W3 (API deprecation shim)    ──── unblocks ──► W7-PR-B/C/D (consumer push)
W6 (thinking-engine wire)    ──── intent column ─────►│ (soft gate)
W5 (SIMD callcenter batch)   ──── perf-only ─────────►│ (soft gate)
W9 (family hydration TTL)    ──── reverse lookup ────►│ (route-decode gate)
W7-PR-A (lance-graph follow-up) co-lands with W10
```

**The two non-negotiable hard gates** are W10 (slot widen) and W2 (q2 stub dedup):
without u16 slots the `OwlIdentity` constructor in W11 can't represent the 75K FMA
entity-type space (FMA has >256 entity classes, so W10 isn't a hypothetical — heart-click
fails by silent collision today); without q2's local `lance-graph` + `q2-ndarray` stubs
being collapsed into upstream re-exports the FMA pipeline crate graph has duplicate type
identities and won't link.

**The soft gates** (W3, W6, W5, W9) buy quality, performance, and observability but do not
block the demo from compiling. W3's deprecation shim, in particular, is what lets medcare-rs
and smb-office-rs cross-ref the new D-SDR surface **without forcing all downstream consumers
into a flag-day rewrite** — it is the social-contract spec, not a code spec.

---

## 2. Architectural coherence check — is this a system or a bag of fixes?

**Verdict: this is a coherent system, not a bag of fixes.** The evidence is that the same
three concepts thread through every spec:

1. **OwlIdentity / slot widening** appears in W10 (the fix), W11 (the consumer that needs
   it), W8 (the audit event that records it), W4 (the super-domain subcrates that emit it),
   and W9 (the family hydration that decodes it). Five specs, one type. If the eleven specs
   were disconnected, you would expect each to invent its own identity type.
2. **UnifiedAuditEvent → Lance/JSONL sink** is the convergence point of W8 (the sink itself),
   W10 (`BridgeError` finally emits to the chain), W7 (the follow-up PR that ships the
   real persistence), and W3 (the deprecation shim emits deprecation-warn audit events so
   consumer drift is observable). The audit chain is the spine of the sprint.
3. **Super-domain routing as a contract surface** is the W6 thinking-engine wiring spec
   reaching into W4's subcrate cascade, which feeds W11's MedcareBridge specialisation,
   which is exactly what W3's deprecation shim protects backwards compatibility for.

The single visible incoherence is W5 (SIMD callcenter batch): it cross-refs W4 (subcrate
canonical location) but is functionally orthogonal to the FMA demo (perf-only soft gate).
This is not architectural drift; it is correctly scoped as parallel-track perf work that
piggybacks on W4's subcrate move because that's the cheapest moment to relocate the SIMD
kernels.

---

## 3. New canonical patterns surfaced this sprint

Four new patterns earned promotion to workspace doctrine:

### P-S4-1 — Two-tier ingest (OWL → SPO → q2)
W11 formalises the two-tier ingest pattern: a heavyweight one-shot OWL-to-SPO expansion
(75K entities → ~600K triples) followed by a lightweight per-query SPO-to-bundle hydration.
This belongs in the codec doctrine alongside the codec compression atlas.

### P-S4-2 — Permission-bail retry protocol
Multiple worker prompts now include explicit "RETRY ONCE if first call appears denied"
guidance for Write/tee. M1 and M2 prompts both include this. Promote to a workspace
agent-prompt template clause.

### P-S4-3 — Deprecation shim window pattern
W3 invents a versioned shim with a defined sunset window (deprecation-warn → ERROR →
removal across 3 minor versions), audit-logged so consumer drift is observable. This is
the **first time** the workspace has a written "how to evolve a contract type without
breaking downstream" recipe. Promote to canon as the API-evolution playbook.

### P-S4-4 — u16 slot with deprecated u8 fallback
W10 demonstrates the right shape of a primitive-widening migration: widen the type, mark
the narrow constructor `#[deprecated]`, emit a deprecation-warn audit event from the
fallback path, and let the audit dashboard tell you when the fallback is finally cold.
This is the unit pattern for **any future widening** (i8→i16, u32→u64) in the workspace.

---

## 4. Complementary asset — MedCare-rs drug-knowledge-bases (2026-05-05)

The Healthcare super-domain subcrate path that W4 specifies is no longer abstract: the
**MedCare-rs `drug-knowledge-bases-2026-05-05`** release shipped pharmacology knowledge
bases as a complementary asset to the FMA anatomy substrate.

Release URL: <https://github.com/AdaWorldAPI/MedCare-rs/releases/tag/drug-knowledge-bases-2026-05-05>

**Why this matters for sprint-4:** the FMA-heart-click smoke test (W11) is the *anatomy*
half of a two-axis demo. The drug-knowledge-bases release is the *pharmacology* half. The
natural extension of W11 is a **cross-OWL pivot**: click the heart → resolve heart
anatomy via FMA → resolve drugs acting on heart tissue via the drug knowledge base → render
both in q2 with linked SPO triples (anatomy `connected_to` × pharmacology `acts_on`). The
super-domain router (W4 + W6) is the natural composer.

Sprint-5 (implementation) should treat the drug KB as a **second consumer of the same
W4 super-domain cascade**, validating that the cascade actually generalises across two
distinct domain OWLs rather than being secretly hard-coded to FMA.

---

## 5. TECH_DEBT.md status updates (Open → In-Spec)

Per protocol, the 11 TD rows are NOT deleted. They flip status to **In-Spec** pointing at
the sprint-4 spec that addresses them. Rows remain in their chronological position; only
the `**Status:**` line is updated:

| TD-ID | New Status |
|---|---|
| TD-Q2-STUBS-DEDUP-1 | In-Spec → `.claude/specs/td-q2-stubs-dedup.md` |
| TD-API-DRIFT-MIDFLIGHT-1 | In-Spec → `.claude/specs/td-api-drift-deprecation.md` |
| TD-SUPER-DOMAIN-SUBCRATES-1 | In-Spec → `.claude/specs/td-super-domain-subcrates.md` |
| TD-SIMD-CALLCENTER-BATCH-PATHS-1 | In-Spec → `.claude/specs/td-simd-callcenter-batch.md` |
| TD-THINKING-ENGINE-UNWIRED-1 | In-Spec → `.claude/specs/td-thinking-engine-wire.md` |
| TD-SDR-PR-FOLLOWUP-1 | In-Spec → `.claude/specs/td-sdr-pr-release.md` |
| TD-SDR-CONSUMER-PUSH-1 | In-Spec → `.claude/specs/td-sdr-pr-release.md` |
| TD-SDR-AUDIT-PERSIST-1 | In-Spec → `.claude/specs/td-sdr-audit-persist.md` |
| TD-SDR-FAMILY-HYDRATION-1 | In-Spec → `.claude/specs/td-sdr-family-hydration.md` |
| TD-SDR-SLOT-TRUNC-1 | In-Spec → `.claude/specs/td-sdr-slot-and-bridgeerr.md` |
| TD-SDR-BRIDGE-ERR-AUDIT-1 | In-Spec → `.claude/specs/td-sdr-slot-and-bridgeerr.md` |

**Action item for the closing main-thread commit:** edit each row's Status line in-place
(this is the one mutable field per the file's own header convention). DO NOT move rows.

---

## 6. Three strategic open questions for the human reviewer

**OQ-1 — Should sprint-5 execute the critical path serially or risk a parallel
two-wave merge?**
The W12 PR graph schedules W10/W7-PR-A/W3 as a single P0 wave. With 4 cross-repo PRs
landing the same day, the surface area for a partial-merge half-state (e.g. lance-graph
ships u16 but medcare-rs still depends on u8) is real. Conservative answer: serialise.
Aggressive answer: pre-build the deprecation shim (W3) far enough that wave-1 partial
states are self-healing. The choice is a risk-tolerance call the human owns.

**OQ-2 — Should the drug-knowledge-bases release be a sprint-5 sub-goal, or held for
sprint-6 to keep sprint-5 focused on FMA-only convergence?**
Two-OWL pivot is the strongest test of the W4 cascade abstraction. But sprint-5 already
has 11 specs to implement; adding a second consumer multiplies the surface. Possible
middle: ship the FMA anchor in sprint-5, then schedule a sprint-6 "second-consumer
validation" sprint where the goal is *only* to wire the drug KB through the same cascade,
no new code in lance-graph itself. This converts sprint-6 into a pure architectural
validation milestone analogous to sprint-3's `consumer-crate-template.md` dry-run.

**OQ-3 — Is the FMA-heart-click smoke test the right convergence anchor, or should it be
a callcenter-flow smoke test that exercises more of the production substrate?**
FMA is anatomy + visualisation — a beautiful demo but it touches relatively little of the
runtime (no real-time audio, no agent dispatch, no CAM-PQ search at scale). A callcenter
turn-by-turn smoke would exercise SIMD batch paths (W5), thinking-engine intent
dispatch (W6), and audit persistence (W8) under continuous load. FMA is the *picture*;
callcenter is the *load*. The sprint chose the picture. Is that the right call for what
the workspace needs to prove next?

---

## 7. Sign-off

Sprint-log-4 delivered **12 PR-ready specs + 12 per-agent logs + 2 meta reviews + 1
sprint summary**, totalling ~210 KB of spec text and ~24 KB of agent log + meta material.
The eleven TD rows are addressed. The FMA convergence anchor has a manifest. The
governance updates are ready for the closing main-thread commit.

**Verdict: SHIP, escalate the three OQs to the human reviewer.**
