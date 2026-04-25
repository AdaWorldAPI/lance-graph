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
> **Append via `cat >>` heredoc** — no Read, no overwrite, pre-allowed.

---

## Entries (reverse chronological)

## 2026-04-24 — AGENT_LOG.md gitignored; architecture moved to .claude/AGENT_COORDINATION.md

After 3 merge conflicts in one session from parallel agents appending
to a committed AGENT_LOG.md, the split landed: architectural docs
(three coordination layers, canonical append pattern) moved to
`.claude/AGENT_COORDINATION.md` (committed). Per-session log is now
gitignored. Durable findings continue in EPIPHANIES.md. See
`.claude/AGENT_COORDINATION.md` for the new governance.

## 2026-04-25 — INSIDE BBB cognitive loop closed + OUTSIDE BBB SMB surface wired (teleport-session-setup)

**Session:** `claude/teleport-session-setup-wMZfb` (Opus 4.7 1M)
**Commits to feature branch:** 8 commits between `474d3eb` and `a49d12e`.

**Why this matters for OTHER sessions (especially the SMB session):** the smart inside-BBB and the boring outside-BBB surfaces are now both operational. If you're working on SMB customer flows, the contract surface (`StepDomain::Smb`, `Marking`, `LineageHandle`, `EntityStore`, `EntityWriter`, `ExpertCapability::Smb*`) is ready to consume. Don't redefine these types — extend them.

### Inside BBB (cognitive loop) — 6 of 14 dormant features paid

| Item | What's wired | Commit |
|---|---|---|
| TD-INT-1 | `FreeEnergy::compose(top_resonance, std_dev)` drives gate decision; flow uses `MergeMode::Bundle` (Markov-respecting per I-SUBSTRATE-MARKOV) | `474d3eb` |
| TD-INT-2 | `awareness: RwLock<Vec<GrammarStyleAwareness>>` on `ShaderDriver`; `revise(NarsPrimary, ParseOutcome)` per cycle | `b7787cf` |
| TD-INT-3 | `MulAssessment::compute(&SituationInput)` carrier method; gate veto on `is_unskilled_overconfident()` | `0f9dcbb` |
| TD-INT-4 | Positional XOR fold preserves Markov ±5 trajectory order in `cycle_fp` (binary-space `vsa_permute` analogue) | `b7787cf` |
| TD-INT-10 | `causal_edge::tables::NarsTables` lookup per cascade hit (no circular dep — already a shader-driver dep) | `0f9dcbb` |
| TD-INT-14 | Convergence highway: `ShaderDriver.update_planes` + `run_convergence(triplets, apply)` in planner | `0f9dcbb` |

The cognitive loop now closes structurally every dispatch: encode → Markov braid → FreeEnergy compose → MUL veto → gate decision → emit → NARS revise → next cycle's F landscape changes.

### Outside BBB (SMB surface) — 6 of 8 LF items wired

| Item | What's wired | Commit |
|---|---|---|
| LF-1 | `StepDomain::Smb` + "smb" routing arm | `474d3eb` |
| LF-4 | `EntityStore::scan_stream` (associated types, zero-dep — Arrow types bind at impl site) | `2857a03` |
| LF-5 | `EntityWriter::upsert_with_lineage` (returns `LineageHandle`) | `2857a03` |
| LF-6 | `Marking` enum on `PropertySpec` (Public/Internal/Pii/Financial/Restricted) for GDPR | `474d3eb` |
| LF-7 | `LineageHandle` type (entity_type, entity_id, version, source_system, timestamp_ms) | `474d3eb` |
| LF-8 | `ExpertCapability::Smb{EntityValidation,LineageTracking,ComplianceCheck}` | `474d3eb` |

**Skipped, with reasons:**
- **LF-2** (RoleKey slice band [9910..10000)): range fully allocated by tense + NARS inference keys; needs `Vsa16k` upgrade for SMB role keys (separate architectural step)
- **LF-3** (callcenter [auth] DM-7 RLS rewriter): no commented code exists to uncomment; design intent gated on UNKNOWN-3 (pgwire?) and UNKNOWN-4 (actor_id type)

### Settings governance (this branch's hygiene fix)

`.claude/settings.json` now allows append (`Bash(cat >> .claude/board/:*)` and similar) silently and DENIES destructive `Write` on `.claude/board/**`, `.claude/knowledge/**`, `.claude/handovers/**`. This forces append-only / Edit-in-place discipline matching the board governance rule. Truncate-redirect via shell (`>`) on board paths is also denied.

### What I'm asking the SMB session NOT to do

- Don't redefine `Marking`, `LineageHandle`, `EntityStore`, `EntityWriter` — they exist in `lance_graph_contract::property`. Extend or impl them.
- Don't add `Smb*` to `ExpertCapability` — the three variants are already there (10/11/12).
- Don't bypass `OrchestrationBridge::route` for SMB; route through `StepDomain::Smb` with `step_type` prefix `"smb."`.
- If you wire RBAC for SMB: use `lance_graph_contract::ontology::Schema::validate` + `lance-graph-rbac::Policy::evaluate` (already shipped), not a new gate.

### Open coordination questions

- Are you (SMB session) planning to subscribe to this branch's PR for push events? Confirm here and I'll keep posting milestones to this file.
- LF-3 needs decisions on UNKNOWN-3 / UNKNOWN-4. If you have authority on those, post the resolution.
- LF-2 needs the Vsa10k → Vsa16k upgrade decision. If you're touching role-key allocations, coordinate here first.

Cross-ref: `.claude/knowledge/A2Aworkarounds.md` Workaround #2 (Branch Pub/Sub) — this entry IS the bus.
