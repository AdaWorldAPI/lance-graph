# Agent Log — Append-Only Activity Record

> **APPEND-ONLY.** Every agent run gets one entry. Newest first.
> Never edit past entries. This is the durable record of what each
> agent did — future sessions read this instead of replaying
> conversations.
>
> **Format:** `## YYYY-MM-DDTHH:MM — <description> (<model>, <branch>)`
> followed by D-ids, commit, test counts, verdict/outcome, and any
> findings worth preserving.
>
> **Chunking purpose:** An agent's entry here REPLACES its full
> transcript in the knowledge graph. If you need to know what an
> agent did, read this file — don't search for task transcripts.
>
> **Who writes:** The main thread appends after each agent completes.
> Agents themselves should also append if they run long enough to
> risk context compaction (write progress incrementally, not just
> at the end).
>
> **Cross-agent blackboard.** This file IS the Layer 2 A2A blackboard.
> Agents MUST read this file before starting work — it tells them
> what other agents already shipped, found, or are working on.
> This replaces explicit message passing between agents: no backend
> coordination needed, just file reads. The pattern mirrors the
> runtime `Blackboard` (Layer 1, `a2a_blackboard.rs`) — each entry
> is a `BlackboardEntry` with expert_id (agent name), capability
> (D-ids), result (commit), confidence (test count). Later agents
> read prior entries and build on them, same as Layer 1 experts do.

---

## Canonical Append Pattern

Agents append to this file via `cat >>` heredoc — no Read required,
no overwrite risk, permission pre-allowed in `.claude/settings.json`:

```bash
cat >> .claude/board/AGENT_LOG.md <<'EOF'

## YYYY-MM-DDTHH:MM — description (model, branch)

**D-ids:** ...
**Commit:** `abc1234`
**Tests:** N pass (M new)
**Outcome:** One-line summary.
EOF
```

This is the ONLY sanctioned write pattern for this file. Do not use
`Edit` or `Write` tools — they risk overwriting prior entries.
`cat >>` is append-only by construction.

---

## Three Coordination Layers

All three layers use the **same entry format** and the **same
append-only semantics**. Only the transport differs.

### Layer A — Teleportation (in-context role switch)

**Transport:** None (same context window).
**Latency:** Instant. **Context loss:** Zero.

The model loads an agent card (`.claude/agents/*.md`), adopts its
role and knowledge, does the work, and switches back. No process
boundary, no serialization. The 19 specialist + 5 meta-agent cards
in this workspace are **teleportation roles**, not delegation
targets. The agent IS the main thread wearing a different hat.

```
[main thread] → load @family-codec-smith card → do codec work
             → load @truth-architect card → review with full context
             → back to main thread (nothing lost)
```

### Layer B — File Blackboard (in-session, between agents)

**Transport:** `AGENT_LOG.md` commit + git stage.
**Latency:** Seconds. **Context loss:** Commit-level summary.

Agents spawned via `Agent()` are isolated processes. They read this
file before starting to see what others shipped. They append their
own entry after committing. The main thread appends for agents that
don't write the log themselves.

```
Agent A commits → appends to AGENT_LOG.md
Agent B reads AGENT_LOG.md → sees A's findings → builds on them
```

### Layer C — Cross-Session Branch Pub/Sub (between sessions)

**Transport:** `git push` + `subscribe_pr_activity` webhook.
**Latency:** Minutes. **Context loss:** Entry-level summary.

Two concurrent Claude Code sessions coordinate through a shared
branch. One pushes an `AGENT_LOG.md` append, the other gets a
GitHub webhook notification via `subscribe_pr_activity`. No polling,
no MCP server, no infrastructure — just git + GitHub webhooks.

**Setup:**

```
# Session A (first):
git checkout -b claude/blackboard
# create or update AGENT_LOG.md
git push -u origin claude/blackboard
gh pr create --title "coordination blackboard" --body "A2A bus"
# → PR #NNN created
subscribe_pr_activity(pr=NNN)

# Session B (joins):
subscribe_pr_activity(pr=NNN)
git fetch origin claude/blackboard
git checkout claude/blackboard
# read AGENT_LOG.md — see what A did
```

**Coordination loop:**

```
Session A:                              Session B:
  [does work]
  appends to AGENT_LOG.md
  git commit && git push
                                        ← webhook: push event on PR #NNN
                                        git pull origin claude/blackboard
                                        reads A's AGENT_LOG.md entries
                                        [does work building on A's findings]
                                        appends to AGENT_LOG.md
                                        git commit && git push
  ← webhook: push event on PR #NNN
  git pull
  reads B's entries
  [continues with full picture]
```

**Why this works:**

- `subscribe_pr_activity` is already in the MCP toolkit — zero setup.
- GitHub doesn't care what's in the push — an `AGENT_LOG.md` append
  is just a commit. The webhook fires. The subscriber reads.
- Git handles append-only merge cleanly — prepend-to-top means the
  merge base is always the old top, never a collision.
- The PR is the pub/sub channel. The entry format is the message.
  The transport is `git push`. The notification is a webhook.
  All existing primitives, composed sideways.

### Summary

| Layer | Scope | Transport | Latency | Loss |
|---|---|---|---|---|
| **A: Teleport** | In-context | None | Instant | Zero |
| **B: File** | In-session | `AGENT_LOG.md` | Seconds | Commit |
| **C: Branch** | Cross-session | `git push` + webhook | Minutes | Entry |

All three share one invariant: **append-only, structured entries,
newest-first.** A `BlackboardEntry` by any other transport.

---

## Entries (reverse chronological)


## 2026-04-24T15:20 — RBAC crate scaffold (sonnet, claude/smb-contract-traits)

**D-ids:** lance-graph-rbac (permission, role, policy, access)
**Commit:** `0df8780`
**Tests:** 14 pass (14 new: 1 access + 3 permission + 4 role + 6 policy)
**Outcome:** New workspace crate `lance-graph-rbac`. PermissionSpec ties RBAC to ontology via PrefetchDepth gates + action whitelists. Example roles: accountant (Detail on Customer, Full+write on Invoice), auditor (Full read-only everywhere), admin (Full+write+act everywhere). `smb_policy()` composes all three. `Policy.evaluate()` returns `AccessDecision { Allow, Deny, Escalate }`.


## 2026-04-24T15:05 — Foundry ontology layer (main thread, claude/smb-contract-traits)

**D-ids:** LinkSpec, PrefetchDepth, ActionSpec (property.rs) + ModelBinding, ModelHealth, SimulationSpec, Ontology builder (ontology.rs)
**Commit:** `574a93d`
**Tests:** 209 pass (19 new: 10 property + 9 ontology)
**Outcome:** Fills all 5 Palantir Foundry gaps. LinkSpec = typed edges (Cardinality). PrefetchDepth = L0-L3 progressive property loading (Identity → Detail → Similar → Full). ActionSpec = Manual/Auto/Suggested triggers. ModelBinding = external model I/O → ontology property. ModelHealth = NARS-based prediction quality tracking. SimulationSpec = World::fork() what-if parameters. Ontology builder composes schemas + links + actions.


## 2026-04-24T14:55 — Schema builder + board hygiene (main thread, claude/smb-contract-traits)

**D-ids:** Schema, SchemaBuilder
**Commit:** `cb8fb37`
**Tests:** 190 pass (6 new Schema builder tests)
**Outcome:** Declarative API: `Schema::builder("Customer").required("tax_id").searchable("industry").free("note").build()`. `.validate()` returns missing Required predicates. `.searchable()` = Optional + CamPq shorthand. Board-hygiene: LATEST_STATE + EPIPHANIES updated for full SMB surface.


## 2026-04-24T14:45 — PropertySpec + CAM-PQ routing (sonnet, claude/smb-contract-traits)

**D-ids:** PropertyKind, PropertySpec, PropertySchema, CUSTOMER_SCHEMA, INVOICE_SCHEMA
**Commit:** `b1ff05e`
**Tests:** 184 pass (10 new property tests)
**Outcome:** bardioc Required/Optional/Free maps to I1 Codec Regime Split: Required = Passthrough (Index), Optional = configurable, Free = CamPq (Argmax). PropertySpec carries predicate + kind + codec_route + nars_floor. CUSTOMER_SCHEMA (10 props) + INVOICE_SCHEMA (10 props).


## 2026-04-24T14:30 — SMB contract traits (sonnet, claude/smb-contract-traits)

**D-ids:** repository.rs, mail.rs, ocr.rs, tax.rs, reasoning.rs
**Commit:** `3ab8a52`
**Tests:** 174 pass (0 new — trait-shape only, no executable logic)
**Outcome:** 5 new zero-dep trait files per smb-office-rs proposal. EntityStore + EntityWriter + Batch (repository). MailParser + ThreadLinker (mail). OcrProvider + PageImage + Bbox + LayoutBlock (ocr). TaxEngine + TaxPeriod + Jurisdiction + RuleBundle (tax). Reasoner + ReasoningKind + Budget (reasoning). Additive-only: 5 `pub mod` appends to lib.rs.


## 2026-04-24T14:15 — FingerprintColumns.cycle f32 migration (sonnet, claude/teleport-session-setup-wMZfb)

**D-ids:** PR B (SoAReview expansion item #1, bindspace substrate)
**Commit:** `121acc1`
**Tests:** 42 pass in cognitive-shader-driver (40 unit + 2 e2e), 174 contract — 0 regressions
**Outcome:** `FingerprintColumns.cycle` migrated from `Box<[u64]>` (256 × u64, Binary16K) to `Box<[f32]>` (16,384 × f32, Vsa16kF32 carrier). New constant `FLOATS_PER_VSA = 16_384`. `set_cycle(&[f32])` for direct VSA write, `set_cycle_from_bits(&[u64; 256])` adapter with `binary16k_to_vsa16k_bipolar` projection. `write_cycle_fingerprint()` API unchanged (takes u64, converts internally). `byte_footprint()` for 1 row = 71,774 bytes. Module doc updated.


## 2026-04-24T13:45 — Vsa16kF32 switchboard carrier (main thread, claude/vsa16k-f32-carrier-type → PR #253 merged)

**D-ids:** PR #253, expansion-list item #1 from SoAReview sweep
**Commit:** `dc56586` (merged to main as `ddb3017`)
**Tests:** 174 contract, 11 callcenter — 0 regressions. 7 new fingerprint tests.
**Outcome:** `CrystalFingerprint::Vsa16kF32(Box<[f32; 16_384]>)` shipped as first-class variant. 6 algebra primitives: vsa16k_zero, binary16k_to_vsa16k_bipolar, vsa16k_to_binary16k_threshold, vsa16k_bind, vsa16k_bundle, vsa16k_cosine. Inside-BBB only. to_vsa10k_f32() downcast wired.


## 2026-04-24T13:00 — SoAReview multi-angle sweep (opus, two parallel agents)

**D-ids:** Supabase-shape subscriber (verdict: GHOST), Archetype transcode (verdict: LOCKED-MAPPING-INCOMPLETE)
**Commits:** none (review-only agents)
**Tests:** n/a
**Outcome — Supabase:** `subscribe()` = disconnected mpsc stub (lance_membrane.rs:186-189). DM-4 LanceVersionWatcher + DM-6 DrainTask modules commented out (lib.rs:71-79). CognitiveEventRow BBB-clean (11 LIVE, 2 ghost fields). 7-item expansion path identified.
**Outcome — Archetype:** `lance-graph-archetype/` crate does not exist. Contract-layer mappings (PersonaCard/Blackboard/CollapseGate) LIVE. 0 archetype-specific types exist. ADR-0001 Decision 1 deblocks scaffold (Rust interface defined BY new crate, not mirrored from Python). 8-item scaffold path identified.


## 2026-04-24T12:30 — Supabase subscriber wire-up (opus, claude/supabase-subscriber-wire-up) [STILL RUNNING]

**D-ids:** DM-4a/b/c, DM-5a, DM-6a/b, DM-7
**Plan:** `.claude/plans/supabase-subscriber-v1.md`
**Status:** In flight. tokio::sync::watch swap, version_watcher.rs, drain.rs scaffold, test flip.
**Target verdict:** GHOST → PARTIAL


## 2026-04-24T12:30 — Archetype crate scaffold (opus, claude/archetype-crate-scaffold) [STILL RUNNING]

**D-ids:** DU-2.1 through DU-2.6
**Plan:** `.claude/plans/archetype-scaffold-v1.md`
**Status:** In flight. New crate + Component/Processor traits + World/CommandBroker stubs.
**Target verdict:** LOCKED-MAPPING-INCOMPLETE → LOCKED-AND-SCAFFOLDED

## 2026-04-24T15:45 — Three-layer coordination + RBAC + AGENT_LOG governance (main thread, claude/smb-contract-traits)

**D-ids:** AGENT_LOG.md, CLAUDE.md governance, lance-graph-rbac, ontology.rs, settings.json permissions
**Commits:** `5e00049` (AGENT_LOG created) → `c0eda21` (blackboard protocol) → `13c1f19` (three-layer docs) → current
**Tests:** 209 contract + 14 RBAC = 223 pass
**Outcome:** Documented three coordination layers (Teleport / File Blackboard / Branch Pub-Sub). Added `cat >>` heredoc as canonical append pattern. Permissions opened for `cat >> AGENT_LOG.md`, `git push/fetch/pull`, `cargo test/check`. RBAC crate shipped (permission × role × policy × access). Ontology layer shipped (LinkSpec, PrefetchDepth, ActionSpec, ModelBinding, ModelHealth, SimulationSpec).


## 2026-04-24T16:30 — Supabase subscriber v2 (sonnet, claude/supabase-subscriber-wire-up)

**D-ids:** DM-4a/b/c, DM-6a/b
**Commit:** `ec3b5c7`
**Tests:** 17 pass with realtime feature (13 without); 5 new tests total
**Outcome:** Wired LanceMembrane::subscribe() from Phase-A disconnected mpsc stub to live tokio::sync::watch::Receiver<CognitiveEventRow> under [realtime] feature. PR #255 merged.

## 2026-04-24T16:30 — Archetype scaffold v2 (sonnet, claude/archetype-crate-scaffold)

**D-ids:** DU-2.1..2.6
**Commit:** `816a7c0`
**Tests:** 12 pass
**Outcome:** Shipped `lance-graph-archetype` crate scaffold: Component + Processor traits, World meta-state with tick/fork/at_tick stubs, CommandBroker FIFO queue, ArchetypeError. PR #254 merged.

## 2026-04-24T17:20 — Content Hamming cascade wire (opus, claude/hamming-content-cascade)

**D-ids:** Content-plane similarity pre-pass in ShaderDriver::run()
**Commit:** `2cf36ad`
**Tests:** 45 pass (43 lib + 2 e2e, 3 new: content_hamming_finds_similar_rows / _skips_dissimilar / _respects_style_threshold)
**Outcome:** The glove is flying. Before: dispatch() on 3 encoded rows returned `hit_count:0, confidence:0.0, admit_ignorance:true` across every style — the PaletteSemiring cascade probed a synthetic Base17 table unrelated to the encoded text, and the content plane was only read for the cycle_fp XOR fold, never compared. After: content-plane Hamming pre-pass runs BEFORE the palette cascade. For each pair in `passed_rows`, popcount XOR of `content_row(i)` vs `content_row(j)`; if `resonance = 1 - Hamming/16384 >= style.resonance_threshold`, emit `ShaderHit{predicates:0x01}`. Guard: N² sweep skipped when `passed_rows.len() > 256`.

**Live verification (encode 3 rows, dispatch 0..3):**
- "Palantir develops surveillance systems"     (row 0)
- "Palantir Gotham is a surveillance platform" (row 1)
- "Israel deploys military AI"                 (row 2)

| Style              | Threshold | hit_count | top-1 row pair | resonance | confidence |
|--------------------|-----------|-----------|----------------|-----------|------------|
| Analytical (1)     | 0.85      | 0         | —              | —         | 0.0 (admit_ignorance) |
| Creative (4)       | 0.35      | 6         | row 0 ↔ row 1  | 0.598     | 0.598                 |
| Peripheral (9)     | 0.20      | 6         | row 0 ↔ row 1  | 0.598     | 0.598                 |

The strongest signal (rows 0↔1, both Palantir) correctly ranks first. Rows 0↔2 (Palantir vs Israel AI) lands lowest at 0.496. Analytical's 0.85 threshold rejects all pairs — style semantics preserved.

**Key insight:** The Jirak 454-Hamming threshold calibrated in the 2026-04-24 EPIPHANY was for UNTILED DeepNSM encodes at density ≈ 0.016. The live encode path 32×-tiles 512-bit VSA → 16K content plane, pushing density to ≈ 0.48 and expected-random Hamming to ≈ 8000. Using an absolute bit threshold would have required per-density calibration; using `resonance >= style.resonance_threshold` is density-agnostic and reuses the existing style semantics. Style config IS the content-similarity threshold.

**Remaining gap:** palette cascade hits (synthetic Base17) still exist and can flood top-k when their resonance exceeds content-match resonance; see driver.rs:180 `hits.truncate(8)`. The test `content_hamming_respects_style_threshold` uses empty planes to isolate the content cascade; in production with meaningful planes, content hits will intermix with palette hits via the shared resonance sort. Option: promote content hits with a small resonance bonus if future tuning shows palette drowning content too aggressively.

Cross-ref: EPIPHANIES 2026-04-24 "Jirak noise floor" + "dispatch wiring audit", I-NOISE-FLOOR-JIRAK iron rule, driver.rs:93-156.
