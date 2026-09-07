# V3 + Harvest inventory — lance-graph, 2026-09-07

All 21 files under `.claude/v3/` read in full (README, INTEGRATION-PLAN,
COMPONENT-MAP, ENTROPY-MILESTONES, MODULE-TABLE [top + tail sections; the
middle is a 304-row mechanical per-file census, sampled representatively —
not individually re-verified row-by-row], FUTURE-DESIGN, VISION,
agents/BOOT.md, 7 knowledge/*.md, 6 soa_layout/*.md). Note: the task brief
said "22 files" — the actual count on disk is 21; nothing appears missing
(`find .claude/v3 -type f` = 21 files, all read).

---

## v3 file table

| File | Role (one line) | Last commit | ⊘/SUPERSEDED/RETIRED markers carried |
|---|---|---|---|
| `README.md` | Orientation + doc map, entry point for `/v3` skill | 2026-07-10 | none itself; points at primer §6 for supersession list |
| `INTEGRATION-PLAN.md` | The W0–W6 wave plan + 15 dated addenda (Addendum-1..15) recording live corrections | 2026-08-04 | Addendum-6→7 (mutual-masking retracted), Addendum-14 (ack eliminated), Addendum-15 (⊘ SUPERSEDED note on W1/W2 rows — "rows stay as written, THIS note is current state") |
| `COMPONENT-MAP.md` | Per-subsystem REUSE/REPURPOSE/EXTEND/RETIRE/NEW/BLOCKED verdict table, 7 sections | 2026-09-05 | one inline ⊘ **PARTLY STALE** note on `KanbanActor<O>` (2026-09-05, actor half deleted 2026-08-05) |
| `ENTROPY-MILESTONES.md` | The N→1 duplication-collapse ledger, M1–M26, each with a mechanical gate | 2026-07-23 | M9 row itself carries "RESOLVED... REOPENED (partial)"; several rows are now stale relative to later PRs (see Uncertain) |
| `MODULE-TABLE.md` | Mechanical per-file census (304 files: lance-graph, lance-graph-contract, + an "ancestry pipeline" addendum for thinking-engine/p64-bridge/cognitive-shader-driver) | 2026-07-10 | none as markers; content itself documents 4+ independent 12-entry "style table" duplications not yet swept |
| `FUTURE-DESIGN.md` | APPEND-ONLY meta-board / wiring-queue index for post-2026-07-02 rulings, newest-first | 2026-07-18 | ⊘ READ-THROUGH note on the "ack is the kanban trigger" row (superseded by ack-elimination) |
| `VISION.md` | The graded WHY (every claim tagged [G]/[RULING]/[ASPIRATION]) | 2026-07-17 | none; self-describes as the append-only "vision-keeping" doc |
| `agents/BOOT.md` | V3 agent-card activation table (4 cards) extending main `.claude/agents/BOOT.md` | 2026-07-02 | none |
| `knowledge/v3-substrate-primer.md` | One-page doctrine summary, §6 "what must NOT be reinvented" table | 2026-07-17 | is itself the canonical supersession table other files defer to |
| `knowledge/mailbox-kanban-model.md` | Executor arms, ahead-firing writer, kanbanstep trigger, budget | 2026-09-05 | ⊘ **PARTLY STALE** inline note (2026-09-05) on the structural-owner row: `KanbanActor`/`KanbanMsg` deleted 2026-08-05, file now read-only meta-awareness surface |
| `knowledge/compiled-templates.md` | The askama↔elixir-DSL analogy, DSL triple, oracle split, gap list | 2026-07-02 | "CORRECTED 2026-07-02" note retracting the original NextAction↔OgarAction 1:1 claim |
| `knowledge/d-mbx-a6-owner-consume-and-persistence.md` | D-MBX-A6 owner-consume adapter + fire-and-forget persistence sink spec | 2026-08-01 | none (itself documents the ack-elimination-compatible design) |
| `knowledge/multi-anchor-ast-resolution.md` | The ruff+odoo multi-anchor AST resolution method (harvest-adjacent) | 2026-07-02 | status line: "[G] for method existence, [H] per-mechanism until ground-truthed" |
| `knowledge/persona-vs-rung-ladder.md` | Demarcation: rung-content ladder (0–4) vs the persona-36 storyline; O1–O9 open items | 2026-08-30 | ⊘ SCOPE CORRECTION (2026-08-30, styles not confined to rung 4); several O-items internally regraded/re-scoped |
| `knowledge/sonnet-worker-guardrails.md` | §1 mandatory Sonnet-worker preamble + §2 vocabulary table + §3 footgun catalogue + §5 STOP triggers | 2026-08-04 | none itself; documents F1–F15 historical footguns |
| `soa_layout/README.md` | Doc map for the 4 soa_layout files | 2026-07-02 | none |
| `soa_layout/consumer-map.md` | 6-consumer write-path audit (T1–T4 tiers, warden table) | 2026-07-02 | none (content is a 2026-07-02 snapshot; not re-verified since) |
| `soa_layout/le-contract.md` | The 4+12 byte facet atom, L1–L8 payload catalogue, slot purity, §3b two-level LE contract | 2026-08-17 | §3a wide-carving waiting room explicitly a "grace period... not a tail revival"; multiple "open reconciliation items [H]" |
| `soa_layout/routing.md` | Address-as-router: prefix routing, mailbox routing, write routing, read-mode routing, adoption-as-range-count | 2026-07-02 | none |
| `soa_layout/tenants.md` | The 16 `ValueTenant` lanes with exact byte offsets (refreshed 2026-08-22/23) | 2026-08-23 | ⊘ 2026-07-28 refresh note: doc had drifted 4 tenants behind code; ⊘ 2026-07-28 TRANSIENT-READ CORRECTION on a since-resolved mismatch |
| `soa_layout/witness-nibble-lane.md` | The A9 `CausalWitnessFacet` sub-byte lane: why it is NOT a 9th §3 layout | 2026-07-29 | opens by naming its own citation error ("shipped citing `le-contract §3 L9 G24N4` — an entry that does not exist") |

---

## Wave status W0–W6

Verdicts cross-checked against `.claude/board/STATUS_BOARD.md` (§V3 rows,
lines ~890–917) and, where STATUS_BOARD itself was stale relative to
INTEGRATION-PLAN's own Addendum-15 (2026-08-04) correction, that correction
is cited instead.

### W0 — Ratify & document
- **Promise:** ship `.claude/v3/` tree + the V3 awareness layer (knowledge docs, 4 agent cards, `/v3` skill, `/v3-audit` command, CLAUDE.md+BOOT.md entrypoints).
- D-ids: D-V3-W0a, D-V3-W0b.
- Evidence (STATUS_BOARD:894–895): `"D-V3-W0a | ... | Shipped (this PR) | complete: 7/7 mappers synthesized; MODULE-TABLE = 304/304 files (21/21 census chunks); soa_layout 5/5 docs"`; `"D-V3-W0b | ... | Shipped (this PR) | 4 knowledge docs, 4 cards, skill+command registered"`.
- **Verdict: CLOSED.** The folder exists, all files present and readable; this session's own inventory confirms the tree.

### W1 — Envelope & ownership (the keystone)
- **Promise:** `mailbox_owner()` stamp, ahead-firing batch writer + cast pairing, delegation cache, MailboxId minting, probes.
- D-ids: D-V3-W1a..e.
- Evidence (Addendum-15, INTEGRATION-PLAN.md:730–761, dated 2026-08-04): `"D-V3-W1b batch writer | SHIPPED — lance-graph-planner/src/batch_writer.rs (...). NOT 'new module' to be built. Zero production call sites — built-undriven"`; `"D-V3-W1c delegation cache | SHIPPED"`; `"D-V3-W1e probes | SHIPPED"`. Then a further, later correction (`knowledge/d-mbx-a6-owner-consume-and-persistence.md` + STATUS_BOARD D-MBX-A6-P3c/P3d/P4, dated 2026-08-01/08-02) shows the writer IS now driven: `"cycle_driver.rs:516 (cognitive_pass)"` is a real production caller, and D-MBX-A6-P4 (`STATUS_BOARD:1665`) shows a full cycle/WAL driver "**P4a–P4f Shipped (slice)**" with a "64k/17 falsifier" test.
- **Verdict: CLOSED for the mechanism, PARTIAL for fleet-wide adoption.** The keystone (cast pairing, delegation cache, ownership stamp, and — per the later P3c/P4 work not reflected in the wave table itself — a real driven caller and cycle/WAL closure) is built and tested. What remains open per the plan's own honesty ledger (STATUS_BOARD:1665): "actor-owned production wiring NOT proven ... cognitive-shader-driver/SoA thought NOT proven ... durability FAKE (contract-probe WalSink, until LanceShardSink)."

### W2 — Kanban executors (two arms + structural owner)
- **Promise:** D-MBX-A6 adapter emit, per-mailbox board-as-TENANT, supervisor wiring, symbiont arm, 550ms budget, dispatch-speed probe.
- D-ids: D-MBX-A6, D-V3-W2a..e.
- Evidence: D-MBX-A6 arm #1 — STATUS_BOARD:1662 `"Shipped"` (StrategyOutcome carrier). **D-V3-W2a is the clearest open/contradictory item**: Addendum-12a (INTEGRATION-PLAN.md:486–510, 2026-07-02) specifies a NEW 10th `BoardAggregates` ValueTenant gated on T1–T6 tests + a batched classid mint. STATUS_BOARD:901 (undated re-read, presumably still 2026-07-10-era): `"Queued (GATED: Addendum-12a...)"`. But Addendum-15 (2026-08-04, INTEGRATION-PLAN.md:744) instead claims: `"D-V3-W2a kanban tenant | SHIPPED as ValueTenant::Kanban — the kanban×Rubicon per-node phase cursor in contract/canonical_node.rs:1622+ (8-byte tenant), per the one-mailbox-one-board ruling."` — but `soa_layout/tenants.md` (refreshed 2026-08-23) still lists `ValueTenant::Kanban` (tenant #9) as the **pre-existing per-row** tenant (`KanbanTenant`), the same one `mailbox-kanban-model.md` explicitly calls a *sibling*, never a substitute, for the per-mailbox board; no `BoardAggregates` tenant exists anywhere in the current 16-tenant table. **This looks like an unresolved contradiction, not a closure** — flagged below in Candidate epiphanies.
  D-V3-W2b (supervisor wiring): STATUS_BOARD:902 `"Shipped"` (2026-07-10) but COMPONENT-MAP's own 2026-09-05 correction says the `KanbanActor` half was DELETED 2026-08-05 and "has NO assigned architectural responsibility... a session reading the W2b row as an open work item would build the exact thing the ruling struck" (`mailbox-kanban-model.md`, quoted verbatim above).
- **Verdict: PARTIAL / CONTRADICTORY.** Arm #1 (planner) shipped; the structural-owner design (ractor `KanbanActor`) was shipped, then architecturally superseded (`E-ACTOR-IS-NOT-THE-PHASE-PATH-1`, 2026-08-04) — real progress happened via `persist_sink::recover_and_apply` instead, but the wave table's own W2b row is stale and would mislead a session into re-building the retired actor path. W2a's real state is genuinely unclear from the docs as written (see above).

### W3 — Compiled templates
- **Promise:** `StepMask` type, ElixirTemplate→graph-flow adapter, Rig oracle + compile-down loop, catalogue.
- D-ids: D-V3-W3a..d.
- Evidence: STATUS_BOARD:905 `"D-V3-W3a | ... | In PR (2026-07-10) | contract::step_mask::StepMask, +5 tests (866 lib green)"`; :906 `"D-V3-W3b | ... | Queued"`; :907 `"D-V3-W3c | ... | Queued"`; :908 `"D-V3-W3d | ... | Queued"`. `compiled-templates.md`'s own gap list (unchanged since 2026-07-02): `"StepMask type does not exist yet"` — contradicted by the shipped `contract::step_mask::StepMask` (the knowledge doc was never updated after W3a shipped).
- **Verdict: PARTIAL, mostly OPEN.** W3a (StepMask) shipped; W3b/c/d remain queued as of the latest STATUS_BOARD rows found. The controlling knowledge doc (`compiled-templates.md`) is stale on this point.

### W4 — DTO ladder
- **Promise:** rename `ResonanceDto`→`PerturbationDto` (D-PERT-1), BusDto/cast pairing call sites, L4 learning-loop probe.
- D-ids: D-PERT-1, D-V3-W4a, D-V3-W4b.
- Evidence: STATUS_BOARD:951 `"D-PERT-1 | ... | Shipped (#630, 2026-07-02; verified in-code 2026-07-10 — row was stale)"`; :909 `"D-V3-W4a | ... | In PR (2026-07-10)"`; :910 `"D-V3-W4b | ... | Queued"`.
- **Verdict: PARTIAL.** D-PERT-1 CLOSED. W4a in-PR-as-of-last-record (likely shipped by now given W1's mainline progress, unverified here). W4b open.

### W5 — Consumer adoption
- **Promise:** 9 sub-items (q2 re-bakes, cpic mereology, bake annotation, W5d probes, ladybug/smb-office pulls, smb-office migration, OGAR emit.rs fix, MedCare born-stamped, q2 dual-bake collapse).
- D-ids: D-V3-W5a..i.
- Evidence: STATUS_BOARD rows :911–917 are almost uniformly `"Queued"` (D-V3-W5a,b,c,e,f implied via consumer-map.md, g, h, i all cite "Queued" or reference the consumer-map §5 items, none marked Shipped). `consumer-map.md` §5 (2026-07-02 snapshot, not re-verified): explicitly names smb-office-rs's `LanceConnector::upsert` as the still-live ORPHAN-WRITE and the "first live migration" target — no evidence it has been migrated since.
- **Verdict: OPEN.** No W5 sub-item found Shipped in the material read; this is the least-advanced wave.

### W6 — Monitor & retirement
- **Promise:** adoption/corpus scanner, `0x1000` marker retirement (P4, operator checkpoint), legacy alias retirement, custom-half opening.
- D-ids: D-V3-W6a, D-CCF-4, D-V3-W6b, D-V3-W6c.
- Evidence: STATUS_BOARD:915 `"D-V3-W6a | ... | In PR (counting logic shipped 9c55646 2026-07-02... runnable examples/adoption_scan.rs added 2026-07-10; Lance-dataset sweep = residue)"`. **D-CCF-4 (STATUS_BOARD:950) has itself been RESCINDED**: `"RESCINDED (operator 2026-07-03, E-V3-DUAL-SCHEMA-0x1000-IS-PERMANENT-1: v2/v3 coexist permanently by schema — retirement off the table)"` — directly contradicting the still-live `INTEGRATION-PLAN.md:110` row text (`"P4's trigger is DEFINED: adoption reads 100%"`) and `soa_layout/routing.md:106` (`"adoption 100% ⇒ P4 trigger (operator checkpoint) ⇒ marker deprecates"`), and the primer's own §5 framing (`v3-substrate-primer.md:103-105`, `"0x1000 ... temporary by declaration"`). None of README.md, INTEGRATION-PLAN.md, or the primer mention the rescission.
- **Verdict: OPEN, and the plan's own premise (0x1000 is temporary / P4 retires it) is STALE — superseded by an operator ruling the wave-plan text does not carry.** This is the single largest doc/board drift found in this task; flagged in Candidate epiphanies below.

---

## COMPONENT-MAP counts

`COMPONENT-MAP.md` uses six verdicts: REUSE / REPURPOSE / EXTEND / RETIRE / NEW / BLOCKED (plus one-off compounds: REUSE/EXTEND, RETIRE-toward-contract, BLOCKED→flip-on, CORRECTED). Tallying every row across §§1,3,4,5,6 (the subsystem sections proper; §2 is prose, §7 is the consumer table, counted separately below):

| Verdict | Approx. count (rows, incl. compounds folded into nearest primary) |
|---|---|
| REUSE (incl. `REUSE (HW)`, `REUSE (disambiguate)`) | ~34 |
| EXTEND (incl. `REUSE/EXTEND`) | ~11 |
| REPURPOSE (incl. `REPURPOSE (rename)`) | ~6 |
| RETIRE (incl. `RETIRE-toward-contract`) | 4 |
| NEW | 3 |
| BLOCKED (incl. `BLOCKED/NEW`, `BLOCKED→flip-on`) | ~5–7 |
| CORRECTED (one-off) | 1 |

The document's own cross-cutting summary (COMPONENT-MAP.md:136–139, verbatim, more authoritative than my re-tally): *"REUSE dominates — the ruled model is mostly wired-not-invented... The load-bearing NEW pieces are exactly three: batch writer + delegation cache (W1), board-as-tenant type (W2a), StepMask + control-flow closure (W3a/b)."*

**§7 Consumers** (6 rows, own vocabulary): openproject-nexgen-rs REUSE; MedCare-rs "REUSE + **born-stamped gate (W5h)**"; OGAR "REUSE + **emit.rs 3× `as u16` post-flip mislabel (W5g)**"; q2 "REUSE/REPURPOSE"; woa-rs EXTEND; smb-office-rs BLOCKED.

**Every RETIRE item, with retirement-PR status:**
1. `CollapseGateEmission` (contract, §1) — **retired, PR cited**: "already tombstoned (PR #477); comment-only remains."
2. `cognitive_stack.rs::ThinkingStyle` (12-space, §3) — RETIRE-toward-contract, **no single PR cited in COMPONENT-MAP itself**, but ENTROPY-MILESTONES M9 records the actual retirement mechanism: 5+3-council pass, commit `1a11038`, "FIVE divergent style tables killed" (2026-07-10), later partially REOPENED (2026-07-18) for 3 untouched 12-entry tables the sweep missed.
3. `bindspace.rs::BindSpace` (§6, "RETIRE (W7)") — **no PR; not yet retired.** Gate = "parity test (mailbox_soa.rs:1145) is the deletion gate"; `sonnet-worker-guardrails.md` §2 confirms current live status: "the legacy global store, MIGRATION IN PROGRESS → MailboxSoA... something to extend or to delete now [FORBIDDEN]... add new writers to it; remove it [FORBIDDEN]."
4. `ladybug-rs bind_space + CogRedis` (§6, "RETIRE (their repo)") — **no PR; external repo, gated on contract-pulls-only migration (W5e)**, itself unshipped per the W5 verdict above.

So of 4 named RETIRE items, only 1 has landed with a cited PR; 1 landed via a different (council) mechanism; 2 remain fully open.

---

## ENTROPY-MILESTONES

26 rows (M1–M26). Rough status distribution as last recorded in the file itself (2026-07-23):

- **SHIPPED / RESOLVED (mechanical gate green):** M5 (PR #477 tombstone), M6 (registry shipped, default-retirement open), M7 (resolved 2026-07-02 correction), M9 (resolved 2026-07-10, then partially reopened 2026-07-18), M15 (resolved 2026-07-02), M25 (shipped v1 2026-07-02), M26 (mechanism shipped 2026-07-23, semantic placement remains).
- **IN-FLIGHT:** M3 (stamp shipped, enforcement gated on W1 — likely more advanced now given W1's Addendum-15 progress, not re-verified against M3's own row), M4 (successor shipped, cutover pending W1), M12 (CycleBudget shipped 2026-07-10, load-balancer consumption open), M13 (doc-side only), M17 (doc corrected, code is W3 — W3 itself only partially shipped per the wave table above), M20 (residual resolved 2026-07-18), M24 (marked "PARTIAL 2026-07-17" — **this is now visibly stale**: STATUS_BOARD's D-MBX-A6-P3d/P4 rows (2026-08-01/08-02) show a full durable-witness + recovery + cycle/WAL closure with a "64k/17 falsifier" test, far beyond what M24's row text describes).
- **QUEUED / PENDING (no shipment recorded):** M8, M10, M11, M14, M16, M19, M21, M22, M23.
- **RULING-NEEDED (blocked on an operator decision):** M18 (sigma-chain Ω→Δ→Φ→Θ→Λ vs KanbanColumn 6-phase mapping — no ruling found in this session's reads).

Net: roughly 7/26 cleanly shipped, ~6/26 in-flight (2 visibly under-reported relative to later board activity), 9/26 untouched, 1/26 blocked on a ruling that was never made.

---

## FUTURE-DESIGN + VISION open items

**FUTURE-DESIGN.md** (append-only, newest-first) — every item that reads as not-yet-done, quoted:

- *"D-MTS-1 — Markov-as-stream parity on the DeepNSM corpus: the stream earns the Markov crown or it doesn't. Gates ALL VSA-path removal. The keystone. [next]"* (VISION.md §8, cross-referenced from FUTURE-DESIGN's migration arc) — **not run**.
- *"`StepMask` in contract (sibling of FieldMask)... W3a"* — shipped per STATUS_BOARD, but the D-V3-W3b/c/d successors it unblocks remain Queued.
- *"D-TTV-1 (engineering, envelope-auditor gated)"* — thinking tenants onto V3 substrate — open, gated on envelope-auditor sign-off never recorded as given.
- *"the load-balancing wiring (W2d) does not exist yet"* (COMPONENT-MAP.md:75, cited by FUTURE-DESIGN's addenda) — open.
- *"W2c symbiont arm: one dependency-uncomment + ~10 min cold build (BlockedColdBuild is deliberate); attempt in-container after 1–3, disk permitting."* — never confirmed attempted in the material read.
- *"D-EPIPHANY-SIG-1 (queued, [H]/CONJECTURE) — Hambly–Lyons epiphany-vs-rumination detector... Probe-gated"* — open, though note the 2026-09-01 literature harvest (see Harvest section below) substantially advances the Hambly-Lyons math this item depends on, **without FUTURE-DESIGN.md itself being updated to reflect it**.
- *"Style-as-class (D-TSC-2/3, after the batched mint)"* (VISION.md §4) — explicitly gated on a mint that per the W2a contradiction above may or may not have happened.
- *"[ASPIRATION, the honest crown]: thinking that compiles its own thinking"* (VISION.md §4) — named as aspiration, not claimed done.

**VISION.md** — items explicitly tagged `[ASPIRATION]` or `[next]` (verbatim):

- *"cannot stop thinking — active inference as the dispatch mechanism... [ASPIRATION-in-operation: the doctrine is designed and partially wired, not yet measured as a closed loop]"* (§0).
- *"whether the REAL awareness loop inherits this is precisely D-MTS-6b"* (§3) — the in-driver validation of the k*=1 comma-bit finding, not yet run.
- *"[ASPIRATION until D-MTS-6b]"* (§3, repeated fence on the "reconstructs more than it stores" claim).
- *"the endgame this ladder points at [ASPIRATION, the honest crown]: thinking that compiles its own thinking"* (§4).
- *"6. The compilation loop closes [ASPIRATION]: traces → templates → runbooks → families"* (§8, the final numbered road item).
- The whole §8 road (`D-MTS-1` → `D-TTV-1` → `D-MTS-6b` → `D-MTS-2/3` → `D-TSC-2/3` → compilation-loop-closes) is presented in **explicit dependency order**, and per the material read in this session, **none of D-MTS-1, D-TTV-1, or D-MTS-6b were found shipped** — meaning the entire VISION.md road is still at its first rung.

---

## O1–O6

From `knowledge/persona-vs-rung-ladder.md` (current state, incorporating the file's own regrades):

- **O1 — rung↔content wiring absent.** Still open. `RungLevel` carries Pearl causal-depth only; "No rung references `sigma_rosetta`, `verb_table`, `recipes`, or `StyleFamily`."
- **O2 — the orchestration→recipe edge does not exist.** Still open; **re-scoped 2026-08-30** from "rung-4→rung-3 edge" to "any-rung orchestration → `recipes::Recipe`" (per the O9 scope correction). `default_runbook()` still points at the persona vocabulary, not the recipe layer.
- **O3 — persona storyline unwired by design.** Still open, and intentionally so ("no rung/dispatch code should consume it as if it were reasoning").
- **O4 — D-TSC-1 residue.** Still open: `PlannerStyleExt` not re-exported at `api::` (the q2 E0599 path); a mis-worded `#[deprecated]` note; two untethered lab-only name→ordinal tables (`wire.rs`, `auto_style.rs`).
- **O5 — probe results to ledger (run 2026-07-14).** Marked as **done as a probe run** (results recorded: p64-bridge `STYLES[ord%12]` dormant, `UNIFIED_STYLES` tethered-not-collapsed), but the two follow-on hardenings it names (`#[deprecated]` on `style_by_ordinal`) are not confirmed shipped in this session's reads.
- **O6 — triangle structure unbuilt.** Still open. "Rung 4's autopoiesis triangle (frozen × learned × exploration) has its three poles in separate subsystems... but no composed macro object." Landing zone named (`E-THINKING-STYLES-ARE-CLASSES-1`) but not built.

(For completeness, not asked but present in the same file: O7 — two divergent rung-2 144-vocabularies, unresolved; O8 — a third off-canon 0–9 rung ladder in `learning::cognitive_frameworks::Rung`, unresolved; O9 — the rung-4 anchoring of StyleFamily itself regraded 2026-08-26 as a "scalar-era artifact" under the new tower/multi-stratum reading — this is the correction that re-scoped O2.)

---

## Harvest — senses found

Grepped `.claude/v3`, `.claude/knowledge`, `.claude/board/EPIPHANIES.md`, `.claude/board/LATEST_STATE.md`, 16 named `.claude/plans/*.md` files, plus (per the coordinator's added-scope message) `.claude/nexgen/` (harvest/ + plans/) and `.claude/knowledge/literature-harvest-2026-09-01-post-1132.md`, all read in full where new. At least **five distinct senses** of "harvest" are in live use in this workspace:

1. **The ruff `ruff_*_spo` AST/SPO harvest producer pattern** — the dominant sense workspace-wide. `ruff_cpp_spo`, `ruff_csharp_spo`, `ruff_python_spo`, `ruff_ruby_spo`, `ruff_sqlalchemy_spo`, `ruff_spo_triplet`, `ruff_spo_address` walk a source AST (C++/C#/Python/Ruby/SQLAlchemy) and emit `(subject,predicate,object)` triples / method-resolution manifests that feed OGAR mints and lance-graph-contract types. Evidence in `.claude/v3/knowledge/multi-anchor-ast-resolution.md:3`: *"READ BY: any session doing ruff-harvest / odoo-rs transcode / OGAR..."*; `.claude/v3/MODULE-TABLE.md:103`: *"harvested via ruff_python_dto_check+AST"*; `.claude/v3/FUTURE-DESIGN.md:39`: *"ruff_cpp_spo Stockfish harvest (oracle-only)"*. Also the huge `E-OCR-*` / tesseract-rs arc in `EPIPHANIES.md` (line 17117 etc.) — a byte-parity-proven consumer-side use of the same pattern.

2. **Domain-specific harvest corpora produced by sense 1** — e.g. `aiwar-neo4j-harvest` (OSINT graph, `.claude/v3/MODULE-TABLE.md:135`, `unified-soa-rubikon-integration-v1.md:63`), `odoo_ontology.spo.ndjson` (`.claude/v3/MODULE-TABLE.md:103`), `r2il-harvest-pass1` (a ruff GitHub Release, `r2il-machine-semantic-contract-v1.md:1319`), `tesseract-rs/.claude/harvest/` (oracle/parity artifacts, confirmed present on disk), `woa-rs/vendor/ogit/v02-harvest/` (vendored OGIT TTL, confirmed present on disk).

3. **Cross-repo pattern-transfer harvest** — the sense used pervasively in the older (pre-V3, 2026-05/06) consumer-integration plans: "harvesting" reusable code shapes (bridge templates, reconciler shells) from one consumer repo (MedCare-rs, smb-office-rs) into another (woa-rs). Heaviest concentration: `lance-graph-in-woa-rs-v1.md` (30 hits, e.g. line 356: *"woa-rs as integration target, harvesting XRechnung + parallelbetrieb"*), `lance-graph-business-logic-poc-via-woa-rs-v1.md` (10 hits), `soa-value-tenant-migration-v1.md` (a dedicated "harvest brief" plan format, 12 hits, e.g. line 1: *"# SoA Value-Tenant Migration — Plan v1 (harvest brief + 5+3 sign-off)"*).

4. **"Idea/literature harvest" as a generic gathered-evidence ledger** — session-scoped sweeps that read prior art and file findings without shipping code. Two concrete instances read in full this task:
   - `.claude/knowledge/literature-harvest-2026-09-01-post-1132.md` — a 5-Opus-auditor literature sweep (NARS/bilattice/EIG/Hambly-Lyons theorems) against shipped `epistemic_bassin.rs`/`ogar-loco` code, dated 2026-09-01, itself later **partially superseded 2026-09-02** by a same-file ⊘ notice (five of six "loco-core calls" it treated as constitutional were retracted; only `TERNLOG (0x86)` survives).
   - `.claude/nexgen/harvest/` (11 files, 2026-09-05) — 4 read-only Sonnet "reader" reports (cascade/rolling-floor/HDR code census, Prozentrang/Shannon doctrine census, proprioception/EWA/Mexican-hat/mask-surface census, sigker/power-kernel census) + 7 Sonnet "sweeper" reports over 65 recent PRs (lance-graph #1126–#1175, OGAR #274–#298, ndarray #277–#301), feeding `.claude/nexgen/plans/nexgen-mask-histogram-thresholds-v1.md` (a live, partially-shipped plan: D-NXG-1/4/5 shipped 2026-09-05 per `STATUS_BOARD.md:74-89`).

5. **A same-word, different-referent collision worth flagging on its own**: the nexgen/literature-harvest material's **"v3"** (`.claude/nexgen/harvest/14-...:6`: *"24-axis basis v3 mirror (`epistemic_bassin::axes`)"*; `literature-harvest...md:39`: *"the 24-axis basis v3 is `0x0334`"*) is a **completely different "v3"** from `.claude/v3/`'s mailbox-kanban-facet substrate — it is version-3 of a now-**retracted** `ogar-epistemic`/`epistemic_bassin` axis design (`E-SIX-SEMANTIC-FAMILIES-MUST-NOT-IMPERSONATE-EACH-OTHER-1`, 2026-09-02). Similarly, `.claude/v3/`'s own text uses "nexgen" only to mean the unrelated **`openproject-nexgen-rs`** consumer repo (`COMPONENT-MAP.md:115,121`; `consumer-map.md:15,29`) — never `.claude/nexgen/`. Two independent naming collisions ("v3" and "nexgen") sit in the same workspace at the same time.

---

## Harvest — v3 coverage

**Sense 1 (ruff SPO harvest producer):** documented as INTEGRATED-BY-REFERENCE. `.claude/v3/knowledge/multi-anchor-ast-resolution.md` is entirely about this sense and is explicitly listed in `.claude/v3/README.md:42` doc map ("ruff/odoo transcode landings"). `MODULE-TABLE.md` cites two live harvest-fed modules (`odoo_ontology.rs`, `aiwar.rs`). **Coverage: YES, first-class.**

**Sense 2 (domain harvest corpora):** referenced only incidentally, as evidence inside sense-1 rows (`aiwar-neo4j-harvest`, `odoo_ontology.spo.ndjson`). No dedicated inventory of harvest corpora exists in `.claude/v3/`. **Coverage: PARTIAL, by citation only.**

**Sense 3 (cross-repo pattern-transfer harvest, the woa-rs/medcare-rs/smb-office-rs plans):** **NOT referenced anywhere in `.claude/v3/`.** These plans predate the V3 rulings (2026-05/06 vs 2026-07-02) and describe consumer-integration work orthogonal to the mailbox/kanban/facet substrate; `.claude/v3/soa_layout/consumer-map.md` DOES audit woa-rs/MedCare-rs/smb-office-rs write-paths (its own, V3-specific concern — write-on-behalf compliance), but never cites or incorporates the harvest-brief plans themselves. **Coverage: NO** (different concern, same consumer names, no cross-reference).

**Sense 4a — `literature-harvest-2026-09-01-post-1132.md`:** **NOT referenced anywhere in `.claude/v3/`.** Grep for `"literature-harvest"` and `"2026-09-01"` across `.claude/v3` returns zero hits. The doc's own `**Scope:**` line frames it entirely in terms of the *nexgen* "v3" (`epistemic_bassin`/`ogar-loco` axes), never `.claude/v3/`'s facet/kanban substrate — genuinely a different subsystem, so the absence of cross-reference may be *correct* rather than a gap, modulo the naming collision noted above.

**Sense 4b — `.claude/nexgen/harvest/` + `.claude/nexgen/plans/nexgen-mask-histogram-thresholds-v1.md`:** **NOT referenced anywhere in `.claude/v3/`.** Grep for `"nexgen"`, `"NestedBands"`, `"E-NXG"`, `"D-NXG"` across `.claude/v3` returns zero hits except the unrelated `openproject-nexgen-rs` string (confirmed above). Conversely, the nexgen plan does touch V3-adjacent vocabulary — `.claude/nexgen/plans/nexgen-mask-histogram-thresholds-v1.md:60`: *"The T1/T2 warden's NAMED test"* (referencing `.claude/knowledge/membrane-tiers.md`, a DIFFERENT membrane-tier doctrine, not the V3 agent cards); room 27 (line 145) explicitly connects to the Think-struct/VSA-trajectory doctrine from the top-level `CLAUDE.md`, and room 21 (line 139) cites the "256:256-by-classid ruling" — but never `.claude/v3/le-contract.md`'s L1–L8 catalogue, `tenants.md`'s tenant table, or the mailbox-kanban model, even though `NestedBands` is explicitly framed as "per (classid, version, idx)" — the exact V3 addressing space. **This is a genuine, unambiguous gap**: the nexgen harvest and plan operate directly on V3-shaped addresses (classid, version-keyed slabs, `WideFieldMask`/`FieldMask`) without citing or updating the V3 doc family that owns that addressing scheme.

**Bottom line for the coordinator's question:** *No* — `.claude/v3/` does **not** currently incorporate either the nexgen harvest sweeps or the 2026-09-01 literature harvest, by any file:line evidence found. The nexgen harvest is the more consequential omission because its subject matter (mask-based row selection over classid/version-keyed slabs) is architecturally adjacent to — and in room 27 explicitly claims kinship with — the V3 substrate's own `Think.trajectory` and tenant-lane machinery, yet the two doc families do not reference each other in either direction.

---

## Harvest — gaps to add

If an "update v3 with the harvest" pass were undertaken, concretely:

- **`.claude/v3/README.md` doc map** — add a row pointing at `.claude/nexgen/plans/nexgen-mask-histogram-thresholds-v1.md` for "row-mask/histogram/rank work over classid-addressed slabs," parallel to the existing `soa_layout/` row, since `NestedBands` is version-keyed exactly like `ValueTenant`/`ENVELOPE_LAYOUT_VERSION`.
- **`.claude/v3/soa_layout/tenants.md`** — the nexgen plan's room 27 (`nexgen-mask-histogram-thresholds-v1.md:145`) explicitly proposes reading `Think.trajectory` as `&NestedBands` (masks) instead of `Vsa16kF32`; if/when `D-NXG-*` lands a consumer wired into a real `ValueTenant`, `tenants.md` needs a new row (it currently has zero mask-histogram tenant) — file that should carry it: `tenants.md` §2.
- **`.claude/v3/knowledge/v3-substrate-primer.md` §6 ("what must NOT be reinvented")** — should gain a line for the newest, most sweeping V3-adjacent ruling found this session but absent from the primer: `E-EVERYTHING-WIRES-TO-SOA-V3-CE64-IS-ALU-LEGACY-1` (`.claude/board/EPIPHANIES.md:815`, 2026-09-05 operator ruling, verbatim: *"anything must be wired into SoA V3 substrate no exceptions except the causaledge64 adjacent as ALU legacy substrat"*) — this directly hardens and updates the primer's own §6 table (the `Vsa16kF32`/CausalEdge64 rows) but is not cited there.
- **`.claude/v3/soa_layout/witness-nibble-lane.md`** — the nexgen literature-harvest's Hambly-Lyons/Pillar-11 findings (D1/D2/D3/D9 in `literature-harvest-2026-09-01-post-1132.md`, esp. D3: *"a single `u8:u8` rail read as ONE scalar axis is d=1"*) are directly about the same sub-byte/rail-reading discipline `witness-nibble-lane.md` polices (its §3–4 "slot purity"/"loci not magnitudes" rules); no cross-reference exists in either direction. File that should carry it: `witness-nibble-lane.md` §5 (its own "Open — flagged as PROPOSALS" section) or a new companion note.
- **`.claude/v3/ENTROPY-MILESTONES.md`** — the nexgen plan's room 4 (`nexgen-mask-histogram-thresholds-v1.md:122`) proposes collapsing `Cascade::Band`, `FloorBand`, `QualityTracker` buckets, and OGAR's `residue_band` into one `NestedBands.bucket_index` — this is textbook entropy-milestone shape (N representations → 1 canonical), yet no M27 row exists for it.
- **`.claude/v3/COMPONENT-MAP.md` §6 (Shader/convergence/foundation)** — if D-NXG's room 27 (`Think.trajectory` as masks) ever ships, the existing `Vsa16kF32`-carrier rows in COMPONENT-MAP and the primer both need a REPURPOSE/RETIRE-candidate row added; today COMPONENT-MAP has no row for `NestedBands` at all.
- **Cross-link back, the other direction:** `.claude/nexgen/plans/nexgen-mask-histogram-thresholds-v1.md` itself never cites `le-contract.md`, `tenants.md`, or `routing.md` even though it is operating on the same classid/version keyspace those files own — a plan-side gap as much as a v3-side one.

---

## Candidate epiphanies

Facts a future session would need the "why" for, none of which are currently recorded as a dated board entry cross-referenced from `.claude/v3/`:

1. **W2a's status is internally contradictory across the same plan file.** `INTEGRATION-PLAN.md` Addendum-12a (line 486–510, 2026-07-02) specifies a NEW gated `BoardAggregates` 10th tenant; Addendum-15 (line 744, 2026-08-04) instead claims W2a "SHIPPED as `ValueTenant::Kanban`" — the pre-existing per-row tenant #9, which `mailbox-kanban-model.md` itself calls a *sibling*, not a substitute, of the per-mailbox board. `soa_layout/tenants.md` (refreshed as recently as 2026-08-23) shows no `BoardAggregates` tenant in the current 16-lane table. Either Addendum-15 is wrong, or the design was quietly re-scoped without regrading Addendum-12a or `mailbox-kanban-model.md`'s "gap list" (which still lists "Per-mailbox kanban board carried as a TENANT" as an open gap, INTEGRATION-PLAN §"What the arms still need," `mailbox-kanban-model.md:123`).
2. **D-CCF-4 (the `0x1000` marker retirement) was RESCINDED 2026-07-03**, one day after the plan that names it as the P4 trigger was written — but `INTEGRATION-PLAN.md:110`, `soa_layout/routing.md:106`, and `knowledge/v3-substrate-primer.md:103-105` all still describe `0x1000` as "temporary" with a defined 100%-adoption retirement trigger, with no ⊘ notice anywhere in `.claude/v3/`. `E-V3-DUAL-SCHEMA-0x1000-IS-PERMANENT-1` is the ruling; it is cited only in `STATUS_BOARD.md:950`, never mirrored into the V3 doc family.
3. **ENTROPY-MILESTONES M2 and M24 are stale relative to their own cross-referenced board rows.** M2 says "QUEUED" for the `PerturbationDto` rename; STATUS_BOARD (D-PERT-1) has said "Shipped (#630)" since 2026-07-02 — before M2's own file was last edited (2026-07-23). M24 says "PARTIAL 2026-07-17"; STATUS_BOARD's D-MBX-A6-P3d/P4 rows (2026-08-01/08-02, well before ENTROPY-MILESTONES' 2026-07-23 mtime is odd — actually P3d/P4 postdate M24's mtime, meaning the row was never updated after real progress landed) describe a materially more complete durable-witness/recovery/cycle-WAL system with a "64k/17 falsifier" test. Neither row was updated to match.
4. **`E-EVERYTHING-WIRES-TO-SOA-V3-CE64-IS-ALU-LEGACY-1` (2026-09-05)** is a sweeping "no exceptions" operator ruling directly on V3 substrate scope, landed the same day as `COMPONENT-MAP.md`'s last edit, yet is not mentioned in COMPONENT-MAP, the primer, or README — a session reading only `.claude/v3/` would not know this ruling exists.
5. **Two independent naming collisions share the string "v3" and "nexgen" respectively** across unrelated subsystems (see Harvest sense 5 above) — worth a one-line disambiguation note in `.claude/v3/README.md` given both are workspace-wide, high-frequency terms.

---

## Uncertain

- **MODULE-TABLE.md's 304-row census was not individually re-verified.** I read the header/lance-graph section in full (rows 1–128) and the tail of the ancestry-pipeline addendum (p64-bridge, cognitive-shader-driver, rows ~430–493), but the middle of the lance-graph-contract section (113 files) and the thinking-engine/other-crate ancestry rows were not exhaustively re-read row-by-row against current source — my confidence in the file's *overall shape and role* is high, but I cannot certify every individual duplication/tech-debt claim it makes is still current (the doc itself is dated 2026-07-10 and several of its own findings, e.g. the `UNIFIED_STYLES`/style-table-proliferation notes, were later partially resolved per `persona-vs-rung-ladder.md` O5 — I did cross-check that one, but did not attempt to verify the rest).
- **Whether D-V3-W4a, D-V3-W2d, D-V3-W3a etc. have shipped since their last "In PR" STATUS_BOARD entry (all dated 2026-07-10) was not independently re-verified against current source** — I relied on the two available cross-checks (STATUS_BOARD rows as last written, and INTEGRATION-PLAN's Addendum-15 self-correction for W1/W2 only) rather than grepping live crate source for `StepMask`, `CycleBudget`, etc. Given how much progress Addendum-15 revealed for W1/W2 alone, it is plausible several other "Queued"/"In PR" rows are also stale-in-the-optimistic-direction; this task's scope (V3 doc inventory) did not include a full source audit.
- **The exact current shape of `ValueTenant::Kanban` vs. any real per-mailbox board object** (Candidate epiphany 1) could not be resolved from documentation alone — it requires reading `crates/lance-graph-contract/src/canonical_node.rs` around line 1622+ and the current `MailboxSoaOwner`/`KanbanColumn` wiring, which was out of scope for a doc-only inventory.
- **The COMPONENT-MAP verdict tally in this report is an approximate hand-count**, not a mechanically generated one; compound verdicts (e.g. "REUSE/EXTEND") were folded by judgment call into one bucket each. The document's own qualitative summary (quoted verbatim) is the more reliable source for "REUSE dominates."
