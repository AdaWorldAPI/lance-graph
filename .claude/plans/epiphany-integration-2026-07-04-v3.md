# Epiphany Integration Plan — Membranes, Parity, and the Unified ruff Phase Sequence

> **Version:** v3 RATIFIED (2026-07-04) — S-wave (§8) + savant wave (§9) + brutal wave (§10) all folded
> **Lane:** medcare-bridge / ruff-harvester session (Fable 5 main thread)
> **Pipeline (complete):** v1 draft → 5× Sonnet PR-history drift audit → 5× Opus savant
> review → v3 → 3× Opus brutal review (overclaim RESTATE·10 / dilution REPAIR·3 /
> baton CATCH-CRITICAL·1+latent·5) → THIS ratified text
> **PR scope:** THIS document + board wiring on lance-graph. Groups A–D execute in
> follow-up sessions per §4; nothing in A–D lands in the plan PR itself.
> **Ground state — cited as a RANGE, per `E-PLAN-GROUND-STATE-IS-A-RANGE-1`:**
> lance-graph ≥ #645 · OGAR ≥ #151 · ruff ≥ #41 (moved #39→#41 *during drafting*).
> Every executable anchor re-derives from `HEAD` at execution time; prose counts cite
> test assertions, never literals.

---

## 0. Sources and provenance

1. First-pass epiphanies E1–E8 (2026-07-02) + second-pass N1–N8 (2026-07-04) —
   ordinals are provenance only and die with this document.
2. rs-graph-llm audit ruling (`E-V3-RIG-ARM-MUST-BE-ARIGRAPH-1`,
   `E-V3-RSGRAPHLLM-IS-ADAPTER-REPATRIATE-1`) + two verb-level sharpenings.
3. Operator rulings in scope: classid canon-high flip (#147/#628); V1 u24+u24 tail
   retirement (OGAR #151 — *strengthens* this plan: `ruff_spo_address::Facet` was
   "already V3" per #151's own message); `E-SEMANTIC-OS-CONVERGENCE-1`.
4. Receipts: S-wave reports `<scratchpad>/plan-audit/S{1..5}-*.md`; savant reviews
   `<scratchpad>/plan-review/*.md`; brutal-wave verdicts returned in-session
   (overclaim O-1..O-10, dilution DC-1..DC-3, baton BH-1..BH-7), folded per §10.

## 1. Board-keyed epiphany registry (implements N2)

**Council filing structure (B1): two new parent laws + 14 instance rows.** Prior-art:
no exact duplicates; 3 UNIQUE, 9 OVERLAPS carrying mandatory cross-refs (cite, never
restate). Parents are one-line laws in the `I-<RULE>`-has-consequences shape; every
instance — including γ's — files as its OWN standalone row citing its parent (DC-1:
discoverability beats consolidation). All keys PENDING council; rows marked
**[CONJ]** file at CONJECTURE grade explicitly.

### Parents (new)

| Key | Law |
|---|---|
| `E-BOUNDARY-FUSE-1` (α) | Where the compiler's cross-boundary guarantee ends, a **mechanical fuse** must begin — a grep, a pinned-literal test, or a serialized-allocation gate (DC-3: the family, not just greps). The type system stops at the repo edge; a fuse takes over. Review-time guards are NOT instances of this law (O-1). |
| `E-REGISTRY-COMPLETENESS-ORACLE-1` (β) | A construct that fails to converge, round-trip, or classify is revealing a gap in the frozen Predicate registry — not a special input. Consequence: soc's Counterexample bucket is a *discovery queue feeding A2a*, not an error bucket. |

(γ TABULATE-THEN-DISPATCH already exists as `E-COMPILED-THINKING-TEMPLATES` + the
attention/NarsTables lineage — no new parent; its instance row below cites it.)

### Instance rows

| Proposed key | Prior-art verdict | Claim (with review-mandated qualifiers) |
|---|---|---|
| `E-CLASSID-OPACITY-1` (α) | OVERLAPS → cite `ogar-consumer-preflight.md`, `OGAR-CONSUMER-BEST-PRACTICES.md`, `E-CLASSID-COMPAT-READER` | Classid decomposition **SHOULD live only** at the licensed site (`ogar_vocab::app` / contract mirror) — an invariant to promote and enforce, not a current fact (O-9): #628 `rbac.rs` shows even the licensed crate leaked it, and the ruff-side grep does not exist yet (the novel action: extend v3-audit's bit-math grep to ruff). **Reconciliation clause:** reading the prefix *via licensed accessors* (`app_of`/`concept_of`, classid→ClassView routing) is NOT decomposition — "read-the-prefix-via-accessor ≠ decompose-the-bits"; without this the entry contradicts OGAR's "key prerenders with zero value decode" P0. Receipt: #147 landed with no ruff companion change *per its own commit message* (`2fc2370`, S3-read — O-7: not an independent diff audit). Prior art: ndarray `guid-prefix-shape-routing.md` (#215). |
| `E-FALSIFIER-FENCE-1` (α) | OVERLAPS → cite `E-BRICK3-RAN-TRUNCATION-DISALLOWED` | The public-design/private-numbers split is E-BRICK3 doctrine already; the NOVEL half is the two-sided tripwire: public CI deny-grep + private *runnable* probe (MedCare-rs archive). A membrane without a tripwire is prose. OGAR's "membrane" word means UI/LLM egress — one disambiguating sentence required (S3). Scope now includes **fixture genericization**: `medcare:*` example identifiers still sit in the public crate (BH-5) and are cleaned in A7's commit. |
| `E-SOC-MINT-CONVERGENCE-1` (β) | OVERLAPS → cite `E-BRICK3-RAN-TRUNCATION-DISALLOWED` | Extension: `soc` classifies, `mint_factored` repairs the two benign classes; the OGAR-SOC diagnostic's fix suggestion IS what `mint_factored` does. Under β, Counterexample = registry-gap discovery queue. |
| `E-REASSEMBLE-INVERSE-MINT-1` (β) | OVERLAPS → cite `E-IDENTITY-WHITEBOX-1`, `E-MINT-TRACE-1` | `reassemble` is the inverse mint; gate = `reassemble(expand(g))==g` per registry predicate. Cite the round-trip-as-gate lineage (`reencode_safety` x256, CausalEdge64 v1/v2, `flip(flip(x))==x`). Novel = building the inverse beyond the C++-plane scope (a FEATURE — see A4). |
| `E-CONVERGENCE-GATE-FIRST-1` (β) **[CONJ]** | OVERLAPS → cite `E-AR-PROJECTION-CONVERGED` | HYPOTHESIS the A5 gate is built to test (O-5): same construct should mint the same Facet across Python/C++/C#. Today only Python+Ruby emit `inherits_from` through the shared expander; C# does not yet, and no convergence test exists anywhere. Second-order form, same grade: the gate as registry ADMISSION ORACLE — a new frontend needing a new predicate has found a registry gap. Both claims graduate to FINDING only when A5 runs green. |
| `E-TURSTEHER-TRANSPILE-CARRY-1` | OVERLAPS → cite `E-CLASSRBAC-PROMOTED-TO-CONTRACT`, `E-AUTH-CLASS-WIRED-TO-RBAC`, `E-RBAC-AUTHORIZE-PROBE-GREEN`, capstone #139 | RBAC-as-const is established doctrine; novel = the *cross-language carry*: `required_role` sits on the structural side of the fence, so transpile carries the floor for free. |
| `E-RUFF-BLACKBOARD-1` | UNIQUE (instance of CLAUDE.md Layer-2 doctrine) | **Split-grade entry (DC-2):** FINDING leg — ruff has no Layer-2 coordination surface at all (S5: no `.claude/board/`), actioned by D2. CONJECTURE leg — that the soc/RadixCodebook collision was *caused* by the missing board (edits were sequential; unverified). Dissent named: a committed board in a rebase-heavy fork is itself a conflict generator → D2 stays MINIMAL (one file, no append-only logs). |
| `E-LITERAL-PIN-PARITY-1` (α) | OVERLAPS → cite `E-CODEBOOK-MINT-IS-A-CROSS-REPO-ARC` | Two literal-pinned tests against one documented layout = cross-membrane agreement where types can't link. Receipts by TEST NAME, not line number (O-10/BH-7b, per this plan's own drift law): lance `facet.rs` `hi_chain`/`lo_chain` pin test ↔ ruff `facet_bytes_match_facetcascade_layout`; plus the #628↔#147 flip fuse and the COUNT_FUSE/`assert_codebook_parity` family. Known residual (BH-7a): the two sides pin the same layout with *different* literal values — not yet a shared golden vector, so a one-sided change can pass both; **novel action upgraded: promote to a shared golden vector at every BBB boundary.** |
| `E-TENANT-ALLOC-SERIALIZED-1` (α) | UNIQUE | Tenant lanes = the second scarce append-only allocation space; give them the #148 serialized-allocation discipline. Numbers (receipt-backed): **10 shipped** (ending `Kanban`, `tenants.md`); `BoardAggregates` = LAYOUT-GATED pending 11th (T1–T6 + classid mint). The polyglot RFC's "31 slots" belongs to a *different* model (32×16B GUID slab) — never conflate. |
| `E-CENSUS-FORWARD-REF-1` | UNIQUE | Census "frozen CORE" tags carry a forward-ref to the board entry that could unfreeze them. Dissent named: the pointer is itself a drift surface; mitigation = point at a board KEY (stable, append-only), prefer executable refs where one exists. |
| `E-RETRIEVAL-INDEX-CONVERGENCE-1` **[CONJ]** | OVERLAPS → cite `E-V3-RIG-ARM-MUST-BE-ARIGRAPH-1`, `E-PANCAKES-IS-RADIX-IS-HHTL` | CONJECTURE (O-4: architectural identity, no code receipt): the rig retrieval leg, ruff's `RadixCodebook`, and the GUID cascade (HEEL/HIP/TWIG over `(part_of:is_a)`) are the same radix walk. The durable, actionable half regardless of grade: **#18 mounts on the existing walk; nobody mints another index.** |
| `E-PLANNER-COMPILES-DISPATCH-1` (γ instance) | OVERLAPS → **standalone row citing `E-COMPILED-THINKING-TEMPLATES` as parent (DC-1: default FLIPPED from merge — a W2e/planner-placement claim inside a template-stack entry is undiscoverable at the point of need)** | W2e's sub-µs question is placement, not performance: planner compiles the dispatch table at plan-time (L4, ms); executor does O(1) lookup at dispatch-time (L2, ns). Sub-µs is a lookup, not a plan. Keywords for discoverability: W2e, planner-placement, L4-compile-L2-lookup. |
| `E-PLAN-GROUND-STATE-IS-A-RANGE-1` | NEW (harvested from this plan's own drift) | A multi-repo integration plan has a drift half-life shorter than its authoring time: ruff moved #39→#41 mid-draft (and #40 *was* item A3's Python half landing); "57" was stale before it was written (actual 62); PR #620 never existed. Consequences: (a) ground state is a RANGE with a re-anchor gate — execution derives from `HEAD`; (b) **cite the executable, never the prose** — the predicate count is the `assert_eq!(Predicate::ALL.len(), 62)` test, not a doc literal. This law caught its own plan twice (O-10's line numbers; the "57"). |
| *(amendment)* `E-V3-RIG-ARM-MUST-BE-ARIGRAPH-1` | correct-as-trimmed (S2-D2; dilution check 4 PASS) | Two verb-level appends only: (a) rig **mounts on** AriGraph, never *acts as* it; (b) keep-shell (graph-flow dispatch ~500 ns) / transplant-organs (pgvector arm → AriGraph combined retrieval). Repatriation content cited from `E-V3-RSGRAPHLLM-IS-ADAPTER-REPATRIATE-1`. |

## 2. Deliverable groups

### Group A — ruff: the unified phase sequence

| ID | Deliverable | Gate / notes |
|---|---|---|
| A1 | **Mint a NEW uniquely-named branch** off ruff `origin/main@HEAD` (e.g. `claude/ruff-phase-a-<fresh-slug>`) — **NEVER re-point the shared `claude/medcare-ruff-csharp-sync-4iahey` name (BH-4 CATCH-CRITICAL: the sibling session holds unpushed commits on that name; re-pointing it is directed data loss)** | D1 + the ruff-side pointer (D2) precede the first push |
| A2a | **Registry freeze (the keystone):** version the closed `Predicate` registry; prose counts replaced by the derived assertion (`assert_eq!(Predicate::ALL.len(), 62)` already exists — prose cites the test) + conformance test | Small; gates A3–A6 |
| A2b | **Opacity invariant into the IR record** — mutates the shared IR surface; cascades to 4 emitter crates | Own PR; **blocked-on marker recorded in ruff's own board file (BH-2): "A2b: BLOCKED-ON B1 council verdict (lance-graph EPIPHANIES)"** |
| A3 | **C# golden fixture only** — Python (#40) and Ruby (pre-#33) already emit `inherits_from` (S5-D3) | Unblocks A5 |
| A4 | **Reassembler generalization — a FEATURE (cascade-impact):** new inverse logic per predicate family (core-7 + AR-shape are not reconstructed today), THEN the existing narrow round-trip test extends over the full registry | Own PR, always; the property gate is the acceptance criterion |
| A5 | **Cross-language convergence gate** — builds the test that would prove/refute `E-CONVERGENCE-GATE-FIRST-1` | Phase exit; hard-blocks backend work; greenfield |
| A6 | `Mint` → ndjson emission seam, V3-clean (ndjson = intake at the edge; landed form = `NodeRow`; IR canonical) + **registry-version stamp on the ndjson path** (iron-rule: the I-LEGACY serialization-version consequence — the stamp is the guard; do NOT rely on an assumed loud-fail, O-8) | Small lift on existing `to_ndjson` infra |
| A7 | CI **falsifier-fence tripwire** + **genericize the surviving `medcare:*` fixtures in `ruff_spo_address` docs/tests (BH-5)** | **Blocked-on Q-A7 resolution (§6) — recorded in the ruff board file**; lands with A2a once unblocked |

### Group B — lance-graph: board + doc wiring

| ID | Deliverable | Gate / notes |
|---|---|---|
| B1 | File §1 through the **epiphany-brainstorm-council**: 2 parents + 14 rows, cross-refs and split grades as specified | Verdict gates A2b + A7-doctrine text; the verdict crossing the repo boundary is BH-2's marker, not memory |
| B2 | v3 census forward-ref convention + W6-AriGraph pointer addendum (pointer form, no tag flips) | Broadcast-first, **with fallback (BH-3): if the V3 session has not applied or acked within 7 days / by its next wave-planning pass (whichever first), THIS lane applies the diff and the broadcast entry records applied-by-fallback** |
| B3 | Extend `.claude/v3/soa_layout/tenants.md` (owner-wave / classid-mint-ref / layout-version columns; 10 shipped + BoardAggregates PENDING-GATED) | Same broadcast-first + 7-day fallback as B2 |
| B4 | Amend `E-V3-RIG-ARM-MUST-BE-ARIGRAPH-1` — two trimmed appends, sibling cited | Same commit family as B1 |

### Group C — OGAR: doctrine text (insertion points pinned, S3)

| ID | Deliverable | Gate / notes |
|---|---|---|
| C1 | Falsifier-fence bullet in CLAUDE.md `## Non-negotiables` + membrane-word disambiguation | **Commit carries the plan back-pointer (BH-6)** |
| C2 | Türsteher-carry as 5th bullet in `OGAR-TRANSPILE-SUBSTRATE.md` §1.6, citing capstone #139 | Same commit family as C1 |

### Group D — cross-cutting (the boundary repairs live here)

| ID | Deliverable | Gate / notes |
|---|---|---|
| D1 | Broadcast entry in lance-graph: lane claims the ruff work + announces this plan | Necessary but NOT sufficient (BH-1): the broadcast lives in the one repo ruff sessions don't open — D2 is the enforceable half |
| D2 | Minimal ruff `.claude/board/LATEST_STATE.md` — one file, no append-only logs — **which MUST carry (BH-1/BH-2/BH-6): the pointer to this plan (repo+path), the A2b/A7 blocked-on markers, and the new-branch name minted by A1** | With A1; this file IS the consumer-side baton home |

## 3. Explicit non-goals (unchanged)

rs-graph-llm migration (task #21, sibling lane) — B4 amends text only · AriGraph code
/ retrieval leg / #18 (probe-gated) · no census tag flips · no classid/tenant mints ·
no history rewrites · ndarray out of scope (S4) · no new γ parent entry.

## 4. Sequencing DAG + PR-split map

```
D1 (lance-graph broadcast)          ← paper half
 └─► A1+D2 ── PR-1 (ruff: NEW branch + board file w/ plan pointer + gate markers)  ← enforceable half
      └─► A2a ── PR-2 (keystone freeze)          [A7 joins PR-2 only if Q-A7 resolved]
           ├─► A3 ── PR-3 ──► A5 ── PR-5 (gate) ──► A6 ── PR-6 (seam + version stamp)
           └─► A4 ── PR-4 (feature: general reassembler; ALWAYS its own PR)
B1+B4 ── lance-graph board commits (this lane's own)
 └─► A2b ── PR-2b (ruff: IR invariant; after B1 verdict, per the ruff-side marker)
B2+B3 ── broadcast-first drafts → V3 session applies OR 7-day fallback applies-with-ack
C1+C2 ── OGAR doc commit (carries plan back-pointer)
Cross-repo edges: D1→A1 · B1-verdict→{A2b, A7-doctrine-text} — both recorded consumer-side (D2 file)
```

## 5. Gates, fuses, and guards (re-carved per O-1/DC-3)

### Mechanical fuses (instances of `E-BOUNDARY-FUSE-1`) — with wiring status

| Fuse | Class | Where | Status | Fires when |
|---|---|---|---|---|
| Round-trip property | CI property gate | ruff tests | **WIRED-BY-A4** (today: C++-plane subset only) | any registry predicate fails `reassemble(expand(g))==g` |
| Convergence gate | CI property gate | ruff CI | **WIRED-BY-A5** (greenfield) | same construct mints different Facets across frontends |
| Falsifier fence | deny-grep | ruff CI | **WIRED-BY-A7** (blocked on Q-A7) | private-corpus identifier pattern in a public crate |
| Literal-pin parity | pinned-literal test (→ shared golden vector) | every BBB boundary | 2 instances LIVE; golden-vector upgrade queued | pinned bytes diverge between mirrored tests |
| Opacity invariant | grep | v3-audit, extended to ruff | **WIRED-BY-B1/A2b** | classid bit math outside the licensed site |
| Registry-version stamp | serialized version gate | A6 ndjson path | **WIRED-BY-A6** | version-less ndjson meets a newer registry |
| Tenant allocation | serialized-allocation gate | `tenants.md` registry (B3) | WIRED-BY-B3 | an unregistered lane claim appears in a PR |

(The "a new frontend *needs* a new predicate" signal is a β *interpretation* of a
convergence-gate failure — prose doctrine, not a tripwire clause; O-3.)

### Review-time guards (NOT α instances — human judgment, named so they don't drift)

| Guard | Fires when |
|---|---|
| Endpoint fence (dto-soa) | A6 attempts to become a standing `/v1/mint` endpoint |
| Service fence (dto-soa) | the dispatch table attempts to become a standing `DispatchTable` service |

## 6. Open questions

1. **Q-A7-patterns (OPEN — BLOCKS A7; council or operator decides):** BH-5 killed
   candidate (b) as stated: a regex tight enough to deny the known private
   identifiers must encode their naming scheme — the guard would reconstruct what it
   guards (meta-membrane leak). Revised candidates: (b′) **prefix-class-only public
   list** — deny the *namespace prefixes* declared private (e.g. the app-namespace
   prefix class) + a blanket "no real-corpus fixtures in public crates" policy
   enforced by review, accepting that the fence is coarser than identifier-level;
   (a) private list in MedCare-rs pulled as a CI secret (rejected-by-default: breaks
   fork builds — itself a membrane violation of another kind). Note: any prefix-level
   fence fires on ruff's own surviving `medcare:*` fixtures immediately — which is
   correct, and A7's fixture-genericization handles it before the fence arms.
2. ~~Q-A3-scope~~ MOOT · ~~Q-B3-home~~ ANSWERED (`soa_layout/tenants.md`) ·
   ~~Q-ndarray~~ ANSWERED (out of scope).

## 7. Status ledger

| Stage | Status |
|---|---|
| v1 draft (Fable 5) | DONE 2026-07-04 |
| S-wave drift audit (5× Sonnet) | DONE — 11 dispositions (§8) |
| Savant review (5× Opus) | DONE — folded (§9) |
| Brutal review (3× Opus) | DONE — 20 findings folded (§10) |
| v3 RATIFIED (Fable 5) | DONE 2026-07-04 |
| Plan PR | THIS commit |

## 8. S-wave drift-fold changelog (receipts: `plan-audit/`)

S1-D1+S2-D1 BoardAggregates→PENDING-GATED 11th · S1-D2 "31 slots" conflation removed ·
S1-D3 opacity scope-qualified · S2-D2 B4 trimmed · S3-D1/D2 C1/C2 pinned ·
S3-D4 #151 strengthens facet assumption · S4 ndarray out + #215 prior-art cite ·
S5-D1 ground state → range + re-anchor · S5-D2 count 62 (derived, not prose) ·
S5-D3 A3 shrunk to C# golden · S5-D4 A4 extends existing test.

## 9. Savant-fold changelog (receipts: `plan-review/`)

iron-rule YIELDS-WITH-AP → A6 version stamp; S1-D3 verbatim ·
dto-soa CLEAN → two review-time guards (now §5, correctly outside α) ·
prior-art 3 UNIQUE / 9 OVERLAPS / 1 merge-candidate → cross-ref column; γ not filed ·
cascade-impact → A2a/A2b split; A4 = feature; PR map; filename `-v3.md`; board-hygiene set ·
creative-explorer RICH → parents α/β; drift-law harvested; dissents named; upsells folded.

## 10. Brutal-fold changelog (verdicts returned in-session)

| Finding | Disposition |
|---|---|
| O-1 / DC-3 (fuse taxonomy) | §5 re-carved: mechanical fuses (grep / pinned-literal / allocation-gate / CI property) with STATUS column; review-time guards moved out of α; α law rewords to the mechanism family |
| O-2 / O-3 (fuses dressed as live) | STATUS column added (WIRED-BY-A4/A5/A6/A7/B1/B3); "needs a new predicate" demoted to prose |
| O-4 / O-5 / O-6 (upsells as facts) | `E-RETRIEVAL-INDEX-CONVERGENCE-1` + `E-CONVERGENCE-GATE-FIRST-1` marked **[CONJ]**; durable halves retained |
| O-7 (#147 "verified") | Reworded: "per its own commit message (S3-read)" |
| O-8 ("fails LOUD") | Removed as premise; the A6 version stamp is the guard |
| O-9 (opacity absolute) | "SHOULD live only … an invariant to promote and enforce" |
| O-10 / BH-7b (line-number cites) | All pins cited by TEST NAME; the drift-law's own rule applied to itself |
| BH-7a (no shared golden vector) | Residual named in `E-LITERAL-PIN-PARITY-1`; golden-vector upgrade = the novel action |
| DC-1 (merge default buries W2e) | Default FLIPPED: standalone γ-instance row with discoverability keywords |
| DC-2 (grade mush) | `E-RUFF-BLACKBOARD-1` split-grade: FINDING (no board surface) + CONJECTURE (collision causality) |
| **BH-4 (CATCH-CRITICAL)** | **A1 rewritten: mint a NEW branch name; never re-point the shared branch — sibling holds unpushed commits on it** |
| BH-1 / BH-2 / BH-6 (batons without consumer-side homes) | D2 upgraded to the enforceable half: ruff board file carries plan pointer + A2b/A7 blocked-on markers + new branch name; C1/C2 commit carries the OGAR back-pointer |
| BH-3 (B2/B3 die silently) | 7-day / next-wave-pass fallback: this lane applies-with-ack, recorded in the broadcast |
| BH-5 (meta-membrane leak + live fixtures) | Q-A7 candidate (b) replaced by (b′) prefix-class-only; A7 scope += genericize surviving `medcare:*` fixtures |
