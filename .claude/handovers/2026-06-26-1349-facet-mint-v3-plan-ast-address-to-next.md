# Handover Capstone — `facet_mint` (brick 2) + V3-plan AST-as-address integration → next session

> **From:** the 2026-06-26 `claude/serene-mayer-1a09he` session.
> **To:** the next session that picks up the AST-as-(part_of:is_a)-address arc
> and/or the SoA value-tenant V3 migration.
> **Append-only.** Correct in place only on the `**Status:**` line.

---

## 0. Paste-this to start the next session (the prompt)

> You are continuing the **AST-as-(part_of:is_a)-address** arc on
> `AdaWorldAPI/lance-graph`, branch **`claude/serene-mayer-1a09he`** (already
> pushed; HEAD `ed06ff44`). The previous session built **brick 2** (the rank-minter
> `contract::facet_mint`) and folded the whole arc into the V3 migration plan.
>
> **First, the mandatory reads** (CLAUDE.md order): `.claude/board/LATEST_STATE.md`,
> `.claude/board/PR_ARC_INVENTORY.md`, `.claude/agents/BOOT.md`. Then the arc:
> `.claude/knowledge/ast-as-partof-isa-address.md` (the what+why, merged #616/#617),
> `.claude/plans/soa-value-tenant-migration-v2.md` (the plan — read **§2.2, §2.3,
> §7** especially), `crates/lance-graph-contract/src/facet.rs` (the shipped
> `FacetCascade` carrier) + `src/facet_mint.rs` (brick 2), and
> `src/canonical_node.rs` around `CLASSID_OSINT/_FMA/_OSINT_V3` +
> `BUILTIN_READ_MODES` + `TailVariant`.
>
> **One-line state:** carrier shipped (`FacetCascade`, #613/#614), minter built
> (brick 2, `360fc720`), the MedCare probe (brick 3) is the single CONJECTURE→FINDING
> gate, and the per-consumer V3 identity migration (FMA, CPIC) has **not** started —
> OSINT-V3 was already shipped in #613 (a *prior* session), FMA-V3 and CPIC-V3 do not
> exist yet. **Canon:Custom is operator-DEFERRED** (§2.3) — do not flip it.
>
> **Git/process:** stay on `claude/serene-mayer-1a09he`; FF or `--force-with-lease`
> only (never bare `--force`); **do NOT open a PR unless explicitly asked**; **no
> model identifier in any committed artifact** (chat only); board-hygiene in the same
> commit (LATEST_STATE / plan); read-before-write. Minting a classid into the shipped
> `canonical_node.rs` needs an explicit operator **go**.

---

## 1. What this session actually shipped (commits on `claude/serene-mayer-1a09he`)

| commit | what | kind |
|---|---|---|
| `360fc720` | **`contract::facet_mint`** — the deterministic `(part_of:is_a)` rank-minter (brick 2) + `pub mod facet_mint;` + LATEST_STATE entry | **code** (additive, zero-dep) |
| `b32a1e0c` | merge `origin/main` — brought in #616 ("what" + my V3 alignment) and #617 ("why"), incl. `ast-as-partof-isa-address.md` | merge |
| `d8983720` | **plan §7** — folded the AST-as-address arc into `soa-value-tenant-migration-v2.md` as the *self-programming payoff* the two phases serve | docs |
| `ed06ff44` | **plan §2.3** — recorded the operator's **Canon:Custom deferral** | docs |

Also this session (not new commits on this branch): commented on **PR #616**
([#issuecomment-4806841615](https://github.com/AdaWorldAPI/lance-graph/pull/616#issuecomment-4806841615))
summarizing the V3 alignment + that brick 2 is built; authored the ast-doc V3-alignment
edits that merged via the #616/#617 line.

**This session touched ZERO lines of `canonical_node.rs`** — it migrated no classid.
Files changed: `facet_mint.rs` (new), `lib.rs` (one line), `soa-value-tenant-migration-v2.md`,
`LATEST_STATE.md`.

## 2. PRs in play

| PR | repo | state | what |
|---|---|---|---|
| #616 | lance-graph | **merged** | AST-as-address "what" + V3 alignment |
| #617 | lance-graph | **merged** | the three-layer "why" (assembler-cost AST / static-dynamic ladder / OGAR-as-importable-ERP-stdlib) |
| #29 | **ruff** (NOT in this session's repo scope) | open | `ruff_csharp_spo` Roslyn harvester — the brick-3 *producer* |
| `facet_mint` | lance-graph | **branch-only, no PR** | not opened — the operator didn't ask |

## 3. The arc in one picture — three bricks

```
source (C#/…) ──ruff_*_spo harvest──► SPO triples ──► NodeDecl{id, part_of_parent, is_a_parent}
                                                            │  facet_mint (BRICK 2 — BUILT)
                                                            ▼
                                          FacetCascade (BRICK 1 — LOCKED, shipped #613/#614)
                                          facet_classid(4) | 6×(8:8);  hi_chain=part_of, lo_chain=is_a
                                                            │  brick 3 (PENDING — the F-gate)
                                                            ▼
                              MedCare ruff_csharp_spo → mint → lance SoA → typeHierarchy/definition
                                          (MedCareV2 = parity oracle)  ⇒ CONJECTURE → FINDING
```

- **Brick 1 — slot allocation: LOCKED.** The 6-pair/12-slot layout *is* the shipped
  `facet::FacetCascade` (`facet_classid(4) | 6×(8:8) = 16 B`, `const _`-asserted).
  Nothing to ratify. The only open ordering is the classid half-order (Canon:Custom),
  which is **orthogonal to per-tier packing** and now **deferred** (§2.3).
- **Brick 2 — the rank-minter: BUILT** (`360fc720`). API:
  - `mint_facets(nodes: &[NodeDecl], facet_classid: u32) -> Result<Vec<FacetCascade>, MintError>`
  - `NodeDecl { id: u32, part_of_parent: Option<u32>, is_a_parent: Option<u32> }`
    (`::root(id)`, `::new(id, po, ia)`) — **producer-agnostic** (ruff C++ + Roslyn C#
    `ruff_csharp_spo` fill the same shape).
  - Walks root→node in each hierarchy, packs coarse→fine **sibling-ranks** (by sorted
    id ⇒ order-independent) into the 6 tiers: `tier.hi = po_rank`, `tier.lo = ia_rank`.
  - **Exact + roundtrip-lossless** for a finite AST (deterministic ranking, NOT learned
    PQ). **Exact-or-error**: `MintError::{DepthOverflow(>6 tiers), FanoutOverflow(>256
    siblings), UnknownParent, DuplicateId, Cycle}` — never silent aliasing.
    `MAX_TIERS=6`, `MAX_FANOUT=256`. `I-VSA-IDENTITIES`-clean (positions, not content).
  - `facet_classid` (row 0) is a **parameter** — bakes in NO half-order, so the minter
    is independent of the Canon:Custom decision.
  - **8 tests green; clippy `-D warnings` + fmt clean.** Verified in an **isolated
    zero-dep harness** (see §7 caveat) — the whole-workspace cargo build does not
    complete in this sandbox.
- **Brick 3 — the MedCare probe: PENDING.** `ruff_csharp_spo` harvest → `facet_mint`
  → lance SoA → `typeHierarchy`/`definition` query, diffed against the C# original
  with **MedCareV2 as the parity oracle.** This is the single step that promotes the
  arc (and the ~2 MB / 10× headline) from CONJECTURE to FINDING.

## 4. The honest migration table (audited — do not conflate plan work with migration)

| consumer | V3 state | by whom |
|---|---|---|
| **OSINT-V3** | shipped `CLASSID_OSINT_V3 = 0x1000_0700`, in `BUILTIN_READ_MODES` | **#613**, prior session |
| **FMA-V3** | shipped `CLASSID_FMA_V3 = 0x1000_0A01` + `ReadMode::FMA_V3` | **#618**, parallel genome session |
| **CPIC-V3** | shipped `CLASSID_CPIC_V3 = 0x1000_0E00` + `ReadMode::CPIC_V3` + `ConceptDomain::Genetics = 0x0E` | **#618**, parallel genome session |

**This (facet_mint) session migrated none of the three** — they shipped in #613 (OSINT,
prior) and #618 (FMA + CPIC, the parallel genome session). **As of the 2026-06-26 merge,
Phase-1 identity mints are COMPLETE** (verified on main). The flip then waits on §2.3
condition 3 (DEBT-1 + DEBT-2 of `v3-migration-debt-ledger-v1.md`; DEBT-4 RESOLVED by
#618) — an operator-adjudication point (the #618 handover reads the flip "unblocked";
condition 3 keeps DEBT-1/2 as gates).

## 5. Operator decision locked this session — Canon:Custom DEFERRED (§2.3)

The classid half-order flip `[custom(hi):canon(lo)] → [canon(hi):custom(lo)]` (so
the prefix sorts by **shared concept**, not render skin) is **deferred — do not flip
now.** It happens **only after ALL THREE**:
1. **Phase 1 complete**, AND
2. **OSINT + FMA + CPIC all re-encoded to V3** (current `custom:canon` convention +
   high-u16 `0x1000_xxxx` gen marker — OSINT done #613; FMA-V3 + CPIC-V3 to mint), AND
3. **the V3-migration technical debt is resolved AND V3 identity is confirmed working**
   (operator refinement 2026-06-26) — *a schema-prefix flip is a structural reorg; the
   operator will not reorder the prefix on top of unresolved migration debt or
   unconfirmed identity.* The debt §2.3 points at: (a) the unreverted POC-`Full`
   default `value_schema` (§2.1 L2 → canonical `Bootstrap`); (b) the §2.1 parity fuse
   is still structural-not-runtime → the OGAR-side `tail_variant` wiring is outstanding
   (OGAR #128 doc-only); (c) the §5 `/home/user/{OGAR,MedCare-rs}` casing-miss sweep;
   (d) the FMA-V3 + CPIC-V3 mints. "Confirm identity works" = the
   `I-LEGACY-API-FEATURE-GATED` field-isolation matrix + version-gate proof per V3 tail.

Then **flip once, atomically, over the whole V3 set.**

**Forcing reason (why not piecemeal):** the flip reinterprets *routing* on every
classid — post-flip `classid_concept_domain` reads the **HIGH** u16, not the low. A
half-flipped corpus mis-routes by construction (same `as u16 >> 8`, two meanings):
the `I-LEGACY-API-FEATURE-GATED` failure mode. So it must be a single coordinated
reorder over a known-complete set, carrying the version-gate + field-isolation
discipline when it lands.

## 6. Where I left off — ordered next moves

0. **Inventory the V3-migration debt — DONE** (`v3-migration-debt-ledger-v1.md`,
   2026-06-26, grounded from a tree audit): **DEBT-1** POC-`Full`→`Bootstrap` default
   (`canonical_node.rs:968-970`, OPEN); **DEBT-2** OGAR codes `tail_variant` + fuse
   structural→runtime (OGAR has it docs-only, OPEN); **DEBT-3** casing-miss sweep
   (MedCare-rs is a *clean* consumer — 0 V3 refs; the weight is DEBT-2); **DEBT-4**
   FMA-V3 + CPIC-V3 mints (+ Genetics slot pick, OPEN). The ledger carries an
   **append-only sign-off section** — other sessions ratify + add perspective there.
   Burn DEBT-1/2/4 green (each with the field-isolation + version-gate exit bar)
   **before** the flip.
1. **FMA-V3 mint** — the clean Phase-1 step: `CLASSID_FMA_V3 = 0x1000_0A01` (Anatomy
   route `0x0A01` intact), a `ReadMode::FMA_V3` const, and the `BUILTIN_READ_MODES`
   entry under `guid-v3-tail`. Mirrors the shipped OSINT-V3 pattern exactly. **Needs
   an explicit operator go** (touches shipped `canonical_node.rs`), per §5/CLAUDE.md.
2. **CPIC-V3 mint** — **blocked on a Genetics domain slot.** `0x0D` is already **HR**
   in `ogar_codebook.rs`; Genetics needs a free slot (`0x03–0x06` or `0x0E`). Pick the
   slot first (operator/codebook decision), then mint `CLASSID_CPIC_V3 = 0x1000_0?00`.
3. **Fix the debt + confirm identity works** — clear the items from move 0 and prove
   each V3 tail with the `I-LEGACY-API-FEATURE-GATED` field-isolation matrix +
   version-gate, end-to-end. This is §2.3 condition 3; the operator flips **only**
   after it is green.
4. **Phase 1 complete + debt clear + identity confirmed** ⇒ then (and only then) the
   **Canon:Custom flip** (§2.3, atomic over the whole V3 set).
5. **Brick 3 — MedCare probe** (can run in parallel with 0–3): scaffold the consumer
   half in `MedCare-rs` (`NodeDecl` ingest + `mint_facets` + `typeHierarchy`/
   `definition` assertions), leaving the `ruff_csharp_spo` wiring as the seam to fill
   once `AdaWorldAPI/ruff` is in scope. This is the CONJECTURE→FINDING gate.

**Open offers the operator has NOT yet taken (don't assume):** open a PR for
`facet_mint`; scaffold brick-3 in MedCare-rs; wire the harvest→`NodeDecl` adapter
(needs `ruff` added to scope).

## 7. Caveats (honest)

- **Whole-workspace cargo build does not complete in this sandbox.** The workspace
  resolves `ndarray` from git (`[patch.crates-io]` + `crates/helix` + `crates/
  perturbation-sim`), and cargo's recursive submodule init fails fetching the ndarray
  fork's `crates/burn/upstream` submodule (`AdaWorldAPI/burn.git @ 9b2b6712`) — proxy
  403 / revision-not-found. **This is pre-existing infra, unrelated to `facet_mint`**
  (a zero-dep additive module). `facet_mint` was therefore verified in an isolated
  zero-dep crate (copy `facet.rs` + `facet_mint.rs`, `cargo test`). The operator's
  steer: **ndarray is wanted; burn is a non-issue — do not chase it.** If you need to
  validate contract-crate changes, the isolated-harness pattern works; don't try to
  "fix" burn or drop ndarray.
- The arc is **CONJECTURE** (`[S]`→`[H]`): carrier + mechanism shipped, the economic
  claim (~2 MB / 10× vs ~250K LOC) **unmeasured**. Brick 3 is the only thing that
  moves it to FINDING. Don't let the §7 prose read as a proven result.

## 8. Iron rules / canon that bind this work

- **`I-VSA-IDENTITIES`** — the minter encodes identity *positions* (which sibling,
  which level), never bundles content. Any "improve the minter with PQ/superposition"
  idea violates this.
- **`I-LEGACY-API-FEATURE-GATED`** — the Canon:Custom flip, and *any* layout reclaim,
  needs a version gate + field-isolation matrix tests. This is why the flip is atomic.
- **CANON node layout** (CLAUDE.md): `key(16) | edges(16) | value(480)`; `classid`
  routes on the **low** u16 today (`0xDDCC` codebook). The `FacetCascade` is a
  *reading* over 16 value bytes — it does NOT touch the locked 480 B layout.
- **Git/model policy:** branch `claude/serene-mayer-1a09he`; FF / `--force-with-lease`
  only; no PR unless asked; **no model identifier in committed artifacts**; board
  hygiene in-commit; read-before-write.

---

**Status:** OPEN — Phase-1 FMA-V3/CPIC-V3 mints + brick-3 probe pending; Canon:Custom
deferred; `facet_mint` built and on-branch (no PR).
