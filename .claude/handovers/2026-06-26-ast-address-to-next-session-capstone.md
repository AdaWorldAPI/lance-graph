# Handover Capstone — AST-as-(part_of:is_a) / OGAR-ERP-stdlib arc → next session

> **From:** session on `claude/medcare-bridge-lance-graph-wmx76z` (2026-06-26)
> **To:** next session picking up the OGAR-ERP / V3 AST-address arc
> **Status of the arc:** CONJECTURE captured + carrier verified; ONE empirical
> probe converts it to FINDING. No V3 *migration* was performed this session.

---

## 0. TL;DR (read this, then the mandatory reads)

This session **did not write product code**. It (a) verified another session's
V3 alignment of a design doc, (b) consolidated a multi-layer architectural
thesis into one knowledge doc, and (c) scaffolded a C# harvester. The thesis:
**a transcode source's structural AST is stored AS the `(part_of:is_a)` GUID
address (shipped `FacetCascade`), making ERP construction in any consumer as
cheap as importing pre-minted OGIT class primitives + classaction pointers.**
The whole arc is CONJECTURE; the gate to FINDING is the **MedCare brick-3 probe**
(§6). Everything else is wiring, not research.

**Single most important correction to inherit:** the carrier and minter are
already SHIPPED/built. Do not re-design them. The remaining work is *wiring*
(mint imported OGIT → codebook; merge the minter branch; run the probe).

---

## 1. Mandatory reads to get up to speed (in order)

1. **`.claude/knowledge/ast-as-partof-isa-address.md`** — THE doc this arc
   produced (merged via #616, extended via #617). The full thesis + the three
   honest boundaries. Start here.
2. **`.claude/knowledge/guid-canon-and-prefix-routing.md`** — the V3 GUID canon.
3. **`crates/lance-graph-contract/src/facet.rs`** — the SHIPPED carrier
   (`FacetCascade` = `facet_classid(4) | 6×(8:8) = 16 B`, content-blind,
   `(part_of:is_a)` is one consumer projection). Four SIMD lanes.
4. **`crates/lance-graph-contract/src/canonical_node.rs`** — `TailVariant::V3`,
   `mint_for`, `CLASSID_OSINT_V3 = 0x1000_0700` (the one wired V3 exemplar).
5. **`crates/lance-graph-contract/src/class_view.rs`** — the static-shape →
   dynamic-ClassView → askama ladder (XML/XSD/XSLT analogy), C2/N3 invariants.
6. **`crates/lance-graph-contract/src/ogar_codebook.rs`** — the import surface
   (`canonical_concept_id`, `render_classid_for_concept`).
7. **`.claude/knowledge/ogar-consumer-preflight.md`** + **`core-first-transcode-doctrine.md`**
   — the pull-don't-reconstruct iron rule + the Core-first doctrine.
8. **OGAR repo:** `docs/OGAR-AST-CONTRACT.md` (the THINK arm `Class` / DO arm
   `ActionDef`+`ActionInvocation`+`KausalSpec` IR) and
   `vocab/imports/ogit/` (complete OGIT, ~1,940 TTL, IMPORTED).

---

## 2. What I shipped this session (all PRs)

| PR | Repo | State | Content |
|---|---|---|---|
| **#616** | lance-graph | **MERGED** | `ast-as-partof-isa-address.md` (the *what*) + I aligned another session's V3 correction onto it. Now on `main`. |
| **#617** | lance-graph | **OPEN** | The *why*: three-layer thesis + laws-as-patterns + OGIT-imported + DO-as-classaction-pointer + the 2 MB headline. Branch `claude/medcare-bridge-lance-graph-wmx76z`, HEAD `e34a7fa`. **Docs only**, safe to merge independently. |
| **#29** | ruff | **OPEN** | `ruff_csharp_spo` — Roslyn (.NET) C#→SPO harvester (brick-3 input). **CI-gated, not locally verified** (no .NET SDK here; proxy blocks workspace `cargo check`). |

**No code/contract surface was changed in lance-graph.** `#617`'s diff is one
knowledge file, +186 lines.

---

## 3. The thesis in one screen (so you don't re-derive it)

- **Layer 1 — rails-shaped semantic AST at *assembler* cost.** `facet.rs`'s four
  SIMD lanes ARE the LSP ops (`definition`=`vpcmpeqd`+`vmovmskps`,
  `typeHierarchy`=`lo_chain` `vpxor`+`tzcnt`, `documentSymbol`=`hi_chain`,
  ancestry=GFNI Morton). Only the *declarative* (rails) class body flattens to
  fixed-width tiles; an imperative syntax tree does not.
- **Layer 2 — static OGAR / dynamic ClassView / askama.** One ClassView projects
  the `N × 6×(part_of:is_a)` SoA two ways: row→typed **view** (THINK→askama) AND
  row→conditional **classaction** (DO→`ActionDef` under `KausalSpec` guard).
  C2 (presence≠semantics) + N3 (append-only positions) = a row view that evolves
  17 years (Redmine) without breaking persisted rows.
- **Layer 3 — OGAR as importable ERP-LAW primitives + lance as compiler.** The
  primitives are *laws/regulations* (universal, reusable), not CRUD:
  `ObjectView`=required fields, `ActionDef`/`KausalSpec`=rules. Content stays
  with the consumer; only the pattern is shared. `ogar_codebook` is the import
  surface. **DO in the consumer = a classaction POINTER** (`classid → ActionDef`
  via `ActionInvocation` DTO) — exactly the OGAR consumer doctrine ("classid is
  pure address; magic at the resolution target").
- **Headline (CONJECTURE):** OpenProject + Odoo as ~2 MB of GUID-encoded
  `(part_of:is_a)` vs ~20 MB / ~250K LOC (~10×). Dimensional check: 16 B/node ⇒
  2 MB ≈ 131K nodes — credible for both ERPs' structural skeleton.

---

## 4. V3 state — what EXISTS vs what I did (IMPORTANT, do not misattribute)

**I performed NO V3 migration this session.** I only *read/verified* the shipped
V3 surfaces to ground the docs. The V3 mints shipped in OTHER sessions
(#613/#614/#615, then **#618** which landed AFTER my #617 merged):

| domain | classid | status (as of #618, 2026-06-26) |
|---|---|---|
| **OSINT-V3** | `0x1000_0700` | wired (first exemplar, #613-615) |
| **FMA-V3** | `0x1000_0A01` | **wired** (#618 — `ReadMode::FMA_V3`) |
| **CPIC-V3** | `0x1000_0E00` | **wired** (#618 — Genetics domain `0x0E` operator-allocated; `ReadMode::CPIC_V3`) |

**#618 "Phase 1 V3 set complete"** — OSINT + FMA + CPIC all minted, Genetics
domain `0x0E` allocated. So CPIC/FMA/OSINT are ALL wired now (none by me).

The V3 marker lives in the **HIGH (custom) u16**; canon/domain in the **LOW u16**
(`classid_concept_domain` routes on the low half — a deliberate Codex-P1 fix; the
low-half `0x1007` alternative was rejected as domain `0x10`=Unassigned). **Open
operator decision: "Canon:Custom"** — whether to flip canon to the high half
(would require re-pointing `classid_concept_domain`). Not settled; flagged in the
doc, not hardened.

---

## 5. The minter (brick 2) — BUILT and verified, on an unmerged branch

- **`lance_graph_contract::facet_mint`** — `mint_facets(&[NodeDecl], facet_classid)
  -> Result<Vec<FacetCascade>, MintError>`. Deterministic sibling ranking,
  roundtrip-lossless, `MintError::{DuplicateId, UnknownParent, DepthOverflow(>6),
  FanoutOverflow(>256), Cycle}`. `NodeDecl{id, part_of_parent, is_a_parent}` is
  producer-agnostic (C++/C#/Ruby/Python harvests all fill it).
- **Branch `claude/serene-mayer-1a09he`, commit `360fc720`. NOT merged to main.**
- **I ran its 8 tests myself: 8 passed**, in the zero-dep `lance-graph-contract`
  crate (no ndarray/burn needed).

---

## 6. THE critical path — brick 3 (the probe that converts CONJECTURE→FINDING)

`ruff_csharp_spo` harvest (MedCare C#) → `facet_mint` → lance SoA → `typeHierarchy`
/`definition` LSP query, **with `AdaWorldAPI/MedCareV2` as the parity oracle**.
It produces the first real numbers for: (a) bytes-of-(part_of:is_a) vs source LOC
(the 10× claim), (b) lossless round-trip of the class graph, (c) query-as-asm-op.

**Its gates are wiring, not research — in order:**
1. **Merge `serene-mayer` (`facet_mint`)** into main (verified green).
2. **Mint imported OGIT → `ogar_codebook`** — the source is ALREADY in OGAR at
   `vocab/imports/ogit/` (~1,940 TTL incl. `NTO/{Audit,Compliance,Legal,Security}`
   + `SDF/Automation` = HIRO DO-arm source). The remaining step is the
   **source→codebook wire** (`facet_mint` → publish), NOT harvesting.
3. **Reserve one `facet_classid`** in the value-tenant migration's codec selector
   for the AST `6×(8:8 part_of:is_a)` reading (a ClassView reading, NOT a new
   `ValueSchema` variant — see `soa-value-tenant-migration-v1-harvest.md §5.1`).
4. Then run the MedCare probe; needs a .NET 8 SDK to run the Roslyn harvester.

---

## 7. Honest caveats / corrections I made (inherit these — don't repeat them)

- **`burn` is NOT a build blocker.** An earlier claim ("workspace compile blocked
  by proxy 403 on ndarray's `burn` submodule") was **wrong** — `burn` is an
  *excluded* workspace member (`ndarray/Cargo.toml:375`) with zero code coupling.
  The minter's crate (`lance-graph-contract`) is zero-dep and builds standalone.
- **The local `owl` repo is NOT W3C OWL.** It's camel-ai's "Optimized Workforce
  Learning" multi-agent framework (arXiv:2505.23885). The operator's "OWL↔OGIT
  bijective-lossless" result refers to W3C OWL (the ontology standard); I cited it
  **attributed, not re-verified** (the converter is plausibly OGIT's
  `bin/ogit-tools-jar-with-dependencies.jar`, unread).
- **Everything in the doc is CONJECTURE.** The carrier is shipped, but no
  end-to-end measurement exists. Keep the label until the probe runs.
- **Sync-first discipline:** every PR this session verified `HEAD..origin/main`
  empty + `diff --stat origin/main..HEAD` showing only intended files BEFORE
  pushing (a prior arc orphaned commits by not doing this). Keep doing it.
- **Workspace iron rules in play:** OGAR — no model identifier in artifacts, no
  German PII labels, no Bash grep/sed/tail/head/awk (use Grep/Read tools).
  Consumer side — pull via `*Port::class_id` / `canonical_concept_id`, NEVER
  construct a `*Bridge` or copy the codebook.

---

## 8. Concrete next actions (pick up here)

1. **Decide:** merge #617 (docs, safe) — or leave for review.
2. **Brick-3 wiring step 1:** merge `claude/serene-mayer-1a09he` (`facet_mint`).
3. **Brick-3 wiring step 2 (highest leverage):** write the OGIT-source→`ogar_codebook`
   mint — the input (`vocab/imports/ogit/`) is already in OGAR; this is the step
   that first produces real byte-size numbers for the 10× claim.
4. **Decide the open gate:** the "Canon:Custom" classid half-order (operator call;
   the importable-stdlib reuse-clustering argues canon-high, but it collides with
   the shipped low-half domain routing — not free).
5. **Optional:** verify ruff #29 in CI (needs network + .NET SDK).

---

## 9. Coordinates

- lance-graph branch: `claude/medcare-bridge-lance-graph-wmx76z` (HEAD `e34a7fa`,
  synced FF to `main` which includes #616 at `b7eb02c`).
- minter: `claude/serene-mayer-1a09he` @ `360fc720` (unmerged).
- ruff branch: `claude/medcare-bridge-lance-graph-wmx76z` (PR #29, scaffold @ `0f463f2`).
- OGIT-in-OGAR: `OGAR/vocab/imports/ogit/` (~1,940 TTL).
- Operator note (not code): MedCare Railway deploy still binds `:3000` because a
  Railway **PORT service variable** overrides `$PORT` — the operator must remove
  it; the app correctly prefers `$PORT`.
