# Issues Log — Open + Resolved (double-entry, append-only)

## 2026-07-04 — ISS-V1-TAIL-RESIDUE — two pre-existing `NodeGuid::new` (V1 `u24+u24`) mint sites must migrate to V3 (`mint_for` / V3-marked classid)

**Status:** OPEN — **MIGRATION MANDATORY** (operator ruling 2026-07-04, `E-V1-TAIL-FORBIDDEN-V3-IS-CONTENT-BLIND-1`). Deferred in *timing*, not in *obligation*; NOT to be churned into unrelated PRs. Owner: whoever next moves each output path onto a V3-marked classid.

**The residue.** The flat V1 tail `family(u24) ++ identity(u24)` is forbidden for new units; V3 is the content-blind `classid(4)+12B` facet (`E-V3-FACET-4-PLUS-12`). A read-only conformance audit of the whole repo found the V1 tail *produced* at exactly two live sites, both hardcoding `NodeGuid::new(...)` instead of the canonical `mint_for(classid_read_mode(c).tail_variant, …)` dispatch:
- `crates/lance-graph-contract/src/ocr.rs:121` — the #496 OCR→`NodeRow` keystone.
- `crates/lance-graph-contract/src/aiwar.rs:104` — the aiwar `NodeRow` builder.

Both currently target V1-default/OSINT classids, so they are **behaviorally correct today** — the defect is that they bypass the `mint_for` dispatch that is supposed to make a class's V1→V3 flip a one-line registry change. Everything else in the repo is either a test (`#[cfg(test)]`) or a legitimate legacy-compat *read* (`family()`/`identity()` fallback arms in `soa_graph.rs`, `hhtl.rs` prefix routing) — reads stay, per `I-LEGACY-API-FEATURE-GATED`; only new *mints* are forbidden.

**Resolution (mandatory, when each output path is next touched).** Route each site through `mint_for(classid_read_mode(classid).tail_variant, …)` with a V3-marked classid; add a `/v3-audit` grep that forbids new `NodeGuid::new(` in non-test code so the guard is mechanical. Blocker to note: `mint_for`/`new_v2` sit behind `guid-v2-tail` (default-off) — un-gate `mint_for` (V1 arm unconditional, V2/V3 under the feature) before pointing production mints at it.

**Cross-ref:** `EPIPHANIES.md` `E-V1-TAIL-FORBIDDEN-V3-IS-CONTENT-BLIND-1`, `E-V3-FACET-4-PLUS-12`, `canonical_node.rs` (`TailVariant`, `mint_for`, `new`), OGAR PR (canon supersession) + `D-V1-TAIL-RETIRED`.

## 2026-07-01 — ISS-Q2-CASCADE3-NIBBLE-ANCESTRY — q2 `cascade3` FNV bytes are byte-hierarchical but NOT nibble-hierarchical; HHTL routing over bake mints is sound only at whole-tier granularity

**Status:** OPEN (falsifier specified, not yet run — q2 push gate WAIVED 2026-07-02, "temporary precaution … you can unarm that"; runnable now, D-VCW-5). Owner:
q2 `cpic/src/lib.rs::cascade3` (+ any bake reusing it) vs the OGAR canon's
256=4⁴ hierarchical-codebook condition. Plan: `v3-convergence-wiring-v1.md` §5.

**The claim tension.** OGAR canon: each tier's 256-entry codebook is a 4-level
4-ary centroid HIERARCHY so a byte's nibbles are the centroid's ancestry —
`is_ancestor_of` = containment, prefix routing rigorous at nibble depth. q2's
`cascade3` derives tier byte `i` as the FNV-1a low byte of the cumulative DN
prefix at depth `i`: siblings share leading BYTES (per-tier prefix routing
holds), but a hash byte's nibbles carry NO ancestry — below whole-byte
granularity the tree structure is noise.

**Falsifier (runnable in q2 when opened):** two DNs sharing a 3-deep prefix
must show `common_prefix_depth` at nibble granularity ≈ random beyond the
shared-byte boundary (vs the 4⁴ condition's prediction of structured nibble
sharing). Confirmed ⇒ either (a) HHTL routing over bake mints clamps to tier
granularity (document the boundary), or (b) the cascade generator moves to a
hierarchical codebook (bigger change, operator call). No routing code should
assume sub-byte ancestry on these mints until this runs.

---

## 2026-07-01 — ISS-Q2-CPIC-MIRROR-DIVERGES-FROM-CPIC-V3-REGISTRY — q2's local `cpic::NodeGuid` mirror is V1-layout-parity-true but diverges from the registered CPIC-V3 read-mode on BOTH domain and tail shape

**Status:** RESOLUTION RULED 2026-07-02 (execution queued as flip P3 / D-CCF-3) — the operator ruling ("Same for cpic also under q2, which has a different domain for separation") + the triggered canon:custom flip settle this as shape (a) below, with the target classid updated by the flip: q2's CPIC class = Genetics:q2 = `0x0E:01::…` (post-flip stored `0x0E01_1000`; the pre-flip root `0x1000_0E00` normalizes `:00`→`:01`). q2 push gate WAIVED. cpic re-mints by PULLING the contract (`mint_for(classid_read_mode(…).tail_variant, …)`), retiring the local mirror + the `0x0C` domain. See `classid-canon-custom-flip-v1.md` §2/§4. Owner: q2 `cpic/src/lib.rs` +
`lance-graph-contract::canonical_node` (CPIC-V3 registry). Surfaced 2026-07-01
by a verify-the-mirror read after the V3 tenant-carve certification.

**Ground truth (read, not grepped).** q2 `cpic/src/lib.rs`:
- `NodeGuid::mint(classid, part[3], isa[3], family, identity)` builds
  HEEL/HIP/TWIG as `(part<<8)|isa` — the V3 `(part_of:is_a)` 8:8 tile. ✓ canon.
- `key16()` packs `classid·heel·hip·twig·family(u24)·identity(u24)` LE —
  **byte-identical to the contract's V1 layout** (its parity doc-comment is
  TRUE at the byte-order level). But that is the **V1 tail**, not the V3
  (`leaf·family·identity` 3×u16) tail the registry's
  `ReadMode::CPIC_V3.tail_variant = V3` reads.
- classids are `0x000C_0001..0x000C_0006` (`CID_GENE..CID_REC`) — domain
  **`0x0C`**, NOT the operator-allocated Genetics `0x0E`
  (`CLASSID_CPIC_V3 = 0x1000_0E00`), and no `0x1000` V3 gen-marker.

**So:** V3 *tiles* on a V1 *tail* under an unregistered *domain* — three
divergences from the wired CPIC-V3 read-mode. A bake produced with this mirror
will not resolve to `ReadMode::CPIC_V3` (falls to `ReadMode::DEFAULT`) and its
tail bytes read differently under the registry's V3 lens.

**Stale-brief correction (same sweep):** `soa-value-tenant-migration-v1.md`
§2.5's blocker — "q2 `osint-bake/fma.rs` calls `NodeGuid::new_v2(...)`, a
7-group API that does **not** exist" — is stale on both halves: `new_v2` DOES
exist (7 groups, feature `guid-v2-tail`, shipped + matrix-tested), and no
`new_v2` call site exists in q2 today (grep: only `cpic::NodeGuid::mint`).

**CORRECTION (2026-07-01, same session — the previous paragraph's second half
is WRONG; truncated-grep error, head_limit cut before osint-bake):** q2
`osint-bake` DOES call `new_v2` — `crates/osint-bake/src/lib.rs:606` mints the
classid-`0x0700` OSINT rows via `NodeGuid::new_v2(NodeGuid::CLASSID_OSINT, …)`
(also `:745`), and it imports the REAL contract
(`use lance_graph_contract::canonical_node::{NodeGuid, classid_read_mode}`) —
no mirror in osint-bake; only `cpic` carries the local mirror. What stands:
the brief's "API does not exist" half is stale (`new_v2` exists and q2 links
it fine). NEW observation for the same operator decision: osint-bake's OSINT
rows mint a **V2 tail** directly (`new_v2`) for legacy `CLASSID_OSINT`, whose
registered read-mode is `tail_variant = V1` — the known per-classid-legacy-tail
pending noted in `ReadMode::DEFAULT`'s docs, while its FMA bins already use the
sanctioned `mint_for(classid_read_mode(c).tail_variant, …)` dispatch. The V3
class `CLASSID_OSINT_V3 = 0x1000_0700` exists precisely for that migration.

**Resolution paths (operator decision):** (a) q2 cpic re-mints via the
contract's `mint_for(classid_read_mode(CLASSID_CPIC_V3).tail_variant, …)`
pull (consumer-preflight shape — pull, never mirror); (b) the registry gains
the `0x0C` pharmacogenomics classids q2 actually minted; or (c) the q2 POC is
declared registry-exempt (bake-only) and its parity comment is scoped to
"V1 byte layout" explicitly. No action taken pending direction.

---

## 2026-07-01 — ISS-OSINT-SYSTEM-ROOT-SLOT-VIOLATION — OGAR shipped `osint_system` at the reserved `0x0700` root slot; the lance-graph mirror canon forbids it (`CC==0x00` = domain root, reserved) — the parallel-mirror is BLOCKED on a remap decision

**Status:** RESOLVED 2026-07-02 (operator ruling, executed in OGAR PR #146) — and the resolution is SHARPER than either recorded option: "OSINT Person was a hallucination"; within the OSINT domain the low byte is APPID space applied domain-wise (`00` = the domain itself, `01` = q2 the consumer), so OSINT contributes **zero vocabulary rows**. OGAR #146 removed BOTH #145 mints (`osint_system@0x0700` AND `osint_person@0x0701`); count 67 → 65 == the mirror's 65 — the COUNT_FUSE balances with ZERO mirror-side changes, and the zero-slot invariant is untouched. Options A and B below are preserved as history; neither was taken (B came closest — its addendum's two-id-spaces reading is confirmed, but even the `0x0701` "concept" was a mislabel on q2's appid slot). See `E-CLASSID-CANON-HIGH-TRIGGERED` + `.claude/plans/classid-canon-custom-flip-v1.md` §0 for the full ruling (which also triggers the canon:custom flip).

**The violation.** The shared codebook canon (documented in `ogar_codebook.rs` module header: *"`CC == 0x00` = the domain root, reserved"*) requires every concept id `0xDDCC` to have `CC ≥ 0x01`; `0x__00` is the domain-root/default, NOT a concrete concept. OGAR main ships **`("osint_system", 0x0700)`** — `CC == 0x00`, the reserved root. `("osint_person", 0x0701)` is valid (`CC==01`, operator-frozen). The lance-graph mirror enforces the canon via the workspace-member test `codebook_has_no_duplicate_ids_or_zero_concept_slot` (`assert_ne!(id & 0x00FF, 0x00)`), so **mirroring `0x0700` fails lance-graph's own default CI** (748 pass, 1 fail). The `COUNT_FUSE` (in the *excluded* `lance-graph-ogar`) is a separate, downstream break; this one is in-tree.

**Current blast radius.** lance-graph main's default CI is GREEN (mirror still 65, zero-slot test passes; the `COUNT_FUSE` lives in the excluded `lance-graph-ogar`). Consumers vendoring `lance-graph-ogar` against OGAR-main-67 vs mirror-65 will break on the count fuse. The parallel-mirror fix is **blocked** because the obvious "+2 rows" fix trips the zero-slot invariant.

**Decision needed (operator).** Two coherent reads of `osint_system @ 0x0700`:
- **Option A — it's a concrete concept → remap.** Move `osint_system` to `0x0702` in OGAR (fresh PR; `0x0701` frozen for `osint_person`); mirror `{0x0701, 0x0702}` (count 67); update q2 `OSINT_SYSTEM_CLASS 0x0700 → 0x0702`. Canon satisfied, but a merged id moves + q2 change.
- **Option B (recommended) — `0x0700` IS the OSINT domain root/default class, not a counted concept.** This is exactly what the canon reserves `0x__00` for ("zero = fall through to the broader default"). OGAR drops `osint_system` from the *concept* `CODEBOOK`/`class_ids::ALL` (keep an `OSINT_SYSTEM = 0x0700` const documented as the domain-root class if useful); `ALL` → 66; mirror carries only `("osint_person", 0x0701)` → 66; the fuse balances at 66; q2 keeps `0x0700` as the renderable domain-default class (canon-legal: the root IS a real default class, just not a codebook *concept* row). No id moves; aligns with the user's "0x0701 is the frozen concept" framing.

Both are OGAR-side follow-ups (OGAR #145 is merged) landed in parallel with the lance-graph mirror rows, per `E-OGAR-LANCEGRAPH-MOVE-IN-PARALLEL`.

**ADDENDUM (2026-07-01, later session — ground-truth strengthening of Option B;
still the operator's decision, no action taken):** the codebase ALREADY lives
Option B's distinction. There are two id spaces aliasing in the lo u16: the
**classid space** (what nodes mint under) and the **concept-vocabulary space**
(what the codebook counts). Evidence: `canonical_node.rs` ships
`CLASSID_OSINT = 0x0000_0700` as a LIVE registered class (`ReadMode::OSINT` in
`BUILTIN_READ_MODES`) and q2 `osint-bake/src/lib.rs:606` mints real `0x0700`
rows — while the mirror's zero-slot invariant only ever governed *vocabulary
rows*. So `0xDD00` = "the ONE class per domain" (valid classid, exactly the
operator's "OSINT is ONE class") and simultaneously "not a nameable concept"
(no codebook row) — no contradiction once the spaces are named. Under this
reading OGAR's `osint_system` mint was the same move lance-graph already made,
just landed in the wrong space (a vocabulary row instead of a classid const).
Option B resolves it without deleting the idea or moving any id.

## 2026-07-01 — ISS-OGAR-OSINT-MIRROR-PENDING — OGAR #145's OSINT mint (+2 to `class_ids::ALL`) breaks the contract-mirror `COUNT_FUSE` on merge; the paired lance-graph mirror rows must land in the same arc

**Status:** RESOLVED 2026-07-02 — dissolved by the same operator ruling as `ISS-OSINT-SYSTEM-ROOT-SLOT-VIOLATION`: OGAR PR #146 removes both #145 OSINT mints (67 → 65), so the fuse balances with NO mirror rows to land; the "2 mirror rows in parallel" path below is moot (preserved as history). The fuse itself stays, per the earlier ruling ("keep the fuse — it IS the dependency contract"). · Original: **Resolution path RULED by operator 2026-07-01: keep the fuse (it IS the dependency contract enforcing OGAR↔lance-graph parallel movement); do NOT pin to a rev — "option 1" is REJECTED. Land the 2 mirror rows + `domains_agree` arm in parallel with OGAR #145 (option 2 / coordinated merge; brief transient red is acceptable — "the fuse is okay for now"). See `E-OGAR-LANCEGRAPH-MOVE-IN-PARALLEL`.** · Owner: OGAR `ogar-vocab` (PR #145) + `lance-graph-contract::ogar_codebook` mirror + `lance-graph-ogar::parity::domains_agree`. Surfaced 2026-07-01 while self-reviewing PR #624 / #145. Same cross-repo-arc shape as `ISS-OGAR-AUTH-MIRROR-DRIFT` (which took medcare CI red) and `ISS-OGAR-GENETICS-MIRROR-PENDING`; cited by `E-CODEBOOK-MINT-IS-A-CROSS-REPO-ARC`.

**READY PATCH (apply to lance-graph the moment OGAR #145 is on OGAR main; NOT to #624 while OGAR main is still 65 — that breaks #624's own fuse):** in `crates/lance-graph-contract/src/ogar_codebook.rs` add the two rows `("osint_system", 0x0700), ("osint_person", 0x0701)` to `mirror::CODEBOOK` (65 → 67); add the `(O::Osint, C::Osint)` arm to `lance-graph-ogar::parity::domains_agree` (the `ConceptDomain::Osint` enum + `0x07 => Osint` route already exist). Then `mirror::CODEBOOK.len() == ogar_vocab::class_ids::ALL.len()` (67 == 67) restored.

**The break.** OGAR PR #145 mints `osint_system` (0x0700) + `osint_person` (0x0701) into `ogar_vocab::class_ids::ALL` (+2). `lance-graph-ogar` pins `ogar-vocab = { git = ".../OGAR", branch = "main" }` (tracks main, NOT a rev), and carries the compile-time `COUNT_FUSE`: `assert!(mirror::CODEBOOK.len() == ogar_vocab::class_ids::ALL.len())` (`lance-graph-ogar/src/lib.rs:119`). The contract mirror `lance-graph-contract::ogar_codebook::CODEBOOK` currently has **65 rows with NO osint entries** (it reserved `ConceptDomain::Osint` + the `0x07 => Osint` route + a domain-nibble test, but not the two concept rows). So **the instant #145 merges to OGAR main, `COUNT_FUSE` fires `error[E0080]` in every consumer vendoring `lance-graph-ogar`** — medcare, smb, woa, etc.

**Why the mirror rows can't just be added to PR #624 now.** #624's `lance-graph-ogar` compiles against OGAR **main**, which still has 65 (osint mint is unmerged on #145). Adding +2 to the mirror now → mirror 67 vs OGAR-main 65 → breaks #624's OWN CI. The two sides are chicken-and-egg across the `branch = "main"` tracking.

**Resolution (coordinated arc, per the auth precedent):** land in lock-step —
1. OGAR #145 merges to OGAR main (ALL → 67); **at this moment lance-graph main's `COUNT_FUSE` goes red** (known transient, as with the auth mint).
2. Immediately merge a lance-graph change adding the 2 osint rows to `ogar_codebook::CODEBOOK` (`("osint_system", 0x0700)`, `("osint_person", 0x0701)`) + the `(O::Osint, C::Osint)` arm to `lance-graph-ogar::parity::domains_agree` → 67 == 67 restored.
   - The `ConceptDomain::Osint` enum + `0x07 => Osint` route already exist in the mirror, so only the 2 CODEBOOK rows + the `domains_agree` arm are missing.

**Merge-ordering decision needed from operator:** whether to (a) merge #145 + the mirror follow-up back-to-back accepting the brief transient red, (b) hold #145 until the mirror PR is staged, or (c) pin `lance-graph-ogar` to a rev instead of `branch = "main"` to decouple the cadence. Flagged to the operator 2026-07-01.

## 2026-06-26 — ISS-OGAR-GENETICS-MIRROR-PENDING — contract mirror gained `ConceptDomain::Genetics` (0x0E) ahead of OGAR; the `domains_agree` arm + OGAR side follow

**Status:** OPEN (tracked) · Owner: OGAR `ogar-vocab` + `lance-graph-ogar` · Surfaced by: CodeRabbit on #618. The same cross-repo-arc shape as `ISS-OGAR-AUTH-MIRROR-DRIFT` / `E-CODEBOOK-MINT-IS-A-CROSS-REPO-ARC`, but **domain-only** so it does not break in isolation.

#618 added `ConceptDomain::Genetics` + `0x0E => Genetics` to the contract mirror (`ogar_codebook.rs`) so CPIC-V3 `0x1000_0E00` routes Genetics (operator-allocated 2026-06-26). OGAR's `ogar_vocab::ConceptDomain` has **no Genetics variant yet**, and `lance-graph-ogar::parity::domains_agree` (`lib.rs:128-148`) still stops at `HR`/`Unassigned`. **Why it's safe in isolation (not a build break like the Auth drift):** the addition is a *domain enum variant + route*, NOT a CODEBOOK **concept** — `mirror::CODEBOOK.len()` is unchanged, so the compile-time `COUNT_FUSE` still holds, and `assert_codebook_parity` iterates CODEBOOK concept-ids (none at `0x0E`), so `domains_agree(0x0E00)` is never called. `domains_agree` is a `matches!` (never exhaustiveness-checked), so adding `C::Genetics` does not break compile either; the `(O::Genetics, C::Genetics)` arm **cannot** be added today because `O::Genetics` does not exist.

**Resolution (the coordinated arc, when Genetics concepts are minted):** (1) OGAR `ogar-vocab` adds `ConceptDomain::Genetics` + `0x0E => Genetics` + any Genetics concept rows; (2) the contract mirror's `CODEBOOK` gains the matching concept rows (keeping `COUNT_FUSE` balanced); (3) `lance-graph-ogar::parity::domains_agree` gains the `(O::Genetics, C::Genetics)` arm. Per `E-CODEBOOK-MINT-IS-A-CROSS-REPO-ARC`, those three land together, never split. Until then the drift guard correctly reflects "contract ahead of OGAR on the Genetics domain."

## 2026-06-23 — ISS-OGAR-AUTH-MIRROR-DRIFT — `0x0B` AuthStore mint broke the contract mirror's COUNT_FUSE in every consumer

**Status:** RESOLVED 2026-06-23 (this commit). OGAR `ogar-vocab` PR #110 minted the `0x0B` AuthStore family (4 concepts: auth_store 0x0B01, auth_zitadel 0x0B02, auth_zanzibar 0x0B03, auth_ory_keto 0x0B04) and merged to OGAR `main`, taking `ogar_vocab::class_ids::ALL` from 39 → 43. The paired `lance-graph-contract::ogar_codebook::CODEBOOK` mirror was NOT updated in the same arc, so the compile-time `COUNT_FUSE` in `lance-graph-ogar` (`assert!(mirror::CODEBOOK.len() == ogar_vocab::class_ids::ALL.len())`) fired `error[E0080]` (`vendor/lance-graph/crates/lance-graph-ogar/src/lib.rs:113`) in **every** consumer vendoring the OGAR git dep — medcare CI went red on `cargo build`. **Resolution:** added the 4 auth rows + `ConceptDomain::Auth` + `0x0B => Auth` to the mirror, and the `(O::Auth, C::Auth)` arm to `lance-graph-ogar::parity::domains_agree` (else the runtime `assert_codebook_parity` test panics). 43 == 43 restored; `cargo test -p lance-graph-contract` green. **Process fix (see EPIPHANIES E-CODEBOOK-MINT-IS-A-CROSS-REPO-ARC):** an OGAR concept mint is a cross-repo arc — the OGAR entry + the contract mirror + the `domains_agree` arm land together, never split across sessions. **Merge note (2026-06-23):** main landed #595 (auth sync) + #597 (PRODUCT + ACCOUNTING_ACCOUNT, OGAR #111) first; on merge this branch took main's superset `ogar_codebook.rs` (45 concepts incl. the `AppPrefix` render layer), so the auth mirror rows here are subsumed — the `domains_agree` Auth arm + this finding stand.

## 2026-06-22 — ISS-CONTRACT-APP-PREFIX-MIRROR — `contract::ogar_codebook` lacks the OGAR#97 `APP_PREFIX` / `render_classid_for` mirror, so membrane consumers must hand-stamp the hi-u16 render prefix

**Status:** RESOLVED 2026-06-22 (`claude/contract-app-prefix-mirror`) · Owner: lance-graph-contract · Surfaced by: `.claude/knowledge/ogar-consumer-preflight.md` (the consumer spellbook).

**Resolution:** `contract::ogar_codebook` now mirrors the hi-u16 APP-prefix layer — `AppPrefix` (the OGAR#95 §2 allocation table as typed data: `0x0001` OpenProject / `0x0002` Odoo / `0x0003` WoA / `0x0004` SMB / `0x0005` Healthcare / `0x0007` Redmine), `render_classid` + `render_classid_for_concept` (compose), `classid_app_prefix` + `classid_concept` (decompose). A membrane consumer (BBB-safe) now pulls BOTH halves from one source — no hand-stamped `0x000N`. Wire-compat parity test `app_prefixes_match_ogar_allocation_table` pins the prefixes against OGAR `PortSpec::APP_PREFIX`; `render_classid_composes_decomposes_and_preserves_the_concept_half` pins the `0x0005_0901` MedCare-patient worked example. Mirrors OGAR#97 (`ogar_vocab::app`), following the OGAR#98 `canonical_concept_name` precedent.

`contract::ogar_codebook` mirrors `canonical_concept_id` / `canonical_concept_name` (the lo-u16 concept pull, BBB-safe for membrane consumers woa-rs / medcare-rs / smb-office-rs) but does NOT mirror OGAR#97's `PortSpec::APP_PREFIX` + `render_classid_for` (the hi-u16 render composition: `render_classid = APP << 16 | concept`, OGAR#95 §2). A membrane consumer (BBB-barrier: contract/ontology/callcenter only — `lance_graph_ogar` forbidden) can therefore pull the shared concept but must re-derive the app prefix from the OGAR#95 allocation table by hand. Per Core-First the consumer MUST NOT hard-code `0x000N`. **Fix:** mirror the app-prefix table + a `render_classid` helper into `contract::ogar_codebook` (the `canonical_concept_name` reverse-map mirror, OGAR#98, is the precedent) so the membrane stamps from one source. Interim: the spellbook's Q5 says "stamp from the allocation table." Cross-ref: `.claude/knowledge/ogar-consumer-preflight.md` § "A Core gap this spellbook surfaces"; OGAR#95/#97/#98.

---

## 2026-06-20 — F64-TENANT-VS-F32-ENERGY — perturbation f64 narrows to the F32 `Energy` tenant; a true-f64 tenant is a canon EXTENSION (operator decision)

**Status:** RESOLVED 2026-06-20 (operator) — **NOT F64.** F32 is the fast NaN-hunt tenant (half of f64; NaN test is one integer exponent mask). The compute tenant pivots to **BF16 + AMX** (operator: "use BF16 and add_mul where possible and use amx"); the perturbation/Spain workload is deprioritised in favour of a BF16 4×4-Morton-tile Domino POC. No F64 canon extension. Cross-ref: AGENT_LOG BF16/AMX pivot.

The D1 bridge (`crates/symbiont/src/bridge.rs`) stores each bus's f64 perturbation magnitude in `ValueTenant::Energy` (F32) — "one external f64 → one internal typed tenant," per the operator's architecture. The operator's phrasing was "F64 tenant," but the canon has **no F64 tenant**: `Energy` is F32 (`canonical_node.rs:410`, `VALUE_TENANTS:481`). The f64→f32 narrowing is exact at f32 but lossy vs f64. **Decision needed:** (a) accept F32 `Energy` (the substrate's deliberate accumulator precision; no change), OR (b) extend the canon with a NEW F64 tenant — a value-slab layout addition (RESERVE-DON'T-RECLAIM; bumps `ENVELOPE_LAYOUT_VERSION`; the canon is operator-locked). Not done autonomously. Cross-ref: EPIPHANIES `E-NODE-IS-SOA-IS-KANBAN-BOARD`.

---

## 2026-05-30 — OD-CANONICAL-SPEC-DISAGREEMENT-TIER-SET — `cognitive-risc-core.md` and `wikidata-hhtl-load.md` disagree on the ProvenanceTier value-set; SPEC-OWNER decision, not Claude-session

**Status:** Open · Owner: spec author (NOT a Claude session) · Blocks: D-ARM-1 (ProvenanceTier in `lance-graph-contract`), D-ARM-2 (`Proposer` trait + `CandidateRule`), D-ARM-SYN-1/2/3 (per PR #436 follow-ups).

The four canonical specs at `.claude/specs/` disagree among themselves:
- `cognitive-risc-core.md:58` → tier set `{Curated, Extracted, ArmDiscovered, Ratified}` marked `[stable]`.
- `wikidata-hhtl-load.md:25` → tier set `{Curated, Extracted, Derived}`.
- `faiss-homology-cam-pq.md:14` → "Reasoning layer = separate indexed store, **Derived tier**" — argues `Derived` is a *separate axis*, not a tier value.
- Code today (`crates/lance-graph-ontology/src/odoo_blueprint/mod.rs:450`) → `OdooConfidence::{Curated, Extracted, Conjecture}` — a third value-set, neither matching the core spec nor the wikidata spec.

4-of-4 council reviewers (2026-05-30, recorded in `AGENT_LOG.md` + `post-438-integration-options-v1.md` §4) verdict: do NOT ship `ProvenanceTier` into `lance-graph-contract` until the spec owner reconciles. Two of the four reviewers (R2 + R4) explicitly call this a SPEC FREEZE issue, not a Claude-session decision.

**Council's recommended default if the spec owner wants one to ratify or reject:** keep the core's stable-4 as the on-byte tier; treat `Derived` as a separate orthogonal "reasoning provenance" axis (per faiss-homology + wikidata "orthogonal=beside, not mixed in"); decide `Conjecture`'s fate by either dropping it from code (it's unused per `git grep`) or mapping it to a proposer-local discovery-time label that never crosses the wire.

Cross-ref: `.claude/knowledge/discovery-origin-provenance-reconciliation-v1.md` §2.1 (full conflict matrix), §6 (OD-1/2/3), §8 (specs-on-branch correction).

---

## 2026-05-30 — OD-PROPOSER-ID-WIDTH-CHOICE — 6-bit (64 slots, u8) vs `u16` for `discovery_origin` proposer-id field; SPEC-OWNER lean exists, decision is pending

**Status:** Open · Owner: spec author · Blocks: D-ARM-1, D-MBX-A6-P3 (if `discovery_origin` rides alongside `KanbanMove`).

`cognitive-risc-core.md:62` explicitly says "Widen proposer field (steal reserved → 6 bits/64, or go u16) before surrealkv WAL hardens the LE wire format" — names two alternatives, does not pick. `cognitive-risc-classes.md:64` restates the same problem as freeze-time move N2.

The current `streaming-arm-nars-discovery-v1.md` §7.2 (committed on PR #435 branch, NOT in code) allocates 2 bits = 4 slots and is already full (AstWalker/PairStats/Aerial/Other). #436's PR-note explicitly defers the contract carrier to D-ARM-1.

Council R1 (architectural-fit): u16 (because `class_id`/N1 must ship in the same freeze pass and u16 fits both decisions). Council R3 (integration-coordination): defer this choice until #439 lands (it's mid-flight on `lance-graph-contract`).

**This issue and OD-CANONICAL-SPEC-DISAGREEMENT-TIER-SET are paired** — both touch the same byte grammar; ship them in one council-ratified pass or wait until both are settled.

Cross-ref: `.claude/knowledge/discovery-origin-provenance-reconciliation-v1.md` §6 OD-1; `cognitive-risc-classes.md` §"NON-DEFERRABLE freeze-time moves" N1+N2; PR #439 (open, kanban Phase 2).

---

> **Append-only ledger.** Every issue (bug, regression, invariant
> violation, blocker) gets a dated entry here. Entries move from
> Open → Resolved by status-flip; they are NEVER deleted.
>
> **Format invariant:** every entry starts with `## YYYY-MM-DD — `
> followed by a short title. Body is short — one paragraph of
> problem + cross-references. Full repro / fix / test details go
> in the PR or in a dedicated doc and are LINKED, not duplicated.
>
> **Mutable field:** `**Status:**` line only (Open / Resolved /
> Wontfix / Superseded). Resolved entries keep a `**Resolution:**`
> line pointing at the PR + commit SHA that fixed them.

---

## Double-entry discipline

Every issue has TWO corresponding rows, both in this file:
1. **Open section** — issue captured when first seen.
2. **Resolved section** — same entry, appended when closed, with
   `**Resolution:**` line pointing at fix.

The resolved entry cites the open entry's date as anchor. Old
"Open" entry's **Status:** flips to `Resolved YYYY-MM-DD` — it
stays in the Open section (never moved) so chronology is
preserved. The Resolved section accumulates fixes for discovery.

This is **bookkeeping discipline**, not a storage optimization:
- Open section = what broke and when.
- Resolved section = how and when it was fixed.
- Both sections keep the same row forever; the view depends on
  which section you're reading.

---

## Governance

- **Append-only.** Never delete a row from either section.
- **Mutable:** `**Status:**` and `**Resolution:**` fields only.
- **`permissions.ask` on Edit** (same rule as PR_ARC_INVENTORY).
  Write for appends stays unprompted.
- **Supersedure:** if an issue turns out to be a duplicate of an
  older one, Status → `Superseded by YYYY-MM-DD <title>`; old entry
  stays.

## Cross-references

- `PR_ARC_INVENTORY.md` — which PR shipped the fix.
- `STATUS_BOARD.md` — deliverable-level view (an issue may block
  one or more D-ids).
- `EPIPHANIES.md` — if debugging surfaced an architectural
  insight, that lands in Epiphanies; this file tracks the concrete
  fix.
- `TECH_DEBT.md` — if an issue is knowingly deferred rather than
  fixed, it moves (via cross-ref) into technical debt.

---

## Kanban Format (priority + scope on every entry)

Every issue carries:
- **Priority** — `P0` blocker / `P1` high / `P2` medium / `P3` low.
- **Scope** — which agent / deliverable / domain owns it. One or
  more of: `@<agent-name>`, `D<N>` (plan D-id),
  `domain:<grammar|codec|infra|arigraph|...>`.

Together they form the ticket tag: `[P1 @truth-architect D5 domain:grammar]`.
Agents filter by their own `@`-mention or their domain; nothing
gets buried.

## Open Issues

## 2026-05-30 — [ARM-JIRAK-FLOOR] Aerial+ proposer (D-ARM-13) ships without the mandatory Jirak Stage-A floor

**Status: OPEN.** Surfaced by the 3-savant brutal review of D-ARM-13 (iron-rule-savant #1 finding, brutally-honest-tester P1). The transcoded Aerial+ proposer (`crates/lance-graph-arm-discovery`) gates rule emission only on classical `min_support`/`min_confidence` (`extract.rs` → `rule::CandidateRule::passes`). `I-NOISE-FLOOR-JIRAK` and `streaming-arm-nars-discovery-v1.md` §4 (line 395 "This is not optional") + §11.1 declare the Jirak weak-dependence significance floor **mandatory at Stage A** — but `jirak` exists nowhere in the crate and **D-ARM-7 (the Jirak module) is Queued**. Consequence: with `c = m/(m+k)` saturating as `m = support×n` grows, a thin-but-frequent spurious rule at a 200K window becomes a high-confidence candidate → "substrate calcifies on noise." **Hard prerequisite:** D-ARM-7 MUST land before this proposer is wired into D-ARM-5 (the first stage where `(f,c)` meets a live `SpoStore` + `TruthValue::revision`). Documented honestly in `rule.rs::passes` doc + the synergy doc §4. Resolve by: implement D-ARM-7 and route `extract_rules` emission through `jirak_significance_threshold` BEFORE the classical floor.

## 2026-04-20 — [E-MEMB-1] Python↔Rust slice layouts are incompatible at the 10 kD membrane

**Status:** Open
**Priority:** P1
**Scope:** @integration-lead @truth-architect domain:membrane

PR #210's `role_keys.rs` (Rust) defines disjoint slices of the 10K VSA: Subject [0..2000), Predicate [2000..4000), Object [4000..6000), Modifier [6000..7500), Context [7500..9000), TEKAMOLO [9000..9900), Finnish [9840..9910), tenses [9910..9970), NARS [9970..10000). Python `adarail_mcp/membrane.py::DIMENSION_MAP` uses a different layout entirely: [0..500) "Soul Space" (qualia_16 / stances_16 / verbs_32 / tau_macros / tsv), dim 285 = hot_level, [2000..2018) = qualia_pcs_18. Any vector round-tripped across the two stacks will be reinterpreted by the other side's slice geometry → semantic noise, silent mis-binding.

**Impact:** blocks cross-language reconciliation for the AGI-as-glove surface (Ada σ/τ/q ↔ Rust BindSpace SoA). Until resolved, the Membrane cannot use raw 10K transfer — only serialized σ/τ/q at the REST edge.

**Secondary blocker:** E-MEMB-7 (Ada has its own 3-space incoherence between `membrane.py` 10kD, `rosetta_v2.py` 1024D Jina, and Fingerprint<256> 16K-bit — reconcile internally before Python↔Rust).

**Substrate constraint (added 2026-04-20 per [FORMAL-SCAFFOLD] reclassification):** any bridge between Python-membrane and Rust-role_keys MUST respect E-SUBSTRATE-1. An identity-map between the two layouts would violate bundle associativity — the two layouts encode different algebraic structures over d=10000. The reconciliation doc must EITHER pick one layout as canonical (likely Rust's `role_keys` disjoint slices) and re-express Python's into it, OR define a projector that preserves commutativity of bundle under translation. **A naive bit-by-bit remap is not acceptable** — it would silently break the Markov guarantee that D7 and the rest of the NARS revision stack rely on (see I-SUBSTRATE-MARKOV in CLAUDE.md).

**Next action (when queued):** author a `slice-layout-reconciliation.md` knowledge doc mapping every Python DIMENSION_MAP region to either (a) a Rust role_keys slice, (b) a dropped region, or (c) a new Rust slice to add. The doc MUST include the substrate-respect analysis above. Not yet scheduled.

Cross-ref: `.claude/board/EPIPHANIES.md` 2026-04-20 E-MEMB-1; `.claude/board/EPIPHANIES.md` E-SUBSTRATE-1 + [FORMAL-SCAFFOLD]; Deposit log E-MEMB-7; PR #210 role_keys.rs; `adarail_mcp/membrane.py::DIMENSION_MAP`; CLAUDE.md I-SUBSTRATE-MARKOV.

---

## 2026-05-13 — ndarray:master missing `hpc-extras` feature (latent downstream build break)
**Status:** Open (upstream-blocked)
**Priority:** P2
**Scope:** domain:infra D-NDARRAY-MASTER-HPC-EXTRAS

The `hpc-extras` feature on `ndarray` lives on `AdaWorldAPI/ndarray` branch `claude/burn-A1-dep-gating` (PR #116, **never merged to master**). lance-graph PR #364 (`a3c753f`) declares `features = ["hpc-extras"]` on its `ndarray` path dep — this works for us because the local `/home/user/ndarray` checkout is on the integration branch that carries the feature. **Any consumer that points at `ndarray:master` (post-#142, pre-#116) will hit `feature hpc-extras not found`** — surfaced by MedCare-rs PR #118 (doc-only investigation, merged 2026-05-13). The fix is upstream: `ndarray PR #116 → master`. Outside this session's scope; tracked here so it doesn't get rediscovered.

Cross-ref: MedCare-rs#118, lance-graph PR #364 commit `a3c753f`, ndarray PR #116 (`claude/burn-A1-dep-gating`), ndarray PR #142 (VBMI+Inf clamp, merged but does NOT add hpc-extras to master).

---

## 2026-05-16 — [W-F9-X1] Subagent Edit/Write permission isolation gap — workers must use python3 heredoc fallback

**Status:** Open
**Priority:** P2
**Scope:** domain:infra domain:cca2a @adk-coordinator
**Filed by:** W-F9 (sprint-12 Wave F sweep); originally surfaced per E-META-8

The Claude Code SDK subagent context used in sprint-11 CCA2A workers had `Edit`, `Write`, and `MultiEdit` tools blocked by permission policy. Every worker that needed to write files was forced to use `python3 << 'PYEOF'` heredocs via the Bash tool as a fallback. This pattern works but is awkward, undiscoverable, and error-prone (heredoc quoting rules differ from Edit semantics). Workaround: explicitly instruct workers in their prompt ("Edit/Write blocked — use `python3` heredocs"). Resolution requires either an upstream SDK permission fix or acceptance of the heredoc pattern as the CCA2A standard for write operations in restricted subagent contexts.

Cross-ref: EPIPHANIES.md E-META-8; `.claude/agents/BOOT.md` subagent spawn policy; sprint-11 W-D2/W-F1..W-F9 agent logs.

---

## 2026-05-16 — [W-F9-X2] Stop-hook fires on uncommitted in-flight state during subagent handoff

**Status:** Open
**Priority:** P2
**Scope:** domain:infra domain:cca2a domain:hooks
**Filed by:** W-F9 (sprint-12 Wave F sweep)

When a CCA2A subagent stops mid-task with uncommitted files, the stop-hook fires and may trigger board-hygiene checks or branch guards against a dirty state. Subsequent workers or branch switches then require a stash dance (`git stash` / `git stash pop`) before they can proceed. The workaround is: commit incrementally and stash before any branch switch. A proper resolution would require the stop-hook to detect known-active-worker state (e.g., via a sentinel file or `STATUS_BOARD.md` marker) and tolerate mid-task uncommitted changes without erroring.

Cross-ref: `.claude/hooks/` (stop-hook scripts); `.claude/board/STATUS_BOARD.md`; sprint-11 Wave D multi-step stash dance notes.

---

## 2026-05-16 — [W-F9-X3] Workspace disk quota at 91%+ during cargo builds; ENOSPC risk recurring

**Status:** Open
**Priority:** P1
**Scope:** domain:infra domain:build
**Filed by:** W-F9 (sprint-12 Wave F sweep); first hit during PR #386 rebase cycle

During the sprint-11 PR #386 cycle the workspace hit ENOSPC mid-rebase; 21 GB was freed by running `cargo clean`. The `target/` directory accumulates incrementally built artifacts from multiple workers building different crates in parallel, and the quota ceiling (~91% at the time of the incident) leaves insufficient headroom for rebase + build operations. Risk is recurring: every sprint with heavy parallel cargo work will approach the ceiling. Resolution options: (a) periodic `cargo clean` as a sprint-start hygiene step, (b) smaller per-worker `CARGO_TARGET_DIR` so artifacts don't accumulate in one location, (c) larger disk quota.

Cross-ref: PR #386 (sprint-11); sprint-11 Wave D rebase log.

---

## 2026-05-16 — [W-F9-X4] `cargo check -p lance-graph` may fail locally due to missing `protoc` binary

**Status:** Open
**Priority:** P2
**Scope:** domain:infra domain:build crate:lance-graph
**Filed by:** W-F9 (sprint-12 Wave F sweep)

`lance-encoding` (a transitive dependency of `lance-graph`) requires the `protoc` system binary for its build script. In sprint-11 this binary was absent from the default environment; W-D2 installed it manually. As a result, `cargo check -p lance-graph` (and any other command that pulls `lance-encoding`) will fail with an opaque `protoc not found` error on any worker environment that has not had the binary pre-installed. **CI is the canonical validator**; workers should note that a local compile failure of `lance-graph` may be an environment issue, not a code issue. Resolution: automate `protoc` installation in workspace setup (see TECH_DEBT.md TD-PROTOC-ENV-SETUP-1).

Cross-ref: TECH_DEBT.md TD-PROTOC-ENV-SETUP-1; D-CSV-6a agent log (W-D2 manual install); sprint-11 Wave D build notes.

---

## 2026-05-16 — [W-F9-X5] Background-worker file collisions during main-thread rebase require multi-step stash dance

**Status:** Open
**Priority:** P2
**Scope:** domain:infra domain:cca2a
**Filed by:** W-F9 (sprint-12 Wave F sweep)

During sprint-11 Wave D, a background worker had modified workspace files while the main thread needed to rebase onto updated `main`. The conflict required a multi-step stash dance: stash local changes → rebase → pop stash → resolve conflicts → continue. The pattern works but is fragile: if the stash contains large or structurally complex diffs the pop may produce confusing three-way conflicts. Proper resolution would coordinate worker commits with main-thread rebase windows (e.g., all workers commit before any rebase is initiated), or use per-worker branches that are rebased independently.

Cross-ref: Sprint-11 Wave D / sprint-12 Wave D rebase log; TECH_DEBT.md TD-PROTOC-ENV-SETUP-1 (related infra gap); `.claude/agents/BOOT.md` handover protocol.

(No other tracked open issues. New issues PREPEND here
in reverse chronological order. Format below.)

```
## YYYY-MM-DD — <short title>
**Status:** Open
**Priority:** P0 | P1 | P2 | P3
**Scope:** @<agent> D<N> domain:<tag>

<one paragraph: what's broken, where it surfaces, rough impact>

Cross-ref: <file:line or PR # or knowledge doc>
```

---

## Resolved Issues

(No resolved issues at initial commit. When an Open issue is fixed,
APPEND a copy here with the same date anchor + `**Resolution:**`
line. Old Open entry's Status flips to `Resolved YYYY-MM-DD`. Old
entry stays in the Open section for chronology.)

```
## YYYY-MM-DD — <same title as Open entry>
**Status:** Resolved YYYY-MM-DD
**Resolution:** PR #NNN (commit SHA) — <one-line description>

<original problem paragraph, verbatim>

Cross-ref: <same as Open entry>
```

---

## How to use this file

**When an issue is found** — prepend to **Open Issues** section with
today's date + `**Status:** Open` + one-paragraph description.

**When an issue is fixed** — append to **Resolved Issues** section
with the same title and date anchor + `**Status:** Resolved
YYYY-MM-DD` + `**Resolution:** PR #NNN`. Don't edit the Open entry
body; just flip its Status to `Resolved YYYY-MM-DD`.

**When an issue is a duplicate** — append a new entry in Resolved
section noting `**Resolution:** duplicate of YYYY-MM-DD <title>`;
flip Open entry to Superseded.

**When an issue is deferred knowingly** — leave it Open here but
also append a row to `TECH_DEBT.md` with cross-ref back.

## ISS-CLASSID-OGAR-DRIFT — 2026-06-20 (cont.) — RESOLVING (operator signed off; landed)
**Status:** RESOLVING — operator greenlit the realign (`AskUserQuestion`: "Realign to 0xDDCC", "Wire-compat now", "FMA = Health 0x09XX"). Landed D-OVC-1/2/4 on the jirak branch: `CLASSID_OSINT 0x0007 → 0x0700` (OSINT domain root), `CLASSID_FMA 0x0008 → 0x0901` (anatomy concept in Health, `0x0900` = Health root); minted `CLASSID_PROJECT = 0x0100` + `CLASSID_ERP = 0x0200` with `ReadMode::{PROJECT,ERP}` registered; NEW `contract::ogar_codebook` (wire-compat mirror, zero-dep — `ConceptDomain` / `canonical_concept_domain` / `classid_concept_domain` / `source_domain_concept` / `CODEBOOK` / `canonical_concept_id` / `LabelDTO::from_canonical`); `soa_graph::{PROJECT,ERP}` DomainSpecs. Drift guard test pins the shared `0xDDCC` ids; contract 710 default / 716 v2 green, clippy clean. **Dependency direction = (b) wire-compat (no OGAR↔contract dep);** the `u16` LE wire is the only contract. D-OVC-3 (cutover/version-gate audit of the *value* realign per `I-LEGACY-API-FEATURE-GATED`) remains; the classids are layout-preserving (a const value change, not a bit-layout reclaim), so no `ENVELOPE_LAYOUT_VERSION` bump. Closes when the PR merges.

## ISS-CLASSID-OGAR-DRIFT — 2026-06-20 — OPEN (needs operator sign-off)
**What:** merged `lance-graph-contract` classids drifted from OGAR `ogar-vocab`'s domain-encoded codebook (`0xDDCC`, `crates/ogar-vocab/src/lib.rs:1073` CODEBOOK + `:1163` `canonical_concept_domain`). `CLASSID_OSINT=0x0007` → `0x00` = OGAR *Reserved* domain (OSINT is `0x07XX`); `CLASSID_FMA=0x0008` → OGAR *OCR* block (FMA/anatomy is clinical → Health `0x09XX`). OGAR's own note (`lib.rs:1204-1212`): codebook id == `NodeGuid.classid` low u16, and `LabelDTO` "long-term belongs in lance-graph-contract." So contract + OGAR currently disagree on what `0x07`/`0x08` mean.
**Impact:** the contract↔OGAR↔q2 triangle has an inconsistent classid space; `canonical_concept_domain(id>>8)` mis-routes contract's OSINT/FMA; project/ERP un-minted.
**Fix (proposed):** `.claude/plans/ogar-vocab-contract-codebook-migration-v1.md` D-OVC-1..4 — host the codebook/`ConceptDomain`/`LabelDTO` in contract, classids follow `0xDDCC` (mint project `0x01XX`+ERP `0x02XX`; realign OSINT→`0x0700`, FMA→Health `0x09XX`). **Realigning merged OSINT/FMA rewrites canon (#557/#560 + CLAUDE.md canon block) → operator sign-off required** (plan §5). Origin: `CLASSID_OSINT=0x0007` minted from the early "OSINT is 0x0007" guess before ogar-vocab's `0xDDCC` layout was consulted.
