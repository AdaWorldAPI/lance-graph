# SPEC v1 — `NodeGuid::new` repurpose audit (5+3 council, READ-ONLY)

**Status:** SPEC v1, Phase 0. **Nothing is implemented. Nothing may be changed.**
The deliverable of this council is an AUDIT — a verdict on whether
`NodeGuid::new` can be repurposed, at what cost, and what would break. No
code, no bake, no schema edit lands from this pass.

**Operator constraint, verbatim (2026-09-06):** *"It's already in production,
if you touch it it would change something you half understand."* The audit
exists BECAUSE the understanding is incomplete. A savant that proposes an
edit has misread this line.

---

## 0. RETRACTED READINGS — do NOT rediscover these as findings

The orchestrator produced four wrong readings before this spec. They are
recorded so the council does not spend a lens re-deriving them, and so no
finding cites them as prior context. **Each is FALSE:**

| # | wrong reading | corrected by |
|---|---|---|
| R1 | "the 12+4 is `in_family`/`out_family`, and the edge block content is a degree histogram, therefore the 12+4 the operator described is V1 naming" | operator: the 12+4 is about **value** bytes, not the edge block. Two different regions. |
| R2 | "no path exists in any row; a DN cannot be derived from the bake" | measured false — see §2 M4/M5 |
| R3 | "edge targets live in side tables (`ro_edges`); consumers walk `ZIPPER_ISA_DEPTH`" | targets are in the value slab's edge lanes; operator: **"Nobody walks anything"** |
| R4 | "the key is `classid(4) + 4×u24`, i.e. the quad IS the key" | operator: **"4x24bit identity is in the value tenant of the SoA"** — the quad is a VALUE tenant, not the key |

**Method note for every savant:** R1–R4 were each produced by reading
accessor NAMES or byte-occupancy STATISTICS instead of the class's declared
schema. `ogar-obo/src/layout.rs` exists to prevent exactly this and says so:
*"the classid picks the reading … a reader never assumes a carve."* Any
finding in this council that rests on a field name or a histogram, rather
than on a declared schema or an operator ruling, is malformed.

---

## 1. FROZEN DECISIONS (cite-only; flag as VIOLATES with evidence, never re-open)

1. **F1 — V1 is banned for new mints.** lance-graph `CLAUDE.md` § CANON:
   *"forbidden for new units"*; MedCare `Cargo.toml:139` records the operator
   ruling *"V1/V2 are completely deprecated — ANY default is Always V3, period."*
2. **F2 — the stride is canon.** `NODE_ROW_STRIDE = 512`, `key(16) | edges(16)
   | value(480)`. Every tail variant is a READING of the same bytes;
   `TailVariant::is_layout_preserving` returns `true` unconditionally
   (`canonical_node.rs:1290+`). No `ENVELOPE_LAYOUT_VERSION` bump is available
   as a lever, and none is needed.
3. **F3 — RESERVE, DON'T RECLAIM.** A zero tier means *not consulted*, never
   *compacted away*.
4. **F4 — the classid picks the reading.** `ogar-obo/src/layout.rs`
   `row_schema_of(concept)`; `ClassView::edge_codec_flavor` for the edge block.
5. **F5 — operator architecture, stated 2026-09-06, FROZEN:**
   - **the first two tenants are the distinguished name**
   - **the 4×24-bit Quad Identität is in a VALUE tenant of the SoA**
   - **lookup-wise the O(1) is in the quad 4×24 bit**
   - the depth-16 path (12 + 4 second tenant) is used for **parent–child
     inheritance and HHTL nodes**
   - **"Nobody walks anything."**
6. **F6 — the V3 generation marker is `0x1000` in the LOW u16**, canon
   `domain:appid` in the HIGH u16 (`canonical_node.rs:94-118`:
   `CLASSID_OSINT_V3 = 0x0701_1000`, `CLASSID_FMA_V3 = 0x0A01_1000`,
   `CLASSID_CPIC_V3 = 0x0E01_1000`).
7. **F7 — the lo-u16 app prefixes `0x0005` / `0x0009` are pre-flip artifacts**
   (operator, this session). They are NOT evidence of a live allocation table.
8. **F8 — do not touch bodyhelix / adjacent older bakes** (operator): they work,
   re-baking is out of scope, and they are the one place V1 may legitimately remain.

---

## 2. INPUT INVENTORY (file:line — all verified this session)

### 2.1 The subject

| item | location | shape |
|---|---|---|
| `NodeGuid::new` | `lance-graph-contract/src/canonical_node.rs:200-224` | `(classid u32, heel u16, hip u16, twig u16, family u32, identity u32)`; asserts family/identity ≤ 24 bits; writes `c[0..4] h[4..6] p[6..8] t[8..10] f[10..13] i[13..16]` |
| `NodeGuid::local` | `canonical_node.rs:228` | default-class bootstrap: identity only |
| `mint_for` | `canonical_node.rs:368` | dispatches on `TailVariant`; **V2 and V3 share one arm** → `new_v2`, asserting family/identity ≤ `0xFFFF` |
| `TailVariant` | `canonical_node.rs:1275` | V1 `family(u24)·identity(u24)` / V2 `leaf·family·identity 3×u16` / V3 = *"the `(part_of:is_a)` 8:8 tile"* |
| accessors whose MEANING a repurpose changes | `canonical_node.rs:240,245,250,256,262,268,305,513,521,527,534` | `classid family identity heel hip twig local_key leaf family_v2 identity_v2 local_key_v2` |
| `CascadeShape` | `lance-graph-contract/src/facet.rs:395-409` | `G6D2` (6×2, shift `i>>1`) / `G4D3` (4×3, divides `i/3`, `is_byte_aligned=false`) / `G3D4` (3×4, shift `i>>2`); `ROTATIONS` = all three; **G·D = 12 each** |
| feature gates | `lance-graph-contract/Cargo.toml:42,54,69` | `default = ["guid-v3-tail"]`; `guid-v3-tail = ["guid-v2-tail"]` |

### 2.2 The producer's declared carve (authoritative)

`ogar-obo/src/layout.rs` `OBO_CORE_ROW`, with `lib.rs` `EDGES_OFFSET=16`,
`VALUE_OFFSET=32`, `NODE_ROW_STRIDE=512`, `edges.rs` `LANE_BYTES=16`,
`LINKS_PER_LANE=4`, `EDGE_LANE_COUNT=23`, `EDGE_SLOT_COUNT=92`:

```
key    [0..16]     classid(4) + 12
edges  [16..32]    row[EDGES_OFFSET + Predicate as usize] = per-predicate DEGREE
value  [32..512]
   entity_type   slab  96 → abs 128, u16
   edge_lanes    slab 112 → abs 144, 23 lanes × 16 B, runs to end of row
                 each lane = classid(4) + 4×u24     ("under the G2 grace carving")
```

### 2.3 Measurements taken this session (reproducible)

- **M1** MedCare mint ratio: **44** `NodeGuid::new` call sites vs **1** real
  `NodeGuid::mint_for(` call site (`medcare-cohorts/src/differential.rs:944`).
- **M2** `identity()` readers off a key: MedCare **66**, lance-graph **5**,
  tesseract-rs **0**, stockfish-rs **0**. `family()`: 2 / 8 / 0 / 0.
- **M3** stockfish-rs does NOT enable `guid-v3-tail` → its one mint is forced V1.
- **M4** `all-lanes-domain-v0.1.0/all-lanes.soa` (S3, 762,041 rows × 512 B):
  key facet bytes 4..9 populated with a descending profile; bytes 10,11 always
  zero; 13,14,15 dense.
- **M5** value-slab depth probe over 24,000 sampled rows: tenant-1 (abs 76..88)
  contiguous-prefix depth histogram `1→10 … 4→3578 … 12→28`; **when tenant-1 is
  full at 12, tenant-2 depth is 4 in 27 of 28 rows** (one at 5); max total 17.
- **M6** classid generations coexist in one bake: `0x9101 0x9202 0x970d 0x9811`
  (new 91..9E) alongside `0x0315 0x0304 0x0309` (retired `0x03`), plus `0x9303`
  which `ogar-vocab` asserts must stay a hole. ~8,996 of 18,000 sampled rows
  were on `0x03`.
- **M7** MedCare app prefixes: `MEDCARE_APP_PREFIX = 0x0005`
  (`class_registry.rs:314`, pinned to `HealthcarePort::APP_PREFIX` by a
  conformance test) vs hardcoded `APP_PREFIX = 0x0009`
  (`medcare-server/src/views/addressed.rs:156`).
- **M8** patient addressing: `patient_guid` (`medcare-soa/src/patient.rs:189`)
  mints `CLASSID_DEFAULT` (0) with family/identity split of a 48-bit id;
  `bake_node_rows` (`medcare-cohorts/src/node_bake.rs:50-73`) mints the DISEASE
  classid with `family=0`, `identity=demo_patient_id`, `edges: EdgeBlock::default()`.
- **M9** cohort size: **750** (15×50), ids `1..=750`,
  `DEMO_PATIENT_ID_BASE = 9_100_000`, reserved block `9_000_000..=9_999_999`
  (fits one u24). MedCare `CLAUDE.md` says "1,200" — drift.
- **M10** `patient` is ALREADY minted `0x0901` in `ogar-vocab` with
  `class_ids::PATIENT` and `HealthcarePort::class_id("Patient")`.

---

## 3. THE PROPOSED RESOLUTION (what the council audits — NOT what it builds)

**Claim under audit (C):** `NodeGuid::new` can be retired as a *V1 mint* and
repurposed as *the* V3 cascade constructor, without a stride change and
without an `ENVELOPE_LAYOUT_VERSION` bump, because:

- C1 — the key's 12 post-classid bytes are the same 12 bytes every
  `CascadeShape` carves (`G·D = 12`), so the shape is a READING (F2).
- C2 — V1's `family(u24)·identity(u24)` occupies bytes 10..13 and 13..16, so
  under a 2-byte rail grid rail 4 = `(byte12, byte13)` **straddles the two u24
  fields**; a flat u24 cannot carry a rail.
- C3 — per F5 the identity is NOT in the key: the quad is a value tenant and
  the first two tenants are the DN. So a repurposed key carries cascade/DN
  addressing, and `identity()`-off-the-key becomes meaningless.
- C4 — the cost is therefore concentrated in M2's readers (66 + 5) and M1's
  mint sites (44), not in the layout.

**The audit must return a verdict on C, not implement it.**

---

## 4. NON-GOALS

- **N1** Any code, schema, bake or Cargo change. Read-only pass (operator).
- **N2** Re-litigating F1–F8. Frozen; verify compliance only.
- **N3** Designing the patient classid mint. Separate, operator-gated.
- **N4** bodyhelix / older bakes (F8).
- **N5** The `0x03` vs `91..9E` generation mix (M6) — surfaced, not resolved here.
- **N6** Proposing a new tenant, shape, or accessor. A savant wanting to
  redesign files ONE `RISK` finding and stops (harness rule).

---

## 5. PRE-REGISTERED GATES (for the AUDIT artifact, not for code)

- **G1** Every claim in ratified v3 carries `file:line` or a named measurement
  (M1–M10), or is labelled CONJECTURE.
- **G2** No claim contradicts F1–F8 without an explicit VIOLATES finding.
- **G3** The verdict on C is one of: `REPURPOSABLE-AS-SPECIFIED` /
  `REPURPOSABLE-WITH-AMENDMENTS` (each amendment named) / `NOT-REPURPOSABLE`
  (with the blocking mechanism named at file:line).
- **G4** The M2 reader census is classified into
  *ordering / uniqueness / display / **sequence*** — only the fourth is a
  semantic blocker (a DN sorts hierarchically, not temporally, and that
  failure still compiles).
- **G5** Zero files modified by this pass. `git status` clean but for the
  audit document itself.

---

## 6. PER-SAVANT QUESTION SETS

### S1 — prior art (`prior-art-savant`)
1. Has a `NodeGuid::new` repurpose/retirement already been proposed, ruled on,
   or partially landed anywhere in `.claude/board/`, `.claude/plans/`, or
   `EPIPHANIES.md`? Cite the E-id / plan row.
2. Does `ISS-V1-TAIL-RESIDUE` or any open issue already own this? Its scope?
3. Is there prior art for a class whose key carries cascade while identity
   lives in a value tenant (F5's shape)? Where?
4. Any DUPLICATE of the M1 44:1 finding already recorded?

### S2 — iron rules (`iron-rule-savant`)
1. Does claim C violate `I-LEGACY-API-FEATURE-GATED`? Specifically: do the
   V1 accessors (`family`/`identity`/`local_key`) become same-name-different-
   semantics under a repurpose — the exact 5-instance Sprint-11 anti-pattern?
2. Does C violate F2 (layout-preserving) at any point? YES/NO + evidence.
3. Does C violate `RESERVE, DON'T RECLAIM` (F3)?
4. Does the V2/V3 shared mint arm (`canonical_node.rs:368+`, u16 asserts)
   contradict F1's "always V3"? VIOLATES or CONFIRMS.

### S3 — code truth (runtime-archaeologist charter, via `general-purpose`)
1. Verify EVERY file:line in §2. Report each as CODED / CLAIMED / ABSENT.
2. Is `NodeGuid::new` reachable from any PRODUCTION path in lance-graph
   itself (not tests, not MedCare)? Cite call sites.
3. Are the 66 M2 readers real reads of `key.identity()`, or do some read a
   different `identity()`? Give the true count.
4. Does anything in-tree already consume `CascadeShape::G4D3`? file:line.
5. Is `ZIPPER_ISA_DEPTH` / `inherited_part_of` reachable from a production
   route, or dormant? (F5 says nobody walks — verify the CODE agrees.)

### S4 — cascade impact (`cascade-impact-savant`)
1. Enumerate every file/test/doc/board row that MUST change if C lands.
   Split mandatory-pre-merge vs informational-follow-up.
2. Which of the M2 readers fall in each G4 bucket (ordering / uniqueness /
   display / sequence)? Name the sequence ones at file:line — they are the
   blocking set.
3. Which committed golden fixtures or `.soa` artifacts would be invalidated?
4. Does any CROSS-REPO consumer (tesseract-rs, stockfish-rs, lance-graph-java,
   a2ui-rs, OGAR) read a `NodeGuid` field that C changes?

### S5 — different views (`creative-explorer-savant`)
1. What is the strongest reading under which C is the WRONG move — i.e. what
   does V1's `family`/`identity` split buy that a cascade reading loses?
2. Is there a second-order consequence of C that the spec has not named?
3. Given F5, is "repurpose `NodeGuid::new`" even the right unit of change, or
   is the real unit something else? (Name it. Do NOT design it.)
4. What would make this audit's verdict WRONG in six months?

---

## 7. Council roster (8)

**Phase 1 — the 5 (parallel, Sonnet, read-only):** S1 `prior-art-savant`,
S2 `iron-rule-savant`, S3 `general-purpose` (runtime-archaeologist charter),
S4 `cascade-impact-savant`, S5 `creative-explorer-savant`.

**Phase 3 — the 3 (parallel, on draft v2 ONLY):** `overclaim-auditor`,
`dilution-collapse-sentinel`, `firewall-warden`.

Savants are READ-ONLY and write no repo file. The orchestrator is the sole
writer. No savant runs `cargo` (shared `target/`, workspace rule).
