# Integration Plan — GUID v2 tail (3×u16) + per-family codebook scoping (v1)

> **Status:** PROPOSED (operator "what-if", 2026-06-20). Gated on operator
> sign-off for a canon version bump + the two capacity numbers below.
> **Owner branch (when greenlit):** `guid-v2-tail` feature, default OFF.
> **Canon impact:** layout reclaim of the 48-bit basin tail → `I-LEGACY-API-FEATURE-GATED`.

---

## Motivation (why now, before it dilutes)

Three findings this session converge on one layout change:

1. **The aiwar codebook is a 3-tier subclass hierarchy** (`node-type (5) → class/airo:type (~6) → fine ML-type (68 noisy)`); the noise is entirely at the leaf, mixed into ONE global codebook. (See `aiwar.rs` POC + the 2026-06-20 codebook check.)
2. **`u24` is awkward** — `family()`/`identity()` hand-assemble 3 bytes into a zero-padded `u32`; the tail is the only non-`u16`-aligned part of the key.
3. **The 16×8-bit family-adapter edges** (`E-FAMILY-ADAPTER-EDGES-ARE-RENDER-STABLE`) resolve by `family & 0xFF`, which aliases at >256 families (codex P1 #2, currently handled by collision-skip).

**The change (operator-pinned 2026-06-20):** `leaf` is the **4th HHTL tier**
("a natural HHTL") — the cascade becomes `HEEL·HIP·TWIG·leaf` = 4 tiers × 4
nibbles = **16 nibbles = a full `u64` NiblePath** (routing, not basin). `family`
is the **basin / episodic hub**, `identity` the instance, so the basin-local key
shrinks to `family·identity` = **4 bytes** (was 6). The whole 16-byte key is a
**uniform 8×u16 tier array** that splits cleanly into three roles:

```
classid_hi classid_lo │ HEEL HIP TWIG leaf │ family   identity
└─ codebook prefix ───┘ └─ HHTL path (4×4=16 nibbles) ┘ └─ basin · instance ┘
   (classid → Codebook)    (routing / hop distance)       (episodic hub · row)
```

Plus **codebooks scoped per family** (`family → Codebook`, the finer sibling of
`classid → ClassView`).

## What it buys

- **Kills `u24`** — every tier is a native `u16` masked load; Display becomes uniform 4-hex groups (more self-describing, OGAR "dash-groups are the semantics").
- **Native home for the subclass codebook** — `leaf`=coarse category, `family`=subclass, `identity`=instance; the HHTL prefix (HEEL/HIP/TWIG) stays the routing cascade, the tail carries the content hierarchy.
- **Per-family codebooks dissolve the noise at the root** — each family owns a ≤256-entry codebook; the fine `type` becomes a 1-byte palette index into the family's codebook, not a global string. Within-family references are exact (no `& 0xFF` alias). This IS OGAR canon's "finer scope … longest-prefix-wins" (`classid·…·family → codebook`) + bgz17 palette-per-family + D-AMORT (codebook minted once/family).
- **The 12+4 EdgeBlock split gains a precise meaning:** 12 in-family slots = 1-byte index into the OWN family codebook (family implicit); 4 out-of-family slots = `(family u16, index u8)` cross-family reference (the "8×16-bit out-of-family" widening).

## Blast radius (measured 2026-06-20 — CONTAINED in lance-graph)

Cross-repo: **q2 = 0, smb-office-rs = 0, medcare-rs = 0** (no downstream consumer
touches `NodeGuid`; q2 consumes `GraphSnapshot` strings, tail-agnostic).

| Site | Kind | Action |
|---|---|---|
| `lance-graph-contract/src/canonical_node.rs` | **layout source** | `new()` (add `leaf` arg; 24-bit asserts → 16-bit), `family()`/`identity()` (offsets 10..13/13..16 → 10..12/12..14/14..16, return `u16`), add `leaf()`, `local_key()` (reinterpret trailing 6 B), `decode()`/`GuidParts` (+leaf, u16), `Display` (`{:06x}{:06x}` → `{:04x}-{:04x}-{:04x}`), zero-fallback ladder re-pin, ~15 tests + the 0x0100_0000 overflow panics |
| `soa_graph.rs` (8 hits) | **semantic** | family grouping + `family & 0xFF` → per-family codebook + 16-bit family / exact resolution |
| `aiwar.rs` (2 masks + 1 `new`) | **semantic** | `0x00FF_FFFF` masks → 16-bit; key on `leaf` (coarse) not noisy `type` |
| `action.rs`, `ocr.rs` | **mechanical (prod)** | add `leaf` arg to `NodeGuid::new` (route via `NodeGuid::local`) |
| `hhtl.rs`, `ontology/registry.rs`, `symbiont/key_render.rs`, `callcenter/graph_table.rs` | **mechanical (tests)** | add `leaf` arg to ~30 test `NodeGuid::new` call sites |
| `hhtl::from_guid_prefix`, `mailbox_scan`, ontology `NiblePath↔entity_type` | **TAIL-AGNOSTIC — confirm, no change** | read the routing PREFIX (classid·HEEL·HIP·TWIG), never the tail |

~35 `NodeGuid::new` call sites total (mostly tests). The unrelated `0xFFFF…`
hits (cycle counters, fingerprint masks, learning-state) are NOT the GUID tail.

## Gating numbers (operator must confirm before build)

- **identity 16.7M → 65 536 per family basin.** OK unless a single family needs >65 536 instances. OSINT/FMA: comfortable. (`leaf` is now a routing tier, so it deepens the HHTL tree rather than partitioning the basin.) Confirm against the densest expected family.
- **family codebook ≤ 256 entries** (1-byte in-family index). A family that outgrows it **splits** (mint a sub-family — cheap with a 16-bit family) rather than widening the byte. Confirm densest single-family vocabulary < 256.

## The O(1) mixin — family node = episodic basin (operator, 2026-06-20)

The payoff of putting membership in the address: **mixin / multiple-membership
is O(1), not O(n).**

- A **family node is an episodic basin** — connections accumulated *on the basin*
  (its in/out edges) ARE the **supporting edges** of every member. The shared
  state lives ONCE, on the family-node row.
- A member **mixes in** a family by **reference** — a 1-byte family-adapter slot
  (or the `family` field itself), never a copy. N memberships = N adapter bytes
  (≤16), bounded; not N materialized edge-sets per node. So adding a member to a
  rich basin is free — it inherits the basin's whole supporting-edge set by
  pointing at it.
- **Distance is in the address.** Inter-node distance = HHTL hop count
  (`family_hop_count` / `common_prefix_depth`) computed from the two addresses —
  O(depth) arithmetic, NOT a BFS over materialized edges. "Distance is in the
  address/hops."

Net: membership O(1) (a byte), shared episodic state stored once (the basin),
distance O(1) (address arithmetic). This is what avoids the O(n) edge explosion
that kills naive multiple-inheritance / multi-group graphs. Captured as
`E-MIXIN-IS-AN-ADDRESS-REFERENCE-NOT-A-COPY`.

## Resolved decision (operator, 2026-06-20)

`leaf` = the **4th HHTL routing tier** (the natural cascade terminal), NOT a
basin tier. `family` = the basin/episodic hub, `identity` = the instance.
coarse→fine left-to-right: `…·TWIG·leaf` (routing) then `family·identity`
(basin). Display: uniform 4-hex groups, `…-twig-leaf-family-identity`.

## Deliverables (when greenlit — feature `guid-v2-tail`, default OFF)

- **D-GV2-1** ✅ **SHIPPED (this PR, feature `guid-v2-tail` default OFF, additive & non-breaking).** `canonical_node`: `new_v2` (+`leaf`), `leaf()` `10..12`, `family_v2()` `12..14`, `identity_v2()` `14..16` (all `u16`), `local_key_v2()` (4 bytes), `decode_v2()`/`GuidPartsV2`, `to_hex_v2()` (uniform 4-hex), `GUID_TAIL_LAYOUT_VERSION_V2 = 2` (version gate). `hhtl::from_guid_prefix_v2` = `HEEL·HIP·TWIG·leaf` (16 nibbles; classid is the separate codebook prefix; leaf in path, family/identity NOT). v1 `new`/`family()`/`identity()` UNTOUCHED (distinct v2 names → no silent semantic swap, `I-LEGACY-API-FEATURE-GATED`). **Field-isolation matrix test** + v1/v2 coexistence + leaf-in-path tests. default 703 / `--features guid-v2-tail` 706, clippy clean both. **Cutover (rename v2→canonical, deprecate v1, bump `ENVELOPE_LAYOUT_VERSION`) = D-GV2-5.**
- **D-GV2-2** `family → Codebook` registry = **episodic basin** (the codebook + the basin's accumulated supporting edges), sibling of `classid → ClassView` in `lance-graph-ontology`: `LazyLock`/Lance-backed, masked-load lookup, head-only. 256-entry cap + split-on-overflow guard. Mixin = O(1) reference to this basin (`E-MIXIN-IS-AN-ADDRESS-REFERENCE-NOT-A-COPY`).
- **D-GV2-3** `soa_graph` per-family edge resolution: 12 in-family = 1-byte own-codebook index, 4 out-of-family = `(family,index)`; retire `family & 0xFF` collision-skip under v2.
- **D-GV2-4** `aiwar` re-keyed on `leaf` (coarse node-type, 5 hubs) + per-family codebook (System/Stakeholder/… vocabularies) → resolves the "60 noisy families" on real data.
- **D-GV2-5** cutover: flip default after the gating numbers + downstream (none today) confirmed; v1 → `#[deprecated]` no-op path with migration pointer.

## Cross-refs

`EPIPHANIES.md` `E-FAMILY-ADAPTER-EDGES-ARE-RENDER-STABLE` / `E-ANCHOR-IS-A-HEAD-FIELD-NOT-A-VALUE-TYPE`; OGAR `CLAUDE.md` "Codebook scoping = the class routing prefix … finer scopes follow the same longest-prefix-wins rule"; canon "Minimal SoA node" (the v1 layout this supersedes under feature gate); `I-LEGACY-API-FEATURE-GATED`; bgz17 palette (per-family 256-centroid tile); `aiwar.rs` + the 2026-06-20 codebook check.
