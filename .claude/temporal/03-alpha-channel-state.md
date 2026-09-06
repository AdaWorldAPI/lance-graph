# 03 — the alpha channel: what it is, what it costs, what breaks

## What it is

An **ephemeral overlay** at the SAME addresses as the baked SoA spine, with
rows existing only where attention landed. Operator's own picture (alpha plan
§0): a Photoshop alpha channel carrying the residue of a search, so the graph
is not contaminated by patients; eye-tracking of the ontology thoughts,
recording WHERE the eye looked, not what it saw.

Not a budget, not a weight, not a mask of importance. `alpha.rs:12-19`:
*allocate* = the address space, every address the base spine already has, at
**zero rows**; *claim* = materialise ONE row at ONE address.

## The payload — 16 bytes in value slot 0

```
 0..4   cycle  (u32)  the thinking cycle the claim belongs to
 4..8   seq    (u32)  claim order = the saccade's position
 8..9   rung   (u8)   which rung of attention landed
 9..11  visits (u16)  how often attention returned (1 on first claim)
11..16  reserved      zeroed, never reclaimed
```

`ALPHA_STAMP_OFFSET = 0`, `ALPHA_STAMP_BYTES = 16`.

`visits` is the eye-tracking **regression** counter, and the module argues for
it: in reading research a regression is the single most diagnostic event, it
says the reader did not integrate the first time. A revisit never rewrites the
first visit's `seq`/`rung`/`cycle`; only the counter moves (saturating).

## The cost — this is the reason a delta was considered

`claim()` allocates a full canonical `NodeRow`: key(16) + edges(16) +
value(480) = **512 bytes**, of which 16 carry the stamp, 16 are a
deliberately-zeroed `EdgeBlock`, and **464 are zeros**. Plus a `HashMap` entry.

## The population half is ALREADY mask-native

`AlphaMask` is a word bitset over base ordinals — `Box<[u64]>` + `len: u32`,
`ordinal/64` word index, `1 << (ordinal % 64)` bit — with and/or/xor/and_not,
`not()` clearing phantom tail bits, and the module's own law that **no unnamed
materializer exists**: the only way ordinals leave mask form is the named
`materialize_ordinals()`, O(n).

So only the PAYLOAD materializes. The ordinal is the base-slice POSITION,
cached lazily in `OnceLock<HashMap<AlphaAddr, u32>>`, deliberately not claim
order (pinned by `the_ordinal_is_the_base_position_not_a_claim_order`).

## Who actually needs a `NodeRow`

`AlphaOverlay::rows()` has three real consumers:

| site | what it does |
|---|---|
| `alpha_tunnel.rs:185` (`merge()`) | only extracts `(key, stamp)` via `stamp_of` |
| MedCare `medcare-nodesoa/src/alpha.rs:27` | `node_rows_to_batch(overlay.rows(), cycle)` |
| MedCare `medcare-nodesoa/src/alpha.rs:82` | `write_node_soa_dataset(path, overlay.rows(), cycle)` |

Every other `.rows()` hit in either repo is a different type (facet, zerocopy,
gotham, graph_feed).

And `merge()` already RETURNS `Vec<(AlphaAddr, AlphaStamp)>` — the row form is
purely internal.

Every caller of `get()` does `.is_some()` / `.is_none()` or immediately
`stamp_of(...)`, with ONE exception: a MedCare test
(`the_overlay_row_carries_the_stamp_and_the_base_row_does_not`) that reads
`.value`/`.edges` directly. So `get` can safely become `Option<AlphaStamp>`.

## The MedCare storage half — and it is test-only

`medcare-nodesoa/src/alpha.rs` (415 lines) keeps 4 pub fns after the migration:
`overlay_to_batch`, `key_bytes_at`, and feature-gated `write_alpha_overlay` /
`read_alpha_overlay`. The schema is `lib.rs:46-53`: **ONE column, `"node"`,
`DataType::FixedSizeBinary(512)`, non-nullable** — the whole `NodeRow` via
`NodeRowPacket::as_le_bytes()` chunked directly, never split into fields.
Write path: `write_node_soa_dataset` -> `Dataset::write(..., WriteParams{mode:
Create, ..default})`; **no stable-row-id config anywhere**; every caller uses a
scratch/tempdir path, not a durable location.

**All four have ZERO consumers outside alpha.rs's own tests.** So nothing in
production depends on the 512-byte form. `AlphaOverlay` itself is consumed
directly from `lance_graph_contract` by `first-thought/{backreference,attention}.rs`,
bypassing medcare-nodesoa as intended.

## Hard constraints any redesign must satisfy

1. **`AlphaTunnel::merge()` requires `(rung, seq)`-monotonic traversal with NO
   sort.** It carries a `debug_assert` and a dead-sort postmortem in source: a
   `sort_by_key((rung, seq))` stood there and was *provably dead* (the disable
   run stayed green because it cannot reorder anything), so it was replaced by
   the assert. The traversal IS the order: lanes ascend by index (= rung), and
   `rows()` yields per-lane claims in `seq` order. **Claim-order storage
   satisfies this; ordinal-order storage does NOT.**
2. The parallel-equals-sequential byte-identity falsifier
   (`parallel_und_sequenziell_sind_byte_identisch`) is a hard gate.
3. `rung_horizon::claim_admitted` requires refusal to be a **true no-op** —
   the gate precedes the write, never filters after, so a Strict lane's
   scanpath carries no trace of the future, not even a `visits` bump.
4. `merged_rows()` must keep round-tripping the 512-byte V3-canon stride.
5. `data-flow.md`'s "No `&mut self` during computation. Ever." **conflicts with
   every current alpha call site** (`&mut AlphaOverlay`, mutated live). A
   standing conflict between a ruled invariant and shipped code — an operator
   decision, not something to silently fix.

## Break analysis of the compact-resident-form change

**5 true BREAK sites, ~15 BREAK-BUT-TRIVIAL, everything else NO-BREAK.**
Only **two need real code** rather than test edits: MedCare's
`overlay_to_batch` and `write_alpha_overlay`, which genuinely need materialized
`NodeRow`s for Arrow encoding and should call the named materializer.

`frontier_dispatch.rs`, `patient_shadow.rs`, `rung_horizon.rs`,
`spog_tenants.rs`'s core logic, and MedCare's `attention.rs` /
`backreference.rs` / `obo.rs` already consume the compact
`(AlphaAddr, AlphaStamp)` form via `merge()` / `scanpath()` and need no change.

## The rung × tenant cross does NOT exist

| axis | accessor | mask |
|---|---|---|
| rung 0..9 | `AlphaTunnel::lane(rung) -> &AlphaOverlay` | `.attended_mask()` |
| tenant / graph | `SpogTenants::tenant(concept) -> &AlphaOverlay` | `.attended_mask()` |

`SpogTenants::claim(addr, rung)` ALREADY takes the rung and stamps it, so the
data is there. But `SpogTenants` exposes no mask surface of its own
(`over`, `claim`, `tenant`, `concepts`, `claimed_len`, `merge`), and a grep for
`rung.*tenant|tenant.*rung` returns nothing but that signature. **No structure
crosses them, in code or in any plan.**

The cross IS the meta-awareness the alpha plan §0 item 6 asks for — *"Der Rung
über dem Rung hat ein Bewusstsein über focus of attention"*. The object is a
10×N mask matrix over one allocation; each cell is one `AlphaMask` AND. It
needs no new stored state (both masks are recomputed projections), it is the
first thing that would make the rung dimension observable at all (today
`merge()` folds ten lanes into one flat scanpath and the rung survives only as
a byte in a stamp), and `tenant_mask AND NOT any_rung_mask` is "this domain was
addressable and no rung ever looked".

⊘ **Trap, pinned by its owner** (`ogar-dismech/src/lib.rs`): never carry a rung
ordinal in the residue band. The band is 3 bits / 8 values and the rung ladder
is a different enum — *"unrelated enums that share four variant names."*
