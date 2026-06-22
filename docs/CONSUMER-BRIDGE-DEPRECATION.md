# CONSUMER BRIDGE DEPRECATION — landing shape for the OGAR migration

> **Status:** preemptive deprecation. Symbols still compile; every use
> emits a `#[deprecated]` warning pointing here. **Nothing removed.**
> Deletion lands later, after all consumer repos have migrated.
>
> **Source of truth:** `AdaWorldAPI/OGAR` — see
> [`docs/APP-CLASS-CODEBOOK-LAYOUT.md`](https://github.com/AdaWorldAPI/OGAR/blob/claude/medcare-bridge-lance-graph-wmx76z/docs/APP-CLASS-CODEBOOK-LAYOUT.md),
> [`docs/CONSUMER-MIGRATION-HOWTO.md`](https://github.com/AdaWorldAPI/OGAR/blob/claude/medcare-bridge-lance-graph-wmx76z/docs/CONSUMER-MIGRATION-HOWTO.md),
> [`docs/CLASSID-RBAC-KEYSTONE-SPEC.md`](https://github.com/AdaWorldAPI/OGAR/blob/claude/medcare-bridge-lance-graph-wmx76z/docs/CLASSID-RBAC-KEYSTONE-SPEC.md),
> tracking PR **AdaWorldAPI/OGAR#95**.

## Why these bridges are deprecated

The agnostic spine (`lance-graph-*`) does NOT own consumer ontology.
The per-consumer bridges that lived here grew up before OGAR was
the AR-shaped class registry it is today; they coupled `lance-graph-ogar`
(and the OGIT cache `lance-graph-ontology`) to consumer-specific shapes
that should live entirely in the consumers, with OGAR providing the
classid + schema + grant.

The replacement is **pull the classid via OGAR PortSpec, then enrich
locally**:

```rust
// BEFORE — bridge constructed in the consumer:
use lance_graph_ogar::bridges::WoaBridge;
let bridge = WoaBridge::new(registry)?;
let entity = bridge.entity("Stundenzettel")?;
let class_id = entity.schema_ptr.entity_type_id();  // 0x0103

// AFTER — static port lookup in the consumer:
use ogar_vocab::ports::{WoaPort, PortSpec};
let class_id: Option<u16> = WoaPort::class_id("Stundenzettel");  // Some(0x0103)
//
// Render lens (per-app) lives in the high u16: 0x0003 << 16 | 0x0103
// = 0x0003_0103 (woa-rs's render classid for billable work entry).
// RBAC + ontology key on the low half (shared).
```

That's it — no `Registry`, no `hydrate`, no `OntologyRegistry` field on
the consumer. The classid pull is a pure function call; cross-fork
convergence is preserved (every consumer of `BILLABLE_WORK_ENTRY`
resolves to `0x0103` whether it's WoA's `Stundenzettel`, SMB's,
OpenProject's `TimeEntry`, or Redmine's). See OGAR
`APP-CLASS-CODEBOOK-LAYOUT.md` §1 (hi/lo split) and §3.5–3.7
(per-app rendering, key-value content, RAG membrane).

## What is deprecated (this PR)

### `lance-graph-ogar::bridges` — OGAR-driven per-port aliases

Six `UnifiedBridge<P>` type aliases — the consumer-facing names. The
underlying `UnifiedBridge<P>` harness is **not** deprecated (it's the
implementation mechanism for `NamespaceBridge`; internal-only).

| Deprecated symbol | Replacement |
|---|---|
| `bridges::OpenProjectBridge` | `ogar_vocab::ports::OpenProjectPort::class_id(name)` |
| `bridges::RedmineBridge` | `ogar_vocab::ports::RedminePort::class_id(name)` |
| `bridges::MedcareBridge` | `ogar_vocab::ports::HealthcarePort::class_id(name)` |
| `bridges::WoaBridge` | `ogar_vocab::ports::WoaPort::class_id(name)` |
| `bridges::SmbBridge` | `ogar_vocab::ports::SmbPort::class_id(name)` |
| `bridges::OdooBridge` | `ogar_vocab::ports::OdooPort::class_id(name)` |

The `Port` types (`OpenProjectPort`, `WoaPort`, …) themselves are
**not** deprecated — they are the replacement.

### `lance-graph-ontology::bridges` — legacy per-tenant structs

Four bespoke structs. These pre-date OGAR's codebook and have no
`PortSpec` impl in `ogar-vocab::ports`. They live in the OGIT cache
crate (`lance-graph-ontology`), which by SoC rule cannot depend on
`ogar-vocab`. Consumers should pull the classid via the OGAR side and
let OGIT continue to do what it does today (legacy TTL/RDF hydration).

| Deprecated symbol | Replacement |
|---|---|
| `lance_graph_ontology::bridges::OgitBridge` | `ogar_vocab::ports::*Port::class_id(name)` for the relevant port |
| `lance_graph_ontology::bridges::WoaBridge` (legacy struct) | `ogar_vocab::ports::WoaPort::class_id(name)` |
| `lance_graph_ontology::bridges::SpearBridge` | (no OGAR port yet — author one per `CONSUMER-MIGRATION-HOWTO.md`) |
| `lance_graph_ontology::bridges::SharePointBridge` | (no OGAR port yet — author one per `CONSUMER-MIGRATION-HOWTO.md`) |

## What this PR does NOT do (read this — flagged by parallel sessions)

This PR is **deprecation only**. It signals the migration target but
does **not** ship the full replacement API. Two specific gaps remain:

### Gap 1 — `lance-graph-rbac` has no `authorize(actor, classid, op)` yet

The keystone in `CLASSID-RBAC-KEYSTONE-SPEC.md` describes a single
`authorize(actor: &Membership, classid: u16, op: Op) -> AccessDecision`
entry point in `lance-graph-rbac`. Today (2026-06-22) the crate ships
the scaffolding (`AccessDecision::{Allow, Deny, Escalate}`,
`role`/`permission`/`policy`/`access` modules) but **no `authorize`
function and no classid-keyed signature**. So a consumer reading the
deprecation note "authorize by classid" cannot, today, *call* that —
the function does not exist.

Why this PR doesn't add it: the keystone is graded `[H]` and gated on
**`PROBE-OGAR-RBAC-AUTHORIZE`** (OGAR `CLAUDE.md` non-negotiables;
`CLASSID-RBAC-KEYSTONE-SPEC.md` §10). Shipping the signature before
the probe locks the shape before falsification — the gate exists
exactly to prevent that. The keystone is its own PR, after the probe
runs green.

Until the keystone ships, consumers should:
- migrate the **classid pull** off the bridges now (this PR's target —
  `Port::class_id(name)`),
- keep their **existing auth** (or none) for the authorize call site,
- **NOT re-introduce a bridge as an auth stopgap**.

### Gap 2 — no `Membership` / `Op` types yet

The keystone's actor type (`Membership` — I-K6 in the spec) and op
type are not defined in `lance-graph-rbac` either. They land with the
keystone PR, not here.

## What is NOT deprecated

- **`UnifiedBridge<P>`** — implementation harness, not consumer surface.
- **`ogar_vocab::ports::*Port`** — these are the replacement.
- **Internal lance-graph bridges** (`unified_bridge.rs` in callcenter,
  `*_bridge.rs` in thinking-engine / learning / cognitive-shader-driver)
  — these are internal cognitive seams, not consumer-migration targets.
- **`OPENPROJECT_CODEBOOK` / `REDMINE_CODEBOOK` constants** — already
  deprecated in prior PR #570 pointing at `ogar_vocab::ports::*_ALIASES`.

## Removal timeline

Deletion of the deprecated symbols is **gated on every consumer being
green**. Per `git grep` at this PR's filing (2026-06-22):

| Consumer | Files still importing `lance_graph_{ontology,ogar}::bridges` |
|---|---|
| MedCare-rs | 33 |
| woa-rs | 6 |
| smb-office-rs | 4 |
| odoo-rs | 0 ✓ |
| openproject-nexgen-rs | 0 ✓ |

The terminal `bridges/` deletion PR opens only after MedCare-rs,
woa-rs, and smb-office-rs ship their migrations. No removal window
is announced — the deprecation is a beacon, not a deadline.

## How to migrate (one-page recipe)

1. Confirm your OGAR `PortSpec` exists (`ogar_vocab::ports::*`). All
   six ports above already exist.
2. Replace the bridge construction call with a static port lookup:
   ```rust
   let cid: u16 = YourPort::class_id(entity_name)
       .ok_or_else(|| /* concept not in your port's alias table */)?;
   ```
3. Form your render classid by stamping your app's high-u16 prefix
   (per OGAR's `APP-CLASS-CODEBOOK-LAYOUT.md` §2 allocation table —
   `0x0001` OpenProject · `0x0002` Odoo · `0x0003` WoA · `0x0004` SMB ·
   `0x0005` Medcare · `0x0007` Redmine):
   ```rust
   const APP: u32 = 0x0003_0000;  // e.g. woa-rs
   let render_classid: u32 = APP | (cid as u32);
   ```
4. Enrich + render by `render_classid`; authorize by `cid` (the low
   half — shared grant lattice).
5. Delete the bridge import + any hand-rolled registry/hydration in
   your repo.

DoD: no `XBridge` / `UnifiedBridge<…>` symbol from `lance_graph_*`
survives in your repo's grep; the classid pull is a pure function call;
your diff touches only OGAR (a port, if new) + your own crate. The
spine is byte-for-byte unchanged.
