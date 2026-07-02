# Routing — how an address finds its class, tenant, owner, and board

> READ BY: v3-envelope-auditor, v3-kanban-executor-engineer, anyone adding a
> lookup/scan/dispatch over V3 keys. Byte layout itself: `le-contract.md`.

## Status: FINDING (address canon operator-locked; kanban routing EXTENDS pending W1/W2)

---

## 1. The address is the router (no lookup tables in the hot path)

The canonical 16-byte LE key routes at every prefix depth
(CANON, `CLAUDE.md` § Minimal SoA node):

```
0..4   classid  u32   [hi u16 CANON concept/domain:appid][lo u16 CUSTOM]
4..6   HEEL     u16   ┐
6..8   HIP      u16   ├ cascade tiers (HHTL path; 256×256 centroid tiles)
8..10  TWIG     u16   ┘
10..13 family   u24   ┐ basin-local key
13..16 identity u24   ┘ (masked load once the trie binds the prefix)
```

Routing consequences, in prefix order:

| Prefix consumed | Routes to | Mechanism |
|---|---|---|
| canon hi-u16 | domain + app (`ConceptDomain` = `canon >> 8`) | range predicate — canon-high IS a clustered index (E-CLASSID-CANON-HIGH-IS-A-CLUSTERED-INDEX) |
| custom lo-u16 | render skin / (post-P4) template lens; `0x1000` = V3-adoption monitor | catalogue dispatch; monitor is a marker, never a semantic |
| classid (full) | `ClassView` + `classid_read_mode(c).value_schema` (tenant schema) | O(1) codebook; longest-prefix codebook binding (OGAR tier rule) |
| HEEL/HIP/TWIG | cascade position | shift/mask only (`tier = level >> 2`); tier distance = 3 table lookups |
| family | basin (neighborhood grouping) | dormant while 0 (zero-fallback ladder) |
| identity | the row | masked 6-byte `local_key()` |

**Zero-fallback ladder:** a zero tier means *not consulted*, never
*compacted away* (RESERVE-DON'T-RECLAIM). Bootstrap addressing = classid 0
+ family 0 + identity alone.

## 2. Mailbox routing — MailboxId IS the NiblePath

Per `docs/architecture/soa-three-tier-model.md` Tier 3: the `u32 mailbox_id`
is not a handle into a table — it IS the HHTL radix-trie key.

- `NiblePath::is_ancestor_of` = prefix ancestry = class ancestry.
- Ontology resolution is O(1) for known classes; JITson for new ones.
- Consequence: per-row `entity_type: u16` is transitional and retires once
  the O(1) path is sole (three-tier doc §Tier-3) — a routing entropy
  milestone in its own right.
- **Thinking metric for free** (handover F4): `from_guid_prefix_v3` +
  `family_hop_count` give O(depth), zero-value-decode graph distance —
  the natural adjacency for AriGraph/episodic tissue.

## 3. Write routing — cast → delegation cache → owner → board

The V3 write path routes by OWNER, not by table (W1/W2 of the
INTEGRATION-PLAN; today only the stamp exists):

```
consumer/engine
   │  cast(on_behalf = envelope.mailbox_owner(), payload = BusDto)
   ▼
batch writer ── delegation cache (cast id vs stamp: hit→proceed, miss→resolve once)
   │  fires AHEAD KanbanMove at cast (never waits for the write)
   ├────────────► owner actor (supervisor kanban_actor) ── MailboxSoaOwner::advance_phase  [SOLE mutator]
   ▼
Lance columnar I/O (LE bytes from the in-place store — the only byte writer)
```

Routing rules:
- Moves route to the mailbox's OWNER actor; nothing else mutates phase.
- A missing/late update never gates a thinking cycle (standing async plan);
  updates only reprioritize (StepMask).
- Delegation is resolved AT the batch writer — consumers never route
  "may I write?" questions themselves.

## 4. Read-mode routing — legacy forms resolve forever

Stored corpora carry pre-flip classids and V1/V2 tails. Readers route
through:

- `classid_canon(id)` — strict, new mints;
- `classid_canon_compat(id)` — surfaces serving BOTH stored forms (RBAC
  grants, un-re-baked corpora);
- `BUILTIN_READ_MODES` `CLASSID_*_LEGACY` alias keys — persisted old-form
  ids resolve without re-baking (mint-forward, never reinterpretation).

Alias retirement routes through the corpus proof only (W6). FORBIDDEN
discriminators on the composed u32: `as u16`, `& 0xFFFF`, `>> 8`, `>> 16`
(`/v3-audit` check 1).

## 5. Monitor routing — adoption is a range count

Because canon-high clusters by domain, both governance metrics are
key-range counts over the same index — ONE two-metric scanner (W6a):

- **adoption%** = rows whose custom half carries `0x1000` (V3 substrate)
  vs total, per domain;
- **corpus proof** = count of old-form rows (legacy order / legacy tails);
  zero ⇒ alias retirement unlocks; adoption 100% ⇒ P4 trigger (operator
  checkpoint) ⇒ marker deprecates ⇒ custom half opens for the
  render/template catalogue.

Cross-ref: `le-contract.md` (bytes), `tenants.md` (what the value lanes
mean), `consumer-map.md` (who writes what), primer §5,
board E-V3-MARKER-IS-A-MONITOR.
