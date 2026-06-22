# OGAR consumer pre-flight — pull the classid, never re-mint the Core (the consumer spellbook)

> **Operational mirror** of `docs/CONSUMER-BRIDGE-DEPRECATION.md` (the migration
> recipe, #589/#590) and the **inverse** of OGAR's
> `SURREAL-AST-TRAP-PREFLIGHT.md` (the producer trap). Read this BEFORE the
> keyboard fires in any consumer that needs a classid — five questions, 90
> seconds, one mirror.
>
> Where OGAR's spellbook stops a *producer* substituting SurrealQL AST for
> `Class` + `ActionDef`, THIS spellbook stops a *consumer* (woa-rs, medcare-rs,
> smb-office-rs, …) re-implementing the OGAR Core locally instead of pulling
> from it. Same boundary, opposite arm.
>
> Status: **SPELLBOOK v0** (2026-06-22). Append-only.
>
> READ BY: `integration-lead`, `baton-handoff-auditor`, `core-first-architect`,
> the `Plan` subagent — and every consumer-crate session touching a classid, a
> `*Bridge`, the OGAR codebook, or an `OntologyRegistry`.
>
> Grounded in: `docs/CONSUMER-BRIDGE-DEPRECATION.md` (the recipe),
> `.claude/knowledge/core-first-transcode-doctrine.md` (pull from the Core,
> never mint a parallel registry), OGAR#95 APP-CLASS-CODEBOOK-LAYOUT (the hi/lo
> classid split), OGAR#97 `PortSpec::APP_PREFIX` + `render_classid_for`, OGAR#98
> `canonical_concept_name`.

## The trap, in one paragraph

You're a consumer — woa-rs, medcare-rs, smb-office-rs — and you need the classid
for a domain concept (`Stundenzettel`, `Patient`, `Rechnung`). The first reflex
is *"construct the bridge: `WoaBridge::new(registry)?.entity(name)?`."* That
reflex is **System-1** — it reads like a clean lookup. But the bridge carries an
`OntologyRegistry` it has to hydrate, couples your customer binary to the spine,
and is `#[deprecated]` as of #589. The **System-2** move is a pure function:
`WoaPort::class_id(name)` (spine) or `ogar_codebook::canonical_concept_id(name)`
(membrane). No registry, no hydration, no construction. The deeper trap — the
one the Core-First doctrine forbids outright — is copying OGAR's codebook *into
your repo* (a local `const *_CODEBOOK`, a hand-rolled `name → id` map, a parallel
`class_ids` registry). The Core owns the codebook; a copy drifts the moment OGAR
mints a concept, and the cross-fork convergence (`Stundenzettel` ⇄ `TimeEntry` ⇄
SMB's hours all resolving to `0x0103`) silently breaks.

The trap is silent at write time and loud at the next OGAR codebook bump.
Avoiding it costs five questions. Recovering from it costs a drift bug nobody
can see until two forks disagree on a classid.

## The five questions — run BEFORE the keyboard fires

```
Q1.  WHAT do I need from OGAR?
     ├─ the classid for ONE concept (by name)
     │      →  PULL it.  *Port::class_id(name) / canonical_concept_id(name).
     │         A pure function returning Option<u16>.  Proceed.
     └─ a registry of ALL concepts / the whole codebook / a name↔id map
            →  you're about to RE-MINT the Core.  STOP.
               The Core is the single source; pull per-concept, don't copy.

Q2.  Am I CONSTRUCTING a bridge object?
     ├─ XBridge::new(registry)?.entity(name)?   (registry, hydrate, .entity)
     │      →  DEPRECATED (#589).  The bridge carries state the pull doesn't
     │         need.  STOP.  Drop the registry; the classid pull is stateless.
     └─ *Port::class_id(name)   (no construction, no &self)
            →  correct.  Pure function.  Proceed.

Q3.  Am I COPYING the codebook / minting a parallel registry?
     ├─ a local `const *_CODEBOOK` / `*_ALIASES`, a hand-rolled
     │  `HashMap<&str,u16>`, or my own `class_ids` mirror
     │      →  Core-First VIOLATION.  It drifts on the next OGAR mint.  STOP.
     └─ pulling from contract::ogar_codebook / lance_graph_ogar
            →  correct.  One source; the parity-guard fuses drift at build.

Q4.  WHICH CRATE am I pulling from?   (the BBB-barrier)
     ├─ customer binary / membrane (woa-rs, medcare-rs realtime, smb-office-rs)
     │      →  lance_graph_contract::ogar_codebook ONLY.  Zero-dep, BBB-safe.
     │         lance_graph_ogar / -engine / -planner in a customer Cargo.toml
     │         is a BBB BREACH.  STOP.
     └─ spine / internal crate
            →  lance_graph_ogar::*Port is fine (it re-exports ogar_vocab).

Q5.  Does my classid CARRY the app prefix?
     ├─ render id = APP(hi u16) << 16 | concept(lo u16)   (0x0003_0103)
     │  RBAC / ontology key on the shared lo u16
     │      →  correct (OGAR#95 §2; spine: render_classid_for::<P>() #97).
     └─ I'm using the bare lo u16 as the render id, or inventing my own
        prefix scheme
            →  drift from the allocation table.  STOP.  Stamp the prefix.
```

Any "STOP" answer catches the trap pre-materialization.

## Diagnostic signatures — what the trap looks like in review

- **`use lance_graph_{ontology,ogar}::bridges::*Bridge`** in a consumer crate,
  followed by `XBridge::new(...)` — the deprecated construction path.
- **An `OntologyRegistry` / `registry` field on a consumer's own struct** — the
  bridge's hydrate state leaking into the consumer; the pull needs none.
- **A local `const *_CODEBOOK` / `*_ALIASES`, a hand-rolled `name → classid`
  map, or a `class_ids`-shaped table** copied into the consumer repo — the Core
  copied, not pulled. This is the loud one.
- **A per-app prefix constant invented locally** (`const APP: u32 = 0x...`)
  instead of `PortSpec::APP_PREFIX` / the OGAR#95 allocation table.
- **`lance-graph-ogar` / `lance-graph-engine` / `lance-graph-planner` in a
  customer-binary's `Cargo.toml`** — a BBB-barrier breach (Q4).
- **A consumer-side `entity_type_id()` / `schema_ptr` chain** to get a classid
  the `class_id(name)` pull returns in one call.

One signature is suspicious; a local codebook copy alone is the trap.

## Remediation — three moves (= the `CONSUMER-BRIDGE-DEPRECATION.md` recipe)

1. **Name the concept.** It's the argument to the bridge's `.entity(name)` —
   `Stundenzettel`, `Patient`, `Rechnung`. That string is all you need.
2. **Pull the classid** — pure function, no registry:
   - spine: `lance_graph_ogar::WoaPort::class_id(name) -> Option<u16>`
   - membrane (BBB): `lance_graph_contract::ogar_codebook::canonical_concept_id(name)`
3. **Stamp the app prefix + delete the old surface.** Render id =
   `APP << 16 | cid`; authorize on the shared `cid` (lo u16). Then delete: the
   `*Bridge` import, any `OntologyRegistry` field, and every local codebook
   copy. Your diff touches only your crate; the spine is byte-for-byte unchanged.

DoD: no `*Bridge` / `XBridge::new` / local `*_CODEBOOK` survives a grep in your
repo; the classid pull is a pure function call.

## Consumer status (the worked examples — from #589's snapshot, 2026-06-22)

| Consumer | Files still on `lance_graph_{ontology,ogar}::bridges` | In scope here |
|---|---|---|
| MedCare-rs | 33 | yes (`medcare-rs`) |
| woa-rs | 6 | yes (`woa-rs`) |
| smb-office-rs | 4 | no |
| odoo-rs | 0 ✓ | — |
| openproject-nexgen-rs | 0 ✓ | — |

These three are the migration backlog. The terminal `bridges/` deletion in the
spine is gated on all three reaching 0.

## A Core gap this spellbook surfaces (honest — flag, don't paper)

`contract::ogar_codebook` mirrors `canonical_concept_id` / `canonical_concept_name`
(the lo-u16 pull, BBB-safe) but does **not** yet mirror OGAR#97's
`PortSpec::APP_PREFIX` / `render_classid_for` (the hi-u16 render composition). So
a **membrane** consumer can pull the shared concept but must hand-stamp the app
prefix from the OGAR#95 allocation table — a small re-derivation a
`contract::ogar_codebook::APP_PREFIX` mirror would remove. Per Core-First the
consumer must NOT hard-code `0x000N`; file the contract mirror (the
`canonical_concept_name` precedent is OGAR#98) rather than minting a local
prefix const. Tracked: `ISS-CONTRACT-APP-PREFIX-MIRROR`.

## When this doc fires + trigger phrases

Author (before keyboard): any consumer session that needs a classid, migrates
off a `*Bridge`, or touches the OGAR codebook. Review (PR landing): any consumer
PR adding a `*Bridge` import, a local codebook copy, or a `lance-graph-ogar` /
`-engine` / `-planner` dep to a customer binary.

Triggers: `*Bridge` · `class_id` · `classid` · `entity_type_id` · `codebook` ·
`*_ALIASES` · `OntologyRegistry` · `pull the classid` · `bridge migration` ·
`render classid` · `APP_PREFIX` · BBB-barrier.

## What this doc is NOT

- **Not a ban on `lance-graph-ontology`.** The OGIT cache (TTL/RDF hydration) is
  a legitimate consumer dep; the trap is the *bridge construction* + the *local
  codebook copy*, not the crate.
- **Not a ban on the bridges existing.** They're `#[deprecated]`, not removed —
  this spellbook governs NEW consumer code and the migration, not a hard cutover.
- **Not retroactive blame.** The 33/6/4 backlog gets the three-move remediation,
  not a citation. Citations are for future consumer code.

## Cross-refs

- `docs/CONSUMER-BRIDGE-DEPRECATION.md` — the migration recipe (the *what*).
- `.claude/knowledge/core-first-transcode-doctrine.md` — pull from the Core,
  never mint a parallel registry (the *why*).
- OGAR `docs/SURREAL-AST-TRAP-PREFLIGHT.md` — the producer-side mirror (the
  *inverse* arm of the same boundary).
- OGAR#95 `APP-CLASS-CODEBOOK-LAYOUT.md` (hi/lo split) · #97 `render_classid_for`
  · #98 `canonical_concept_name`.
