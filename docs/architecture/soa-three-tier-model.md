# SoA Three-Tier Model — Mailbox Lifecycle, Kanban, and Ontology

> **Branch:** `claude/stoic-turing-M0Eiq`
> **Date:** 2026-06-06
> **Authority:** Supersedes any prior baton/emission/CollapseGateEmission framing.

---

## The invariant

**Every SoA envelope is zero-copy from creation to Lance tombstone.**

**Target state:** there is no baton, no emission, and no inter-mailbox handoff
type. No bytes leave the backing store until Lance's own columnar I/O writes
them to disk — and even then the in-memory store is unchanged, not serialized
and freed.

**Current state:** legacy `MailboxSoA::emit()` and `CollapseGateEmission`
artifacts still exist in source and are scheduled for removal (see Tier 1
below). Treat them as migration-only; do not call or extend them.

---

## Tier 1 — MailboxSoA (primary, owned, zero-copy)

The `MailboxSoA<const N>` is the single thought envelope. One mailbox owns one
SoA. The columns (`energy [f32;N]`, `plasticity [u8;N]`, `last_active_cycle [u32;N]`,
`edges [CausalEdge64;N]`, `qualia [QualiaI4_16D;N]`, `meta [MetaWord;N]`,
`entity_type [u16;N]`) are allocated once at mailbox creation and released at
Lance tombstone.

```
creation
   │
   ▼
MailboxSoA<N>  (backing store in-place; column LE contract = MultiLaneColumn)
   │              (envelope LE contract = SoaEnvelope trait)
   │
   ▼  Lance write on each version tick (LE bytes → columnar store)
DatasetVersion(v)  →  DatasetVersion(v+1)  →  ...
   │
   ▼
Lance soft-delete (tombstone)  ← sole lifecycle event that ends the store
```

**Access contract:**
- `MailboxSoaView`: read-only, `&[T]` borrows, `edges_raw() -> &[u64]`
- `MailboxSoaOwner`: `advance_phase(&mut self, to: KanbanPhase)` — sole mutator

**Idempotency guard:** `last_active_cycle [u32;N]` marks the cycle a row was
last written. It is a same-cycle guard, not a history column. (Rename from
`last_emission_cycle` in source — the emission framing is wrong.)

**`MailboxSoA::emit()` and `CollapseGateEmission` are legacy artifacts from a
superseded design and are scheduled for removal.** Until that lands, treat them
as migration-only and non-canonical. There is no intended inter-mailbox handoff
type.

---

## Tier 2 — KanbanColumn / Rubicon lifecycle (sole secondary data)

The only data that is *secondary* to the SoA backing store is the Kanban
phase. This is triggered by the Lance writer, not by the SoA itself.

```
Lance writer  →  VersionScheduler::on_version(&view, at, exec)
                         │  read-only &V: never mutates
                         ▼
                   Option<KanbanMove>  { mailbox, from→to, libet_offset_us }
                         │  caller applies
                         ▼
              MailboxSoaOwner::advance_phase(to)  ← SOLE mutator

KanbanPhase lifecycle (6 states):
  Planning → CognitiveWork → Evaluation → Commit → Plan → Prune
```

Above the SoA mailboxes, ractor (`lance-graph-supervisor`, ractor 0.14,
`supervisor` + `supervisor-lifecycle-audit` features) provides actor-level
meta-orchestration. Each mailbox is a ractor actor. The single-owner invariant
(no virtual ownership pointer needed) is enforced by Rust move semantics through
ractor's message-passing model.

The Kanban column is the only data outside the SoA backing store that reflects
SoA lifecycle. There is no baton, no emission stream, no secondary truth column.

---

## Tier 3 — OGAR classes + OGIT ontology (inherited, O(1))

The identity of a mailbox SoA resolves O(1) to its OGAR class and OGIT
ontology schema. This is Tier 3 because it is *inherited*, not stored per-row.

**Resolution chain:**

```
mailbox address  =  u32 mailbox_id  =  NiblePath  (MailboxId IS the NiblePath)
        │
        ▼  HHTL radix-trie prefix walk — the u32 itself is the trie key
        │  NiblePath::is_ancestor_of:
        │    (other.path >> (4*(other.depth-self.depth))) == self.path
        │  = prefix ancestry = class ancestry (Confirmed)
        ▼
OGIT radix-trie codebook  (O(1) for known classes at compile time)
        │
        ├─ class identity string  ("ogit-op/WorkPackage")
        ├─ schema (fields, assoc, enums, attributes)  — stored ONCE per class in OntologyRegistry
        └─ label inheritance  (parent, mixins, STI)

For new/runtime classes: JIT via lance-graph-planner (JITson / Cranelift)
```

**Thinking style is owned by the Kanban (Tier 2), not the class.** The class
does not dispatch a thinking style — the Kanban column does. Thinking styles
are an **O(1) lookup over an I4-32D address space** (32 nibbles × 4 bits =
128 bits = 2^128 distinct style addresses). The Rubicon phase the Kanban
assigns selects the style; the OGAR class supplies schema + tools, not the
style. (`MappingRow.thinking_style` as a per-class field is therefore the wrong
home — the style belongs to the Kanban lifecycle state, addressed O(1) in the
I4-32D space, not stored per ontology class.)

**MailboxId IS the NiblePath.** The `u32 mailbox_id` field in `MailboxSoA` is not
a handle into a separate lookup — it IS the NiblePath key that the HHTL radix trie
walks. No separate prefix field survives. This makes `entity_type: u16` in every
SoA row entirely redundant: if the ontology resolves O(1) from the address, the
per-row handle violates SoC and defeats radix-trie cheapness. **`entity_type: u16`
removal from SoA rows is total** once O(1) lookup is the sole path. The current
linear scan in `OntologyRegistry::enumerate_first_with_entity_type_id` is a defect
— it should be an O(1) `Vec` index keyed by the 1-based ordinal, or removed entirely
once the per-row handle is gone.

**OGAR active record / DLL AST adapter:** OGAR classes get pragmatic mapping
to inherited tools at compile time. These are cheap inherited registers, not
per-instance data in the SoA. The `Adapter::map` static identity transform in
OGAR + `KnowableFromStore` trait at the lance-graph boundary is the seam.

**surrealdb / kv-lance:** OGAR's DLL AST → SurrealQL path (`ogar-adapter-surrealql`)
requires surrealdb with the `kv-lance` feature. This is BLOCKED(C) — the
`kv-lance` feature is only in the AdaWorldAPI surrealdb fork, coordinates
(git URL, branch) unknown. `surreal_container/Cargo.toml` dep is commented
out pending resolution. **Do not fall back to crates.io surrealdb.**

---

## What does NOT exist (and must not be invented)

| Concept | Status |
|---|---|
| `CollapseGateEmission` as cross-mailbox carrier | **WRONG** — scheduled for removal |
| `MailboxSoA::emit()` | **WRONG** — scheduled for removal |
| "Baton" as inter-mailbox handoff | **WRONG** — superseded |
| `wire_cost_bytes() = 13 + 10·baton_count` | **WRONG** — from CLAUDE.md E-BATON-1, now superseded |
| `Vsa16kF32` as a cross-mailbox carrier | **WRONG** — deprecated, lives only as legacy `cycle` column in `BindSpace` |
| Secondary data beyond KanbanColumn | **WRONG** — Kanban is the only secondary tier |
| BindSpace as the envelope | **MIGRATION IN PROGRESS** — BindSpace is the global legacy; MailboxSoA is the target |

---

## Iron rules that fall out of this model

1. `MailboxSoA` backing store is never copied, never serialized, never transmitted.
   Lance writes LE bytes from it; the store itself stays in place.
2. `VersionScheduler` is read-only (`&V`). It proposes; `MailboxSoaOwner` disposes.
3. `MailboxSoA::emit()` and `CollapseGateEmission` are removed in the next
   pass — they are not part of the correct design.
4. ractor provides the single-owner invariant for mailbox actors — no virtual
   ownership pointer is needed.
5. Ontology resolution is O(1) HHTL prefix lookup for known classes. JITson
   for new ones. The `entity_type: u16` per-row handle may be eliminated once
   the O(1) lookup is the sole path.
6. surrealdb requires the AdaWorldAPI fork with `kv-lance`. Never fall back to
   crates.io. BLOCKED(C) until fork coordinates are provided.

---

## Register-file model — SoA as LE bytecode registers, OGAR class as instruction-set descriptor

This is the load-bearing mental model. Read it before touching the SoA layout,
the class hierarchy, or any codegen template.

### SoA columns = LE registers

The `MailboxSoA<N>` columns are CPU-style registers: fixed width, fixed byte
offset, little-endian, indexed by position. There is no schema in the row.
The row is a register bank.

```
  Byte offset   Width    Column              LE kind
  ──────────    ─────    ──────              ───────
  0             4·N      energy[N]           f32 × N
  4N            N        plasticity[N]       u8  × N
  5N            4·N      last_active_cycle[N] u32 × N
  9N            8·N      edges[N]            u64 × N   (CausalEdge64 LE)
  17N           N        qualia[N]           u8  × N   (QualiaI4_16D packed)
  18N           2·N      meta[N]             u16 × N   (MetaWord LE)
  20N           2·N      entity_type[N]      u16 × N
  22N           4        mailbox_id          u32
  22N+4         4        current_cycle       u32
  ...           ...      (scalars follow)
```

The `SoaEnvelope` trait is the register-file descriptor: it names each
register's byte offset, width, and LE element kind — exactly what a CPU ABI
document does. `ColumnDescriptor` is one register descriptor. `verify_layout()`
is the ABI conformance check.

`MultiLaneColumn` in ndarray is the load/store unit: it iterates the LE bytes
of one typed register into SIMD lanes. Nothing above this level cares about
byte order — it is resolved at the `from_le_bytes` boundary inside the lane
iterator.

### OGAR class = instruction-set descriptor + DTO store for active record

The OGAR class does NOT live in the register. It describes what the register
means. A class is:

- **Label** — the OGIT identity string (`"ogit-op/WorkPackage"`), the
  human-readable name, and the full label-inheritance chain up the HHTL trie.
- **Schema** — the field set: which fields exist, what types they are, which
  are required vs optional. Stored once per class in `OntologyRegistry`
  (`MappingRow`); never duplicated into SoA rows.
- **Tools** — the methods / adapters that operate on the register. These are
  inherited from the class hierarchy (HHTL prefix ancestry = class ancestry).
  A subclass inherits all parent tools without restating them. The default
  mechanism is **compile-time Rust trait impls**: one `impl Tool for ClassFoo`
  per class, monomorphized at build time from the HHTL inheritance chain.
  Zero-cost — no vtable, no `dyn`, no runtime dispatch. "Cheap inherited
  registers" in the architecture doc is literal: the trait impl IS the register,
  and the compiler erases it to a direct call.

  **Dispatch escape hatch.** Runtime dispatch is *not* the default and is not
  baked into the substrate. If a class genuinely needs runtime dispatch, it adds
  a **dispatcher class** as an escape hatch attached to a register — an opt-in,
  per-class override. The base case stays monomorphized and zero-cost; only the
  classes that ask for dispatch pay for it.
- **Codegen templates** — Askama/Jinja `Class<Template>` views (see below).
- **Active record** — the class instance wraps a slice of the register bank
  and provides the domain API. Methods come from the class hierarchy; data
  comes from the register slice. The active record is the class speaking about
  one register bank row, not a separate struct.

```
  mailbox address  ──HHTL O(1)──►  OGAR Class
                                       │
                              ┌────────┼────────────┐
                              ▼        ▼             ▼
                           label    schema         tools
                          (string) (fields)  (inherited methods)
                              │        │             │
                              └────────┼─────────────┘
                                       ▼
                             active record instance
                          (class + register bank slice)
                          no new fields; register IS the data
```

### Askama/Jinja codegen = masked selection from the class DTO — split by known vs JIT

A `Class<Template>` is not a new type. It is a **masked selection** over the
class DTO: the template declares which fields it reads, the codegen emits only
those fields as strongly-typed Rust from the class schema, and the result is a
zero-overhead view — one `select` mask over one class, compiled to a concrete
struct by the Askama/Jinja template engine.

**When does this codegen happen? Both — split by known vs JIT:**

- **Known OGAR classes** (those in the codebook at compile time): `build.rs` /
  proc-macro emits concrete Rust structs at build time. Zero runtime cost;
  the compiler sees the full type.
- **Runtime-discovered / new classes**: `JITson` path in `lance-graph-planner`
  (Cranelift backend). The template is instantiated at first encounter and
  compiled to native code. Same masked-selection semantics, deferred to first use.

**The template engine is not the only codegen front-end.** The SurrealDB AST can
resolve the same masked selection as **Elixir-syntax templates**, but it is
compiled through the *same* runtime split — known classes compiled at build time,
new classes via JIT Cranelift / JITson in `lance-graph-planner`. Askama/Jinja and
the SurrealQL/Elixir-template path are two surface syntaxes over one codegen
backend; both bottom out at the build-known vs JIT-new split, never at runtime
interpretation.

```
  OGAR Class { field_a, field_b, field_c, tool_x, tool_y, template_T }
                     │
                     ▼  Class<TemplateFoo> mask = { field_a, tool_x }
                     │
              codegen (Askama/Jinja)
                     │
                     ▼
  struct FooDto { field_a: TypeA }   // only selected fields
  impl FooTool for FooDto { ... }    // only selected tools
```

The template is a compile-time projection. No runtime dispatch. No new
inheritance hierarchy. The mask is the whole mechanism — the class already
owns the full schema; the template carves the slice it needs.

This makes every DTO in the system a **derived view of an OGAR class**, never
an independent schema. A session that proposes a new DTO struct without a
corresponding OGAR class entry is creating schema drift.

### Why this is the correct mental model

| CPU register file | SoA mailbox |
|---|---|
| Register bank | `MailboxSoA<N>` columns |
| Register descriptor (ABI doc) | `SoaEnvelope` + `ColumnDescriptor` |
| Load/store unit (MOV instruction) | `MultiLaneColumn` SIMD iterator |
| Instruction-set definition | OGAR class (label + schema + tools) |
| Subroutine calling convention | HHTL inheritance (prefix = class ancestry) |
| Object code (compiled binary) | Askama/Jinja codegen from `Class<Template>` |
| Active register state | register bank slice wrapped by active record |

The SoA is dumb. It holds bytes. The class makes the bytes meaningful.
The template makes the class usable without the full schema in scope.
Nothing in the register knows about the class; nothing in the class knows
about which register it describes at runtime — the address is the link.
