# sap-glove-quack-v1 — SAP as the sixth glove over lance-graph-quack

> Status: PROPOSAL (2026-10-10). Read-only draft; nothing below is built.
> Sibling of `lance-graph-java` (the Java glove). Same law, different front.
> Grades: `[G]` = read in source at the pinned SHAs below, `[H]` = design
> hypothesis that needs its probe, `[S]` = operator ruling required.

## 0. Pins (what this draft was read against)

| repo | sha | read |
|---|---|---|
| lance-graph | `0c7d4e1` (2026-10-10) | `crates/lance-graph-sap/*`, `lance-graph-quack/src/lib.rs`, `lance-graph-mask-risc/src/ir.rs`, `lance-graph-dir-sim/src/{lib,exec,cloud}.rs` |
| lance-graph-java | `0b835f1` (2026-10-09) | `CLAUDE.md`, `native/lgj-abi/src/exports.rs` (export list) |
| OGAR | HEAD 2026-10-10 | `CLAUDE.md`, `README.md`, `crates/ogar-vocab/src/ports.rs`, `crates/ogar-dir-sim/src/plan.rs`, `crates/ogar-from-schema/src/xsd.rs` |
| stalwart (fork) | `01aaceb` | `crates/directory/src/backend/dirsim/mod.rs` (dir-sim recipient backend) |
| SIMAF / SIMAFPort | cloned, **not yet inventoried** | W0 harvests them; CATS is already harvested into `lance-graph-sap` |

## 1. The thesis in one table

`lance-graph-java` settled the shape: *Java sees boring `sql()`; lance-graph
does the masking; nothing proportional to rows ever reaches Java except
through a method named `materialize*`.* SAP is the same glove pointed at a
different crowd. An SAP system, an EDI partner, xSuite or SAPconnect sees
the boring wire **it already speaks**; underneath, every population
operation is one Quack program on the one mask-RISC evaluator.

| tier | the caller writes | the caller never learns |
|---|---|---|
| consumer crate → ndarray | `U8x64::cmpeq_mask(..)` | which SIMD backend ran |
| Java → lance-graph (lgj) | `sql("select …")` | masks, hops, ternlog |
| **SAP world → lance-graph (this plan)** | a BAPI call, an IDoc, a SCOT send request, an xSuite mailbox drop, an EDIFACT interchange | that the "SAP system" answering is a set of schemas over Quack |

The emulation and the later real connection are **one pipeline**. Only the
two ends change:

```
source adapter ─► bind (cold, fallible) ─► Quack lower ─► mask-RISC fold ─► pivot ─► named sink
   fixture/file/                schema-                    Program            ordinal     BAPI params /
   mailbox/RFC                  driven                     (one evaluator)    map         IDoc flat file /
                                                                                          EDIFACT / SMTP
```

The simulation uses fixture sources and side-effect-free sinks that only
produce an execution plan. Docking means pointing a source at the real port
and letting an actuator consume the plan. The engine, the schemas and the
pivots stay as they are. That is what "jederzeit andocken" means
operationally.

## 2. What already exists — the template is `lance-graph-sap` `[G]`

`crates/lance-graph-sap` (own workspace; deps: `lance-graph-contract`,
`lance-graph-mask-risc`, `lance-graph-quack`, `hmac`, `sha2`; **no OGAR
dependency**) already implements the full pattern for one surface, SAP CATS:

| module | role | the generalisable piece |
|---|---|---|
| `schema.rs` | 23 `FieldDescriptor`s harvested from the ABAP leaf declarations, append-only ordinals, `abap_group`/`technical_name`/`csharp_name`/`native_type`/`width`/`carrier: LaneKind`; `CatsSchema: ClassView`; class id supplied by the caller's registry | **the schema is the moving part** |
| `bind.rs` | cold ingestion into owned `U32`/`I32`/`U64` columns borrowed as `LaneRef`; NUMC, ABAP_BOOL (`X`/space ≡ `true`/`false`), exact decimal scale, UTF-16 width check; strings → first-occurrence dictionary codes kept as edge metadata | the edge discipline |
| `query.rs` | `CatsQuery::prepare` lowers `Filter::and[EqU32, GeI32, LeI32]` + `Agg::GroupSumI32` and `Agg::Rows`; `execute_into` / `select_into` on caller-owned scratch | **the fold** |
| `edge.rs` | `HASH_ORDINALS` (HMAC-SHA512 projection order, two profiles that deliberately disagree), `BAPI_ORDINALS` + `BAPI_PARAMETERS` → `bapi_sink` for `BAPI_CATIMESHEETMGR_INSERT`; the only row materialisation is `materialize_rows` inside the sink | **the pivot** and the named sink |

The comment on `BAPI_ORDINALS` already states the algebra this plan
promotes: *"a permutation of a selection of FIELDS, not a second field
vocabulary."*

## 3. The three Bordmittel and how far each stretches

### 3.1 Schema — the only thing that is authored per surface

A surface is a const table of `FieldDescriptor`s plus a `ClassView`
implementation, exactly as CATS. Generalise by lifting the CATS-specific
parts out of `bind.rs` behind a trait, without changing CATS behaviour:

```rust
pub trait WireSchema: ClassView {
    const FIELDS: &'static [FieldDescriptor];   // append-only ordinals
    const SOURCE_PIN: &'static str;             // file:line or document id it was harvested from
}
pub struct WireBatch<S: WireSchema> { /* today's CatsBatch body, generic */ }
```

Every field comes from a pinned source, as CATS's do (`schema.tsv` is
checked against `FIELDS` in a unit test). Harvest sources per surface:

| surface | harvest source | harvester |
|---|---|---|
| CATS | SIMAF ABAP + SIMAFPort C# mirror | done |
| IDoc control/data/status records | the IDoc type's XSD / parser-format export from the SAP system (WE60 / IDoc documentation export) | `ogar-from-schema::xsd` — already a byte-exact XSD walker for MARS; a second consumer `into_classes` is the plan it names |
| BAPI parameter structures | function module interface export (SE37) or SIMAF call sites | table transcription, oracle-checked |
| SCOT/SAPconnect send request & status | SIMAF/SIMAFPort mail channel classes (W0) | harvest |
| xSuite inbound document | xSuite Interface scenario config + the stack/POBox fields already observed in the EWS→Graph migration | harvest |
| EDIFACT / X12 | UN/EDIFACT directory segment specs for the chosen message (e.g. INVOIC D.96A) | table transcription |

### 3.2 Mask fold — the evaluator, unchanged

Every SAP-side question is a `Filter` + `Agg` lowered by Quack and run by
`execute_into`. The fold vocabulary present at the pin `[G]`:

| SAP question | Quack spelling | IR terminal |
|---|---|---|
| CATS hours per activity (exists) | `Agg::GroupSumI32` | `GroupSumI32` |
| IDocs per status / message type (WE02/BD87 shape) | `Agg::GroupReduce{Local(status), Count}` | `GroupReduce` |
| IDocs per partner × message type | `GroupAddr::Pair{hi, lo, stride}` | `GroupReduce` (fused composite key) |
| data segments of control records in status 51 | `Filter::EqU32Via{fk: docnum, key, v}` | `Pred::EqU32Via` (one pass, no foreign plane) |
| line items whose header passes a resident plane | `Filter::Semijoin` | `MaskOp::Gather` |
| header ← any of its lines (one-to-many back) | `Agg::ScatterOrU32` | `ScatterOrU32` |
| sum of amounts per document / partner | `Agg::GroupSumViaI32` / `GroupReduce{SumI32}` | see gap G1 |
| "which rows go to the sink" | `Agg::Rows` | `Keep` → caller's mask |

IDoc hierarchy (SEGNUM/PSGNUM, header → item → sub-item) is fk lanes, and a
descent is the lgj hop shape, `src_mask → hop → dst_mask`. Segment order is
placement order: bind preserves SEGNUM order as row order and
`materialize_rows` yields ascending rows, so serialisation needs no
`ORDER BY` (Quack does not have one, and does not need one here).

### 3.3 Pivot — ordinal maps as the cross-surface algebra

Every wire shape is a **pivot**: positions of a target shape → ordinals of
one canonical field basis. CATS already has three of them
(`HASH_ORDINALS`, `BAPI_ORDINALS`, C# field order). Promote the pattern to a
type:

```rust
/// Target position k reads canonical ordinal `map[k]`. Const, cold, validated.
pub struct Pivot<const N: usize> { pub map: [u16; N], pub names: [&'static str; N] }
```

Laws (all checkable at compile time or in one unit test):

- **validity**: every `map[k] < FIELD_COUNT`; duplicates only where the
  target genuinely repeats a field, and then declared;
- **composition**: `(p ∘ q).map[k] = q.map[p.map[k]]` — IDoc segment →
  canonical → EDIFACT segment is a composition, never a hand-written
  second mapping;
- **identity**: canonical order is `0..N`;
- **inverse**: exists exactly when the pivot is a bijection; selections
  (BAPI's 8 of 23) have none, and the type says so;
- **pivots never move data**: a sink reads `batch.edge_value(map[k], row)`
  over the rows the mask kept. The pivot is metadata, the mask is the
  population, the sink is the one materialiser.

Home: start inside `lance-graph-sap` (`pivot.rs`). Promote to
`lance-graph-contract` (next to `WideFieldMask`, which is the unordered
sibling of the same idea) when a second consumer needs it. `lgj`'s
projection order is the likely second consumer `[H]`.

## 4. Surface catalogue — emulated behaviour and docking point

| # | surface | populations (SoA) | emulated behaviour (pure rules + folds) | sink (simulation → docked) |
|---|---|---|---|---|
| S1 | **CATS / BAPI** | time entries | exists | `bapi_sink` records → RFC call to `BAPI_CATIMESHEETMGR_INSERT` |
| S2 | **IDoc / ALE** | control records (EDI_DC40), data records (EDI_DD40, one population per segment type), status records (EDIDS) — fk `DOCNUM`, parent `PSGNUM` | inbound processing as a rule producing a new version (status 64 → 53 / 51 with reason), partner profile check (WE20 as a resident plane `partner × message type × direction`), counts per status | flat-file IDoc (fixed-width records through pivot + width) → file port or tRFC port |
| S3 | **EDI (EDIFACT / X12)** | interchange, message, segment populations | conversion = pivot composition IDoc segment ↔ canonical ↔ EDIFACT segment; envelope (UNB/UNH/UNT/UNZ) counts as folds | segment text renderer → AS2/SFTP partner (Lobster/SEEBURGER-type converter replaced or fed) |
| S4 | **SCOT / SAPconnect** | send requests, recipients, status (SOST shape) | queue/status transitions as rules; recipient eligibility from `lance-graph-dir-sim` (already wired into the stalwart fork's `dirsim` recipient backend) | SMTP submission to the stalwart fork → real SAPconnect SMTP node pointing at that host |
| S5 | **xSuite inbound invoice** | mailbox items, stacks, documents (`0x080B`), extracted invoice header/lines | POBox routing, stack status, duplicate detection as folds; posting proposal as pivot onto the MM invoice BAPI or IDoc INVOIC | execution plan → `BAPI_INCOMINGINVOICE_CREATE` or INVOIC inbound |
| S6 | **SIMAF channels / DTO** | SIMAF DTO base / document / notification | HMAC-SHA512 parity profiles (exist for CATS in `edge.rs`) | Graph / SendGrid / SCOT channel per SIMAF |

Field lists for S2–S6 are deliberately not written here. Each one arrives
through its harvester with a source pin, as CATS did.

## 5. Simulation — the dir-sim shape, reused

`lance-graph-dir-sim` already proves the simulation shape `[G]`:

```
observed G0 ──rule──► G1 ──rule──► G2 ──validate──► desired ──diff(G0,G2)──► ExecutionPlan ──X
```

with OGAR owning meaning (`ogar-dir-sim`: `Change`, provenance, `Violation`,
`ExecutionPlan` whose operations each carry the `Precondition` that held in
the observed basis) and lance-graph owning execution (versions as
`Arc<Snapshot>` + delta overlay, invariants as Quack programs).

The SAP twin is the same split:

| directory (exists) | SAP (proposed) |
|---|---|
| `ogar-dir-core` (records, `Guid128`, `Dn128`) | `ogar-sap-core` — IDoc / CATS / SOST record shapes, numeric keys only |
| `ogar-dir-sim` (`Change`, `Violation`, `ExecutionPlan`, `Operation`) | `ogar-sap-sim` — `Operation::{CallFunction, PostIdoc, SetIdocStatus, SubmitMail, …}` with preconditions; semantic only, no endpoints, no credentials |
| `lance-graph-dir-sim` (own workspace, path-deps OGAR) | `lance-graph-sap-sim` (own workspace, path-deps OGAR + `lance-graph-sap`) |
| `lance-graph-quack` / `mask-risc` | unchanged |

`lance-graph-sap` stays OGAR-free, as it is today. The plan is the docking
contract: an actuator (RFC client, file port writer, SMTP submitter)
consumes `ExecutionPlan`, re-reads each precondition, then acts.

## 6. OGAR vocabulary `[S]`

A `SapPort: PortSpec` maps SAP public names onto existing canon concepts.
Candidates, every one of which resolves to an id already minted:

| SAP name | canon concept |
|---|---|
| `CATSDB`, `CATS` | `BILLABLE_WORK_ENTRY` `0x0103` (converges with SMB/WoA `Stundenzettel`, OpenProject `TimeEntry`, Odoo `account.analytic.line`) |
| `INVOIC`, `ORDERS`, `VBRK`, `VBAK` | `COMMERCIAL_DOCUMENT` `0x0202` |
| `E1EDP01`, `VBRP`, `VBAP` | `COMMERCIAL_LINE_ITEM` `0x0201` |
| `KNA1`, `LFA1` | `BILLING_PARTY` `0x0204` |
| `PA0001` / `PERNR` | `HR_EMPLOYEE` |
| SAPconnect send request | `EMAIL` `0x0B05` (shared with `SpearPort`) |
| archive / xSuite document | `DOCUMENT` `0x080B` |

Operator rulings this needs: the `APP_PREFIX` slot (`0x0000`–`0x000A` are
taken at the pin; `0x000B` is the next free value, allocation lives in
`APP-CLASS-CODEBOOK-LAYOUT.md` §2), whether EDIFACT gets its own port/skin,
and whether the IDoc control record (a transport envelope) is a concept at
all or stays edge metadata.

## 7. Substrate gaps — closed substrate-first (STOP rule)

| id | gap | where it closes |
|---|---|---|
| G1 | money: per-row amounts above `i32` (≈ 21.4 M in cents) and grouped sums. `GroupFold` at the pin has `Count`/`MinI32`/`MaxI32`/`SumSymI32`; CATS bounds hours inside `i32` per row and checks the total in `i64` | probe first: is a scaled split (hi/lo `i32` lanes, two folds, recombined at the sink) exact and cheap enough? If not, an `I64` lane fold lands in `mask-risc` on an `ndarray::simd` primitive |
| G2 | fixed-width text in sinks (IDoc flat file, EDIFACT) | sink-only, edge metadata; no execution change |
| G3 | `Pivot` as a shared type | `lance-graph-sap` first, `lance-graph-contract` on the second consumer |

## 8. Waves and gates

| wave | content | gate (red-then-green, measured) |
|---|---|---|
| W0 | read-only: inventory SIMAF / SIMAFPort (mail channels, inbound pickup, DTOs), re-verify the fold table in §3.2 against `mask-risc` by test, probe G1 | report with `file:line`; no code |
| W1 | `pivot.rs`; re-express `HASH_ORDINALS` / `BAPI_ORDINALS` / C# order as `Pivot`s | existing CATS tests unchanged and green; pivot law tests |
| W2 | `WireSchema` + `WireBatch<S>`; `CatsBatch` becomes `WireBatch<CatsSchema>` | `no_alloc`, `view_convergence`, `fold`, `binding` suites unchanged |
| W3 | IDoc: XSD harvest via `ogar-from-schema`, three populations with fk, flat-file sink | bind → sink round trip byte-identical on a real exported IDoc; allocation independent of row count |
| W4 | `ogar-sap-core`, `ogar-sap-sim`, `lance-graph-sap-sim`: status rules, validation, `ExecutionPlan` | dir-sim-style suites: version sharing, violations, plan order |
| W5 | S4 SCOT → stalwart (dir-sim recipients), S5 xSuite inbound | end-to-end in simulation: send request → plan → SMTP submission to the local stalwart; mailbox drop → posting proposal |
| W6 | S3 EDIFACT by pivot composition | INVOIC IDoc ↔ EDIFACT INVOIC round trip on a fixture pair |

Each wave lands with its board entry in the same commit, per the
lance-graph board rules.

## 9. The glove test for every SAP-side addition

Adapted from lgj's practical test: *would an SAP basis or ABAP person
recognise this from habit alone* — a function module name, an IDoc type, a
SCOT node, a mailbox? If yes, it belongs on the front. If it needs our
vocabulary (masks, lanes, folds) to be understood, it belongs behind the
front, in lance-graph.
