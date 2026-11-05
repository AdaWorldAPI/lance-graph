# D-DRDT-TILE-POC-1: Holographic ordinal tile routing, structural parity and lawful folds

**Status:** PROPOSED, plan only. **Date:** 2026-11-05. **Owner:** DR/DT64 and Bagop contracts in lance-graph; OGAR owns source identity correspondence; ndarray owns positional execution. No runtime admission or code change is implied.

## 1. The canonical picture

**SAP / PSCustomObject is a logical, normalized, one-key ordinal-addressed hashtable over typed SoA lanes.** It is not a physical hash table or one allocated object per row.

- **Key:** source-qualified \`Guid128\` (16 bytes) or another explicitly bound source identity. Never mistake an ephemeral row ordinal, classid or display label for that key.
- **Row coordinate:** a snapshot/frame-local ordinal that selects one row in an addressed SoA population.
- **Property coordinate:** a stable ordinal in the class's ordered field basis, resolved through \`ClassView\` and codebooks. \`WideFieldMask\` expresses structural presence and method footprints, not numeric business values.
- **Behavior coordinate:** the method/word identity of a DT64 binding; this is a pointer to an operation, not a materialized operation result.
- **Object:** a projected view of these coordinates and selected fields, materialized only for an explicitly demanded external result.

**DR is the Google-Maps teleportation metaphor:** route the point of computation through identity × property × predicate coordinates without moving source objects. **DT is the law preserving structural and behavioral parity:** bind the right \`ClassView\` basis, immutable behavior, signature, guards and effect order at the route version. **Quack lowers admissible calculations to mask-RISC; ndarray executes on borrowed numeric/positional lanes.** The DR route log is not an evaluator. The DT binding store is not a business-value database.

\`\`\`text
SAP / PSCustomObject logical tile
    scoped Guid128 → source population + local row coordinate
    classid → ClassView → field ordinal + WideFieldMask
    MethodRef / AlgebraRef → operation identity
                  │
            DR coordinate route
                  │
       DT basis/behavior parity check
                  │
        Quack → mask-RISC → ndarray
                  │
    scalar / mask / joined property projection at demand
\`\`\`

This is a **holographic representation/interpretation of a Bag**, not the definition of DR itself. DR also serves non-Bag address spaces. DR reservations of Cartesian **coordinates** are not allocations of dense Cartesian products, nor guarantees that every join/fold is legal or implemented.

## 2. Verified substrates and exact boundaries

| Existing contract | Source | Consequence |
|---|---|---|
| Canonical \`NodeRow\` ABI | [canonical_node.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/canonical_node.rs) | \`key[16] + edges[16] + value[480] = 512\` bytes, aligned to 64. An external \`Guid128\` is not the canonical minted \`NodeGuid\` key; preserve it without truncation through typed source/identity lanes. |
| ClassView / WideFieldMask | [class_view.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/class_view.rs) | Late semantic projection over ordered field positions. \`FieldMask\` is presence-only; \`WideFieldMask\` widens the basis without moving existing bits. |
| DR routing | [dt64/dr.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dt64/src/dr.rs) | Identity × Property × Predicate extents, DR64/DR256 reservation, position/class routes and \`RouteVersion\`. It stores no value or method meaning. |
| DT binding | [dt64/lib.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dt64/src/lib.rs) | \`ClassSchema\`, \`MethodRef\`, \`MethodId\`, \`SigRev\`, \`SchemaRev\`, guard and ordered effects; behavior immutable per classid; derive + reroute for changes; routing replay is versioned. |
| Query execution | [quack/lib.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-quack/src/lib.rs) | Quack *lowers* into mask-RISC; the consumer executes; ndarray provides SIMD/positional mechanics. No second expression evaluator. |
| SAP tile precedent | [lance-graph-sap/query.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-sap/src/query.rs) | CATS typed columns, cold bindings, Quack grouped fold without constructing per-row DTOs. |
| SoA directory precedent | [dir-sim/snapshot.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dir-sim/src/snapshot.rs) | Distinct \`Guid128\`-sorted user/group SoA populations with snapshot-local \`u16\` ordinals, masks and cold dictionaries. |
| Hybrid correspondence | [OGAR correspond.rs](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-dir-core/src/correspond.rs) | Separate AD / Entra / EXO families; scope-qualified \`Guid128\` identity witnesses; cold sorted \`(scope, u128 key, row)\` index; no hot string work. |
| Identity decoding | [OGAR identity.rs](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-dir-core/src/identity.rs) / [OGAR ogar-ad](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-ad/src/lib.rs) | \`User_{GUID}\` / \`Group_{GUID}\` backsync labels are decoded **once** into canonical GUID bytes, with source-kind and encoding validation. |

\`Dn128\` is a 16-byte **hierarchy coordinate** under a directory scope, not the 16-byte **source identity** \`Guid128\`. Byte-width equality is not semantic equality. An AD \`objectGUID\` and Entra object \`id\` need not match: sourceAnchor → immutableId is an independent witness; AD \`msDS-ExternalDirectoryObjectId\` backsync → Entra \`id\` → EXO \`ExternalDirectoryObjectId\` is the common cloud identity when those observations are available and consistent.

## 3. Smallest useful proof: chained late materialization

\`\`\`text
AD tile:   objectGUID, sourceAnchor, msDS-ExternalDirectoryObjectId = User_{cloudGUID}
                             │  decode at ingress; qualified GUID identity
Entra tile: id = cloudGUID, onPremisesImmutableId, onPremises* fields
                             │
EXO tile:   ExternalDirectoryObjectId = cloudGUID, ExchangeGuid, mailbox fields
                             │
        cold correspondence, conflict/multiplicity/scope checks
                             │
     DR-addressed coordinate descriptor: (AD row, Entra row, EXO row)
                             │
          DT method/basis and layout/version witnesses
                             │
        late field selection → admitted Quack fold / ndarray reads
                             │
          virtual SAP / PSCustomObject projection
\`\`\`

The **operation pointer** (for example an XVERWEIS-like lookup method) is a \`MethodRef\` / polyfill reference: it names *what to execute*. A resolved ordinal coordinate, if produced, is an **execution result** and may be retained as a version-witnessed descriptor. The eventual property value is a later read, not a mandatory result of the lookup. Ordinary Excel XVERWEIS semantics are a consumer adapter, **not** the definition of the operation pointer or the DR map.

A resolved coordinate chain can avoid repeated GUID matching and per-row object hydration. This does **not** automatically admit arbitrary cross-table lane-derived Gather. The existing OGAR correspondence builds cold sorted indexes; \`ndarray/src/simd_stride.rs\` provides positional access forms (\`lut_u8\`, \`lut9\`), not a shipped general-purpose GUID join. The implementation must either use already lawful co-addressed views/borrowed lanes or obtain the missing owner admission with an independent oracle. Keep OP-14 / NotASide / witness-derived-mask and ADMISSION B restrictions explicit.

## 4. Immutability and replay obligations

1. **DT behavior parity:** same \`classid\`, field basis, \`MethodId\`, \`SigRev\`, guard and ordered effects means the same method meaning; changed behavior requires a derived classid and reroute. Unknown/foreign/mismatched binding **refuses**, never maps to an empty success.
2. **DR historical routing:** \`RouteVersion\` makes past class/position routing queryable. A route cannot alone reproduce data from an old time.
3. **Data replay requires data:** preserve source observation/snapshot identity, value revision, layout digest/frame, source scope, codebook/schema version and any external-effect evidence. No claim of full deterministic historical output without those inputs.
4. **Coordinate lifetime:** global GUID identity survives reordered rows; local ordinals do not. A permuted tile demands rebinding or a matching layout witness; NEVER silently reuse stale ordinal paths.
5. **One authority:** codebook vocabulary and canonical source identities are not to be reminted by Excel, reporting or SAP adapters. Derived lookup indexes are disposable performance structures, not alternate identity stores.

## 5. Independent falsifier (lab before promotion)

Use one synthetic AD/Entra/EXO trio plus a CATS-style property tile. Include: valid backsync \`User_{GUID}\`, a separately observed source-anchor witness, mismatched anchor, ambiguous duplicate cloud id, absent backsync, foreign tenant with colliding visible labels, malformed GUID, conflicting \`ExchangeGuid\` and changed snapshot/layout. Verify cold strict decoding and compare 128-bit bytes, not strings. The independent oracle resolves identity by scope and exact GUID, not by candidate local ordinals.

- **F1:** same source GUIDs under reordered ordinals yield equal semantic values after witness-checked rebind; stale references refuse.
- **F2:** distinct foreign scopes never collapse, even if cloud GUID bytes or display values collide.
- **F3:** conflicting/ambiguous AD↔Entra↔EXO correspondence remains explicit, never an implicit first-match.
- **F4:** one typed SAP property fold executes over borrowed SoA lanes, preserving null/presence/multiplicity laws; no per-row PSCustomObject, joined DTO or string lookup in the hot path.
- **F5:** immutable DT method behavior and ordered effects survive an unrelated route; derived class + reroute changes the interpretation only at the new route version.
- **F6:** a projected value can be read through a bound coordinate only with accepted scope, data snapshot and layout witnesses; disable each witness to demonstrate a wrong answer/refusal.
- **F7:** count separately: ingress decode, cold index, DR binding, witness checks, warmed reads, folded result and egress bytes. Do not claim speedup from an isolated warm read.

**STOP conditions:** if a proposed hop needs an unadmitted Gather, computed-mask reuse, alternate join authority, missing DRISA M1/M2 layout witness or D4/S-INGRESS write authority, keep it LAB/NOT ADMITTED and say which gate is missing. Read-only demonstrators may use existing lawful paths without claiming production admission.

## 6. Program boundaries and status

| Plan | PR | Meaning for this POC |
|---|---|---|
| P0 this plan | [lance-graph #1960](https://github.com/AdaWorldAPI/lance-graph/pull/1960) | Architectural law, source-backed minimal proof |
| P1 positional mechanics | [ndarray #408](https://github.com/AdaWorldAPI/ndarray/pull/408) | Lawful 16-byte/position reads; no implied GUID join |
| P2 method-pointer polyfill | [lance-graph #1961](https://github.com/AdaWorldAPI/lance-graph/pull/1961) | Optional operation-dispatch slice only if needed |
| P3 GridCraft | [gridcraft #1](https://github.com/AdaWorldAPI/gridcraft/pull/1) | Deferred spreadsheet-shaped consumer |
| P4 Java folding evidence | [lance-graph-java #93](https://github.com/AdaWorldAPI/lance-graph-java/pull/93) | Non-blocking evidence only |
| P5 report-rs | [report-rs #34](https://github.com/AdaWorldAPI/report-rs/pull/34) | Deferred reporting adapter |
| P6 Excel integration falsifiers | [gridcraft #2](https://github.com/AdaWorldAPI/gridcraft/pull/2) | Deferred Excel parity tests; does not gate core POC |
| P7 F### formula consumer | [gridcraft #3](https://github.com/AdaWorldAPI/gridcraft/pull/3) | Deferred formula binding; DT64 behavior-lens law reused |

**Critical path:** existing OGAR correspondence + lance-graph-dir-sim + lance-graph-sap + DR/DT binding + one admitted Quack/ndarray fold. **Not critical:** Excel/XLSX, GridCraft rewrite, DataFusion/Databricks, Java, Codegraph, all-formula support, persistence or SharePoint synchronization.

This is a documentation proposal only. No production code, merge, new semantic authority, or automatic owner approval.
