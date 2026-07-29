# copy-cognitive — `derive(Clone, Copy)` verdicts for cognitive / callcenter / ontology

Run date: 2026-07-29. Branch `claude/x265-x266-plans-review-h9osnl`.
Operator order: *"copies are forbidden, borrows are only for the same mailbox"*;
*"only cognitive achievements > tenant"*.
Mandatory reads done: `zero-copy-lens-law.md`, `data-flow.md` §2 (via ndarray
CLAUDE.md), `borrow-strategy.md` (q2 rules). Census:
`.claude/board/exec-runs/copy-derive-blast-radius.txt`.

EDIT ONLY — no cargo run. 2 derives changed, both zero-cascade.

## Headline

**Not one `Copy` type in the three crates carries a borrow.** Zero lifetime
parameters across all 119 `Copy` sites. The four Tier-A callcenter entries the
census flagged (`FamilyEntry`, `OwlPivot`, `MetaAnchors`, `SuperDomainEntry`)
were flagged on `&'static`, which is the *opposite* of the violation — see
§"The `'static` verdict".

The two VIOLATIONs found are a different shape than the census heuristic
looked for: **mutable state records whose `Copy` forks a single-writer
invariant.** Both were invisible to a lifetime-or-`&` grep.

## ⚠ The census is INCOMPLETE for these three crates

`copy-derive-blast-radius.txt` claims *"matches both orderings"*. It does not.
It matched `derive(Clone, Copy…` only; every `derive(Debug, Clone, Copy…)`
site is absent. Measured:

| crate | in census | actual `Copy` derives | missing |
|---|---:|---:|---:|
| lance-graph-cognitive | 23 | 44 | 21 |
| lance-graph-callcenter | 22 | 30 | 8 |
| lance-graph-ontology | 11 | 45 | 34 |
| **total (these 3)** | **56** | **119** | **63** |

Whole files are missing: `cognitive/fabric/{zero_copy,firefly_frame,gel,scheduler}.rs`,
`cognitive/spo/gestalt.rs`, `cognitive/search/{causal,certificate,hdr_cascade}.rs`,
`callcenter/{policy,rls,savant_reasoners,audit}.rs`,
`callcenter/transcode/{cam_pq_decode,spo_filter}.rs`,
`ontology/odoo_blueprint/**` (~30 sites), `ontology/soa_bake/mod.rs`.
**The global 369 figure is therefore a floor, not a count.** I screened all 63
extras here myself (none carry a borrow); the other crates' extras have not
been screened by anyone.

Reproduce: `grep -rn "Copy" --include=*.rs -A2 <crate>/src | grep derive`.

## Changes made (2)

### 1. VIOLATION — `ChunkHeader`, `lance-graph-cognitive/src/core_full/scent.rs:24`

`#[derive(Clone, Copy, Debug)]` → `#[derive(Clone, Debug)]`.

A record OF the substrate, not a value. `ScentIndexL1` **owns**
`Box<[ChunkHeader; BUCKETS]>` and **mutates it in place** — `on_append` writes
`scent`/`count`/`last_access`, `set_decision` and `set_plasticity` write their
fields. `Copy` mints a second, immediately-stale reading of live bucket state,
and `scent` is itself a lossy projection of fingerprints that already live in
the data file (a second stored reading of bytes that already have one).

Cascade: **none.** Every read path is already `&`-borrowed (`headers.iter()`,
`&mut self.headers[i]`); construction is `std::array::from_fn` (no `Copy`
needed — a `[expr; N]` repeat literal would have required it, and none exists);
`write_headers`/`read_headers` take `&`/`&mut` slices; `ScentIndexL2` composes
`ScentIndexL1` by value. `ChunkHeader` appears in **no other file in the
workspace** (grepped). `Clone` retained.

### 2. VIOLATION — `AuditChain`, `lance-graph-callcenter/src/unified_audit.rs:196`

`#[derive(Clone, Copy, Debug)]` → `#[derive(Clone, Debug)]`.

The strongest finding of the run. `AuditChain` is single-writer chain state
(`advance(&mut self)` writes `self.last_root`), and `last_root` is a second
holding of a root **already durably recorded** on the last emitted event. A
`Copy` silently forks the chain: two advancers stamping distinct events as
successors of the same `prev_merkle` — which is precisely the tamper signature
`verify_chain` exists to detect, minted by the type system with nothing in a
diff to point at. §13.4's cross-domain unlinkability rests on one salt, one
root, one writer.

The correct shape is already present structurally and the derive was
undermining it: `UnifiedBridge` holds `Mutex<AuditChain>` (one-writer mailbox)
and the sanctioned way to continue a chain elsewhere is
`AuditChain::resume(.., last_root)`, which is *explicit about which root it
claims*.

Cascade: **none.** All uses are `AuditChain::new` / `::resume` /
`Mutex::new` / `&mut` `advance` through the lock; `audit_root()` copies out
`last_root` (an `AuditMerkleRoot` u64 — untouched). `Clone` retained
deliberately: it forks too, but an explicit `.clone()` is greppable and
nothing calls it today. **Follow-up worth an operator call: `Clone` on a
merkle advancer is arguably also forbidden** — I did not remove it because
that exceeds the stated scope (the census targets `Copy`).

## The `'static` verdict (why the 4 Tier-A callcenter entries are NOT violations)

`FamilyEntry` (`family_table.rs:132`), `OwlPivot` (`odoo_alignment.rs:65`),
`MetaAnchors` (`super_domain.rs:125`), `SuperDomainEntry` (`super_domain.rs:181`)
were flagged for `&'static str` / `&'static [u8]` / `&'static [OgitFamily]`.

**A `'static` borrow has no mailbox to escape.** It borrows immutable rodata
that outlives every compartment and has no writer, so it can neither dangle
nor drift. And `Copy` on a struct of `&'static` fields duplicates **pointers,
not content** — the grey/white fence in `zero-copy-lens-law.md`: *"cross-tenant
pointers are legitimate; cross-tenant values are not."* These rows are the
already-correct shape; forcing them to own their strings would be the
violation. Same verdict covers ontology's ~30 `odoo_blueprint` baked rows —
`OdooEntityPairing` holds `&'static OdooEntity`, the reference-not-copy shape
done right.

## ELEVATED — the audit-event question, answered

The operator asked whether `UnifiedAuditEvent` / `AuditChain` /
`AuditMerkleRoot` are higher-rung achievements or copies of something already
recorded. They split three ways, which is the interesting part:

- **`AuditMerkleRoot` (`:94`) — ELEVATED.** `chain(prev_root, salt, bytes)` is
  a **cross-term**: a computation across multiple reads yielding a value of a
  different KIND — a fact about the *sequence*, not a member of it. It is
  reproducible by no cast from any lane. This is exactly the Gadamer-refined
  test in `zero-copy-lens-law.md` and the shape of the shipped
  `Locus::Quorum` / `Contradiction` precedent. Rung: inputs are rung 0–1
  observation (one authorize() decision); output is a chain-integrity witness
  about the history of decisions. Storing is legitimate — that IS the
  calcification. `Copy` kept (a `u64`).
- **`UnifiedAuditEvent` (`:135`) — ELEVATED, keep `Copy`.** Its input fields
  (tenant/owl/op/role-hash) do each exist elsewhere, so field-by-field it
  looks like a copy — but the event *as emitted* carries `merkle_root` +
  `prev_merkle`, so the row is the elevation, not a gathered view of the
  request. It is **immutable once stamped**, so a copy cannot drift; 42 bytes
  of scalars, no borrow. The doc-comment's stated reason for hashing the role
  rather than storing `&'static str` ("so the event is `Copy` + fixed-size")
  is a durability argument, and it holds.
- **`AuditChain` (`:196`) — VIOLATION.** Not the witness; the *advancer*. See
  change 2. The distinction that matters: **the witness may be copied because
  it is finished; the advancer may not because it is still being written.**

## Full verdict table

`L` = LEGITIMATE (owned value microcopy, data-flow.md §2 REQUIRES `Copy`),
`E` = ELEVATED, `V` = VIOLATION (edited).

### lance-graph-cognitive

| path:line | type | V | reason |
|---|---|---|---|
| container_bs/adjacency.rs:33 | `PackedDn` | L | newtype over `u64`; a hierarchical address = white matter (a displacement, not content). Watch-listed as "record OF substrate" — it is not: no backing store exists that it projects; the packed `u64` IS the address. |
| container_bs/adjacency.rs:232 | `InlineEdge` | L | 2 bytes (`verb`,`target_hint`), produced by `unpack(u16)` and consumed immediately; never gathered (`grep` finds no `Vec<InlineEdge>` / `[InlineEdge; N]`). The stored form is the `u16` in container words 16-31; this is a transient decode, not a second store. |
| container_bs/adjacency.rs:282 | `EdgeDescriptor` | L | newtype over `u64`, same argument. The zero-copy views in this file are the separate `*View<'a>` types (lines 336-606) — correctly **not** `Copy`. |
| core_full/index.rs:74 | `Key` | L | `#[repr(transparent)] u64`. |
| core_full/index.rs:134 | `Entry` | L | 3×`u64` = `(prefix, offset, target)`: pure displacements, write-once, read via `&Entry` iterators. White matter. |
| core_full/scent.rs:24 | `ChunkHeader` | **V** | mutable owned substrate state + a content projection (`scent`). **Edited.** |
| core_full/scent.rs:480 | `BucketAddr` | L | ≤3-byte address enum (`L1/L2/L3`). |
| fabric/subsystem.rs:4 | `Subsystem` | L | fieldless enum. |
| grammar/causality.rs:37 | `DependencyType` | L | fieldless enum. |
| search/cognitive.rs:63 | `QualiaVector` | L | 8×`f32` + `RelevanceScores`; the `QualiaColumn` read of the AGI-as-glove doctrine, passed by value. Textbook data-flow §2. |
| search/cognitive.rs:280 | `SearchVia` | L | fieldless enum. |
| search/cognitive.rs:309 | `RelevanceScores` | L | 5×`f32` score record. |
| spo/cognitive_codebook.rs:22,70,95,112,129,148,165,190,215 | `CognitiveDomain`, `NsmCategory`, `QualiaChannel`, `NarsCopula`, `NarsInference`, `CausalityType`, `TemporalRelation`, `YamlTemplate`, `ThematicRole` | L | 9 fieldless codebook enums — the register (I-VSA-IDENTITIES Test 0). |
| spo/cognitive_codebook.rs:248 | `CognitiveAddress` | L | packed `u64` address. |
| spo/sentence_crystal.rs:85 | `Coord5D` | L | 5×`usize` grid coordinate. |
| *(21 uncensused)* fabric/{zero_copy,firefly_frame,gel,scheduler}.rs, spo/gestalt.rs, search/{causal,certificate,hdr_cascade}.rs | `AddrRef`, `EdgeRef`, `FrameHeader`, `ConditionFlags`, `Instruction`, `LanguagePrefix`, `ExecutionContext`, `Location`, `MexicanHat`, `AntialiasedSigma`, `TiltReport`, `GestaltState`, … | L | screened: all scalar/address/enum, no lifetimes, no `&` fields. `zero_copy.rs`'s `EdgeRef` = 2×`AddrRef` + `u32`; the borrowing type there (`ZeroCopyExecutor<'a>`) is correctly not `Copy`. |

### lance-graph-callcenter

| path:line | type | V | reason |
|---|---|---|---|
| family_table.rs:132 | `FamilyEntry` | L | `&'static` baked row — see §"The `'static` verdict". |
| odoo_alignment.rs:65 | `OwlPivot` | L | same. |
| super_domain.rs:125 | `MetaAnchors` | L | same (`Option<&'static str>` ×2). |
| super_domain.rs:181 | `SuperDomainEntry` | L | same (`&'static [OgitFamily]`). |
| audit_sink/composite.rs:16 | `FanoutMode` | L | fieldless enum. |
| audit_sink/mod.rs:63 | `NoopAuditSink` | L | ZST. |
| dn_path.rs:18 | `DnPath` | L | 6×`u64` segment hashes = an address (heel/hip/branch/twig/leaf). |
| family_table.rs:62 | `OwlCharacteristics` | L | `#[repr(transparent)] u8` bitfield. |
| family_table.rs:181 | `PerFamilyCodebook` | L | ZST placeholder. |
| lance_membrane.rs:117 | `ActorState` | L | 3 scalars; its doc says it exists to be an **atomic snapshot under one lock** defeating the F-01 identity-tear race — i.e. the owned-microcopy pattern applied deliberately, and `Copy` is what makes the tear-free read cheap. |
| super_domain.rs:45,109,153 | `SuperDomain`, `DolceMarker`, `ComplianceRegime` | L | `#[repr(u8)]` fieldless enums. |
| transcode/parallelbetrieb.rs:71 | `DriftKind` | L | fieldless enum. |
| transcode/zerocopy.rs:48 | `ArrowTypeCode` | L | enum, one `usize` payload variant. |
| unified_audit.rs:51,71 | `AuthOp`, `AuthDecision` | L | `#[repr(u8)]` fieldless enums. |
| unified_audit.rs:94 | `AuditMerkleRoot` | **E** | the cross-term. See §ELEVATED. |
| unified_audit.rs:135 | `UnifiedAuditEvent` | **E** | finished witness, immutable once stamped. See §ELEVATED. |
| unified_audit.rs:196 | `AuditChain` | **V** | single-writer advancer; `Copy` forks the merkle chain. **Edited.** |
| unified_bridge.rs:75,110,169 | `OgitFamily`, `OwlIdentity`, `TenantId` | L | 1/3/4-byte identity addresses; `OgitFamily`'s own doc: *"Pure address. No reasoning, no string lookup."* |
| *(8 uncensused)* policy.rs, rls.rs, savant_reasoners.rs, audit.rs, transcode/{cam_pq_decode,spo_filter}.rs | `PolicyKind`, `Op`, `RedactionMode`, `DpMechanism`, `RegistryMode`, `SavantError`, `StatementKind`, `PassthroughDecoder` | L | screened: enums + a ZST decoder. |

### lance-graph-ontology

| path:line | type | V | reason |
|---|---|---|---|
| bridge.rs:144 | `EntityRef` | L | despite the name, carries **no borrow** — wraps `SchemaPtr` (2×`u32`). A pointer by value. |
| bridge.rs:149 | `EdgeRef` | L | same. |
| hydrators/dolce_odoo.rs:32 | `DolceCategory` | L | fieldless enum. |
| hydrators/owl.rs:139 | `Format` | L | fieldless enum. |
| namespace.rs:26 | `NamespaceId` | L | `u8` newtype. |
| namespace.rs:124 | `SchemaPtr` | L | packed `u32` + context `u32`; the canonical address. Note `with_context_id` returns a new value — builder, not a compute path. |
| namespace.rs:196 | `SchemaKind` | L | `#[repr(u8)]` enum. |
| proposal.rs:114 | `IdentityCodec` | L *(with a note)* | 19 bytes of owned scalars, no borrow → `Copy` is not the violation. **But** structurally it is 4 lossy readings (`cam_pq_code`, `base17_head`, `palette_key`, `scent`) of a fingerprint whose warm form its own doc says *"stays on `BindSpace`"* — the "second stored projection" silhouette. That is a **storage-design** question (the HHTL codec ladder needs all rungs resident to skip), not something removing a derive fixes. Reported, not edited; flagging it here so it is not rediscovered as a derive problem. |
| proposal.rs:124 | `QualiaMeta` | L *(with a note)* | 80 bytes of scalars, transient dispatch bundle → derive is fine. Note for the record: it bundles three of the four SoA axes (`qualia`/`meta`/`edge`), which the AGI-as-glove doctrine warns against wrapping ("breaks the SIMD sweep"). Again a shape question, not a `Copy` question. |
| proposal.rs:213 | `MappingHandle` | L | receipt = `(SchemaPtr, row_index)`; addresses. |
| ttl_parse.rs:560 | `SubjectKind` | L | fieldless enum. |
| *(34 uncensused)* odoo_blueprint/** (~30), soa_bake/mod.rs (3), … | `OdooEntity`, `OdooField`, `OdooMethod`, `OdooStateMachine`, `OdooEntityPairing`, `StructuralSignature`, `SchemaVersion`, `EdgePair`, … | L | screened: all `&'static` baked blueprint rows (see §`'static`) plus scalar signatures. `OdooEntityPairing` holds `&'static OdooEntity` — reference-not-copy, the shape we want. |

## Cascades refused

**None.** Both edits are confined to their declaring file; no caller relied on
`Copy` for either type. Nothing was left unremoved for cascade reasons.

The one thing I deliberately did **not** do: remove `Clone` from `AuditChain`
(see change 2). Scope call, flagged for the operator rather than taken
unilaterally.

## Gates

Not run — brief is EDIT ONLY; orchestrator compiles centrally. Expected
surface: `cargo test -p lance-graph-cognitive`, `-p lance-graph-callcenter`.
Both edits remove a trait impl, so any breakage would appear as
`error[E0507]: cannot move out of ...` / `use of moved value` at a call site —
I found no such site by grep, but the compiler is the authority.
