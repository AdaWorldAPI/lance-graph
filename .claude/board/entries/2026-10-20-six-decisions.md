# 2026-10-20 — ADR-2026-10-20-SIX: ADMISSION, D-RA-2, WIRE-50, ACTION-GATE, FLOAT-REFS, O30

**Status:** DECISION (2026-10-20). **Basis:** the operator's rulings in session
`session_013z2e1YXkm6o6HNrDn5BPWA`, given on the options in
`.claude/research/2026-10-recovery/LEDGER-REVISION-2026-10-20d.md` §6. The
qualifications below are part of each decision, not suggestions.

**Principle (operator's words):** "Preparation may establish coordinates.
Execution operates on co-addressed sets. Neither phase gets to conceal an
unaccounted relational gather." Representation, permission and evidence stay
distinct: none of them acquires meaning because a bit pattern or a missing
field happens to permit it. The six rulings below share one constraint: "A
value cannot acquire authority merely by being representable."

| decision | ruling |
|---|---|
| ADMISSION | B: same-table, version-witnessed seeks; no implicit move to C |
| D-RA-2 | A: the contract owns the single-table set-algebra rewrites |
| WIRE-50 | A: pyramid order; meaningful only under a presence condition |
| ACTION-GATE | A: `commit_via` is canonical; fail closed |
| FLOAT-REFS | B: `fold` off by default; this is isolation, not reproducibility |
| O30 | A: bounded source identities, at most 32 per derived cell at first |

## 1. ADMISSION — option B

**DECISION** (operator's words): "A query-preparation phase may perform
counted, version-witnessed, same-table key-range seeks over an attested sorted
key. It may construct a projection for subsequent co-addressed evaluation. A
position derived from another relational column or table is not admissible as
a read address merely because the lookup occurs during preparation. Frozen
codebook decoding is admissible as representation translation, provided its
mapping is immutable within the attested version, its cost is accounted for,
and it does not become a relational lookup mechanism. Execution remains
co-addressed."

- **Accounting:** report `attest_cost`, `projection_build_cost` and
  `query_read_cost` separately. Arm B's one-time attestation (a full scan of
  both key columns, 7–10 ms for 100,002 keys) must not disappear behind its
  per-query count (about 3,000 rows).
- **D-MPS-8 stays conditionally blocked.** #1621's "address translation only
  in preparation" does not authorize cross-table addressing. If it turns an
  external row's value into a read position in the relationship table, it is
  option C whatever the phase. Rewritten as a sequential, witnessed
  relationship projection it could come under B.
- **`graph_hop.rs`** (`OutgoingHop`, #1616) stays quarantined.
- `Filter::Semijoin`, `Pred::EqU32Via` and `MaskOp::Gather` need call-site
  classification before any removal. `seek_from` and `gallop_partition_point`
  are admissible only for attested same-table seeks.
- **SCOPE:** every preparation-time read in lance-graph, Quack, mask-risc,
  relalg and the probes. Closes the ruling half of GATHER-DECISION;
  ISS-ADMISSION-PREP-BINDING and ISS-GATHER-HOP-1616 carry the status.
- **REVISIT WHEN:** a measured cross-table preparation operation shows a
  benefit that cannot be had by materializing a co-addressed projection. That
  would justify evaluating C, never implementing it silently.

## 2. D-RA-2 — option A

**DECISION:** `lance-graph-contract` owns `Runs`, `Res`, the algebra laws and
`normalize`; relalg, Quack and crepe / R2IL consume and lower. "The contract
owns mathematical truth; consumers own execution strategy." Associative
flattening, idempotence, absorption and other valid single-table identities
have exactly one implementation.

- relalg keeps relational planning, joins and lowering. Datalog values stay in
  `metta::datalog` (council v3).
- The slot budget is defined on **normalized** structure, not parser nesting:
  `Or(a, Or(b, c))` flattens before the admission check, and normalization must
  not expand expressions exponentially to remove nesting.
- **Required tests:** differential tests against materialized bitsets;
  relalg's expectations unchanged; complement and universe scope handled
  explicitly before any negation identity is admitted.
- **Order:** D-CMR-1 (types and laws) → port the existing rules one at a time
  → D-CMR-6 publishes the catalogue → repoint crepe. This is the first
  implementation PR.
- **REVISIT WHEN:** not stated.

## 3. WIRE-50 — option A, with presence outside the field

**DECISION:** bits 50..52 use the pyramid order, 7 = Astonishment … 0 =
Wisdom. Three bits hold eight values; eight stages plus "not on the ladder" are
nine states, so the field cannot carry both. The bits encode a maturation
stage **only when a separate presence condition is true**.

- At the typed boundary this is `Option<Maturation3>`.
- The presence condition lives in a versioned envelope, a validity sidecar or
  a separately ruled field. It must not borrow another reserved CE64 bit.
- A raw zero never establishes Wisdom for an uninitialized or legacy edge.
- **Migration PR:** the shared table for bits 40..63, field-isolation tests,
  explicit old-layout decoding, `wire.rs` migrated, coordinated tests for
  `learn`, `MooreNars16` and `CausalEdgeV3`. `isa.rs`'s old reading of 46..49
  goes behind ISA-LEGACY. D-CGW-9 is corrected in the same ledger pass (done in
  this commit).
- No serialized meaning changes before the presence contract exists.
- **REVISIT WHEN:** the storage layout offers a better lossless location for
  maturation presence.

## 4. ACTION-GATE — option A, fail closed

**DECISION:** `commit_via` is the only authoritative commit path.
`required_role = None` no longer means everyone is authorized; it means the
definition is incomplete unless it is declared public.

- **Invariant on `ActionDef`:**
  - `Public`: no role grant needed, but definition, state and impact checks
    still apply.
  - `Restricted(role)`: needs the role and registry authorization.
  - Missing or contradictory authorization metadata: refused.

  An enum or a validated marker; absence is never permission.
- **Migration:** deprecating `commit` while its weaker check stays callable is
  not enough. It either delegates through `commit_via` or becomes a
  feature-gated legacy API that fails closed without a grant registry.
- MedCare: review the existing roleless definitions before migrating, but do
  not keep open-by-default behaviour for compatibility.
- **Tests:** inventory every `ActionDef` and every `commit` caller; then show
  that unauthorized execution cannot succeed through either entry point.
- **REVISIT WHEN:** a consumer needs a different authorization policy model,
  not an easier bypass.

## 5. FLOAT-REFS — option B, as isolation

**DECISION:** `fold` becomes an optional feature, off by default, so the core
default build no longer follows OGAR `main` (`lance-graph-r2il/Cargo.toml:19-20`).

- This is dependency **isolation**. A build with `fold` on can still follow
  moving branches.
- Reproducible deployment is a separate contract: a committed, resolved
  lockfile for the actual application build, tested with `--locked`. No fixed
  git revisions go into internal manifests.
- Lab probes stay outside the default surface.
- Test core with both feature configurations and the `fold_path` fixture.
- **Note for the implementer:** lance-graph tracks no `Cargo.lock` by project
  rule (`.gitignore`, ISS-STALE-AUTHORITY-LOCKS). So the committed lockfile
  belongs to the deploying application's build, not to this library
  workspace. Which build carries it is OPEN.
- **REVISIT WHEN:** the fold surface is ready to be a supported default, with
  stable dependency compatibility tests.

## 6. O30 — option A, 32 sources at first

**DECISION:** at most 32 distinct direct evidence sources per derived cell in
the first implementation. This is a provisional engineering capacity, not a
property of CE64, and not tied to the 64 thought slots.

- **Identity:** `SourceRef(table_id, version_witness, lane, row_position)`. A
  bare `(lane, row_position)` is insufficient across table versions or layout
  changes.
- **Storage:** a cold, sparse sidecar indexed by the derived cell, never CE64
  hot-path bits.
- **Handling:**
  - canonicalize and deduplicate;
  - on overflow, refuse explicitly and keep the previous committed state;
  - no modulo folding, no silent truncation, no invented overlap.
- Direct support and transitive evidence are kept distinct; a later
  derivation DAG may reach many underlying sources without flattening them
  into every cell.
- A per-cell source list does not by itself give selective retraction.
  D-RPR-2 needs a versioned reverse-dependency index or an equivalent
  invalidation plan.
- **Tests first:**
  - the modulo-64 stamp overlap (C37);
  - AriGraph's revision with no source record;
  - 32 and 33 distinct sources;
  - identical row positions in different table versions.
- **REVISIT WHEN:** measured source cardinalities produce meaningful refusals,
  or a provenance DAG shows a better bound.

## Execution order

1. **Ledger and documentation.** This commit records the six decisions and
   updates the two admission issues. It corrects D-CGW-9, D-CMR-2 and D-CGW-0,
   and records D-RA-2, D-CMR-1, D-HL-1 O2 and D-MATURATION3-1. It also adds the
   two missing agent card headers and regenerates the agents README index.
2. **First architecture PR:** D-CMR-1 / D-RA-2.
3. **Independent:** the admission call-site classification and named legacy
   gates; ACTION-GATE with no permissive alternate entry point.
4. **Independent:** FLOAT-REFS feature isolation; WIRE-50 with the presence
   contract before any serialized meaning changes.
5. **Provenance:** O30 source identities, bounded cold storage and the
   failing fixtures; then D-TGA-3 and D-RPR-2.

OPEN: where the deploying application's lockfile lives (FLOAT-REFS); where
WIRE-50's presence condition lives (envelope, sidecar or a separately ruled
field); the classification of each gather call site.
