# medcare-consumer-pull-thinking-proof-v1 — one real medical thought through the already-live OGAR consumer path

> **Status:** ACTIVE (consolidation + proof target). **Date:** 2026-08-02.
> **Supersedes as the single active MedCare plan:** the host-side
> manifest/supervisor/actor revival implied by `pr-e-1-manifest-modules.md`,
> `pr-g1-manifest-modules.md`, `pr-g2-ractor-supervisor.md §8` (those specs stay
> as historical records; see §3). Grounded in the 2026-08-02 read-only
> investigation (six-agent sweep + unshallowed git history, HEAD `71d1db1`).
> **Separation of concerns:** lance-graph is PUBLIC; MedCare-rs is PRIVATE. This
> plan names consumer crates/tests structurally only — no clinical schema
> detail, no sourcing/licensing reasoning, ever (MedCare-rs commitment #9).

## 1. Canonical classification (the investigation's verdict, pinned)

There are **two MedCare integration lineages, not one partially-assembled path**:

| Surface | Classification | Evidence |
|---|---|---|
| Bridge migration (`MedcareBridge = UnifiedBridge<HealthcarePort>`) | **COMPLETED** — deprecated compatibility alias only | commit `ddb6c840` (2026-06-21) collapsed the last bespoke per-tenant bridge; `#[deprecated]` landed `10e717d4` (2026-06-22) |
| Healthcare codebook promotion (`0x0901–0x090C`, 7 port aliases) | **COMPLETED** — canon in OGAR `ogar_vocab`; `contract::ogar_codebook` is a **verified wire-compatible mirror** (checked slot-for-slot this pass) | OGAR `ports.rs:286-309`, `lib.rs:1262-1273`; the 7-vs-12 gap is **intentional** (harvest mints carry no OGIT entity → no alias; pin at `ports.rs:598-606`) — do NOT create a Healthcare subset codebook |
| OGAR vocabulary plug-ins (`ogar-vocab`, `ogar-obo`, `ogar-fma`, `ogar-cpic`, ClassView, DDL adapters) | **EXISTING SUBSTRATE** — available plug-in infrastructure, NOT part of the live MedCare execution path (no `MedCare-rs → ogar-obo` edge exists; verified) | OGAR `crates/ogar-obo` (MONDO/HPO/Uberon/PATO); consumer grep clean |
| Host-side manifest/supervisor/actor path | **DORMANT** — decision required before any revival (see §3) | frozen since birth 2026-05-13 (81 days at investigation HEAD) |
| Callcenter authorization/audit composition (`callcenter::UnifiedBridge<B>` ⊗ `ogar::UnifiedBridge<P>`) | **LATER integration + hardening** — conceptually right, **currently unbuildable** (callcenter has zero dep on lance-graph-ogar); not a prerequisite for the first thought | dep-graph verified |
| Real cognitive thought over MedCare data (shader-driver / MailboxSoA) | **ACTIVE PROOF TARGET** — this plan's §4 | the honest gap: cognition, not plumbing |
| Generic manifest-driven actor factory | **POST-PROOF generalization** (only if §3 rules "revive") | — |

## 2. The live path (build on THIS)

```
MedCare-rs (private)
  └─ crates/medcare-bridge ──(vendor/lance-graph softlink)──►
       lance_graph_ogar::MedcareBridge = UnifiedBridge<HealthcarePort>
         ├─ entity()/entity_by_uri() — namespace-locked, codebook synthesis
         ├─ ogar_vocab::ports::HealthcarePort  (class_id("Patient")=0x0901, APP_PREFIX=0x0005)
         └─ OntologyRegistry (TTL hydration)
  └─ crates/medcare-rbac ──► ogar-vocab (direct)
```

Compiles and is tested on the consumer side (`healthcare_hydrate.rs` scope-lock)
and on this side (`bridge_scope_lock.rs`, `medcare_bridge_conforms`, all green —
NOTE: `lance-graph-ogar` is workspace-EXCLUDED; test via
`--manifest-path crates/lance-graph-ogar/Cargo.toml`, a green parent-workspace
build proves nothing about it).

## 3. The dead lineage (dormant — do NOT revive to prove thinking)

```
modules/medcare/manifest.yaml → build.rs → MANIFEST_METADATA ╳ (one caller: a test)
CallcenterSupervisor::spawn_consumer_actor → StubConsumerActor  (unconditional)
DispatchToG: Health → ok; ALL else → DispatchNotImplemented (child never reached)
MedcareConsumerActor — never constructed; all 8 arms tracing::debug! + TODO
manifest actor.type = MedCareActor / MedCareMessage — exist in NEITHER repo
rbac_policy: medcare_policy — symbol does not exist
entity codes 100–105 / action_capabilities / actor.message_type — parsed then DISCARDED
```

Reviving this is **not a wiring patch — it would be a new implementation based
on a stale declaration.** The literal MedCare surfaces that ARE live and useful
as workbench material — `StepDomain::Medcare::profile()`, `medcare_ontology()`,
the ontology_table/conformance fixtures, `ConsumerEnvelope` types — may be
reused as proof fixtures, but are NOT the canonical runtime composition and
must not be presented as such.

**Open decision (operator):** retire vs revive. If revived: a real typed actor
factory keyed on live types, never on the fictional YAML names; and the
manifest either emits everything it declares or stops declaring it.

## 4. The proof target — first medical thought (the drill hole)

**Home: the proof lands primarily in MedCare-rs** — the live composition root
per §2. lance-graph contributes only **generic** seams genuinely missing (e.g.
the deferred `MailboxSoaView::qualia()` when sanctioned, driver entry surface)
— never a new "MedCare-rs-shaped" host adapter in this repo (that would
reconstruct the dead lineage in miniature). Public/private boundary:
MedCare-specific schema and clinical mappings stay in MedCare-rs; generic
SoA/cycle/driver primitives may land here.

```
MedCare-rs input (existing schema surface, private side)
  → existing medcare-bridge → UnifiedBridge<HealthcarePort>   (classid, namespace lock)
  → existing MedCare-rs policy/schema path (or narrow proof policy — fail-closed)
  → cognitive-shader-driver + REAL MailboxSoA                 (the honest gap)
  → non-vacuous medical cognitive result
  → owner_adapter::emit_bootstrap_intent / BatchWriter        (write-on-behalf cast)
  → cycle_driver (#879, merged): collect → seal (one WAL write) → apply sparse → next intent
```

**The driver requirement (hard):** the proof must invoke the EXISTING
`cognitive-shader-driver` + `MailboxSoA` operational unit. Reading qualia from
the real SoA column is necessary but NOT sufficient — feeding extracted values
straight into `shade_owner` bypasses the driver and proves only the MUL gate
(already proven). `shade_owner` participates only as the existing downstream
gate where the driver actually reaches it, never as a substitute for the
driver.

**Trace-and-report obligation:** the proof's deliverable includes the exact
active chain, symbol by symbol:
`MedCare input type → HealthcarePort resolution → SoA projection →
cognitive-shader-driver entry point → material SoA/result change → MUL gate
(if reached) → StrategyOutcome → owner_adapter cast → sparse sealed
transition`. Every arrow named with file:symbol; no arrow asserted without
evidence.

**Falsifiers (per the P0 falsifiability rule):**
- F1 — the cognitive result is **non-vacuous**: two different medical inputs
  produce two different DRIVER outcomes (a material SoA/result change that
  discriminates; not a constant, and not gate-only).
- F2 — the classid on the thought's carrier is the OGAR canon (`0x0901`-family
  via `HealthcarePort::class_id`), not a local literal.
- F3 — the intent round-trips: cast in cycle N is collected, sealed (exactly one
  WAL write) and applied in N+1; unrepresented owners byte-identical.
- F4 — the policy leg is fail-closed: an unrecognized actor/role against the
  Healthcare classid is DENIED (this test is currently ABSENT — finding h).

## 5. Optional OBO slice (only if it makes the proof observably medical)

`ogar-obo` exists (MONDO/HPO/Uberon/PATO). If the proof wants a visibly
clinical hop: `HealthcarePort::Diagnosis → MONDO term → optional HPO phenotype
→ cognitive operation`. No full vocabulary hydration, no ontology federation,
no cathedral. Skip entirely if F1–F4 pass without it.

## 6. Open decisions carried (not blocked on — the proof proceeds around them)

1. **OQ-2 retention:** 2190d (`StepDomain::Medcare`, tested) vs 3650d
   (manifest, dead path) — HIPAA 6yr vs BMV-Ä §57 10yr. Operator/regulatory.
2. **`Ueberweisung` / `Anamnese` canon:** first-class consumer concepts;
   `anamnesis` = `0x0908` (no port alias), `Ueberweisung` has NO canonical id.
   OGAR-side mint decision.
3. **Dead-lineage retire/revive** (§3).
4. **Bridge composition edge** (callcenter⊗ogar) — later hardening.
5. **`.grok/` Zone-3/spear lineage** — unreconciled third model; needs a ruling
   or a staleness banner.
6. **Audit default = `NoopAuditSink`** with `SuperDomain::Unknown`/salt 0 and
   swallowed emit errors — for a HIPAA-regime domain, "configured" and
   "recording" are indistinguishable. P0 candidate, own PR.

## 7. Exclusions (scope fence)

- NO manifest/supervisor/actor revival for the proof.
- NO new Healthcare subset codebook; NO local classid literals.
- NO callcenter⊗ogar dependency edge as a proof prerequisite.
- NO provisional-intent recovery ledger in #879 (merged). The authoritative
  pre-commit rule (operator-ruled): **sealed `Vn` + unchanged Kanban task +
  deterministic computation = the same provisional intent on the next sweep.**
  Commit fails before `Vn+1` exists → publish nothing · mutate no owner ·
  advance no watermark · **discard** provisional slots, held moves and planning
  results · rerun the unchanged task from `Vn`. Correctness never requires
  retaining a byte-identical frozen cycle — `SealFailure{casts}` is classified
  as an **optional retry cache / implementation convenience only** and must not
  become a planning ledger (so documented in `cycle_driver.rs`, with the
  deterministic-regeneration falsifier
  `pre_commit_failure_discards_everything_and_regenerates_from_vn` that DROPS
  the cache and proves the same semantic sparse cycle re-derives).
  `recover_fleet` is **committed-history recovery ONLY** (`Vn+1` exists,
  application/restart interrupted) — never conceptually grouped with ordinary
  write failure; the two mechanisms share no state. (Latency figures attached
  to these paths in review prose are **operator-provided measurements**, not
  workspace-reproduced benchmarks — their exact values are not the
  architectural foundation and are not restated here.)
- NO clinical schema detail or sourcing/licensing reasoning in this public repo.

## 8. Status ledger for the older documents

| Doc | Disposition |
|---|---|
| `pr-e-1-manifest-modules.md`, `pr-g1-manifest-modules.md`, `pr-g2-ractor-supervisor.md` | historical — describe the dormant lineage |
| `pr-e1-medcare-super-domain.md` | historical + carries the open OQ-2 |
| `foundry-consumer-parity-v1.md`, `foundry-roadmap.md`, `MEDCARE_POLICY_GAP.md`, `td-super-domain-subcrates.md` | stale-but-unmarked (investigation §8-P3); banner candidates |
| `ogar-sink-in-and-consumer-bridge-removal-v1.md` | substantially DONE (the `ddb6c840` collapse) |
| `lance-graph-in-medcare-rs-v1.md`, `unified-bridge-consumer-migration-v1.md` | partially landed; remaining items fold into §6 here |
| `.grok/board/*` | unreconciled second lineage; §6.5 |

**The central realignment:** OGAR and the bridge pattern are NOT the unfinished
part. The unfinished part is **cognition and composition**. Prove one real
medical thought through the already-live consumer path, then generalize the
exact seam that worked.
