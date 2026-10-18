# D-CPG-v1: Causal Pickaxe over resident Git, Lance 13 refs, stable identity and temporal horizons

**Status:** PROPOSAL / PRE-REGISTERED FALSIFIERS (2026-10-18). No runtime implementation, new persistent carrier, dependency upgrade, causal certificate, or benchmark claim.
**What changes:** A **Git porcelain-to-epistemic-algebra contract**, version-plane adapter and causal-debugger query contract joining existing D-TGA thought Git (#1599), candidate resolutions (#1604), the Lance 13/LanceDB 0.40 capability surface, immutable NodeGuid identity, and `crates/lance-graph-planner/src/temporal.rs`.
**Why:** The user asks whether Git can *literally* fit around Lance versioning × stable rows × `temporal.rs` and whether semantic Pickaxe can become a causal debugger. Earlier 64k kanban work shows a fold-to-write cost ratio requiring **resident speculative history with selective durable squashes**, not per-step Lance writes.

## 0. The answer, bounded

**Yes for the user-facing version-control API; no for equating different identities.** Lance 13 has dataset versions, tags, branches, read-only checkout, restore, transactions and dataset delta capabilities. LanceDB 0.40.0 uses Lance 13.0.0; its Rust Table API exposes version/checkout/tags/create_branch/checkout_branch. Native refs eliminate the need to emulate *durable* tags and snapshot branching. They do **not** provide the semantic merge rules, per-thought in-RAM commit DAG, evidence lineage, Git-style content-addressed commits, or causal interventions. Those remain a narrow adapter atop existing carriers.

The preferred topology:

```text
   [Canonical reference bake B_v; immutable NodeGuid keys and codebook seal]
                                  |
                 +----------------+----------------+
                 | RESIDENT EXECUTION / NO I/O      |
                 | thought 0 givens; 1..=63 branches|
                 | RelAlg/Datalog -> R2IL/mask-risc  |
                 | ephemeral attention / many folds |
                 | sparse typed deltas at same i    |
                 | logical commit DAG / semantic diff|
                 +----------------+----------------+
                                  |
                semantic artifact election / G1..G7 /
                authorized change / nonrepeatable receipt
                                  |
              one canonical writer -> sparse batch seal
                     |            |             |
                     v            v             v
              Lance 13 MVCC    version refs    resident replay
              data/checkpoint  branch/tag       & delta masks
                     |
            [temporal.rs horizons + local trajectories]
                     |
           pickaxe(NodeGuid, time, query, provenance)
                     |
            temporal -> derivation -> counterfactual
```

**Terms:** "Git" is an *interface/metaphor* for versioned reasoning; actual Git object storage is not a dependency. Lance refs are storage pointers; branch names/versions are not automatically Git SHA hashes. An in-RAM thought commit is **not** a Lance commit, and a single Lance version is not necessarily one thought.

## 1. Map the three identities, four clocks, and two causal orders

| Plane | Canonical identity | Semantics | Must NOT be mistaken for |
| --- | --- | --- | --- |
| semantic | `NodeGuid` (16 bytes), registered codebook/bake | *what* entity or proposition, immutable identity in a sealed frame | Lance `_rowid` or `_rowaddr` |
| speculative | `ThoughtCommitId` (opaque, stable while retained), parent IDs | which in-memory explanation/branch revision produced changed slots | CE64 W 0..63; Git's SHA without actual content digest |
| durable | `DurableRef{dataset_id, branch, version}` | Lance-visible sealed snapshot; may also have tag | physical row address or wall-clock timestamp |
| witness | independent source root + rule/program identity + derivation read-set | why a conclusion was derived and which alternative supports it | written-by popcount / Git ancestry |

Clock and order axes:

1. **LIVE**: resident computation steps, transient saccades, and local thought commit sequence.
2. **PERSISTENCE**: one serialized application writer publishes nonempty semantic batch versions.
3. **RECOVERY**: checkpoint + history-tail reconstruction, not the hot query loop.
4. **EVENT / KNOWLEDGE TIME**: owner-local sequence and HLC/reader horizon mediated by `temporal.rs`. Commit wall time cannot replace these semantics.

Three **distinct orders**: (a) DAG parent/merge ancestry, (b) owner-local evidence event order and cross-server causal metadata, (c) logical derivation dependency. None is interchangeable with another. Two independent branches can have the same wall-clock minute yet be causally unrelated. A future datum visible to an `Aware`/`Retro` reader must not become an input to a historically `Strict` conclusion.

## 2. Source-grounded preflight: what's really there

| Piece | Checked source | Current status |
| --- | --- | --- |
| Thought API | `.claude/plans/thought-git-api-v1.md` (#1599) | PROPOSAL. `branch`, `diff`, `status`, `rebase`, `squash_into`, `land`, `blame`, `MergeQueue`; D-TGA-1/3 etc still test-gated. **Reuse it, do not rewrite it.** |
| Candidate identities | #1604 `docs/research/versioned-semantic-candidates-64-thoughts-rfc.md` (PR branch, not main) | PROPOSAL. Different bound claims require different identities; semantic ambiguity survives. |
| NodeGuid and aperture | `crates/lance-graph-contract/src/canonical_node.rs`; `.claude/knowledge/epistemic-frontier-capstone.md` | CURRENT-CONTRACT. `NodeGuid` key, one sealed SoA allocation. No new key, hot String or foreign-index gather. |
| Temporal | `crates/lance-graph-planner/src/temporal.rs` | ACTUAL IMPLEMENTATION: `LanceVersion = u64`, `QueryReference::at(v,rung)`, statuses `Contemporary / Anachronistic / Spoiler / Unknowable`, `LocalCausalRow` and `local_trajectories`. `DependsClosure` seam currently uses trivial `NoDeps` by default. Cross-server HLC comparison policy deferred. |
| Persistence writer | `.claude/plans/persistence-artifact-backed-commit-v1.md`; `crates/lance-graph-planner/src/persist_sink.rs` | RATIFIED rule: no changed artifact ⇒ no write/version. One application writer, `(cycle,batch_hash)` reconciliation; writer-side canonical stable ordering, separate from temporal read projection. |
| Three clocks | `.claude/board/entries/2026-09-24-three-clocks-never-meet-implicitly.md` | DECISION. Live != persistence != recovery. Explicit backpressure, durability demand or recovery are the rendezvous. |
| Fold/write economics | `.claude/board/entries/2026-09-24-cost-scales-with-dirty-rows-replay-budget-before-materialization.md` | MEASURED on Lance 11, not 13: ~1.7ns axis fold, ~1.7ms commit floor, ~1M comparable folds per minimal commit; 64k dirty rows ~533–583ms seal. DO NOT port these timings to new backends without remeasure. |
| Resident log | `.claude/knowledge/split-tunnel-sparse-append-resident-fold.md` | MEASURED in HubSPO-rs, not this end-to-end model. Append sparse records, fold resident; notes different version-guard/idempotency limitations in reference consumer. |
| Direct same-coordinate reads | H14c/H14d plans | MEASURED probe: `Thought(t)[Table(a)[i] : Table(b)[i]]`; mask prefilter. General foreign-key graph join remains unsolved. |
| Lance 13 dataset | https://docs.rs/lance/13.0.0/lance/dataset/struct.Dataset.html | Versioned docs list `create_branch` / `checkout_branch` / `tags` / `delta` / `read_version_transaction` / `checkout_version` / `restore` and stable row-ID migration. Capability *exists* in pinned upstream, but integration with the repository's workspace versions is NOT proven. |
| LanceDB 0.40 | https://docs.rs/lancedb/0.40.0/lancedb/table/struct.Table.html ; https://docs.rs/lancedb/latest/lancedb/table/struct.Table.html | Stable release 0.40.0 uses Lance 13.0.0. Version, checkout, tag and branch API documented in release's current crate docs; check local-vs-remote branch operations with a compiled probe. |
| Stable row ID behavior | https://github.com/lance-format/lance/blob/main/docs/src/format/table/row_id_lineage.md | Upstream **current** spec differentiates stable `_rowid`, moving physical `_rowaddr` and row-level version lineage. Current spec may reflect post-13 behavior; pin 13.0.0 binary probe before asserting update/remap or CDC support. `NodeGuid` remains identity across table copies, forks and bakes only under a validated codebook/identity mapping. |

### Two explicitly dangerous temporal defaults

- `QueryReference::default()` means `ref_version = u64::MAX`, hence nominally Strict but **unbounded**. For Pickaxe/as-of/audit MUST use `QueryReference::at(concrete_observed_head, rung)`, and for a strict historical answer set `mode=Strict` explicitly rather than inherit `Aware` or `Retro` from high rung.
- `NoDeps` returns `ready()` for all subjects, a correct nonblocking placeholder **not proof of a causal dependency closure**. Fail closed or downgrade a result to "unverified dependency" when Pickaxe asks causal questions.

## 3. Would native Lance branches literally replace Git branches?

**Only at the durable snapshot level, not at the million-fold live level.** Native Lance branches/tags can carry saved **dataset-level** experiments and immutable/retained versions. For a large, separately funded experiment, a real storage branch might be appropriate. Most speculative thought branches should stay in RAM and share the sealed base read handle.

| Git-like operation | Proposed cheap resident execution | Durable storage binding | Semantic check |
| --- | --- | --- | --- |
| `branch()` | allocate thought 1..=63 on same sealed frame; make logical DAG parent | **none** for ordinary speculative branch; optionally `Dataset::create_branch` for long-lived experimental dataset branch | no 65th slot / no shadow mutation of given 0 |
| `diff(a,b)` | `W_a XOR W_b` then typed cell compare under aperture | `Dataset::delta()` across *durable* versions as independent oracle where available | presence XOR != content difference; same claim/field needed |
| `checkout(rev)` | pin immutable logical revision + base, read by ordered delta fold | `checkout_version` or read-only Table checkout for persisted revision | no future leakage and no mutable checked-out table |
| `pickaxe` | intersect candidate/field change mask, revision interval and thought selection; inspect surviving records | optionally enumerate `read_version_transaction` and version deltas at recovery/audit clock | changes != derivation dependencies != physical cause |
| `blame` | original source-root bitsets + derivation proof + landing version | durable receipt keyed by branch+version (or dataset identity) | source-count not thought-count |
| `rebase` | D-TGA masks R/W/M + DRed on deleted premises | compare pinned head; resolve changed NodeGuid mapping | never allow ambiguous stale read or unexamined deletion |
| `merge / squash` | typed agreement/conflict resolution over versioned candidates; write dirty slots once | one canonical `LanceCycleWriter` non-empty batch | evidence independence, negative strata, no default latest-wins |
| `restore / revert` | create new compensating *semantic* version under explicit approval | physical `restore` is **not** semantic Git revert and can hide later rows; avoid on main path | never erase later evidence to simulate temporal rewind |
| `gc / compact` | discard resident intermediates only if receipts preserve promised pickaxe/bisect questions | apply tested retention policy on tags/history; disallow cleanup of pinned versions | refuse exact historical answer if sufficient lineage was discarded |

**Important:** Native Lance branch API and per-branch version numbers do not guarantee Git's 3-way merge, conflict representation, full ancestry DAG, SHA identity, or semantic root independence. Do not promise these from storage primitives. The adapter must decide and test its own narrow behavior.

## 4. Proposed Pickaxe syntax as **query**, not new VM

No additional instruction set. Proposed user-facing *sketch* routed through D-TGA and current RelAlg/R2IL execution.

```rust
// Proposed surface only; these types and functions do not exist.
let q = mind.pickaxe(ClaimId::of("fox", "steals", "key"))
    .within(EventWindow::last_minutes(20))
    .at(QueryReference::at(observed_lance_head, 0))
    .in_thoughts(ThoughtMask::selected(&[24, 48]))
    .changed_values()
    .with_evidence_roots();

let hits = q.execute_on_resident_masks()?;   // no Lance read on hot path
let why = hits[0].derivation_witnesses()?;    // truth maintenance, not chronology
let test = why.counterfactual_replay(RemoveRoot(source_a))?;
```

**Cold path:** for a historical range not resident, resolve **exact** `DurableRef(dataset, branch, version)`, pin manifest, replay/deserialize necessary sealed deltas, then execute same mask program. Never run one Lance checkout/query per logical fold or commit. Never silently use the latest table handle if the query says as-of V_old.

**Hot semantic-change discovery**: To avoid 10^10 commit walks, maintain *only* a bounded resident change summary per candidate/field/time bucket when cost-positive. `Select = AddressAperture ∩ ChangedFieldMask ∩ ThoughtMask ∩ PinnedTemporalWindow`; inspect actual typed values and receipts at surviving positions. The mask indicates where a relevant change *could* be; doesn't prove the value changed. An event-window query uses HLC/owner-local sequence or specifically recorded event stamps, NOT `LanceVersion` as minutes.

**Pickaxe causal grades**:
1. **Temporal change**: the value/proposition differed between revisions.
2. **Provenance**: a source/program/root contributed under registered derivation rules.
3. **Counterfactual computation**: changing a specified input alters the conclusion **inside the fixed, deterministic model**; warn on alternative sufficient proofs and overdetermination.
4. **World causal claim**: requires separately justified intervention/identification assumptions and Pearl-grade evidence, never inferred from Git parentage or NAL confidence.

`temporal.rs` classifies what was knowable and owner order; Datalog provenance answers what was derivationally relevant; Pearl-style tests answer **intervention under a model**. This does NOT establish real-world cause by provenance alone. The current CE64 remains one unchanged ISA with 40..42 Pearl question, 43..45 registered Haspelmath orientation, 46..49 activation i4 (decision/code alignment open), 53..58 thought W (decision/code alignment open), 59..63 one EpistemicState5. No Pickaxe opcode or witness-pointer abuse.

## 5. Important data correctness boundaries

1. **Three rows:** semantic `NodeGuid` != logical Lance `_rowid` != physical `_rowaddr`. `_rowaddr` may move on compact/update; stable IDs are an opt-in feature and not a cross-dataset UUID. Always resolve the correct version-scoped address map for masked I/O. Test migration, compaction, update, delete, reopen and branch independently with Lance 13 pinned.
2. **Version coordinate:** a numeric version alone is insufficient when branches, datasets or clones exist. A `DurableRef` includes dataset identity + branch + version; potentially manifest hash / codebook seal after verification. A movable tag is not a commit ID.
3. **Time and causality:** `TemporalStatus::Contemporary` isn't a certified derivation or causal dependency. Enforce `knowable_from` and pinned `V_ref`, preserve independent owner trajectories; HLC policy for cross-server is deferred. Concurrent thought finish order is not causal order.
4. **Visibility:** a thought may read only sealed givens at its pinned base unless a **declared dependency** explicitly admits resident speculative outputs; no accidental unsealed sibling reads. Authorization and tenant/field masks filter **before** payload.
5. **Merge:** merge two overlapping claims through G1–G7 and source-root independence; if roots duplicate or conflict, retain both perspectives but refuse false double evidence. Negative dependency graph topological order is required; FIFO alone is not stratification.
6. **Retention:** a squash may preserve a compact result **only if** it retains a sufficient replay receipt for the advertised query level. If exact pickaxe/bisect of discarded micro-steps is requested, return `LineageUnavailable`; never invent an erased step.
7. **Nonrepeatable information:** independently witnessed external observations/action receipts need required durability. Speculative computations are regenerate-able only if program version, input roots and deterministic assumptions are pinned.
8. **No per-fold allocation:** a theoretical billion folds becomes nonsense if every fold instantiates a Git commit/HashMap. Use bounded reused scratch/mask words; commits only for semantically significant working-tree transitions. Bench allocations and peak RSS.
9. **No implicit causal model:** do NOT shortcut the pending `DependsClosure` SPO implementation using ancestry; test that `NoDeps` cannot produce a certified dependency result.

## 6. Prototype stages and falsifiers (registered before execution)

Harness to land **first**, in an isolated probe (no default workspace pins or production writer mutation). Machine-readable fixtures: `docs/research/causal-pickaxe-version-temporal-fixtures.json`.

| ID | Probe / independent oracle | Must demonstrate / kill |
| --- | --- | --- |
| CPG-00 | Compile exact versions: Lance `=13.0.0` and LanceDB `=0.40.0` in standalone tool; inventory API | refuse any "verified" claim if missing methods/feature flags; report native vs remote capability separately |
| CPG-01 | Stable NodeGuid over append/update/delete/reopen/compaction; stable vs legacy `_rowid`; `_rowaddr` | exact content/identity parity after relocation; ensure stale physical address is rejected |
| CPG-02 | Native Lance branch/tag/checkout/restore probe | branch writes isolated; checkout read-only; restore behavior explicitly demonstrated **not** Git revert |
| CPG-03 | Speculative resident history 0, 1, 63, 64 attempts | 63 hypotheses plus givens; next refused; write-only changed cells; no Lance call on ephemeral branch/commit |
| CPG-04 | 1,000,000 small resident folds vs (a) each-step persist, (b) batched dirty squash | correctness equal at seal; benchmark p50/p95 ns/step and bytes/write, not borrowed Lance11 cost |
| CPG-05 | Event HLC and observed-head fixture, with future evidence available in storage | strict pinned past read never observes future; explicit `Aware`/`Retro` flagged not rewritten as Strict |
| CPG-06 | Two owner trajectories interleaved in global durable order | `local_trajectories` restores each owner's durable monotonic order; reset sequence collisions refused |
| CPG-07 | Source root duplicated in 3 thoughts; separate roots in 1 thought | evidence roots counted once vs twice appropriately; thought-popcount != independent witness count |
| CPG-08 | Candidate flip caused by changed rule vs changed evidence | pickaxe attributes correct dependency and can say "underdetermined" when both explanations possible |
| CPG-09 | Alternative independent sufficient derivations of same Y | removing one root may leave Y; Pickaxe does NOT equate no output change with no contribution |
| CPG-10 | Remove a previously used premise on a rebase | exact from-scratch oracle equals DRed rebase; stale derived B disappears |
| CPG-11 | Squash and drop unneeded transient reasoning, then historical pickaxe | retained receipt answers advertised queries; deleted exact-step query returns `LineageUnavailable` |
| CPG-12 | Two branches both number a version `7` | version alone never resolves a branch; use dataset+branch+version |
| CPG-13 | `NoDeps` and HLC-None default | no certification of actual data-causal dependency or cross-server order |
| CPG-14 | Timestamp window with out-of-order commits and late evidence | use event time/HLC plus horizon, not only commit-time filter |
| CPG-15 | Native tag/version retention and cleanup | pinned historical version recoverable OR explicit retention refusal; no silent broken citation |
| CPG-16 | Concurrent write/race, lost ACK, retry | one logical writer and durable idempotency; no false CAS claim for Lance Append |
| CPG-17 | Access-controlled historical row | auth/field/tenant mask applies to historical checkout too; denied row not decoded |

**Arms:** A = in-memory list of microsteps + per-step Lance writes (expensive control); B = D-TGA-style resident DAG + one sparse squash; C = resident DAG + Lance native persisted experimental branch; D = resident masks with no per-step commit, recompute on demand. Must test equal results, identical sealed root/lineage, independent evidence policy and version-awareness *before* timing claims. Report 64k-address tile, 0/1/10/100/1k/65,536 dirty sweep; 1, 63 thoughts; history depth and branch count; p50/p95 latency, allocations, writes, fragments, bytes, replay time, masked payload reads and cold recovery.

**Decision gate:** keep the simplest variant that preserves exact identity, temporal/factual correctness, stable source roots, reproducibility and target recovery latency. Reject native durable branches as a hot thought primitive if they spend metadata/manifest I/O. Never trade away semantic falsifiers for a synthetic throughput win.

## 7. Dependency and rollout order

- **CPG-0:** inventory Cargo pins and cross-crate lockstep; create a minimal, excluded Lance13/LanceDB0.40 executable API probe (no global pin bump).
- **CPG-1:** pin row-ID semantics and branch/tag/version behavior using the exact release; correct any previous Lance11 assumptions (#1276) before migration.
- **CPG-2:** map one reusable `DurableRef` adapter to the *existing* `LanceCycleWriter`/D-TGA boundary; retain single-writer contract and artifact election.
- **CPG-3:** resident logical history and selective squash against D-TGA-1/3/4, preserving candidate receipts from #1604 and NAL independent evidence.
- **CPG-4:** Pickaxe as a **RelAlg/Datalog read plan** over versioned masks; `temporal.rs` performs horizon/deinterlace, without adding a new interpreter.
- **CPG-5:** counterfactual replay/ablation and independent oracles; differentiate model dependence, derivation necessity, causal Pearl inference.
- **CPG-6:** user-facing Git vocabulary and benchmarks if CPG-0..5 pass, including bounded memory and crash/cleanup recovery.

**Prerequisite:** protect any existing consumers that require older Lance pins (root `Cargo.toml` documents older 7/0.30 lockstep and excluded Lance13 probes). Do not claim whole-workspace migration has happened. If Lance13 and LanceDB0.40 APIs introduce incompatible crate versions, leave the default workspace unchanged and publish only an isolated probe/failing report.

## 8. What would falsify the idea?

- Native Lance branch version identifiers or stable row IDs are insufficient to satisfy the exact-version, cross-branch identity required, and the needed mapping is too expensive.
- Maintaining resident logical commit metadata costs more memory/time than regenerating intermediate results under realistic 64k×64 workloads.
- A Pickaxe report changes based solely on commit chronology, while the genuine derivation and evidence roots do not.
- Speculative rebase produces different conclusions from a fresh run under the same evidence.
- Recovery after squash silently loses an advertised reconstruction capability.
- The explicit causal query is answered using only `NoDeps`, "Strict" default sentinel, copied witnesses, or grammatical/Haspelmath orientation as a causal proof.

**Bottom line:** Git **around** Lance13 stable and temporal versions is promising. Git **inside each fold** is precisely the problem we originally eliminated. First demonstrate that a resident branch-and-squash graph can buy causal debugging without paying per-step I/O, and that temporal and semantic causality remain separately justified.

## 9. Git porcelain is the public surface of epistemic algebra (2026-10-18 amendment)

**Operator insight:** "The one API that despite the complexity nobody can resist as a surface of masking relalg Datalog AST NAL." The follow-up decision is to **expand this PR**, not open #1606: #1599 already owns thought lifecycle, #1604 owns candidate semantics, and this plan owns the *public Git vocabulary → typed operation → existing evaluation substrate* seam. The interface needs to look familiar; the semantic engine must remain singular.

**Architectural law:** **Git is the porcelain, not the VM.** Git verbs compile or route to operations already owned by the D-TGA lifecycle, RelAlg normalization, the external Datalog fixpoint driver, R2IL / mask-risc, NAL/ONA evidential revision, temporal read-horizon checks, and Lance durable version refs. A Git command may compose them, but MUST NOT introduce a new inference evaluator, witness counter, causal edge model, graph-key lookup, independently materialized row universe, or per-fold persistence.

### 9.1 Vocabulary mapping and its limits

| Public porcelain | Meaning for an epistemic user | Existing execution / contract | Refusal or caveat |
| --- | --- | --- | --- |
| `status` | pending proposals, dirty/clean/stale, conflicts, unresolved evidence | D-TGA `R/W/M` masks, `popcount`, candidate outcomes (#1604), NAL-root status | No "supported" claim from thought-popcount alone; no implicit hydration/mutation |
| `branch` | start another possible interpretation | D-TGA branch slot 1..=63, shared immutable NodeGuid spine, parent logical revision | Not a new Lance dataset by default; slot != historical commit ID |
| `add -p` / `stage` | select some proposed semantic deltas for review/landing | address/field aperture AND changed mask; typed candidate and policy gates | **Staged ≠ true, proved, causal, authorized, or persisted**; selection alone never upgrades CE64 F/C |
| `diff --semantic` | compare interpretations, not just occupancy | `P_a XOR P_b` **candidate locations**, then typed value comparisons at same address under aperture; explicit binding algebra where needed | XOR of presence != semantic disagreement; identical presence and different values still differ |
| `merge` | reconcile compatible branches without dropping alternatives | D-TGA dependency-ordered queue, DRed, NAL independent-root checks, G1–G7 gate | No "ours/theirs/latest wins" as epistemic truth; conflicts remain visible |
| `rebase` | re-evaluate with newly sealed evidence | D-TGA read/write masks, stale comparisons, semi-naive Datalog re-run + retraction | Rebase after premise deletion must equal from-scratch recomputation; no future evidence leakage |
| `blame` | inspect origin of a claim, rules, independent roots, landing version | root bitsets, derivation receipt, logical DAG/version mapping | Git commit author/CE64 W identifies a producing perspective, not independent evidence |
| `bisect` | locate earliest revision that changes an oracle result | pinned logical commit DAG, deterministic reconstruction, predicate oracle | Only meaningful if oracle stable and replay coverage sufficient; otherwise `LineageUnavailable` |
| `pickaxe` | find claim/value changes and investigate derivational cause | change-mask + temporal aperture + pinned revision, then witness dependency and optional model-intervention replay | Temporal/provenance/ablation/real-world causal grades must not collapse |
| `squash` | retain justifiable semantic progress and sufficient reconstruction | D-TGA partial squash; artifact gate; one writer; receipts | No micro-commit per fold; explicit query fidelity/retention limitations |
| `checkout` / `log` | view previous warranted states and why they landed | pinned resident revision or `DurableRef{dataset,branch,version}`; `temporal.rs` | No unbounded `QueryReference::default()` for historical audit |
| `revert` | propose reversing a semantic change | new reviewed inverse/retraction proposal + DRed at next version | Do NOT silently invoke Lance `restore`, erase independent evidence, or rewrite published history |

**Command spelling is a proposal, not a released CLI.** User-facing `mind status` or `git know status` can be aliases to the same typed operation. Avoid hijacking the system's actual `git` executable or assuming a filesystem implementation. Names are label-codebook lookups at admission, not hot-path String keys.

### 9.2 A single lowering path, never an implementation per verb

```text
human | AI agent | Rust caller
       ↓
Git-like porcelain (admission, parsed options, explicit scope)
       ↓
typed intent: operation, NodeGuid/claim aperture, thought/revision,
              event window, pinned observed head, authorization,
              evidence policy (no "just true" checkbox)
       ↓
RelAlg normalize (∩ ∪ complement / selection / projection where supported)
    ↘ Datalog external fixpoint / DRed only for genuinely recursive dependencies
       ↓
R2IL / mask-risc  → bitset address mask + typed resident payload terminal
       ↓
candidate outcomes / NAL evidence revision gates / temporal classification
       ↓
read-only report  OR  D-TGA speculative change  OR  reviewed sparse Lance land
```

The lowering chooses a pre-existing operation by **semantic effect** (read, speculative edit, evidential review, durable land), not by the CLI verb spelling. Two porcelain expressions with the same typed intent and sealed inputs must normalize to equivalent results. When a command requires true variable-binding joins rather than already coindexed addresses, it may only use a validated relational operator/projection. Boolean mask AND is not a general Datalog join or an established WCOJ.

The "Git" similarity stops at meaningful semantics:
- `branch`/`diff` are often cheap mask/overlay operations; `merge` is **not** a 3-way text merge, but independent-root revision plus contradictory alternatives.
- `add -p` is candidate *admission to a stage*, **not epistemic certification**; staging must retain negative witnesses and rejected alternatives.
- `status` is a version-specific *view*; no derived cache writes just because someone asked.
- `bisect`/`pickaxe` are only as exact as the **retained** input, codebook, program, source roots and commit lineage; reject unsupported historical precision instead of fabricating it.
- A user agent cannot upgrade authorization by checking out an earlier branch. Apply row/field/tenant/temporal masks **before** payload.
- No new CE64 field: bits 43..45 retain Haspelmath registered function, 53..58 thought W; causal Pickaxe requests and receipts live outside those bits.

### 9.3 Make the familiar commands useful without hiding epistemic nuance

Illustrative porcelain, entirely **NOT_IMPLEMENTED**:

```sh
mind branch possible-inheritance
mind query 'Why does this user have this effective permission?' --as-of main@V421
mind status --evidence
mind diff main@V421..possible-inheritance --semantic
mind add -p --scope permission:effective      # propose selected changed cells
mind rebase possible-inheritance --onto main@V422
mind blame permission:effective --roots
mind pickaxe permission:inherited --since '20 minutes' --explain
mind merge possible-inheritance --revision-policy nal
mind squash --receipt sufficient-for-pickaxe
```

Illustrative status *semantics*:

```text
On thought: possible-inheritance
Pinned base: dataset/main@V421
Compared head: dataset/main@V422

  43 candidate claims
  11 with independent supporting roots
   7 conflicting alternatives
  25 unresolved
   3 selected for review
   2 require additional evidence
  0 durable changes until authorized, successful land

Changes are proposals; confidence and proof status come from evidence,
not from staging, membership, or Git ancestry.
```

Counts above are fictional UI fixture values, not repository benchmarks or verified NAL verdicts. Maintain human and AI parity: an agent issuing the same command against the same sealed context sees the same provenance-bearing result, except fields hidden by its authorization.

### 9.4 Extended falsification: porcelain is a compilation surface

The JSON preregistration adds `CPG-18`–`CPG-25` and **does not constitute executed tests**:

- Two different Git command spellings with equivalent typed intent produce identical masks and results; no distinct evaluator executes.
- `status` on unchanged inputs produces zero durable writes and does not grant source independence by thought-count.
- `diff` catches a value change when both presence planes are equal.
- `add -p` selects a proper changed subset but never certifies candidates or silently drops conflicting evidence.
- `merge` preserves incompatible alternatives and refuses duplicated source-root inflation.
- `bisect` with a nonmonotonic oracle is either exhaustive/validated or refuses the unsafe binary search; pruned lineage gives `LineageUnavailable`.
- A historical unauthorized branch remains unreadable and respects a concretely pinned strict knowledge horizon.
- A command alias and its machine-authored AST share the identical audited execution path and outcome.

**Review gate:** treat the porcelain feature as **DESIGN-ONLY** until D-TGA's falsifiers, H14c/H14d operators, strict temporal horizon enforcement, source-root policy, and stable-row/branch API probes have actual passing fixtures. A presentable CLI is not evidence of a correct reasoning engine.

**Summary:** **One porcelain, one typed meaning, one mask execution substrate, one evidence policy, one durable writer.** Git makes epistemic algebra approachable precisely because it does not replace that algebra.
