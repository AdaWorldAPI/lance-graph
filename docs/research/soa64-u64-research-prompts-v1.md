# SOA64-U64: research prompt pack and EXO-50 handoff protocol

**Status:** Research prompts, not operator authorization. Companion to [research map](./soa64-u64-demand-amortization-map-v1.md).
**Target:** `AdaWorldAPI/lance-graph` on a dedicated research branch. Read upstream current main before interpreting historical locations. No value execution admitted here.

## A. Mandatory session boot instructions (paste before every specialty prompt)

> You are investigating **SOA64-U64**, a hypothesis that 64 logical typed u64 property words per identity may be inexpensive **when only demanded SoA lanes are read**. You must attempt to disprove it and publish alternatives that win. Never equate virtual addressability with 64 dense resident lanes, or `simd_stride` with an admitted cross-lane executor.
>
> **First read** `CLAUDE.md`, `.claude/v3/README.md`, `crates/lance-graph-dt64/src/dr.rs`, `crates/lance-graph-dt64/src/lane.rs`, `crates/lance-graph-report/src/lane_source.rs`, and the two research files in `docs/research/soa64-u64-*`. Read applicable board rulings and Linear issues by their IDs. Record SHA, exact file:line and observed/assumed/test-only/admitted status.
>
> **EXO-50:** Use the connected Linear MCP tool to fetch [EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id) and read *new comments* before a task. It is an append-only message bus, not a decision ledger, and there is no guaranteed push delivery. Do not claim to have read comments you have not fetched. Actual CC workers must include **full session ID** in messages. Post each result as a **new comment** with the issue's envelope; never edit another agent's comment. Do not post credentials or real PII. If posting is unavailable, produce a copyable envelope in the hand-back rather than claiming delivery.
>
> **Board governance:** [EXO-186](https://linear.app/exo-red/issue/EXO-186/decision-shared-ledgers-move-to-linear-epiphanies-agent-log-and): new epiphanies P-EXO-12; agent runs P-EXO-10; actual decisions P-EXO-11. Never append new entries to the frozen EPIPHANIES/AGENT_LOG ledger files. Research recommendations are not operator rulings. Existing EXO-174 workbook u64 decision is not permission to change general SoA execution.
>
> **Tiers:** Haiku performs read-only source inventory and independent fact enumeration. Sonnet drafts bounded research text/fixtures from a complete brief. Opus plans, evaluates conflicting claims, validates measured evidence, handles cargo/git and decides what to recommend. No autonomous merge; no production writer, IAM, Gather or cross-coordinate admission. Keep report-rs #33 a draft.
>
> **Baseline invariants:** mandatory source-owned SoA; canonical/versioned NodeGuid ordinal; DR address/frame; DT64 field/method witness; separate authorized WHO scope; WHERE row mask distinct from WHAT field mask; WHY dependency footprint; typed values explicit; source-owned borrowing; fold without owned tuples; no hidden gather; scope and frame fail closed.
>
> **Every hand-back format:** question; SHA + exact locations; observed facts; competing hypotheses; smallest independent oracle; one falsifier and disabled-guard expected failure; measured allocations/lookups/bytes; admitted vs test-only vs proposed; refusals; recommendation **without** inferred approval; follow-up Linear links.

## B. Parallel research tracks (read-only audits first)

### B1. Canonical X and DR geometry, G2L amortization — Haiku inventory, Opus verdict

**Question:** Can an owner certify stable ordered NodeGuid ordinals so that a full 256-wide sweep needs only `tile=i>>8` and `slot=i&255`, without extra row-ID lookup or mask? When do sparse selections truly need a mapping?

**Read:** DT64 `dr.rs` `LocalMap`/`MapTable`/`FrameStore`, D-LANESOCK-0, [EXO-46](https://linear.app/exo-red/issue/EXO-46/a1-decide-counted-globalfocus-binding-membrane-for-production), [EXO-151](https://linear.app/exo-red/issue/EXO-151/d-drf-immutable-dr64-focus-leased-foreign-cache-sparse-writeback).

**Fixtures:** 64, 256, 257, 513 IDs, sorted and reordered snapshots, pure append and stale frame. Include two tenants with identical display labels but different canonical IDs.

**Must report:** number of global→local resolutions before and inside each tile sweep, frame invalidation mechanism, map bytes, range/mask/vector representation, and whether a second coordinate owner is being created. Reject if a source change renumbers an existing sealed ordinal without witness.

**Do NOT:** assume a GUID is a dense ordinal; implement a replacement MapTable; confuse `DR256_GLOBAL_X64` with `256×256×16`.

### B2. Logical 64 fields versus 32/64 resident lanes — Sonnet research table, Opus cost review

**Question:** Is the 64-field logical schema cheaper to describe while 1/3/8/32/64 real lanes are selected and borrowed? What is the actual descriptor size and residency pressure?

**Read:** DT64 `MAX_FIELDS`, `WideFieldMask`, `ValueDemand`, `MethodBinding::footprint`, `LaneSource` address/borrow, [EXO-121](https://linear.app/exo-red/issue/EXO-121/d-qr-cap-classviewwidefieldmask-relational-isa-binding-capstone).

**Compare:** `64 × u64` dense; `32 × u64` resident budget; field-basis-only; selective borrowed u64 lanes; u32/u16/u8 physical encoding. Include 64, 256 and 65,536 identities. Quantify exact byte budgets, cache-line traffic and metadata, not just row widths.

**Falsifier:** demand only A,C,E but deliberately poison unused fields; any access or hydration of B,D or 59 irrelevant fields fails. Field-basis reordering under equal field count also fails binding.

### B3. Typed u64 codec and precision — Haiku existing-code census, Sonnet comparator, Opus decision request

**Question:** Which typed 64-bit word interpretations are lawful for integer, scaled decimal, text code, identity handle, relation reference, null, unobserved and error? What does a tag subtract from payload?

**Read:** [EXO-174](https://linear.app/exo-red/issue/EXO-174/decision-rpt-xl-delta-03-the-workbook-is-a-projection-over-the-64k-soa), [EXO-185](https://linear.app/exo-red/issue/EXO-185/brief-pr-33-acid-test-dt64-simd-stride-audit-read-only), `lance-graph-sap/src/bind.rs` exact-decimal path, `report-rs#33` admission ledger.

**Fixtures:** signed min/max, overflow, two equal-value strings with distinct source IDs, Unicode, null vs missing, 64-bit boundary, `12345678901234567890.12` exact-decimal string.

**Must refuse:** silent `f64` rounding, `u32` truncation, arbitrary tag reuse and codebook authority minted by a downstream consumer. Determine whether wide or external representation is indispensable; do not force all values into 64 bits.

### B4. SIMD, stride and virtual transpose — independent ndarray review

**Question:** Do fixed-width 64-bit loads plus known strides beat narrow SIMD on real architectures for 1/3/8/32/64 demanded fields?

**Read:** AdaWorldAPI/ndarray `src/simd_stride.rs`, `src/simd_masking_ops.rs`, `src/simd.rs` and backends; [EXO-134](https://linear.app/exo-red/issue/EXO-134/goldstaub-count-invocation-and-allocation-boundaries-before-optimizing).

**Compare:** continuous per-lane sweep, 5-field virtual object read, register-local transpose, genuine AoS scan, u8/u16/u32 narrowing. Count SIMD registers, loads, shuffles, decode operations, LLC misses, physical bytes, allocations. Identify actual ISA paths (AVX2, AVX-512, NEON, scalar); no assumed 64-element u64 vector.

**Falsifier:** a narrow physical lane wins on cold scan/bandwidth or u64 incurs cache amplification. Report that outcome as evidence, not as a reason to suppress the benchmark.

**Boundary:** `simd_stride` is a narrower positional helper; `simd_masking_ops` handles packed mask ergonomics. Do not add raw per-backend intrinsics in a consumer.

### B5. Quack composability, X×Y and no mandatory intermediate materialization

**Question:** Can a relational contract be expressed independent of available physical realization, including second hop, join and composed reusable mask, while execution refuses cross-space paths that require forbidden Gather?

**Read:** Quack `lower`, mask-RISC `LaneRef`/`Pred`/`Agg`, relalg, [EXO-140](https://linear.app/exo-red/issue/EXO-140/admission-b-call-sites-unclassified-semijoin-equ32via-gather-and-no), [EXO-115](https://linear.app/exo-red/issue/EXO-115/quack-contract-set-algebra-route-only-the-res-expressible-filter), [EXO-108](https://linear.app/exo-red/issue/EXO-108/gap-set-semantics-distinct-over-a-projection-and-bag-count-over).

**Oracle:** independent mathematical set/bag model on symbolic coordinates, *not* tuple materialization as a proposed production implementation. Clearly mark oracle-only reference data and fold separately.

**Falsifier:** two unrelated domains have equal `i` but different underlying identities; executor must refuse. Distinguish expressible symbolic `R⊆X×Y` from currently admitted relation evaluation.

### B6. Scope, provenance and replay — Opus security lead

**Question:** Which authority can seal a production reader scope for one actor/class/row scope/field mask/frame, and which mutations invalidate it?

**Read:** D-LANESOCK-4, OGAR verified login/authorization, [EXO-135](https://linear.app/exo-red/issue/EXO-135/crumb-trail-what-may-be-built-before-execution-under-what-witness-and), [EXO-46](https://linear.app/exo-red/issue/EXO-46/a1-decide-counted-globalfocus-binding-membrane-for-production).

**Falsifiers:** counterfeit numeric scope with same ID, different tenant same label, revoked policy, foreign frame, changed property order, unknown source. Verify that `scope: None` never means all customers without explicit grant.

**Boundary:** do not construct `AbiLanes` in production; no IAM/config hacks. Record missing interface only and separate future admissible work.

### B7. Source ownership, mutation and compression

**Question:** If an upstream writer supplies versioned compressed/source-owned property lanes, can DR/DT64 hold only typed addresses, guarantee exact interpretation and preserve no-copy folds? When does decompression or relayout become necessary and costly?

**Read:** D4/S-INGRESS, DT64 lane socket, upstream Lance writer contract, `docs/COMPRESSION_MINDSET_SHIFTS.md`, `docs/CODEC_INVARIANTS_AND_EXPERIMENTS.md`.

**Compare:** source-native narrow column, dictionary/codebook, RLE/delta, memory-mapped dense u64, remote source. Report cold write/read, snapshot invalidation, disk I/O, overhead and mutation costs separately.

**Kill:** second canonical ID/codebook store, silent lane repacking interpreted as zero-copy, or unversioned source-local row order.

### B8. Consumer economics and management legibility — report-rs read-only survey

**Question:** Can Cockpit/Hub/Matrix share one authorized answer and vary only presentation demands? Does a wider logical schema remove customer joins or merely hide their source alignment cost?

**Read:** report-rs `docs/plans/rpt-bagop-workbook-convergence-v1.md`, #33 draft, EXO-164 benchmark, EXO-174 decision, EXO-185 read-only audit.

**Fixture:** 2 tenants, same labels, 64 properties, 3 selected fields, status COUNT vs 5-field detail, XVERWEIS duplicate and no-match.

**Report:** same authorized value and provenance across views; forbidden data stays hidden in search/filter/export. Do not implement another workbook store or change report-rs from this research PR.

### B9. Adversarial economics and prior art — independent skeptic

**Question:** Under which workloads does 64×u64 become a regression?

**Countercases:** analytical systems with SIMD-friendly dictionary u8/u16; sparse entities; 60/64 demanded fields where row locality wins; heavily compressed scans; highly mutable snapshots; random remote reads; many small source frames; expensive tag decode; cache pressure under 400 MB–1.2 GB RAM.

**Outputs:** table of counterexamples including storage, working set, bandwidth, time and uncertainties. Do not credit an untested prediction as a measurement. Scope external prior-art references to verified versions/benchmarks.

## C. Shared benchmark protocol

Pre-register data/seed and exact mathematical oracle. Use independent expected results and a disabled-guard test. Separate ingestion, G2L, sealing, bind, cold read, warm fold, output and egress. Report count of source bytes read, cache lines, allocations, masks and lookups. Include semantic parity/refusals before speed results. Compare effective demanded-field densities `1,3,8,32,64` and full/sparse populations. Record architecture, CPU, rustc, optimizer, feature flags, warmup, runs, uncertainty and memory residency. Never quietly time a different operation to flatter u64.

**Progressive kill gates:**
1. Correctness + scope/coordinate witness (if failed, stop).
2. Refusal coverage and oracle independence (if failed, stop).
3. No hidden row materialization or illegal cross-space Gather (if failed, stop).
4. Fair baseline methodology and cold/warm decomposition.
5. Report measured crossover or report *no crossover observed*.

Do not start experimental code from this prompt alone; obtain separate operator authorization for the lab PR.

## D. Coordination across Linear MCP and GitHub

| Purpose | Destination | Explicit behavior |
|---|---|---|
| Agent inbox | [EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id) | read comments since last check; append one new comment per handoff; include real CC session ID |
| Research-run record | [P-EXO-10, lance-graph Agent2Agent](https://linear.app/exo-red) | new `A2A: SOA64-U64 — …` issue if a real run needs a durable record, not a shared file |
| New epiphany (if proven) | P-EXO-12 | new issue only after the existing provenance/admission rules; no speculative epiphany |
| Proposed decision | P-EXO-11 | operator approval separately; this PR is not approval |
| Reader scope | [EXO-46](https://linear.app/exo-red/issue/EXO-46/a1-decide-counted-globalfocus-binding-membrane-for-production) | cross-link existing gate, don't duplicate |
| Existing workbook decision | [EXO-174](https://linear.app/exo-red/issue/EXO-174/decision-rpt-xl-delta-03-the-workbook-is-a-projection-over-the-64k-soa) | preserve scope, propose supersession only in explicit new decision |
| Benchmark comparator | [EXO-164](https://linear.app/exo-red/issue/EXO-164/drf-rpt-lab-xverweis-binding-dr64-shootout-arms-a-oracle-b-lance-c) | separate experimental arm and gate |
| #33 read-only audit | [EXO-185](https://linear.app/exo-red/issue/EXO-185/brief-pr-33-acid-test-dt64-simd-stride-audit-read-only) | no PR/change under its authority |
| Shared-ledger policy | [EXO-186](https://linear.app/exo-red/issue/EXO-186/decision-shared-ledgers-move-to-linear-epiphanies-agent-log-and) | no new shared append-only repo entries |

### EXO-50 message template

```
MESSAGE-ID: <registered-sender-session-id>/<unique-id>
UTC: <actual ISO-8601 timestamp>
FROM: GPT | X | CC
SESSION-ID: <full CC session ID | registered GPT/X ID>
TO: ALL | GPT:gpt-ada-01 | CC:<full session ID>
TOPIC: SOA64-U64 / <GitHub research PR>
TYPE: FINDING | QUESTION | REQUEST | DECISION-PROPOSAL | ACK | RESULT
REPLY-TO: none | <prior MESSAGE-ID or Linear comment URL>
REFS: <research PR, source SHA/file:line, EXO issue(s)>
BODY: <short factual finding + one explicit falsifier + no guessed admission>
ACK: requested | not-required
```

**Handshake:** recipient actually reads the comment; no automatic delivery, timer, daemon or implicit ACK. A suggestion is not operator permission.

## E. Synthesis prompt (Opus only, after independent hand-backs)

> Independently check the cited code and the preregistered fixture; reject claims supported only by diagrams or hypothesis. Compare 64×u64 dense SoA, 32-resident/64-logical virtual lens, narrow/compressed SoA and AoS across scan, point, sparse, full-field, cold/warm, mutation, source I/O, serialization and cache pressure. Reconcile disagreements, preserve the strongest counterexample, and produce:
>
> 1. **Verdict:** supported / falsified / conditional / not tested, separately for H1–H4.
> 2. **Architecture truth table:** present implementation vs laboratory proof vs proposed interface vs prohibited.
> 3. **One ownership map:** NodeGuid source/DR/DT64/Quack/mask-RISC/ndarray/OGAR/report-rs.
> 4. **Minimum admitted witness:** exact issuer, source frame, coordinate, field basis, reader scope, method and value kind; first missing interface.
> 5. **Quantitative cost sheet:** bytes, cache lines, allocations, lookups and timing for each arm, with environmental evidence.
> 6. **Counterexamples:** explicitly where narrow/compressed/AoS wins; no selective publication.
> 7. **Next 3 gates MAX:** scope/coordinate, single typed u64 lane, independent demanded terminal and benchmark; all remain separate proposed PRs.
> 8. **Decision request:** narrow operator question with exact falsifier, never silently granting production execution.
>
> Send a short EXO-50 hand-back with links; write any genuine new epiphany/decision only to its separate Linear board following EXO-186. Leave all source code and production admission unchanged unless a later operator instruction explicitly authorizes it.

## F. Research-PR acceptance checklist

- [ ] All statements separated into verified code facts, existing rulings, test-only evidence, conjectures and pending admission.
- [ ] All GitHub/Linear references point to named real objects.
- [ ] At least one narrow-storage-winning counterexample is in the plan.
- [ ] Independent scalar oracle and disabled-guard check are specified.
- [ ] Same-index condition is witnessed; cross-coordinate paths refuse.
- [ ] U64 codec exactness, tag, overflow and missingness remain open questions.
- [ ] 32 resident vs 64 logical vs 64 resident are differentiated.
- [ ] Every new agent message follows EXO-50 envelope and real session ID requirement.
- [ ] No `Cargo.toml`, Rust source, tests, writers, auth or production settings changed by this documentation PR.
- [ ] Existing EXO-164 / EXO-174 / EXO-185 decisions are not overwritten by conjecture.
- [ ] Operator can authorize or reject future experiments independently.
