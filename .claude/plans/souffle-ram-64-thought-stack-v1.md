# D-SRAM-v1: Soufflé AST→RAM compiler techniques for 64 thought layers

**Status:** PROPOSAL / PREREGISTERED FALSIFIERS (2026-10-19). This document introduces **no runtime, opcode, dependency, storage layout, optimizer result or authorization to implement**. No code or benchmarks are claimed.
**Intent:** assimilate the Datalog-compatible subsets of Atomese, MeTTa and Narsese into one *bound relational program*, learn from Soufflé's AST→RAM compiler, preserve Datafrog as an independent semi-naive oracle, and execute eligible operations through existing RelAlg/quack → R2IL → mask-risc folds over the canonical SoA/mask frame. Integrate with thought Git (#1599, #1605): **at most 63 simultaneously resident speculative thought identities**, not 63 recursive iterations.

## 0. Read-before-modify and precedence

- [Epistemic frontier](../knowledge/epistemic-frontier-capstone.md) (#1597): NodeGuid/address is identity; thought 0 is immutable givens; thoughts 1..=63 are reusable speculative overlays; 10 rung lanes are a different axis; CE64 W is a **thought slot**, not proof provenance or a persisted history ID.
- [Thought Git API](thought-git-api-v1.md) (#1599): branch/diff/rebase/squash/land/merge queue, liveness of sparse overlays, source-bit receipts and DRed are proposals with open gates, **not** a shipped Git VM.
- [Causal Pickaxe / resident Git / temporal horizons](causal-pickaxe-resident-git-lance13-temporal-v1.md) (#1605): Git verbs are porcelain over resident epistemic algebra; logical thought commits differ from durable Lance versions; no per-fold commits; strict historical queries pin actual horizons; derivation ≠ commit ancestry.
- [MeTTa assimilation](metta-datalog-assimilation-v1.md), [council v3](metta-datalog-assimilation-v1-council-v3.md): **council has a VIOLATION banner for gathers / foreign-lane-value index hops**. Earlier preimage / gather passages are superseded. Reuse the D-ASM owner, W1–W3 gates and Datafrog parity tests; do not re-ratify D-ASM here.
- [H14c](d-holo-h14c-thought-first-v1.md) and [H14d](d-holo-h14d-logic-mask-read-v1.md): the hot operator is `Thought(t)[T_a[i] : T_b[i]]`; symbolic logic → admitted address mask → masked payload read → terminal. No foreign key, no runtime string, no row objects or unproved join-as-AND.
- [RelAlg](relalg-v1.md): `crates/lance-graph-relalg/src/ir.rs` currently describes Boolean selection **within one row universe**, not full Datalog rules, multi-variable joins or mu. D-RA-2's ownership question and D-RA-4's recursion driver are **OPEN**; do not silently extend this IR.
- [Semantic candidates RFC](../../docs/research/versioned-semantic-candidates-64-thoughts-rfc.md) (#1604): different claims need different NodeGuids; grammar confidence is not NAL confidence; root overlap does not become independent evidence; deletions require DRed/replay.
- [Lance-graph-nal](../../crates/lance-graph-nal/src/lib.rs): compiled NAL operations already pass through `FoldDialect` / mask-risc and revision requires source-root independence. Compounds/full NAL scope remain separately gated.

**Boundary:** this is a **compiler integration plan**, not an alternative cognitive planner, a competing Datalog AST owner, a second evaluator, a new CE64 interpretation, or a fork of Soufflé's C++ runtime.

## 1. Why AST→RAM aligns with the thought stack

Soufflé's RAM means **Relational Algebra Machine**, an imperative relational intermediate representation. It is not a promise that each AST node becomes a separate memory object, process, or thought layer. Soufflé specializes declarative Datalog through semi-naive rule evaluation into a RAM plan and optionally a compiled C++ program. We want its **analysis and plan transformations**, but the **only production executor remains existing R2IL → mask-risc**.

```text
 Atomese/AtomSpace         MeTTa/Hyperon       Narsese/ONA       Datalog
       |                        |                   |              |
       +---------- parse / admit / typed-refuse / resolve codebooks --------+
                                  |
                     Bound relational rule form
                     (reuse D-ASM's chosen owner;
                        never hot String/JSON)
                                  |
                 Soufflé-inspired AST analyses (cold)
       type/safety, binding modes, SCC, negation strata, magic-set,
              symbolic normalization, proof of co-addressing
                                  |
                     RAM-like logical schedule (cold)
          loops/fixpoint regions, masks, folds, terminals, liveness
                  /        |               \
        co-addressed    proven projection   unrepresentable
           |                 |                   |
     mask algebra      gated mapping      typed refusal
           +-----------------+
                     |
         existing RelAlg / quack / FoldDialect (R2IL)
                     |
           compiled mask-risc + ndarray::simd
                     |
         resident sparse thought overlays t=1..63
          t=0 sealed givens; no fold-time Lance I/O
                     |
        D-TGA resident version/diff/rebase/land gates
                     |
        one canonical Lance writer only on elected artifact
```

**The compiler produces operations *for* a thought, not one thought per operation.** A single thought may run 10,000 semi-naive iterations in its own fixpoint without occupying 10,000 slots. Different active counterfactuals or alternative interpretations need distinct slots only while they have live mutable epistemic state. Do not conflate **thought depth, iteration count, NAL levels, rung levels, Datalog strata and Git commit ancestry**.

## 2. AST/RAM parity subset and semantic firewalls

| Source construct | Compilable intent (conditional) | Required conservation |
| --- | --- | --- |
| Atomese facts / `PresentLink`, pure conjunction / alternatives | Bound relations, mask selection, unions | Explicit variable bindings and canonical statement IDs; open-world absence is not false |
| Atomese `MeetLink`/`QueryLink` patterns | Rule/query plan | Satisfy real binding semantics; do not assume arbitrary cross-coordinate joins are mask AND |
| MeTTa `match`, flat facts and Datalog-fragment `=` rules | Query/monotone rule plus terminal | `BindingsSet` **bag multiplicity** preserved at named exit; nondeterminism cannot be silently deduplicated |
| Narsese inheritance/implication inference rules | Datalog-like premise shapes + typed truth-law tags | NAL F/C and evidential stamps use existing laws; multiple readings and argument order audited against ONA/OpenNARS |
| Recursive Horn rules | SCC / semi-naive delta driver outside mask-risc | Least fixpoint / termination, dedup for cycles; no one-thought-per-iteration |
| Negation | Strictly stratified, bounded closed-world segment | Non-monotone dependencies and retractions tracked; cycle through negation refused |
| Symbolic/higher-order constructions | **Not automatically lowered** | Grounded calls, arbitrary term creation, stateful effects, dependent types, `superpose/collapse` outside supported fragment route/refuse per D-ASM; compiler cannot quietly erase semantics |

**Status of compiler ownership:** `lance-graph-metta::datalog` / quack-bound plans were selected for W1–W3 by the D-ASM council, while D-RA-2 remains unresolved. Treat the shared AST and RAM-like schedule as **logical contracts only** until their owning plan approves a placement. A complete second AST or VM is forbidden.

**Set-versus-bag warning:** pure Datalog relations are sets; Hyperon answers can be bags. Keep support multiplicity through a typed terminal or refuse. Never use a presence popcount as the number of proofs, and never use an evidence-source popcount as the number of thoughts.

## 3. Source harvest: what to transcode and what to decline

Port **algorithms / invariants** from the upstream Soufflé reference after a versioned source inventory, not C++ object layout or engine state:

| Soufflé reference seam | Use in our cold compiler | Refusal / measurement |
| --- | --- | --- |
| `src/ast/transform/MagicSet.cpp` | Bind-aware rule specialization, propagate apertures and question constraints backward | Only if equivalence of the query answer + witness set holds; magic sets are not always faster |
| `src/ast` analyses + transformation passes | Rule safety, types, binding modes, dependency SCCs, strata, normalization | Reject unsafe variables and negation cycles; never equate strata with thought numbers |
| `src/ast2ram/seminaive/` | Generate delta-driven recursive regions from rule SCCs | Driver remains external to mask-risc; count read/write effects per iteration |
| `src/ram/transform/MakeIndex.cpp`, `ReorderConditions.cpp`, `EliminateDuplicates.cpp` | Translate to cost-based mask/aperture order, selected fold terminals, lawful dedup | **Index choice is advisory only**; no new trie/foreign gather; dedup may violate bag semantics |
| Soufflé RAM execution and C++ synthesis | Reference for expected behavior and oracle comparisons | **Do not port** native RAM interpreter, C++ codegen, parallel runtime, relation object model or duplicated storage |
| Provenance extensions | Source for proof/replay test cases | Git lineage is not causal proof; NAL source roots remain distinct from rule derivation provenance |

References: https://souffle-lang.org/docs/translate/ , https://souffle-lang.github.io/magicset , https://github.com/souffle-lang/souffle/tree/master/src/ast2ram and https://github.com/souffle-lang/souffle . Source is UPL-1.0; preserve notices for any adapted material and track file/revision/licensing in the harvest. **No upstream source is copied by this PR.**

`AdaWorldAPI/datafrog` is the existing Rust semi-naive comparison path (D-MTP-0 measurements, D-ASM oracle). Keep the fork as oracle, not as a hidden new production evaluator. Match the evaluated query and category (set, bag or explicit count), not merely cardinality. A failure in one oracle is logged, not averaged away.

## 4. A RAM-like thought scheduler, without allocating 64 AST nodes

The proposed schedule is a compile-time description (names below are **concepts, not shipped Rust APIs**):

```text
ReadPinnedFrame(v, schema_seal, aperture)
RunFold(t, compiled_plan, read_set, write_set)
Fixpoint(t, SCC, delta, stable, iteration_budget)
Branch(base_commit, semantic_candidate)
Resolve(candidates, evidence_policy, strict_horizon)
Rebase(changed_read_set -> retract/rederive)
SquashOrDiscard(t, retain_receipt, free_slot)
Land(elected_artifact -> one canonical writer)
```

A **thought** is an isolated writable epistemic overlay with identity and lifecycle. A **RAM temporary**, mask scratch slot, relational intermediate or Datalog delta is **not** a thought. A **stratum** orders dependencies and may execute within one thought; only genuinely independent or conflicting mutable interpretations require separate thought slots. Even nested syntactic scopes need not imply nested resident thoughts.

Before allocating a speculative thought, run a conservative *lifetime/interference analysis* on the **speculative thought dependency graph** (not AST variable graph alone):

1. Determine each live hypothesis's pinned base, required reads (including negative reads), proposed sparse writes, independence/provenance, prospective downstream consumers, and whether its state must remain independently mutable.
2. Reuse a free slot only once **all consumers are finished and a replayable receipt survives**, or the thought was explicitly discarded. An active derivation can never be silently overwritten.
3. Prefer reuse, serial scheduling, or explicit semantic consolidation into an existing thought **only when the D-TGA merge and evidence gates certify it**. Consolidation is not simply bitwise OR; conflicting candidates survive.
4. If all 63 speculative slots are simultaneously live and non-consolidatable, return a typed `BranchesFull` / scheduling refusal. **Do not evict thought 0, alias two live thoughts or allocate a 64th speculative W value.**
5. Unbounded recursive depth is not limited by the number of slots; it is separately constrained by dataflow fixpoint/termination budget. If >63 *live alternatives* are essential, scheduling must refuse or explicitly wait for legal retirement, never claim that a loop automatically solves capacity.
6. Historical identities do **not** use the recyclable 6-bit thought slot as a durable witness. PR #1605's logical commit IDs, source roots, pinned `DurableRef` and receipts carry history outside live slots.

The target budget is **`0 <= live_speculative_thoughts <= 63`** (64 total including givens), *not* "63 possible thoughts over all time", *not* "64 independent evidence sources", and *not* "64 steps of reasoning".

## 5. Physical lowering gates

A RAM plan may use existing operations only after proof of its coordinate semantics:

- **P0 symbolic:** normalize Boolean fragments and shrink apertures before any payload read. Return `Empty` / `All` where proven.
- **P1 co-addressed:** `T_a[i]` and `T_b[i]` share the same sealed allocation and ordinal; select `A & P_a & P_b` then read survivors at `i`. Different versions or codebooks cannot AND without an independently verified mapping.
- **P2 canonical pre-arrangement:** a relation was *already* co-located or legally projected into the same coordinate frame at admission/seal, with schema/codebook witness. Charge the build/maintenance cost; **never** sneak an extra foreign-lane dereference into the hot executor.
- **P3 not expressible:** an unaligned `path(x,z) :- path(x,y), edge(y,z)` is **not a mask AND**. Refuse this proposed physical lowering and record a capability gap until a correct, separately authorized layout/projection is demonstrated. Datafrog/Soufflé may compute the oracle off-path, never invisibly substitute in production.
- **P4 recursive:** call one compiled `Program` per frontier delta via a recursion driver that does not exist yet (D-RA-4's d′ preimage driver is a banned gather shape, so D-SRAM-4 is BLOCKED); datafrog's `recent/stable` is reference semantics, not a new runtime in mask-risc.
- **P5 NAL:** weighted frequency/confidence stays an existing typed fold with source independence; derivation support is not a Boolean membership bit.
- **P6 Git:** `status`, `diff`, `blame`, `rebase`, `pickaxe` lower through existing resident selections. `land` follows G1..G7 and the canonical single writer. Compile-time scheduling does not certify a semantic conclusion.

No new opcode, key format, hot String, runtime JSON, row allocator, per-fold Lance transaction, implicit materialized join or CE64 bit reassignment. New primitives require an insufficiency proof and a separate review under the existing wardens.

## 6. Measurable implementation sequence (gated, not authorized)

| ID | Phase and deliverable | Exit / red gate |
| --- | --- | --- |
| D-SRAM-0 | Inventory pinned Soufflé AST/ast2ram/RAM passes; map to D-ASM/D-RA/D-TGA owners and current shipped code; establish source notices | Explicit evidence for every mapped pass; no code imported; list gaps and superseded gather plans |
| D-SRAM-1 | Define a **conceptual bound AST → RAM-like schedule** for pure conjunctive selection, including bag vs set, address seal and negative dependencies | Same canonical answers and read sets on hand-built fixtures; every unsupported operation typed-refused |
| D-SRAM-2 | Prototype cold optimizer rewrites: SCC/strata, magic-set/aperture propagation, predicate ordering, safe dedup; keep optimizer out of hot path | `optimized == unoptimized == source oracle` for query **and** evidential roots. Disabling a rewrite must leave the answer unchanged and only remove the optimized plan shape (assert the shape is absent, never a wrong answer). Negative controls disable a **semantic correctness guard** (e.g. SCC stratification of a negative dependency, dedup under bag semantics), and those must fail (2026-10-19 correction, #1612 review) |
| D-SRAM-3 | Investigate co-addressed physical selection and coordinate proof; exact R2IL `Program` reproducibility across eligible front ends | P1/P2 agree with scalar reference; P3 always refuses; no foreign-gather or phantom equal-row assumption |
| D-SRAM-4 | **OPEN / BLOCKED (2026-10-19, #1612 review).** The earlier target, the "already-authorized d′ driver", is D-RA-4, which STATUS_BOARD marks as a gather violation (preimage hop). There is no lawful recursion driver to hook into yet; this phase waits for a co-addressed recursion design under D-HOLO H14c/H14d, not invented here. Once one exists: compare with Datafrog, then Soufflé | Identical least fixpoint for set workloads, terminated cycles, consistent bag categorization; one thought remains live across many iterations |
| D-SRAM-5 | Analyze *speculative thought* liveness and nested-alternative lifetime with the existing 0+63 slots; integrate with D-TGA when implemented | Exhaustion refused, deterministic re-use, negative-read strata safe, no tombstoned branch reused before proof receipt |
| D-SRAM-6 | Candidate resolution and Git porcelain/admission/rebase/land under pinned temporal horizon; source-aware DRed/replay | Deleted support removes only dependent proofs; independent proof survives; stale-horizon and fake-causality controls red |
| D-SRAM-7 | Evaluate hierarchical **chunk** promotion as a compilation optimization only after correctness gates | Cold source graph + versioned receipts can unfold; no confidence inflation; compare physical reads/bytes/ops/time, not just row count |

**Release order:** D-SRAM-4 is blocked until a lawful (co-addressed, non-gather) recursion driver is designed and approved; D-RA-4 is not that driver; D-SRAM-5/6 depend on D-TGA-1/3 plus #1605's temporal/provenance gates. D-SRAM-2 is research/probe until D-ASM's Datalog AST placement and D-RA-2 are resolved. This plan **adds no new D-ASM/W wave implementation authorization**.

## 7. Measurement and decision rules

The independent arms are **Datafrog fork**, **Soufflé** (external CLI, never production dependency), **unedited current mask-RISC**, **planned optimized lowering** and **scalar reference**. Compare on the *same pre-bound canonical facts*, same query shape and result category; record time to bind/compile separately from resident warm execution. Inputs include 64k same-allocation rows at densities 0.1/1/10/50%, recursive chain/cycle, multiset fan-out, stratified negation and multi-horizon revisions.

Report: schema seal, CPU SIMD tier, tool/source revisions, cold optimization time, hot execution median/distribution, number of folds, address words scanned, payload bytes, materialized tuples, allocation count, active thought high-water mark, memory, result identities, root/proof signatures and refusal counts. **No assumed speedup.** Magic-set, predicate reordering and compression are permitted only when a correct oracle equivalence test passes. A good mask win must also win after all sealing/projection costs required by the workload are charged.

`docs/research/souffle-ram-64-thought-falsifiers.md` is the preregistered **symbolic** fixture/kill-condition map. Its cases are **not executed tests**.

## 8. Stop conditions / review questions

Stop, file a separate issue and do not implement if: an optimizer assumes same-row across differently sealed populations; a join needs gather or a hidden foreign-index read; a bag collapses to a set; two evidence roots overlap yet confidence increases as though independent; a negative cycle sneaks through scheduling; an over-capacity thought slot aliases a live one; a merged chunk destroys reconstructible alternatives; a pinned as-of query sees a newer version; or a new opcode/VM appears without prior gate.

**Questions for review:** (1) Can Soufflé's magic-set/adornment analysis be expressed entirely as compile-time aperture restriction under existing H14d operations? (2) What is the smallest well-typed RAM-like *schedule view* that requires no new IR owner while D-RA-2 remains open? (3) Does proof-aware thought liveness improve reuse without forcing premature squash? (4) What fraction of real MeTTa/Atomese/Narsese fixtures is truly co-addressable rather than requiring a forbidden hop? (5) Is compiling the same bound program through different front ends byte-identical once bag and truth-law tags are explicitly separated?
