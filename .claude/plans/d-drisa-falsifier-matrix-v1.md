# D-DRISA falsifier matrix: local register, Abakus ALU, async driver, cognition and economics

> **2026-11-04 · PRE-REGISTERED / UNEXECUTED.** Every row below is a **proposed** test, unless a precisely matching earlier PR is cited as PARTIAL evidence. Passing previous P2/P3 correctness is **not** completion of remote or locator falsifiers. No performance value is fabricated.
>
> Owner queue and exact PR boundaries: [staged charter](d-drisa-staged-pr-charter-v1.md). Architecture: [D-DRISA-0](d-drisa-remote-register-shader-v1.md). Tracking: [EXO-153](https://linear.app/exo-red/issue/EXO-153), coordination: [EXO-50](https://linear.app/exo-red/issue/EXO-50).

## A. Method for judging a gate

Each experiment must name (1) immutable source/version and original independent scalar oracle, (2) exact input, including scope and truth mode, (3) expected answer **and** explicit refusal, (4) disable/mutation that should break an assertion, (5) warm/cold costs, allocations, bytes and source receipts, (6) owner and commit SHA. Every successful refusal needs a paired successful authorized control. An oracle that calls the same map, source-locator helper or mask program as the candidate is **not independent**.

**Truth:** Canonical corpus fixture with source global IDs and known typed values generated independently of the register/packing code; Boolean support and Bag multiplicity calculated separately; Kleene UNKNOWN explicit for missing value; value 0 distinct from absence. Source leases and permissions are fake-but-attested only for the first *local* stages; remote authorization is not proven by a fake token. A null path does not silently become FALSE unless exactly the admitted SQL WHERE rule says so.

**Measurements:** absolute p50/p95/p99 and variance, Q=1/2/8/16/64/256; single DR64 / DR256 / multiple tiles; 0, 0.1, 1, 10, 50, 95, 100% selection; clustered/scattered/fragmented; large and small result; in-memory, page cache and disk distinct. Include request/result bytes, logical and physical rows, *actual* pages read if provider counters support it, copies, allocations, CPU cycles, scratch footprint, prefetch hits/misses. No arbitrary 1ms remote-SLA claim.

## R. DR register, identity, lease, locator: R1–R16

| ID | Exact fixture / independent oracle | Deliberate disable / expected refusal | Owner / status |
| --- | --- | --- | --- |
| R1 | Two frames map same 64 identities to permuted local positions; oracle by global ID compares COUNT/EXISTS/mask | Reuse old ordinal without frame seal → wrong answer/refusal | DT64/map-risc, PR1 LAB |
| R2 | DR256 tile with nontrivial end tail and one-hot at final active bit | Disable tail clear → false extra bit | mask-risc, PR1 LAB |
| R3 | 1M identities over multiple DR256 tiles with two disjoint selections, a shared global id | Join on local index rather than global identity → oracle divergence | DR, PR1 LAB |
| R4 | MAP-only COUNT over pre-bound population plane; input source value array not mapped | Any payload-lane read or source I/O fails; count remains exact | DR/Quack, PR1 LAB |
| R5 | ClassView demands exactly 3 of 64 fields in physical order permuted; oracle maps field IRIs | Physical ordinal shortcut or field label shortcut → refusal/wrong sum | ClassView, PR1 LAB, #1950 partial |
| R6 | Authorization allows 3 fields; ask for fourth and ask for unknown IRI | No widen-to-all; Unauthorized / UnknownProperty | DT64/security, PR1 LAB |
| R7 | Same map/frame, 2 methods, Q=1,2,8,16,256; source unchanged | Force reroute per call; measure extra cost but answer parity | DR, PR1 LAB; #1950 bundled reference |
| R8 | One frame remains, canonical value mutates and a derived WHERE mask is reused | Source-value witness changed → mask must refresh/refuse; frame may remain valid | source/Quack, PR5 BLOCKED |
| R9 | Canonical source compacts, reorders, splits fragment or changes stable row ID under old DR frame | Reuse stale physical locator → typed StaleLocation, never wrong data | D4/source, PR5/6 BLOCKED |
| R10 | Focus slot recycled to another tenant while old remote job queued | ABA / scope check bypass must turn test red | EXO-46/151/security, PR5 BLOCKED |
| R11 | Source lease expires after binding, before borrowed read | No use-after-release; stale typed refusal, not empty result | source owner, PR1/5 GATED |
| R12 | Memoized locator hit → evict → miss on same still-valid source | Answer invariant, only lookup/cost changes; no persisted truth in receipt | EXO-151, PR5 BLOCKED |
| R13 | Two readers share leased same canonical bytes; a writer invalidates during one execution | Correct pinned snapshot or typed refusal, no torn/read-after-free | source/lease owner, PR5 BLOCKED |
| R14 | Old frame read with historical SoA snapshot available, then snapshot dropped | Correct old value with valid snapshot, then HistoricalSourceUnavailable | DR/source, PR5 BLOCKED |
| R15 | DR bool mask, ClassView field mask, Lance row-address mask all same bit width but different meanings | Deliberately pass one as another → typed failure; no raw transmute | DT64/Lance, PR1/6 |
| R16 | Source location receipt and auth capability are removed, without removing sealed DR relationship | Relation truth unchanged; only value read refusal/slower rebind | EXO-46/89/151, PR5 BLOCKED |

**Important measured predecessor:** #1950 P2 `corrections` uses `Binding::prepare`, which already copies `gather_lane` payload values. Its Q≥2 win is not R7/R12 proof of source-location memoization. #1951 reports no production locator-lease host; PR5 cannot become production by inventing one.

## A. Abakus / shader ALU: A1–A8

| ID | Exact fixture / oracle | Disable / negative control | Owner / status |
| --- | --- | --- | --- |
| A1 | k-of-n with n=1,2,4,5,16,24; k=0,1,ceil(n/2),n,n+1; oracle per identity summing Boolean input truth | Wrong threshold boundary and k>n must be caught | ndarray/Quack, PR3 LAB |
| A2 | 64 independent identities: one-hot, alternating, 0/full, upper-tail/random, high/low carry collisions | Swap significance carry plane or shift by one *identity* → fail | ndarray, PR3 LAB |
| A3 | Same expression over DR64, DR256 and multi-tile frames with 3/64 demanded fields | Use physical lane ordinal as field position → wrong answer | ClassView/Quack, PR3 LAB |
| A4 | Packed 4×u8 in u32 with zero/present/absent sentinel, threshold 0,64,170,255 and out-of-range | Ordered-u8 comparator must not be claimed shipped; unauthorized codec refuses | ndarray, separate PR6 proposal, #1950 gap |
| A5 | SQL Kleene UNKNOWN for NULL, empty population, logical NOT, present zero | Replacing UNKNOWN with false *before* a permitted WHERE terminal fails | Quack/semantics, PR3 LAB |
| A6 | Boolean support vs Bag multiplicities 0,1,2,255; same support different total count | Count support instead of multiplicity → fail | Quack/RelAlg, PR3 LAB |
| A7 | Native AND/OR/ternlog vs Abakus on 2-mask AND and a true ≥3-of-5 expression | Counter extra work must not be default; compare cold+warm, disable adder carry | ndarray/mask-risc, PR3 LAB |
| A8 | 4 ISA categories where supported (scalar, AVX2, AVX-512, NEON/wasm); seeded independent oracle | Backend wrong tail/overflow/sign must turn test red; skip unsupported hardware honestly | ndarray owner, PR3 LAB |

The `U24x5` packing microtile and `Abacus<BITS>` examples remain **examples-only** per #1951 census, not a shipped API. `half_add_mask`/`full_add_mask` are the existing legal words. The packed-u8 ordered comparison is **separate** from k-of-n carry arithmetic.

## N. Async driver and transport: N1–N14

| ID | Exact fixture / oracle | Deliberate break and expected status | Owner / status |
| --- | --- | --- | --- |
| N1 | Same typed COUNT locally synchronously and via loopback queued worker on one immutable source | Mismatch result or effective witness fails | runtime, PR2 LAB |
| N2 | 64 admitted concurrent requests with max_inflight=8 and bounded scratch; same oracle | Overfill queue → ResourceExhausted, bounded RSS | runtime, PR2 LAB |
| N3 | Two jobs reuse request_id but differ in operation digest or scope | Replay/Conflict refusal; never return first job's answer | security, PR2 LAB |
| N4 | Cancel before admit, after admit, queued, already running and after completion | Cancellation cannot fabricate FALSE/0; CancelAck not rollback | runtime, PR2 LAB |
| N5 | Timeout and network disconnect after job ran but before reply | Indeterminate, retry only same immutable snapshot and digest | transport, PR2/4 LAB |
| N6 | Authorization revocation while request queued and while running | Out-of-scope data must not escape; lease/policy decision explicit | security, PR2/4 GATED |
| N7 | Untrusted client changes class/schema revision but keeps same frame and result shape | SchemaMismatch, not automatic default | DT64/security, PR2 LAB |
| N8 | Untrusted program requests forbidden opcode/F### or invalid scratch slot | UnsupportedProgram/UnboundSlot, not arbitrary native evaluation | R2IL/mask-risc, PR2 LAB |
| N9 | Execution returns MaskReceipt then client reuses on a different frame, source, scope or expired TTL | Reject wrong binding; do not read stale scratch slot | register/results owner, PR2/7 GATED |
| N10 | B1: client fetches source data; B2: remote data-local executes the same query; compare outputs and bytes | Force a payload column on B2 wire → violation, even if faster | transport, PR4 LAB |
| N11 | B2 server sends only scalar; mask output size > maximum allowed result | ResultTooLarge or bounded receipt, not silent row transfer | transport, PR4 LAB |
| N12 | Delayed request after source overwrite or compaction | WrongSourceVersion, never silently rebase operation | source/security, PR4/6 GATED |
| N13 | Two source owners with coincident local DR ordinals, no cross-owner join witness | Refuse unsupported cross-owner operation; no accidental join | D4/EXO-46, PR4 LAB |
| N14 | Network partitions with 2 hosts, duplicate completions, bursty load, misrouted locality hint | No cross-tenant data, bounded resources and measured timeouts | transport/security, PR4/8 GATED |

**Transport baseline:** packet capture or explicit transport byte counters must show command and result bytes. Sending only a handle is not enough if the server secretly fetches the entire source into a new buffer.

## C. Cognitive semantics and continuation: C1–C10

| ID | Fixture and oracle | Deliberate break / refusal | Owner / gate |
| --- | --- | --- | --- |
| C1 | #1948 German grammar `geben/gefallen`, identical dependent relation and Case, distinct head frame; independent annotation oracle | Confuse frame/candidate role → mismatched result | grammar/DT64, PR7 GATED |
| C2 | #1948 cold co-addressed head-frame producer, source lexicon language + revision changed while DR frame stays | Revision-only check insufficient if language changes; stale producer refused | EXO-46/140, PR7 GATED |
| C3 | Two read-only methods consume the same scope-bound WHERE/result plane in *one* admitted program | Scratch slot reused across epoch → StaleSlot, not a valid continuation | R2IL/Quack, PR1/7 |
| C4 | Bounded remote result-mask continuation across *two* requests, same frame/scope/source and live result sink | Without admitted result sink, refuse; no global mask cache | EXO-46/result owner, PR7 BLOCKED |
| C5 | One grammatical input gives competing Direction/Purpose functions | Preserve both candidate masks and UNKNOWN; no arbitrary winner | grammar owner, PR7 |
| C6 | XLOG simple acyclic 3-way join with independent enumerative Boolean oracle | Row/tuple reconstructed in hot path or foreign gather bypass → refusal | XLOG/Quack, PR7 |
| C7 | NAL/polyfill truth mode, Boolean support versus Bag and epistemic status | Mislabel support as weighted truth/causal fact → refusal | NAL semantic owner, PR7 |
| C8 | Any unadmitted recursive request (`EXO-73` open) | typed UnsupportedRecursion, no hidden fixpoint driver | EXO-73, PR7 |
| C9 | THOUGHT candidate with witness and provenance, request asks to write WORLD | no hydration/Rubicon gate → NotAdmitted, no side effect | EXO-78 / Thought-Git, PR7 |
| C10 | Two methods reference one another through WHEN, or require hop-shaped `EqU32Via` | detect cycle / EXO-140 unclassified foreign lookup, not extra interpreter | DT64/EXO-140, PR7 |

## P. Cost and experimental validity: P1–P9

| ID | A/B comparison and counted cost | Falsifier / decision consequence |
| --- | --- | --- |
| P1 | B0 local borrowed vs B1 remote fetch vs B2 remote instruction+scalar on same source | If B2 does not beat B1 on whole-query time/bytes for a meaningful region, no throughput claim |
| P2 | Q=1/2/8/16/64/256 on unchanged source: routing per query vs one permitted binding | Savings not present without copied payload? separate address-only result, no bundled inference |
| P3 | Borrow direct vs memoized location vs prefetch, input locality clustered/scattered and false lookahead | If warm borrow loses 2.3–2.5× at 50% as in #1950, investigate, not mandate gather |
| P4 | 0.1/1/10/50/95% selectivity, one/4/16 projected columns, range counts varied independently | `range_count × 4096` is a bound/proxy, not actual pages touched; choose measured source costs |
| P5 | Cold/warm page cache and real disk IO, plus S-INGRESS tree build and attest costs | Report physical source bytes, not just logical selected rows or lane hits |
| P6 | Identical end-to-end semantics with and without repeated mask egress, bounded result receipt if admitted | A hidden remote mask store, wrong scope or unknown lifetime fails |
| P7 | End-to-end k-of-n Abakus vs native Boolean on small and large K, cold+warm | A single faster microkernel does not prove a driver speedup |
| P8 | Two-process 1-host vs two-host with latency, TLS/auth and congestion varied | If network cost dominates or bytes moved are unchanged, report NO BENEFIT |
| P9 | Reproduce on another ISA/source version with disable tests, raw logs + SHA | No cherry-picked fastest cell; uncertainty and hardware explicitly reported |

## Existing evidence crosswalk (never over-credit)

| Earlier gate | What can be credited | Still missing |
| --- | --- | --- |
| #1949 L9 / C4 | #1950 Lance A-full/B-allow/B-block/C-range semantic parity, ingress as copied, D4 explicit | No real DR-to-row-id link, no remote driver |
| #1949 L10 / C2 | #1950 1M/10M multi-tile composition | Bag multiplicity separate and cross-source identity not fully falsified |
| #1949 L1/L2 / C1 | #1950 prepared binding reused Q≥2 | Bundled gathered values; **not** memoized SoA locators |
| #1949 A3/A4 | #1950 packed layout C2 cost/correctness; #1951 lab-only U24x5 census | Native ordered-u8 kernel and Abakus k-of-n still not measured |
| #1951 P3-1 | range fragmentation heuristic with local 63-test suite | physical pages touched, 95% cluster crossover, transport selection |
| #1948 grammar | 16/16 lab + co-addressed candidate | production producer admission, cross-call lifetime, source scope |

## Verdict format required for every future PR

```text
D-DRISA-<stage>-<gate>
SOURCE: repository path:line + commit SHA
OPERATION: semantic program / witness / truth mode
FIXTURE: independent versioned data source, expected answer/refusal
ARM: B0/B1/B2/B3/B4/B5/B6 and disable variant
MEASURED: median, p10, p90, p95, allocations, CPU, wire, page/disc bytes
ORACLE: independently named, never the same shared function
STATUS: PASS | FAIL | NOT-RUN | GATED | NOT-COMPARABLE
ADMISSION: owner and EXO issues, explicit operator decision if needed
NO-GO: source violation / unauthorized path / stale witness / semantics drift
```

**Refusal boundary:** If an experiment cannot use an admitted owner, answer `BLOCKED_EXO` or `BLOCKED_D4` with a precise source citation. Do **not** fake a source identity, wire pointer, hash-table stand-in or ask a microbenchmark to ratify the architecture.
