# D-DRIVER-ZC independent falsifier matrix: virtual buses, source borrow, foreign bind, storage and enforcement

> **2026-11-04 · ALL TESTS BELOW ARE PROPOSED, NOT RUN.** Results from #1955, #1956 and #1950 are explicitly identified as *predecessor* lab evidence; they do not pass these new virtual-bus/call-graph/storage obligations. A matching answer alone cannot certify a no-copy path. [Master](d-driver-zc-virtual-bus-contract-v1.md) · [PR charter](d-driver-zc-staged-pr-charter-v1.md) · [EXO-154](https://linear.app/exo-red/issue/EXO-154).

## Method and independent reference

**Source fixture:** Independently generated table keyed by global ID, with three uncorrelated typed x/y/z values, missing/unknown/present-zero cases, relation edges, explicit snapshots and scope. Separate oracle enumerates identities and applies declared semantic operators; it **never** calls the candidate driver, reuses the candidate binding, asks `TileMap::local` for oracle answers, or receives candidate-created derived masks.

**Disable protocol:** Every cited deliberate violation must yield **an observable different answer or typed refusal**, not merely "test still passes" or a panic. For refusal tests, a positive control under a valid witness must succeed. Anti-vacuity: nonzero and unequal expected outputs where a bad result could otherwise escape.

**Physical evidence:** source buffer identity/pointers + address source ownership, no-value-vector build in candidate code, source read/copy counters and alias invariants, allocations (including stack-copy caveat), scratch/result and mask ingress separated, number of tiles/ids touched, peak resident bytes, page-cache/disk separately, p50/p95/p99 with variance, Q=1/2/8/16/64/256.

**Notation:** `V` = bus algebra, `Z` = resident no-copy, `F` = foreign route/identity, `S` = async/storage, `E` = enforcement/economics. Tests below do **not** supersede D-DRISA's R/A/N/C/P pre-registration.

## V. Virtual bus semantics and composition (V1–V12)

| ID | Independent fixture / expected | Disable falsifier and refusal | PR |
| --- | --- | --- | --- |
| V1 | Bound same-identity XY view: x=2,y=7,z=13, independently oracle-defined operator F(x,y) | swap x/y deliberately: noncommutative F changes answer | ZC-2 |
| V2 | XZ view on same IDs and same source X; expected differs from XY due to independent Z values | substitute Y lane for Z → wrong answer | ZC-2 |
| V3 | XX view reads one canonical X twice as logical operands | physical duplicate X vector allocated or wrong alias → fail | ZC-2 |
| V4 | YY and ZZ on different fields, distinct negative, zero and positive values | alias wrong lane or mix field identities → fail | ZC-2 |
| V5 | All XX/XY/XZ/YY/ZZ on same sealed DR64 and DR256 populations | silently use local ordinal from different frame → divergent oracle/refusal | ZC-2 |
| V6 | XY and XZ share the same source X pointer and witnessed tile mapping, Q≥2 | create different copied X arrays → fail representation checker | ZC-2 |
| V7 | Two-step XY→XZ program with intermediate mask held in existing in-program scratch, output COUNT | force object-row intermediate or stale scratch epoch → fail | ZC-2 |
| V8 | Three-step XY→XZ→ZZ and independent scalar expression evaluated by global ID | materialize intermediate or reorder step/frame → fail | ZC-2 |
| V9 | YZ symmetric control, foreign Y excluded from local same-identity rule | accidental treat foreign id as Y[local_i] → typed refusal | ZC-2 |
| V10 | ClassView field mapping through physical permutation, predicate_iri not ordinal/label | ordinal or name shortcut silently selects wrong column → fail | ZC-1/2 |
| V11 | Boolean support vs Bag multiplicity on one bus, Kleene UNKNOWN retained for absent value | force Boolean collapse/UNKNOWN=false → oracle divergence | ZC-2 |
| V12 | Requested N outputs vs aggregate scalar over N identities | false O(1) output claim disproved by counted bytes; explicit egress required | ZC-2/7 |

## Z. Resident no-copy membrane and lease (Z1–Z10)

| ID | Independent fixture / expected | Disable / failure state | PR |
| --- | --- | --- | --- |
| Z1 | Existing #1955 Soa under known source/frame, two demanded borrowed lanes, direct pointer alias, exact fold | swap to owned `Vec<i32>`: CI/anti-copy instrumentation must detect | ZC-1 |
| Z2 | Map-only COUNT and EXISTS after value buffers unmap/unavailable | any canonical value-lane read → forbidden; valid answer remains | ZC-1 |
| Z3 | 3 demanded fields of 64, differing physical order and field basis | full 64-field gather or forbidden field → fail | ZC-1 |
| Z4 | Same DR frame and source layout, one value-only write | map-only valid; derived predicate stale until new value witness | ZC-1 |
| Z5 | New SoA incarnation resets numeric revision to 0 | numeric revision match alone must not grant stale read | ZC-1 |
| Z6 | Same IDs but permuted source tile local order | without source-owner layout witness, wrong oracle; with witness refuse | ZC-1 |
| Z7 | Source buffer released while a candidate view escapes callback | Rust UI compile-fail / lifetime proof, no use-after-free | ZC-1 |
| Z8 | A stack-based copy of value tuples and a heap owned column both attempted inside admitted module | heap allocation count alone must not falsely certify; policy/lint rejects both | ZC-1/6 |
| Z9 | An eager value gather happens inside fake "prefetch" stage | count canonical duplicate bytes / explicit no-copy refusal | ZC-1/3 |
| Z10 | Candidate returns scalar/mask while scratch/result allocator is active | allowed scratch/result bytes separate from value-copy counter; do not falsely fail | ZC-1 |

## F. Local/foreign addressing and bind amortization (F1–F9)

| ID | Independent fixture / expected | Deliberate break / status | PR |
| --- | --- | --- | --- |
| F1 | Co-addressed DR tile/local population, 64/256/multi-tile, repeat Q=256 | any hidden `MapTable::locate` per method fails route counter | ZC-3 |
| F2 | Global IDs pre-bound once to 256 DR256 tiles using #1955 route, oracle by IDs | route per call loses time; honest ≥1ms FAILED-GATE if applicable | ZC-3 |
| F3 | Two populations include same global ID at different local ordinals | join on local position → wrong answer/refusal | ZC-3 |
| F4 | One legal foreign relationship with independently enumerated edges and source-owner permit | invalid/unclassified `EqU32Via`/gather hop refused under EXO-140 | ZC-3 BLOCKED |
| F5 | Same foreign source/frame and reused binding for Q=1/2/16/256 | copy values to fake "accelerator" cache → forbidden | ZC-3 BLOCKED |
| F6 | Foreign source compaction under original DR coordinate frame | stale row location/source index seal → source-lease refusal | ZC-3 BLOCKED |
| F7 | Foreign source scope revoked while owner still hosts data | scope mismatch fails without returning old mask | ZC-3 BLOCKED |
| F8 | Source index/mapping receipt evicted but underlying relation unchanged | relation answer invariant; admitted rebind or refusal, no lost truth | ZC-3 BLOCKED |
| F9 | Two owners have identical integer tile/local coordinates | never join identities merely because coordinates coincide; refusal | ZC-3 BLOCKED |

## S. Async, disk, source isolation (S1–S8)

| ID | Independent fixture / expected | Disable and decision | PR |
| --- | --- | --- | --- |
| S1 | In-process async worker receives typed bound query, source unchanged | same scalar oracle; no remote/payload claim from loopback | ZC-4 gated |
| S2 | Value changes while a descriptor is queued; map-only vs value-scoped methods | value operation stale refusal; map-only may remain valid | ZC-4 gated |
| S3 | Caller supplies matching FNV digest for foreign layout, executor knows actual source | reject caller's untrusted "authority"; source-owning witness wins | ZC-4 gated |
| S4 | Two-process request contains only method/witness; source owner computes and returns COUNT | packet/transport counter detects any unexpected source field payload | ZC-4 gated |
| S5 | Real Lance storage 1M/10M, selected 1/16 fields, 0.1–95% distributed rows | D4 missing must produce BLOCKED_D4, not accept handwritten row IDs as integrated truth | ZC-5 blocked |
| S6 | Fragment compaction, stable row-id option, source revision and local frame mapping | forged `_rowid` or wrong page silently accepted → fail | ZC-5 blocked |
| S7 | Page cache warm vs disk cold, clustered vs scattered; physical byte counter | cannot claim disk-level no copy from resident-only borrow evidence | ZC-5 blocked |
| S8 | Partial/missing source pages, mid-execution cancel and error | typed I/O error or indeterminate, **never** zero count or empty mask | ZC-5 blocked |

## E. Enforcement and economic proof (E1–E8)

| ID | Independent test / expected | Negative control / no-go | PR |
| --- | --- | --- | --- |
| E1 | Compile-fail: attempted source buffer ownership through the sealed migrated consumer | bad direct accessor must fail or be policy rejected | ZC-6 |
| E2 | Compile-fail: borrowed value view escapes source lease or wrong lifetime | UAF path rejected at compile time; no `unsafe` bypass | ZC-6 |
| E3 | Compile/policy failure: inject `to_vec` field copy into one selected hot-path module | CI fails; do not flag legitimate descriptor cloning | ZC-6 |
| E4 | Enumerate reachable data-access sites from declared caller roots and classify migrated/excluded/unknown | missing dynamic edge means INCOMPLETE coverage, not "17k proved" | ZC-6 |
| E5 | Reference tuple oracle independently computes global-ID truth | test oracle equals production implementation by sharing function → fail independence | ZC-6/EXO-109 |
| E6 | Same query B0 local borrowed vs B1 client payload fetch vs B2 source-local instruction, full total time and bytes | equal/greater B2 end-to-end time: no speedup claim; report why | ZC-7 |
| E7 | Same fixed-address XY/XZ operation before/after lens fusion; count payload copies and scratch | zero allocations but wrong frame or hidden copies → fail; memory metrics separate | ZC-7 |
| E8 | Whole program output N rows, or all pairs N², while claim promises O(1) time | output lower bound contradiction -> reject broad theorem claim | ZC-7 |

**Total:** 12 V + 10 Z + 9 F + 8 S + 8 E = **47 new proposed gates**. None executed in this documentation PR.

## Stage / evidence crosswalk

| Current evidence | Reuse here | Still unresolved |
| --- | --- | --- |
| #1955 merged local register | Existing SoA borrowed planes, layout and scope refusal, route-once | ZC-1 enforceable membrane, five bus projections, historical-call coverage |
| #1956 merged ALU | k-of-n / Bag / Kleene correctness, measured native advantage | No virtual-bus / remote speedup inference |
| #1949 and #1951 | D-DRAM locator proposal and `U24x5` census | EXO-46/89/151 lease owner, no public Abakus type |
| #1950 / #1633 | selective source reading, known ingress copy | D4 and actual source-owner witness |
| Zero-copy lens law | operator invariant, extant transparent view precedents | no automatic full-repo enforcement or physical I/O guarantee |
| EXO-104/105/109 | precise tuple/rollback/oracle risks | open owner decisions; no automatic mask-risc modifications |

## Reproducibility record required on every future PR

~~~text
GATE: D-DRIVER-ZC/<id>
PR/COMMIT: URL + immutable head SHA
OWNER: owner and outstanding EXO decisions
PROGRAM: ClassView fields, method, bus roles, semiring, NULL mode
SOURCE: canonical source incarnation / source-version / DR frame / tile digests
ORACLE: independent global-ID scalar reference, source-independent
POSITIVE: exact result, permitted scope
DISABLE: exactly one deliberate failure with expected divergence/refusal
COPY: owned canonical payload bytes created; scratch/result/ingress separately
COST: Q, source page/disk bytes, rpc bytes, allocations, p50/p95/p99, cold/warm
SCOPE: local | source-backed | two-process | NOT-RUN
VERDICT: PASS | FAIL | PARTIAL | BLOCKED_OWNER | NOT-COMPARABLE
~~~

**No-go**: a test that proves only equal answers, only zero heap allocations, or only a fast microkernel cannot certify the driver zero-copy law or the "Thinking Hard Drive" theorem. The driver cannot itself certify a caller excluded from the migrated/closed access surface.
