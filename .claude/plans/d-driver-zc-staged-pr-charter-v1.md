# D-DRIVER-ZC: staged PR charter for the virtual access membrane

> **2026-11-04. STATUS: PROPOSED, PR0 DOCS ONLY.** This is a plan for independently reviewed, *future* executable PRs. Nothing here authorizes a new production driver/ISA, a global index, or a network listener. [Master contract](d-driver-zc-virtual-bus-contract-v1.md); [falsifier matrix](d-driver-zc-falsifier-matrix-v1.md); [EXO-154](https://linear.app/exo-red/issue/EXO-154); [EXO-153](https://linear.app/exo-red/issue/EXO-153); EXO-50 is communication, not admission.
>
> **Relationship to D-DRISA:** Stages D-DRISA-0 (#1953), D-DRISA-1 (#1955) and D-DRISA-3 (#1956) are **merged**. This plan does **not** duplicate local register or Abakus ALU tests. Stages D-DRISA-2 / 4 remain separate async/remote owner projects; our stages ZC-4/5 are *integration gates*, not competing drivers.

## Staging and dependency graph

~~~mermaid
flowchart TD
    Z0["ZC-0: contract, ownership, proof matrix"] --> Z1["ZC-1: local sealed borrow membrane"]
    Z1 --> Z2["ZC-2: five virtual bus lenses and fusion"]
    Z1 --> Z3["ZC-3: address reuse and foreign bind"]
    Z2 --> Z4["ZC-4: async descriptor integration only"]
    Z3 --> Z4
    Z4 --> Z5["ZC-5: storage-backed/Lance attestation"]
    Z2 --> Z6["ZC-6: migrated caller firewall and CI"]
    Z3 --> Z6
    Z5 --> Z7["ZC-7: Thinking Hard Drive proof verdict"]
    Z6 --> Z7
    G1{"EXO-46/89/151/140"} -.-> Z3
    G2{"D-DRISA-2/4 & security"} -.-> Z4
    G3{"D4 + S-INGRESS + EXO-114"} -.-> Z5
    G4{"class/semantic owners"} -.-> Z6
~~~

Stages labelled GATED only proceed as decision memos until owner authorization. Do not open stacked implementation PRs before reviewing their prerequisites. Docs PR ZC-0 can be merged only after CI, reviewers and governance agree; **no auto-merge**.

## ZC-0: docs, status board, EXO map (this PR)

**Ownership:** operator / architecture review. **Files:** master, this staged charter, independent falsifier matrix, board entry, board status/integration/index and Linear EXO-154 / EXO-50 handoff.

**Acceptance:**
- The existing zero-copy law remains in force, including no implicit copy below a size threshold.
- Dynamic scope and virtual bus semantics separated from physical layout and from user-level object egress.
- All identifiers labelled logical/hypothetical; no public Rust type, opcode, layout, wire protocol or F### presumed.
- Precise distinction among local source borrowing, disk/page ingest, and remote data-local computation.
- Existing D-DRISA-1/3 evidence attributed correctly, existing owner gates preserved; no investigation of the legacy DTO/shader-consolidation stack.
- Generated board entry index diff preserves all existing curated findings and passes repository check.

**Exit:** reviewed architecture, no runtime.

## ZC-1: smallest local no-copy borrow membrane

**Status:** FUTURE LAB, no public API or source rewrite. **Owner review:** DR/DT64, ClassView, mask-risc and source layout owner.

**Goal:** Consume #1955 `RegisterFile` on a borrowed source-owned SoA; define a *scoped internal access membrane* which can project a typed pair of fields at one witnessed DR ordinal. Do not add a new hash table, helper row cache, generic `Vec<Row>`, duplicated fields or new mask-risc executor. Prefer extending the existing excluded `drisa-register-probe` fixture to avoid a new crate.

**Positive tests:** one tile DR64, one DR256, 256 tiles (65,536 identities), demanded 2 of 64 fields, same physical buffer pointers as source, borrowed slices/strides, 0 new owned canonical bytes, mask-only COUNT/EXISTS without source value reads; compare direct reference oracle by global ID.

**Anti-evasion tests:** intentionally call a view materializer (must be unavailable under the admitted interface or rejected by a lint/compile-fail fixture), request unknown/unauthorized field, change source incarnation with numeric revision reset, permute the source layout, escape a borrowed slice beyond the lease lifetime (compile-fail). A separate in-process bad/legacy arm may materialize solely as an independent reference oracle, never an admitted execution route.

**Measured:** cold bind cost, warm fold, route calls (0 after reuse), allocations and payload copy counters (heap and stack where instrumentation can detect), source bytes read, source pointer alias, scratch/result bytes reported separately. Do **not** claim universal no-copy because one compiled scope passed.

**Go:** precise guarantee limited to migrated local consumer path. **Stop:** if raw source exposes uncontrolled owned-copy access to that path; report coverage gap, do not bypass with unguarded `unsafe`, `transmute` or a "fast Vec" path.

## ZC-2: five virtual buses and composition without materialization

**Status:** FUTURE LAB; depends ZC-1. **Owner review:** Quack/RelAlg and DT64 method semantics.

**Goal:** Prove `XX, XY, XZ, YY, ZZ` each project and fold on the SAME canonical X/Y/Z SoA rows without building five physical arrays or rows. `YZ` may be included as symmetric negative/positive control. Do not assign any fixed semantic meaning (product, covariance, adjacency) to the bus labels without a typed operator declared by the semantic owner.

**Fixture:** three deliberately non-interchangeable field lanes with differing values (include zeros/negative/missing), 64/256/multi-tile, source-layout permutation, 2–3 typed methods, input population mask. Compare:
- direct per-row scalar oracle by global ID and declared field operators;
- admitted one-lens folds;
- two-step `XY → XZ`, three-step `XY → XZ → ZZ` compiled into existing mask-risc scratch / native algebra and returned as terminal;
- materializing reference implementation **outside timed/admitted path** to catch semantic mismatch, counted as B1.

**Representation proof:** no new owned `Vec<(X,Y)>`, `Vec<PSCustomObject>` or 2D bus table; X remains one source buffer and one coordinate binding across XY and XZ. Explicitly test XX/YY/ZZ roles do not double-load a physically identical field just to fill two logical slots.

**Negative controls:** force the same ordinal from two different frames; delete an edge connecting a foreign Y; accidentally swap X and Y; vary NULL/Bag semantics; cross-call scratch epoch; hidden full-field eager gather; violate tail bits.

**Go:** five bus views and multi-step composition pass independent oracles with a stable source owner and zero intermediate source-value representation. **Stop:** if composition forces an owned value table or drops frame/source/scope; refuse as `NOT_FUSABLE` rather than copying.

## ZC-3: one address binding, local tiles and foreign/remote source-owner hop

**Status:** read-only decision memo **BLOCKED EXO-46/89/151/140** for production foreign navigation. Local co-addressed tile proof may run with no new foreign authority.

**Question:** Can an already-resolved address be reused across multiple virtual buses and methods while all requests retain one source/layout/permission witness? Compare:
- co-addressed input already in tile/local form;
- bind once on first call, use Q=1/2/8/16/64/256;
- per-query global→local routing (#1955 failed 256-tile gate);
- FOREIGN case when legally admitted, resolved on source owner and reused under scope with no new global index.

**Exact cost:** cold routing, per-call source lookup, tiles visited, bytes, allocations, total p50/p95 and negative misprediction/eviction; Q=1 must not be hidden. Include mismatched foreign source, expired foreign source lease, source compaction, wrong field schema and cross-tenant refusal.

**Proposed owner preference:** co-addressed masks plus existing `MapTable` / `Frame` first. No independent hash map or trie by default; a versioned index is a **separate EXO-46 operator decision** only if the first alternative cannot meet a measured requirement.

**Stop:** neither name coincidence nor matching local ordinal authorizes cross-source relationship truth. A remote pointer never crosses the process boundary.

## ZC-4: async source-owner driver bridge, NOT a second async engine

**Status:** **GATED D-DRISA-2 / 4, security/transport owner**. This plan provides *consumer compatibility tests*, not a competing driver. No examination or reorganization of the legacy Ladybug/P64/StreamDTO/BusDTO/PerturbationDTO/cognitive-shader-driver stack in this stage.

**Positive:** take the same witnessed `bind/project/execute` semantics from ZC-1/2; send a small typed method, source/frame/layout/scope reference into an existing, authorized **data-owner** loopback runner; source owner validates and borrows its own values, returns COUNT/EXISTS/mask scalar. Asynchronous queue, cancellation and invalidation tests remain D-DRISA-2's responsibility; here only require that a submitted virtual-bus lens does not acquire a new payload owner.

**Negative:** forged caller-supplied layout digest (FNV insufficient), stale source while queued, scope revoke, early release, foreign owner mismatch, oversized result, secret value bytes on the request wire. In-process "no wire payload" is not a remote performance result.

**Go:** driver lens semantics survive the async descriptor; do not assert interprocess/no-payload transfer until D-DRISA-4 runs an actual transport counter. **Stop:** second execution engine, public per-op endpoint or remote raw pointer.

## ZC-5: persistent source / Thinking Hard Drive physical proof

**Status:** **BLOCKED D4/S-INGRESS, EXO-114, source/lifetime owner**.

**Goal:** Test a source-backed tile whose canonical value planes reside in disk-backed storage and can be selectively read without reconstructing per-row objects. The CPU still computes; source I/O still occurs. Attest DR frame → correct Lance _rowid under ONE existing physical authority only, and count copied row-mask ingress. Reuse #1633 / #1950 source read arms.

**Fixture:** 1M/10M rows, 1 or 16 demanded columns, 0.1/1/10/50/95% selection, clustered/scattered, page-cache warm/cold and disk-cold; fragment split/compaction, stable row-ID toggle, value-only update, remote request. Project XY/XZ/ZZ and compare independent stored-row oracle. Never equate range_count×4096 with pages touched.

**Go:** the admitted resident borrowed portion adds **0 secondary canonical payload bytes**, while physical read and ingress costs are accurately reported. **Stop:** missing D4, reading wrong identity, row ID = row address without witness, or claiming disk reads are zero-copy computation.

## ZC-6: access firewall and incremental legacy migration

**Status:** FUTURE POLICY LAB + operator ownership. This is the **17,000-call** risk reduction program, not a claim that 17,000 call sites have been examined or are all reducible.

**Phase 1:** static *reachable access-surface* census from chosen admitted consumer roots. Classify canonical borrowed reads, explicit result egress, source-owner decode, old raw direct access, tuple materialization, reference-only oracle, and unknown dynamic/plugin paths. Do **not** scan whole repo merely to create a giant list: use Kythe/Codegraph if already ready, else a narrow static AST/call graph with completeness caveats.

**Phase 2:** seal canonical source fields and expose only the validated capability in a selected migrated module; add Rust UI compile-fail tests for value-owner escape, raw reference past lifetime, direct constructor/copy bypass and unauthorized field projection. Use lints/checks for forbidden `Vec<Row>`, `to_vec`, `collect`, `clone` only *inside admitted hot-path modules* and manually classify false positives (cloning an instruction descriptor or a result scalar isn't source duplication).

**Phase 3:** prohibit calling an oracle implementation as a normal hot execution path (EXO-105/109 unresolved); preserve independent scalar or tuple oracle in test/reference scope. Do not edit frozen mask-risc without owner authorization.

**Phase 4:** expand migrated coverage incrementally. Report explicit numerator/denominator and roots; only then claim e.g. "all 37 reachable read paths behind this facade" instead of "all 17,000 calls." Negative fixture deliberately inserts a copied field buffer into a migrated path, and CI must fail.

**Go:** local compile-time/CI coverage is credible within declared roots, with an independent runtime copy/materialization probe. **Stop:** no completeness analysis, caller can bypass private boundary, or a linter flags all `clone` as wrongdoing.

## ZC-7: proposed "Thinking Hard Drive" theorem / engineering verdict

**Status:** FUTURE INDEPENDENT REVIEW, no performance or mathematical theorem certified now. **Dependencies:** enough ZC-1/2/5/6 data; EXO-154, EXO-153, semantic owners.

**Claim categories:**
1. **Semantics:** same witnesses, exact Boolean/Bag/Kleene/absence parity and typed refusals.
2. **Representation:** same canonical source address and zero newly owned canonical *value payload* per admitted resident fold, no virtual bus replicas.
3. **Economics:** whole-query throughput, cold/hot, disk/page bytes, ingress, network bytes and output size; a requested N-row output costs Ω(N).
4. **Enforcement:** migrated reachable call graph cannot bypass the tested facade within the stated threat model.
5. **Scope of theorem:** bounded operator set and layout conditions, not proof of O(1) all-pairs computation or exemption from physical I/O.

Report `PASS_LOCAL_ONLY`, `PASS_SOURCE_BACKED`, `FAIL_SEMANTIC`, `FAIL_COPY`, `BLOCKED_OWNER` or `NOT_COMPARABLE`. New production adoption or a consolidated driver identity is a **separate operator ruling**.

## PR discipline

- Open one PR per stage; ZC-0 is docs-only. Use current main, no accidental stacking or auto-merge. Keep independent falsifier and deliberate-disable logs, code refs, SHA, and board status. Research data can live on a pinned branch with source links.
- No new ClassId, F###, mask-risc opcode, ndarray backend, hash table, global row-id map, source value clone or direct production driver module in the initial probes.
- Keep D-DRISA staging and ownership intact: do not start a second async engine. If D-DRISA-2/4 later ships, ZC-4 may be a *compatibility* probe only.
- EXO-50 messages are append-only coordination and cannot admit architecture changes.
- Keep migrated cognitive-shader-driver/P64/StreamDTO/BusDTO/PerturbationDTO investigation **explicitly deferred**, per operator request.
