# The lance 9 cache surface — what moka holds, who can size it, and what that means for RAM-billed deployments

> **READ BY:** `lance-cache-cartographer`, `lance-cache-contents-auditor`,
> `integration-lead`, `truth-architect`, and any session that reasons about
> RAM footprint of a lance-graph consumer, proposes a disk cache backend,
> touches `.claude/plans/idle-flush-dataset-eviction-v1.md`, or debugs a
> `"No object store provider found for scheme"` error.
>
> **Companions:** `.claude/knowledge/s3-hydration-lifecycle.md` (the
> three-layer model this doc's question sits under) ·
> `.claude/plans/idle-flush-dataset-eviction-v1.md` §8a (the measured
> hydration cost) · `zero-copy-lens-law.md` (why a materializing read path is
> a violation, not a convenience).

## The operator's question, stated before any answer

> *"Der Gedanke warum ich S3 sink-in to Harddisk möchte ist, daß lance nur
> die aktiven Bestandteile in den Speicher zieht und die Railway-Rechnung für
> RAM usage kleiner wird."* — and, one turn later: *"man müsste dann halt nur
> radikal moka flushen."*

Two candidate mechanisms for a small RAM bill over a large dataset:

- **A. Disk sink-in:** hydrate S3 → local disk, rely on demand paging, evict
  after idle (the idle-flush plan).
- **B. Capped moka:** read S3 (or disk) directly and bound the in-memory
  cache — `radikal flushen` as configuration.

Which one is real depends on two facts about lance 9 that this doc exists to
pin: **what the cache actually holds** and **who can size it from outside**.

## Evidence status (workspace rule: label everything)

| claim | status | evidence |
|---|---|---|
| `moka` is an UNCONDITIONAL dependency of `lance-core` — no feature gate, present in local-only builds | **FINDING** (source-read 2026-08-07, lance-core 9.0.0) | `Cargo.toml` `[dependencies.moka] version = "0.12"` with no `optional`; `cache/mod.rs:51` `mod moka;` ungated |
| The `aws` feature gates only the object-store PROVIDER, not any cache | **FINDING** (measured) | with every `AWS_*` var set correctly, `connect("s3://…")` without `lancedb/aws` fails `No object store provider found for scheme: 's3'` — the error names the scheme, not a credential. With the feature: first-try success |
| `MokaCacheBackend::with_capacity(bytes)` is byte-weighed, not entry-counted | **FINDING** (source-read) | `moka.rs`: `.max_capacity(capacity)` + `.weigher(\|key, entry\| key_footprint(key) + entry.size_bytes)` |
| `MokaCacheBackend::no_cache()` exists (`Cache::new(0)`) — "radikal flushen" as a constructor | **FINDING** (source-read) | `moka.rs` |
| `CacheBackend` is a pluggable trait; the docs name "persistent backends" as intended; **only moka ships** | **FINDING** (source-read) | `cache/{mod,backend,codec}.rs`; `codec.rs` describes "scanning a persistent store at startup"; the only implementor in-tree is `MokaCacheBackend` |
| lance mmaps nothing — `memmap` absent from lance / lance-io / lance-file / lance-core; local reads are seek+read into heap `Bytes` | **FINDING** (source-read) | zero grep hits across the four manifests and sources; `object_store` `local.rs` seeks and reads |
| Lance's read path gives NO 64-byte alignment guarantee (varies with allocator state) | **FINDING** (measured, this branch) | commit `a27b06a` message + `osm-soa-bake` `slab.rs` — the same read passed alone and failed in-suite |
| `FixedSizeBinaryArray::try_from_iter` copies chunk-by-chunk; `Buffer::from_vec` adopts the allocation | **FINDING** (source-read + pointer-identity test before removal) | arrow-array 58 `fixed_size_binary_array.rs:553` (`MutableBuffer`), arrow-buffer `immutable.rs:141` |
| **What `LanceCache` actually holds on the DATA path** — decoded pages / batches vs only manifests, schemas, index metadata | **OPEN — P-CACHE-1** | nothing read yet; this decides whether a capacity cap bounds the RAM that matters |
| **Whether cache capacity / `no_cache` is reachable from `lancedb::connect` or `DatasetBuilder`** (public API, env, session object) | **OPEN — P-CACHE-2** | `LanceCache::with_capacity` found only in lance-core; plumbing untraced |
| **Empirical RAM shape**: RSS across scans at different capacities, re-fetch behaviour over S3 on a second scan | **OPEN — P-CACHE-3** | needs the probe; timing + `/proc/self/status` VmRSS, honest about network variance |

**Nothing below is promoted past its row above.**

## Why P-CACHE-1 is the load-bearing question

If the data path (decoded column pages, record batches) does NOT go through
`LanceCache`, then capping it bounds only metadata — and the RAM that shows up
on a Railway bill lives in whatever the *caller* holds (the failure mode
already caught once on this branch: a `read_batch` that collected the whole
table and concat-copied it, commit `a27b06a`). In that world, mechanism B is
an illusion and the RAM answer is *streaming discipline in the consumer*, not
cache configuration.

If the data path DOES go through the cache, `with_capacity(n)` is a hard
byte ceiling on exactly the memory the operator is billed for, and mechanism
B beats mechanism A on every axis except S3 request count.

## Decision table this doc must end up supporting

| finding | consequence |
|---|---|
| data cached + capacity reachable | **B wins for RAM**; idle-flush plan remains a *request-cost* optimisation only |
| data cached + capacity NOT reachable | small upstream ask (expose the knob), NOT a fork — lance is upstream-authoritative (`E-LANCE-IS-UPSTREAM-AUTHORITATIVE-1`) |
| data not cached | RAM bill is the consumer's streaming discipline; B is inert for data; re-reads hit S3 every time → A (disk sink-in) regains its case as *request* mitigation |

## Probe queue

| probe | question | pass/fail shape | status |
|---|---|---|---|
| P-CACHE-1 | do decoded data bytes enter `LanceCache`? | enumerate every `CacheKey` implementor + every insert site; classify metadata vs data with file:line | NOT RUN |
| P-CACHE-2 | is capacity / `no_cache` settable via lancedb / DatasetBuilder / env / Session? | the exact public call chain, or a definitive "unreachable" | NOT RUN |
| P-CACHE-3 | does RSS track capacity? does a second S3 scan re-fetch? | probe binary, RSS + wall time, both capacities; network variance stated | NOT RUN |

## Traps already paid for (do not re-pay)

1. **The scheme error is not a credential error.** Provider absent ⇒ fail at
   registration; every credential correct and irrelevant.
2. **"Usually aligned" passes CI.** Never assert an alignment VALUE; assert
   the invariant (`rows().is_some() == (ptr % 64 == 0)`).
3. **A pointer-identity test proves nothing if its reference is the copy.**
   The removed module's test compared the slab against the *concatenated*
   buffer — the copy it should have ruled out was one call earlier.
4. **`try_from_iter` is a copy wearing a constructor's name.**
