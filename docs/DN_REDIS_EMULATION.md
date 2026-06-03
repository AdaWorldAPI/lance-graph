# dn_redis is Redis-protocol EMULATION, not a Redis dependency

## TL;DR

`lance-graph-cognitive::container_bs::dn_redis` is a **protocol/shape adapter** for HHTL-keyed hot lookups. It is NOT a dependency on an external Redis server. Adopters reading the module and the lab-vs-canonical-surface discipline doc correctly identify that `dn_redis` is "protocol-only"; they sometimes incorrectly conclude that an external Redis server is therefore required.

This doc clarifies that **"protocol-only adapter" means the SHAPE is the contract — the backend is open**. Three valid backends:

- (a) an external Redis cluster (traditional hybrid deployment)
- (b) **in-binary**: an OGIT class view + DataFusion query over the local Lance dataset, projected through the `dn_redis` adapter shape — Redis-protocol responses emitted from your own data, **no Redis service required**
- (c) anything else that produces the `ada:dn:{hex}` / `ada:spine:{hex}` key shape with the documented operations

## Audience

Anyone reading `lance-graph-cognitive::container_bs::dn_redis` (the 253-line protocol-only adapter) and wondering whether they need to deploy a Redis server. Also anyone reading SPRINT plan docs that reference `dn_redis` as "protocol-only" without further qualification.

## The pattern (FalkorDB / KuzuDB precedent)

[FalkorDB](https://www.falkordb.com/) (formerly RedisGraph) and [KùzuDB](https://kuzudb.com/) both "talk Redis" — they expose the Redis wire protocol — while the storage and execution underneath are completely different. They are NOT Redis; they emulate the wire protocol so existing Redis clients work.

`dn_redis` follows the same pattern. It defines:

- Key conventions: `ada:dn:{hex}` for individual nodes, `ada:spine:{hex}` for spine entries
- Walk-to-root operation shape (`MGET ada:dn:{ancestor1} ada:dn:{ancestor2} ...`)
- Children/subtree patterns (`KEYS ada:dn:{prefix}??` for one-level; `KEYS ada:dn:{prefix}*` for subtree)
- Pipeline operations (`RedisPipeline` + `RedisCommand` enum)

These are **shapes**, not service dependencies. The shape can be served by any backend that produces the right responses.

## Why this matters

Consumers reading the `dn_redis` module + the surrounding lance-graph discipline docs (lab-vs-canonical-surface, the "protocol-only adapter" annotation that appears across the brutal-fix planning material in downstream consumers like `AdaWorldAPI/bardioc`) correctly identify that it is protocol-only. They incorrectly conclude that an external Redis server is therefore required. This doc clarifies that "protocol-only adapter" means the SHAPE is the contract — the backend is open.

For a single-binary deployment (e.g. `bardioc/substrate-b`) the natural backend is option (b): the in-binary OGIT class view + DataFusion query over Lance. This produces Redis-protocol-shaped responses without running a Redis server. The single-binary "zero application-level boundaries" deployment shape (see `docs/APPEND_ONLY_RAFT_DOVETAIL.md` for the broader argument) survives the addition of HHTL-keyed hot lookups precisely because `dn_redis` is emulation, not a service.

For consumers that already operate a Redis cluster, option (a) is fine. For consumers that prefer purpose-built graph engines exposing Redis wire protocol, FalkorDB and KùzuDB demonstrate that the pattern is well-established.

## Recommended consumer pattern

For the in-binary backend (option b):

1. Implement a `dn_redis::Backend` impl (or whatever the trait name resolves to in the adapter's surface) that intercepts the documented operations
2. Project them through DataFusion queries over the local Lance dataset
3. Wrap the responses in the Redis-protocol shape the adapter expects
4. Mount the backend at the same call sites that a Redis client would be mounted

See the bardioc B1 substrate-b reference for a worked example once that consumer's reference-implementation doc lands (separate proposal under upstream-contributions T2.1).

## What this doc does NOT claim

- **It does not claim FalkorDB or KùzuDB use `dn_redis`.** Those are independent products demonstrating the wire-protocol-emulation pattern; `dn_redis` borrows the pattern but is its own thing.
- **It does not claim option (a) is wrong.** External Redis is a perfectly valid backend for `dn_redis`. The point is that adopters are not LOCKED IN to it.
- **It does not specify the in-binary backend implementation.** That's consumer code; this doc explains why such consumer code is well-formed, not how to write it.

## Cross-references

- `lance-graph-cognitive::container_bs::dn_redis` — the protocol adapter
- `lab-vs-canonical-surface.md` — the canonical-vs-lab discipline that frames adapters
- FalkorDB project (Redis wire protocol emulation precedent)
- KùzuDB project (similar pattern)
- `AdaWorldAPI/bardioc` PR #15 conversation thread — where this clarification surfaced
