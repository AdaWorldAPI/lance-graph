# dn_redis is a key-shape protocol + Rust command type model — not a Redis service dep

## TL;DR

`lance-graph-cognitive::container_bs::dn_redis` is a **key-shape protocol definition + a Rust-side command type model** for HHTL-keyed hot lookups. It is NOT:

- A network-protocol implementation (no RESP encoder/parser, no listener, no command executor)
- A drop-in replacement that lets existing Redis clients talk to lance-graph
- A wire-protocol emulator (the FalkorDB / KùzuDB precedent does NOT apply here — those projects implement the Redis RESP protocol; dn_redis does not)

What dn_redis IS:

- A **Rust API** that exports key-construction helpers (`dn_key`, `spine_key`, `walk_to_root_keys`, `children_pattern`, `subtree_pattern`) producing `String` keys like `ada:dn:{hex}` and `ada:spine:{hex}`
- A **command type model** (`RedisCommand` enum + `RedisPipeline` struct) that adopters can populate to describe the operations they need to execute
- A **serde layer** for `CogRecord` payloads (`cog_record_to_bytes`, `cog_record_from_bytes`)

## What is missing — and what adopters have to provide

The module exports the SHAPE of the operations; adopters provide the EXECUTOR. There is no `Backend` trait, no listener, no `connect()` function, no `execute()` method on `RedisPipeline` — those are what consumer code must implement.

To use dn_redis, a consumer must:

1. Decide on a backend (see below)
2. Write an executor that takes `RedisPipeline` (or individual `RedisCommand` values) and runs them against the chosen backend
3. Mount the executor at the call sites where the consumer's cognitive substrate needs hot-key lookups

This doc proposes formalizing two valid backend categories so adopters do not have to discover them by reverse-engineering the call sites.

## Two valid backend categories

### Backend A: standalone Redis (or Redis with hash-tag re-keying)

A consumer can run a Redis server and pipe `RedisCommand`s to it via any Redis client (`redis-rs`, `fred`, etc.). The consumer writes the executor: receive `RedisCommand`, translate to the client's API, return results.

**Important caveat (per codex P2 review on PR #455):** the documented `walk_to_root` operation uses `MGET ada:dn:{ancestor1} ada:dn:{ancestor2} ...`. In a **Redis Cluster** deployment, `MGET` requires all keys to belong to the same hash slot — which means they must share a `{hash_tag}` substring per the [Redis Cluster specification](https://redis.io/docs/latest/operate/oss_and_stack/reference/cluster-spec/). The current key layout `ada:dn:{hex}` does NOT include a hash tag, so cluster `MGET` will return `CROSSSLOT Keys in request don't hash to the same slot` errors.

For Redis Cluster to be a valid backend, EITHER:
- (i) The key layout must be re-shaped to include a shared hash tag for keys that are co-queried (e.g. `ada:dn:{root_basin}:0102...` so all descendants of a basin hash to one slot)
- (ii) The consumer must split multi-key operations into per-key calls (defeats the pipeline)
- (iii) The consumer runs standalone Redis (not Cluster)

This caveat is the practical reason most consumers should treat dn_redis's key shape as "designed for standalone Redis OR for an in-binary executor", not for a clustered Redis deployment without re-shaping.

### Backend B: in-binary executor over Lance via DataFusion

A consumer can implement an executor that takes the same `RedisCommand` shape and runs it against the local Lance dataset via DataFusion queries. The result is Redis-protocol-shaped responses (via the Rust types — not wire-protocol bytes) emitted from the consumer's own data, with **no Redis service required**.

What the consumer writes (per codex P2 #3 — there is no shipped trait to implement; this IS new consumer code):

```rust
// Consumer code, NOT lance-graph code
struct LanceBackend { /* ... */ }

impl LanceBackend {
    fn execute(&self, cmd: RedisCommand) -> Result<RedisValue, Error> {
        match cmd {
            RedisCommand::Get(key) => {
                let dn = self.parse_dn_from_key(&key)?;
                let row = self.lance.read_by_dn(dn).await?;
                Ok(RedisValue::Bulk(cog_record_to_bytes(&row)))
            }
            RedisCommand::Mget(keys) => { /* batch lookup over Lance */ }
            RedisCommand::Keys(pattern) => { /* DataFusion scan filtered by prefix */ }
            // ... etc per the enum's variants
        }
    }
}
```

This is "Redis-shape over Lance" — the consumer projects Lance results through the command-result type. There is no protocol parsing because no wire protocol is involved; everything is in-process Rust types.

## What this doc does NOT claim

- **It does not claim FalkorDB or KùzuDB use dn_redis.** Those are independent products that implement the actual Redis RESP wire protocol; dn_redis does not implement RESP. The earlier version of this doc cited them as a precedent for "talk Redis without being Redis" — that framing was wrong. The honest framing is more constrained: dn_redis provides the key-shape and command-type contract that a consumer can EITHER pipe to a real Redis (standalone) OR execute in-binary against their own data.
- **It does not document a network protocol.** There is no listener, parser, or executor shipped by lance-graph for dn_redis. Calling it "wire-protocol emulation" (as a prior version of this doc did) misleads consumers about what is implemented vs what they must implement themselves.
- **It does not claim Redis Cluster is plug-and-play.** Per the caveat above, the current key shape is incompatible with cluster `MGET` slot routing.

## Why this matters

For single-binary deployments (e.g. `AdaWorldAPI/bardioc/substrate-b`) the natural backend is option (B): an in-binary executor over Lance. This means "zero application-level boundaries within substrate-b" (per the PR #452 / #454 append-only-Raft doc) survives the addition of HHTL-keyed hot lookups precisely because dn_redis is a TYPE MODEL, not a service dependency.

For consumers who already operate a standalone Redis (with no clustering or with cluster + re-keyed hash tags), option (A) is straightforward: write the executor that translates `RedisCommand` to a Redis client's API.

The earlier version of this doc framed dn_redis as wire-protocol emulation; codex review on PR #455 caught the inaccuracy. The corrected framing is more constrained but more honest: dn_redis is the SHAPE, the consumer provides the EXECUTION.

## Cross-references

- `lance-graph-cognitive::container_bs::dn_redis` — the key-shape protocol + command type model
- `lab-vs-canonical-surface.md` — the canonical-vs-lab discipline that frames adapters
- Companion docs `APPEND_ONLY_RAFT_DOVETAIL.md` + `CLUSTER_ASYMMETRY.md` (PR #452 / #453 / #454 — merged)
- `AdaWorldAPI/bardioc` PR #15 conversation thread (where this doc + the corrections originated)
- [Redis Cluster specification](https://redis.io/docs/latest/operate/oss_and_stack/reference/cluster-spec/) for the hash-slot constraint
