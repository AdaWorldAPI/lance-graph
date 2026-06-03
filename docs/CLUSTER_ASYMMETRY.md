# Cluster ≠ cluster: how lance-graph clusters differ from Cassandra-era clusters

## TL;DR

There are two distinct reasons to cluster a distributed system:

1. **Capacity-forced clustering**: the data does not fit on one node.
   You shard. Cross-node queries are required. (Cassandra, JanusGraph,
   classic Elasticsearch deployments.)

2. **Availability-chosen clustering**: the data fits on one node.
   You replicate it across N nodes for HA + geo + load distribution.
   Each node holds the full dataset. Cross-node queries are NOT
   required for the hot path. (CockroachDB, etcd, peer-Raft Lance.)

Lance-graph consumers are virtually never capacity-forced (the
encoding cascade + columnar layout + radix-trie deduplication
collapse storage by 1-3 orders of magnitude vs LSM-tree wide-column
stores). Clustering is always availability-chosen.

Don't import the Cassandra cluster operations playbook into a
lance-graph deployment. The failure modes, the operational rhythms,
the budgeting assumptions are all qualitatively different.

> ### Scope: external architecture pattern, not a built-in lance-graph feature
>
> Per codex P1 review on PR #453: the deployment shapes described
> below — particularly "peer-Raft + Lance-local-per-node" — describe
> an EXTERNAL ARCHITECTURE PATTERN that adopters can build on top of
> lance-graph, NOT a built-in lance-graph capability. Lance-graph
> itself provides the columnar storage, the DataFusion query path,
> the encoding crates, and the Rust API surface. The Raft layer, the
> substrate binary, and the consensus-replication path are
> downstream consumer code. Adopters implementing this pattern reach
> for `openraft` (or `surreal-cluster` if their stack is
> surrealdb-shaped) to provide the Raft layer; they would NOT
> inherit one from lance-graph.
>
> The doc documents WHY this pattern works well WHEN built on
> lance-graph (the append-only storage model, cheap anti-entropy,
> per-node fit), not a feature lance-graph itself ships. Adopters
> who only consume lance-graph's columnar + DataFusion path should
> NOT assume their data is automatically replicated.


## The two cluster shapes, side by side

| Concern | Capacity-forced (Cassandra+JG) | Availability-chosen (peer-Raft Lance) |
|---|---|---|
| Reason to cluster | Data does not fit on one node | Data fits on one node; replicate for HA + geo |
| Each node holds | 1/N of the data (consistent-hash shards) | 100% of the data |
| Hot-path reads | Coordinator pattern; fan-out to shard owners | Local; no cross-node hop |
| Replication factor + storage amplification | R=3-5 per replica; total cluster storage = N_shards × R (sharding COMPOUNDS replication amplification) | R=3 per replica; total cluster storage = R × dataset_size (same per-replica amplification; no sharding compound) |
| Effective node count | 3N to 5N where N is shards needed for capacity | 3 (or more if geo distribution is required) |
| Compaction shape | LSM SSTable: tombstone-reclaim + run-merge; coordinates with replication; lag spikes during | Lance file (`DatasetOptimizer.compact_files`): merge small fragments for layout; independent of consensus; no replication-lag interaction |
| Cross-shard transactions | Hard; needs 2PC or careful avoidance | Not applicable (no shards) |
| Anti-entropy | Merkle-tree comparison; expensive | Manifest hash compare; O(1) decision |
| Read latency | Depends on coordinator + slowest shard | Local-replica latency; predictable |
| Operator burden | High (capacity planning, compaction scheduling, rebalancing) | Low (HA discipline only; no shard management) |


**Storage amplification honesty (per codex P2 review on PR #453):**
Both shapes have replication storage amplification — three replicas
of a 5GB dataset is 15GB of total cluster disk regardless of
distribution shape. The architectural advantage of availability-chosen
clustering is NOT that amplification disappears (it doesn't); it's
that amplification does NOT compound with shard count. Cassandra at
RF=3 with 10 capacity shards consumes ~30× the single-replica
storage; peer-Raft at R=3 consumes exactly 3×. Same per-replica
amplification, no shard multiplier on top.

## Why lance-graph consumers fit on one node

The encoding stack typical of lance-graph consumers compresses data
by 1-3 orders of magnitude vs LSM-tree wide-column representations:

- **highheelbgz SpiralAddress**: 3 integers (12 bytes or 6 bytes
  for u16 variants) representing a φ-spiral walking address — NOT
  a copy of the weight vector, an address into a deterministic
  spiral. Per-row footprint approaches the information-theoretic
  minimum for the addressing dimension.

- **bgz-hhtl-d Slot D / Slot V**: 4 bytes total per row — 2-byte
  Slot D (HEEL basin 2 bits / HIP family 4 bits / TWIG centroid 8
  bits / polarity 1 bit / reserved 1 bit) + 2-byte Slot V (BF16
  residual magnitude). The hierarchical addressing is in the bit
  layout; the residual captures the per-row delta from the centroid.

- **bgz17 palette256**: 256-archetype compose table for multi-hop
  semantic relations. 8 bits per archetype reference; multi-hop
  composition in O(1) via the compose table.

- **CAM-PQ leaf vectors**: 6 bytes per row for the leaf-exact-match
  vector projection. HHTL-banked variant (16 family subcodebooks)
  reduces ANN scan space by ~16× under empirical intra-family
  locality (98.6% per the `lance-graph` PR #444 probe).

- **`lance-graph-contract::hhtl::NiblePath` (shipped) + Lance
  `versions()` (shipped)**: HHTL identity is a 16ⁿ nibble path packed
  into a `u64` (`FAN_OUT = 16`, `MAX_DEPTH = 16`). Adopters who want
  to dedupe shared HHTL prefixes in memory typically derive an
  adaptive radix-trie index over `NiblePath` addresses — heel + hip
  nibbles common across many entities can be stored once per path
  segment via consumer-side structures (O(k) lookup, prefix-sharing).
  **The dedup-by-prefix data structure itself is consumer code, not
  a built-in lance-graph crate.** Lance's own `versions()` log is the
  time-axis (cross-session index of which identity positions changed
  when). An earlier version of this doc cited `vort/vart` as if it
  were a shipped crate; corrected per peer review — the radix-shaped
  trie at the cognitive layer is a proposed pattern (no shipped crate
  name) and the identity primitive + the time-axis are the two
  shipped surfaces this bullet should have cited from the start.

Concrete example: Wikidata (~115M entities). In Cassandra+JG, the
indexed graph form is multi-TB with replication factor 3 → multi-TB
× 3 across the cluster. In lance-graph with the above encoding stack,
the same corpus compresses to low single-digit GB total (including
indexes and the Lance version log). Fits on a modest single node.
Each peer replica holds the FULL dataset.

This is not an academic claim; it's the deployment shape proven in
the `AdaWorldAPI/bardioc` B1 substrate-b reference implementation.

## Knock-on consequences of availability-chosen clustering

### 1. Three-node deployment is the starting recommendation, not the toy

For Cassandra-era thinkers, three-node clusters are toys. Production
Cassandra deployments often run 12-24 nodes. The reasoning: each node
holds 1/N of the data; you need many N to fit the corpus; replication
factor 3 multiplies again.

For peer-Raft Lance: three-node clusters are production. Each node
holds the full dataset. Three replicas give you majority quorum (one
failure tolerated). More replicas are added for geographic distribution
or read-load fanout, not for capacity.

Default starting deployment: 3 substrate-b instances, one per AZ in
the same region, peer-Raft replicated.

### 2. No coordinator pattern; no fan-out lag

Cassandra reads route through a coordinator node which fans out to
the shard owners and aggregates the response. The query latency is
bounded below by the slowest shard's response time. Hot shards drag
the cluster.

Peer-Raft Lance reads are LOCAL. The client connects to any node;
the node has the full dataset; it returns the answer without contacting
peers (for eventually-consistent reads) or with a single Raft read-index
round (for linearizable reads). No fan-out, no aggregation, no
slowest-shard lag.

### 3. Compaction is qualitatively different (lighter coordination)

Cassandra-style LSM compaction rewrites SSTables to reclaim tombstones
and merge sorted runs. It is CPU + IO heavy and creates replication
lag spikes; if too many nodes compact simultaneously, the cluster's
effective replication factor temporarily drops. Operators schedule
compactions across nodes to avoid that.

Lance has compaction too, but of a different shape:
`DatasetOptimizer.compact_files` merges small fragments into larger
ones for query layout optimization (many small appends produce many
small fragments which slow scans; periodic file compaction restores
good layout). It is NOT a tombstone-reclaim cycle — Lance is
append-only at the version level, so there are no tombstones to
reclaim in the LSM sense.

The qualitative difference:

- **LSM compaction** is a CORRECTNESS + SPACE concern (tombstones must
  be reclaimed to bound storage; runs must merge for read performance);
  coordination with replication is unavoidable.
- **Lance file compaction** is a LAYOUT OPTIMIZATION concern (queries
  get faster when fragments are larger; correctness is unaffected);
  it can be scheduled independently per node, produces new fragments
  that are themselves append-only, and does NOT block or interact
  with consensus replication.

So Lance compaction exists and operators should plan for it (Lance's
table-maintenance docs describe `DatasetOptimizer.compact_files`).
The operational burden is lower than Cassandra LSM compaction because
the coordination requirements are weaker — but it is not zero. (Per
codex P2 review on PR #453.)

### 4. Anti-entropy is cheap

Cassandra anti-entropy (catching up a lagging replica) compares SSTable
Merkle trees node-by-node and streams the diffs. This is expensive and
creates load on both sides.

Lance peer-Raft anti-entropy: compare the manifest hash between nodes.
If equal, sync. If not, ship missing fragments + the new manifest. The
IDENTIFICATION step is O(1). The streaming step is bounded by the actual
divergence, not by the dataset size.

### 5. Rebalancing is not a thing

Cassandra rebalancing (when adding or removing nodes) requires data
movement. Token ranges shift; data streams from old owners to new
owners; the cluster operates in a degraded state during the rebalance.

Peer-Raft Lance: adding a node = bring up a new substrate-b instance,
let it catch up via Raft, mark it a voter. No data movement needed
beyond the catch-up (and the catch-up is the same wire pattern as
the per-write replication — no special "rebalance" mode). Removing
a node: stop the substrate-b, mark it non-voter, retire it. No data
movement.

## Knock-on consequences for the Raft consensus tax

Raft consensus IS on the per-request budget for writes and linearizable
reads. This is irreducible for the distributed-OLTP property. (See
the companion doc `append-only-raft-dovetail.md` for why this lands
lighter in Lance than in LSM-based systems.)

In an availability-chosen cluster, the consensus tax lands EVEN lighter
than in a capacity-forced cluster:

- **Smaller per-node datasets**: Raft logs are smaller; commits propagate
  faster
- **Fewer replicas needed**: 3 substrate-b instances vs 12-24+
  Cassandra+JG nodes → less coordination overhead per write
- **No cross-node fan-out**: linearizable read-index can be served by
  the local leader of the relevant Raft group; doesn't require remote
  round-trip when the local instance IS the leader

## When you actually DO need capacity-forced sharding

Rare for lance-graph consumers, but possible:

- Corpora dramatically larger than Wikidata (multi-billion entities or
  multi-PB raw data)
- Workloads with hot-key distributions that exceed a single node's
  IO bandwidth
- Specific compliance requirements that mandate physical data isolation
  per tenant

In those cases, the appropriate response is application-level sharding
or tenant-level partitioning — NOT a Cassandra-style consistent-hash
ring. Each shard or tenant gets its own peer-Raft + Lance cluster.
The capacity dimension is solved by horizontal application-layer
partitioning; each underlying cluster remains availability-chosen.

## What this doc does NOT claim

- **Single-node deployments are sufficient for production.** They're
  not. HA requires multiple replicas (or accepted downtime during
  failures). Single-node is a development / staging shape.

- **Three-node is universally optimal.** Geographic distribution
  may require more (one replica per region). Specific availability
  targets may justify more replicas.

- **All Lance + Raft stacks ship the same compression.** The specific
  encoding cascade described above is from the `bardioc` B1 reference
  consumer; other consumers will have different per-row footprints.
  The qualitative property (orders-of-magnitude vs LSM wide-column)
  remains, but specific numbers vary.

- **Cassandra+JG choose their shape wrongly.** The Cassandra design
  is correct FOR THE STORAGE MODEL IT HAS. The architectural choice
  that produces a different deployment shape is the choice to use
  Lance + Raft, with append-only storage and columnar compression.
  The doc names the consequence, not the wrongness of the alternative.

## Recommended deployment pattern (reference)

> **Reminder (per codex P1):** This pattern is the bardioc B1
> substrate-b reference architecture, NOT a built-in lance-graph
> feature. Adopters provide the Raft layer themselves (openraft /
> surreal-cluster / external TiKV). Lance-graph contributes the
> columnar storage + DataFusion + encoding crates that MAKE this
> pattern cheap, not the pattern itself.

See [reference consumer implementation in bardioc B1 substrate-b]
(separate proposed doc) for a worked example. Briefly:

- Three substrate-b instances, one per availability zone within a region
- Each substrate-b is one Rust binary AND one Raft node (Raft impl
  from openraft or surreal-cluster — NOT inherited from lance-graph)
- Lance dataset local to each instance (full dataset, not a shard)
- Reads serve from local Lance with no cross-node coordination
  (eventually-consistent) or from a Raft read-index round (linearizable)
- Writes serve via Raft quorum to the local leader; replicated as Lance
  fragment appends
- Add a fourth+ instance only for geographic distribution or read-load
  fanout, not for capacity

This is the shape proven against Wikidata-scale workloads in the
bardioc reference consumer. Operational complexity is significantly
lower than the Cassandra-era equivalent at the same availability target.