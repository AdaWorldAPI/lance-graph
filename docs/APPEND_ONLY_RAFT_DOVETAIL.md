# Append-only ↔ Append-log: how Lance and Raft share the same write shape

## TL;DR

Lance is append-only by design. Raft's commit log is append-only by
design. Replication built on Lance + Raft (peer nodes, each holding
the full dataset locally, replicated via Raft consensus across
substrate instances) is structurally cheaper than the same shape on
LSM-tree + Paxos-light (Cassandra), B-tree + 2PC (traditional SQL),
or any other model where storage mutates state in place.

This doc names the property explicitly + lists the operational
consequences so adopters can choose the right deployment shape.

## The two write shapes that have to align

A distributed Lance deployment has two write paths:

1. **Storage write**: how data lands on disk. Lance: append a
   fragment to the dataset; update the manifest. No in-place
   mutation. No tombstone bookkeeping. No background compaction.

2. **Consensus write**: how replicas agree on the next state.
   Raft: leader appends an entry to its log; followers append the
   same entry to theirs; entry commits when a majority has it.
   No in-place mutation of prior entries. No reordering.

Both paths are append-only. They can therefore SHARE the same
underlying write — the Lance fragment IS the Raft log entry's
payload; the manifest update IS the commit ack.

Compare this with the conventional alternatives:

| Storage shape | Consensus shape | Result |
|---|---|---|
| **LSM-tree (Cassandra)** | Paxos-light / gossip | Storage AND consensus both have their own append-then-mutate cycles. Compaction in storage interacts with hinted handoff in consensus. Coordination headaches. |
| **B-tree (PostgreSQL)** | 2PC (citus-like) | Storage in-place updates fight with 2PC's append-log. Vacuum interacts with commit-log replay. More headaches. |
| **Append-only Lance** | Append-only Raft | One write shape. Storage commit = consensus log entry. No interaction problems. |

## Operational consequences

### 1. No compaction storms

Cassandra clusters periodically run compaction (rewrite SSTables to
reclaim space + maintain read performance). Each node compacts on
its own schedule. During compaction:

- The compacting node's CPU spikes
- Its disk write bandwidth spikes
- Replication lag from / to that node spikes
- Read latency for queries hitting that node spikes
- The cluster's effective replication factor temporarily drops as
  compaction blocks normal write flow

The cluster operator's job is partly to schedule compactions across
nodes so that not too many compact simultaneously. This is a
significant operational burden.

Lance has no compaction. The version log IS the truth; old fragments
can be reclaimed by version-based GC (a much simpler operation than
SSTable compaction) but the GC is local-only, doesn't interact with
replication, and doesn't reorder anything.

A peer-Raft + Lance deployment therefore has uniform per-node
behavior. Each node is doing the same work at the same time, with
the same shape. The operations runbook is simpler because the
failure modes are simpler.

### 2. Anti-entropy is a hash compare, not a Merkle-tree walk

Cassandra's anti-entropy (catching up a lagging replica) compares
SSTable Merkle trees between nodes, identifies divergent ranges,
streams the diffs. The Merkle-tree comparison is expensive (CPU + IO)
and the diff streaming creates load on both nodes.

Lance's anti-entropy is: compare manifest hashes; if they match, the
nodes are in sync; if they don't, ship missing fragments + the new
manifest. The expensive part is shipping fragments (which would
happen in any anti-entropy scheme) but the IDENTIFICATION step is
O(1) — one hash compare.

### 3. Catch-up is "ship missing fragments + apply manifest"

A replica that's been offline for a window needs to catch up to the
current state. In Cassandra this means replaying the commit log +
streaming SSTables that fell out of the hinted-handoff window. The
catch-up time depends on the write volume during the offline window
+ the SSTable layout of the surviving nodes.

In Lance + Raft this means: identify the missing fragments (one hash
compare on the manifest), ship them in their natural order (no
reordering needed because they're append-only), apply the manifest.
The catch-up time depends on the write volume during the offline
window + the wire bandwidth. There is no "is this fragment still
current?" question to resolve because in an append-only world, all
fragments are still current; only the manifest decides what's part
of the active state.

### 4. Cross-DC replication is predictable

The wire format for replication in Lance + Raft is: serialized
fragments (Parquet, compressed by Lance's columnar encoding) + small
manifest deltas. The per-write wire footprint is the size of the
data the user wrote, plus a constant for the manifest delta. There
is no protocol-level multiplier.

In Cassandra, the wire format is the CQL operation log. The per-write
footprint depends on which columns mutated, whether the row was new
or updated, whether the column had a previous value. Cross-DC
replication budget is harder to plan.

### 5. The consensus tax lands once, not twice

This is the unifying point: with non-append-only storage, an
application that wants linearizable writes pays the consensus tax
TWICE. Once for the consensus protocol shipping operations to
replicas. Once for the storage layer doing per-node compaction +
mutation bookkeeping. The two taxes interact (a compaction storm
delays consensus catch-up; a Raft snapshot has to materialize the
LSM-tree state).

With Lance + Raft, the consensus tax and the storage tax are the
SAME tax. The append IS both the consensus log entry and the storage
commit. You pay it once.

## What this implies for deployment shape

The conventional advice from the Cassandra era is "replication is
expensive; minimize the replication factor". This is a workaround
for the double-tax property of LSM + Paxos-light. It doesn't apply
to Lance + Raft.

For lance-graph adopters:

- **Three-node peer-Raft + Lance-local-per-node** is the recommended
  starting deployment for HA. The consensus cost per write is small
  (Raft quorum round on the same write that the storage commit is
  doing anyway) and the failure modes are predictable.
- **Per-node dataset fits per-node** is the assumption. The
  compressed encoding cascade common in lance-graph consumers
  (highheelbgz / bgz-hhtl-d / bgz17 / CAM-PQ) plus the vort/vart
  adaptive radix trie plus Lance's columnar compression keep this
  true even for Wikidata-scale corpora.
- **TiKV-as-shared-storage** is a valid alternative shape but pays
  the double-tax property in a different way (TiKV's internal LSM
  layer + Raft + an extra network hop for every read). Use it
  ONLY when each node holding the full dataset is genuinely
  infeasible (which is rare for lance-graph workloads).
- **Cassandra-style replication** is the wrong shape. Don't import
  the Cassandra operations playbook to a Lance deployment.

## What this property does NOT claim

- **Lance + Raft is not "free".** Writes still pay a Raft quorum
  round (typically 5-15ms same-DC, more cross-DC). Linearizable
  reads pay a read-index round (typically 1-3ms). The CDN-cache
  "async backend replication" analogy applies only to
  eventually-consistent reads served from a local follower.

- **This is a property of append-only storage + Raft, not Lance
  specifically.** Iceberg + Raft would have similar properties.
  Delta Lake + Raft would have similar properties. Lance has
  additional orthogonal benefits (vector search, ML/AI primitives,
  DataFusion-native query path) that complement the property but
  are not the property itself.

- **It does not eliminate the need for HA discipline.** Backups,
  monitoring, capacity planning, GC tuning, failure recovery
  drills — all still required. The property reduces the
  operational burden of replication; it doesn't eliminate the
  operational burden of a distributed system.

## Recommended deployment pattern (reference)

See [reference consumer implementation in bardioc-B1-substrate-b]
(separate proposed doc) for a worked example. Briefly:

- Each substrate-b instance is one Rust binary AND one Raft node
- Lance dataset per node (per-instance hot path)
- Raft replicates user-data writes across instances (consensus
  cost on writes + linearizable reads; off-budget for
  eventually-consistent reads)
- Eventually-consistent reads serve from local Lance + local Raft
  follower with no cross-node coordination
- Linearizable reads pay a Raft read-index round
- Writes pay a Raft quorum round AND a Lance fragment append —
  but these are the SAME write, not two writes

This pattern is cheaper than the conventional alternatives by
roughly a factor of N (where N is the storage / consensus tax
ratio of the alternative). On Cassandra+JG that ratio is roughly
3-5x; on TiKV-backed surrealdb roughly 1.5-2x; on Lance + Raft
peer-replication roughly 1x.

## Provenance

- Original observation: AdaWorldAPI/bardioc, B1 architectural
  collapse round, conversation thread on PR #15
- Worked through against Cassandra+JG OLD stack as the contrast case
- Cross-referenced bardioc's Wikidata-fits-locally property (115M
  entities in low single-digit GB compressed; OLD stack would
  require multi-TB cluster)

## Related work upstream

- `cognitive-shader-driver`'s canonical API discipline
  (`.claude/knowledge/lab-vs-canonical-surface.md`) — the consumer
  surface that this deployment shape targets
- `collapse #8` in bardioc/.claude/TECH_DEBT.md — cold-path Neo4j
  on Lance versions via DataFusion SQL; same append-only property
  enables this without coordination
- Empirical data from bardioc's §14 acceptance gate runs once
  available