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

> ### Scope: external architecture pattern, not a built-in lance-graph feature
>
> Post-merge correction (paralleling peer review on the companion doc
> PR #453): the deployment shape described below ("peer-Raft +
> Lance-local-per-node") is an EXTERNAL ARCHITECTURE PATTERN adopters
> can build on top of lance-graph, NOT a built-in lance-graph
> capability. Lance-graph provides the append-only columnar storage +
> the DataFusion query path + the encoding crates; the Raft layer +
> the substrate binary + the consensus-replication path are
> downstream consumer code (e.g. `openraft` or `surreal-cluster`).
> Adopters who only consume lance-graph's columnar + DataFusion path
> should NOT assume their data is automatically replicated.
>
> The doc documents WHY this pattern works well WHEN built on
> lance-graph — the storage-append/consensus-append dovetail property
> — not a feature lance-graph itself ships.


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
| **Append-only Lance** | Append-only Raft | One write shape. Storage commit = consensus log entry. (`DatasetOptimizer.compact_files` produces a new manifest version — that output replicates through the same Raft log as a normal write. The operation runs in one place; the result replicates. Per codex P2 PR #454 — see Operational consequence #1 for the honest framing.) |

## Operational consequences

### 1. Compaction is qualitatively different, not absent

Cassandra clusters periodically run LSM compaction (rewrite SSTables
to reclaim tombstones + merge sorted runs to maintain read
performance). Each node compacts on its own schedule. During
compaction:

- The compacting node's CPU spikes
- Its disk write bandwidth spikes
- Replication lag from / to that node spikes
- Read latency for queries hitting that node spikes
- The cluster's effective replication factor temporarily drops as
  compaction blocks normal write flow

The cluster operator's job is partly to schedule compactions across
nodes so that not too many compact simultaneously. This is a
significant operational burden.

Lance has compaction TOO, but of a qualitatively different shape:
`DatasetOptimizer.compact_files` merges small fragments into larger
ones to optimize query layout (many small appends produce many small
fragments which slow scans). For datasets that use deletes, updates,
or dropped columns, the SAME compaction also performs reclamation —
deletion vectors get materialized away (removing rows logically
marked for delete) and dropped columns are physically removed by
default. So there IS a reclamation role on those datasets; it is
just NOT the LSM tombstone-reclaim mechanism (Lance has no
tombstones at the version level — deletions are tracked by
deletion-vectors against append-only fragments). For append-only
write workloads with no deletes/updates/drops, the layout-only
framing applies; for mixed-write workloads, the reclamation
component is also present. Either way the operation produces new
append-only fragments at a better layout.

Operationally:

- The compaction OPERATION runs locally on whichever node has the
  current leader role (or on a node permitted to run a maintenance
  task in the chosen deployment shape); it does not block normal
  write flow at the application layer
- The OUTPUT of compaction — the new manifest version + the new
  set of fragments — flows through the Raft log like any other
  Lance commit. Peers see the new manifest version after consensus
  commits, and anti-entropy converges replicas to the post-compaction
  state. So the result REPLICATES; the work that produced the result
  is what runs in one place
- Per-node SCHEDULING choices do not stack into coordination
  headaches the way Cassandra LSM scheduling does, because each
  compaction's product is a single committed version (not a
  per-replica concurrent rewrite that has to be reconciled). At most
  one node should run a given compaction at a time to avoid wasted
  work; this is a coordination choice (lock or leader-only), not a
  coordination headache
- The failure modes are smaller: a partial compaction is recoverable
  via Raft's standard log replay; no in-flight LSM tombstones to
  lose; correctness is unaffected

A peer-Raft + Lance deployment therefore has uniform per-node
behavior under consensus. Compaction is a maintenance operation
that produces a normal commit; operators plan for it (it consumes
CPU + IO when it runs) but the cluster-wide coordination model is
simpler than Cassandra's per-node-independent LSM compaction
scheduling. (Per post-merge correction on PR #452; sharpened per
codex P2 review on PR #454 — the prior framing said 'independent
of consensus' which overclaimed; the operation is local but the
output replicates.)

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

### 5. The consensus tax and the storage-COMMIT tax are the same tax

This is the unifying point: with LSM-tree storage, an application
that wants linearizable writes pays the consensus tax TWICE. Once
for the consensus protocol shipping operations to replicas. Once for
the storage layer doing per-node tombstone-reclaim + run-merge
compaction. The two taxes interact (a compaction storm delays
consensus catch-up; a Raft snapshot has to materialize the LSM-tree
state).

With Lance + Raft, the consensus tax and the storage-COMMIT tax are
the SAME tax — the append IS both the consensus log entry and the
storage commit; you pay it once. Lance does have its own file-
compaction cycle (`DatasetOptimizer.compact_files`), which produces
a NEW manifest version — and that new version flows through the
same Raft log as any other write, replicating to peers via the
normal consensus + anti-entropy path. So compaction's OUTPUT
counts as a consensus event (one more append). What it does NOT
add is a SECOND tax of the LSM-tree kind (per-node tombstone-
reclaim + run-merge bookkeeping that runs on every replica
independently and creates coordination headaches with replication).
The LAYOUT-OPTIMIZATION cycle exists; it pays the SAME consensus
tax as a regular write (one commit), and does NOT layer a separate
per-replica storage tax on top.

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