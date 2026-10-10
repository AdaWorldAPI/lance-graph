# Handover: HubSPO-rs → lance-graph — wishlist (2026-10-10)

> APPEND-ONLY. Consumer-side wishlist from HubSPO-rs (the CRM consumer on
> `HubSpoPort` `0x000B`). Read against lance-graph `a75bcc42`.
> - The meaning half of W-1..W-3 is in
>   `OGAR/.claude/handovers/2026-10-10-1600-hubspo-rs-to-ogar-dir-sim.md`.
> - The thread-key item is in spear (`spear/.claude/handovers/`).
>
> Every item is a missing capability the consumer is not allowed to build
> locally (STOP rule). Evidence comes from HubSPO-rs PR #17
> (`.claude/knowledge/comms-api-surface.md`, probe P-QDB-1) and the
> HubSPO-rs `ISSUES.md` entries named per item.

## dir-sim execution (`crates/lance-graph-dir-sim`)

**W-1 — Mailbox delegation, executed** (meaning: OGAR W-1)
- Store delegations as a sparse relation lane, the way memberships are
  (`snapshot.rs:33-41`: sorted `u16` ordinal pairs, unresolved endpoints
  kept by identity).
- Add a query, e.g. `View::may_act_for(delegate, owner, right) -> bool`.
- Cover delegations in `simulate` / `diff` / `plan`.
- Emit them from `ad::project`: `publicDelegates` for SendOnBehalf.
- HubSPO-rs needs this for send-as / on-behalf (D-MAIL-3) and for working a
  colleague's inbox (D-MAIL-5).
- A search over both trees for `send.?as|on.?behalf|FullAccess|publicDelegates`
  finds nothing.

**W-2 — An `ActorSource` over a `View`**
- Implement `lance_graph_contract::rbac_plug::ActorSource` with roles taken
  from group membership in one version, and scope from the owner (spear's
  `DirectoryActors` is the worked example).
- Implementations of `ActorSource` today:
  - `lance-graph-contract/src/rbac_plug.rs:449` (test);
  - `lance-graph-rbac/src/authorize.rs:501` (test);
  - OGAR `IdentityActors`, built from an authenticated user, not a
    directory.
- HubSPO-rs needs this for team-scoped record access (D-IAM-2).
- P-AUTH-1 measured the scope mask this would feed: ≤ 4.3 % kernel delta,
  word-wise.

**W-3 — Persist a version**
- `VersionStore` is in memory only (`store.rs`, `Vec<Version>` + map).
- What HubSPO-rs needs:
  - write and read back a version as a Lance dataset version, with
    provenance and tags;
  - restoring a version undoing a directory change;
  - a restart that does not re-observe AD.
- `ad::project` (#1452) is export only.
- This also unblocks the one remaining path to a stalwart dir-sim directory
  (HubSPO-rs OD-12 option b).

**W-4 — Transitive group members**
- Nested groups are held by identity and never expanded (`snapshot.rs:38-41`).
- spear computes the closure itself (`mailbox_members`, cycle-safe), and
  HubSPO-rs would need a third copy for ticket routing.
- Ask: `View::members_transitive(group) -> impl Iterator<Item = Guid128>`,
  in user-ordinal order, cycles walked once.

## Report seam (`crates/lance-graph-report`)

**W-5 — A borrowed lane variant** (HubSPO-rs `ISS-LG-REPORT-OWNED-LANES`)
- `LaneData` owns `Arc<[i32]>` / `Arc<[u32]>` / `Arc<[u64]>` (`batch.rs:26-34`).
- So publishing a Lance version into an `AbiBatch` copies every lane once,
  and graph-flow-lance and z8run-lance inherit the copy.
- mask-risc itself borrows (`LaneRef`). P-QDB-1 ran nine Quack programs
  straight over Arrow buffers from Lance (`LaneRef::U32(array.values())`).
  That works today without the report seam; the report seam is the only
  copy left.
- Ask: a variant over an Arrow buffer, or a borrowed slice with a lifetime.

**W-6 — Generation width** (HubSPO-rs `ISS-LG-REPORT-GENERATION-WIDTH`)
- `AbiBatch::generation` and `plan.rs:288` are `u32`; `LanceVersion` is
  `u64`.
- With "generation = Lance version", the mapping must fail closed past
  `u32::MAX`, unless the field widens.
- Ask: widen to `u64`, or name the fail-closed mapping in the contract.

## Not asked

- A table catalog for Quack. HubSPO-rs implements `quack::bind::Registrar`
  and `Binder` itself; P-QDB-1 did this in about 80 lines.
- `ORDER BY` or strings in Quack: storage order and codebooks cover
  HubSPO-rs.
- Anything from DataFusion (grace period).
