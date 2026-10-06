# 2026-10-06 — D-ALPHA-G-0 + D-WA-0: WorldG from the Alpha route; Witness × Angle as a declared local source

## VERIFIED-IN-CODE — inventory (main @ 705509fe)

1. **Where WorldG is selected.** `lance_graph_contract::spog_tenants`:
   `graph_of(addr) = classid >> 16` (`spog_tenants.rs:41`) and
   `SpogTenants::claim` (`:167`), which routes an attended address to the tenant
   of its world and returns `TenantClaim::NoTenant` for an undeclared one.
   `AlphaTunnel` (`alpha_tunnel.rs`) is the same split tunnel keyed by rung, not
   by world.
2. **What preserves G across the saccade.** Every claimed row keeps the base
   row's `NodeGuid` byte-for-byte (`alpha.rs` `claim`), so the world is in the
   attention record. `SpogTenants::route` records the world of each fresh claim;
   `merge_in_claim_order` replays the saccade in time.
3. **Can two worlds feed one `MailboxSoA` in production?** No path exists:
   `MailboxSoA::apply_edges` has no production caller (only its own tests and
   the probes). The driver's mailbox backing expects at most one designated
   mailbox (`debug_assert` under `mailbox-thoughtspace`, `driver.rs:208`;
   multi-mailbox routing is the unbuilt W5). Nothing connects `SpogTenants` or
   `AlphaTunnel` to a mailbox.
4. **#1371's divergence** is constructible only, by handing a foreign-world edge
   to `apply_edges` directly.
5. **Where the split tunnel is tested.** `alpha_tunnel.rs` (parallel ==
   sequential, byte-identical; cross-rung revisit counting) and
   `spog_tenants.rs` (tenant isolation, claim-order replay).
6. **Saccade provenance exists:** `AlphaStamp` (cycle, seq, rung, visits) plus
   `SpogTenants::route`.
7. **G recoverable on replay without CE64:** yes, from the attended key and the
   route.
8. **16-byte hosts for an explicit row-local SPOG later:** `Register128`
   (classid-free, meaning from the resolved SPOG context), the 4 + 12
   `FacetCascade` (also the edge block), `Quad8`. None carries a LocalG today.
9. **Odoo/ERP LocalG:** none at runtime. `company_id` exists only as a
   blueprint field (`odoo_blueprint/l3.rs`); the Odoo hydrator keys an OWL
   context bundle by namespace.
10. **Witness (53..58) and bits 43..45** are disjoint and readable together
    without moving bits. Both already have shipped readings that must refuse a
    new one: bits 43..45 = pathology triad (`direction()`), bits 53..58 =
    cohort `WitnessTable` index (`witness_table.rs`, not wired).

## MEASURED

- `alpha_world_provenance_probe.rs`: 5 tests, 5 disable runs red (world read
  from the edge, grouped merge instead of the route, all deliveries to one
  mailbox, a mixed received count, `NoTenant` accepted). The first mailbox
  disable came back green; the test now reads each mailbox's own
  `plasticity_at` counts. After review (#1372) the mailbox deliveries are built
  from the tunnel's per-world shadows (`tenant(w).scanpath()`), not routed by
  the test; feeding both mailboxes from one shadow fails it.
- `witness_angle_probe.rs`: 7 tests, 8 disable runs red (law ignores class,
  ignores generation, reading check off, generation check off, Witness 0
  accepted, v1 accepted, Angle read from other bits, law keyed by world instead
  of class); 0 allocations. The world-keyed disable needed a second
  source-coordinate class in the same world, added after review (#1372).

## RESOLVED

`ISS-MAILBOX-ROUTES-WITNESS-WITHOUT-GRAPH`: not a `MailboxSoA` defect.
`MailboxSoA` stays graph-blind; no G check in `apply_edges`. The residue is the
unbuilt join below.

## OPEN

- Where production will join an attended address to the CE64 it produced: no
  caller does this yet, so the event pairing exists only in the probe. When
  built, it must feed mailboxes from the tunnel's per-world output (the probe's
  `tenant(w).scanpath()` shape), never by re-deriving the world itself.
- Whether `graph_of` is the right WorldG granularity for every consumer.
- Whether the Witness × Angle declaration (and the D-SPOG-W-0 one) becomes a
  contract declaration beside `band_reading`.
- LocalG (row-local SPOG G in a 16-byte register): not modelled.
