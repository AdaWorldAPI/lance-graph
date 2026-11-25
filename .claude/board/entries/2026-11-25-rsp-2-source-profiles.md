# 2026-11-25 — D-RSP-2: source profiles shared by tenants

**Status:** TEST-PINNED. Plan: `.claude/plans/registered-source-profiles-v1.md` D-RSP-2.

- `SourceProfile::new(class, [(label, iri)], config)` (`lance-graph-intake`)
  registers one class once — classid, fields under caller-supplied source
  IRIs, the hand config, the `BindStore` it is registered and routed in — and
  is shared as an `Arc`.
- `Table::from_rows_in(name, &profile, columns, rows)` builds a tenant table
  with its own `FrameStore`, `MapTable` and `CodeTable`. `Table::from_rows` is
  `from_rows_in` over a private profile with the `intake:{table}/{column}`
  IRIs, unchanged for every existing caller.
- Gate (`tenants_share_a_profile_but_never_a_frame_or_a_code`): two tenants of
  one profile carry the same IRIs (`graph:license/prepaidUnits` under any table
  name); tenant A's frame is `ForeignFrame` on B; code 1 is `north` in A and
  `south` in B, and A's frame decodes nothing in B; columns that are not the
  profile's labels are `ProfileMismatch`. Disable runs (mismatch check off;
  per-table IRIs in the owner) are red.

OPEN: the scope issuer stays D-LANESOCK's item (every table still reads under
intake's fixed `READER` scope); Quack's `Binder` still binds by column name
(the D-RSP-5 field seam).
