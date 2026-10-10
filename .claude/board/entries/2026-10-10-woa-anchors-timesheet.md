# D-XGP-8 — WoA anchors: the hours row is TimeSheet (2026-10-10)

**Status:** FINDING, VERIFIED-IN-CODE + TEST-PINNED (`lance-graph-glove-parity/tests/authority.rs`). Plan `cross-glove-business-parity-v1.md` §C.9.5.

**Finding.** OGAR's `WoaPort` resolves `TimesheetActivity` (and `Stundenzettel` / `TimeEntry` / `Zeiterfassung`) to `BILLABLE_WORK_ENTRY` (`0x0103`). In WoA, `TimesheetActivity` is a child row holding only `beschreibung` and `created_at`. The hours live in its parent `TimeSheet`: `datum` (DATE NOT NULL), `minuten` (INTEGER), `user` (→ `User`), `tenant_id`, `abgerechnet` (BOOLEAN). `TimeSheet`, `User` and `Tenant` all carry classid `0x0000_0000` in the harvest. So the convergence pin is on the description row, not the hours row.

**Shipped (`lance-graph-glove-parity::basis`):** `WOA_FIELDS` claims the three anchors from `TimeSheet`, all `Hypothesized`:

- `user` → `performed_by`;
- `minuten` → `duration` (an integer minute count; SAP `hours_logged` and Odoo `unit_amount` are hours, so a conversion is ×60);
- `tenant_id` → `tenant`.

`ANCHORS = [performed_by, duration, tenant]`; each glove (SAP, Odoo, WoA) claims each anchor exactly once (test-pinned). Unmapped on purpose:

- `datum`: the class has no temporal role, same as SAP `work_date_utc` and Odoo `date`;
- `abgerechnet`: means "already invoiced", an invoicing state, not `billable` ("may be invoiced"). Mapping it to `billable` would be wrong both ways.

**Two-sided pin** (`woa_pin_is_on_the_description_row_not_the_hours_row`): `WoaPort::class_id("TimesheetActivity") == Some(0x0103)` and `TimeSheet` / `User` / `Tenant` resolve to `None`. When OGAR moves the pin or mints those concepts, the test fails and the WoA claims get re-read.

**For the operator (OGAR change, not made here):** should `WoaPort` resolve `TimeSheet` to `0x0103` (and `TimesheetActivity` become a description child of it)? `Stundenzettel` is the German name of `TimeSheet`, not of `TimesheetActivity`, so today's alias table points the German name at the wrong row too.

**Next:** with the three anchors claimed by all three gloves, the remaining step is a `Converted` grade for one anchor: `duration` is the candidate (SAP hours × 60 = WoA minutes, provable on rows once a WoA fixture exists).
