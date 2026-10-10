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

**For the operator (OGAR change, not made here):** should `WoaPort` resolve `TimeSheet` to `0x0103`? See the correction below for "Stundenzettel".

**Correction (operator, 2026-10-10: "I believe OGIT has the time sheet concept minted. Stundenzettel should be a dto label"; verified against `AdaWorldAPI/OGIT` `2315167`).**
- OGIT mints `ogit.WorkOrder:TimeSheet` (`NTO/WorkOrder/entities/TimeSheet.ttl`), plus `ogit.WorkOrder:User` and `ogit.WorkOrder:Tenant` in the same folder. "Has classid `0x0000_0000`" above is true only of the OGAR harvest and codebook: those concepts have no OGAR id and no `WoaPort` alias. They are not unminted.
- OGIT confirms the reading: `minuten` is "duration … in minutes (integer)", rounded up to 15 minutes for billing; `abgerechnet` is the "billed-flag … transferred onto an invoice document"; `TimeSheet` `belongs` `Customer` and `Tenant`, and `relates` `User` (the verb `logsTime`: User → TimeSheet).
- **"Stundenzettel" is a DTO label of `ogit.WorkOrder:TimeSheet`, not a concept name** (its OGIT description begins "Stundenzettel — …"). OGIT has no `TimesheetActivity` concept.
- So the open OGAR question is: give `ogit.WorkOrder:TimeSheet` its `0x0103` alias in `WoaPort`, and treat "Stundenzettel" as a label of it rather than an alias name of its own.

**Next:** with the three anchors claimed by all three gloves, the remaining step is a `Converted` grade for one anchor: `duration` is the candidate (SAP hours × 60 = WoA minutes, provable on rows once a WoA fixture exists).
