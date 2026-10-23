# Torque — ideas we are moving on right now (append-only)

> **Append-only.** Prepend new rows. Never delete a row; the mutable fields
> are `**Status:**`, `**Last moved:**` and `**Evidence so far:**` (append
> lines, never rewrite). Added 2026-10-23.

## What a torque row is

An `IDEAS.md` Open row, or a named crumb trail, that a session is actively
probing. The record exists so the momentum survives a context reset or a
handover: the next session reads the evidence chain and continues instead of
restarting.

Every row carries:

- `**Owner:**` session id(s); co-owners welcome, claim in Linear EXO-50;
- `**Fruit:**` `Low-hanging` | `High-signal` | `Heavy` (admits the row to `FRUITS.md`);
- `**Source:**` the IDEAS row or crumb trail it came from;
- `**Evidence so far:**` probe → result → next, appended;
- `**Next probe:**` one, pre-registered before code;
- `**Kill:**` the result that ends it;
- `**Promote:**` the result that sends it to the EPIPHANIES gate;
- `**Last moved:**` date;
- `**Status:**` `TORQUE` | `STALLED → IDEAS` | `KILLED (<receipt>)` | `PROMOTED → EPIPHANIES`.

## The stall rule

A row whose `Last moved:` is more than **14 days** old is regraded
`STALLED → IDEAS`. Its IDEAS row is reactivated with the evidence chain
appended. Nothing is deleted. The `/board` sync reports stale rows. The 14-day
window is a policy pin, not a measurement. Revisit it once a month of rows
exists.

## Promotion

`PROMOTED → EPIPHANIES` needs the probe to have run, the three-way gate
(NEW / LOAD-BEARING / DURABLE, root `CLAUDE.md` § Closeout) to pass, and the
operator's word. The EPIPHANIES row cites this row and the entry the probe
landed in (`epiphany_provenance.py`).

---

## Rows

### T-2026-10-23-1 — does a PaletteLut learned from data compose where ComposeTable does not?
**Owner:** CC:session_01G22yT6htkcdyXsihxxXdrg
**Fruit:** High-signal
**Source:** BREADCRUMBS trail T-CLOSURE; Linear EXO-129 (High-signal, Open end).
**Evidence so far:**
- 2026-10-23: P-S2 (#1726). `ComposeTable` is UNSTABLE in both worlds
  (agreement 0.025 / 0.043 at k=2), because the codebook is not closed
  under bind.
**Next probe:** pre-register, then build a synthetic world with a known true
transition map. Compare a `PaletteLut` filled from observed transitions
against `ComposeTable` and the exact chain at k = 1..6, in two unlike worlds.
**Kill:** the learned LUT also falls below 0.90 agreement at k=2 in both
worlds.
**Promote:** it holds ≥ 0.90 at k=2 in both worlds, and a disable run
(shuffled transitions) collapses it.
**Last moved:** 2026-10-23
**Status:** TORQUE
