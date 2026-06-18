# mailbox-cycle-aware-write-contract-v1 — every SoA write carries/checks its cycle

> **Status:** CONJECTURE / design. 5+3-gated before code.
> **Date:** 2026-06-18.
> **Parent:** `bindspace-singleton-to-mailbox-soa-v1.md` rule 1 (`E-SOA-CYCLE-OWNERSHIP`).
> **Owns:** *"the timing needs to be assigned to the mailbox; nothing buffers a stale/older
> mailbox or writes without cycle awareness — per cycle crystal-clear via the LE contract of
> the tenant AND the envelope."*

---

## Epiphany (less is more)

**Today every per-row write is cycle-blind.** `consume_firing` is the *only* cycle-aware
mutator. The 11 `set_*` setters on `MailboxSoA` and the 8 on `BackingStoreWrite` do
`self.col[row] = v` — no clock, no gate. A late batch from cycle *c-3* can silently overwrite
a row the owner already advanced past. **The fix is one cycle-aware write path; the per-field
setters stop being public mutation doors.**

---

## The gap (precise)

| Surface | Cycle-aware? |
|---|---|
| `MailboxSoA::consume_firing(row)` | ✅ checks `last_active_cycle[row] == current_cycle`, stamps it |
| `MailboxSoA::set_{edge,qualia,meta,entity_type,temporal,expert,sigma,content,topic,angle}` | ❌ blind `self.col[row] = v` |
| `BackingStoreWrite::set_{content,qualia,edge,meta,entity_type,temporal,expert,sigma}` | ❌ blind passthrough |
| `engine_bridge` writers (`dispatch_busdto`, `persist_cycle`, `write_qualia_17d`, `ingest_codebook_indices`) | ❌ call blind setters |

`last_active_cycle: [u32; N]` (sentinel `u32::MAX`) is the **consumption** stamp — overloading
it for writes would corrupt `consume_firing`'s same-cycle idempotency guard
(`I-LEGACY-API-FEATURE-GATED`: same name, different semantics under a flag = forbidden).

---

## Council resolutions (5+3, 2026-06-18)

The 5+3 (convergence / dto-soa / trajectory / integration-lead builders + brutally-honest
critic, with operator architecture direction) resolved the OQs and fixed two P0s:

- **OQ-A → two stamps.** Add `last_write_cycle: [u32; N]`; never overload `last_active_cycle`
  (would break `consume_firing`'s exact-match same-cycle idempotency — confirmed DROP). The
  phase-bit single-field pack is type-isomorphic but guard-divergent → deferred probe
  `OQ-CSV-CYCLEPACK`, not now. Add `last_write_cycle` to **`reset_row`** + the **field-isolation
  matrix test** (a new `[u32; N]` that `reset_row` forgets is the exact leak the test catches).
- **OQ-B → no header field, no version bump.** The batch/owner cycle is already `self.current_cycle`
  (and `SoaEnvelope::cycle()` on the trait side). `write_row`'s `cycle` param is compared against
  `self.current_cycle`. `ENVELOPE_LAYOUT_VERSION` stays 2.
- **OQ-C → Aware-buffer** for `cycle > current_cycle` (Strict-reject loses concurrent producer
  work in the multi-producer interlace target). Use a write-side `WriteDisposition`, do NOT reuse
  the reader's `EpistemicMode` verbatim.
- **OQ-D → infallible `WriteOutcome::{Accepted, Stale, Future}`.** The W4b `Arc::get_mut` Result
  precedent is a CATEGORY ERROR here: ownership is compile-proven (`&mut self`, E-CE64-MB-4), so
  stale/future is a valid in-domain *outcome*, not an aliasing failure. (If a sub-call can fail,
  that sub-call is `Result`, separately.)
- **OQ-E → S2.5**, its own pre-S3 node behind `mailbox-thoughtspace` (default-OFF); S3 then
  dissolves the singleton onto an already-cycle-gated surface. Not blocked by surrealdb/D-MBX-9;
  #535 cleared the `with-engine` break.

### P0 fix — de-interlace is ADDRESSING, not planner-routing (operator direction)

The original §2 routed stale writes to `lance-graph-planner/src/temporal.rs`. **Rejected** —
`temporal.rs::deinterlace()` is a read-only projection (no write sink), and the planner is
unreachable from a `mailbox-thoughtspace` build (feature-graph: `with-planner`/`lab` only).
Replacement:

- **Mailbox pointer at spawn = the GUID identity tail (last 6 hex / `identity(24-bit)`)**, the
  canonical bootstrap address; or an **ephemeral version-time-series-aware pointer** mapped to it.
  A write finds its mailbox by *identity*, so "which mailbox owns this write" is an addressing
  lookup, not a query-time de-interlace.
- **Stale-write handling is LOCAL** to the mailbox — a small `stale_writes` buffer (Aware) or a
  drop-with-telemetry counter (Strict) — **no `lance-graph-planner` dependency**. Testable in
  isolation with synthetic stale batches.
- **Cold time-series + kanban stay Lance-native** (lancedb 0.30 / lance 7.0.0) — Lance versions
  ARE the time series; do NOT invent a surreal-specific TS format (resolves OQ-11.6/D-MBX-9
  direction: stay native).
- **Setters stay `pub` + `#[doc(hidden)]`** with a migration pointer to `write_row` — `pub(crate)`
  breaks `tests/w2_differential.rs` (separate compilation unit). A test/debug_assert proves no
  production path reaches a blind setter once `write_row` is the live door (blind+gated coexistence
  is the I-LEGACY-API-FEATURE-GATED hazard; the guard is mandatory).

### Scale framing (operator, deferred — NOT this deliverable)

- **Ractor recycling:** reuse a pool of ~16k ractor mailboxes across SoA generations; **16k
  mailboxes per cycle = one table prefix** (one basin of the ~1024).
- **Full-sweep latency:** 1024 prefixes × ~0.5–2.5 s/cycle (16k substrate cost) ≈ **8–40 min** for
  a full 16M-envelope pass. Sizing only; not in scope until the cycle-aware foundation lands.

---

## Primary proposal — P1: stamped envelope + gated `write_row`

1. **One mutator.** `fn write_row(&mut self, row, cycle: u32, cell: &WriteCell) -> WriteOutcome`.
   The 11 `set_*` stay **`pub` + `#[doc(hidden)]`** with a migration pointer to `write_row`
   (NOT `pub(crate)` — that breaks `tests/w2_differential.rs`, a separate compilation unit).
2. **Gate — WRAP-AWARE, against `self.current_cycle` (NOT planner-routed):**
   - Compare via `current_cycle.wrapping_sub(cycle)`: `== 0` → **Accepted** (stamp
     `last_write_cycle[row] = cycle`); `< 0x8000_0000` → **Stale**; else → **Future**. Naive
     `</>` misclassifies post-wrap stragglers as Future across the 8–40 min sweep — wrap-aware is mandatory.
   - **Stale/Future handling is LOCAL** (per the P0-fix block above): a `stale_writes` buffer
     (Aware/`WriteDisposition::Buffer`) or a drop-with-telemetry counter (Strict). **No
     `lance-graph-planner`/`temporal.rs` dependency** — the rejected §-original temporal routing
     is dead; do not re-introduce it.
   - **Singleton arm is cycle-blind BY CONSTRUCTION (CATCH-CRITICAL fix).** `BindSpace` owns no
     `current_cycle`; `BackingStoreWrite::Singleton` returns `WriteOutcome::Accepted`
     unconditionally with a `debug_assert`/doc: *"the cycle gate is a Mailbox-only guarantee until
     W7 deletes BindSpace."* The differential harness asserts cycle-gating ONLY on the Mailbox arm
     (else the uniform signature is a C2-divergence sentinel-lie / I-LEGACY hazard).
3. **Two stamps, not one.** Keep `last_active_cycle` (consumption). Add
   `last_write_cycle: [u32; N]` (write). `consume_firing` untouched. **Both `last_write_cycle` AND
   the new `identity` field MUST be added to `reset_row` + the field-isolation matrix test in the
   SAME commit** (a `[u32;N]`/`u32` that `reset_row` forgets is the exact leak the test catches —
   iron-rule mandatory). **(OQ-A)**
4. **LE contract, byte-explicit — NO version bump (OQ-B).** The owner cycle is already
   `self.current_cycle` (and `SoaEnvelope::cycle()` trait-side); `write_row`'s `cycle` compares
   against `self.current_cycle`. `ENVELOPE_LAYOUT_VERSION` stays **2** — holds as long as
   `identity` lives in the GUID **key** (recomposed positionally from arena `family` + row), NOT a
   new persisted value column.
   - *Spawn pointer:* new `identity: u32` field = canon GUID identity tail (last 6 hex / 24-bit;
     high byte zero, `debug_assert`). Do **not** overload `mailbox_id` (its meaning is the
     corpus/`classid` handle). `family`/basin is a table-level constant (one per arena, 16k
     mailboxes = one prefix table). Constraint: a `mailbox_id` used as a spawn pointer must be
     `<= 0x00FF_FFFF` (else `NodeGuid::new` panics) — route through `NodeGuid::local(identity)`.
   - *Tenant:* `WriteCell { row, cycle, <field-presence payload> }` — carries `(row, cycle)` so a
     buffered stale write stays self-describing. Plan MUST enumerate which of the 8 columns each of
     the 4 writer sites (`dispatch_busdto` / `persist_cycle` / `write_qualia_17d` /
     `ingest_codebook_indices`) writes, so no site forgets `cycle`.
5. **Feature gate.** Land behind `mailbox-thoughtspace` (default-OFF, same gate as `BackingStore`).
   The blind setters coexist during transition ONLY with a `debug_assert`/test proving **no
   production path reaches a blind setter once `write_row` is the live door** — this guard ships in
   the SAME commit as `write_row` (its absence = VIOLATES I-LEGACY-API-FEATURE-GATED).

### Alternatives (for the council to weigh)

- **P2 — stamp-only (observability).** Record `last_write_cycle`, never reject. Cheap, weak: a
  stale write still lands; only detectable post-hoc. (Likely too weak for "nothing buffers stale".)
- **P3 — type-state `CycleToken<'c>`.** Setters require `&CycleToken<'c>` borrowed for one cycle;
  cross-cycle write = compile error (E-CE64-MB-4 flavor). Strongest, most invasive; touches every
  call site. Possibly S3-era, not now.

---

## Open questions — ALL RESOLVED by the 5+3 (see "Council resolutions" above)

- **OQ-A → two `[u32; N]` stamps** (phase-pack deferred as `OQ-CSV-CYCLEPACK`).
- **OQ-B → no header field, no version bump** (reuse `current_cycle`; identity stays in key).
- **OQ-C → Aware-buffer** via write-side `WriteDisposition` (not `EpistemicMode` reuse).
- **OQ-D → infallible `WriteOutcome::{Accepted,Stale,Future}`** (W4b `Arc` precedent = category error).
- **OQ-E → S2.5**, own pre-S3 node behind `mailbox-thoughtspace`.

Remaining (deferred, not gating this deliverable):
- **OQ-CSV-CYCLEPACK:** can a phase-tagged single `[u32; N]` carry both stamps without coupling
  `consume_firing`'s exact-match guard to the write gate's ordering guard? (Probe behind a
  four-invariant differential test, incl. the `2^31` wrap + `u32::MAX` sentinel.)

---

## Cascade — split into 3 landable increments (integration-lead)

**Inc 1 (one PR) — contract floor.** `lance-graph-contract` confirm `SoaEnvelope::cycle()` reuse;
**NO `ENVELOPE_LAYOUT_VERSION` bump** (OQ-B). No engine gate, no surrealdb.

**Inc 2 (one PR) — gated mutator + stamps + identity.** `mailbox_soa.rs` (`write_row` wrap-aware,
`last_write_cycle: [u32;N]`, `identity: u32`, setters stay `pub`+`#[doc(hidden)]`+pointer,
`reset_row` clears both new fields, `stale_writes` local buffer/telemetry) + `backing.rs`
(`BackingStoreWrite` cycle param; **Singleton arm cycle-blind-by-construction**) + tests
(differential stale/current/Future, `consume_firing` unaffected, field-isolation matrix EXTENDED to
`last_write_cycle`+`identity`, no-production-blind-path guard). All behind `mailbox-thoughtspace`.

**Inc 3 (splittable) — consumers.** `engine_bridge.rs` 4 writer sites carry `cycle` via `WriteCell`
(post-#535 `with-engine` fix). **`temporal.rs` is NOT in the cascade** — de-interlace is addressing
(GUID identity tail) + local stale handling; no planner dep.

Cold time-series + kanban (S4-era) stay **Lance-native** (lancedb 0.30 / lance 7.0.0) — Lance
versions ARE the time series; no surreal-specific TS format.
