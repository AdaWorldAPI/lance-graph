# D-DRISA decision memos: the owner questions the local register proof leaves open

> **Status:** PROPOSED. Written by this session after D-DRISA-1 (local register proof) and D-DRISA-3 (ALU semantics). Each memo states the question, the source it rests on, a proposed ruling, the falsifier that would decide it, and what follows from approval or refusal. Nothing here is a decision; each needs the named owner. Architecture: #1953 (D-DRISA-0, a plan by a GPT session, adopted after review). Tracking: EXO-153; handoff: EXO-50.

## D-DRISA-M1: a layout witness on the canonical SoA

**Question.** May the canonical SoA state which DR frame version and tile digests its rows are laid out under, so a register binding can verify the planes it borrows?

**Source.** `MapTable` owns layout (`dr.rs:805`); `TileMap::digest` (`:780`); nothing on the SoA side states the layout (D-DRISA-1 census). The lab `Soa::layout` carries `(TileId, digest)` per tile.

**Proposed ruling.** The SoA owner records `(MapTable identity, RouteVersion, per-tile digest)` when it lays rows out, read-only to consumers. No new map, no copy of ids.

**Falsifier.** D-DRISA-1 `a_permuted_coordinate_assignment_is_refused_and_would_be_wrong`: same ids, another local order. Refused with the witness; wrong answer without it.

**Approve:** register binding becomes verifiable without trusting the caller. **Refuse:** every binding must trust that planes and frame agree; a permuted layout answers wrong silently.

## D-DRISA-M2: a global id → tile index (EXO-46)

**Question.** Who owns routing a set of global ids to tiles? `MapTable::locate` reads every tile per id.

**Source.** `dr.rs:955`. D-DRISA-1 arm B at 256 tiles: 842,138 lookups per bind, above 1,000 µs per query (failed gate).

**Proposed ruling.** Either (a) an index owned by `MapTable`, versioned with it, or (b) no index: callers obtain selections co-addressed (the #1948 C′ form) so routing never happens per query. Not a new hash table outside DR.

**Falsifier.** The same arm B with the chosen mechanism: lookups per id ≤ 1 and bind under 1,000 µs at 256 tiles, with the D-DRISA-1 oracle unchanged.

**Approve (a):** binding cost falls to O(ids). **Approve (b):** no new structure; selections must arrive co-addressed. **Refuse both:** register files must be bound once and reused, never per query.

## D-DRISA-M3: result-register lifetime across calls

**Question.** May a derived mask (the D-DRISA-1 derived predicate register) outlive one call?

**Source.** In-program scratch is epoch-safe (`ExecError::ScratchReadBeforeWrite`). The derived register in the lab lives in one in-process object, stamped with the value revision; nothing owns it across calls.

**Proposed ruling.** A bounded result register keyed by `(frame issue, layout digests, scope, value revision)`, evicted without changing truth, refusing on any mismatch. No global mask store.

**Falsifier.** D-DRISA-1 `a_value_write_invalidates_values_but_not_the_map` and `disabling_the_value_check_returns_a_wrong_derived_answer`, run across two calls; plus #1953 N9 (reuse on another frame/scope/source must refuse).

**Approve:** repeated value predicates need not be recomputed. **Refuse:** results are scalars or egress masks only, recomputed per call.

## D-DRISA-M4: source-location receipt (EXO-46 / EXO-89 / EXO-151)

**Question.** Who hosts a receipt of resolved canonical locations?

**Source.** #1951 census: no production lease host; `drf-focus-probe::FocusKey` and `lance-graph-nsl::FocusLease` are excluded lab crates. D-DRISA-1 needed no receipt: within one process the SoA is borrowed directly.

**Proposed ruling.** Defer. D-DRISA-1 shows a local binding needs a layout witness (M1), not a location receipt. A receipt matters only once the source is not directly borrowable (Lance, remote).

**Falsifier.** #1953 R8–R16 once a host exists.

## D-DRISA-M5: DR → Lance row authority (D4 / S-INGRESS)

**Question.** Unchanged from #1950 and #1951: exactly one attested DR ordinal → `_rowid` seam, with ingress counted.

**Source.** #1950 `where_lance_arm` (`BLOCKED_D4`); ingress measured as a copy (0.4–5.4 µs for one range, 123,162 µs for 3.26M ranges).

**Proposed ruling.** No change by this work. D-DRISA-1 adds one input: the layout witness (M1) is the same kind of statement on the Lance side (a frame → fragment/row binding), so M1 and D4 should be designed together.

## D-DRISA-M6: the Abakus ALU as an instruction family

**Question.** Should the shipped carry words become the ALU for threshold, Kleene or Bag instructions?

**Source.** D-DRISA-3: correct on every case (19/19 tests, 7 disables red), but 1.5–4.8× slower than native words on the same facade in all eight measured shapes, including 3-of-5 against its Boolean expansion and Kleene against two byte counters. P3-5 (#1954): 1.7–4.0× slower for plain k-of-n.

**Proposed ruling.** Refuse as the default ALU on this evidence. Keep the semantics (the tests and the oracle) as the reference; lower k-of-n, Kleene and Bag thresholds to `U8x64` byte counters. Revisit only with a wide carry word (ndarray owner) or a shape byte counters cannot hold (counts above 255).

**Falsifier.** A carry implementation that beats `U8x64` on any of the eight shapes, with the D-DRISA-3 oracle unchanged.

## D-DRISA-M7: asynchronous loopback (stage 2)

**Question.** May stage 2 (in-process queue, deadline, cancel) proceed as a lab?

**Source.** #1953 charter PR2. D-DRISA-1 gives the descriptor it would carry: frame issue, layout digests, scope, value revision, compiled method.

**Proposed ruling.** Yes, as a lab, after review of D-DRISA-1. A queued descriptor must re-run the same revalidation at execution, so a source change between admit and run is refused by the existing checks (N1, N4, N12).

**Falsifier.** #1953 N1–N9 against the D-DRISA-1 refusals.

## Checklist

- [x] D-DRISA-1 local register proof (`entries/2026-11-04-d-drisa-1-register-proof.md`)
- [x] D-DRISA-3 ALU semantics (`entries/2026-11-04-d-drisa-3-alu-semantics.md`)
- [ ] Owner rulings M1–M7
