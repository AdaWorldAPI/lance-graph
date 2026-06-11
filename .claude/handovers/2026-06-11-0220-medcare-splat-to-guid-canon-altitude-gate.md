# Handover — the Altitude Gate (+ receipts owed) → guid-canon session

> **From:** session `claude/medcare-gaussian-splat-8z76jc` (splat / MTPPS / MedCare web-app arc).
> **To:** the guid-canon crystallization session (`01PBTGaPCSnnt`) — owner of the
> theorem-checker / PP-13 charter surface and `OGAR/CLAUDE.md` canon.
> **Date:** 2026-06-11. **Type:** APPEND-ONLY handover. **Code:** none (docs/proposal).
>
> This is a **proposal + receipts**, not a unilateral canon edit. The two
> canon-landing actions (an `EPIPHANIES.md` entry + a charter rule) are left for
> the owning session to take if it adopts the gate.

---

## 1. PROPOSAL — the Altitude Gate (`[H]`, sibling to theorem-checker rule-0)

**Before accepting a red probe OR a correction as a verdict, classify what it
attacks: the *flesh* (an implementation/code defect) or the *spine* (the
thesis / contract / the probe's own expectation). A red may be encoding the
probe's wrong assumption, not a code defect — and "fixing" the code to satisfy
it can break a correct mechanism.**

### Why it's needed (the gap it fills)

The charter already has two altitude-adjacent guards, and **neither covers this
case**:

- `theorem-checker rule-0` ("pin the unit system first") — catches a lens
  auditing the **wrong unit** (hex-nibbles read as bits).
- the φ-quorum anti-eigenvalue-theater contract — catches a **measurement
  masquerading as a verdict** (raw-XOR-u64 ordering as "nearest").

What's uncovered: a **probe whose own expectation is wrong, firing red against a
correct mechanism**. That red pattern-matches to "code is broken" (flesh), and
the reflexive fix breaks the spine.

### The grounding receipt (one worked case, one named near-miss)

**PROBE-HILBERT-L4.** `hilbert3d_encode([15,15,15],4) → 2925, expected 4095`
shipped as a P0 **red blocker** ("blocks every L4 cascade-addressing claim").
First-hand verification (ndarray #215, commit `44d104d7`) found the encoder
**correct and bijective** — 13/13 tests including `level4_all_indices_unique`
(onto `[0,4096)`, exactly what cascade addressing needs). The `==4095`
expectation was a wrong **orientation assumption**; `2925` is a valid endpoint
under the shipped orientation. **Near-miss:** a reflexive "make it 4095" fix
would have broken a correct space-filling curve to satisfy a wrong probe.

This is the gate in miniature: the red attacked the *spine* (a wrong
expectation), not the *flesh* (the encoder). Classifying altitude before fixing
is what saved the curve.

### Proposed shape (for the owning session to place)

A `rule-1` in the theorem-checker / PP-13 brutally-honest-tester family:
> *On any red probe or proposed correction, state its altitude: flesh
> (implementation) or spine (contract/expectation/thesis). If spine, the probe's
> own assumption is in scope — verify the expectation before touching the
> mechanism. A locally-correct fact must never stand in as a spine-refutation.*

Confidence **`[H]`** — one receipt + one near-miss; the generalization (that
this is a recurring, distinct failure mode) is conjectural until a second
independent instance lands. Falsifier: if every "wrong-expectation red" turns
out to also be a units error or a theater case, rule-0 + quorum already cover it
and this rule is redundant.

---

## 2. NOTE — address⊕payload = altitude (`[S]`, a lens, NOT doctrine)

Across this session's surface — GUID `key(128)+value(3968)`, spine/flesh,
Cesium tile-address/tile-content, theorem-checker units-vs-bits — the recurring
shape is **address ⊕ payload**, and the recurring bug is **altitude confusion:
reading the key as value or the value as key.** The three known altitude errors
fit it: geo/graph (coordinate-space read as traversal-machinery), NodeGuid
(struct read as canon), rule-0's origin (hex read as bits).

Held at **`[S]`** on purpose. It's a *description* that unifies known cases, not
a buildable with a falsifiable consequence yet. It earns promotion only if it
**predicts an altitude bug nobody found by other means**. Dressing it as
doctrine now would be the synthesis-spiral the session was explicitly steered
away from. Recorded so it isn't re-derived; not proposed for canon.

---

## 3. RECEIPTS OWED (honesty + anti-fork — not new ideas)

### 3a. MTPPS ↔ canon convergence — cross-link, do not fork

`ndarray/.claude/plans/mtpps-markov-tile-pyramid-substrate-v1.md` (this session)
independently lands on the guid-canon findings — they must cross-reference so
the same insight doesn't divide the search surface under two ids:

- MTPPS `tile_perturb_paint` ≡ **deterministic phase** (exponent, location,
  phase, magnitude; only the magnitude envelope stored; helix `CurveRuler`).
- MTPPS Markov cascade ≡ **WH-on-VSA / two-algebra rule** (sign=XOR `vsa_bind`,
  magnitude=`vsa_bundle` NEVER `MergeMode::Xor`). Reached independently this
  session ⇒ corroboration, not novelty; the canon statement is authoritative.
- `AmxBf16Grid = BlockedGrid<u16,16,16>` ≡ the transition-tile-as-AMX-tile shape
  — with the corrected caveat: one **build-time VNNI repack** (K=32, B
  pair-interleaved), zero per-query adaptation; "same shape" holds at the 16×16
  f32 accumulator, not the operands.
- MTPPS octree gate references HILBERT-L4: **already destaled** this session
  (commit `3927969e`) — addressing is green; only the 2-D `BlockedGrid` *storage*
  container remains deferred.

### 3b. NodeGuid authority — correction I owe

This session's `MedCare-rs/docs/NORTHSTAR_VISION.md` and
`cesium/docs/GENERIC_RENDERER_CORNERSTONE.md` cite the Rust `NodeGuid` struct as
the GUID source-of-truth. The canon (`OGAR/CLAUDE.md`, #50) is explicit that the
direction is the **reverse**: wrappers are *audited against* the canon
group-by-group, never the reverse. I had authority backwards — same altitude
error as §1 (struct = value read as canon = key). **Fix owed:** those two docs
should cite `OGAR/CLAUDE.md` as the GUID canon and mark the NodeGuid mapping
"audited against canon, Phase B (groups 3–4 → HIP/TWIG) pending," not as the
definition.

---

## 4. DESIGN NOTE — quorum-k scales with altitude (downstream of `contract::quorum`)

When the `contract::quorum` scaffold (#411 `todo!()`) becomes real: a correction
that touches a **spine** claim should require a higher quorum-k than one touching
**flesh**. That folds the §1 Altitude Gate into the quorum certificate as a
typed threshold rather than a review reflex. Buildable only after quorum lands;
recorded as a forward pointer, not a present deliverable.

---

## What the receiving session might do

- **Adopt §1** as a charter rule + an `EPIPHANIES.md` entry if it agrees (its
  surface to own; this handover is the proposal).
- **Cross-link §3a** from the canon's MTPPS-adjacent docs.
- **Accept §3b** as my correction to make on my own branch (I will).
- **§2 / §4** are forward notes — no action owed.

_End of handover._
