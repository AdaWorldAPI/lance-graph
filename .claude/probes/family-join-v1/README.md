# PROBE-FAMILY-JOIN-1 — the two `family` namings give OPPOSITE answers, and `from_be_bytes` is the plausible wrong one

**Re-runnable, zero external data.** Links both real types; nothing is reimplemented.

```sh
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  cargo run --manifest-path .claude/probes/family-join-v1/Cargo.toml
```

Standalone crate with its own empty `[workspace]` table (same posture as
`perturbation-sim` / `jc` / `sigker`), so it never enters the lance-graph workspace build.
It path-deps `lance-graph-contract` with `features = ["guid-v2-tail"]` (default OFF in that
crate) and `perturbation-sim` for the real `CascadeKey`.

## What it was built to answer

`ISS-FAMILY-IS-FOUR-WIDTHS-TWO-AT-OPPOSITE-ENDS` claimed that `family` denotes four different
things, and that the two sharing a `u16` width sit at **opposite ends of the key** —
`CascadeKey::family` **is** HEEL (root-most), while `NodeGuid`'s v2 `family` is the
second-finest field, after `leaf`. The issue's own closing condition: compute a shared-prefix
under both namings for the same entity and report whether they agree.

**Anti-vacuity, enforced in code** (`assert!` per pair): every non-control pair must differ
somewhere in bytes 4..14, or both sides return the same trivial answer and the comparison
proves nothing.

## Result 1 — the namings are not merely different, they INVERT

| pair | cascade | v2tail | |
|---|---|---|---|
| P1 same HHTL, different v2 family | **3** | **1** | disagree |
| P2 different HEEL, same v2 tail | **0** | **3** | disagree |
| P3 identical (control) | 3 | 3 | agree |
| P4 differ in both (control) | 0 | 0 | agree |

**P2 is the extreme case: 0 versus 3.** On one pair of entities, one naming reports *nothing
shared* and the other reports *everything shared*. Not a subtle divergence — a complete
inversion. P1 is the same hazard milder (3 vs 1).

The two controls are what make this a measurement rather than a broken comparison: the methods
agree **exactly** when they should (identical keys → 3/3; differ everywhere → 0/0). A
comparison that always disagreed would prove nothing.

## Result 2 — the byte-order trap, and `from_be_bytes` is the dangerous one

Nibble level of first divergence, 0..=32 (32 = identical):

| pair | `from_le` | `from_be` | recomposed (correct) |
|---|---|---|---|
| T1 differs in classid's **TOP** nibble | 24 | 6 | **0** |
| T2 differs in identity's **LAST** nibble | 3 | 29 | **31** |

- **Recomposed from decoded values: 0 and 31** — exactly as
  `E-THE-SEMIRING-IS-FREE-...-JOIN-IS-THE-SAME-XOR-1` predicted.
- **`from_le_bytes`: 24 and 3 — fully INVERTED.** A root-level difference reads as *nearly
  identical*; a leaf-level difference reads as *nearly maximally distant*.
- **`from_be_bytes`: 6 and 29 — the plausible wrong answer.** It gets the *direction* right
  (T1 < T2, so a naive monotonicity check passes) while being wrong at **every** point: 6
  instead of 0, 29 instead of 31. This is the nibble-order-reversed-within-each-field error,
  and it is the one that would actually ship, because it survives the obvious sanity check.

Both `from_be` values were hand-derived before running (T1: bytes 0-2 identical, byte 3 XOR
`0x80` → 24 leading zeros → level 6) and the measurement reproduced them, so the mechanism is
understood rather than merely observed.

## What this does NOT show

It does not show anyone has *written* the wrong join — `NodeGuid` has no join surface at all
(`ISS-NODEGUID-HAS-NO-JOIN-SURFACE`), which is why this probe had to supply the comparison. It
shows what the two namings *would* yield, which is the point: the hazard is live precisely
because the operation has not been written yet.

Nor does it indict the shipped API. `family_v2` is distinctly named and feature-gated, and its
own doc comment already says *"different name, different bytes — no silent semantic swap."*
The exposure is a reader or a design treating "family" as one concept across the two types.
