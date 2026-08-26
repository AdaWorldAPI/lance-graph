## 2026-08-11 — E-THE-REUSE-IS-THE-PROCESS-AND-IT-EXPOSED-A-FIT-PROBLEM-1

**Status:** FINDING `[G]` on the measurement (`crates/helix/tests/bearing_encode_paths.rs`,
N=65536); `[S]` on the API shape, which is an operator call.

**Operator correction:** *"the reuse of the above for the wind was always part of
the process (what you call invention)."* A prior entry
(`E-THE-DOCTRINE-DOC-EXISTED-AND-I-NEVER-READ-IT-1`) filed the missing
bearing-encode under *invention* and declined to build it. **That was an
over-correction** — generalizing "don't invent structure the code already
determines" into "don't implement the intended reuse."

**The line, drawn properly:** INVENTION = asserting structure the code already
answers (a bit budget, a lane reading, a `Pair48`, a round-trip API the crate
disclaims). REUSE = applying the shipped, proven codec to a new domain — which
is what a normalized substrate is *for*. **A missing entry point for a designed
reuse is a plumbing gap, not a design refusal.**

**And the reuse immediately earned its keep.** `encode_signed` derives all three
direction-bearing fields from `n` alone (`residue.rs:182-204`). The knowledge doc
`helix-cartesian-vs-fisher2z.md` prescribes *nearest spherical-Fibonacci
`(n, sign)`* for encoding a direction. **Measured, that prescription does not fit
weather:**

| horizontal bearing | nearest-`n` (prescribed) | direct `(polar, azimuth)` |
|---|---|---|
| 0° / 90° / 270° | 1.933° / 2.706° / 1.897° | **0.000°** each |
| mean, 24 bearing×elevation cases | **0.972°** | **0.097°** — 10× |

**Mechanism:** the golden spiral couples latitude and azimuth through ONE index.
Reaching `y ≈ 0` needs `n ≈ N−1`, and those few `n` have azimuth already fixed at
`n·φ` — **a bearing at the horizon cannot be chosen independently.** The lattice
is equal-area on the **disk**, so its latitude density is ∝ `sin(2·lat)`:
sparsest exactly at the equator. Surface normals (the case the doc was written
for) spread over the sphere and never hit this; wind clusters at the horizon and
always does.

**Rule extracted:** *a doctrine written for one domain is not automatically
right for the next one that reuses it.* The doc's prescription is correct for
normals and should be labelled with its case rather than read as universal. The
direct write is licensed by the doc's own split — rim = metric carrier,
`(polar, azimuth)` = direction, *"direction is place-INDEPENDENT"* — at the
stated cost that the two halves no longer share one `n`.

**Corrects:** `E-THE-DOCTRINE-DOC-EXISTED-AND-I-NEVER-READ-IT-1` (its "the real
gap… deliberately not built" framing). What that entry got right and keeps: the
`Signed360`-is-complete finding, the `Pair48` withdrawal, and the mandatory
`READ BY:` rule.

