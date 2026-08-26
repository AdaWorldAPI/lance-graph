## 2026-08-11 — E-HELIX360-IS-THE-NORMALIZED-SUBSTRATE-NOT-A-BIT-BUDGET-1

**Status:** FINDING `[G]` for every code fact (file:line in the doc); the weather
mapping is `[H]`; the floor policy is `[S]` pending probes. **⊘ Two corrections:
"helix360" is this session's coinage — the symbol is `Signed360`, no `helix360`
exists in any repo `[G-absence]`; and the "2×24 = from AND to" reading below is
FALSE — one `Signed360` is already a complete full-sphere direction. See
`E-THE-DOCTRINE-DOC-EXISTED-AND-I-NEVER-READ-IT-1` (top of file).**

**The finding, in one line:** helix360 is a Fisher-2z-hydratable golden-spiral
projection — two 24-bit equal-area hemispheres glued at the equator (Poincaré
rim densification via `arctanh`), stored as the existing 6-byte content-blind
`ValueTenant::HelixResidue` lane (`U8×6 @ row_offset 112`) whose 2×24 reading IS
"wind coming from AND going to" — and palette256 is the same mechanism one rank
down (fisher **z** → u8 → 256×256 LUT = the cosine replacement). The z-form is
kept **normalized, not materialized**: variance stabilization puts unlike
variables on one scale, which is what makes cross-variable correlation a
legitimate LUT operation ("correlate what normally can't be correlated") and
what makes 8 bits sufficient per scalar. `hyperbolic_depth` being exposed but
never called in the encode path is the DESIGN, not an unwired seam.

**The one real open decision found by reading `quantize.rs`/`distance.rs`:**
`RollingFloor` is per-instance and `quantize` is linear over `[lo,hi]`, so
cross-variable code comparability is a calibration POLICY (shared canonical
z-floor vs per-variable floors vs hybrid) the weather ingest must set
explicitly, stamped via `floor_version` and proposed to align with Lance
dataset versions. Probes P1–P6 queued.

**The correction arc (operator-caught, five instances, one species):** an
invented 48-bit budget where `Signed360` already existed; a 12-byte `Pair48`
proposal for a lane already 2× by construction; "unwired 2z" where
not-materializing is the point; "scalar similarity" where the value is the
normalization; plus the earlier bearing-snapping/wind-speed measurement
inversions. Meta-rule: *consult-don't-guess applies to one's own prior
in-session statements* — they carry no evidentiary weight until checked
against the tree.

**Doc:** `.claude/knowledge/weather-normalized-substrate.md` (full detail:
projection, wire shapes incl. the #498 sign-partition regression, floor
mechanics + loss accounting, measured-evidence ledger with conditions, ingest
spec with the corrected ARCO-ERA5 object name, verification battery, probe
queue, correction ledger). Companion plan §0 gains ⊘ C3 (wrong Zarr object
name) in the same commit.

