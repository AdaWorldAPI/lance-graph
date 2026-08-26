## 2026-08-21 — E-HHTL-IS-MINTED-IN-THE-ARTIFACT-NOBODY-CITES-1 — "zero on every baked row in both production bakes" is precise about the two it names and silent about the third, where the five OBO namespaces are 100% minted

**Status:** FINDING (measured directly on the pinned `.soa` bytes, SHA-256
verified against `MedCare-rs/data/config/bakes.tsv`). **Confidence:** High —
every number is a count over 512-byte rows, tiers read at bytes 4..10.

The board headline (`INTEGRATION_PLANS.md` ARC-B entry) and
`EPIPHANIES.md:899` both state HHTL is **"zero on every baked row in both
production bakes"**, citing `ogar-obo` (68,797 rows) and MedCare's
`join-map.md` (68,797 rows). Both citations are correct. **Both describe the
same artifact set of two.** A third pinned artifact exists:

| artifact | rows | HHTL != 0 |
|---|---:|---:|
| `obo-core.soa` | 68,797 | **0 (0.00%)** |
| `spine.soa` | 7,641 | **0 (0.00%)** |
| `all-lanes.soa` | 770,360 | **164,031 (21.29%)** |

Per lane in `all-lanes.soa`: **MONDO `0x0301` 32,095 rows 100% · HPO `0x0302`
19,836 100% · UBERON `0x0303` 14,975 100% · PATO `0x0304` 1,887 100% ·
ICD-10-GM `0x030F` 16,905 100% · OMIM `0x0319` 18,712 100% · OPS `0x0311`
98.9% · Orphanet `0x0317` 67.6%**; LOINC / CUI / RxNorm / FMA ~0%.

**Why this matters and is not a footnote:** the five 100%-minted namespaces are
exactly the ones a DisMech overlay grounds against. The generalized claim
("HHTL is dormant, the gap is mint + read") licenses a session to skip HHTL
entirely; the measured claim says HHTL is available **if a reader names
`all-lanes.soa`** and unavailable **if it names `obo-core.soa`**. So *which
artifact a consumer reads* is a first-class design decision, not a detail — and
a ladder level claiming "+ HHTL topology" must state its artifact or it is
measuring nothing.

**The generalization error is the transferable part.** Two independent
citations agreeing is evidence about the *rows they counted*, not about the
artifact set. Two sources counting the same 68,797 rows is ONE measurement
reported twice. Before promoting "on every row" from "on these rows", enumerate
the artifacts, not the citations.

**⊘ CORRECTION 2026-08-21 (same day, operator-prompted) — the entry above
measured ONE of TWO readings.** Operator: *"Obo HHTL ist meines Wissens mit
zipper bereits indirekt hydriert."* Correct. The census above counted the
**cascade tiers** (bytes 4..10); `rails::HhtlMode::of_row` PREFERS the
**RailHead** reading and uses Cascade only as the fallback. Measured on the
Zipper rail registers (`rails.rs:130-147`): MONDO **32,094/32,095**, HPO
**19,835/19,836**, UBERON **14,973/14,975** (plus 8,525 `part_of`, the only
lane with mereology), PATO 1,886/1,887 — median logical-DN depths 6/7/8/5, and
264 rows deep enough to use the continuation slab. So the OBO hierarchy IS
hydrated, indirectly, exactly as stated. `obo-core.soa`/`spine.soa` remain zero
on BOTH readings. New in this correction: cascade and rail are **independent** —
Orphanet (14,063 cascade) and OMIM (18,712) carry **zero** rail DN, so a
prefix-containment consumer silently gets depth 0 there. **The lesson compounds
the entry's own:** it is not enough to enumerate the artifacts — a claim about
a field must also name which READING of it was counted, when the accessor
picks between two registers. Full table: plan §8a ⊘ correction.

Cross-ref: `.claude/plans/dismech-causality-v3-v1.md` §8a; ARC-B
`docs/architecture/ARC-B-OWNERSHIP-AND-ADDRESSING-REASSESSMENT.md:23` (regraded
in place: its conclusion holds for `obo-core`/`spine`, needs the `all-lanes`
qualifier); `EPIPHANIES.md:899`.

