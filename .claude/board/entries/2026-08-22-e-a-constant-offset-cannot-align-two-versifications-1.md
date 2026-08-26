## 2026-08-22 — E-A-CONSTANT-OFFSET-CANNOT-ALIGN-TWO-VERSIFICATIONS-1 — the versification map's KJV side is exact, and 51 of its offsets address a verse that does not exist; the shape, not the values, is the defect

**Status:** FINDING (arithmetic on the map's own columns — no scoring
involved). **Confidence:** High for the 51; the head/tail split (22) reuses the
original detector's cheap prefix-anchor signal and is a diagnosis, not a proof.

**The map is right where it can be.** `rosetta-pd-bundle/versification_map.tsv`
declares `kjv_verse_count` for every (lane, book, chapter); it matches the
actual KJV lane for **3,567 / 3,567** rows (1,189 chapters × 3 lanes), with no
gaps, no extras, and a dense KJV lane (count == max verse number everywhere).
The offset census reproduces its report exactly (luther1545 36 /
elberfelder1905 3 / bkr 8).

**51 rows declare an offset that cannot be applied end-to-end:** 47 have
`kjv_verse_count + offset > lane_verse_count`; 4 have `1 + offset < 1`. The
dominant cluster is luther1545 Psalms carrying `offset=+1` on chapters where
**both lanes have identical verse counts** — self-contradictory: an endpoint
must fall off.

**Psalm 84, the report's own worked receipt, shows why.** Luther counts the
superscription as v1, so `+1` is correct for KJV v1–v10 — but the lane has
**dropped KJV v11** ("For the Lord God is a sun and shield"), so KJV v12
aligns at `+0`. The detection is linguistically right; the *carrier* cannot
hold the answer.

**The generalization:**

> An integer offset is position made portable by arithmetic. It cannot express
> either degenerate case an alignment actually needs — an **address with no
> witness**, or a **witness with no address** — so on any interior insertion or
> deletion it is forced to pick a number that is right for part of the chapter
> and wrong for the rest.

This is also why the map's measured payoff is thin (+21 verses, 0.14 pp over
the naive join): where an offset was found, it is partly wrong. The report's
caveat covers whole-*chapter* divergence and explicitly observed none; the
failure here is intra-chapter verse loss, which that model does not name.

**Operator ruling arising (2026-08-22):** *when referencing other versions you
can never use token position — always the verse ADDRESS, distinguished-name
style.* Alignment across lanes is a relation between addresses, including the
two degenerate forms; repairing individual offset rows keeps the shape that
cannot represent them.

Corroborating contrast from the same pass: the Composite Gospel Index carries
601 references as `Book.Chapter.Verse` ranges rather than offsets, is versified
against **RSV**, and **600 of 601 resolved against the KJV lane on first
contact** (the single failure is a typed book mismatch,
`hasReference="Matt.4.33-Mark.4.34"` under `isPartOf="#Mark"`). An address
ported across versifications; an offset did not.

**Cross-refs — same asset, distinct defect classes** (added by the 5+3 council so the
asset's finding history does not fragment across four ids): `E-VERSIFICATION-IS-PER-EDITION-NOT-PER-TRADITION-1` (:7487) and `E-A-MARGIN-IS-NOT-A-QUALITY-SCORE-1` (:5813) find *value* and *confidence-column* defects in this same map; this entry finds the **carrier shape** defect. `E-RCC-1-FOUR-LANES-ONE-KEY-1` (:7499) is the design the map serves.

Artifacts + full defect census: `.claude/handovers/2026-08-22-corpus-addressing-session-to-next.md`.

