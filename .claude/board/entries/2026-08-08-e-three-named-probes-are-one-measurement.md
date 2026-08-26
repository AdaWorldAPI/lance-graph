## 2026-08-08 — E-THREE-NAMED-PROBES-ARE-ONE-MEASUREMENT — F-1, helix's unrun fidelity gate, and the weather-encoder question are the SAME probe

**Status:** FINDING `[G]` on the identity (three docs, one measurement shape), `[H]` on the outcome (unrun). Surfaced by the 5-agent recon behind `.claude/plans/weather-substrate-poc-v1.md`.

Three probes have been carried independently, in three places, each named as blocking:

1. **F-1 (codebook fidelity)** — `substrate-unification-thesis.md` §4.1: does hierarchical-4⁴ (a byte's nibbles = its centroid's ancestry) preserve rank-distance vs flat k-means-256? Named in the OGAR canon, **un-run**. Its KILL: if it fails, "prefix-is-ancestry" degrades from *code* to *router* and a large fraction of the `[H]`/`[S]` map collapses at once.
2. **helix's own gate** — `crates/helix/KNOWLEDGE.md:338-343`: fidelity vs certified `Base17Fz` is **CONJECTURE, probe NOT RUN**, gate ≥0.9980 Pearson.
3. **The weather-encoder question** — does a 48-bit `Signed360` preserve *synoptic* distance? (the gate the whole retrieval/analogue lane stands on).

**They are one measurement:** *does a quantized hierarchical code rank like the continuous quantity it encodes?* Only the dataset differs. And `bgz17` already ships **both arms** — `Palette` (flat k-means 256, `palette.rs:112`) and `HierarchicalPalette` (16 coarse × 16 leaves with `coarse_is_ancestor_of`, `palette.rs:214,451-493`) — plus measured anchors on other data (`PaletteResolution`: Full256 ρ=0.992 / Half128 ρ=0.965 / Quarter64 ρ=0.738, `palette.rs:543-582`). So the *instrument* has been sitting in-tree, built, the entire time; what was missing was a dataset whose ground truth the workspace did not author.

**Why weather is the right dataset for it, not merely a convenient one.** Every prior corroboration in this arc has been internal — four sessions converging, probes designed by the same people holding the thesis (thesis §5 warns explicitly that convergence can be shared blind spots as easily as shared truth). Weather supplies a **physical** ground-truth distance (field RMSE) that no one here chose, at a scale (10⁶ points) that exercises the hierarchy rather than a toy. Running the bake-off on weather answers all three questions with one number — and a **negative is thesis-relevant, not weather-relevant**: it would demote prefix-ancestry workspace-wide, which is precisely why it must be run before more is built on top.

**Consequence for planning:** do not schedule these as three separate probes in three repos. One bake-off (`D-WX-5`), arms C vs D carrying F-1, arms A/B carrying helix's gate, all under Jirak-bounded significance (`I-NOISE-FLOOR-JIRAK` — grid fields are weakly dependent by construction; classical Berry-Esseen is wrong here). Mandatory anti-vacuity per `E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1`: include a shuffled-codebook encoder that **must** fail — a bake-off where every arm passes has measured nothing.

