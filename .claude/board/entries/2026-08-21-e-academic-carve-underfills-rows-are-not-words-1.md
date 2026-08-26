## 2026-08-21 — E-ACADEMIC-CARVE-UNDERFILLS-ROWS-ARE-NOT-WORDS-1 — 20,845 COCA rows are 18,559 distinct words, so the 80×256 academic carve fills 90.6% and basins 73..79 are empty

**Status:** FINDING (measured on the committed
`crates/deepnsm/word_frequency/academic_20k.csv`, produced while building the
deepnsm-v2 academic codebook for S3). **Confidence:** High — a count over a
committed file, reproducible with `csv.DictReader` and a `set`.

**The falsified claim** is in `plans/causal-rung-standing-wave-v1.md:465`:

> **Vocabulary 4096 → 20k academic:** **fits** the palette256² pair carve
> NATIVELY — 20480 = 80×256, `(basin, identity) = (id>>8, id&0xFF)`

It does not fit. It under-fills.

| | measured |
|---|---|
| rows in `academic_20k.csv` | 20,845 |
| **distinct surface forms** | **18,559** |
| duplicate rows (same word, different `Pos`) | **2,286 (11.0%)** |
| `PaletteVocab::ACADEMIC_20K` reserved slots | 20,480 |
| **carve fill** | **18,559 / 20,480 = 90.6%** |
| last id 18,558 lands at | basin **72**, slot 126 |
| **empty basins** | **73..79 (7 of 80)** |

**The inference that produced the error is arithmetic on the wrong noun.**
`plans/deepnsm-morton-comma-facet-v1.md:140` records "20,845 rows" and calls
them "the 20k most-frequent COCA-Academic words". 20,845 > 20,480, so the carve
looks like it fits with 365 to spare. But `PaletteVocab::from_frequency_ranked`
admits by **surface form**, deduplicating (`duplicates_keep_first_id` is its own
pinned test), and COCA lists a word once per part of speech. The same plan file
even notes the multiplicity two paragraphs later — *"The `Pos='v'` rows in
`academic_20k.csv` (thousands of verbs)"* — without drawing the consequence for
the carve. **The fact was present, the subtraction was not.**

**Why no test caught it, and this is the reusable half.** The crate's own
`academic_20k_carve_spans_80_basins` asserts exactly the property that fails on
real data — `pair("w20479") == Some((79, 255))`, i.e. basin 79 is reached — and
it passes, because its fixture is `(0..20480).map(|i| format!("w{i}"))`:
**20,480 synthetic words that are all-distinct by construction.** The real
input is 11% duplicates. A dedupe-sensitive property was tested on a
duplicate-free fixture, so the test could only ever confirm the arithmetic it
was written from.

This is the fixture-shape failure this board already carries in other clothes
(a single-band fixture cannot see a multi-band defect; a justified page cannot
exercise the ragged branch). Stated for this class: **when the code under test
deduplicates, filters, or otherwise reduces its input, a fixture with nothing to
reduce measures the reservation, not the fill.** The falsifier is one line —
give the fixture a duplicate and assert the admitted count drops.

**What is NOT claimed.** 90.6% fill is not by itself a defect: `RESERVE, DON'T
RECLAIM` is canon, and a partially-filled carve with `(basin, slot) = (id>>8,
id&0xFF)` is correct, addressable, and stable. Nothing needs to move. What must
change is the **claim**: the carve is under-filled by 1,921 slots, seven basins
are empty, and any design that assumed basin 79 is populated — or that reads
basin occupancy as a frequency signal — is reasoning from a number the data
never supported. Whether to fill the tail (a larger COCA slice, or admitting
`(word, Pos)` pairs rather than surface forms) is a deliberate decision with a
semantic cost either way, not a gap to be quietly padded.

> **⊕ CORRECTED SAME-DAY (2026-08-21, codex P1 on #975 + the operator's
> "wieso pyyaml — hattest du nicht für dismech-rs schon was für Rust
> gebaut").** The counts above came from a throwaway Python line-scanner and
> were **wrong**. Codex caught it structurally without needing the corpus:
> the module documents `pathophysiology[].downstream[]` as a SUBSET with
> **2,497** mediator-bearing edges, and I had written **2,489** corpus-wide —
> *a subset cannot exceed the whole*. Re-measured with the parser that
> already existed (`dismech_oracle_census`, dismech-rs, over
> `graph::build_causal_graph`), cross-checked against an independent pyyaml
> structural parse — identical on every figure:
>
> | quantity | line-scan (wrong) | structural (authoritative) |
> |---|---:|---:|
> | label-KNOWN with ≥1 mediator | 2,489 | **2,512** (63.1%) |
> | label-KNOWN naming nothing | 1,489 (37.4%) | **1,466** (36.9%) |
> | distinct mediator strings | 3,048 | **3,095** |
> | exact node references | 45 (1.5%) | **45** (1.5%) — unchanged |
> | groundable by label alone | 59.4% | **59.3%** (1,834) |
> | ungrounded prose | 40.6% | **40.7%** (1,261) |
> | oracle diseases | not measured | **549** |
> | `UNKNOWN_INT` that DO name mediators | **missed entirely** | **92** |
>
> There is no contradiction once the number is right: 2,512 corpus-wide vs
> 2,497 in the downstream subset, the 15 sitting in `influences_mechanisms`
> (115) and `sequelae` (19). **Every qualitative claim survives** — the
> mediators are prose, the oracle is ~⅔ of the label-KNOWN set, the
> label-only edges need a third bucket. The magnitudes moved by <1%.
>
> **Two lessons, and the second is the expensive one.** (1) A line-oriented
> YAML walk is a parser you did not write and cannot test; it fails silently
> on shapes you did not imagine. (2) **The repository already had the
> structural parser** — `model::parse_disorder_raw` is serde_yaml over a
> committed type and `graph::build_causal_graph` already carries
> `intermediate_mechanisms: Vec<String>` on the edge. The right answer was
> one `cargo run` away, is sub-second against pyyaml's two minutes, and is
> now committed and re-runnable instead of living in a `/tmp` script. *A
> measurement a committed parser can make must not be made by an ad-hoc
> script.* This is the same family as
> `E-ABBREVIATION-GREP-MANUFACTURED-AN-ABSENCE-1` two entries up — reaching
> for an improvised tool over the one the repo already ships — twice in one
> session, which makes it the session's dominant failure mode rather than an
> incident.
>
> **A third finding neither pass had:** 92 `INDIRECT_UNKNOWN_INTERMEDIATES`
> edges DO name mediators. The source contradicts its own label. They are
> neither oracle nor restraint control, and left in the control they read as
> hallucinated closure by the benchmark's own definition.

**Artifact.** The derived codebook was uploaded round-trip-verified to S3 at
`lance-graph/codebooks/deepnsm-v2-academic-coca-v1/` (TSV + source CSV +
manifest), with the shortfall stated in the file's own header so a consumer
cannot pin it without reading it. It reports 18,559 entries; it does not pad.

