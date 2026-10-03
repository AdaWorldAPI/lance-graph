# Known is not alpha; the premise gate found it blind (2026-10-03)

**Status:** MEASURED · DECISION recorded in `deepnsm-v2-lexical-address-v1.md` §3.1 and
`.claude/knowledge/reference-frame-vs-motion.md`.

- **The finding.** A 5+3 council on #1307 asked "which kind of alpha carries
  known/unknown?" and verified four options. "Known" is a property of the calibrated
  reference set (baked, digested, immutable per version, written by the bake); alpha is
  same-coordinate runtime motion (per cycle, discardable, written by `claim`,
  `alpha.rs:7-16`). Different signatures, so the question's premise was wrong.
- **The gate.** `.claude/agents/premise-auditor.md` builds a per-concept signature
  (answers, coordinates, lifetime, persistence, writer, cardinality; representation never
  decides identity) and runs four tests. Measured with a copy stripped of the worked
  example (no mention of alpha, #1307 or "known"):

| case | expected | verdict |
|---|---|---|
| #1307 §3.1 as the council left it | fire | **PREMISE-SPLIT**: (a) is the answer, typed apart from `AlphaMask`; (d) bundles an unrelated attention-recorder question; (b), (c) category errors |
| #1306 OQ-CML-1, picked as a clean control | stay silent | **PREMISE-SPLIT** — the control was not clean: "linked" covered build graph / binary residue / call path / public types, and plan:122 (`Refusal` carries `GraphError`) contradicted plan:142. Verified against `error.rs:43-47`, `Cargo.toml:25-53`, `lib.rs:43`; plan corrected |
| D-LXA-2 generator, Rust example vs Python script | stay silent | **PREMISE-SOUND** |

- **OPEN.** Three cases is a small sample; the gate's false-positive rate on real option
  sets is not measured. The clean-control run also noted that `genre_shapes.rs:18-21`
  marks `academic_20k.csv` "license: unverified, do not redistribute", which bears on
  committing a 20k-derived TSV (D-LXA-2); not yet checked.
