# Reference frame vs motion — calibrated data is the fixed point, not state

**READ BY:** `premise-auditor` (every signature table), any session touching
`ReferenceSet`, codebooks, bakes, coverage planes, `contract::alpha` /
`alpha_tunnel`, the cognitive-shader-driver fold loop, or any proposal to
"unify" two bitmaps, lanes or tables that look alike.

**Status:** DECISION (2026-10-03, `ISS-LXA-ALPHA-FIT`).
**BASIS:** `alpha.rs:7-16` (alpha is "a second table at the *same*
addresses as an already-baked SoA spine", "not a bake", "a record of where
attention went"); `deepnsm-v2-lexical-address-v1.md` §3.1; the blind
premise-gate test that reached the same split without being told.
**REVISIT WHEN** a value that this doc places on the reference side must
change within one reference version (by this decision it cannot; that is a
new version).

## The two categories

```
CALIBRATED REFERENCE (frame)              COGNITIVE SESSION (motion)

corpus, up to ~10⁹ observations
      │  compressed offline, once
      ▼
ReferenceSet vN                           same coordinates i
  ├── LexicalAddress[i]                         │
  ├── lemma / PoS                               ▼
  ├── frequency                            thought / rung r0
  ├── lemma_evidence                             │ alpha Δ
  └── coverage[i]  (defined / measured)          ▼
                                           thought / rung r1 …
```

| | reference frame | motion |
|---|---|---|
| answers | what did a large measurement of the world establish at coordinate *i*? | what is active, attended or different at coordinate *i*, now? |
| source | corpus measurement, compressed offline | one session, one thought, one rung |
| lifetime | immutable for one reference version | per cycle / rung |
| persistence | digested, reproduced byte for byte | discardable whole, or versioned as runtime state |
| writer | the bake | the runtime (`claim`) |
| examples | coverage, frequency, `lemma_evidence`, PoS, the Fisher-z LUT | alpha overlay, attended mask, per-rung lanes |

**The lever.** The expensive part (corpus → calibration → reference set)
runs once. Everything after it is cheap: reference set → lookup → fold →
fold → … A single fold is almost nothing; a million folds are still cheaper
than re-deriving what the reference already holds, and each of them stands
on the statistics of the original observations. The reference set is the
fixed point the folds lever against.

**What a large N buys and what it does not.** At very large N, frequency
approaches a stable empirical distribution for the measured population. It
is still not "the truth of the language": corpus choice, genre, period and
speaker population remain bias sources. That is exactly why the reference
set is **versioned and digested**: we know which world was measured.

## The rules

1. **Same coordinates, different category, different type.** A coverage
   plane and an alpha mask may share a bitmap layout. They must not share a
   type, a name, or a conversion. Representation is allowed to repeat;
   meaning is not.
2. **Motion never writes the frame.** Alpha and every other runtime overlay
   read the reference set and write beside it at the same coordinates. They
   never redefine, refine or "correct" a reference value. A changed
   reference value is a new reference version, made by a new bake.
3. **The frame is not a cache.** Calibrated data is not derived state to be
   invalidated, recomputed per session, or optimized away. Treating it as a
   cache is the failure this doc exists to stop.
4. **The code carries the category, not the session's memory.** Every
   reference-side type states in its doc comment that it is calibrated,
   immutable per version, derived from measurement, and not attention /
   alpha / runtime confidence. Every overlay type states that it is
   same-coordinate, session-local, and must not redefine reference data. A
   type-level guard (no conversion between the two) backs the prose.

## The tell

A local, reasonable-sounding simplification — *"these four bitmaps are all
per-coordinate state, let's make one overlay"* — is the signature of this
failure. Before merging two things that look alike, fill their signatures
(`.claude/agents/premise-auditor.md` § Step 1). If they differ on *answers,
lifetime, persistence or writer*, they stay apart however similar their
bytes are.
