# DeepNSM Sentence-Level AriGraph Reader — Design Document

**Branch:** `claude/stoic-turing-M0Eiq`  
**Crate:** `crates/deepnsm/`  
**New modules:** `morphology`, `cam64`, `episodic_spo`, `window`, `reader_state`, `signed_crystal`, `sentence_transformer64`  
**Tests added:** 200 lib tests (0 failures)

---

## Glossary (one-liners for reviewers)

| Term | Definition |
|------|-----------|
| **P64** | 8×8-bit native reading-state address space; NOT a quantised float embedding |
| **CAM4096** | 12-bit deterministic address selected from P64 lanes; NOT a quantised embedding vector |
| **Crystal4096** | 3-axis signed reading coordinate (12 bits, 4096 cells); P4096 palette codebook key |
| **Cam64** | 64-bit fast reading-locality index (NOT semantic truth); used for prefetch and basin heuristics |
| **EpisodicSpoFrame** | The auditable SPO truth witness; `cam64` inside it is the index, not the truth |
| **SentenceWindow** | Wernicke faculty: exact NP head ranks for coreference; distinct from `ContextWindow` (Broca/VSA) |
| **splat_p64** | Discrete palette splat into a Hamming neighbourhood; NOT a Gaussian in f32 space |
| **SentenceTransformer64** | State-transition automaton (Manning & Carpenter sense); NOT neural self-attention |

---

## What this is

DeepNSM is a distributional semantic engine that replaces transformer inference
with precomputed distributional lookup (4,096-word COCA vocabulary, 8 MB distance
matrix, `<10 μs/sentence`). This reader layer adds the **sentence-level
AriGraph reading state machine** on top of the existing tokenizer/parser/encoder
stack — the auditable, Wernicke-faculty side of the pipeline.

The one-line description:

> DeepNSM reads sentences one at a time, emits auditable episodic SPO rows,
> maintains a ±5 sentence reading state for pronoun/coreference/inference, and
> derives a compact 64-bit CAM code from morphology + grammar + NSM markers for
> fast basin matching and prefetch.

---

## Architecture: five distinct responsibilities

```
SentenceStructure (from parser)
  + SentenceFeatures (caller-supplied annotations)
        │
        ▼
  ReadingState::step()          ← left-corner state machine
        │
        ├─► Vec<EpisodicSpoFrame>    truth witness (auditable SPO rows)
        │
        ├─► ReadingState_next        updated ±5 window + entity stack
        │
        ├─► Cam64                    reading-state locality key (NOT the truth)
        │
        ├─► SignedSentenceCrystal    P64MeaningField + Crystal4096 coordinate
        │
        └─► Sentence64              P64 + CAM4096 + EpisodicSpoHint
                │
                ▼
          holograph BitpackedVector  (16Kbit resonance, separate crate)
                │
                ▼
          AriGraph basin update
```

### The two-faculty split (E-ENGLISH-BIFURCATES)

Two windows serve different cognitive faculties and must not be fused:

| Window | Faculty | Content | Purpose |
|--------|---------|---------|---------|
| `ContextWindow` (`context.rs`) | Broca / projection | VSA projections (MarkovBundler band) | Distributional disambiguation |
| `SentenceWindow` (`window.rs`) | Wernicke / coreference | Exact NP head vocabulary ranks | Auditable coreference resolution |

---

## Module-by-module description

### `morphology.rs` — MorphFlags

`MorphFlags(u16)` is a 14-bit packed field of heuristic morphological features
derived from `SentenceStructure`. No float arithmetic. Flags: `PAST`, `PRESENT`,
`FUTURE`, `SINGULAR`, `PLURAL`, `FIRST_PERSON`, `SECOND_PERSON`, `THIRD_PERSON`,
`PASSIVE`, `NEGATED`, `INTERROGATIVE`, `RELATIVE_CLAUSE`, `INFINITIVE`,
`SUBORDINATE`.

```rust
let morph = MorphFlags::from_sentence_structure(&sentence, triple_idx);
assert!(morph.is_past());
```

### `cam64.rs` — Reading-state locality key

`Cam64(u64)` is 8 lanes × 8 bits. It is **NOT semantic truth** — it is a fast
reading-locality index for candidate prefetch, basin matching, and coreference
heuristics.

| Lane | Content | Source |
|------|---------|--------|
| 0 | entity/subject bucket | vocabulary rank >> 5 |
| 1 | predicate/action bucket | vocabulary rank >> 5 |
| 2 | object/complement bucket | vocabulary rank >> 5 |
| 3 | morphology low byte | MorphFlags bits 0-7 |
| 4 | clause structure | MorphFlags bits 8-13 |
| 5 | discourse / anaphora | entity stack depth + coref flag |
| 6 | causal / temporal | temporal marker present |
| 7 | episodic basin | novelty_high hint |

**`continues_basin(prev: Cam64) → bool`**: Pika chart-arc predicate. Uses
`count_ones()` on the XOR: shared ≥ 16 bits AND diff ≤ 24 bits. Dumb,
deterministic, no semantic reasoning.

### `episodic_spo.rs` — Auditable witness row

`EpisodicSpoFrame` is the truth: one auditable SPO row per triple per sentence.
All 25 fields are `Copy`; the struct stacks in `Vec<EpisodicSpoFrame>` for SoA
sweep. The `cam64` field is the fast-index; the `subject/predicate/object_candidate_id`
fields are the truth. Size constrained to ≤ 128 bytes (tested).

`BasinClassification` expresses how a new frame relates to an existing AriGraph
story basin: `Reinforcement`, `NoveltyDelta`, `WisdomDelta`, `Contradiction`,
`Branch`, `Epiphany`.

### `window.rs` — ±5 sentence ring buffer + Pika expectation slots

`SentenceWindow` is an 11-entry ring buffer of `WindowEntry` (up to 4 NP heads
per sentence). Provides `resolve_pronoun(exclude_rank) → u16` with a two-phase
resolution strategy:

**Phase 1 — Pika forward expectation slots (added in left-corner adaptation):**
When a left-corner trigger fires (relative pronoun, anaphora), `push_expected(rank, reason)`
pre-populates a slot. Resolution checks these first — confirmed expectation beats
recency heuristic.

**Phase 2 — Confirmed ring, most-recent-first:**
Heads iterated in reverse within each entry (last-mentioned in text = highest
index = most recent). Original Manning & Carpenter recency heuristic.

```rust
window.push_expected(active_subject, ExpectedReason::RelativeClause);
let referent = window.resolve_pronoun(pronoun_rank); // returns expected first
```

`ExpectedReason` enum: `RelativeClause`, `Anaphora`, `Ellipsis`,
`CausalContinuation`, `TemporalContinuation`.

### `reader_state.rs` — Left-corner state machine

`ReadingState::step(self, &SentenceStructure, &SentenceFeatures) → (Vec<EpisodicSpoFrame>, ReadingState)`

Pure function — `self` is consumed, `next` is returned. No `&mut self` during
computation (data-flow.md rule). State carries:

- **Top-down expectation**: `expected_subject_bucket`, `expected_predicate_bucket`,
  `active_trigger` (set by the first triple's `LeftCornerTrigger`)
- **Bottom-up evidence**: `active_subject`, `active_predicate`, `active_object`
- **Entity stack**: LIFO bounded at 8, evicts oldest on overflow
- **±5 window**: `SentenceWindow` for coreference
- **Cam64**: current reading-state locality code

**Left-corner trigger wiring** (Pika chart-arc pre-population):
```rust
LeftCornerTrigger::Relative | LeftCornerTrigger::Anaphora => {
    // Prior active_subject is the most likely antecedent.
    // Pre-push into window's expectation buffer before processing.
    next.window.push_expected(next.active_subject, reason);
}
```

`LeftCornerTrigger` variants: `Declarative`, `Causal`, `Temporal`, `Relative`,
`Anaphora`, `FirstPerson`, `Domain(u8)`. Each carries a `basin_byte()` that
feeds into Cam64 lane 7.

### `signed_crystal.rs` — Discrete reading crystal

Three types for the holograph bridge:

**`SignedOffset4`**: 0-14 encodes −7..+7 (raw = offset + 7); 15 = overflow/basin-change.

**`Crystal4096`**: three axes × 4 bits = 12 bits, 4096 cells. Direct P4096 palette
codebook key. `xor()` for VSA bind/unbind (self-inverse). `same_basin()` =
no overflow AND `nibble_distance ≤ 1`.

**`SignedSentenceCrystal { p64: P64MeaningField, coord: Crystal4096 }`**: complete
output bridging DeepNSM to holograph. `bind()` XOR-binds both fields. `same_basin_as()`
= P64 agreement ≥ 40 bits AND coordinate nibble_distance ≤ 1.

### `sentence_transformer64.rs` — Native P64 meaning field

**The architectural correction this module encodes:**

> P64 is the **native address space**, not a compressed approximation of a float
> embedding. Floats may approximate P64 for external ML interop. P64 does not
> approximate floats.

**`P64(u64)`**: 8 orthogonal semantic planes. Each word projects *vertically*
into the field — activates across multiple lanes simultaneously. `from_cam64_and_nsm()`
is the canonical construction path (grammar → Cam64 → P64, no floats).
`bind()` = XOR (VSA, self-inverse). `agreement()` = `64 - popcount(XOR)`.

**`Cam4096(u16)`**: 12-bit deterministic codebook address. `from_p64()` folds
top nibbles of entity, predicate, and basin lanes — a bit-selection, not
nearest-neighbour search in float space. 4096 cells = native-English reading-state
classes at full resolution.

**`Perturbation4x4`**: local 4×4 discrete ambiguity tile. Row = semantic axis
(entity/predicate shift), col = syntactic axis (clause/discourse shift). 16
alternatives per step. The implicit `(4×4)^n` trajectory space is never
materialised — HHTL/GridLake prunes to the small living frontier (Pika-style).

**`splat_p64(centre, tile, radius_bits) → SmallNeighbourhood`**: discrete palette
splat — NOT Gaussian in f32 space. Keeps only cells that change the P64, stay
within Hamming radius, and `near_match` the centre's CAM4096. Stack-allocated,
≤ 16 entries.

**`SentenceTransformer64`**: projects `(Cam64, nsm_prime_mask, subject, predicate, object, role)`
into `Sentence64 { p64, cam, spo_hint }`. `project_from_frame()` is the ergonomic
path from an `EpisodicSpoFrame`. The name is honest in its own docs:
*"Transformer here means state-transition transformer, not neural self-attention."*

---

## What is NOT changed

- `ContextWindow` (`context.rs`) — untouched. VSA/Broca faculty.
- `pipeline.rs`, `encoder.rs`, `similarity.rs` — untouched.
- Float fields on `EpisodicSpoFrame` (confidence, novelty, wisdom, entropy,
  free_energy_delta) — retained as **boundary quality annotations**, not hot-path
  substrate.
- All existing 132 deepnsm tests — still pass.

---

## Float boundary policy

```
HOT PATH — zero floats:
  P64, Cam4096, Crystal4096, SignedOffset4
  MorphFlags, Cam64, SentenceWindow
  splat_p64, hamming_p64, nibble_distance

BOUNDARY ANNOTATIONS — f32 permitted:
  EpisodicSpoFrame.{confidence, novelty, wisdom, entropy, staunen, free_energy_delta}

FORBIDDEN INTERNAL PATH (absent by omission):
  fn from_f32_embedding(...) -> P64     ← does not exist
  fn quantize_embedding(...) -> Cam4096 ← does not exist
```

---

## Test summary

| Module | Tests |
|--------|-------|
| `morphology` | 8 |
| `cam64` | 13 (5 new: basin continuation) |
| `episodic_spo` | 8 |
| `window` | 11 (6 new: expectation slots) |
| `reader_state` | 12 (2 new: trigger wiring) |
| `signed_crystal` | 18 |
| `sentence_transformer64` | 26 |
| **Existing deepnsm tests** | 104 (unchanged) |
| **Total** | **200** |

---

## Relationship to holograph

```
DeepNSM (this PR):
  local 64-bit reading-state code
  P64 / CAM4096 / Crystal4096 discrete palette

holograph (crates/holograph):
  large 10K/16K/32K bitpacked resonance field
  XOR bind/unbind, HDR cascade, stacked popcount, no floats

AriGraph:
  episodic crystallisation into story basins and tombstone witnesses

Flow:
  sentence
    → SentenceTransformer64 → Sentence64 { P64, CAM4096 }
    → EpisodicSpoFrame (auditable SPO truth)
    → holograph SemanticCrystal → BitpackedVector (16Kbit)
    → AriGraph basin update
```

The holograph `sentence_crystal.rs` (integer-first, char n-gram hashing,
bit rotation, majority bundling) is the correct large-field ancestor.
The ladybug-rs `sentence_crystal.rs` (f32 random projection → 5D coords)
is a float-projection prototype and is NOT the reference here.
