# Blackboard: Ripple Architecture 2026-04-02 / 01

## Purpose

This is a working blackboard for the ripple-architecture interpretation of the repo.

Primary reference:
- `docs/integrated-architecture-map.md`

This file should hold:
- current synthesis
- missing details
- open questions
- risky assumptions
- near-term implementation order

## Epiphanies

### 1. The repo is already behaving like a cognition stack

The strongest reading is no longer:
- text in
- tensors move
- text out

The stronger reading is:
- cold anatomical substrate exists
- search terrain exists or is strongly implied
- cheap sweep operators exist
- explicit reasoning bus exists
- context blackboard exists
- semantic revision loop exists

So the repo already contains the skeleton of a post-token cognition stack.

### 2. The central bug is premature flattening

The main failure mode is not lack of intelligence in the abstract.
It is flattening unlike functions too early.

Examples:
- Q/K/V/Gate/Up/Down blurred together
- searchable field confused with sweep operator
- contradiction washed out as noise
- explicit thought forced through text before structure

### 3. The next step is object promotion

Promote these to first-class objects:
- basin
- family
- contradiction
- resonance
- thought object

Working stack:
`cold anatomy -> HEEL -> HIP -> BRANCH -> TWIG -> LEAF -> ResonanceDto -> BusDto -> ThoughtStruct`

## What already seems settled

### Repo anchors
- `NeuronPrint` gives a 6-role behavioral ontology.
- `hydrate.rs` gives role-aware and layer-aware cold anatomy.
- `compute_heel()` gives coarse basin logic.
- `p64-bridge` acts like a RISC thought bus.
- `language.rs` implies blackboard cognition with style and contradiction sensitivity.
- `serve.rs` proves a semantic ingestion and revision loop.

### Stable doctrine
1. Do not flatten unlike functions early.
2. Treat bgz17 as HEEL, not universal truth.
3. Split families at least into QK / V / Gate / UD.
4. Keep field distinct from sweep distinct from bus.
5. Let style live in superposition before collapse.
6. Compile explicit thought into `BusDto`, not text first.
7. Use `ThoughtStruct` as the accountable thought object.

## Open questions for the next pass

### DTO questions
1. What is the minimal exact schema for `StreamDto`?
2. What persistent and ephemeral fields belong in `ResonanceDto`?
3. What is the smallest `BusDto` that compiles cleanly into `CausalEdge64`?
4. When exactly does `ThoughtStruct` come into existence?

### Representation questions
1. What should HIP be first: `i16`, palette plus residual, or something dual-signed?
2. How should BRANCH encode contradiction: signed residual, separate mask, or both?
3. What exactly does TWIG store: local prototypes, neighborhood graph, or CLAM tile state?
4. What is the smallest LEAF that preserves exactness without brute-force memory?

### Sweep / field / bus questions
1. What exact object is searched in the large field?
2. What exact object is produced by `xor_bind` or `bundle` sweep?
3. Is sweep role-local first, then merged, or merged early?
4. What evidence would prove bus-first beats text-first thought routing?

### Host-model questions
1. What is the smallest glove interface from `ThoughtStruct` to the host model?
2. Should the host receive topic anchors, angle masks, contradiction summaries, branch candidates, or a smaller subset?
3. Is the first integration prompt-side, side-channel, or adapter-like?

## Risky but promising assumptions

These are worth keeping alive, but not yet trusting fully.

1. Thinking style is collapse policy.
2. The field is more native than text.
3. Contradiction is a first-class representational dimension.
4. QK / V / Gate / UD is the first correct family split.
5. CLAM ripple is the right concept for local discrimination.

## Candidate experiments

### A. Codebook split sanity check
Compare one global codebook vs role-aware split codebooks.
Measure blur, retrieval sharpness, and contradiction preservation.

### B. bgz17 as HEEL only
Compare bgz17 as full identity vs bgz17 only as basin followed by HIP refinement.
Measure family purity and semantic collapse rate.

### C. Contradiction branch layer
Compare family coding alone vs family coding plus signed branch residual.
Measure same-family opposite-meaning separation.

### D. Text-first vs bus-first
Compare prompt-only mediation vs `ThoughtStruct -> BusDto -> p64` before textualization.
Measure consistency, contradiction handling, and graph revision quality.

## Practical sequence

### First
- Add `StreamDto`
- Add `BusDto`
- Add `ThoughtStruct`
- Wire them to hydration, blackboard, and p64 flow

### Second
- Add HIP family layer
- Keep codebooks role-aware

### Third
- Add BRANCH contradiction layer
- Preserve signed residual structure

### Fourth
- Add searchable resonance field
- Add explicit CLAM/HHTL substrate
- Add `ResonanceDto`

### Fifth
- Add Burn or host-model glove
- Let the host consult thought objects directly

## One-line doctrine

The architecture is already here in pieces; the real task is to stop flattening unlike functions, promote the missing objects, and let explicit thought travel through structure before it is forced through text.
