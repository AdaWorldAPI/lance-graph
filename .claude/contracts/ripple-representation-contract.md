# Contract Proposal: Representation Layers

## Purpose

This document proposes a stable boundary between coarse basin routing, family sharpening, contradiction preservation, local discrimination, and exact member resolution.

The point is not to claim the final codec.
The point is to stop asking one representation to do five different jobs.

## Layer contract

### HEEL

Role:
- coarse basin selection
- broad wake-up
- cheap routing identity

Likely substrate:
- `bgz17`
- `Base17`
- `compute_heel()`

Must not be asked to do:
- final family identity
- contradiction preservation
- exact member representation

### HIP

Role:
- family sharpening after basin routing

Likely substrate:
- `i16` first
- later `i32`
- palette around 16384 for first serious pass

Must preserve:
- family purity
- role-aware grouping

### BRANCH

Role:
- contradiction or polarity split inside family

Likely substrate:
- signed residual
- contradiction mask
- branch-local delta code

Must preserve:
- same-family opposite tendency
- inversion pressure
- semantic tension

### TWIG

Role:
- local neighborhood discrimination
- CLAM-scale micro-routing

Likely substrate:
- local prototypes
- neighborhood graph
- tile-local pressure descriptors

Must preserve:
- local geometry
- branch relevance
- near-neighbor discrimination

### LEAF

Role:
- exact member plus residual

Likely substrate:
- family identity plus exact delta

Must preserve:
- exactness without brute-force flattening

## Family split contract

At minimum, representation must distinguish:

- QK = compatibility family
- V = payload family
- Gate = modulation family
- UD = transform family

Rule:

No global codebook should span QK, V, Gate, and UD unless a benchmark explicitly demonstrates benefit.

## Search / sweep / bus contract

### Search field
- searchable terrain
- broad candidate space
- not itself explicit thought

### Sweep
- cheap local collapse operator
- `bundle` / `xor_bind` / equivalent
- emits lookup object or temporary gestalt

### Bus
- accountable structured execution object
- topology plus style plus contradiction transport

## Success criteria

A correct representation stack should improve at least one of:

- family purity
- contradiction preservation
- local neighborhood sharpness
- bus compilation quality
- host-model guidance quality

without collapsing the others.

## Open questions

- What exact data types should represent HIP and BRANCH first?
- What exact object does TWIG store?
- How should LEAF preserve exactness without exploding storage?
- What benchmark is the fairest first test of role-aware split codebooks?
