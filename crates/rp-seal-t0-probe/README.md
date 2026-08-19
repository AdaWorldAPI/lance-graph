# rp-seal-t0-probe — the X-C2-1 injection harness

RP-SEAL Tier 0 (charter `.claude/plans/erasure-seals-compaction-research-v1.md`,
consolidation `docs/lotus/RP-SEAL-CONSOLIDATION-PASS1.md` M11; C2 §EXPERIMENT).
Fault injection over a synthetic cycle; **ground truth is the injection
record**; a false accept is an affected chunk the auditor did not flag. No
wall-clock quantity anywhere (T0.3: version distance is the epistemic metric;
time is economics).

Verify: `cargo test --manifest-path crates/rp-seal-t0-probe/Cargo.toml`
Report: `cargo run --manifest-path crates/rp-seal-t0-probe/Cargo.toml --example xc21_report`

## Anti-vacuity gate (charter: reproduce the known in-tree false accepts FIRST)

| control | target | status (2026-08-19) |
|---|---|---|
| (a) metadata blind zone | REAL `ndarray::hpc::merkle_tree` | **reproduced** — words 50/70/200 corrupted, `hamming == 0`; zone is WIDER than C2 recorded (48..56, 64..96, 112..256 all unhashed) |
| (b) `verify_ecc` 100%-accept | firefly (wip) | **unreachable by construction** — the `wip` feature fails to compile (104 errors); fault class proven on the fold algebra instead |
| (c) XOR-fold permutation | container_bs (wip) | same as (b): paired-flip cancellation + permutation invariance proven directly |
| (d) unbound digest wrong-slot | REAL `ndarray::hpc::seal::Plane` | **reproduced** — identical content at another slot verifies `Wisdom` against the stored root |

## Measured matrix (2026-08-19, n = 4096, multiplicity 1; I7–I9 as stated)

S1U (locus-unbound content digest — the shipped `seal.rs` shape) false-accepts
**exactly** the substitution class: I4 wrong-slot, I5 stale, I6 duplicate — 1/1
each — and nothing else. S6 (locus+version-bound): **zero false accepts on all
of I1–I9**, zero false alarms. Null control: 10⁶ distinct clean chunks per
scheme, 0 spurious flags. Full-size pass (65,536 × 512 B = 32 MiB, I9 stride
4096): 16 affected, 0 FA, 0 alarms. This is C2's truth table, confirmed
empirically at the hash tier.

## What plugs in next (X-C2-3)

S2 flat RS / S3 row-col P+Q / S4 product / S5 cascade implement the same
`Scheme` trait with a repair verdict layered on; the kill condition stands:
any scheme with a non-zero false-accept rate on I1–I3 at multiplicity 1 is
deleted.
