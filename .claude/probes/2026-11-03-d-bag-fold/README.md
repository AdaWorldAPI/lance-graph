# D-BAG-FOLD evidence: multi-hop existence from resident presence

**Status:** MEASURED, lab only. No production code changed. mask-risc, Quack and
the xlog cube are consumed unchanged.

**Source:** lance-graph `514948203` (merge of #1936, = `origin/main` when measured).
ndarray: the sibling checkout at `../ndarray` relative to the repo root
(here a symlink to the AdaWorldAPI/ndarray checkout). Toolchain: `rustc 1.99.0`
(`ndarray 0.17.2` requires it; the repo default 1.97 refuses).

## Contents

| path | what | provenance |
|---|---|---|
| `original/COADDRESSED-FOLD.md` | first report + its addendum | session scratchpad, byte copy, sha256 `632f0f9d…677e` |
| `original/SIDE-BOUNDARY.md` | the `Cube::side()` boundary audit | session scratchpad, byte copy, sha256 `5789faf8…300c` |
| `original/cofold-main.rs` | the first probe, unchanged | sha256 `0dde1fb5…5854` |
| `original/cofold-Cargo.toml` | its manifest (absolute paths, scratchpad-only) | sha256 `d9085726…af0c` |
| `probe/src/bin/v1_prefix_vs_presence.rs` | the first probe, byte-identical to `original/cofold-main.rs`, built here with relative paths | — |
| `probe/src/bin/v2_witnessed_existential.rs` | this pass: the existential, eight cases, two arms, oracle | new |
| `REPORT.md` | conclusions | new |
| `ARM-L-DECISION-MEMO.md` | decision preparation for Arm L; the ruling in it is DRAFT / NOT IN FORCE | new |
| `original/ARM-L-DECISION-MEMO.first-draft.md` | the memo as first written, before the operator's revision requests | session scratchpad, byte copy, sha256 `5743b1ee…` |

The originals are copies of artifacts that existed in this session's scratchpad.
Nothing in `original/` was reconstructed.

## Run

```sh
cd .claude/probes/2026-11-03-d-bag-fold/probe
CARGO_PROFILE_DEV_DEBUG=0 cargo +1.99.0 run -q --bin v1_prefix_vs_presence
CARGO_PROFILE_DEV_DEBUG=0 cargo +1.99.0 run -q --bin v2_witnessed_existential   # exits 1 on any oracle mismatch
```

The probe has its own `[workspace]` table, so the root workspace never builds it.
Set `CARGO_TARGET_DIR` outside the repo; `target/` must not be committed.

## Expected output (v2, measured)

```
1 all present                                  got=Ok((true, 1)) oracle=(true, 1) OK
2a R deleted (prefix-only says true)           got=Ok((false, 0)) oracle=(false, 0) OK
2b S deleted (prefix-only says true)           got=Ok((false, 0)) oracle=(false, 0) OK
2c T deleted (prefix-only says false)          got=Ok((false, 0)) oracle=(false, 0) OK
3 three candidates, one valid                  got=Ok((true, 1)) oracle=(true, 1) OK
4 three candidates, none valid (prefix-only says true) got=Ok((false, 0)) oracle=(false, 0) OK
3b three candidates, two valid                 got=Ok((true, 2)) oracle=(true, 2) OK
5 permuted rows, checked: prepare = Some(OrderNotAttested)
5 permuted rows, UNCHECKED arm: Ok((false, 0)) vs oracle (true, 1)
5 sealing a permuted lane: Some(OrderNotAttested)
6 duplicate d1, seal: Some(DuplicateKey)
6 duplicate d1, forged witness: Ok((true, 2)) vs oracle (true, 1)
7 other class holds the only valid chain       got=Ok((false, 0)) oracle=(false, 0) OK
7 edge crossing the classid boundary: build = Some("refused (NonRefiningEdge)")
8 before change: Ok((true, 1))
8 after S(b1,c1) deleted, stale projection, checked: Some(StaleVersion)
8 stale projection, UNCHECKED: Ok((true, 1))
8 re-prepared at version 2                     got=Ok((false, 0)) oracle=(false, 0) OK
8 arm L under the stale witness: Some(StaleVersion)
8 arm L at version 2                           got=Ok((false, 0)) oracle=(false, 0) OK
0 mismatches against the oracle
```

Every checked case also asserts that the two arms agree. **Disable runs (red):**
removing arm L's `Semijoin` on `S_in`, or the one on `R_in`, trips the
arm-agreement assertion. The `UNCHECKED` lines are the witness-check disables:
each removed check produces a silent wrong answer (cases 5, 6, 8).

## Limitations

- Toy fixture: one class, 9–13 nodes, 8-bit tiers, depth fixed at 4.
- The witness is a probe-local mirror of `OrderedLaneWitness` plus a presence
  version. Production `SealedFacetLane` and `LanceVersion` were not exercised.
- Edge membership is stamped at the child row, and only one-tier refining edges
  are representable. A general graph is outside this layout.
- No timing: correctness only, so the no-ms rule is not engaged and nothing here
  is a performance claim.
