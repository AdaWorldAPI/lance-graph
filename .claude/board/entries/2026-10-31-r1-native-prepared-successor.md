# R1 successor: native prepared plans over CompiledProgram

**Date 2026-10-31 · Status MEASURED (lab) · Branch `ccr-85acaff0-oszq40-r1b` (successor to #1892)**

`binding_native::prepare_native` runs a `binding::Prepared` plan once through `FoldDialect`, keeps each fold's program as a mask-risc `CompiledProgram` in step order, and records the output layout. `run_native` checks guards, route and permit exactly as `run` does, then runs the programs with `execute_compiled` and fills a fixed-size receipt. No interpreter runs, nothing is allocated, and there is no per-run `BTreeMap`. Plans that compiling once would freeze are refused by name:
- `FoldResultFeedsFold`: a fold fed by an earlier fold's result;
- `ScalarAfterFold`: a scalar computed after a fold;
- `DataScalarNotPrepared`: a fold-free scalar read from the planes;
- `TooManyOutputs`;
- `PlanesChanged`.

`ExecPermit` stays a lab permission, not a production authorization.

Benchmark (`r1_vs_prebound`, `MPF_REPS=31`, 4 cores). Median ns per call from the clean run of three; the other two were disturbed by VM load, one doubling several arms. Every timed call is checked against the oracle.

| rows | mask-risc prepared | R2IL prepared | R1 run | R1 native |
|---:|---:|---:|---:|---:|
| 64 | 105.5 | 107.2 | 405.8 | 164.2 |
| 512 | 204.0 | 207.0 | 499.1 | 256.0 |
| 4096 | 1086.9 | 1089.1 | 1435.8 | 1139.7 |
| 65536 | 18670.8 | 18812.1 | 19039.0 | 18726.3 |

Allocations per call: R1 native 0, R1 run 5.

R1 native sits about 50–60 ns above prepared mask-risc at every size, so the gap is constant and does not grow with rows. The guard check alone measures about 30 ns (prebound vs prebound+guard), and route lookup and the permit scan are the rest. This split is a working model, not separately timed.

OPEN:
- plans that compute a scalar from a fold result (`ORDER_A`, `ORDER_B`, `SUB_*`) are refused, not executed natively;
- the permit check is a linear scan.
