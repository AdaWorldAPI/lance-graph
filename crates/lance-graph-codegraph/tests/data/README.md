# lance-graph-codegraph test data

Store artifacts (rows v2) written by ruff `crates/ruff_charon` (`ruff_charon import <dir> <file.llbc> <crate>`, ruff PR #132, merged `ed8fd4194f`) from the LLBC in ruff `crates/ruff_charon/tests/data`. That LLBC comes from the AdaWorldAPI Charon fork `c69e231` over `crates/sag-ra-probe/fixtures/codegraph`.

- `rev1/`: `cg_lib` and `cg_app` at rev1.
- `rev2/`: both at rev2 (deletes `triple` and `area_dyn`, drops the `area_dyn` call).
- `mixed/`: `cg_lib` at rev2 and `cg_app` at rev1, so `cg_app::run → cg_lib::area_dyn` dangles (`ruff_charon dangling` reports exactly that edge).
- `fnref/`: the one-file crate `fnref` (a function used only as a pointer, an unused private function, a call through a pointer argument).
