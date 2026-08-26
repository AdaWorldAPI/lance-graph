## 2026-08-10 — E-CLAUDE-MD-KEY-DEPENDENCIES-WENT-STALE-AND-PROPAGATED-A-WRONG-PIN-INTO-A-PLAN-1

**Status:** FINDING `[G]` (verified against `Cargo.lock` + every crate manifest).

`CLAUDE.md` § Key Dependencies — **the mandatory first read for every session** —
carried four stale pins, dated *"Verified against Cargo.lock 2026-06-14"*, which
pre-dates the lance-9 sweep (`b2b08b07`, 2026-08-05):

| CLAUDE.md said | tree actually has |
|---|---|
| `lance = "=7.0.0"` | **`=9.0.0`** |
| `lance-linalg = "=7.0.0"` | **`=9.0.0`** |
| `lancedb = "=0.30.0"` | **`=0.33.0`** |
| `datafusion = "53"` | **`54`** |

**The propagation is the finding, not the staleness.** The wrong `datafusion 53`
travelled: stale `CLAUDE.md` → restated in-session → written into
`weather-substrate-poc-v2.md` §6 (PR #915, merged) → *and* mischaracterized there,
because `datafusion 54.1.0` in `Cargo.lock` was filed as suspicious drift when 54 is
in fact our correct, deliberate, **MEASURED** direct pin
(`.claude/plans/lance9-datafusion54-upgrade-probe-v1.md`, 2026-08-05). **A POC
session trusting either document would have pinned lance 7 against a lance-9 tree
and failed to build.**

**⊘ Self-correction, same hour (operator-caught).** The first version of THIS entry
then called `datafusion 53.1.0` a *"residual transitive."* Wrong, and more dangerous
than the original error: it frames a required dependency as cruft, inviting a future
session to collapse `Cargo.lock` to a single major and silently break the `delta`
feature. **Both majors are REQUIRED and the dual state is documented upstream:**
`deltalake-core 0.32.4` pins `datafusion 53.1.0` (+ `datafusion-datasource`,
`datafusion-physical-expr-adapter` at 53.1.0); the lance family pins 54. Cargo
permits the coexistence precisely because they are different semver majors. It
resolves only when deltalake moves to DF 54 — not by any action here.

**The meta-lesson:** "this version looks unexpected" has three possible causes —
*stale doc*, *real drift*, and **legitimate multi-major coexistence**. I reached for
the first two and skipped the third, twice in opposite directions. Before labelling
a version anomalous, read *who requires it*; a lockfile entry with a live requirer
is a constraint, never a leftover.

The lockstep discipline itself was never violated: every manifest carries exact
`=9.0.0` / `=0.33.0`. **Only the docs lagged** — which is the more dangerous
failure, because the code compiles and the doc is what a new session reads first.

**Rule:** a version block asserting *"verified against Cargo.lock <date>"* is a
claim with an expiry. When a dependency sweep lands, the sweep's PR must update
every doc that restates its pins — the same-commit board-hygiene rule, applied to
version facts. Grep for the old version string across `.claude/` and `CLAUDE.md`
before closing a bump.

Corrected in this PR: `CLAUDE.md` (all four, + `lance-index`, + a `rust` line
pointing at `rust-toolchain.toml` as authoritative), the plan's §6 with a dated
`⊘ CORRECTION` block (append-only, not a silent edit).

