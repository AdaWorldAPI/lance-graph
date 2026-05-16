# Codex Savant — P1 Anti-Pattern Catalogue + Stable-Rust Toolchain Reference

> **READ BY:** `brutally-honest-tester` (canonical); also any agent reviewing
> a PR diff before commit; any worker self-checking before reporting DONE
> (per `worker-template-v2.md` §9 codex-P1 self-scan).
>
> **Status:** FINDING (every pattern below was caught at least once by
> the codex GitHub bot during sprint-11 or sprint-12; severity counts
> reflect observed-in-the-wild instances, not theoretical risks).
>
> **Predecessors:** `.claude/board/sprint-log-11/meta-review-opus.md`
> CSI-2; `.claude/board/sprint-log-12/meta-review.md` CSI-7/8/9/10/15;
> `EPIPHANIES.md` E-META-10.
>
> **Promotion track:** these patterns drove the iron-rule
> `I-LEGACY-API-FEATURE-GATED` (CLAUDE.md, added 2026-05-16). The
> catalogue below is the operational counterpart — *what to grep for,
> what to fix.*

---

## 1. The codex bot's review shape (observed)

The `chatgpt-codex-connector` bot reviews every PR diff and posts inline
comments with one of three severity badges:

| Badge | Meaning | Sprint-11/12 frequency |
|---|---|---|
| **P1** | Likely a real bug or breaking behaviour change — fix before merge | 5 (all caught + resolved pre-merge) |
| **P2** | Strong recommendation — fix unless explicitly justified | 3 (rename, doctest, naming) |
| **P3** | Style / style-adjacent / nitpick | not tracked separately |

P1 findings always precede behaviour bugs; P2 findings always precede
API-shape regrets. The catalogue below names every P1/P2 pattern the
bot has caught in this workspace and the grep pattern that catches
each one *before* the bot does.

The brutally-honest-tester agent (`.claude/agents/brutally-honest-tester.md`)
runs this catalogue locally + the toolchain in §3 before main thread
pushes to GitHub — so codex's reviews become a *secondary confirmation*
of work already gated, not a first-pass discovery.

---

## 2. Anti-pattern catalogue (AP1..AP8)

Each entry has the same shape: **Pattern** | **Grep** | **Rule** | **Fix** | **Sprint instance**. Grep patterns assume `--include="*.rs"` and run against the
PR diff (not the whole tree).

### AP1 — v1-API-under-v2-feature alias *(P1, 4× in PR #383, 1× in W3 spec)*

> **Iron-rule home:** `I-LEGACY-API-FEATURE-GATED` (CLAUDE.md).

**Pattern:** a v1 accessor or setter reads/writes the OLD bit positions
under a v2 feature that reclaimed those bits. Same function name silently
produces different semantics depending on which feature is active.

**Grep targets in the diff:**

```bash
# Any pub fn that reads/writes raw bits AND has a #[cfg(feature = ...)] in scope
grep -nE 'pub fn .*\(.*\) -> (i8|u8|u16|u32|u64|i16|i32|i64)' <file> | head
grep -nB 3 -A 3 '#\[cfg\(feature' <file> | head
grep -nE 'self\.0 (>>|<<|&|\|)' <file>
```

**The rule:** every v1 API path under a v2-layout feature must
transparently route through the canonical mapping OR be feature-gated
to a documented no-op with a migration pointer.

**Fix patterns:**

- **Route through canonical mapping:**
  ```rust
  pub fn inference_type(self) -> InferenceType {
      #[cfg(feature = "v2-layout")]
      { InferenceType::from_mantissa(self.inference_mantissa()) }
      #[cfg(not(feature = "v2-layout"))]
      { InferenceType::from_bits(((self.0 >> INFER_SHIFT) & BITS3_MASK) as u8) }
  }
  ```

- **Feature-gate to no-op:**
  ```rust
  pub fn set_temporal(&mut self, t: u16) {
      #[cfg(feature = "v2-layout")]
      let _ = t;  // bits 52-63 are reclaimed; see pack_v2 / chain-position migration
      #[cfg(not(feature = "v2-layout"))]
      { self.0 = (self.0 & !(BITS12_MASK << TEMPORAL_SHIFT))
          | (((t as u64) & BITS12_MASK) << TEMPORAL_SHIFT); }
  }
  ```

**Sprint-11 instances (5 total, all PR #383):**

| # | Function | Bug | Fixed in |
|---|---|---|---|
| 1 | `pack(..., temporal=X)` | wrote bits 52-63 into v2 reclaim zone | `42b3215` |
| 2 | `inference_type()` | read 3-bit unsigned where v2 stores 4-bit signed mantissa | `42b3215` |
| 3 | `set_temporal()` | wrote bits 52-63 even under v2 (clobbered W/lens/spare) | `42b3215` |
| 4 | `pack(InferenceType::Counterfactual)` | wrote raw discriminant 6 instead of `to_mantissa()` -6 | `42b3215` |
| 5 | W3 spec `pal8_v1_v2_round_trip_zero_default` test | used `temporal=1023` v1 edge that aliases v2 reclaim zone | `33509ab` |

---

### AP2 — Bit-position collision under reclaim *(P1, surfaced via field-isolation test absence)*

**Pattern:** a layout-bit boundary is reclaimed (one field shrunk, another grown) without field-isolation matrix tests that verify writing each field leaves all other fields unchanged.

**Grep targets in the diff:**

```bash
# New SHIFT constants near each other
grep -nE 'const [A-Z_]+_SHIFT: (u8|u32|usize) =' <file>
# Setters that mask + or
grep -nE 'pub fn set_[a-z_]+\(&mut self,' <file>
```

**The rule:** when a PR reclaims bits, the test suite MUST include a
matrix test of the form: *for each field F, write a max-value to F and
zero to all others; assert all-others read back as zero*. Symmetric:
write zero to F and max to all others; assert F reads zero.

**Fix:** Append to the layout test file:

```rust
#[test]
fn field_isolation_matrix() {
    let mut e = CausalEdge64::ZERO;
    e.set_w_slot(63);
    assert_eq!(e.truth(), TrustTexture::Crystalline, "W=63 must not bleed into truth");
    assert_eq!(e.spare(), 0, "W=63 must not bleed into spare");
    // ... repeat for each field
}
```

**Sprint instance:** the absence of this test in PR #383 was *why*
AP1 instances 1-4 went unnoticed until codex review. W2 spec
addendum in PR #381 added the matrix test pattern; sprint-12 Wave G
W-G1 cutover applied it explicitly to the QualiaColumn migration.

---

### AP3 — Sub-crate `[workspace]` table trap *(P1, 1× in Wave F W-F1, fixed in W-G6)*

**Pattern:** a new crate's `Cargo.toml` declares its own `[workspace]`
section, which forces the crate into standalone mode and silently
EXCLUDES it from the parent workspace. `cargo metadata --no-deps` will
not list the crate; cargo `-p <crate>` will fail without
`--manifest-path`.

**Grep targets in the diff:**

```bash
# Any [workspace] table in a sub-crate Cargo.toml
grep -rn '^\[workspace\]' crates/*/Cargo.toml
```

**The rule:** sub-crate Cargo.toml MUST NOT have a `[workspace]` block
when the crate is intended as a workspace member. Either:
- Add the crate to parent `Cargo.toml` `[workspace] members =` AND remove
  the sub-crate's own `[workspace]` table, OR
- Add to `[workspace] exclude =` (standalone crate; built via
  `--manifest-path`).

**Fix:** delete the `[workspace]` section from the sub-crate Cargo.toml +
add the crate path to parent workspace `members =`.

**Sprint instance:** `crates/sigma-tier-router/Cargo.toml` in PR #388
(W-F1) declared `[workspace]` — caught by W-Meta-Opus as CSI-7; fixed
in W-G6 + main-thread commit `d4e5bbc`.

---

### AP4 — lib.rs orphan modules *(P1, 2× in Wave F, fixed in W-G commit `d4e5bbc`)*

**Pattern:** a worker creates a new `crates/<crate>/src/<module>.rs` but
forgets to add `pub mod <module>;` to that crate's `lib.rs`. The file
compiles standalone but is invisible to downstream consumers; its
tests never run.

**Grep targets in the diff:**

```bash
# New .rs files in crates/*/src/
git diff --diff-filter=A --name-only main..HEAD | grep -E '^crates/.*/src/.*\.rs$'
# For each new file, verify lib.rs registers it
grep -n "pub mod <stem>;" crates/<crate>/src/lib.rs
```

**The rule:** every new `crates/<C>/src/<M>.rs` must be paired with a
`pub mod <M>;` line in `crates/<C>/src/lib.rs` *in the same PR*. No
orphan modules.

**Fix:** add the `pub mod` line. Place alphabetically in the existing
`pub mod` block.

**Sprint instances:** `crates/cognitive-shader-driver/src/attention_mask.rs`
+ `attention_mask_actor.rs` in PR #388 (W-F2 + W-F3) were orphans —
caught by W-Meta-Opus as CSI-8; fixed in `d4e5bbc`. Worker template
v1 *instructed workers NOT to touch lib.rs*; worker-template-v2
(`.claude/agents/worker-template-v2.md` §5.1) reverses this and makes
lib.rs registration *part of worker scope*.

---

### AP5 — Cross-repo `mod.rs` orphan *(P1, 2× in ndarray PR #147, fixed in ndarray commit `2a1a1e3`)*

**Pattern:** same as AP4 but cross-repo. A worker writes new files to
`/home/user/ndarray/src/hpc/stream/*.rs` but only registers ONE of them
in `hpc/stream/mod.rs`, leaving the other two as orphans.

**Grep targets in the diff:**

```bash
# In ndarray repo (cross-repo aware)
cd /home/user/ndarray && \
git diff --diff-filter=A --name-only master..HEAD | grep -E '\.rs$'
# For each new file, verify the local mod.rs registers it
grep -n "pub mod" /home/user/ndarray/src/hpc/stream/mod.rs
```

**The rule:** same as AP4, but apply per-repo. Cross-repo PRs need
each-repo's `mod.rs` updated *in the corresponding repo's PR*.

**Sprint instance:** `qualia.rs` (W-F4) + `splat_field.rs` (W-F6) in
ndarray PR #147 were orphans (only `inference.rs` self-registered).
W-Meta-Opus CSI-9; fixed when main thread aggregated.

---

### AP6 — Speculative new abstraction *(P2, 0× observed in workspace; documented for prevention)*

**Pattern:** a worker introduces a new trait, wrapper struct, or
indirection layer to "make things more flexible for future use" when
the current PR has only one concrete use case.

**Grep targets in the diff:**

```bash
# New traits with only one impl
grep -nE '^pub trait ' <file> | head
# New newtype wrappers
grep -nE '^pub struct [A-Z][a-zA-Z]+\(.*::[A-Z]' <file>
```

**The rule:** workspace doctrine per CLAUDE.md "Don't add features,
refactor, or introduce abstractions beyond what the task requires."
A bug fix doesn't need a helper. A one-shot operation doesn't need a
trait. Three similar lines is better than a premature abstraction.

**Fix:** delete the abstraction; inline the concrete use case. If a
second use case appears in a future PR, *then* refactor.

**Sprint instance:** zero in the workspace so far — codex has not
flagged this in sprint-11/12. Documented preemptively because the
v3 plan has wide scope and new abstractions could land in sprint-13.

---

### AP7 — `unsafe` without `// SAFETY:` *(P1, CLAUDE.md hard rule)*

**Pattern:** a new `unsafe { ... }` block lands without a
`// SAFETY: <invariants caller must uphold>` comment immediately
above it.

**Grep targets in the diff:**

```bash
# New unsafe blocks
git diff main..HEAD -- '*.rs' | grep -E '^\+.*unsafe\s*[\{(]'
# Verify SAFETY comment in lines just above
```

**The rule:** CLAUDE.md "Hard Rules" line: *"Every `unsafe` block needs a
`// SAFETY:` comment."* Enforced via `cargo geiger` (which counts
`unsafe` regions and flags newly-added ones) + visual review.

**Fix:** add the `// SAFETY: ...` comment with the explicit invariant
list. If you can't articulate the invariant, the `unsafe` is wrong.

**Sprint instance:** zero in sprint-11/12 — workspace has been mostly
safe Rust. Brutally-honest-tester runs `cargo geiger` to catch any
sprint-13 violations.

---

### AP8 — REST/gRPC endpoint drift *(P2, prevented by `lab-vs-canonical-surface.md`)*

**Pattern:** a PR adds a new `/v1/<thing>` REST endpoint or `Wire*` DTO
per the "System-1 easy path" instead of extending `OrchestrationBridge`
+ `UnifiedStep`.

**Grep targets in the diff:**

```bash
git diff main..HEAD -- '*.rs' | grep -nE '\.route\(.*/v1/'
git diff main..HEAD -- '*.rs' | grep -nE 'pub struct Wire[A-Z]'
```

**The rule:** `.claude/knowledge/lab-vs-canonical-surface.md` (MANDATORY
read before any REST/gRPC/Wire DTO/endpoint/shader-lab work). The
canonical consumer surface is `UnifiedStep` via `OrchestrationBridge`.
REST/gRPC + per-op Wire DTOs are LAB-ONLY scaffolding.

**Fix:** extend the canonical bridge instead. Read the Decision
Procedure in the knowledge doc before adding any handler.

**Sprint instance:** prevented by the knowledge doc + the LAB-ONLY
feature-gate convention. Codex hasn't flagged this in sprint-11/12
because no PR has attempted it — but the prevention pattern is worth
having in the catalogue.

---

## 3. Stable-Rust toolchain reference (Miri-free)

The brutally-honest-tester runs a tiered toolchain on every PR diff
(in approximate order — bail on first P0 failure):

### Tier 1 — Mandatory (every Rust PR)

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
cargo audit                    # RustSec advisory scan
cargo deny check               # license + dep + advisory + bans
```

Severity: any failure here is **P0 / blocker**.

### Tier 2 — Strict (reported as P2 unless severe)

```bash
cargo clippy -- -D clippy::pedantic -D clippy::nursery \
    -A clippy::missing_errors_doc \
    -A clippy::missing_panics_doc
```

### Tier 3 — Recommended (most PRs)

```bash
cargo machete                  # unused-dep detector
cargo geiger                   # unsafe code scanner — see AP7
cargo semver-checks check-release   # public-API SemVer compat
cargo outdated --depth 1       # informational
```

### Tier 4 — Targeted (when scope applies)

```bash
cargo spellcheck check         # comments + docs (public-API PRs)
cargo public-api               # surface diff (lance-graph-contract re-exports)
cargo bloat                    # binary size delta (release builds)
cargo nextest run              # faster parallel test runner (drop-in)
```

### Tier 5 — Formal-ish (heavy; opt-in per-deliverable)

```bash
cargo kani                     # bounded model checker (works on stable)
# loom — concurrency model checker; used in tests, not CLI
cargo mutants                  # mutation testing (slow; opt-in benchmark gate)
cargo tarpaulin                # coverage (Linux only)
```

### Explicit non-fit (nightly only — NOT in the brutally-honest-tester gate)

- `miri` — under-the-hood UB detector (nightly)
- `cargo-careful` — extra UB detection (nightly)
- `cargo-fuzz` — libfuzzer integration (nightly)

`cargo-udeps` (nightly historically; recent versions stable) is in
Tier 3 if the stable build is available.

---

## 4. Severity convention

| Severity | Meaning | Auto-action | Examples |
|---|---|---|---|
| **P0** | Blocker; PR must not merge | `HOLD` verdict; reject push | clippy `-D warnings` fail; `cargo audit` advisory hit; AP1-AP4 missed; `cargo deny` advisory or critical-license |
| **P1** | Strong fix-before-merge recommendation | `HOLD` unless explicit justification | AP7 (unsafe without SAFETY); semver-checks breaking-change without major bump; AP2 (field-isolation matrix missing); AP5 (cross-repo orphan) |
| **P2** | Style / API-shape regret; fix unless justified | `LAND` allowed with note | clippy pedantic; AP6 (speculative abstraction); naming pre-commits (CSI-15 type) |
| **P3** | Nitpick; informational | `LAND` | spellcheck typos; clippy style; `cargo outdated` advisories |

The verdict block produced by the brutally-honest-tester maps each
finding to a severity and renders a ternary `LAND` / `HOLD` / `REJECT`
recommendation.

---

## 5. Workflow integration

```
plan → review → correct → sprint → review code → fix P0 → commit → repeat
                                          ^
                                          brutally-honest-tester runs here
                                          before main thread commits
```

The tester's verdict is the gate between worker DONE and main-thread
commit. It runs *before* GitHub CI fires, so codex bot's eventual
review is a secondary confirmation rather than a first-pass
discovery.

**CI workflow draft:** `.github/workflows/honest-tester.yml`
(flagged as sprint-13 D-CSV-18 in the v3 plan). Runs Tier 1 + Tier 3
unconditionally on every PR; Tier 4 conditional on diff scope.

---

## 6. Maintenance protocol

This knowledge doc is the **operational** catalogue. When codex flags
a NEW pattern (one that doesn't fit AP1-AP8):

1. **Triage** in the relevant sprint-log meta-review (Wave H+ CSI-N)
2. **Promote** if observed in N≥2 PRs → APPEND a new AP entry here
   (this doc is APPEND-ONLY within the catalogue section)
3. **Iron-rule candidate** if observed in N≥3 PRs + substrate-level
   consequence → follow the promotion ceremony in
   `.claude/knowledge/iron-rules-doctrine.md` §3

The brutally-honest-tester agent reads this doc at Tier-1
mandatory-load. New AP entries take effect immediately on the next
worker spawn.

---

## 7. Cross-references

- `CLAUDE.md` §Substrate-level iron rules (the iron-rule formalization
  of AP1)
- `.claude/knowledge/iron-rules-doctrine.md` (PP-2; meta-pattern across
  the 4 iron rules)
- `.claude/agents/brutally-honest-tester.md` (PP-13; the agent that
  runs this catalogue + the toolchain)
- `.claude/agents/worker-template-v2.md` (PP-8; the impl-side discipline
  that prevents AP3/AP4/AP5 from landing in the first place)
- `.claude/board/sprint-log-11/meta-review-opus.md` CSI-2 (the
  original 5-instance cluster)
- `.claude/board/sprint-log-12/meta-review.md` CSI-7/8/9/10/15
  (the sprint-12 instances that drove the workspace + orphan fixes)
- `.claude/knowledge/lab-vs-canonical-surface.md` (AP8 prevention)
- `.claude/knowledge/frankenstein-checklist.md` (composition failure
  modes — complementary to AP2)

---

## 8. One sentence that should survive any refactor

> When the same function name produces different semantics under
> different feature flags, you have already written tomorrow's
> codex P1.
