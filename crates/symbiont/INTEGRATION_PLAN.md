# Integration plan — loose ends → the Spain-grid acceptance gate

Status legend: ☐ open · ◐ in progress · ☑ done (this session)

---

## Done this session (the foundation)

- ☑ **ractor messaging compiles.** `MessagingErr::Saturated` handled at all
  three match sites (`actor.rs`, `thread_local/inner.rs`, `derived_actor.rs`).
  This is the kanban backpressure valve. (AdaWorldAPI/ractor#2, merged.)
- ☑ **kv-lance feature gates proven + documented.** Lite-unified surreal
  compiles without RocksDB/C++ storage. (AdaWorldAPI/surrealdb#47, #48, merged.)
- ☑ **Golden image compiles + links.** `cargo build` exit 0, 19m18s,
  `target/debug/symbiont` 4.2 MB, 912 packages, zero errors. The five forks
  resolve AND compile+link into one binary; lockstep pins held. (This is a
  compile milestone — it proves nothing about runtime data flow; see the
  loose-end ledger below.)
- ☑ **Perturbation-sim NaN foundations.** `cascade.rs` preserve-last-finite
  abort + `perturbation_shape_is_always_finite` test; `stats.rs` empty-slice
  guards on `mean`/`pop_var`. (lance-graph, merged.)

---

## Council findings (5+3 hardening, 2026-06-19) — read before §A

An 8-agent council (5 research + 3 brutal reviewers) audited the gap between
"compiles" and the win condition. The one finding everything reduced to:

> **The five crates are linked into one binary with ZERO runtime edges
> between them.** "Compiles" proves the dependency graph; it proves nothing
> about data flow. There are **three incompatible "node" representations and
> no adapter between any of them:**
> 1. canonical `NodeRow` (4096-bit, `lance-graph-contract::canonical_node`) — what the win condition means by "16K-node SoA"
> 2. `VersionedGraph::NodeSchema` (SPO triple planes, `FixedSizeBinary(2048)`, `blasgraph/columnar.rs`) — what `LanceVersionScheduler` *actually* reads today
> 3. perturbation-sim's `Grid`/`PerturbationShape` (plain `f64`) — what the cascade produces

**☐ D0 — PREREQUISITE DECISION (gates all of §A): pick which representation
"the 16K-node SoA" is.** A2 says "canonical 4096-bit node"; the only wired
Lance substrate (`VersionedGraph`) uses a *different* SPO-plane schema. They
cannot both be "the 16K-node SoA." Decide canon (`NodeRow`) and the §A work
targets it; until written down, the Grid→substrate bridge can't be aimed.

**Corrected prerequisite chain** (the plan's flat checkboxes hid these):
`D0 (pick representation)` → `A1 fixture` (also: create the `tests/` dir — it
doesn't exist) → `#1 perturbation-sim gains lance-graph-contract dep` →
`A2 Grid→NodeRow bridge` → `#3 NodeRowPacket→Lance writer` → `A3/A4`.
`C2` (clippy, §C) is independent and **failing now** — cheapest to clear.
The entire kanban loop (ractor scheduler, jitson dispatch, surrealdb version
stream) is **genuinely post-gate** — the 3-part gate needs none of it.

**Key-encoding probe (gates whether A2 is mechanical):** the *value* side of
the bridge is a 0-friction OPPORTUNITY (`basin.rs::as_row()[5]` +
`buffer.rs::inertia_buffer_column()` → `ValueTenant` slots, algebra aligned).
The *key* side is WORTH-EXPLORING: `hhtl.rs::HhtlKey` is the binary-Cheeger
1-bit/tier instance, **not** OGAR's 16-ary/256-centroid production key — it
type-aligns (`u16×3`) but isn't prefix-routable. Probe first: does the binary
key give acceptable HHTL routing locality on the Spain grid, or must the
centroid encoder (compose `basin.rs::spectral_embedding` + `splat.rs::morton2`)
be built before A4's cascade routing is meaningful?

**Honesty corrections applied to the docs (overclaim-auditor):** the README
no longer states the substrate "carries" Spain's grid in present tense; the
build milestone is scoped to compile/link (done) vs data-flow (not); the
"912 packages" claim is scoped to resolution+build, with the two-`object_store`
caveat noted.

### Reviewer findings — golden-image setup correctness (P0/P1 reviewers)

Verdicts: brutally-honest-tester = **HOLD**, baton-handoff-auditor =
**CATCH-LATENT**. The image links cleanly today; these harden it into a
*reproducible* foundation. None blocks the current green build.

- **☐ R1 (latent, top item) — the AdaWorldAPI `ndarray` fork is linked TWICE.**
  lance-graph uses `path = ../../../ndarray` (local HEAD `786110a`);
  surrealdb-core uses `git ...ndarray.git rev=0129b5c8` (older), non-optional.
  symbiont's `[patch]` covers surrealdb-* but NOT `ndarray.git`, so two
  distinct `0.17.2` crate identities compile + link. (A third `ndarray 0.16.1`
  from crates.io via `lance-index` is the *real* numerical ndarray — a
  different crate sharing the name; harmless.) Latent because no ndarray type
  crosses the surrealdb↔lance-graph seam today; drops the baton if a future
  workload passes a `Fingerprint`/array across it (mismatched `TypeId`).
  **Fix (do carefully):** align the source — either checkout local ndarray to
  `0129b5c8`, or bump surrealdb's pin to local HEAD, then add
  `[patch."https://github.com/AdaWorldAPI/ndarray.git"] ndarray = { path = "/home/user/ndarray" }`.
  Verify API compat first; this is a 19-min rebuild that can break surrealdb-core
  if the fork's API drifted between the revs. NOT attempted now (the green
  build is preserved).
- **☐ R2 — commit `symbiont/Cargo.lock`.** It exists on disk (the build
  generated it) but isn't tracked. Without it, `branch`-pinned git deps
  (OGAR's surrealdb `main`, ndarray) can resolve to different commits on
  different days → not byte-reproducible.
- **☐ R3 — pin OGAR's surrealdb git dep to an exact `rev`.** `OGAR/Cargo.toml`
  uses `branch = "main"`, but symbiont's `[patch]` silently substitutes the
  local tree on a *different* branch. Compiles today (AST shape matches);
  drops the baton if the local branch advances the AST or the patch is removed.
- **☐ R4 — regenerate `/home/user/surrealdb/Cargo.lock`.** It resolves lance
  **6.0.0** / lancedb 0.29 — contradicting surrealdb's own `=7.0.0` manifest
  pin. surrealdb's kv-lance-on-lance-7 path was **never resolved inside
  surrealdb's own workspace**; symbiont is the first witness. Regenerate so
  the fork's CI exercises lance 7.
- **note — absolute paths are deliberate** (`publish = false`); the image is
  intentionally machine-pinned to `/home/user/{...}`. Switch to relative
  (`../`) only if portability is wanted.

**NaN coverage (reviewer-confirmed, strong):** `cascade.rs:146` finite-guard,
`perturbation.rs` `FRAGMENTATION_SENTINEL = +∞` (deliberately not NaN,
finiteness-checkable), `eigen.rs:123` div-guard, `stats.rs` divisor floors.
One real P2 gap: a `+∞` sentinel reaching `stats::pearson` makes `saa*sbb=+∞`
→ `sqrt`→ ratio → **NaN**, and the `<1e-12` guard does NOT catch `+∞`. Add an
`is_finite` filter at the stats boundary + a `pearson_rejects_nonfinite` test.
This folds into §B (the NaN-free win condition).

## The acceptance gate (the biggest goal)

> **16K-node SoA substrate carries every Spanish electricity node; the
> perturbation cascade runs NaN-free; `cargo clippy` + `cargo machete` clean.**

### A. Substrate carries the Spanish grid

- ☐ **A1 — source the Spanish grid topology.** REE / ENTSO-E node + line
  list (buses, lines, transformers, susceptances). Deterministic fixture
  checked into `perturbation-sim/tests/fixtures/` (no network at test time).
- ☐ **A2 — map each grid node → one canonical 4096-bit node.**
  `key(16) = classid(u32) | HEEL | HIP | TWIG | family(u24) | identity(u24)`.
  Grid nodes start in the default basin (classid=0, family=0); `identity`
  alone discriminates (16.7M capacity — Spain's ~10³–10⁴ buses fit trivially).
  Edges (12 in-family + 4 out-of-family) carry the line adjacency.
- ☐ **A3 — load the grid into a `MailboxSoA` view over a Lance dataset.**
  The 16K-node column is the Lance-backed SoA; this is where `kv-lance`
  earns its place (zero-copy columnar, versioned).
- ☐ **A4 — run the cascade over the full node set.** `cascade.rs`
  (Weyl/Davis-Kahan spectral perturbation ∘ DC-power-flow/LODF) +
  `basin.rs` (Kron-reduced cross-border super-nodes) + `scorecard.rs`
  (ES `policy_mult` 1.3, `H` 2.0). Output: the perturbation SHAPE per node.

### B. NaN-free, enforced

- ☐ **B1 — NaN linter guard.** A clippy lint / debug-assert pass that fails
  if any `f32`/`f64` in the cascade, spectral step, or scorecard is non-finite.
  Build on the existing `is_finite()` guards; promote them to a checked
  invariant at module boundaries (not just the cascade loop).
- ☐ **B2 — property test over the grid fixture.** Extend
  `perturbation_shape_is_always_finite` to the full Spain fixture (every
  node, every cascade round) — the regression that proves B1 holds on real
  topology, not just synthetic input.

### C. Tight graph

- ☐ **C1 — `cargo machete` clean.** Remove unused deps from the golden-image
  graph and from `perturbation-sim`. (Machete reads manifests; cheap.)
- ☐ **C2 — `cargo clippy --all-targets -- -D warnings` clean** across the
  symbiont graph (at least the first-party crates; upstream warnings triaged).

---

## Other loose ends (post-gate)

- ☐ **surreal_container `BLOCKED(C)`.** The `surreal_container` consumer still
  has the kv-lance fork dep unwired in its `Cargo.toml`. The golden image
  proves the dep graph works; porting that wiring into `surreal_container`
  clears the block.
- ☐ **ndarray-simd in perturbation-sim.** Enable the `ndarray-simd` feature
  (Walsh-Hadamard via ndarray AVX-512 under `target-cpu=x86-64-v4`) and
  `[patch]` perturbation-sim's git ndarray to the local fork. Deferred from
  the first image to keep the AVX/git-patch risk out of the initial compile.
- ☐ **Kanban loop wiring.** Stand up `LanceVersionScheduler` (ractor) →
  `KanbanMove(ExecTarget::Jit)` → jitson formula → `MailboxSoaView` write →
  Lance commit. The perturbation cascade becomes the first *formula* the
  scheduler dispatches.
- ☐ **main.rs as a real harness.** Replace the probe `println!` with a CLI
  that loads the grid fixture, runs the cascade, prints the scorecard, and
  asserts finite — so `cargo run` IS the acceptance-gate demo.
- ☐ **Optional: no-C++ image.** Drop S3 cloud object-store features + flip
  `jsonwebtoken` to `rust_crypto` (see INSTALLATION.md). Nice-to-have only.

---

## Risks / watch-items

- **Two `object_store` versions** appear in the resolved graph (lance vs
  surrealdb transitive). Allowed by cargo (distinct majors); watch for any
  public-type mismatch if they ever meet at an API boundary.
- **Disk:** the full `target/` is multi-GB; build in one shared target dir,
  clean sibling `target/`s (build residue, not research data) if headroom
  drops below ~3 GB.
- **edition 2024 (OGAR)** requires the 1.95 toolchain in the active override —
  `rust-toolchain.toml` pins it; don't run the image build under 1.94.
