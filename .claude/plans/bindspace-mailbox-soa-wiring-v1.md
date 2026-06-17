# bindspace → mailbox_soa — WIRING plan (W2→W7), consolidated (v1)

> **Status:** CONSOLIDATED PLAN — synthesized from 5 specialist consolidation
> agents (sequencing / read-shim / two-path-discipline / 0-friction-swap /
> W2-differential). Pending the 3-brutal-critic pass before any wiring lands.
> **Date:** 2026-06-17.
> **Parent:** `bindspace-mailbox-soa-dependency-map-v1.md` (W0 map; columns W1+W1b DONE).
> **Operator constraints (binding):** two paths step by step; never delete the old
> before the new is tested; CausalEdge64 dedup precise; delete BindSpace LAST.
> **Column migration is COMPLETE** (W1 sigma/temporal/expert; W1b content/topic/angle).
> This doc is the **wiring** half — read/write re-point, feature-gate, cutover.

---

## 0. The one decision gate for the operator (W2.5 — mutability surface)

The write path today is `Arc::get_mut(&mut st.driver.bindspace)` (4 sites:
`grpc.rs:136`, `serve.rs:150/601/692`). An **owned `&mut MailboxSoA` is unreachable**
from a shared `Arc<ShaderDriver>` whose `mailboxes` is a plain field — `&mut self.mailboxes`
needs `&mut ShaderDriver`, which `Arc<ShaderDriver>` cannot give. So a mutability surface
must be chosen **before any write re-point (W3/W4b)**. Two clean options:

- **(A) `RwLock<HashMap<MailboxId, MailboxSoA<N>>>` on the driver** — mirrors the existing
  `planes: RwLock<…>` and `awareness: RwLock<…>` already on `ShaderDriver`; keeps the
  `Arc<ShaderDriver>`-shared topology intact; matches the data-flow rule's "interior
  mutability or built once." **Recommended** (lower blast radius, prior art in the same struct).
- **(B) Move the mailbox set out of `Arc<ShaderDriver>` into the bin handler state** so the
  handler owns `&mut` directly — higher blast radius on `serve.rs`/`grpc.rs`.

**This is the only genuine fork; everything else below is determined.** Default = (A).

---

## 1. The read-surface decision (productive disagreement, RESOLVED)

Three agents diverged, then reconciled:
- **convergence:** parameterize `run` over a read trait; extend the contract `MailboxSoaView`
  with `qualia()`/`content_row()`; give `BindSpace` a migration-scoped impl.
- **container:** do NOT extend the contract — `&dyn MailboxSoaView` adds a per-row indirect
  call in the hot O(n²) Hamming loop (wrong for SIMD); content/qualia on the contract
  over-commits the **cold** `SurrealMailboxView`; a `BindSpace: MailboxSoaView` impl is a
  **sentinel lie** (`mailbox_id`/`w_slot`/`phase` are nonsense for a singleton).
- **two-path:** gate the **owner**, not the function body (I-LEGACY-API-FEATURE-GATED); one
  `dispatch()` body written against a stable read surface, the flag picks which object is handed in.

**RESOLUTION — a driver-crate-local, MONOMORPHIZED read surface (no `&dyn`, no contract change):**

```rust
// crates/cognitive-shader-driver/src/backing.rs  (new, driver-local)
enum BackingStore<'a> {
    Singleton(&'a BindSpace),          // path A — live default
    Mailbox(&'a MailboxSoA<1024>),     // path B — feature-gated
}
```
exposing exactly the six-read dispatch surface, dispatching over **inherent accessors that
already exist on both stores** (W1+W1b shipped them with identical signatures):

| shim method | Singleton arm | Mailbox arm | notes |
|---|---|---|---|
| `prefilter(win, &MetaFilter) -> Vec<u32>` | `bs.meta_prefilter(win,f)` | inline sweep over `mb.meta_raw()`, **clamped to `n_rows()`** | the one *behavior* (not data) the mailbox lacks; must be byte-identical |
| `content_row(row) -> &[u64]` | `bs.fingerprints.content_row` | `mb.content_row` | byte-identical stride (`[row*256..]`, `WORDS_PER_FP=256`), proven by W1b test |
| `qualia_row(row) -> QualiaI4_16D` | `bs.qualia.row` | `mb.qualia_at` | Copy |
| `edge_raw(row) -> u64` | `bs.edges.get` | `mb.edge(row).0` | driver reattaches `causal_edge::CausalEdge64` ONLY; ndarray twin barred |
| `entity_type(row) -> u16` | `bs.entity_type[row]` | `mb.entity_type_at` | |
| `len() -> usize` | `bs.len` | `mb.n_rows()` (=N) | the prefilter bound |

`run()` keeps **one body**, written against `BackingStore` (or a generic `R: DriverRead` —
equivalent; monomorphized either way). The feature flag selects which variant is constructed,
**never** `#[cfg]`-branches inside `run`. `ontology` is **NOT** a shim method — it stays a
shared `Arc<OntologyRegistry>` field on the driver, resolved against the `entity_type(row)` the
shim returns. **No `lance-graph-contract` change for this migration** (a contract `qualia()`
lands later, gated on the *planner* consumer, per the `soa_view.rs:71` note — not here).

---

## 2. Feature-gate invariants (I-LEGACY-API-FEATURE-GATED-safe)

- **I-FG-1 (single arc).** ONE `mailbox-thoughtspace` feature spanning W3→W6. No per-W
  sub-features (a half-migration W4-on/W3-off is a silent-divergence state nobody tests).
- **I-FG-2 (gate the owner).** The flag selects which `BackingStore` variant is built; it must
  NOT introduce `#[cfg]` branches inside `dispatch()`/`ingest_codebook_indices`/`persist_cycle`
  that read different columns under the same fn name. The §1 monomorphized shim is the
  mechanism that makes the single flag safe.
- **I-FG-3 (default OFF until W7).** `default = [...]` must NOT include `mailbox-thoughtspace`
  until W7. Singleton stays the default-build behaviour while both paths live; the mailbox path
  is opt-in and differentially tested. (Contrast `causal-edge-v2-layout` = default-ON because
  that migration is *finished*; this one is not.)

---

## 3. Boundaries that must stay distinct (do NOT flatten early)

| id | distinction held | collapses at |
|---|---|---|
| D-DIST-1 | singleton store ≠ mailbox store (two owners, separately allocated; mailbox OWNS, never `&mut`-borrows from the singleton — `E-CE64-MB-4`) | **W7** |
| D-DIST-2 | read surface ≠ write surface (`MailboxSoaView` vs `MailboxSoaOwner`; cold side read-only) | never |
| D-DIST-3 | typed `causal_edge::CausalEdge64` ≠ raw `u64` column; ndarray twin **barred**; reattach points do NOT multiply | never (firewall permanent) |
| D-DIST-4 | ephemeral thoughtspace ≠ calcified shared (ontology/CAM-PQ stay by `&`/`Arc`, never per-mailbox-owned) | never |
| D-DIST-5 | hot mutation ≠ cold persistence (`persisted_row` is a pointer, not a re-encode; no Arrow/JSON marshalling step) | never |
| membrane | `engine_bridge` `bind/unbind_busdto`/`busdto_to_binary16k` re-encode seam — the ONE thing allowed to dissolve | re-point W3; body deleted W7 |

---

## 4. Divergence hazards (the sharp risks)

- **H-DW-1 (headline).** W3 re-points *writers* to the mailbox while W4 still reads the
  singleton → the shader reasons over stale singleton rows; the two stores **diverge silently on
  the same row index**. **Guard:** writers+readers flip **atomically as one gated unit** (collapse
  W3+W4a, or keep the feature non-default-buildable between them AND run the W2 differential on
  every commit in the window).
- **H-DW-2.** Single-writer-per-logical-row is **absolute**. No dual-write "to keep both warm"
  (a rounding/quantization/`temporal`-no-op disagreement → undetectable divergence). Off → singleton
  sole writer; on → mailbox sole writer; never both.
- **H-DW-3.** The four `Arc::get_mut(&mut bindspace)` sites convert to owned `&mut MailboxSoA`
  (via the W2.5 surface) in the W4b unit. A surviving `get_mut`/**`make_mut`** on `bindspace` under
  the feature is a migration leak — `make_mut` silently **clones the whole 4096-row singleton**
  = an instant second diverging store.

---

## 5. The `cycle` (Vsa16kF32) plane — drop discipline

Singleton keeps `write_cycle_fingerprint`/`cycle_row` until **W7**; `MailboxSoA` **never** gets a
`cycle`/`set_cycle`/`cycle_row` method (compile-level assertion). Under W3:
- **F-CYCLE-1** `dispatch_busdto` (`engine_bridge.rs:243`) — stop writing the column; compute the
  Binary16K accumulator transiently for the current resonance, discard.
- **F-CYCLE-3** `persist_cycle` (`engine_bridge.rs:707`) — stop the `write_cycle_fingerprint`;
  keep the legitimate `edges`/`meta` column writes.
- **F-CYCLE-2** `unbind_busdto` (`engine_bridge.rs:334`) — **REWRITE, do not re-point.** It reads the
  *persisted* cycle plane back; that plane won't exist on the mailbox. Per parent-plan §2.6 it
  must collapse to a column read (`edges`/`qualia`/`meta`). This is the one function that cannot be
  mechanically migrated.
- **F-CYCLE-5** `driver.rs:288` `cycle_fp` local is the CORRECT transient model (a `[u64;256]`
  Binary16K fold on the stack, never a column). Footprint check: hot cost ≈ 6 KB/row
  (content/topic/angle), NOT ≈ 71.6 KB/row (a leak of the 64 KB cycle plane).

---

## 6. The hardened W2→W7 sequence (dependency-correct)

```
W2   read-parity differential harness (test-only)         READY NOW   additive   ∥-safe
W2.5 mailbox &mut surface — DECISION GATE (A: RwLock rec)  READY NOW   additive   operator pick
W3   engine_bridge mailbox writes (feat-gated)            BLOCKED:W2.5 behaviour  serial   +cycle-reader audit +temporal field-isolation
W4a  dispatch READ re-point via BackingStore (feat-gated) BLOCKED:W2,W2.5 behaviour serial  (W2-validated; flip atomic w/ W3 per H-DW-1)
W4b  Arc::get_mut → owned &mut (4 sites)                  BLOCKED:W2.5,W3 behaviour serial  (fixes latent "multiple references" 500; never make_mut)
W5   bins build mailbox set (feat-gated)                  BLOCKED:W4b  behaviour  serial   +mailbox-set topology sub-gate (N, count, keying)
W6   death→SPO-G+Lance tombstone                          HARD-BLOCKED (surrealdb #41)  design ∥-only, NO code  ← DECOUPLED from W7
W7   delete BindSpace + cycle plane; drop feature gate    BLOCKED:W3,W4a,W4b,W5 + §2e test migration  serial  LAST  (NOT gated on W6)
```

**Per-step gates:**
- **W2 — READY NOW.** Differential test only; reads the **concrete** `MailboxSoA` (not the contract
  view). Build the mailbox with **populated rows == BindSpace `len`** so the prefilter sweeps the
  same set. Assert `ShaderResonance` **bit-identical** via `f32::to_bits()` (top_k row/distance/
  predicates/cycle_index + resonance bits, entropy bits, std_dev bits) + `cycle_fingerprint` +
  **`MaterializeProvenance`** (so a future read re-point can't silently drift #515's provenance);
  assert `gate` only with `entity_type == 0` rows (neutralizes the `ontology()` read W2 can't serve).
- **W2.5 — operator decision (A vs B).** Default A (`RwLock`). The gate for the whole write arc.
- **W3 — +2 mandatory in-PR gates:** (a) a `read_cycle_fingerprint`/`cycle_row` reader audit
  promoting "cycle→transient" from CONJECTURE to FINDING; (b) the `temporal` field-isolation matrix
  test (write temporal, assert edges/meta/sigma/expert unchanged) per I-LEGACY-API-FEATURE-GATED.
- **W4a/W4b — split** (read re-point vs ownership-model change). W4a validated by the W2 harness;
  W4b changes handler error semantics (removes the "multiple references" 500).
- **W5 — sub-gate:** define the mailbox-set builder (`N`, count, keying) before the bin move; do not
  invent the sea-star topology ad-hoc in `main()`.
- **W6 — external blocker** (surrealdb #41); design proceeds as docs/contract-only, NO code.
- **W7 — decoupled from W6.** Prereq is **zero singleton readers** + the two §2e singleton-bound
  tests (`end_to_end.rs`, `busdto_bridge_test.rs`) migrated to mailbox FIRST. Then delete
  `bindspace.rs` + the cycle plane + the feature flag in one PR.

---

## 7. #515 MaterializeProvenance interaction

`materialize_provenance` (now on `main`, `driver.rs`) is **provenance-only** — it consumes
already-derived observables (`free_energy`, `std_dev`, `top_resonance`, `awareness_skill`,
hits), reads **no** `BindSpace` directly → adds no new singleton reader and does not gate the
migration. The one interaction: its inputs come from the dispatch read path W4a re-points, so
**W2 must assert `MaterializeProvenance` byte-equality** across stores (§6 W2 gate). `ctx_id`/MUL
still read `ontology()` from the shared `Arc` (not the mailbox).

---

## 8. Acceptance checks (per the operator's "test the new before delete the old")

- W2 differential: bit-identical `ShaderResonance` + `cycle_fingerprint` + `MaterializeProvenance`,
  singleton-backed vs mailbox-backed, same rows — run in BOTH the W3-landed/W4-pending state and
  the W4-complete state (H-DW-1 window).
- Field-isolation matrix (W3): each migrated column write leaves all others byte-unchanged.
- CausalEdge64 firewall (CI): `ndarray::hpc::causal_diff::CausalEdge64` has **zero** imports in
  `cognitive-shader-driver`; an `edges_raw()` reattach round-trips to the typed
  `causal_edge::CausalEdge64` it was packed from.
- Cycle-drop (compile + bench): `MailboxSoA` exposes no `cycle*` symbol; hot footprint ≈ 6 KB/row.
- W7 readiness: grep proves zero `self.bindspace.*` reads under feature-on; the two §2e tests pass
  on the mailbox path before deletion.
