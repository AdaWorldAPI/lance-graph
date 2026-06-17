# bindspace → mailbox_soa — WIRING plan (W2→W7), consolidated + critiqued (v2)

> **Status:** CONSOLIDATED + 3-BRUTAL-CRITIC PASS APPLIED. v1 verdict was **HOLD**;
> v2 integrates every P0/P1. Architecture (mailbox-as-owner, zero-copy, CausalEdge64
> firewall, cycle-drop) confirmed sound by all 3 critics; the fixes below are
> sequencing + two corrected premises, not redesign.
> **Date:** 2026-06-17.
> **Parent:** `bindspace-mailbox-soa-dependency-map-v1.md` (W0 map).
> **Operator constraints (binding):** two paths step by step; never delete the old
> before the new is tested; CausalEdge64 dedup precise; delete BindSpace LAST.
> **Column status (CORRECTED):** W1 (sigma/temporal/expert) **merged** (#517); **W1b
> (content/topic/angle) is OPEN in #518, NOT on `main`** — so `MailboxSoA::content_row`
> does not exist until #518 merges. The wiring half below is **gated on #518**.
> This doc is the **wiring** half — read/write re-point, feature-gate, cutover.

### v2 critique-integration changelog (what the 3 brutal critics overturned)
- **DROPPED the "W2.5 mutability decision gate" — it was a non-issue** (PP-13 P0-2):
  there is **no `Arc<ShaderDriver>`**. The bins hold `Arc<Mutex<ServerState>>` /
  `Arc<Mutex<ShaderDriver>>`; the existing `Arc::get_mut(&mut st.driver.bindspace)` works
  because the **Mutex guard already yields `&mut ShaderDriver`** → `&mut self.mailboxes`
  (plain `HashMap`) is free. **Do NOT add a `RwLock`** — option (A) would create a new
  `Mutex<ShaderDriver>` → mailbox-`RwLock` → `awareness.write()` (`driver.rs:556`)
  lock-order edge (PP-13 P0-3). No operator decision is needed here.
- **W2 is BLOCKED on #518**, not "READY NOW" (PP-13 P0-1 / PP-16 P0): `content_row` is the
  hottest dispatch read and only exists on the #518 branch.
- **prefilter off-by-(N−len)** (PP-13 P1-3): a zeroed `MetaWord` *passes* `MetaFilter::accepts`
  (`0>=0`, `thinking_mask==0` accepts all), so a 1024-row mailbox returns 1024 phantom rows
  vs BindSpace's `len`. `MailboxSoA` has **no populated-count**. → add a high-water-mark
  (W1c below) and clamp the prefilter to *that*, not `n_rows()`.
- **`BusDto` round-trip P0** (baton-handoff): `unbind_busdto`'s non-headline `top_k` indices
  are recovered from the *persisted cycle-plane bits*, which the mailbox does not carry →
  W3 needs a `BusDto` differential gate, and D-DIST-5 is reclassified to name the cycle
  exception (§3, §5).
- **W3+W4a must be ONE atomic PR** (PP-13 P0-4): sequencing them serially lands H-DW-1.
- **Ontology re-home** (baton-handoff P1): `MailboxSoA` has no ontology handle → at W7
  `ctx_id` silently falls to `unwrap_or(0)`. Re-home `Arc<OntologyRegistry>` onto
  `ShaderDriver` (W4); extend W2 to assert `ctx_id` parity on `entity_type != 0`.
- **Shim edge arm** uses `mb.edges_raw()[row]` (raw), not `mb.edge(row).0` (typed bounce) —
  one reattach at the `run()` site (baton-handoff P1).
- **Emit-path `pack(temporal=cycle_index)`** (`driver.rs:402-413`) is a live v2 no-op; W3 adds
  an observability assertion (baton-handoff P1).
- **THIRD `CausalEdge64`** exists — `thinking_engine::layered::CausalEdge64` (in-graph via
  `with-engine`); the firewall CI lint bars **both** it and the ndarray twin (baton-handoff P2).
- Cosmetic: `unbind_busdto` fn is `engine_bridge.rs:315` (the cycle read is `:334`);
  `busdto_bridge_test.rs` binds the singleton in **5** cases (migrate all 5 at W7).

---

## 0. Mutability surface — NOT a decision gate (corrected)

**There is no operator fork here.** The bins own the driver behind a `Mutex`
(`serve.rs:70` `Arc<Mutex<ServerState>>` with `driver: ShaderDriver` by value; `grpc.rs:37`
`Arc<Mutex<ShaderDriver>>`). The existing 4 write sites (`grpc.rs:136`, `serve.rs:150/601/692`)
call `Arc::get_mut(&mut st.driver.bindspace)` — and that works **only because the `Mutex` guard
already hands out `&mut ShaderDriver`**. That same `&mut ShaderDriver` yields `&mut self.mailboxes`
(a plain `HashMap` field, `driver.rs:88`) **for free**. So the mailbox write path (W3/W4b) needs
**no new mutability surface**: the W4b sites become a `mb = st.driver.mailboxes.get_mut(id)` plus
the owned `&mut MailboxSoA`, reached through the guard the handler already holds.

**Do NOT add a `RwLock<HashMap<…>>`** (the v1 "option A"): it would nest a third lock inside the
existing `Mutex<ShaderDriver>` and create a `Mutex → mailbox-RwLock → awareness-RwLock`
(`driver.rs:556` `awareness.write()`) acquisition edge with no documented order — manufacturing
the deadlock surface the migration must avoid. `planes`/`awareness` are independent leaf RwLocks
never co-held across a write; a mailbox write lock held while `awareness.write()` fires is a *new*
edge. The plain-field-through-the-guard path has none of this.

**The only genuinely new prerequisite is W1c (populated-count), not a lock** — see §6.

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
| `prefilter(win, &MetaFilter) -> Vec<u32>` | `bs.meta_prefilter(win,f)` | ascending sweep over `mb.meta_raw()` **reusing `MetaFilter::accepts`**, **clamped to `mb.populated()` (W1c), NOT `n_rows()`** | **CRITICAL (PP-13 P1-3):** a zeroed `MetaWord` passes `accepts`; clamping to `n_rows()=1024` returns (N−len) phantom rows. Must iterate `0..populated` ascending so `passed_rows` order matches `BindSpace::meta_prefilter`'s `for row in start..end`. |
| `content_row(row) -> &[u64]` | `bs.fingerprints.content_row` | `mb.content_row` | byte-identical stride (`[row*256..]`, `WORDS_PER_FP=256`), proven by W1b test — **but `mb.content_row` exists only on the #518 branch; W2 gated on #518** |
| `qualia_row(row) -> QualiaI4_16D` | `bs.qualia.row` | `mb.qualia_at` | Copy |
| `edge_raw(row) -> u64` | `bs.edges.get` | **`mb.edges_raw()[row]`** (raw — NOT `mb.edge(row).0`) | one reattach at `run()` only; `mb.edge(row).0` is a typed→raw→typed bounce that softens the single-reattach firewall (baton-handoff P1). Driver wraps in `causal_edge::CausalEdge64`; ndarray twin AND `thinking_engine::layered::CausalEdge64` barred. |
| `entity_type(row) -> u16` | `bs.entity_type[row]` | `mb.entity_type_at` | |
| `len() -> usize` | `bs.len` | `mb.populated()` (W1c) | the prefilter bound — **populated count, not the const `N`** |

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
| D-DIST-5 | hot mutation ≠ cold persistence (`persisted_row` is a pointer, not a re-encode) — **EXCEPT the `cycle`/`BusDto` path (see below)** | never, except the named exception |
| membrane | `engine_bridge` `bind/unbind_busdto`/`busdto_to_binary16k` re-encode seam — the ONE thing allowed to dissolve | re-point W3; body deleted W7 |

> **D-DIST-5 RECLASSIFIED (baton-handoff P0).** The "no re-encode" invariant is **false for the
> `cycle`/`BusDto` path**: `set_cycle_from_bits` (`bindspace.rs:114`, Binary16K→Vsa16kF32 bipolar)
> ⟷ `unbind_busdto`'s `vsa16k_to_binary16k_threshold` (`engine_bridge.rs:337`) is a **lossy
> round-trip**, and `unbind_busdto`'s **non-headline `top_k` indices are recovered from the
> persisted cycle-plane set-bits** — information the mailbox does not carry on any column. So
> dropping the cycle plane (by design, OQ-1) **necessarily downgrades the `BusDto` round-trip
> contract**: only the headline index (stored in `qualia[9]`, codex-P2 fix) survives; non-headline
> indices with positive energy at encode are **not recoverable from columns**. This is a
> deliberate, named exception — NOT a silent F-CYCLE-2 "rewrite." It is `with-engine`-gated (lab
> path). **W3 gate:** before the cycle plane is touched, add a `BusDto` differential
> (`unbind_busdto(mailbox)` vs `unbind_busdto(bindspace)`) that asserts headline + energies +
> cycle_count + converged parity and **explicitly documents the non-headline-index loss** (update
> `busdto_bridge_test.rs` tolerance). If full round-trip must be preserved, that requires a NEW
> column to carry the top_k indices — escalate to the operator before assuming the loss is OK.

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
  (reached via the existing `Mutex` guard — §0, no new lock) in the W4b unit. A surviving
  `get_mut`/**`make_mut`** on `bindspace` under the feature is a migration leak — `make_mut`
  silently **clones the whole 4096-row singleton** = an instant second diverging store.

---

## 5. The `cycle` (Vsa16kF32) plane — drop discipline

Singleton keeps `write_cycle_fingerprint`/`cycle_row` until **W7**; `MailboxSoA` **never** gets a
`cycle`/`set_cycle`/`cycle_row` method (compile-level assertion). Under W3:
- **F-CYCLE-1** `dispatch_busdto` (`engine_bridge.rs:243`) — stop writing the column; compute the
  Binary16K accumulator transiently for the current resonance, discard.
- **F-CYCLE-3** `persist_cycle` (`engine_bridge.rs:707`) — stop the `write_cycle_fingerprint`;
  keep the legitimate `edges`/`meta` column writes.
- **F-CYCLE-2** `unbind_busdto` (fn at `engine_bridge.rs:315`; the persisted-cycle read is `:334`) —
  **REWRITE, do not re-point**, and **with a downgraded contract** (see D-DIST-5 reclassification).
  It reconstructs non-headline `top_k` indices from cycle-plane set-bits (`:337-352`) — the mailbox
  has no column for those. The rewrite collapses to a column read (`edges`/`qualia`/`meta` + the
  `qualia[9]` headline) and **explicitly loses non-headline indices**; its W3 differential gate +
  `busdto_bridge_test.rs` tolerance update must land in the same PR. The one function that cannot be
  mechanically migrated AND cannot preserve its full contract without a new column.
- **F-CYCLE-5** `driver.rs:288` `cycle_fp` local is the CORRECT transient model (a `[u64;256]`
  Binary16K fold on the stack, never a column). Footprint check: hot cost ≈ 6 KB/row
  (content/topic/angle), NOT ≈ 71.6 KB/row (a leak of the 64 KB cycle plane).

---

## 6. The hardened W2→W7 sequence (dependency-correct)

```
W1c  add MailboxSoA::populated() high-water-mark          BLOCKED:#518  additive   ∥-safe  (prefilter bound; PP-13 P1-3)
W2   read-parity differential harness (test-only)         BLOCKED:#518,W1c additive  ∥-safe  (content_row only on #518)
W3+W4a  ATOMIC: engine_bridge mailbox writes + dispatch    BLOCKED:W2  behaviour  serial  ONE PR (H-DW-1: writer+reader flip together)
        read re-point, both feat-gated `mailbox-thoughtspace`        +cycle-reader audit +temporal field-isolation +BusDto gate +emit-path temporal assertion +feature-on CI row
W4b  Arc::get_mut → owned &mut (4 sites) + ontology re-home BLOCKED:W3  behaviour  serial  (Mutex guard, no new lock; never make_mut; Arc<OntologyRegistry> → ShaderDriver field)
W5   bins build mailbox set (feat-gated)                  BLOCKED:W4b  behaviour  serial   +mailbox-set topology sub-gate (N, count, keying)
W6   death→SPO-G+Lance tombstone                          HARD-BLOCKED (surrealdb #41)  design ∥-only, NO code  ← DECOUPLED from W7
W7   delete BindSpace + cycle plane; drop feature gate    BLOCKED:W3,W4a,W4b,W5 + §2e test migration (5 sites in busdto_bridge_test)  serial  LAST  (NOT gated on W6)
```

**Per-step gates:**
- **W1c — BLOCKED on #518.** Add `MailboxSoA::populated() -> usize` (a high-water-mark set as rows
  are written) + clamp the shim prefilter to it. Without this, a zeroed `MetaWord` passes
  `MetaFilter::accepts` and the mailbox returns (N−len) phantom rows (PP-13 P1-3). Additive, tested.
- **W2 — BLOCKED on #518 + W1c.** Differential test only; reads the **concrete** `MailboxSoA`
  (`content_row` exists only on #518). Use **one driver instance, two backing arms** (same
  `awareness`/`planes`/`semiring`, vary only the `BackingStore`). Build the mailbox with **populated
  rows == BindSpace `len`** and clamp the prefilter to `populated()`. Assert `ShaderResonance`
  **bit-identical** via `f32::to_bits()` (top_k row/distance/predicates/cycle_index + resonance bits,
  entropy bits, std_dev bits) + `cycle_fingerprint` + **`MaterializeProvenance`**; assert `gate`
  with `entity_type == 0` rows (neutralizes `ontology()`), AND (after W4b ontology re-home) a second
  `gate` parity assertion on `entity_type != 0` with a registry attached.
- **W3+W4a — ONE ATOMIC PR** (PP-13 P0-4 / H-DW-1): the writer re-point (`engine_bridge` mailbox
  arm) and the reader re-point (`dispatch` `BackingStore` mailbox arm) flip **together**; never two
  serial steps. In-PR gates: (a) `read_cycle_fingerprint`/`cycle_row` reader audit promoting
  "cycle→transient" CONJECTURE→FINDING; (b) `temporal` field-isolation matrix test
  (I-LEGACY-API-FEATURE-GATED); (c) **`BusDto` differential gate** + `busdto_bridge_test.rs`
  tolerance update for the documented non-headline-index loss (D-DIST-5 exception); (d) **emit-path
  assertion** that `driver.rs:402-413` `pack(temporal=cycle_index)` either lands in the mailbox
  `temporal` column or is paired with a `temporal_dropped_under_v2` observability test; (e) a
  **`--features mailbox-thoughtspace` CI matrix row** (build + the W2 differential) — mandatory from
  this PR on, or the feature path rots.
- **W4b — owned `&mut` + ontology re-home.** The 4 `Arc::get_mut` sites become
  `st.driver.mailboxes.get_mut(id)` through the **existing `Mutex` guard** (NO new lock, §0); never
  `make_mut`. Re-home `Arc<OntologyRegistry>` onto `ShaderDriver` as a first-class field (sourced
  once, not via `bindspace.ontology()`) so `ctx_id` does not fall to `unwrap_or(0)` at W7.
- **W5 — sub-gate:** define the mailbox-set builder (`N`, count, keying) before the bin move; do not
  invent the sea-star topology ad-hoc in `main()`.
- **W6 — external blocker** (surrealdb #41); design proceeds as docs/contract-only, NO code.
- **W7 — decoupled from W6.** Prereq is **zero singleton readers** + the §2e singleton-bound tests
  (`end_to_end.rs`; `busdto_bridge_test.rs` — **5** `BindSpace::zeros` sites) migrated to mailbox
  FIRST. Then delete `bindspace.rs` + the cycle plane + the feature flag in one PR.

---

## 7. #515 MaterializeProvenance interaction

`materialize_provenance` (now on `main`, `driver.rs`) is **provenance-only** — it consumes
already-derived observables (`free_energy`, `std_dev`, `top_resonance`, `awareness_skill`,
hits), reads **no** `BindSpace` directly → adds no new singleton reader and does not gate the
migration. The one interaction: its inputs come from the dispatch read path the W3+W4a atomic PR
re-points, so **W2 must assert `MaterializeProvenance` byte-equality** across stores (§6 W2 gate).
`ctx_id`/MUL read `ontology()` from a shared `Arc` — re-homed onto `ShaderDriver` at W4b (not the
mailbox), so deleting `BindSpace` at W7 does not strand the registry (baton-handoff P1).

---

## 8. Acceptance checks (per the operator's "test the new before delete the old")

- W2 differential: bit-identical `ShaderResonance` + `cycle_fingerprint` + `MaterializeProvenance`
  (one driver, two backing arms), same rows — run on the W3+W4a atomic PR.
- `BusDto` differential (W3): `unbind_busdto(mailbox)` vs `unbind_busdto(bindspace)` — headline +
  energies + cycle_count + converged parity; non-headline-index loss explicitly documented in the
  updated `busdto_bridge_test.rs` tolerance (D-DIST-5 exception).
- Field-isolation matrix (W3): each migrated column write leaves all others byte-unchanged; plus the
  emit-path `temporal` assertion (column-landed OR `temporal_dropped_under_v2`).
- **CausalEdge64 firewall (CI, literal lint not prose):** a `deny`/grep rule asserting
  `cognitive-shader-driver` imports **neither** `ndarray::hpc::causal_diff::CausalEdge64` **nor**
  `thinking_engine::layered::CausalEdge64` (the THIRD same-named type, in-graph via `with-engine`);
  + an `edges_raw()` reattach round-trip to the typed `causal_edge::CausalEdge64`.
- Cycle-drop (compile + bench): `MailboxSoA` exposes no `cycle*` symbol; add `MailboxSoA::byte_footprint()`
  + a one-row assertion mirroring `bindspace.rs:485` proving hot footprint ≈ 6 KB/row (not ≈ 71.6 KB/row).
- Feature path: a `--features mailbox-thoughtspace` CI matrix row (build + W2 differential), live from the W3+W4a PR.
- W7 readiness: grep proves zero `self.bindspace.*` reads under feature-on; the §2e tests (incl. all
  5 `busdto_bridge_test.rs` sites) pass on the mailbox path before deletion.
- W7 readiness: grep proves zero `self.bindspace.*` reads under feature-on; the two §2e tests pass
  on the mailbox path before deletion.
