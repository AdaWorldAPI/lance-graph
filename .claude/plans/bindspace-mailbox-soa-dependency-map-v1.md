# bindspace → mailbox_soa — dependency map & two-path wiring preflight (v1)

> **Status:** MAP / preflight. No source wired yet. Read-based (full reads of
> every consumer on the critical path; two Opus inventory agents for the rest).
> **Date:** 2026-06-17.
> **Operator constraints (binding):**
> 1. The BindSpace → `MailboxSoA<N>` replacement is a **given** (decided; not re-litigated here).
> 2. **Two parallel paths, step by step** — never delete the old until the new is tested.
>    ("better 2 paths step by step than one deleted and 'oops what did that do'.")
> 3. **Map dependencies before wiring/dedup.** Be careful.
> 4. **`CausalEdge64` is duplicated — handle it precisely, no handwaving.**
> **Parent plan:** `bindspace-singleton-to-mailbox-soa-v1.md` (S0–S5 staging; §8 OQ-1 RESOLVED).
> **Companion:** `soa-migration-diff-resolution-2026-06-13.md`.

---

## 0. The two stores, side by side (what migrates)

`BindSpace` (`crates/cognitive-shader-driver/src/bindspace.rs`) — the singleton, **still live**:

| column | type | per-row | → MailboxSoA destination | status |
|---|---|---|---|---|
| `fingerprints.content` | `Box<[u64]>` (256/row) | 2 KB | **own, dense, hot** (OQ-1 RESOLVED §2.7) | **GAP** |
| `fingerprints.topic` | `Box<[u64]>` (256/row) | 2 KB | own, dense, hot | **GAP** |
| `fingerprints.angle` | `Box<[u64]>` (256/row) | 2 KB | own, dense, hot | **GAP** |
| `fingerprints.cycle` | `Box<[f32]>` (16 384/row, `Vsa16kF32`) | **64 KB** | **DROP** — transient local, never a column | n/a |
| `fingerprints.sigma` | `Box<[u8]>` (1/row) | 1 B | own `[u8; N]` (Σ-codebook ref) | **GAP** |
| `edges` | `EdgeColumn(Box<[u64]>)` (**raw u64**) | 8 B | own `[CausalEdge64; N]` (typed) | **SHIPPED** |
| `qualia` | `QualiaI4Column` | 8 B | own `[QualiaI4_16D; N]` | **SHIPPED** |
| `meta` | `MetaColumn(Box<[u32]>)` | 4 B | own `[MetaWord; N]` | **SHIPPED** |
| `temporal` | `Box<[u64]>` | 8 B | own `[u64; N]` (OQ-2 fallback; v2 edge can't carry it) | **GAP** |
| `expert` | `Box<[u16]>` | 2 B | subsume into `mailbox_id`/`w_slot`, or `[u16; N]` | **GAP** |
| `entity_type` | `Box<[u16]>` | 2 B | own `[u16; N]` | **SHIPPED** |
| `ontology` | `Option<Arc<OntologyRegistry>>` | shared | **stays shared** (cold Zone-2, by `&`/`Arc`) | n/a |

`MailboxSoA<N>` (`crates/cognitive-shader-driver/src/mailbox_soa.rs`) — the target, **already shipped (D-MBX-A1)**:
`mailbox_id`, `energy: [f32;N]`, `plasticity_counter: [u8;N]`, `last_active_cycle: [u32;N]`,
`edges: [CausalEdge64;N]`, `qualia: [QualiaI4_16D;N]`, `meta: [MetaWord;N]`,
`entity_type: [u16;N]`, `current_cycle`, `w_slot`, `threshold`, `phase` (Rubicon).
Implements `MailboxSoaView` + `MailboxSoaOwner` (contract), with the `repr(transparent)`
`edges_raw()`/`meta_raw()` zero-copy casts (const-asserted).

**The D-MBX-A2 gap (what S1 still owes):** `content`/`topic`/`angle` (dense, hot — NOT a tiny
ref; OQ-1 resolved), `sigma`, `temporal`, `expert`. Note the content planes are **heap**
(`Box<[u64]>` of `N*256`, like BindSpace) — they cannot be `[u64; N]` stack arrays and
`[u64; N*256]` is not stable; design choice is a parallel `Box<[u64]>` field or a small
`FingerprintColumns`-shaped sub-struct owned by the mailbox.

---

## 1. The `CausalEdge64` duplication (precise — no handwaving)

**Two distinct types, same name, both `#[repr(transparent)]` over `u64`, both with `pub .0`:**

| | `causal_edge::edge::CausalEdge64` | `ndarray::hpc::causal_diff::CausalEdge64` |
|---|---|---|
| file | `crates/causal-edge/src/edge.rs:155` | `/home/user/ndarray/src/hpc/causal_diff.rs:151` |
| semantics | **SPO / thinking edge** — the EdgeColumn baton: S/P/O palette + NARS⟨f,c⟩ + Pearl 2³ + plasticity + (v2) signed inference mantissa + 6-bit W-slot | **weight-diff codec** — which transformer (block, projection) row shifted how far (L1) between two model checkpoints + verb |
| `pack` | 10 scalars | one `&WeightEdge` |
| `frequency`/`confidence` | `/255` | `/1023` |
| `truth()` | `TrustTexture` (2-bit, v2) | `NarsTruth` (f32 pair) |
| `w_slot` / `inference_mantissa` | **yes** | **no** |
| imported into lance-graph? | **yes, everywhere** | **NEVER** (0 imports; only a doc cross-ref at `edge.rs:154`) |

**The hazard:** both are bare `u64` newtypes with public `.0` and overlapping method *names*
carrying *incompatible* semantics. A raw `u64` baton packed by one codec and unpacked by the
other round-trips with **zero compile error and zero runtime signal** — same silent-corruption
class as `I-LEGACY-API-FEATURE-GATED`. Today there is **no** dual-import site; the only vector
is `u64`-level.

**The firewall (already in place — keep it):**
- The contract carries the edge column as **raw `u64`**: `MailboxSoaView::edges_raw() -> &[u64]`
  (`soa_view.rs:46`, "kept raw so the contract stays zero-dep — `causal-edge` is not a contract
  dep").
- Typed `causal_edge::CausalEdge64` is reattached **only at two trusted boundaries**: the hot
  owner `MailboxSoA<N>` (via the const-asserted `repr(transparent)` cast in `edges_raw()`,
  `mailbox_soa.rs:377`) and the conversion template `p64-bridge` (`lib.rs:19`, typed).
- The cold view (`surreal_container::SurrealMailboxView`) and `LanceVersionScheduler` stay on
  the raw `u64` path and never name the type.

**Safe rule (lock for the migration):** the *only* `CausalEdge64` that may be
`CausalEdge64(raw)`-reconstructed from an `edges_raw()` slice is
`causal_edge::edge::CausalEdge64`; the ndarray twin is **barred** from the mailbox/baton path.
Do not `use ndarray::hpc::causal_diff::CausalEdge64` anywhere in `cognitive-shader-driver`. The
contract's raw-`u64` edge column is the dedup firewall — it keeps the contract zero-dep and
confines the typed reattach to one trusted crate boundary.

**v1/v2 layout note (relevant to `temporal`):** the protocol edge has a feature-gated layout
(`causal-edge-v2-layout`, **default-on** since 0.2.0). v2 **drops `temporal`** (bits 52-63
reclaimed for signed mantissa / plasticity-shift / W-slot / truth-lens / spare; `set_temporal`
is a no-op). ⇒ the BindSpace `temporal` column **cannot** fold into the v2 edge — the `[u64; N]`
fallback (OQ-2) is the correct destination. Any code touching `temporal` must obey
`I-LEGACY-API-FEATURE-GATED` (no v1 temporal accessor under v2).

---

## 2. Consumer dependency map (who touches what)

### 2a. The column hot-spots — `engine_bridge.rs` + `driver.rs` (the bulk)
- **`driver.rs` dispatch hot path** reads the singleton per cycle: `self.bindspace.fingerprints.content_row(row)` (resonance/Hamming search — the heaviest read), `self.bindspace.edges.get(row)` → `CausalEdge64(raw)`, `self.bindspace.meta.get`, `self.bindspace.qualia.row`, `self.bindspace.entity_type`, `self.bindspace.ontology()`.
- **`engine_bridge.rs`** = the re-encode membrane (S2 dissolve target): `ingest_codebook_indices(&mut BindSpace)` (writes content/meta/temporal), `dispatch_busdto`/`unbind_busdto` (with-engine; cycle/qualia/meta/expert), `persist_cycle(&mut BindSpace)` (cycle/edges/meta), `write_qualia_observed`/`read_qualia_decomposed` (qualia).

### 2b. Other BindSpace consumers (driver crate)
- `serve.rs` — `bs.fingerprints.set_content` in `encode_handler`; `qualia` read via `read_qualia_decomposed`; `bs.len`.
- `grpc.rs` — `ingest_codebook_indices`; `qualia` read; `bs.len`.
- **Everything else is comment-only or zero-coupling:** `wire.rs`, `sigma_rosetta.rs`, `cypher_bridge.rs`, `auto_style.rs` (test-only `QUALIA_DIMS`), `codec_*`, `decode_kernel.rs`, `rotation_kernel.rs`, `token_agreement.rs`, `planner_bridge.rs`.
- ⚠️ **`attention_mask.rs` / `attention_mask_actor.rs` define their OWN `AttentionMaskSoA`** — share only the `MailboxId`/`w_slot` vocabulary. **Do NOT conflate** with `MailboxSoA<N>`.

### 2c. Construction (allocation) sites — S3 must change
- `bin/grpc.rs:31` — `Arc::new(BindSpace::zeros(4096))`
- `bin/serve.rs:31` — `Arc::new(BindSpace::zeros(4096))`
- (`BindSpaceBuilder` is used only inside `bindspace.rs` tests.)

### 2d. The structural ownership hazard — `Arc::get_mut`
Four sites assume single ownership of the singleton and **break under per-mailbox ownership**:
`grpc.rs:136`, `serve.rs:150`, `serve.rs:601`, `serve.rs:692` (`Arc::get_mut(&mut …bindspace)`
→ `&mut BindSpace`). Each errors if the `Arc` has >1 ref. These are the write escape hatches
the mailbox model replaces with owned `&mut MailboxSoA`.

### 2e. Tests bound to the singleton (must keep green through both paths)
- `tests/end_to_end.rs` — `BindSpace::zeros`, `ingest_codebook_indices`, `write_qualia_17d`, `read_qualia_decomposed`, `bindspace(Arc<BindSpace>)`, `persist_cycle`, `bs.meta.get`.
- `tests/busdto_bridge_test.rs` (with-engine) — `BindSpace::zeros` ×4, `dispatch_busdto`/`unbind_busdto`, `bs.meta.get`.

### 2f. The cold side is ALREADY built (the second path's far end)
- `surreal_container::SurrealMailboxView` (`view.rs:159`) **already implements `MailboxSoaView`** (read-only; deliberately NOT `MailboxSoaOwner`). Reads `energy: &[f32]`, **`edges_raw: &[u64]`** (raw path — never names `CausalEdge64`), `meta_raw: &[u32]`, `entity_type: &[u16]` + scalars. The kv-lance projection (`read_via_kv_lance`) is a stub (`BlockedColdBuild`) until the surrealdb fork dep lands; the trait surface + `from_columns` are complete.
- `lance-graph::graph::scheduler::LanceVersionScheduler` is **generic over `MailboxSoaView`** (OUT-direction; "propose, not dispose"); never names `CausalEdge64` or `BindSpace`.
- `p64-bridge` maps **typed** `causal_edge::CausalEdge64` → palette (storage-ward template).

### 2g. The two-path scaffold is ALREADY in `driver.rs`
`ShaderDriver` holds **both**: `bindspace: Arc<BindSpace>` (live) **and**
`mailboxes: HashMap<MailboxId, MailboxSoA<1024>>` (`driver.rs:88`, "transitional per-mailbox
routing surface (slice A2)… purely additive… Removed at cutover (plan S3)"), with `with_mailbox`
builder + `mailbox(id)` accessor. **Dispatch still reads the singleton** — the mailboxes are
populated but not yet consumed by the hot path. This is exactly the "two paths" cradle.

---

## 3. Two-path, step-by-step wiring sequence (delete-last)

Each step keeps **both paths live** and adds tests for the new before removing anything.

- **W0 (this doc).** Map ratified. — *DONE.*
- **W1 — D-MBX-A2 small columns (additive, tested). DONE.** Added `temporal: [u64;N]`,
  `expert: [u16;N]`, `sigma: [u8;N]` to `MailboxSoA<N>` + accessors + `reset_row` +
  `test_mailbox_soa_column_parity_with_bindspace` (writes matched per-row values to a
  `BindSpace` window and a `MailboxSoA`, asserts `edges`/`qualia`/`meta`/`entity_type` +
  `temporal`/`expert`/`sigma` read back identically). 13 mailbox_soa tests green, clippy clean.
  Deletes nothing.
- **W1b — D-MBX-A2 dense identity planes (additive, tested).** Add `content`/`topic`/`angle`
  (dense, hot, heap `Box<[u64]>` of `N*256`, mirroring `FingerprintColumns` minus `cycle`) to
  `MailboxSoA<N>` + a zero-copy `content_row(row)->&[u64]` accessor + parity test vs
  `BindSpace.fingerprints`. **Design note:** the mailbox is otherwise all-stack `[T;N]`; the
  planes MUST be heap (`N*256` u64 ≈ 2 MB at N=1024 cannot be stack, and `[u64; N*256]` is not
  stable) — own them as `Box<[u64]>` fields or a small `MailboxFingerprints` sub-struct. The
  `cycle` (`Vsa16kF32`) plane is NEVER added (OQ-1/§2.7). This is the hot-path-critical column
  (`driver.rs` resonance read = `content_row`), so it gets its own focused step.
- **W2 — read-parity harness on the hot path.** Add a *read shim* so a `MailboxSoaView` can
  serve the columns `driver.rs` reads (content_row, edge, meta, qualia, entity_type). Run the
  dispatch resonance read against BOTH the singleton and a mailbox built from the same rows;
  assert identical hits/resonance. (Differential test — proves the new before any swap.)
- **W3 — `engine_bridge` onto mailbox rows (feature `mailbox-thoughtspace`).** Re-point
  `ingest_codebook_indices`/`persist_cycle`/`write_qualia_*` to write a `MailboxSoA` row behind
  the feature; v1 `&mut BindSpace` path stays as the default. `cycle` becomes a transient local.
  Field-isolation matrix tests on the temporal/expert boundary (`I-LEGACY-API-FEATURE-GATED`).
- **W4 — driver dispatch reads the mailbox (feature-gated).** Behind the feature, the hot path
  reads `mailboxes` instead of `bindspace`; the four `Arc::get_mut` write hatches become owned
  `&mut MailboxSoA`. Both bins still allocate the singleton when the feature is off.
- **W5 — bins stop allocating the singleton.** `bin/{serve,grpc}.rs` build a mailbox set
  (sea-star) instead of `BindSpace::zeros(4096)`, under the feature.
- **W6 — death → SPO-G + Lance tombstone-witness** (gated on `surreal_container` unblock OR the
  `lance-graph-callcenter` path). The cold `SurrealMailboxView` is the read end.
- **W7 — delete `BindSpace` + the `cycle` plane; remove the feature gate.** Only after W1–W6 are
  green and the singleton has no remaining readers (the two tests in 2e migrated to mailbox).

**Guardrails baked into the order:** the new path is *tested against the old* at W1 (parity) and
W2 (differential) **before** any behaviour swaps at W3/W4; the singleton is removed **last** (W7),
not first. CausalEdge64 stays typed only at the two trusted boundaries; the ndarray twin never
enters. `attention_mask*` SoA is never touched. The content planes stay hot (OQ-1).

---

## 4. Dedup hazards checklist (carry into every wiring PR)

1. **CausalEdge64:** never `use ndarray::hpc::causal_diff::CausalEdge64` in the driver; the
   mailbox edge is `causal_edge::edge::CausalEdge64`; cross-boundary stays raw `u64`.
2. **`temporal` under v2:** no v1 temporal accessor; `[u64; N]` column is the home (v2 edge
   dropped temporal). `I-LEGACY-API-FEATURE-GATED` field-isolation tests mandatory.
3. **`Arc::get_mut` (4 sites):** each is a single-owner assumption that must become owned
   `&mut MailboxSoA` — they are the precise spots that "oops" if the store is shared.
4. **`attention_mask*`:** independent `AttentionMaskSoA`; do not fold into the migration.
5. **`cycle` plane:** never migrate; compute transiently. Dropping it is what makes the 64k–256k
   hot working set fit (§2.7).
6. **Content planes stay hot:** `content`/`topic`/`angle` own dense per-row in the mailbox
   (~6 KB/thought), not a 6 B ref (OQ-1 resolved). The CAM-PQ/codebook ref is the *cold* form.
7. **Singleton deleted LAST (W7):** the two singleton-bound tests (2e) migrate before deletion.
