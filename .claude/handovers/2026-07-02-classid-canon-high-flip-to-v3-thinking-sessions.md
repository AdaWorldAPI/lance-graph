# 2026-07-02 — classid canon-high flip SHIPPED → integration guide for every V3-substrate session

> From: the flip-execution session (lance-graph #626/#627/#628 arc).
> To: any session touching classids, V3 keys, SoA tenants, RBAC, rendering,
> or the thinking layer. APPEND-ONLY per handover protocol.

## What is true now (all merged 2026-07-02)

- **`CLASSID_ORDER = CanonHigh`** (lance-graph #628): a composed classid is
  `[hi u16: CANON concept/domain:appid][lo u16: CUSTOM marker/render-prefix]`.
  Stored `0x0701_1000` = human-readable `0x07:01::1000` (OSINT:q2, V3 marker).
- Mint constants: v1 `0x0700_0000` (OSINT) / `0x0A01_0000` (FMA) /
  `0x0100_0000` (PROJECT) / `0x0200_0000` (ERP); V3 `0x0701_1000` /
  `0x0A01_1000` / `0x0E01_1000` (OSINT + CPIC appid normalized `:01` = q2).
- **OGAR #147** flipped `ogar_vocab::app::{render_classid, app_of, concept_of}`
  in lockstep; `APP_PREFIX` VALUES unchanged (only the position moved to LOW).
  The OGAR#95 reconciliation resolved by construction: the app prefix IS the
  custom half; the #95 allocation table is the custom-half render catalogue.
- Consumers merged: q2 #71, openproject-nexgen #68 (+ trailing lock PR #69),
  MedCare-rs #180, woa-rs #177. Zero-impact verified: OGIT, tesseract-rs,
  openproject (Ruby).

## The five iron consumer rules (post-flip)

1. **Never local bit math.** Compose via `contract::render_classid` /
   `compose_classid`, or `ogar_vocab::app::*`. A `(x << 16) | y` on classids
   in a consumer is a defect (this is how three latent bugs shipped pre-flip).
2. **Discriminators:** `classid_canon(id)` (strict, new mints) or
   `classid_canon_compat(id)` (surfaces serving BOTH stored forms — RBAC
   grants, reads over un-re-baked corpora). FORBIDDEN on the composed u32:
   `as u16`, `& 0xFFFF`, `>> 8`, `>> 16` — post-flip each reads the wrong
   half or the wrong byte; grep-gate your crate for them.
3. **Mint-forward, never reinterpretation.** Persisted pre-flip ids
   (`0x0000_DDCC` / `0xAAAA_DDCC` / `0x1000_DDCC`) resolve forever via the
   `CLASSID_*_LEGACY` alias keys in `BUILTIN_READ_MODES` + the compat reader.
   Re-baking is optional hygiene, never a correctness prerequisite. Alias
   retirement is gated on a corpus proof (a scan showing zero old-form rows).
4. **The custom half is spoken for until P4.** Current tenants: the `0x1000`
   V3 marker (temporary by declaration), §2 app render prefixes
   (`0x0000..0x0007`), and interim kind slots (q2 cpic `0x0E01_000N`). The
   canon slot exactly `0x1000` (domain-0x10 root) is reserved-unusable until
   the marker retires. P4 (operator checkpoint) opens the half for the real
   64k dynamic ClassViews × bitmask render catalogue.
5. **Fix layout docs BEFORE their phase lands.** A stale doc comment
   describing the old order seeds the wrong implementation (the woa-rs
   Phase-3 near-miss). Conversely: a sweep's "all sites routed" claim needs
   a whole-crate `as u16|>> 16|& 0xFFFF|u32::from` grep, not a plan-inventory
   walk (`E-CLASSID-COMPAT-READER` — the P0 sweep missed rbac.rs).

## Thinking-migration to V3 — findings

- **F1 — the canon u16 is now the single cross-layer currency.** The same
  value is: the EntityType SoA tenant stamp, the RBAC `ClassGrant` key, the
  `ConceptDomain` route (`canon >> 8`), the codebook id, and the cross-app
  join key. Thinking layers should key on the u16 canon and treat the
  composed u32 as address-only (pure address; the magic is at the resolution
  target).
- **F2 — thinking styles are render lenses.** "Two renders, one concept"
  (OpenProject/Redmine) generalizes: post-P4 the custom half can carry a
  per-style ClassView selection — 36 thinking styles as 36 lenses over the
  SAME canon concept, dispatched exactly like app render skins. That is the
  natural landing zone for style-conditioned field-masks/templates without
  any new struct (AGI-as-SoA invariant holds).
- **F3 — cpic's kind slots are register-laziness.** Per the operator's
  mereology directive ("basins are genomic mereology, not labels") the six
  CPIC kinds want to be cascade POSITIONS (HEEL/family coordinates under one
  `0x0E01_1000` class), not classid customs. The interim `0x0E01_000N`
  scheme is documented as interim; the contract-pull re-mint should fold
  kind into position, which also frees the custom half for P4.
- **F4 — V3 nible paths are ready-made thinking metrics.**
  `from_guid_prefix_v3` + `family_hop_count` give O(depth), zero-value-decode
  graph distance on V3 keys — the natural adjacency for AriGraph/episodic
  tissue (`Think.graph` / `Think.episodic`) without touching value slabs.

## Suggested continuation (ordered)

1. **CI re-bakes** (q2): `cargo run -p osint-bake` + `--bin fma` in an
   environment with normal egress; commit refreshed `osint_scene.soa` /
   `fma.soa`; re-release `body.soa` then drop `FMA_V3_CLASSID_LEGACY` from
   `BodyV3.tsx`.
2. **Run the waived probes:** D-VCW-3 (q2 P7 render probe) + D-VCW-5
   (cascade3 nibble falsifier) — specs ready, q2 gate waived; they validate
   V3 keys end-to-end through the cockpit.
3. **Phase-2 tenant shaping** (`soa-value-tenant-migration-v2`): the thinking
   layer reads/writes tenant lanes via `classid_read_mode(c).value_schema`
   — CPIC gene expression as the coordinate VALUE on the Compressed schema;
   keep the field-isolation matrix mandatory for every new tenant.
4. **Wire F4:** AriGraph/episodic distance via V3 nible-path hop counts;
   CausalEdge64 provenance rows keyed by V3 addresses (RungElevator already
   feeds rung; the edge column is next).
5. **cpic contract pull with mereology (F3)** — dissolves the interim scheme
   and `ISS-Q2-CPIC-MIRROR`.
6. **Corpus-proof scanner:** a small utility that counts old-form classids in
   Lance datasets → the mechanical gate for legacy-alias retirement.
7. **P4 marker retirement (operator checkpoint)** → open the custom half for
   the render catalogue; then F2 (styles-as-lenses) becomes implementable.
8. **Remaining consumers:** ladybug-rs (still pre-V3/rustynum-era),
   smb-office-rs (consumer-preflight applies) — contract pulls only.

## Outstanding operator checkpoints

- **P4**: `0x1000` marker retirement / custom-half opening.
- **Alias retirement**: only after the corpus proof (step 6).
