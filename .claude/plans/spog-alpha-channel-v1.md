# spog-alpha-channel-v1 — the SPOG alpha channel in MedCare-rs: domain is a mask, the cycle is a sealed batch, the rung is read from the stamp

> **Status:** SPEC (Phase 0), 2026-09-07. Register-before-code. Every "exists"
> claim below was read this session (four Sonnet inventories, orchestrator-
> verified where cited; banked outside the public repo because they quote a
> private consumer). Every "absent" claim names the search that backs it.
>
> **Operator mandate (2026-09-07, verbatim):** *"probe autoattended autonomous
> decision making until you get MedCare-rs SPOG alpha channel to work / keep in
> mind that rows are experimental in lance 11 and only required for tombstones
> which we avoid by having sealed batch per cycle / also keep in mind that the
> relevant bakes in S3 might not be per domain separate (palpitations hpo,
> heart uberon/FMA, heart attack (mondo disease), nitroglycerin (UMCU), triage,
> bypass (snomed actions, interventions) heart rate (Loinc), chebi, mesh
> (research) statistical normalization (cob, iobc) dismech"* — plus *"also
> check Mississippi queen hexagon board game effect vs masking algebra ternlogq
> chaining amortization / same bit that masks mq might 'mask' SPO 'angle' akin
> to multidisciplinary blasgraph"*.
>
> **READ BY:** anyone touching `contract::spog_tenants`, `contract::alpha`,
> `contract::alpha_tunnel`, `contract::wave_dispatch`, MedCare-rs
> `medcare-nodesoa` / `medcare-first-thought::attention` / `medcare-cohorts::
> {quad_slab,rails,bake_data}`, or citing "SPOG", "alpha channel", "rung ×
> tenant", "per-domain bake", "sealed batch per cycle".
>
> **Standing rulings this spec rests on:** `E-EVERYTHING-WIRES-TO-SOA-V3-CE64-
> IS-ALU-LEGACY-1` (R2); `E-PLANNING-MIGRATES-TO-LOCO-R2IL-DATAFUSION-IS-GRACE-
> PERIOD-1`; the alpha-overlay governing choice (`alpha-channel-rung-overlay-
> v1.md` §3k: *"explizite masking ABI traversal wie bei java … sonst verwässern
> wir unsere Architektur"*); the temporal plan (`.claude/temporal/09-plan.md`
> Stage 2: the rung × tenant cross lives ONE crate out of the zero-dep
> contract); `ndarray/.claude/rules/data-flow.md` (no `&mut self` during
> computation — the alpha `claim` sites are the operator's standing exception,
> recorded in temporal 06, not re-litigated here).

## §0 — What "the SPOG alpha channel works" means, measurably

The alpha channel is `AlphaAllocation` → `AlphaOverlay::claim` → 16-byte
`AlphaStamp` in value slot 0 (`crates/lance-graph-contract/src/alpha.rs:72-88`).
SPOG adds the fourth coordinate G — *"No fourth column: G is read from the
key"* — as the canon-high concept half of the classid, `graph_of`
(`crates/lance-graph-contract/src/spog_tenants.rs:38-40`, body
`(addr.classid() >> 16) as u16`), and routes each claim to the tenant owning
that G in `SpogTenants::claim` (`spog_tenants.rs:85-94`). Today
`SpogTenants` has **zero consumers** anywhere (grep `SpogTenants` over
`lance-graph/crates` and `medcare-rs/crates`: hits only in its own file and
tests). The consumer that exists calls the lower entry point instead:
`dispatch_thought(base, &seed, &keys, cycle)` at MedCare-rs
`crates/medcare-nodesoa/src/frontier_dispatch.rs:81`, over the 60,478-row
`obo-core.soa` spine, with `cycle` pinned to the constant `SHADOW_CYCLE = 1`
(`patient_shadow.rs`) and the scanpath read only by the debug HTML view
(`medcare-server/src/views/reasoning_debugger.rs:877-903`).

"Works" is therefore five measured facts, each with a falsifier (§6):

1. **Routed.** A real patient frontier's claims land in per-G tenants over the
   full 762,041-row `all-lanes.soa` — every claim is `TenantClaim::Routed`, or
   is `NoTenant(g)` for a G the caller did not declare (never silently absorbed).
2. **Domain is a mask.** A domain view is `OR` over its G tenants' masks
   (disease = MONDO ∪ ICD-10-GM ∪ Orphanet ∪ …), never a file boundary; a
   horseshoe lane (CUI, SNOMED) is fenced per row by a TUI-equality mask over
   `value[0..2]`, never assigned to a domain wholesale.
3. **Rung × tenant is observable.** The 10 × N `AlphaMask` matrix exists as a
   computation (ternlog on the masks' words), and "this domain was addressable
   and no rung looked" is one read.
4. **Sealed batch per cycle.** One `FixedSizeBinary(512)` append per cycle,
   `cycle = dataset.version() + 1`; no stable row ids, no delete, no merge.
5. **The rung byte is the attention rung.** `AlphaStamp.rung` is written by
   exactly one writer semantics (the ladder step), and the domain reflection
   MedCare reads today through `domain_rung` becomes a tenant read.

## §1 — Input inventory (verified this session)

### 1a. The contract surface, verbatim signatures

| symbol | file:line | shape |
|---|---|---|
| `AlphaMask { words: Box<[u64]>, len: u32 }` | `alpha.rs:224-230` | private fields; `and/or/xor/and_not/not` via `zip` with a **release-mode** `assert_eq!(self.len, other.len)` (`:289`); `materialize_ordinals` the one named materializer (`:345`) |
| `AlphaAllocation::over(&[NodeRow])`, `ordinal`, `mask_of` | `alpha.rs:377,394,417` | ordinal = base position, lazily indexed |
| `AlphaOverlay::{over_shared,new,claim,claim_path,attended_mask,scanpath,rows}` | `alpha.rs:498-661` | `claim(&mut self, addr, rung) -> Result<AlphaClaim, AlphaError>` |
| `graph_of`, `SpogTenants::{over,claim,tenant,concepts,claimed_len,merge}`, `TenantClaim::{Routed,NoTenant,Substrate}` | `spog_tenants.rs:38,74,85,98,107,113,123,44-52` | tenants = `Vec<(u16, AlphaOverlay)>`, linear `find` per claim |
| `AlphaTunnel::{over,lane,lane_mut,run_wave,run_wave_parallel,merge,merged_rows}` | `alpha_tunnel.rs:82-223` | `merge` is `(rung, seq)`-ordered with NO sort (`debug_assert!` at `:197-200`) |
| `dispatch_thought(base, seed, keys, cycle) -> WaveDispatchOutcome{scanpath, waves}` | `wave_dispatch.rs:62-67` | constructs allocation + tunnel internally; `seed.clone()` per lane is RULED intentional (temporal 06) |
| `RowFocusMask` (D-ACR-1) | `attention_facet.rs:370-455` | a **facet-prefix** set (`AttentionFocusFacet`, `covers`/`common_prefix`), NOT a row bitmask — a different object from `AlphaMask`; both stay |

**Absent, by search:** no `AlphaMask::words()` accessor (grep `fn words|as_words|from_words` in `alpha.rs`: 0 hits) — the words are unreachable from any crate, so nothing outside the contract can run a SIMD op on them. No `mask_ternlog`/`AND3` call anywhere in `lance-graph-java/native/lgj-abi/src/*.rs` (grep `ternlog|AND3`: 0 hits at lgj `dbac826`); `lgj_hop` is two sequential `simd_mask_and_assign` (`exports.rs:1818,1822`). **This falsifies two standing claims** — `EPIPHANIES.md` E-NXG-8 *"`AND3` = conjunctive narrowing (`lgj_hop`, shipped)"* and `.claude/knowledge/membrane-tiers.md` §"The polyfill is the worked instance" *"`exports.rs` names `kernels::ternlog::AND3`"* — corrected on the board with this spec.

### 1b. The consumer surface (MedCare-rs, private — quoted minimally)

| fact | where |
|---|---|
| ONE production caller of the alpha channel: `dispatch_thought(base, &seed, &keys, cycle)` | `medcare-nodesoa/src/frontier_dispatch.rs:81`; `base = obo_store::store().node_rows()` (60,478 rows) |
| `cycle` is the constant `SHADOW_CYCLE = 1` — *"Fest, damit zwei Requests … byte-gleich sind"*; there is no cycle loop anywhere (temporal 06: `git grep` for `cycle + 1|cycle++|for cycle in` → 0 hits) | `medcare-nodesoa/src/patient_shadow.rs` |
| The second `AlphaStamp.rung` writer: `domain_rung(classid) = Domain::of_classid(..) as u8 + 1` (1..=8) at the claim site `self.overlay.borrow_mut().claim(*addr, rung)`; `reflection(ov, domain)` filters the scanpath on `stamp_of(r).rung == domain as u8 + 1` | `medcare-first-thought/src/attention.rs:127,153,221` |
| Persist path: `overlay_to_batch` / `write_alpha_overlay` → `node_rows_to_batch(rows, cycle)` → one column `node: FixedSizeBinary(512)` NOT NULL, append | `medcare-nodesoa/src/alpha.rs:26,77`, `lib.rs:46-56` — both functions have **zero callers outside their own tests** |
| Domain grouping: `Domain::of_classid` → `FacetRegime::{Single(d), PerRowTui, Unassigned}`; `Domain::of_row` resolves `PerRowTui` via `cui::tui_of_row(row)` (`value[0..2]`) | `medcare-cohorts/src/quad_tenant.rs`; `cui.rs:123` |
| Row-resident FK register: the 4×u24 identity quad at value `[32,44)`; `quad_slab::project(rows, domain, from, to)` is the scalar reference — filters `Domain::of_row(r) == Some(domain)` FIRST (codex P1 on #410), then compares slot values | `quad_slab.rs:63,78` |
| Per-domain rail bakes: `RailStore { bakes: BTreeMap<u32, (RailBake, RailBake)> }` keyed by classid, grouped `je_domaene` before baking — *"every domain needs a separate bake, the nodes dilute"* (operator 2026-08-12; 180,107 false cross-domain ancestries measured otherwise) | `medcare-cohorts/src/rails.rs:928,965-975` |
| Combined artifacts: `obo-core.soa` = 5 classids in one file (`obo-core-combined-bake-confounds-facets`, RAIL_OFFENE_POSTEN); `all-lanes.soa` = **every lane** node-matched, 762,041 rows, classid-major numeric sort, `soahead:762041@0x03040000`; mmap via `bake_data::soa_map()` / `node_rows_for_classid()` (partition_point over the sorted classids) | `data/config/bakes.tsv`; `bake_data.rs:307,356` |
| Live classids are the domain form for 7 lanes (MONDO `0x9101_0000`, HP `0x9202_0000`, UBERON `0x9303_0000`, LOINC `0x9407_0000`, … OPS `0x9811`), legacy `0x03xx` for PATO/RO/CUI/Orphanet/ATC/RxNorm; readers fold via `domain_block` | `crosswalk.rs:173-177`, `loinc.rs:36`, `cui.rs:95`, `ontology_map.tsv` |

**The operator's hint is confirmed, not assumed:** the bakes in S3 are NOT
per-domain files. `all-lanes.soa` is one image for twelve lanes, `obo-core.soa`
one image for five. Per-domain separation exists today only in `RailStore`'s
in-memory grouping. So "domain" must be a **mask over the combined image**,
and this spec makes it exactly that (§3).

### 1c. The kernels (ndarray T1, verbatim)

`eq_u32_to_mask(values, needle, out)` `simd_int_ops.rs:562`;
`eq_u32_strided_to_mask(bytes, first_offset, stride_bytes, count, needle, out)`
`:629`; `mask_and/or/andnot(_assign)` `:775-932`;
`mask_ternlog::<IMM>(a, b, c, dst)` `:983`; `mask_ternlog_assign::<IMM>(a, b, c)`
`:1015`; the eight named immediates `simd.rs:570-587` (`AND3 0x80`,
`AND2_ANDNOT 0x40`, `AND_ANDNOT2 0x10`, `OR2_AND 0xA8`, `XOR3 0x96`, `MAJ3 0xE8`,
`AND2 0xC0`, `OR3 0xFE`). Convention: `index = (a<<2)|(b<<1)|c`, result bit =
`(IMM >> index) & 1`. Existing in-repo consumer of the named immediates:
`lance-graph-planner/examples/probe_nxg_hist_1.rs:51-136`.

Measured bounds that ride with every speed claim below (temporal 09 P3,
2026-09-06, `--release`): chaining amortization holds (T3/T1 → 0.50 at K ≥ 8,
K = 1 control reads 1.03), **contingent on the mask set fitting L2**; mask
loses to a sparse arm below 0.1 % active. A 762,041-bit mask is 11,907 words =
**93 KiB**; ten rungs × one tenant = 0.93 MiB (fits a 2 MiB L2); the full
10 × 15 matrix does not, and never needs to be resident at once (§3.3).

## §2 — Frozen decisions (each cited; none re-opened below)

| # | decision | why / source |
|---|---|---|
| F1 | **No Lance row ids, no delete, no tombstone.** The alpha channel APPENDS one sealed batch per cycle; addressing is `(version, NodeGuid)`. `enable_stable_row_ids` stays OFF everywhere. | operator 2026-09-07; temporal 01/05 (all three delta arms need stable ids — measured D-LNC-5a); alpha never deletes |
| F2 | **Domain = mask over the combined image**, never a per-domain file. Tenant mask for G = `eq_u32_strided_to_mask(bytes, 0, 512, n_rows, classid, out)` over the mmap; domain view = `OR` over its Gs; horseshoe lanes fenced by a `value[0..2]` TUI-equality mask. | §1b; operator hint; `RailStore`'s grouping is the in-memory precedent |
| F3 | **G is the contract's `graph_of` (canon-high u16)**, one tenant per G. MedCare's coarser `Domain` (byte `0x91..0x9D`, `domain_block.rs`) is a GROUPING of Gs, expressed as mask `OR` — no second G reading is minted, and no bit math on a composed classid appears in consumer code (`Domain::of_classid` already answers it). | `spog_tenants.rs:38-40`; worker rule 4 |
| F4 | **The rung byte carries the attention rung only.** `domain_rung` (Domain+1) is retired as a rung writer once the tenant read replaces `reflection` — the domain is `graph_of(addr)`, read from the key, never from the stamp. Until D-SPG-5 lands, the two writers stay separate overlays (they do today). Trap: never carry a rung ordinal in the residue band (temporal 09 Stage 2). | D-RLR-5 (a); temporal 06 "unrecorded semantic collision" |
| F5 | **Placement:** the SIMD cross cannot live in the zero-dep contract. The contract gains ONE method (`AlphaMask::words(&self) -> &[u64]`, plus the paired `from_words` constructor guarded by the same length law) — a method on the carrier, not a type. The live cross runs in MedCare (`medcare-cohorts` already depends on `ndarray`; the BBB rule keeps `lance-graph-planner` out of the customer binary). An agnostic synthetic probe may live in `lance-graph-planner/examples/`. | temporal 09 Stage 2; CLAUDE.md litmus (method on carrier) |
| F6 | **Cycle = Lance version.** `cycle = dataset.version() + 1` at seal time; `SHADOW_CYCLE = 1` stays for the byte-identical debug view (it is a different consumer). | operator "sealed batch per cycle"; temporal 06 "the cycle loop is absent" |
| F7 | **Explicit mask ABI traversal, never VSA.** Nothing here bundles; `I-VSA-IDENTITIES`' niche is untouched. | alpha-overlay plan §3k (operator, 2026-08-21) |
| F8 | **DataFusion is not extended.** Step 5 (containment: `with_row_id`/`with_row_addr` OFF + a test) comes AFTER the tenant masks exist, or the flags are the only identity the consumer has. | IDEAS 2026-09-07 containment card; grace-period ruling |
| F9 | **The hand-rolled MedCare alpha migrates ONTO the #1198 contract alpha, not beside it.** Operator, 2026-09-07 (verbatim): *"make sure to migrate the handrolled MedCare-rs alpha to LG 1198 alpha"*. "LG 1198 alpha" = the contract path as audited and staged in lance-graph #1198 (`.claude/temporal/`, merged `3797237b`; Stage 1b landed in the successor PR): `AlphaAllocation` → `AlphaTunnel` rung lanes → `SpogTenants` G routing → `merge()` in `(rung, seq)` order → `wave_dispatch`. The hand-rolled surfaces are MedCare's own overlay drivers that bypass that path: `attention::WatchedRows { overlay: RefCell<AlphaOverlay> }` with its `domain_rung` writer and `into_overlay`, `backreference::combined_base` (a second base assembled per patient), and the `medcare-nodesoa::alpha` writer that takes a bare `AlphaOverlay`. Each becomes a consumer of the contract path (D-SPG-5, broadened) — no MedCare-local overlay driver survives the migration, and behaviour is preserved by the set-equality falsifier. | operator 2026-09-07; PR_ARC_INVENTORY 2026-09-06 (#1198); temporal 06 ("the second rung writer") |

## §3 — The design, in the order the operator gave (1 probe → 2 masks → 3 lane-local → 4 r2il → 5 containment)

### 3.1 Tenant masks are the per-G bakes (step 2, the unblock)

Over `all-lanes.soa` mmapped (`bake_data::soa_map()`), for every declared G:

```text
tenant_mask[G] = eq_u32_strided_to_mask(bytes, 0, 512, n_rows, classid_of(G), out)
domain_mask[D] = OR_{G ∈ D} tenant_mask[G]                   // disease = MONDO ∪ ICD-10-GM ∪ Orphanet ∪ OMIM…
horseshoe[D]   = tenant_mask[CUI] ∧ OR_{tui ∈ tui_domains(D)} eq_u16(value[0..2] == tui)
```

Computed ONCE per Lance version (the mask generation), served to every rung —
the Mississippi-Queen M1b amortization stated as a cache key
`(generation, G)` (§4). `SpogTenants::over(alloc, cycle, &concepts)` is then
declared with exactly the Gs that have a non-empty tenant mask; a claim to any
other G is `NoTenant(g)` — visible, not absorbed. The `AlphaAllocation` is
`AlphaAllocation::over(node_rows)` over the whole image, so ordinals are image
positions and every mask in this spec shares one `len`.

### 3.2 The crosswalk is a chain of masked equality sweeps (step 1's subject)

Row-resident FKs only — a sidecar `HashMap` join (`cui::mondo_to_cui`) is the
scalar REFERENCE, never the mechanism. Hop n: `eq_u32_strided_to_mask` on the
FK column of tenant n's rows for each needle of the incoming survivor key set,
`OR`-accumulated, then `mask_ternlog::<AND3>(sweep, tenant_mask[n], rung_gate)`
— the survivors' key set is the needle set of hop n+1. The forbidden move is a
mask-`AND` across two tables (nexgen room 18; `E-…-CHAIN-OF-MASKS` on the
board). Which FK columns are u32-aligned in the real image (key tail at byte
12; quad slots are u24 at value 32..44 and are NOT eq_u32-addressable without
a masked compare) is pinned by the probe's own W0 read, not guessed here —
see D-SPG-4's pre-registration rule.

### 3.3 The rung × tenant cross (step 2's meta-awareness layer, temporal Stage 2)

```text
cell[rung r][G] = mask_ternlog::<AND2>(lane_r.attended_mask().words(), tenant_mask[G], _)
unlooked[D]     = mask_ternlog::<AND_ANDNOT2>(domain_mask[D], any_rung, any_rung)
```

No new stored state: both operands are recomputed projections; the cross is
computed per read for the (r, G) pairs asked, so at most three 93 KiB masks
are live per op (fits L2 — the P3 bound). Kill condition (temporal 09): if
P3-style amortization does not show on THIS shape, build it scalar and say so;
the shape win stands without the speed win.

### 3.4 Sealed batch per cycle (step 3's write side)

`SpogTenants::merge()` → the merged `NodeRow`s (stamp in value slot 0, key
unchanged) → `node_rows_to_batch(rows, cycle)` → ONE `write_node_soa_dataset`
append. `cycle = dataset.version() + 1`, read before the append, asserted equal
to the committed version after. Time travel = read at version; no row identity
is ever needed because the key IS the identity (P0 canon).

### 3.5 Steps 4 and 5, queued behind the probe

Step 4: the first `ogar-r2il` consumer through `lance-graph-ogar` = `RANK` to
admit + `TERNLOG 0x86` per hop, with the falsifier that a lifted crosswalk
program yields the hand-written chain's survivor mask bit-for-bit (IDEAS
2026-09-07). Step 5: DataFusion containment (F8). Neither starts before D-SPG-4
is green.

## §4 — Mississippi Queen, ternlog chaining, and the SPO "angle" (the operator's second check)

Source: `ndarray/.claude/plans/gemm-ternlog-mask-consolidation-v1.md` §9/§11
(M1 reveal-ahead [G], M1b tile-serves-every-boat [G] = the amortization with
cache key `(mask generation, panel)`, M2 lookahead [H], M3 coal budget [H], R1
hexagon [H]; §11.5 `TriadicProjection {Abc, AbAskC, AcAskB, BcAskA, AOnly,
BOnly, COnly, Background}` = K0..K7 as a ternlog immediate indexed
`(a<<2)|(b<<1)|c`, graded [H] *pending one operator word*).

**The mapping, stated as something that can fail.** `mask_ternlog::<IMM>(S, P,
O)` computes per row `IMM[(s<<2)|(p<<1)|o]`. So the immediate's eight bits ARE
the eight K-projections of one quad row (which of S/P/O are present), and the
six ways of wiring the S/P/O presence columns onto the kernel's A/B/C inputs
are the six hex directions — the "angle". A crosswalk hop's immediate, read as
a K-set, is the hop's declared projection; `AND3` is K7 alone (all three
present), `AND_ANDNOT2` is K4 (A only). The operator's interjection — *"same
bit that masks mq might 'mask' SPO 'angle'"* — is the claim that the mask bit
and the projection bit are one bit. It is true by construction of the
immediate's index; whether it is USEFUL is what gate (f) measures: the eight
minterm masks must partition the row population, and a permutation of the
wiring must permute the immediate's bits without changing any set.

**Reveal-ahead = mask generation.** M1's "reveal the tile ahead of the cursor"
is the tenant masks being computed once per Lance version, before any rung
reads (§3.1); M3's coal budget is the per-cycle re-chain budget — how many
hops a rung may run before the next seal. Both are cache-key statements, not
new mechanism. D-GTM-0l's own next probe (*"re-run the identical instrument
against an OGAR-minted address space"*) IS this spec's probe: `all-lanes.soa`
keys are minted from the concept hierarchy, which is the substrate the prefix-
routing hypothesis was proposed for and never measured on.

**lgj correction carried here:** the fused `AND3` hop is an OPPORTUNITY, not
shipped code — `lgj_hop` does two ANDs (`exports.rs:1818,1822`). Whether the
fusion pays on the hop's shape is gate (c) below; nothing in this spec assumes
it does.

## §5 — Deliverables

| D-id | scope | repo | gate / falsifier |
|---|---|---|---|
| **D-SPG-0** | This spec; the lgj `AND3` correction on the board (E-NXG-8 regraded, `.claude/knowledge/membrane-tiers.md` §"The polyfill is the worked instance" corrected in place); IDEAS PROBE-CROSSWALK-MASK-1 card → In progress | lance-graph | citation-decay + append-only gates green |
| **D-SPG-1** | **SHIPPED 2026-09-07** (`alpha.rs`, +78 lines: two methods, three tests). `AlphaMask::words(&self) -> &[u64]` + `AlphaMask::from_words(words: Box<[u64]>, len: u32) -> Self` (same tail-clearing law as `not()`; a `words.len() != len.div_ceil(64)` input is REFUSED, release-mode). No other contract change. | lance-graph | can-fire: `from_words` with a wrong word count panics; can-stay-silent: round-trip `from_words(m.words().into(), m.len()) == m` for `len % 64 != 0`; existing 10 alpha tests untouched |
| **D-SPG-2** | Tenant masks over the combined image: `eq_u32_strided_to_mask` per declared G over `soa_map()`, domain = `OR`, horseshoe = TUI fence; `SpogTenants::over` declared from the non-empty Gs | MedCare-rs | every `tenant_mask[G].count()` equals `node_rows_for_classid(G).len()` (the partition_point answer is the independent reference); `Σ_G count == n_rows` over the declared set (anti-vacuity: the union is the whole image, no row in two tenants) |
| **D-SPG-3** | The rung × tenant cross via `mask_ternlog` on `words()` (§3.3), with the `unlooked[D]` read | MedCare-rs | `cell[r][G].count() == lane_r.scanpath().filter(graph_of == G).count()` for every (r, G) (materialized reference); `AND_ANDNOT2` differs from `AND3` on the same operands wherever `any_rung` is non-empty (the immediate is not decoration) |
| **D-SPG-4** | **PROBE-CROSSWALK-MASK-1**, gates (a)–(h) below, on the real image. **Pre-registration rule:** the probe's W0 READ pins the exact FK columns (byte offsets, widths, needle encoding) in its own header BEFORE any timing runs; a column that is not u32-aligned is either read through a documented masked compare or excluded and said so | MedCare-rs (probe) + lance-graph (record) | §6 |
| **D-SPG-5** | **The migration (F9):** every hand-rolled MedCare alpha driver moves ONTO the #1198 contract path — (i) `attention::WatchedRows`' `RefCell<AlphaOverlay>` + `domain_rung` writer → claims routed through `SpogTenants` inside an `AlphaTunnel` lane whose rung is the attention rung; `reflection(domain)` = `OR` over the domain's tenants' `attended_mask()`; (ii) `backreference::combined_base` → one `AlphaAllocation` over the image, patient rows as a declared tenant, never a second base; (iii) `medcare-nodesoa::alpha::{overlay_to_batch, write_alpha_overlay}` take the tunnel/tenants' `merge()` rows, not a bare overlay; (iv) `frontier_dispatch` routes through tenants over `all-lanes.soa`. `domain_rung` retired as a rung writer (F4) | MedCare-rs | on a fixed frontier, tenant-read reflection == `domain_rung` reflection as address SETS for all 8 domains (the migration is behaviour-preserving) AND the stamps' `rung` bytes now carry ladder values 1..=9 (can-fire: a fixture where the two would differ if the byte were still Domain+1) |
| **D-SPG-6** | Sealed batch per cycle: `merge()` → `node_rows_to_batch(rows, cycle)` → one append; `cycle = version + 1` | MedCare-rs | two cycles = two versions, rows readable at each; a read at version v never sees cycle v+1's stamps; `enable_stable_row_ids` absent from the writer (grep fence) |
| **D-SPG-7** | ogar-r2il consumer (`RANK` + `TERNLOG 0x86`) through `lance-graph-ogar` | lance-graph | lifted program's survivor mask == hand chain, bit-for-bit — **Queued, gates on D-SPG-4** |
| **D-SPG-8** | DataFusion containment (F8) | MedCare-rs | schema of the scan carries neither `_rowid` nor `_rowaddr`; a test that flips red if either flag returns — **Queued, gates on D-SPG-2** |

**Order is not negotiable:** D-SPG-0 → D-SPG-1 (the one contract line) →
D-SPG-2 (masks exist) → D-SPG-4 (probe; may run its synthetic arm before
D-SPG-2 lands, its real arm after) → D-SPG-3 → D-SPG-5 → D-SPG-6 → D-SPG-7 /
D-SPG-8. The loop per the autoattended pattern: plan → preflight → sprint →
review → fix P0 → commit → repeat; every board write in the same commit as
the code it describes; MedCare-rs edits stay in MedCare-rs (private).

## §6 — PROBE-CROSSWALK-MASK-1, gates (pre-registered; the IDEAS card's (a)–(e) plus three)

| gate | pass condition | what a fail means |
|---|---|---|
| (a) sets | survivor SETS of the mask chain == the scalar reference (`quad_slab::project` / sidecar path), every hop, as `materialize_ordinals` vectors | the FK reading of the columns is wrong — never the mask algebra |
| (b) bytes | 0 bytes/step under the counting allocator (D-GTM-0k's instrument, `hex_trie_vs_gemm_probe.rs`) on the hop hot path | something materializes |
| (c) flat | per-hop ns flat in chain length K while the live masks fit L2; report the K = 1 control (must read ≈ 1.0) and the bandwidth column | the win is residency, not chaining — say so |
| (d) seal | a deliberately cross-family `AND` (two tenants' FK masks) is REJECTED at the seal (can-fire) AND a same-family `AND` passes (can-stay-silent) | the fence is decoration |
| (e) sparse | a hop with < 0.1 % survivors is routed to the sparse arm and the two arms agree on the set | the density crossover moved |
| (f) angle | the eight `mask_ternlog::<K_i>(S,P,O)` minterm masks over the quad-stamped rows are pairwise disjoint and sum to the population; each of the six S/P/O→A/B/C wirings permutes the immediate's bits without changing any set; a wrong immediate (`0x80` vs `0xC0`) differs on real data | K0..K7-as-immediate is not a projection basis on this substrate — the §4 mapping is regraded |
| (g) reflection | tenant-read reflection == `domain_rung` reflection as sets (D-SPG-5's falsifier, run early as a read-only comparison) | the G grouping and the Domain grouping disagree somewhere — find the row |
| (h) amortization | tenant masks computed once per generation and reused across 10 rungs cost ≤ 1/10 + ε of recomputing per rung | M1b does not hold on this shape — cache key regraded |

Falsifiers are two-sided where a guard is involved ((d), (f)); assertions run
at the END so every claim is measured before any can abort (probe_nxg_hist_1's
lesson); the K = 1 control and the counting allocator are mandatory arms.

## §7 — Non-goals

- **No new type.** `AlphaMask`, `SpogTenants`, `AlphaTunnel`, `RowFocusMask`
  all stay; `words()`/`from_words()` are methods on an existing carrier.
- **No stable row ids, no `cleanup_old_versions`, no retention** (temporal 09
  non-goals 3 and 5). **Do not touch `temporal.rs`.**
- **No band derivation.** `ReasoningBand` is never `RungLevel`; the rung byte
  never encodes a domain (F4) and never lands in the residue band.
- **No VSA.** F7.
- **No DataFusion extension.** F8; containment only, after the masks exist.
- **No per-domain bake FILES.** F2 — the operator's hint is honoured by making
  the domain a mask, not by splitting artifacts.
- **No claim about recall or proof.** The alpha channel is a pruner (alpha-
  overlay plan piece 7); the SPOG channel does not change that.

## §8 — Corrections banked while writing this spec

1. **`lgj_hop` does NOT use `AND3`.** E-NXG-8 and `.claude/knowledge/membrane-tiers.md` §"The polyfill is the worked instance" said
   it did; lgj-abi at `dbac826` has zero `ternlog`/`AND3` symbols. Regraded on
   the board (dated entry) and in the knowledge doc (⊘ in place).
2. **IDEAS PROBE-CROSSWALK-MASK-1 named "the existing DataFusion path" as the
   reference.** No DataFusion crosswalk exists for this chain in MedCare-rs
   (DataFusion sits in `medcare-analytics` RLS/column-mask and `medcare-server`
   `state.rs`/`seed.rs`/`routes/patient.rs` — patient scans, not ontology
   crosswalks). The reference is the scalar sidecar/quad path (§3.2, gate (a)).
3. **The quad slab is populated on a SUBSET.** `all-lanes.soa` carries quads
   on 3,551 stamped rows (bakes.tsv 2026-08-10 repin note), not on every row —
   gate (f)'s population is those rows, declared as such.
