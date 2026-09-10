# PLAN v1 — the NaN CI mode: making dormant ABI absence visible

> **Status:** PROPOSAL — **`D-NCI-1..5` are unbuilt. No NaN mode exists.** No
> board rows, no minted D-ids; the `D-NCI-*` labels below are **this
> document's own** proposed deliverable names, not entries on
> `STATUS_BOARD.md`. Every design question in §8 and §9.4 is still open, and
> §3.6's `#[cfg]`-vs-hot-plug choice changes a frozen decision (N1), so it is
> a ruling, not an implementation detail.
>
> **One PREREQUISITE is pushed to this branch
> (`claude/lance-graph-1218-plans-z8hzqr`) — it is NOT on `main`, and it is
> not part of the mode.** `6e5e674` fixed `NarsEngine::revise_fast`, which was
> indexing `tables.deduction` — the wrong NARS rule, not merely dropping its
> confidence arguments. It now delegates to `NarsTables::revise`;
> `deduce_fast` names the other rule explicitly; `NarsEngine::with_c_levels`
> lets a caller buy a real confidence resolution. Three disable-verified
> tests (the can-fire / can-stay-silent pair plus the rule pin). It has
> **zero callers**, so it changed no production behaviour — it made a dead
> function correct and named the limitation the live threshold code had
> already run into. It matters here only as §4.5's confidence prerequisite,
> and **only** for the NARS revision path — `contract::revision` has no
> numerics at all (see §4.5's correction block).
>
> **§2.2's writer census was corrected after external review.** The original
> table named only two builders (`with_topology` / `with_reasoning_band`) and
> reported 0 non-test writers of bits 59-63 — true as far as it goes, but an
> undercount: `edge_v3::rehydrate` (`edge_v3.rs:263-291`) also writes bits
> 53-63, via `set_w_slot` / `set_truth` / `set_spare`, and it is a real,
> non-test-gated `pub fn`, not scaffolding. Checked both ways: `rehydrate`
> PRESERVES a value already carried in the V3 payload, it does not originate
> one; and every call site of `rehydrate` in this tree today sits inside a
> `#[cfg(test)] mod tests`. So the dormancy thesis is unchanged — the reason
> is one layer deeper than the original census stated. Full table: §2.2.
>
> **CI-verified at `7a5790e`:** 10/10 green — `format`, `clippy`, `test`,
> `test-with-coverage`, `member-tests`, `linux-build`, `regenerate-and-diff`,
> `citation-decay`, `added-plans-have-dids`, `no-shrink`. Cited because a plan
> that names a landed prerequisite owes the sha the claim was checked at.
>
> **Operator framing (2026-09-10, verbatim in substance):** the biggest debt is
> a large fraction of the ABI reading as *absent* — dormant and invisible.
> Run absence as a **debug/verbose mode during CI**, brutal and loud; certify
> per wire against the LE contract; the entry tax is paid once at the boundary
> and the wire stays cheap.
>
> **Thesis in one line:** a field nobody wrote is `0x00`, `0x00` currently
> decodes as a legitimate value, and therefore **the substrate cannot tell
> "never stamped" from "stamped with the default"** — anywhere. This plan makes
> that distinction visible in CI **without changing a single stored byte**.

## §1 FROZEN DECISIONS (cite-or-VIOLATES; not re-opened on taste)

| # | Frozen | Source |
|---|---|---|
| N1 | **Storage never changes.** Release builds are byte-identical to today, in what they store AND in what they decode. Only a CI/verbose build differs, and only in what it OBSERVES. ⊕ The hot-plug variant (§3.6) would relax the *decode* half — release could be flipped per activation. That variant is NOT chosen here; if it is, N1 is restated, not quietly broken. | operator, 2026-09-10 |
| N2 | **OBSERVE and FAIL, never branch-and-continue.** The mode may count, report, and abort. It may never take a different code path and proceed. The moment it changes control flow it becomes the pattern `I-LEGACY-API-FEATURE-GATED` forbids. | this plan §3.2 |
| N3 | **No new CE64 bit, no `ENVELOPE_LAYOUT_VERSION` bump, no new address type.** Inherited verbatim from `D-ACR-7` F7. | `dacr7-band-reading-contract-v1.md` §1 F7 |
| N4 | **A guard needs BOTH a can-it-fire and a can-it-STAY-SILENT test on non-trivial input.** | `CLAUDE.md` falsifiability rule |
| N5 | **A detector that fires on most reads is a census, not a gate.** Enforcement is an allowlist that SHRINKS, never a global assert that would be red forever and therefore ignored. Corollary of N4 applied to the instrument itself. | this plan §4.2 |
| N6 | **The mode is not switched off after certification.** It costs nothing in release either way; "off" buys only the loss of the ratchet. | this plan §7.3 |
| N7 | **Absence-detection is not correctness.** A wire can be fully wired, fully non-absent, and still carry the wrong value. This plan measures ONE axis and says so. | this plan §5 |

## §2 INPUT INVENTORY (measured 2026-09-10; file:line where verified)

### 2.1 The primary constructor cannot express the tail

`CausalEdge64::pack` under the default `causal-edge-v2-layout` writes bits
0..52 and stops:

- `crates/causal-edge/src/edge.rs:225-234` — the v2 arm writes S/P/O, freq,
  conf, causal mask, direction, mantissa, plasticity, then
  `// v2: temporal is IGNORED. Bits 52-63 are reclaimed ... silently drop it.`
  followed by `let _ = temporal;`.
- `pack_v2` (`edge.rs:835-844`) takes no W-slot, no truth, no band either.

**Consequence:** every edge built by either constructor is born with
`w_slot = 0`, `topology = 0`, `band = 0`. Those are not "unset" — they decode
as the *legitimate* values `Direct` / `Crystalline` / `Surface`. The named
builders that write bits 59-63 as fresh input — `with_topology()`
(`edge.rs:1009`) and `with_reasoning_band()` (`edge.rs:1057`) — have 0
callers outside `v2_layout_tests.rs`. A second writer exists too,
`edge_v3::rehydrate`, and it PRESERVES rather than originates (§2.2 has the
corrected census). `layout.rs:70-72` states that nothing DERIVES the band
automatically — that claim is unaffected either way.

### 2.2 The census: writers, readers, contract callers

> **Corrected after external review.** The first version of this table
> counted only two named builders and reported 0 non-test writers for bits
> 59-63; that undercounted a second, real writer function. Two questions
> were bundled under one column header and need to stay apart: (1) does a
> non-test-gated `pub fn` exist that writes the bits at all, and (2) is that
> function ever REACHED from a path outside `#[cfg(test)]`? The table below
> answers both, separately, for every dormant field this plan tracks —
> W-slot (53-58), truth/topology (59-60), spare/band (61-63).

| layer | (1) writer exists, non-test-gated? | (2) reached outside `#[cfg(test)]`? |
|---|---|---|
| bits 53-58 (W-slot) via `with_w_slot` / `with_routing` (`edge.rs:988,1069`) | yes, but **0** callers outside `v2_layout_tests.rs` | n/a — never called |
| bits 59-60 / 61-63 via `with_topology` / `with_reasoning_band` (`edge.rs:1009,1057`) | yes, but **0** callers outside `dismech_counterfactual.rs`'s own `#[cfg(test)] mod tests` | n/a — never called |
| bits 53-63 (all three) via `edge_v3::rehydrate`'s `set_w_slot` / `set_truth` / `set_spare` (`edge_v3.rs:279,288-289`) | **yes** — a real, non-test-gated function; but it PRESERVES a payload-resident value, it does not originate one | **no** — every call site (`cognitive-shader-driver/src/edge_v3_compare.rs:68-69`, `lance-graph-planner/src/cache/stage26_v3_parity.rs:294-295`) sits inside that file's own `#[cfg(test)] mod tests` |
| readers of bits 59-63 | — | **1**, reachable — `lance-graph-planner/src/dismech_counterfactual.rs:251-252`, via the raw accessors, not the contract projection |
| readers of bits 53-58 (W-slot) | — | **1 live filter, unreachable today** — `cognitive-shader-driver/src/mailbox_soa.rs:355`'s `apply_edges` drops any delivery whose `edge.w_slot() != self.w_slot`; `apply_edges` itself has zero callers outside its own `#[cfg(test)] mod tests` (`mailbox_soa.rs:1089,1120`) |
| callers of the `band_reading` surface (`BandReading`, `EdgeProvenance`, `project_truth`, `project_band`, `admits`, `admits_band`, `BandDeclarations`) | — | **0, anywhere in the tree** |
| classes overriding `ClassView::band_reading` | — | **0** — one impl, the default returning `ZERO_FALLBACK` (`class_view.rs:1231-1237`) |

`BandReading::ZERO_FALLBACK` is `{Trust, Absent}` (`band_reading.rs:230-234`),
so `project_band` would refuse `BandAbsent` for **every class in the tree
today**. The read contract is armed and fail-closed; nothing has ever opted in.

Note the one reader reads a field no production path writes: on any chain whose
edges came through `pack`, it reports the constant `(Direct, Surface)`.

**The `apply_edges` row is the thesis made concrete, not abstract.** A real
delivery filter compares `edge.w_slot()` against a real mailbox's own
`w_slot` and silently drops on mismatch — the module's own doc comment says
so directly (`mailbox_soa.rs:341-355`: "Mismatched edges are silently
dropped in `apply_edges`"). Because no production path originates a
non-zero W (the row above), and because `apply_edges` is not yet called
from outside its own tests, the comparison is latent rather than live
today — but it is the exact failure this plan exists to make visible: a
`CausalEdge64::ZERO`-derived baton's `w_slot() == 0` is indistinguishable
from a real mailbox that also happens to be `w_slot == 0`, the moment
`apply_edges` gains a live caller.

### 2.3 The mantissa already aliases absence to a legitimate value

- `InferenceType::to_mantissa` (`edge.rs:65-82`) emits only
  `{1, 2, -1, 4, 5, 6, -6, 7}` — **never `-8`**.
- `InferenceType::from_mantissa` (`edge.rs:90-94`) does
  `let mag = m.unsigned_abs() & 0x7;` then `0 => Self::Deduction` with the
  comment `// 0 = Identity/neutral -> treat as Deduction`.

So on that field `0` means **three** things at once: "Identity/neutral",
"never stamped", and "the `-8` bucket". `-8` is unwritable by construction and
aliased on read.

### 2.4 Carriers that erase the tail

- `SpoHead` (`lance-graph-planner/src/cache/nars_engine.rs:28-38`) mirrors
  CE64 at 8 bytes but carries `temporal: u8`, which is **dead under v2**:
  `from_causal_edge` hardcodes `temporal: 0` (`:513`, with a doc explaining the
  v2 sentinel) and `to_causal_edge` (`:469`) feeds it to `pack`'s ignored
  argument. It never carries topology or band. It is a **v1-shaped mirror of a
  v2 carrier** — it did not follow the reclaim its own model made.
- `CausalEdgeV3::from_v1_tail_unstated` (`crates/causal-edge/src/edge_v3.rs`)
  zeroes bytes 8 and 9, producing a register indistinguishable from one stamped
  `Direct` / `Surface`. This is the transitive half of the v1 provenance trap
  `band_reading`'s module doc already names (council BLOCK 1).

### 2.5 The convention is already "zero means absent" — by hand

- `causal_witness::elected` (`causal_witness.rs:428-437`): *"Zero maps to
  `None` because `0` is the register's own zero-fallback sentinel for
  'unbound' ... it is never 'offset zero, meaning self'."* A legitimate value
  was **forfeited** because there was no sentinel.
- `probe_witness_presence_2bit.rs`'s `presence_2bit` puts `v < 0` in *before*,
  `v > 0` in *after*, and `v == 0` in **neither**.
- `CausalEdgeV3::anaphora()` returns `Option<i8>` with `0 = none`.

Three shipped sites already treat zero as absence, each by local convention,
none by contract.

### 2.6 The instrument family already exists

Four probes in this tree are the right idiom, so this is a fifth in a shipped
family, not new architecture:

`probe_witness_presence_2bit.rs`, `probe_mask_algebra_invariance.rs`,
`probe_copula_group_mask.rs`, `sigma_probe_masked_traverse.rs`.

The last states the discipline verbatim: *"changes no library code: it calls
`mxm` directly for the unmasked baseline and `masked_traverse` for the masked
result, and reports per-call rows."*

### 2.7 The i4 SIMD surface (for §3.4 cost)

- `I8x16::from_i4_packed_u64` (ndarray `src/simd_int_ops.rs`, W1a primitive)
  unpacks 16 packed nibbles into 16 **sign-extended i8 lanes**; its own tests
  pin `0x8 -> -8` and `0x7 -> +7`.
- `masked_sum_i32` (`:1117`), `masked_strided_group_sum` (`:1208`),
  `mask_ternlog_assign` (`:1015`) already take mask words.
- `cmp_gt` / `cmpgt_mask` / `movemask` / `mask_blend` are on the polyfill
  SURFACE (`src/simd.rs`), which is a re-export catalog over per-arch backends
  (`simd_avx512` / `simd_avx2` / `simd_neon` / `simd_wasm` / `simd_scalar`).
- `src/simd_scalar.rs` is a **backend of that same surface**, not an alternative
  path — its own module doc: *"Mirrors the API of `simd_avx512`, `simd_avx2`,
  and `simd_neon::aarch64_simd` so consumer code reading
  `use crate::simd::F32x16` compiles and runs uniformly across all supported
  targets."* Dispatch is **compile-time** (`#[cfg]`), so the W1a tests run
  *"against exactly one backend per build"*.
- `nibble_above_threshold` (ndarray `src/nibble.rs:227`) is an AVX2 compare
  over packed nibbles — but **unsigned** (Minecraft light levels), returning a
  materialised `Vec<usize>`.


## §3 THE DESIGN

### 3.1 What "NaN" means here

**A decode-time verdict, not a stored bit pattern.** In a CI/verbose build, a
read of a field that was never written returns/reports `Absent` instead of the
value the bits happen to spell. In a release build the identical read returns
the identical value it returns today.

This is `debug_assert!` generalised to the decode boundary. The precedent is
shipped: `BandReading::project_truth` already carries
`debug_assert!((truth_raw as usize) < TRUTH_STATES, ...)`, documented as *"the
G7' compile-time/precondition pin (F9-exempt, stated)"*.

### 3.2 The iron rule that keeps it legal (N2)

`I-LEGACY-API-FEATURE-GATED` forbids *"the same function name silently
producing different semantics under different feature flags"* — and a decode
mode looks exactly like that. The distinction that makes it legitimate:

> Release returns a value. CI returns **"this was never written"** and stops.
> CI is strictly MORE informative, never DIFFERENTLY informative.

Operationally: the mode may `count`, `log`, `collect`, and `panic`/fail. It may
not `if absent { ... } else { ... }` and continue. A reviewer checking this
plan's output checks exactly that one property.

### 3.3 Two designs, different reach — this plan authorises only the first

**Design 1 — POISON-FILL (no encoding change). THIS PLAN.**
In CI builds, constructors fill never-written fields with a canary instead of
zero: `pack()` fills bits 53-63 with the canary, likewise `CausalEdge64::ZERO`,
`Default::default()`, `from_v1_tail_unstated`, and the equivalent SoA/tenant
initialisers. Anything still reading the canary was **never stamped by this
producer**. Classic poisoned-memory technique.

- Reach: producer-side gaps — which is what an *ABI* debt is.
- Works on **every** field, not only signed i4.
- Zero storage change, zero encoding change, zero release cost.
- Cannot answer "was this ever stamped by anyone, ever" for a **persisted** row:
  a stored zero and a never-written zero are the same byte on disk.

**Design 2 — SWAPPED ENCODING. NAMED, NOT AUTHORISED HERE.**
Reassign `0b0000 -> NaN` and `0b1000 -> 0` on signed-i4 fields, making the
default byte the absence sentinel in storage, permanently. Real value set
becomes `{-7..-1, 0, +1..+7}` — 15 values, symmetric, median exactly 0.

- Reach: persisted corpora, not just freshly-constructed registers.
- Costs: reinterprets every existing byte (same accessor, different value —
  `I-LEGACY-API-FEATURE-GATED` at corpus scale); legitimate legacy zeros become
  absent; two ndarray W1a tests re-pin; NaN arithmetic must be defined.
- **Deliberately out of scope.** Reach for it only when certifying a stored
  corpus becomes the live question, and give it its own plan and its own
  version gate.

### 3.4 Cost, honestly split

**Release: zero.** The mode is `#[cfg]`-ed out. No lane predicate, no branch,
no extra op in any kernel. The whole SIMD question is priced at zero because
nothing runs.

**CI, Design 1:** a canary fill in the affected constructors and a compare at
the affected reads. On the i4 lanes this lands where the tree already unpacks
to i8 (`from_i4_packed_u64`), so a canary compare is an ordinary i8 compare and
`masked_sum_i32` / `masked_strided_group_sum` already accept the resulting
mask — **no new reduction kernel**. At most one new surface function would be
an unpack-plus-presence returning `(I8x16, mask)`, in the module that already
owns the unpack. Not required for Design 1's first wave.

**No scalar arm, and no scalar cross-check** (operator, 2026-09-10). The
polyfill surface is transparent in the Valhalla/Panama sense — Panama's vector
surface lowers to the best available ISA with a guaranteed fallback, Valhalla's
value types flatten a wrapper onto a register, and `ndarray::simd` does both,
at **compile time** rather than by JIT. A consumer writes `use ndarray::simd::*`
once; ndarray fills that surface with SIMD per target. Consequences for this
plan, stated so a later session does not re-derive them:

- **Nothing here authors a scalar path.** `simd_scalar` already mirrors the API
  (§2.7); a hand-written scalar arm beside it would be a second implementation
  of a backend that exists.
- **Nothing here authors a scalar cross-check either.** Dispatch is
  compile-time and one backend runs per build, so the scalar backend *is* the
  cross-check when CI builds a non-x86 target. A cross-check written into this
  plan would test the polyfill, which is not this plan's subject.
- **"Per backend" is not a unit of cost for a caller.** A new surface function
  is authored once on the surface; where its arch implementations live is
  ndarray's internal structure, and no consumer of the polyfill ever sees it.
  Earlier drafts of this section priced backends as a plan cost — that was
  wrong, and it is corrected here rather than deleted.

### 3.5 The collapse taxonomy — the product, not a by-product

The mode's output is not a boolean. Each finding lands in one of five classes,
and each class has a **different fix**. All five are already instanced in the
tree:

| mode | measured instance | fix |
|---|---|---|
| **Silent-zero** — absent becomes 0, nobody says so | bits 59-63 through `pack` (`edge.rs:225-234`) | sentinel or declaration |
| **Silent-alias** — absent acquires a *different legitimate identity* | `from_mantissa(-8)` -> `& 0x7` -> 0 -> `Deduction`, which `to_mantissa` re-emits as `+1` | close the alias (free here: `-8` is unwritable) |
| **False-assert** — absent becomes a positive claim | `quorum::AxisProjection::nars_frequency` maps `position = 0` to `8/15 = 0.5333`, and `nars_frequency_range` (`quorum.rs:357-365`) *asserts* `mid > 0.5` | **arithmetic**, not a sentinel — see §5.2 |
| **Declared-collapse** — collapses to 0 and the contract says so | `causal_witness::elected` -> `None`; `anaphora()` -> `Option` | acceptable — or reclaim the forfeited value |
| **Refusal** — never collapses; the read errors | `BandReading::project_band` -> `BandAbsent` | **the target state** |

Silent-alias is the worst of the four defect classes: the value does not
vanish, it **becomes something else**, so the downstream reader gets a
confident wrong answer rather than an empty one.

### 3.6 ALTERNATIVE — the switch as a hot-plug property (operator, 2026-09-10)

> *"you can put even NaN switch into the contract via hotplug.rs — which would
> be a little brutal."* Recorded as a live alternative to the `#[cfg]` design
> above. **Not chosen here**; choosing it changes N1 and §3.4, so it is a
> ruling, not an implementation detail.

**Why it fits, better than a new surface would.** `hotplug::Activation`
**already carries the reading** — `read_modes`, ruled `D-BLOCKS-HOTPLUG-1`
(operator, 2026-09-07): *"how a row addressed under each hot-plugged concept is
READ: which tail the key carries, which value tenants materialise, how the edge
block is carved."* "What does a field nobody wrote decode to" is a **reading**
decision, so it belongs in the struct that already answers that question.

Four properties come free:

1. **Per consumer, per classid — which is §4.1's certification unit exactly.**
   A consumer that has certified its wires activates without the mode; one that
   has not gets it on. §4.2's allowlist stops being a separate file and becomes
   a property of the activation.
2. **Fail-closed by construction.** `Activation` deliberately has **no
   `Default`** — *"an activation is something an authority RESOLVED; there is
   no meaningful empty one."* So there is no accidental green activation
   carrying no NaN policy.
3. **Drift machinery already exists.** `ActivationDrift` /
   `verify_against_mirror` / `mirror_disagreement` — a policy mismatch between
   consumer and authority bangs once, on the path that already bangs.
4. **Zero-dep forces the right vocabulary.** The contract cannot see
   `causal-edge`, so the switch must be expressed in raw/contract terms —
   the same discipline `band_reading` gets by taking raw ordinals.

**What it costs, stated plainly.**

- **N1's decode half.** The mode becomes a runtime property, so a *release*
  binary can be flipped. That is strictly more powerful — a live deploy becomes
  certifiable, not only CI — and it is why the operator calls it brutal. But
  §3.4's "Release: zero" stops being unconditional and must be re-priced.
- **The re-pricing is smaller than it looks.** An activation is resolved
  ONCE, at plug time (`Activation::read_mode_for`), so the policy does not have
  to be read per decode: the consumer already holds a resolved `ReadMode` and
  branches on a value it has. That is the dispatch cost the tree already pays,
  not a new per-read branch — but this is a *reasoned expectation, not a
  measurement*, and it needs one before N1 is relaxed on its strength.
- **N2 is unchanged and non-negotiable.** Runtime or not, the mode may observe
  and fail; the moment it takes a different branch and continues, it is the
  defect this plan exists to find.

**The hybrid, if the trade is unwelcome.** Hot-plug carries the POLICY (which
classids are certified, what absence means for them); `#[cfg]` carries the
ENFORCEMENT (whether a violation aborts). Release resolves the policy once at
activation and pays nothing further; CI adds the abort. This keeps N1 intact
and still lands the certification unit where the reading already lives.

## §4 THE CAMPAIGN

### 4.1 The certification unit is a WIRE, not a crate

A "wire" is one `(producer, field, consumer)` path with a declared LE reading.
It maps onto surfaces that already exist:

- `BandReading` per `(classid, rail)` — the declaration
- `ColumnDescriptor` / `SoaEnvelope::verify_layout` — the byte range
- `EdgeProvenance` — the epoch the raw ordinal was written under

A wire is **certified** when: a producer stamps it, a class declares it, a
consumer projects it through the contract (not a raw accessor), and the CI mode
reports zero canary reads on that path. Progress is then a countable fraction,
not a feeling.

### 4.2 Enforcement is an allowlist that SHRINKS (N5)

Day one, the mode will fire on a large fraction of reads. A gate that is red
everywhere carries exactly as much information as one that never fires, and it
will be routed around within a week.

So the enforcement shape is:

1. **Wave 0 — census only.** The mode runs, reports, and **does not fail**. Its
   output is a table: wire, field, canary-read count, collapse class.
2. **The allowlist is seeded from that census** — every currently-absent wire is
   listed as a known exception, with the class from §3.5.
3. **CI fails on any read NOT in the allowlist.** New unwired sites are red
   immediately; existing ones are documented debt.
4. **Each certification removes an entry.** The allowlist only ever shrinks;
   growing it requires the same review as any other debt admission.

### 4.3 The first deliverable is the real number

The first Wave-0 run **is** the debt measurement, per field, per wire, free —
a COUNT. §4.5 turns it into a RANKING, which is the more useful object.
The operator's working figure for the ABI is large; this plan does not restate
it as measured, because it has not been measured here. Everything in §2 points
the same direction — no production PATH ever writes bits 53-63 (the one real
writer, `edge_v3::rehydrate`, is itself reachable only from test harnesses,
per §2.2's corrected census), `pack` zeroing 53-63, zero callers of the
entire `band_reading` surface, zero `ClassView` overrides — but the number
comes from the run, not from the plan.

### 4.4 Proposed deliverables (this document's own labels)

| id | deliverable | depends on |
|---|---|---|
| `D-NCI-1` | The mode itself: a `#[cfg]`-gated canary constant + fill in the affected constructors, and a read-side observer. Observe-and-fail only (N2). | — |
| `D-NCI-2` | Wave-0 census run + the report table (wire / field / count / collapse class). | D-NCI-1 |
| `D-NCI-3` | The allowlist, seeded from D-NCI-2, plus the CI job that fails outside it. | D-NCI-2 |
| `D-NCI-4` | First certified wire, end to end: stamp -> declare -> project -> zero canary reads. Also the first caller of `admits_band` / `project_band`, which today have none. | D-NCI-3 |
| `D-NCI-5` | `SpoHead` reclaim: it is a v1-shaped mirror (dead `temporal` byte) of a v2 carrier. Preservation fix under `I-LEGACY-API-FEATURE-GATED`, not a feature. | D-NCI-4 |

`D-NCI-1..3` are the instrument. `D-NCI-4..5` are the first two repayments.
Nothing beyond `D-NCI-5` is planned here on purpose — the census decides the
order, and pre-deciding it would be the plan overruling its own measurement.

### 4.5 Ranking by consequence — "if 0 were NaN" is a COUNTERFACTUAL, literally

> Operator, 2026-09-10: *"wiring NaN mode into revision and recalculate a CE64's
> known-unknowns — 'if 0 would be NaN' kind of counterfactual probing and
> revision."*

> ⊘⊘ **CORRECTED 2026-09-10 on a re-read of both modules, and on the operator's
> caution — *"0 vs NaN is only a sidestep via `ogar-loco`; it never replaces the
> whole."* Three errors in the first version of this section, kept visible
> rather than deleted:**
>
> 1. **A HOMONYM collapsed in my own prose.** This section cited
>    `NarsEngine::revise_fast` — **NARS truth revision** (u8 frequency /
>    confidence, `NarsTables`) — while being framed around the revision
>    **docket**, which is `contract::revision::GadamerRevision`. They are
>    unrelated. `contract::revision` is **pure set algebra over
>    `EvidenceMask`**: nine `RevisionKind`s from six booleans, and
>    `EvidentialEffect` is *"deliberately coarser than numerical confidence"*.
>    **It has no numerics at all**, so it never had a confidence axis to be
>    blind on. The `revise_fast` fix (`6e5e674`) is real and unblocks the NARS
>    path; the commit message's claim that it "unblocks §4.5's stated
>    blindness" over-reached to this one. This repo has a standing record of
>    exactly this trap — the two-`GateDecision` collision in
>    `probe_revision_kanban_hinge`, `TrustTexture` as a four-way homonym in
>    `D-ACR-7` §2.4.
> 2. **`contract::counterfactual` does not RUN counterfactuals.** Its whole v3
>    is `todo!()` — `CounterfactualMailbox::{new, poll, cancel}` and
>    `revise_if_minority_wins` — and `AwarenessRevise` is an explicitly
>    **BLOCKED placeholder** whose doc says the canonical `awareness.revise`
>    signature is *"not confirmed on the current contract surface"*. Only v2 is
>    real: the 4-bit mantissa deposit and the spawn gate. The module that
>    actually runs one is `planner::dismech_counterfactual::counterfactual_replay`
>    — cited correctly below, but framed as if the contract module were the
>    runner. It is not.
> 3. **The routing was missing, and it is the point.** See "Where it actually
>    attaches" at the end of this section.

A count says how many fields are unstamped. It does not say which absences
**change an answer**. That second question is not a new mechanism: it is the
shape `dismech_counterfactual::counterfactual_replay` already implements —
*"the SAME W1 replay with one edge cut ... a thresholded verdict ... a
load-bearing edge moves the chain's truth ACROSS the threshold; a redundant one
moves it and stays on the same side."*

Substitute the cut and the shape carries over unchanged:

| | factual arm | counterfactual arm | verdict |
|---|---|---|---|
| shipped (`counterfactual_replay`) | chain as recorded | chain with step `i` removed | was that EDGE load-bearing |
| **this probe** | chain read as stored (`0` = a value) | chain read with `0` = **absent** | were those ABSENCES load-bearing |

Both arms go through the same replay, so a divergence can only come from the
reading — never from two implementations drifting apart, which is the property
`dismech_replay` was built to guarantee.

**It respects the direction ruling.** Nothing here feeds bits into revision.
Revision and counterfactual stay complete thinking; the probe runs the *same*
script twice over *two readings of the same bytes*. The scripts are untouched.

**What the verdict buys:**

- `Necessary` — the chain's conclusion moves when the unstamped fields are
  treated as absent. **The absence is load-bearing**: this wire's silence is
  already changing answers, and it ranks first.
- `Dispensable` — the conclusion holds either way. The wire is unwired and
  nothing downstream depends on it: real debt, low priority.

That is the ordering §4.2's allowlist wants, derived rather than argued — and
it repairs §7.4's weakness, because a small census of *load-bearing* absences
is worth more than a large census of inert ones.

**Scope and honesty:**

- This is a **probe, not a gate**. `counterfactual_replay` has no production
  caller today (measured: tests only), and replaying every chain is not a CI
  budget. It belongs after `D-NCI-2`, not inside `D-NCI-1`.
- **The confidence axis is available, but only if the caller buys it.**
  ⊘ This bullet first read *"blind on the confidence axis as things stand —
  `revise_fast(f1, _c1, f2, _c2)` discards BOTH confidences ... fix that
  first"*. The fix landed the same day, and it was a bigger defect than the
  dropped arguments: `revise_fast` was indexing `tables.deduction` — the wrong
  NARS rule — and the deduction table has no confidence axis at all, which is
  why both arguments were `_`-prefixed. It now delegates to
  `NarsTables::revise`, and `deduce_fast` names the other rule explicitly.

  **The blindness did not vanish; it MOVED**, and the new location is the one
  that matters here. `revise` selects its table by quantizing `c1`/`c2` into
  `c_levels` buckets, and `NarsEngine::new` builds **one** bucket: equal
  weights, so the frequency is a plain mean and `c_out` is the constant 170 —
  the same fixed point `DEFAULT_FREQUENCY_BAR` documents. **A ranking built on
  `new` is still frequency-only.** Use `NarsEngine::with_c_levels` (cost
  `c_levels² × 128 KB`) or state in the result that the axis was never
  consulted. Pinned two-sided:
  `revise_fast_honors_confidence_at_multiple_levels` /
  `revise_fast_confidence_is_inert_at_one_c_level`.
- `DEFAULT_FREQUENCY_BAR` already carries the right warning for whoever tunes
  this: confidence saturates to a fixed point under `NarsTables::build(1)`, so a
  confidence-based verdict would be *"a vacuous threshold — every chain on the
  same side of every bar."* The same trap is one substitution away here.

**Where it actually attaches — a sidestep, never a replacement.**

`GadamerRevision::revise(prior, encounter, ancestry)` consumes **masks**:
`independent_roots`, `resistance`, `contradictions`, `proposed_claims`. There
is no field in it a CE64 bit could be written into, and nothing here proposes
one. So "recalculate a CE64's known-unknowns through revision" cannot mean
modifying `revise` — it means **constructing a different `EncounterEvidence`**:
does an absent field still count as an independent root contacted, a
resistance met, a contradiction live?

That construction is **upstream of the docket**, and its home is `ogar-loco`
— the operator-ruled planning/execution target (2026-09-05, *"every planning
is in migration to ogar-loco and ogar-r2il"*), where `recipe_vocab` already
lowers the 34 NARS recipes to loco ops and `dismech_replay` /
`dismech_counterfactual` already reference it.

So the shape is:

```text
loco program A: read as stored (0 = a value)  → masks → docket → verdict
loco program B: read with 0 = absent          → masks → docket → verdict
                                                   ↑
                          THE SAME docket, unmodified, run twice
```

**The NaN reading is one more loco program, run BESIDE the docket.** It does
not enter `revise`, does not add a field to `EncounterEvidence`, does not
substitute for the counterfactual attack, and does not become the thinking. It
changes what the thinking is handed — and if this section is ever read as
licence to put a NaN branch inside `revision.rs` or `counterfactual.rs`, it has
been read wrong.

## §5 NON-GOALS (each with its why)

1. **The swapped encoding (Design 2).** Named in §3.3, not authorised. It is a
   storage reinterpretation and needs its own plan, its own version gate, and a
   persisted-corpus question that is not yet live.
2. **The `/15` divisor family.** Six sites carry a `15` derived from the
   asymmetric `-8..+7` range, and they split into **two kinds**:
   - *offset-and-scale* `(x + 8) / 15`: `quorum.rs:137`,
     `mul.rs:898`, `mul.rs:1422` (allostatic load, **duplicated**)
   - *max-distance* `|d| / 15`: `recipe_substrate.rs:237`, `:255`, and the
     `logical_dissonance` pin in
     `tests/d_pop_2_producer_reaches_consumers.rs:252`

   Both become `14` **for different reasons** (span 15->14 steps; max distance
   15->14). Two consequences worth stating so a later session does not
   grep-and-replace its way to a right answer for the wrong reason:
   - The **max-distance sites are not buggy today.** `/15` is correct for the
     current range; they become wrong only *after* the range narrows. So the
     order is: change the range first, then all six follow.
   - The **clamp couples to the divisor.** `AxisProjection::settled` clamps to
     `-8` and `position_clamps_to_i4_range` (`quorum.rs:350-353`) pins it.
     Divisor to `/14` with the clamp still at `-8` yields `(-8+7)/14 = -0.071`,
     a negative frequency. Same commit or neither.

   This is a sibling arithmetic fix. A detector does not touch it.
3. **The i4 dequant overshoot.** `quantize_f32_to_i4` uses
   `scale = abs_max / 7.0` (ndarray `hpc/quantized.rs:671-673`) while
   `dequantize_i4_to_f32` maps `0x8 -> -8 -> -8*scale`, i.e. about
   `-1.143 * abs_max` — outside the codec's own declared `min_val = -abs_max`.
   Same family as (2), same fix direction, separate change.
4. **Correctness.** This measures whether a field was written, never whether
   the value is right (N7). A wire can be certified green and still assert the
   wrong thing.
5. **Any consumer-repo work.** This plan is lance-graph-internal. Downstream
   consumers consume `main`; their side is a separate, later question.


## §6 PRE-REGISTERED GATES (decided BEFORE any code)

| # | Gate | Falsifier |
|---|---|---|
| G1 | **Release is byte-identical.** A release build with and without the mode's code present produces identical output on the golden paths. | Any observable difference in a release build = N1 violated, revert. |
| G2 | **Observe-only.** No `cfg`-gated branch changes control flow; the mode's only effects are counting, reporting, and failing. | A reviewer finds an `if <absent>` with a non-failing `else` arm = N2 violated, block. |
| G3 | **Can-fire.** At least one wire reports a NON-zero absence horizon. `project_band` refuses today, so a positive control exists on day one. | If every row reports total collapse, the probe is broken, not the substrate. |
| G4 | **Can-stay-silent.** `frequency_u8` / `confidence_u8` survive a CE64 round-trip and must report **present**. | If those report absent, the probe is wrong. |
| G5 | **The allowlist shrinks.** Every PR after D-NCI-3 either leaves the allowlist unchanged or removes entries. | An addition without an explicit debt admission = block. |
| G6 | **No new CE64 bit, no `ENVELOPE_LAYOUT_VERSION` bump** (N3). | Any layout constant moves = out of scope, split the PR. |

G3 and G4 together are the N4 pair applied to the instrument itself, and they
are the two that a vacuous version of this work would skip.

## §7 RISKS

### 7.1 It looks exactly like the pattern the iron rule forbids

`I-LEGACY-API-FEATURE-GATED` was written against feature-gated semantic
divergence, and Sprint-11 caught that pattern five times. A decode mode is
adjacent to it by construction. The only thing separating them is N2, and N2 is
a property a reviewer must actually check rather than assume. **If N2 ever
softens, this plan becomes the defect it was written to find.**

### 7.2 A red-everywhere gate gets ignored

Covered by N5/§4.2, restated as a risk because it is the likeliest failure
mode: shipping enforcement before the census produces a job that is red on
day one and disabled by day ten.

### 7.3 Turning it off after certification (operator's own last clause)

The plan's one disagreement with its own framing. Certification is not a finish
line; it is a ratchet. In release the mode already costs nothing, so "off" buys
only the loss of the ratchet, and new unwired sites reappear silently.

This repository has receipts on both halves of that failure:

- the supersession index went stale **within the hour** of first landing,
  because regeneration was manual — *"a generated artifact with no staleness
  gate is a hand-maintained artifact with extra steps"* (`CLAUDE.md`);
- two `tesseract-core` fixtures sat red for 13 days, invisible three ways,
  because nothing re-ran them where anyone looked.

Recommendation: keep the mode, keep it CI-only, drive the allowlist to zero and
leave the job armed.

### 7.4 The census could be smaller than expected

If Wave 0 reports a low absence rate, the premise weakens and the campaign
should be re-scoped rather than pushed. That outcome is a legitimate result of
D-NCI-2, not a failure of it — and pre-committing to it here is what stops the
measurement from becoming a formality.

### 7.5 Design 2's trap, recorded now so it is not rediscovered later

If Design 2 is ever taken up: under the swap, a **stale reader** still doing
naive two's-complement decode produces

- absent (`0b0000` -> 0) -> `(0 + 7) / 14 = 0.5` — perfectly neutral,
  perfectly plausible, **invisible**;
- a real zero (`0b1000` -> -8) -> `(-8 + 7) / 14 = -0.071` — out of range,
  **loud**.

The two failure directions are asymmetric in the worst way: absence fails
silent at the one value nobody questions. Any Design-2 plan must guard the
formula with the absence check, not merely re-base the divisor.

## §8 OPEN — needs an operator ruling before D-NCI-1

1. **The canary value(s).** One per field width, or one global pattern? A value
   that is *itself* a plausible datum re-creates the problem one level down.
2. **Failure granularity.** Does an un-allowlisted absent read fail the test
   that touched it, or does the job fail once with the full table? The second
   is kinder to a large first wave; the first localises better.
3. **Scope of the first wave.** The whole crate graph, or `causal-edge` +
   `lance-graph-contract` only? §2's census was entirely inside those two
   before the correction; the corrected §2.2 also cites `cognitive-shader-
   driver` (`edge_v3_compare.rs`, `mailbox_soa.rs`'s `apply_edges`), so a
   two-crate first wave would now exclude the one live-shaped filter
   (`apply_edges`) the census found.
3a. **`#[cfg]`, hot-plug, or the hybrid (§3.6).** The `#[cfg]` design is what
   §3 specifies; the hot-plug variant relaxes N1's decode half in exchange for
   reaching production and folding §4.2's allowlist into the activation. The
   hybrid keeps N1 and still moves the policy to the socket. This is the one
   open item that changes a frozen decision.
4. **Whether `D-NCI-5` (`SpoHead`) rides in this plan or its own.** It is a
   preservation fix, independent of the mode, and could ship first.
5. **Where the runtime disposition of absence lives (operator, mid-session
   2026-09-10): "NaN as Staunen unhydrated trigger for CE64 bits 59-60."**
   Absence is not only debt; at runtime it is *surprise* — the one surprise the
   current Staunen cannot see, because every shipped Staunen is derived from
   what IS present (`nars/basin_resonance.rs:183` mean stakes;
   `nars/insight.rs:168` mean committed contradiction depth;
   `nars/ghost_prior.rs:152-157` `GhostEcho::Staunen`). Staunen has **no
   primitive carrier** — it is not among the 17 `AXIS_LABELS`
   (`qualia.rs:28-46`), and `QualiaI4_16D` packs only the first 16, so minting
   an axis for it would widen a fixed-width column. See §9.

## §9 THE RUNTIME DISPOSITION — one detector, two consumers (OPEN)

The CI mode and a runtime absence-signal are **the same detection with two
dispositions**, and the plan does not fork to accommodate the second:

| context | absence means | response |
|---|---|---|
| CI | debt — a producer that never stamps | fail the build (§4.2 allowlist) |
| runtime | surprise — unhydrated | raise Staunen -> gather |

### 10.1 The three-way split already exists in the contract

Conflating "never wired" with "not yet hydrated" would be a defect: the first
must fail CI, the second must retry, and an eternal retry on the first is worse
than silence. `band_reading` already separates all three, and its module doc
states the distinction verbatim — *"`None` = never declared; `Some(band:
Absent)` = **explicitly** declared band-free. Folding the two would make 'opted
out' and 'never considered' indistinguishable to a migration audit."*

| contract state | meaning | disposition |
|---|---|---|
| `BandPresence::Absent` (declared) | the class carries no band, on purpose | **not** a surprise — a fact. No Staunen. |
| `EdgeProvenance::Unknown` -> `BandReadError::UnknownProvenance` | origin unstated; the bits are not readable | **the unhydrated case** -> Staunen -> gather |
| `BandDeclarations::get` -> `None` (never declared) | nobody ever considered this wire | **the debt case** -> CI fail |

So the placement question has a cheap answer: **the trigger lives in the read
contract that already refuses, and the refusal variant already carries the
reason.** Nothing new is minted — no new bit (N3), no new qualia axis, no new
error type.

### 10.2 Why bits 59-60 in particular

The 2-bit field already encodes degrees of epistemic murk under either lens —
`TrustTexture` = `Crystalline / Solid / Fuzzy / Murky`, `CausalTopology` =
`Direct / IndirectKnown / IndirectUnknown / Unknown`. Absence is the limiting
case of the same axis: `Unknown` is *"I do not know the path"*; a refused read
is *"I was never told"*. It reads as a fifth state **without costing a bit**,
because it is the absence of the field rather than a value in it.

### 10.3 The response already exists too

`mul::GateDecision::Hold` returns `None` from `advance_on_gate`, and the owner
is HELD and re-polled — mechanically *"stay, gather more"*. That is the
hydration response, already shipped. Free energy is the natural consumer
(surprise raises F; F above the homeostasis floor re-fires dispatch), so the
chain is: refusal -> Staunen -> F -> `Hold` -> hydrate -> re-read.

### 10.4 What is genuinely open

1. Does the refusal raise Staunen **directly**, or does it raise free energy
   and let Staunen fall out of the existing derivation? The second changes no
   Staunen formula; the first gives Staunen its first primitive carrier.
2. Retry policy. A refusal that never hydrates must degrade to the debt case
   rather than loop — the boundary between "not yet" and "never" is a count or
   a deadline, and it needs a number, not a feeling.
3. Whether this rides in this plan at all, or becomes its own. The detector is
   shared; the dispositions are not.

## §10 BOARD HYGIENE OWED (not performed here)

Per `CLAUDE.md`'s Mandatory Board-Hygiene Rule, a PR carrying a new integration
plan also owes a PREPEND to `.claude/board/INTEGRATION_PLANS.md`. That file is
append-only and protected; this plan does not touch it. Whoever opens the PR
adds the entry in the same commit.
