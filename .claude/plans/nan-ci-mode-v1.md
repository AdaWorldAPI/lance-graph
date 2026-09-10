# PLAN v1 — the NaN CI mode: making dormant ABI absence visible

> **Status:** PROPOSAL — **`D-NCI-1..5` are unbuilt. No NaN mode exists.** No
> board rows, no minted D-ids; the `D-NCI-*` labels below are **this
> document's own** proposed deliverable names, not entries on
> `STATUS_BOARD.md`. §3.6's `#[cfg]`-vs-hot-plug choice changes a frozen
> decision (N1), so it is a ruling, not an implementation detail.
>
> **§8 and §9 were run through a 5+3 hardening council (2026-09-10) and now
> carry COUNCIL-HARDENED resolutions — still awaiting operator confirmation,
> never a substitute for it.** 5 savants + 3 brutal reviewers, one full
> streamline→attack→fix→ratify cycle, 0 BLOCKs, several real fixes applied
> (a defect in the council's own first-draft R7 was caught and repaired, not
> argued away). See §8/§9 for the resolutions and
> `.claude/board/AGENT_LOG.md` for the run record. No Rust was written; this
> is a plan-text + board-hygiene commit only.
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
In CI builds, never-written fields are marked absent instead of decoding as
zero. **The mechanism splits in two, by field shape — a 5+3 council finding
(2026-09-10, §8/§9 resolution below), not stated when this section was first
drafted:**

- **Saturated fields** (`TrustTexture`/`CausalTopology` 2-bit,
  `ReasoningBand` 3-bit, W-slot 6-bit — every bit pattern in each field's own
  width already names a legitimate variant or slot, zero spare codes) have no
  VALUE available to poison-fill with. These are marked via a **CI-build-only,
  out-of-band sidecar** — a `#[cfg(feature = "nan-ci-mode")]`-gated tracker
  populated by an OUTER constructor/builder that calls the existing,
  UNMODIFIED setters and separately records touch state. The setters
  themselves (`with_topology` / `with_reasoning_band` / `with_w_slot`) never
  change body or behavior in any build.
- **The mantissa** (partial exception): `to_mantissa` never emits 8 of the 16
  possible raw nibbles (`-8` among them) — a real, disclosed reuse of
  already-unused value-space within the existing field width, checked before
  `from_mantissa`'s masking, the same point `BandReading::project_truth`'s
  `debug_assert!` already occupies.
- **Every other byte-range field** (frequency/confidence, S/P/O): a literal
  in-value canary fill applies as originally described below — these are read
  as raw bytes, never decoded through an exhaustive enum match, so a chosen
  canary risks only a documented, avoidable VALUE collision, not the
  saturated fields' guaranteed STRUCTURAL impossibility.

**⊘ "Avoidable" is not yet "avoided" (codereview finding, confirmed valid —
open, needs an implementation-time decision before `D-NCI-1`).** `pack`/
`pack_v2` accept unrestricted `u8` for S, P, O, frequency and confidence, so a
literal canary is, today, just a `u8` value picked without a stated exclusion
rule — a legitimate producer emitting that exact byte would be misread as
absent. This plan does not yet name the sentinel(s) or a producer-side
invariant that rules them out (§6 gate G7 below makes this a pre-registered
requirement rather than an implementation afterthought); the choice itself
(which byte, and whether it is a true exclusion or a measured-and-accepted
low-probability collision) is `D-NCI-1`'s to make, with a collision test as
its own evidence, not this plan's to pre-decide in the abstract.

`pack()` fills the plain byte-range fields with the canary, likewise
`CausalEdge64::ZERO`, `Default::default()`, `from_v1_tail_unstated`, and the
equivalent SoA/tenant initialisers; the three saturated fields and the
mantissa use their own mechanisms above. Anything still reading absent was
**never stamped by this producer**. Classic poisoned-memory technique,
generalised to fields that have no free value to poison with.

- Reach: producer-side gaps — which is what an *ABI* debt is.
- Works on **every** field — the plain byte-range fields directly, the
  saturated fields via the sidecar, the mantissa via its disclosed reuse.
- Zero storage change, zero encoding change, zero release cost — the sidecar
  is a CI-only type that never exists in a release build.
- Cannot answer "was this ever stamped by anyone, ever" for a **persisted** row:
  a stored zero and a never-written zero are the same byte on disk, and the
  sidecar does not survive a round-trip to storage either.

**⊘ Boundary named, not yet enforced (codereview finding, confirmed valid —
open, needs an implementation-time decision before `D-NCI-1`).** The canary
write itself is an in-memory mutation of the plain byte-range fields at
construction time, in a CI build — it happens BEFORE the read-side observer
runs, and N1's own wording ("Release builds are byte-identical... Only a
CI/verbose build differs, and only in what it OBSERVES") does not on its face
authorize a CI-build WRITE-side change; it was written with the read-side
compare in mind. Concretely: if a CI-built binary's canary-poisoned object is
ever serialized — a saved test fixture, a CI-produced Lance snapshot compared
across runs, anything that outlives the process that poisoned it — the canary
bytes leak into what is supposed to be release-shaped data. **This plan does
not yet resolve that boundary; `D-NCI-1` owes one of:** (a) restrict poisoned
objects to construction-and-immediate-read within one CI process, with a
debug assertion refusing any serialization call on a still-poisoned object
before that boundary is crossed, or (b) treat N1 as scoped to release-vs-CI
*decode* only (as its own text literally says) and accept that a CI build's
in-memory representation may differ, so long as nothing CI produces is ever
consumed outside that same CI run. Either resolution is compatible with N6
("not switched off after certification") and N2 (observe-and-fail only); the
plan currently asserts neither explicitly, and should before `D-NCI-1` lands.

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
consumer projects it through the contract (not a raw accessor), **that path has
been observed to execute at least once** (a positive read count, not merely the
absence of canary reads), and the CI mode reports zero canary reads across those
observed reads. Progress is then a countable fraction, not a feeling.

**⊘ Gap closed (codereview finding, confirmed valid):** "zero canary reads" alone
is necessary but not sufficient — a dead or never-exercised consumer path ALSO
reports zero canary reads, for the same reason a light switch nobody has flipped
reports no failures. Certifying that would be certifying silence, not
correctness (the same `can-fire`/`can-stay-silent` pairing N4 already requires
of every guard in this plan, applied here to the certification criterion
itself). The fix is the added clause above: `D-NCI-3`'s allowlist-seeding census
and `D-NCI-4`'s per-wire certification both need a read-count, not just a
canary-count, before marking a wire green.

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

> **Council-hardened, awaiting operator confirmation (2026-09-10) — this
> mechanics description assumes §3.6's `#[cfg]` design, not the hybrid.**
> If §8 item 3a's hybrid is confirmed, the allowlist described above STOPS
> being a bare shrinking file and BECOMES a property of the `Activation`
> resolved once at plug-time (per §3.6's own already-drafted consequence:
> "§4.2's allowlist stops being a separate file and becomes a property of
> the activation"). Steps 1-4 above would then read as: Wave 0 populates
> per-classid certification state on `Activation` rather than a
> free-standing table; step 3's enforcement is the `#[cfg]`-gated abort
> half of the hybrid, resolved against that state once per activation, not
> re-read per decode. Not rewritten as fact here because the hybrid is not
> yet operator-confirmed — see §8 item 3a.

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
| `D-NCI-5` | `SpoHead` reclaim: it is a v1-shaped mirror (dead `temporal` byte) of a v2 carrier. Preservation fix under `I-LEGACY-API-FEATURE-GATED`, not a feature. | **none — splits into its own PR, council-hardened 2026-09-10 (§8 item 4)** |

`D-NCI-1..3` are the instrument. `D-NCI-4` is the first repayment.
**`D-NCI-5` no longer depends on `D-NCI-4`** — it needs none of the mode's
machinery, is the same class of fix as the five Sprint-11
`I-LEGACY-API-FEATURE-GATED` catches, and ships as its own independent PR,
before or alongside `D-NCI-1`, per the council resolution at §8 item 4.
Nothing beyond `D-NCI-4` is planned here on purpose — the census decides the
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
- **"`0` = absent" here means canary-detected absent, never a blanket
  reinterpretation of every stored zero (codereview finding, confirmed
  valid).** §3.3's own Design 1 already discloses the limit this probe must
  respect: "a stored zero and a never-written zero are the same byte on
  disk" for an ALREADY-PERSISTED row — poison-fill cannot tell them apart
  there, and nothing here changes that. The probe's counterfactual arm is
  legitimate ONLY over freshly-constructed, CI-instrumented objects where the
  canary (or the saturated-field sidecar) makes "never written" a real,
  distinct signal from "written as zero" — never over rows loaded from
  storage, where `0` remains a value, full stop. A future implementation
  scopes the probe's input to canary/sidecar-tagged objects explicitly; it
  does not run this arm against persisted corpora.
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
| G7 | **Byte-range canaries are collision-audited, not merely disclosed** (codereview finding, added post-council). Each plain byte-range field (S/P/O, frequency, confidence) that adopts a literal in-value canary names its exact sentinel value(s) and a producer-side exclusion invariant, backed by a test proving the sentinel is distinguishable from every value a real producer emits. | A canary shipped without a stated exclusion invariant, or without a collision test, is not ready — block until named. |
| G8 | **The sidecar has one stated ownership rule per edge instance** (codereview finding, added post-council), covering `Copy`, array/`Vec` storage, `ZERO`/`Default` resets, and direct setter calls that bypass any outer constructor. | A sidecar read that can return another edge's touch state, or silently under-reports touch state on a bypassed path, is a defect in the mechanism itself — block until the ownership rule is written down. |

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

## §8 COUNCIL-HARDENED RESOLUTION (awaiting operator confirmation)

> 5+3 hardening council, 2026-09-10 (run record: `.claude/board/AGENT_LOG.md`).
> **None of R1-R5 below is a ruling.** Each is a committed, savant-verified,
> reviewer-attacked proposal — the council's job was to harden a specific
> resolution well enough that the operator's confirmation is a yes/no, not a
> re-derivation. Nothing here authorizes writing `D-NCI-1`'s Rust code
> (Non-Goal, this council's own spec §3 item 1).

1. **R1 — The canary value(s): two mechanisms, by field shape.**
   `TrustTexture`/`CausalTopology` (2-bit), `ReasoningBand` (3-bit), and the
   W-slot (6-bit) are each independently re-verified **fully saturated** —
   every bit pattern already names a legitimate variant or slot, zero spare
   codes in any of them (checked against the complete enum bodies in
   `layout.rs`, not a truncated excerpt, by two separate agents across the
   council's two phases). A poison-fill VALUE therefore does not exist for
   these three fields. Resolution: a **CI-build-only, out-of-band sidecar**
   (`#[cfg(feature = "nan-ci-mode")]`-gated, e.g.
   `TouchedTail { topology_set: bool, band_set: bool, w_set: bool }`),
   populated by an OUTER constructor/builder that calls the EXISTING,
   UNMODIFIED setters and separately records touch state — never by adding a
   branch inside `with_topology` / `with_reasoning_band` / `with_w_slot`
   themselves, which keeps `I-LEGACY-API-FEATURE-GATED` satisfied on its
   letter (same function, same body, same behavior, always). This is a new
   TYPE, which N3 does not forbid (N3 forbids new CE64 bits / layout-version
   bumps / new address types; a tracker held entirely OUTSIDE the 64-bit
   register is none of those, and it never exists in a release build).
   *Considered and rejected:* a global aggregate counter
   (`tenant_counter.rs:29-44`'s already-shipped `LazyLock<[AtomicU64; N]>`
   pattern) — it can only answer "was this setter called anywhere this run",
   not "was THIS edge's field set before THIS read", producing false
   negatives on edges built before the first call and false positives on
   every edge after. **The mantissa** is a partial exception: `to_mantissa`
   never emits 8 of its 16 possible raw nibbles (`-8` confirmed among them),
   so a raw-nibble canary checked before `from_mantissa`'s
   `unsigned_abs() & 0x7` masking is available and disclosed as touching the
   register directly — reuse of already-unused value-space, not a new bit.
   **Every plain byte-range field** (frequency/confidence, S/P/O) has no
   saturation problem — read as raw bytes rather than decoded through an
   exhaustive enum match, a chosen canary risks only a documented, avoidable
   VALUE collision there, never the saturated fields' structural
   impossibility. See §3.3 for the mechanics text.

   **⊘ Ownership left unspecified (codereview finding, confirmed valid — open,
   needs an implementation-time decision before `D-NCI-1`).** `CausalEdge64`
   is a plain public `Copy` `#[repr(transparent)]` value with public setters;
   the sidecar as described has no stated rule for WHICH edge instance a
   sidecar entry belongs to once that guarantee is exercised. Concretely,
   unaddressed here: (a) a `Copy` of an edge — does the copy's sidecar entry
   move, alias the original's, or start fresh (correctly reporting the copy's
   own fields as untouched, even though the bits were copied touched)? (b) an
   array/`Vec<CausalEdge64>` of edges, or `CausalEdge64::ZERO`/
   `Default::default()` used as a reset — same question, at scale. (c) a
   DIRECT call to `with_topology`/`with_reasoning_band`/`with_w_slot` that
   bypasses whatever "outer constructor" the sidecar is populated through —
   nothing in the type system forces a caller through that constructor, so a
   direct call risks a stale sidecar entry silently under-reporting touch
   state (a false absence, the OPPOSITE direction from a poisoned false
   presence, and arguably worse: it would suppress a real ABI-debt finding).
   The mechanism needs a keying scheme (edge identity, not edge VALUE — two
   edges with identical bits are not the same provenance) before `D-NCI-1`
   can implement it soundly; naming that scheme is implementation work, not a
   plan-text decision, but its ABSENCE is a real gap this plan should not
   paper over.
2. **R2 — Failure granularity: aggregate during Wave 0, per-test after.**
   Wave 0 (census-only, N5) reports one aggregate table, no test fails —
   matching §4.2 step 1 exactly. Once the allowlist exists (post `D-NCI-3`),
   a NEW violation fails the specific test/call site that produced it, not
   the whole job — confirmed orthogonal to N5's shrink-only rule (the
   allowlist's membership direction and a failure's reporting granularity are
   independent), and consistent with `G3`/`G4`'s own per-instrument phrasing.
3. **R3 — Scope of the first wave: three crates.** `causal-edge` +
   `lance-graph-contract` + `cognitive-shader-driver` — the third crate
   carries the one artifact with real production shape today,
   `apply_edges`'s live `w_slot` filter (`mailbox_soa.rs:355`), even though it
   currently lacks a live caller. Certifying that wire first (as `D-NCI-4`'s
   target) demonstrates the mode's value on the wire most likely to matter
   once `apply_edges` gains a caller, rather than one dormant on both ends.
3a. **R4 — `#[cfg]` vs hot-plug vs hybrid: THE HYBRID.** Chosen over pure
   `#[cfg]` because the hybrid captures every real win hot-plug offers
   (per-classid certification matching §4.1 exactly; `Activation`'s
   fail-closed-by-construction with no `Default`; the existing
   `ActivationDrift`/`verify_against_mirror` drift machinery) at zero cost to
   N1 — hot-plug carries the POLICY, `#[cfg]` carries the ENFORCEMENT, and
   release resolves the policy once at activation with no new per-read
   branch. Chosen over pure hot-plug because relaxing N1 is a capability this
   campaign has not measured the re-pricing for (§3.6 already says so). This
   restates the already-drafted §3.6 paragraph — §3.6 is explicitly not
   itself a frozen decision (N1 is; §3.6's choice among its alternatives is
   what changes one), so nothing here treats it as pre-settled. **Build-time
   constraint, confirmed novel composition** (no prior pairing of hotplug
   with a `#[cfg]`-gated enforcement half exists in this tree, and
   `ActivationDrift` carries no `#[non_exhaustive]`): whoever builds
   `D-NCI-1` under this hybrid must NOT add a new `ActivationDrift` variant —
   the certification signal belongs on a new field or sibling type.
4. **R5 — `D-NCI-5` (`SpoHead`) splits into its own PR.** It needs none of
   `D-NCI-1..4`'s machinery — it is the same class of fix as the five
   Sprint-11 `I-LEGACY-API-FEATURE-GATED` catches. Ships independently,
   before or alongside `D-NCI-1` (§4.4 updated). This PR is the FIRST-EVER
   `STATUS_BOARD.md` / `LATEST_STATE.md` / `PR_ARC_INVENTORY.md` entry any
   `D-NCI-*` id has had (confirmed zero prior hits in all three files) —
   whoever ships it adds the `STATUS_BOARD.md` row in the same commit.
5. **Where the runtime disposition of absence lives — R6/R7/R8, resolved at
   §9.** Absence is not only debt; at runtime it is *surprise* — the one
   surprise the current Staunen cannot see (`nars/ghost_prior.rs:152-157`
   `GhostEcho::Staunen`; Staunen has no primitive carrier, not among the 17
   `AXIS_LABELS`, `qualia.rs:28-46`).

## §9 THE RUNTIME DISPOSITION — resolved, splits into its own companion plan

> Council-hardened 2026-09-10, same run as §8. **R8's conclusion: this
> section's content SPLITS out of `nan-ci-mode-v1`.** The CI-side disposition
> (§4's campaign, `D-NCI-1..5`) and the runtime-side disposition below share
> ONE detector but need nothing from each other to ship, and the runtime side
> needs Staunen/free-energy wiring this plan's own deliverables don't touch.
> What follows is the RATIFIED CONTENT for that companion plan, kept here
> until it is filed, so the resolution is not lost between council and
> filing. **Awaiting operator confirmation, same as §8** — this is not a
> ruling.

The CI mode and a runtime absence-signal are **the same detection with two
dispositions**, and neither this plan nor its companion forks the detector to
accommodate the second:

| context | absence means | response |
|---|---|---|
| CI | debt — a producer that never stamps | fail the build (§4.2 allowlist) |
| runtime | surprise — unhydrated | raise Staunen -> gather |

### 9.1 The existing three-way split, and where R7's new state attaches

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

The 2-bit field already encodes degrees of epistemic murk under either lens —
`TrustTexture` = `Crystalline / Solid / Fuzzy / Murky`, `CausalTopology` =
`Direct / IndirectKnown / IndirectUnknown / Unknown`. Absence is the limiting
case of the same axis: `Unknown` is *"I do not know the path"*; a refused read
is *"I was never told"*. It reads as a fifth state **without costing a bit**,
because it is the absence of the field rather than a value in it.

**R7's `RetryExhausted` (§9.2) is NOT a fourth contract-level sibling in the
table above — it is a report-time refinement of the `EdgeProvenance::Unknown`
row**, marking the moment that row's "still resolving" phase is deemed to
have run out of chances. This table does not yet represent that refinement's
crossing from the runtime context into a CI-facing outcome; whoever files the
companion plan adds that row, rather than silently folding the refinement
into one of the three existing rows.

### 9.2 R6/R7 — resolved: route via free energy, with a labeled third disposition on exhaustion

`mul::GateDecision::Hold` returns `None` from `advance_on_gate`, and the owner
is HELD and re-polled — mechanically *"stay, gather more"*. That is the
hydration response, already shipped.

**R6 — does the refusal raise Staunen directly, or via free energy: VIA FREE
ENERGY.** Staunen has no primitive carrier today, and a direct write would
either widen a fixed-width qualia column or overload an existing axis's
semantics — exactly the kind of new-surface cost this plan avoids everywhere
else (Design 1 over Design 2, the hybrid over new bits, the sidecar over new
fields). The PATTERN is precedented (`EPIPHANIES.md`, 2026-04-24, "SMB as
cognitive-stack testbed": a missing required property already routes to free
energy rather than a hard fail — the same pattern, in an unrelated
subsystem, not a literal reuse). **What is not yet true, checked rather than
assumed:** `ghost_prior::echo_for` — the function that actually raises
`GhostEcho::Staunen` — is exercised only by its own test module;
`cognitive-shader-driver/src/driver.rs`'s live `FreeEnergy::compose` call
never feeds it. R6 names the right mechanism and the right precedent for its
shape; the specific wiring from a `BandReadError::UnknownProvenance` refusal
through `kl` into `echo_for` is new plumbing the companion plan must build,
not something already connected that a refusal merely joins.

**R7 — retry policy: `N_RETRY_CYCLES = 3`** (a POLICY PIN needing its own
later measurement, following `DEFAULT_FREQUENCY_BAR`'s own precedent),
counted in mailbox cycles (`MailboxSoA::current_cycle: u32`), not wall time.
The one real count-based give-up precedent in this tree
(`supervisor.rs::ESCALATION_CRASH_COUNT = 10`) pairs its count with a
wall-time backoff interval (100ms -> 30s) because a ractor respawn crosses
an async I/O boundary; a mailbox hydration retry does not — it is already
paced by the substrate's own cycle cadence — so only the give-up THRESHOLD is
borrowed from that precedent's shape, deliberately not its backoff mechanism.
**After `N_RETRY_CYCLES` failed attempts, the disposition becomes a labeled
THIRD state, `RetryExhausted`** — never folded into either `never-declared`
(the doctrine quoted in §9.1 forbids exactly that fold) or `still-hydrating`.
It still fails the CI/audit report (the practical "stop looping forever"
outcome R7 wants), but the report LABELS it separately from the debt case, so
a reviewer can tell "nobody ever touched this wire" apart from "this wire
tried to hydrate and gave up" — the two mechanisms are structurally different
(D-NCI-1..3's static census/allowlist vs. this dynamic runtime signal) and
**not yet connected**; building that connection is itself companion-plan
work, not something the existing instrument already does.

**R6/R7 interaction, stated as a requirement on the companion plan, not a
claim about current behavior:** the companion plan MUST allow Staunen to
fire on every still-unresolved read within the retry window, not only once
`RetryExhausted` is reached — a read that has not yet happened is surprising
each time it is observed, and `N_RETRY_CYCLES` governs only when the
CI/audit disposition stops calling it "not yet"; it must never gate whether
Staunen may fire earlier.

### 9.3 R8 — this section files as its own plan

Splits per the reasoning at the top of this section. When filed, the
companion plan carries §9.1-§9.2 above, forward-referenced from here; §8
item 5's cross-reference is updated to name that file once it exists.

## §10 BOARD HYGIENE

Per `CLAUDE.md`'s Mandatory Board-Hygiene Rule, a PR carrying a new integration
plan owes a PREPEND to `.claude/board/INTEGRATION_PLANS.md` — discharged in the
same commit as the 2026-09-10 council landing (§8/§9 above).

The council run itself (5 savants, 3 reviewers, verdict counts, the v1→v2→v3
change ledger) is recorded in `.claude/board/AGENT_LOG.md`, per
`.claude/agents/5plus3-council.md`'s Phase-5 requirement — that PREPEND, not a
section inside this plan, is the authoritative run record.
