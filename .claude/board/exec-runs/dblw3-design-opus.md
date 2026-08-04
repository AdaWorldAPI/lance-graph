
## 2026-08-04 — [Opus filigree / design-only] D-BLW-3 / D-KIA-D3 — the Horizontverschmelzung fusion falsifier

**Branch:** `claude/x265-x266-plans-review-h9osnl`. **Scope:** THIS FILE ONLY.
No cargo was run. No `.rs` file was written. `crates/jc` and
`crates/lance-graph-planner/src/temporal.rs` were read and NOT modified.
`AGENT_LOG.md` read, not written.

**Reads performed in full:** `.claude/plans/cycle-loop-closure-driver-v1.md`
§12 (all of §12.1–§12.7 incl. §12.3a/§12.3a′/§12.3a″/§12.3b/§12.3c/§12.6);
`/tmp/kanban64k.md` (whole file); `/tmp/primer.md` §3 + §5.7;
`crates/lance-graph-planner/src/temporal.rs` (all 870 lines);
`crates/lance-graph-planner/examples/blw_tenant.rs` (all 1,056 lines);
`crates/jc/src/stats.rs` (κ / ω / φ / `BinaryAssociation` / KR-20 surfaces +
their degeneracy contracts); `CLAUDE.md` § falsifiability rule;
`crates/lance-graph-contract/src/kanban.rs` §DAG; `mailbox_soa.rs`
`write_row` / `set_populated`.

Everything below is stated as *checked at file:line* or marked **UNVERIFIED**.

---

# ⊘ CORRECTION (same session, from the parallel inventory lane) — the mechanism changed, and I had made the substitution it warns about

**A first draft of this note mapped `knowable_from()` to a PER-VERSE corpus-entry
version. That was wrong and is retracted here rather than quietly reshaped.**

`temporal.rs:324-325` defines it verbatim as *"The schema-frame clock (when this
row's **class** became knowable). Sourced by `ogar-adapter-surrealql`'s
`DEFINE TABLE` registration."* — a **class-level registration clock**. A
single-class verse corpus has exactly one such event, so the field is
**CONSTANT across every row**, and per-verse variation would be an invented
semantic. The primer's own iron rule condemns what I did in one line: *"**Do not
substitute the nearest available version field for a missing one.** Read-at ≠
snapshot ≠ changed-at ≠ observation identity"* (primer §3). The missing field is
`per-row last-modified version`, which primer §3 lists as **MISSING**; I reached
for the nearest available one. That is the error, and it is the same shape as the
errors this arm has been recording all day.

**What replaces it — the mode axis, which has a shipped precedent test:**

> **a-priori = a `Strict` read pinned at `V_pin`. hindsight = an `Aware` read at
> the SAME `V_pin`, over the SAME row set.** Only the reader's *permission*
> varies.

Re-verified in source before adopting:
- `classify` (`temporal.rs:185-199`) decides in order: `knowable_from > ref` →
  `Unknowable`; `row_version <= ref` → `Contemporary`; `mode == Retro` →
  `Spoiler`; else → `Anachronistic`.
- `admits` (`temporal.rs:103-112`): `Strict{Contemporary}`;
  `Aware{+Anachronistic}`; `Retro{+Spoiler}`; **`Unknowable` refused in every
  mode**.
- The mode is **not independently settable** — `QueryReference::at(v, rung)`
  derives it via `for_rung` (`temporal.rs:167-175`, `:91-97`) and always sets
  `server_id = 0`, `hlc_tick = None`. The harness selects the mode **by choosing
  the rung**.
- `no_hindsight_streamed_known_game` (`temporal.rs:791-869`) **is already this
  experiment**: 10 plies, `lance_version == ply`, `knowable_from == 0` for ALL
  rows; a Strict reader at `v` sees exactly `0..=v`; a reader at the SAME `v`
  with a hindsight-admitting mode sees past it.

**Two precision points I owe back to the correcting lane:**
1. *Adopted with one substitution — `Aware`, not `Retro`.* With `knowable_from`
   constant and `≤ ref`, **`Aware` and `Retro` admit the IDENTICAL set**: a
   future row classifies `Anachronistic` under Aware (admitted, `:106-108`) and
   `Spoiler` under Retro (admitted, `:109`), and no row exists that only Retro
   admits. So the measurement is the same either way — but `Aware`'s own doc is
   literally *"May also use `ANACHRONISTIC` rows — **hindsight** from a future
   frame"* (`temporal.rs:80`), while `Retro` is *"an **intentional** `V_now` read
   past the horizon"* (`:82-83`) — a deliberate spoiler, which is a different
   Gadamerian object. **Hindsight = rung 5 (`Aware`).** Gate **G1c** asserts the
   extensional identity with rung 9 as a free, falsifiable check.
2. *The cited `temporal.rs:214` for `classify`'s first check is off by ~24 lines.*
   `classify` is `:185-199` and the `knowable_from > ref_version` branch is
   **`:190`**; `:214` is `pub struct DepEdge`. Substance unaffected; recorded so
   the wrong number does not propagate into the harness's doc comments.

**What SURVIVES the correction unchanged, and why that matters:** blocker **B2**
— the shipped D-BLW-1 series cannot move under *any* read — is untouched, and so
is prerequisite **P1** (incremental seating), though **P1's justification
changes**: it is now required because the *ranking pool must grow*, **not**
because `knowable_from` needs a per-verse value. **P2** (a horizon-relative
criterion) is likewise untouched: if every horizon's verdict is identical, a
Strict fold and an Aware fold over the same rows return the same verdicts and
`Δ ≡ 0` by construction. The mode axis changes *how the two horizons are read*;
it does not create *something to read*.

**What CHANGED below:** §1.2 (`knowable_from` is constant, and states it),
§1.4 (P1's justification), §3.6 (pinned parameters: one pin `V_pin = V4`, mode
contrast rung 0 vs rung 5), §4 G1/G2/G3 (G3 is rewritten from a can-fire test
into a *not-exercised* disclosure), §5 (the mechanism), §6 item 9, §7 B5/B7 and
a new **B9**. Nothing else moved.

---

# 0. HEADLINE — a re-scope, stated before the design

**§12.3's D-BLW-3 row, as literally written, cannot be built.** Its declared
input is *"pairwise lens agreement"* between the four stances. That input is
dead three times over, all recorded in the plan itself:

- §12.3a″ **MEASURED**: 3 of 4 stances are UNREACHABLE on the SPO/TSV path; the
  single formable pair had **both** lenses DEGENERATE, so both ∃-quantifiers
  were false *by construction, not by measurement*.
- §12.3c **retired κ over per-verse binaries**; §12.7 recorded the texture
  rewrite as a **KILL** (3 loci of 24 written, agreement capped at 1 before any
  verse was read).
- The D-BLW-2 rebuild (`examples/blw_binding.rs`) is, by its own tag file,
  **NOT COMPILED, NOT RUN**.
- And structurally: the stance instruments run over a TSV, **not over the
  tenant**. §12.7 defect 2 is precisely that a harness bypassing the substrate
  cannot be evidence for a substrate claim. This brief scopes D-BLW-3 to *"the
  sealed version series produced by the D-BLW-1 tenant"* — so a stance-fed
  D-BLW-3 would reintroduce the defect §12.7 just recorded.

**Recommendation, and the whole design below assumes it:** re-scope D-BLW-3's
inputs to the *kanban-64k plan's own* D3 wording — **"two ontology projections
of one cohort (e.g. two catalog-derived criteria views)"** — with the
projections defined over the **tenant's own row data**. The four-stance framing
was §12.3's narrowing of D3, not D3 itself. This is a **reduction**, recorded as
one, with reasons. §7 carries it as blocker **B1**.

---

# 1. The verdict row shape

## 1.1 The type

```rust
/// One (verse, horizon, projection) verdict. Emitted as the sealed series is
/// produced; never reconstructed from a version.
#[derive(Clone)]
struct VerdictRow {
    /// The tenant row's stable verse reference, e.g. "kjv:00417".
    /// Owned `String` because `DeinterlaceRow::subject(&self) -> &str`.
    subject: String,
    /// The SEALED VERSION this verdict was COMPUTED FROM.        → lance_version()
    horizon: u64,
    /// A | B | Z  (see §2)
    projection: Proj,
    /// The binary verdict itself.
    verdict: bool,
}

impl DeinterlaceRow for VerdictRow {
    fn subject(&self) -> &str            { &self.subject }
    fn lance_version(&self) -> LanceVersion { self.horizon }
    /// CONSTANT. The verdict-row *class* is registered once, at series start.
    /// This is a class-registration clock, NOT a per-row warrant time — see
    /// the ⊘ correction block. `0` matches the shipped
    /// `no_hindsight_streamed_known_game` fixture (`temporal.rs:799-801`).
    fn knowable_from(&self) -> LanceVersion { 0 }
    // hlc_tick() DEFAULTED — see §1.3.
}
```

There is deliberately **no `entered` field.** A per-verse entry version exists in
the harness (it is what drives the ranking pool, §2.3) but it is **not** exposed
through `DeinterlaceRow`, because no method on that trait means it.

## 1.2 What the two clocks MEAN in this corpus — and the corpus justification

**`lance_version()` = the horizon the verdict was computed from.**
`temporal.rs:322` defines it as *"The storage-frame clock (this row's Lance
version)"*. This verdict row **comes into existence at the seal of that cycle** —
it is the record of a reading performed from that frame. Stamping it with any
other version would be a false statement about when it was computed. There is
no competing reading.

**`knowable_from()` = a CONSTANT, and this design says so out loud.**
`temporal.rs:324-325` defines it as *"The schema-frame clock (when this row's
**class** became knowable). Sourced by `ogar-adapter-surrealql`'s `DEFINE TABLE`
registration."* — a **class-registration** event. This corpus has **one** class
of verdict row, therefore **one** such event, therefore one value for every row.
It returns `0`, matching the shipped `no_hindsight_streamed_known_game` fixture
(`temporal.rs:799-801`, `Row::new(ply, 0, None)`).

**Consequences, stated rather than hidden:**

- `knowable_from ≤ ref_version` for every row at every pin the design uses, so the
  `Unknowable` branch of `classify` (`temporal.rs:190`) **never fires**. The
  `TemporalStatus::Unknowable` state and the `admits`-refuses-it-in-every-mode
  property (`temporal.rs:110`, asserted at `:646-652`) are **not exercised by this
  harness**, and §4 **G3** discloses that instead of faking a gate for it.
- **`knowable_from` carries ZERO information in this corpus** and contributes
  nothing to any κ. Any result line implying otherwise is false.
- **A per-verse entry version does exist** in the harness — it is what determines
  the ranking pool (§2.3) — but it is deliberately **not** routed through
  `DeinterlaceRow`, because no method on that trait means "when this row's
  subject entered the corpus". The field primer §3 lists as **MISSING**
  (*per-row last-modified version*) stays missing; it is not simulated.

**Why the two clocks being one constant apart is fine here:** the a-priori /
hindsight contrast does **not** run on the clock axis at all. It runs on the
**mode** axis (§5), where `Strict` and `Aware` differ in what they admit at one
identical pin over one identical row set. The clocks' job is only to place each
verdict row at the horizon it was computed from — which `lance_version` does
alone.

## 1.3 `hlc_tick()` stays defaulted `None` — deliberately

Receipt: primer §3 — the HLC mechanism is *implemented and tested* but
**production wiring is dormant**; no substrate call site sets a non-zero
`server_id` or `Some(hlc_tick)` (`temporal.rs:126-151` hardcode `0`/`None`).
Setting one here would fabricate a cross-server clock this arm has no writer for.

**Consequence, stated:** `deinterlace`'s sort key
(`temporal.rs:369-374`) degenerates to `(lance_version, lance_version)`, i.e.
**horizon order**. That is correct and sufficient here — the fold in §5 depends
on exactly that ordering, and gate **G7** proves it is the sort and not the input
order doing the work. **No HLC or multi-writer claim is made.**

## 1.4 THE PREREQUISITE THE SHIPPED HARNESS DOES NOT MEET — read this before building

Two facts about `blw_tenant.rs` as it stands make a naive D-BLW-3 **vacuous**:

1. **All verses are seated before V1.** `seed_tenant(&mut owner, &verses)` runs at
   `blw_tenant.rs:615`, *before* the cycle loop at `:758`. So the reader's
   knowable corpus is **identical at every horizon**, the ranking pool never
   grows, and a rank criterion (§2.3) returns the same verdict at V1 as at V8.
2. **The row body is horizon-independent.** The sweep is a bloom-containment read
   against a FIXED probe (`probe_plane("god", 0)`, `:688`) over content planes
   that are **never rewritten after seeding** — the only per-cycle write is
   `temporal` on fired rows (`stamp_fired_rows`, `:569-582`). Therefore the
   `fired` set is **identical at every cycle**, a per-verse verdict derived from
   it is **constant across V1..V3**, and `κ_apriori ≡ κ_hindsight` **exactly**.
   The "drop the distinction" branch would fire without measuring anything.

So D-BLW-3 requires two changes to the harness shape. **Both are minimum
conditions for the concept to be instantiated at all — neither is a trick:**

- **P1 — INCREMENTAL SEATING.** Verses are seated in `S` slices, one slice per
  sealed cycle, **strictly between the post-apply of cycle `c` and the pre-eval
  snapshot of cycle `c+1`**, so `blw_tenant.rs`'s intra-cycle "untouched
  remainder is byte-identical" falsifier (`:783-806`) is untouched. `populated`
  grows via `set_populated`, which its own doc sanctions as caller-managed
  (*"This is a declaration, never an implicit per-write counter — callers manage
  it explicitly"*, `mailbox_soa.rs:487-497`). Writes use the owner's
  `current_cycle` so `write_row` returns `Accepted` (`mailbox_soa.rs:418-462`).
  **Justification, corrected:** P1 is required because the **ranking pool must
  grow** — that is the only thing that makes a verdict horizon-dependent. It is
  **not** required in order to give `knowable_from` a per-verse value (that
  mapping is retracted; see the ⊘ correction block). Without P1, "what a reader
  could know at V4" is the whole corpus and Gadamer has no purchase — under
  *any* mode.
- **P2 — A HORIZON-RELATIVE CRITERION.** At least one projection's verdict must
  be a function of *the corpus as known at the horizon*. Otherwise the verdict
  cannot move and the falsifier is decoration. §2.3 is where the manufacturing
  risk actually lives, and it is addressed there explicitly.

---

# 2. The two projections being fused

## 2.1 What the tenant actually carries (and what must NOT be used)

Per-row columns as seeded at `blw_tenant.rs:533-546`:

| column | value | usable as a criterion? |
|---|---|---|
| `content` plane | bloom of the verse's tokens, salt `0` | **YES** — the only real per-verse signal |
| `topic` plane | bloom of **the same tokens**, salt `0xA5A5_A5A5` | **NO** — a second hash of the same data |
| `energy` | `text.len() * 0.01` | **NO** — same variable as `meta` |
| `meta` | `text.len() & 0x00FF_FFFF` | **NO** — same variable as `energy` |
| `entity_type` | `row % 251` | **NO** — arbitrary |
| `temporal` | row index, then a per-cycle stamp | **NO** — a harness artifact |

**Explicitly rejected:** "content vs topic" as the two projections. They are two
salted blooms of one token set, so their κ would measure hash collision, not two
horizons — the purest form of the landmine the brief names (*two loci distinct by
RULE, identical by DATA*; measured 27/27 in the prior arc). Same for
`energy` vs `meta`, which are literally an affine image of each other.

## 2.2 The two projections

Both are per-verse binary criteria over the **content identity plane**.

- **Projection A — the SIGHT basin.** Seed set
  `S_A = {eyes, opened, knew, know, saw, see, wise, understanding}`.
- **Projection B — the BODY basin.** Seed set
  `S_B = {naked, nakedness, ashamed, shame, clothed, covered, garment, skin}`.

**Provenance of the seed sets, and its limit.** Both are drawn from §12.6 **A1**'s
pre-registered anchor vocabulary (Gen 2:25 / Gen 3:7) **for provenance only** —
they were written into the plan *before* this instrument was designed, so they
cannot have been selected to make a number move. **NO A1 CLAIM IS MADE.** A1's
own text says a lexical instrument sees "naked" in both verses and scores them
similar; this instrument **is** lexical, so it is exactly the kind A1 predicts
fails on 55↔62. D-BLW-3 does not attempt A1 and must not be reported as
bearing on it.

**Why these two and not another pair:** they are (a) semantically distinct
lexical fields, so κ ≈ 1 is not forced; (b) drawn from one passage's field, so
κ ≈ 0 is not forced either; and (c) fixed in advance, so neither can be swapped
after seeing a table. Whether they collapse *in this corpus* is an empirical
question that gate **G4-COLLAPSED** answers rather than an assumption.

## 2.3 The criterion FORM — top-quartile rank, and why not a threshold

For projection `j` at horizon `V`:

1. `score(i, j) = popcount(content_plane[i] & bloom(S_j))` — the number of
   seed-bloom bits the verse's plane covers. The plane is a **borrowed row slice**
   (`view.identity_plane_at(row, IdentityPlane::Content)`, exactly as
   `sweep_rows` at `blw_tenant.rs:317-346`); the score is an **owned `Copy` u32
   microcopy**. `data-flow.md` §1/§2 respected; no `&mut self` during
   computation.
2. **The RANKING POOL is every tenant row seated by horizon `V`** — everything the
   reader at `V` could know. The per-verse seating version is a **harness-internal
   value** used at *emission* time to build the pool; it is deliberately NOT
   exposed through `DeinterlaceRow` (§1.2).
3. `verdict(i, j, V) = true` iff `score(i, j)` is in the strict top `q` of the
   **pool as of `V`**, **ties broken by ascending row index**. The tie-break is arbitrary but
   FIXED and documented: popcount ties are common at small scores, and an
   unspecified tie-break would make the verdict depend on iteration order.

**The anti-manufacture argument, in three steps:**

- An **absolute threshold** on a horizon-independent score is horizon-independent
  ⟹ cannot move ⟹ decoration.
- An **accumulating-reference** criterion (verdict = overlap against the OR-bundle
  of everything seen so far) *does* move — but **monotonically toward
  everything-positive**. An OR-bloom saturates, the positive rate → 1, and the
  lens becomes the 99.61 % (§12.3a″) / 88.2 % (§12.7) defect. It is *a number
  that cannot help but move* — the Kant tautology one level up, which §12.3b
  names in exactly those words.
- A **rank criterion at fixed `q`** pins the **pool's** positive rate at `q` by
  construction. This is not my invention: §12.3a rescued the Kant binary the same
  way — *"Ranking is relative, so promotions and demotions balance and the
  positive rate cannot reach 1 by construction."* It can move, and it cannot
  drift into degeneracy on the pool.
- **Crucially**, the *measured* marginal — the positive rate on the **fixed
  prefix**, a sub-population of the pool — is **NOT** pinned. It moves iff early
  verses' scores sit systematically differently from later ones. That is a real,
  informative, falsifiable quantity, and it is precisely what the full
  `BinaryAssociation` table exposes.

**`q = 0.25`** — pre-registered, non-adjustable. Chosen as the coarsest standard
quantile (the upper quartile) that leaves BOTH already-pre-registered marginal
guards an order of magnitude of headroom: the degeneracy band `[0.01, 0.99]` and
the can-agree marginal guard `[0.05, 0.95]` (both §12.3a). It is a quantile, not
a fitted cut.

## 2.4 Projection Z — the INERTNESS CONTROL (not a fused projection)

`Z(i) = "the verse's content plane contains ALL bits of bloom("god")"` —
literally D-BLW-1's shipped probe (`blw_tenant.rs:688`). **Z is
horizon-independent by construction**: content planes are never rewritten after
seeding, and containment against a fixed probe consults no pool.

Z is **not** an item of the same construct as A and B and is **never** pooled
with them (see §6 on KR-20). Its single job is gate **G2**.

---

# 3. THE PRE-REGISTERED BAND — fixed here, before any run, NON-ADJUSTABLE

> **A miss is a miss.** Nothing in this section may be changed after a number
> exists. Every threshold below is either an **external convention** or a value
> **already pre-registered in the plan for a different deliverable** — which is
> the strongest available evidence that it was not fitted to this corpus. No
> threshold here is newly invented.

## 3.1 The fusion band, on κ(A, B)

| κ(A, B) | verdict | where the number comes from |
|---|---|---|
| **κ > 0.80** | **REDUNDANCY** — two names for one horizon; nothing fused | Landis–Koch "almost perfect" floor; already pre-registered as §12.3a's can-discriminate ceiling |
| **0.20 ≤ κ ≤ 0.80** | **IN BAND** — two horizons sharing structure without collapsing. The ONLY regime in which the word "fusion" is permitted | the interval between the two |
| **κ < 0.20** | **NO SHARED HORIZON** — nothing to fuse | Landis–Koch slight/fair boundary; already pre-registered as §12.3a's can-agree floor |

**Reasoning for reusing §12.3a's two numbers rather than picking new ones:** they
were fixed for the discrimination twin *before this deliverable was designed*,
for the same stated reason (an external convention that predates this corpus and
therefore cannot have been fitted to it). Choosing fresh numbers here — even
defensible ones — would forfeit that argument. Reuse is the anti-fitting move.

## 3.2 The band is evaluated at BOTH horizons — and the crossing case is adjudicated NOW

Four outcomes, named before the run so none can be adjudicated after seeing a
number:

- **IN / IN** → the only case in which the §3.3 movement result may be reported
  as fusion.
- **OUT / OUT, same side** → recorded as **REDUNDANCY** or **NO SHARED HORIZON**.
  The movement number is still printed; it may **not** be called fusion.
- **IN / OUT or OUT / IN** → **BAND-EXIT**, its own named outcome: *"the pair left
  (or entered) the fusion band between Vk and Vm."* Never silently counted as
  fusion; never silently counted as a miss.
- **κ undefined at either horizon** → see §3.2a, which separates two cases the
  first draft of this note conflated. Printed as `undefined(p_e=1)` — **never
  `0.0`, never blank, never omitted** (§12.3a degeneracy discipline).

### 3.2a `binary_association` has TWO distinct failure shapes — do not conflate them

Read at `jc/src/stats.rs:653-693`:

| shape | meaning | outcome |
|---|---|---|
| the function returns **`None`** | length mismatch or empty input — **structurally unusable** (`:654-656`) | **KILL**, naming the pair. This is a harness bug, never data. |
| returns **`Some(assoc)`** with `assoc.kappa == None` | `p_e == 1` (`:672-680`) — both raters used one identical category throughout | **NOT a kill.** The **counts remain informative** and are printed in full; the pair is stamped **UNDEFINED-κ** and cannot be band-classified. |
| returns **`Some(assoc)`** with `assoc.phi == None` | one variable is constant → zero variance (`phi` → `pearson`, `:596-600`) | printed as `undefined(constant)`; κ may still be defined. |

**Ordering matters, so it is pre-registered:** compute each projection's positive
rate → stamp DEGENERATE (§3.5) → **pair only the non-degenerate**. A constant
projection has rate `0` or `1`, outside `[0.01, 0.99]`, so it is excluded
*before* pairing. Consequently `kappa == None` on a **claimed** pair is a
**harness-ordering bug (KILL)**, while `kappa == None` on a **printed excluded**
pair (e.g. G4's artificial degenerate probe) is **expected and correct**. The
distinction is fixed here so it cannot be adjudicated after a table exists.

## 3.3 Movement (can-fire) — carried VERBATIM from §12.3b, not re-derived

- `|κ_hindsight(k, m) − κ_apriori(k)| ≥ 0.10`, **both κ defined**, fixed prefix
  `k ≥ 1000`.
- `0.10` = one-fifth of the `0.20 … 0.80` band span, per §12.3b's own rationale.

## 3.4 The distinction must earn its keep — VERBATIM from §12.3b

- If for EVERY pair and EVERY prefix `|Δκ| < 0.01`, the a-priori / hindsight
  distinction is **DROPPED from the write-up, not narrated**.
- `0.01` = the two-decimal reporting precision floor for κ.
- **Extended by one clause** (§5.3): Δκ alone cannot distinguish "identical
  reads" from "reads that churned in cancelling directions". The Hamming
  companion is mandatory and is never averaged into Δκ.

## 3.5 Guards, all VERBATIM from §12.3a

- **Corpus floor:** `N ≥ 1000` measured units. Below it the twin is not reported.
- **Degeneracy:** each projection's positive rate on the MEASURED set is computed
  *before* pairing; outside `[0.01, 0.99]` ⟹ stamped **DEGENERATE**, excluded
  from every claim, **and the exclusion printed** — never silent.
- **Unstable:** a pair with `expected_agreement > 0.95` ⟹ stamped **UNSTABLE**;
  cannot support a fusion claim.
- **COLLAPSED** (the *distinct-by-rule, identical-by-data* landmine): if
  `n01 + n10 < 0.05·N` at **both** horizons, the pair is stamped **COLLAPSED** and
  excluded from every fusion claim. Same count clause §12.3a used, for the same
  reason. Here the mechanism would be seed-term co-occurrence; the table detects
  it directly rather than assuming it away.

## 3.6 The run shape, pinned now (changing any of it after a κ exists is fitting)

- **`S = 8` sealed cycles.** The Rubicon DAG loops:
  `Planning → CognitiveWork → Evaluation → Plan → Planning → …`
  (`kanban.rs:101-109`; `Plan => &[Planning]` at `:106`) — VERIFIED, so an
  8-cycle series is two full loops and is legal.
- **`m = 2000` verses, slice = 250/cycle.** The fixed prefix `k = 1000` is
  complete after cycle 4 ⟹ **`V_pin = V4`**, and **V8 is the latest horizon
  present in the row set**. Both pinned.
- **ONE pin, TWO modes** (corrected — see the ⊘ block). Both reads use
  `V_pin = V4`; a-priori is **rung 0 → `Strict`**, hindsight is **rung 5 →
  `Aware`** (`temporal.rs:91-97`). The pin is a **middle** horizon on purpose: at
  V8 nothing is anachronistic, so the contrast would be trivially empty — which
  is precisely gate **G1**'s silence half, not the measurement.
- **The two-pin variant (Strict@V4 vs Strict@V8) is NOT used**, and the reason is
  recorded so it is not "simplified" back in later: it varies the pin *and* the
  permission at once, so a Δ could not be attributed to either. The mode contrast
  holds the pin, the row set, and the code path identical and varies **only what
  the reader is permitted to know**.
- **Feasibility receipt:** `N_CAP = 2048`, `DEFAULT_VERSES = 2000`
  (`blw_tenant.rs:107,110`). `k = 1000` clears the floor with exactly **2×
  headroom and nothing more** — see blocker **B3**.
- **Trajectory honesty:** all eight κ values are REPORTED. **No trend claim is
  made.** The movement test is a **two-point contrast** (V4 vs V8); the six
  intermediate values are reported and never fitted. A "trajectory" over 8 points
  is eight numbers, not a trend.

---

# 4. Gates — can-fire AND can-stay-silent, each with the concrete non-trivial input

> Both halves of every gate use NON-TRIVIAL input. No empty-input silence case
> appears anywhere below (`CLAUDE.md` falsifiability rule, second bullet-twin).

## G1 — THE MECHANISM: the a-priori/hindsight split IS the mode, and it discriminates

This is now the central gate, not a guard on one. `QueryReference::at(v, rung)`
derives the mode via `for_rung` and hardcodes `server_id = 0` / `hlc_tick = None`
(`temporal.rs:167-175`), so **the harness selects the mode by choosing the rung
and by nothing else**.

- **G1a — can-fire:** read the SAME full row set at `at(V4, 0)` (Strict) and
  `at(V4, 5)` (Aware). Aware must admit **strictly more** rows — the V5..V8
  verdict rows classify `Anachronistic` (`temporal.rs:190-198`) and Aware admits
  them (`:106-108`) while Strict refuses (`:104-105`) — and the folded A-verdict
  vector must **differ on at least one verse**. *Input:* the complete row set
  spanning horizons V1..V8. Non-trivial by construction.
- **G1b — can-stay-silent:** `at(V8, 0)` vs `at(V8, 5)`. No row carries a horizon
  above V8, so nothing classifies `Anachronistic` and the two reads must be
  **byte-identical**. *Input:* the same complete row set, the same code path,
  the same two rungs — **only the pin moved**. This is what proves G1a's
  difference came from the existence of future-horizon rows and not from
  mode-dependent plumbing.
- **G1c — `Aware` ≡ `Retro` extensionally, asserted rather than assumed.** With
  `knowable_from` constant and `≤ ref` (§1.2), no row exists that `Retro` admits
  and `Aware` does not: a future row is `Anachronistic` under Aware (admitted)
  and `Spoiler` under Retro (admitted, `:109`). Assert `at(V4, 5)` and
  `at(V4, 9)` return **the same row set**. *Input:* the same non-trivial full row
  set. This is free, it is falsifiable, and it would catch any future change to
  `for_rung`, `classify`, or `admits`. It is also **why `Aware` is the chosen
  hindsight mode** — the measurement is identical either way, so the choice is
  made on doctrine (`Aware`'s doc says *"hindsight"*, `:80`; `Retro`'s says
  *"intentional … read past the horizon"*, `:82-83`), not on numbers.

## G2 — the inert projection must NOT move (the mechanism-level control)

- **can-stay-silent:** `κ(Z_apriori, Z_hindsight)` over the fixed prefix must be
  **exactly 1.0** and the folded Z-vectors byte-identical. Z is
  horizon-independent by construction, so **any** movement is a plumbing leak and
  **voids the (A,B) movement result**. *Input:* the same 1000 verses, the same
  pin, the same two rungs, the same fold — **only the criterion form differs**
  (absolute containment vs rank).
- **can-fire twin:** the same comparison for projection A must produce at least
  one flipped verse — `hamming(A_apriori, A_hindsight) > 0`. If A is *also*
  frozen, the rank mechanism is not horizon-sensitive in this corpus and the
  design's central assumption is **refuted** — report that as the result; do
  **not** patch the criterion to make it move.

> G2 is the strongest control in this design: same corpus, same pin, same reads,
> same fold; one criterion is horizon-relative and one is not; only the former
> may move. It is what lets a measured Δκ be attributed to the growing **pool**
> rather than to the harness.

## G3 — `knowable_from` and `Unknowable` are NOT EXERCISED (a disclosure, not a gate)

**A first draft of this note put a can-fire/can-stay-silent pair here. It was
built on the retracted per-verse `knowable_from` mapping and is withdrawn.**

`knowable_from` is a **class-registration clock** and is CONSTANT at `0` across
every row (§1.2), so:

- The `Unknowable` branch of `classify` (`temporal.rs:190`) **never fires** in
  this harness, and the "refused in every mode" property (`:110`, asserted by the
  shipped `admits_per_mode` at `:646-652`) is **never reached**.
- Constructing an input that *would* fire it requires a **second class
  registration** — a `DEFINE TABLE`-shaped event this single-class verse corpus
  does not have. Manufacturing one would be inventing the semantic the ⊘
  correction block warns against.

**What G3 is instead — a structural assertion that keeps the disclosure true:**
assert that **every** emitted row returns the **same** `knowable_from`. That is a
real, falsifiable check (it fires the moment a future author starts varying the
field) and it costs nothing. It is deliberately **not** dressed up as a
falsifier for `Unknowable`.

**Stated for the record:** the `Unknowable` axis is already covered by
`temporal.rs`'s own shipped tests (`classify_time_axis` at `:623-636`,
`admits_per_mode` at `:638-653`). A consumer is not obliged to re-gate every
branch of a library it does not use — it *is* obliged to say which branches it
leaves untouched, which is what this section does.

## G4 — the band guards themselves can fire and can stay silent

- **DEGENERATE can-fire:** an ARTIFICIAL projection defined as `score > 0` over a
  corpus-ubiquitous seed. The harness already records that a Genesis prefix is
  ~90 % "God" (`blw_tenant.rs:704-706`). Fed through the same pipeline it must be
  stamped **DEGENERATE and PRINTED**, never reported as a projection. *Input:*
  real corpus, real pipeline.
- **DEGENERATE can-stay-silent:** A and B at `q = 0.25` must **not** be stamped —
  their measured rates must land inside `[0.01, 0.99]`. If they do not, that is a
  **result** (the fixed prefix is unrepresentative of the pool), reported, not
  tuned away.
- **COLLAPSED can-fire:** run the pair **(A, A)** — identical by construction ⟹
  `n01 + n10 = 0` ⟹ stamped **COLLAPSED**. This proves the guard reads the
  **counts**, not the labels.
- **COLLAPSED can-stay-silent:** (A, B) must **not** be stamped, i.e.
  `n01 + n10 ≥ 0.05·N`. If it IS stamped, the two seed sets are one criterion in
  this corpus and **no fusion claim may be made** — a real possible outcome,
  pre-accepted here.

## G5 — the fixed verse set is actually fixed (the §12.3b anti-sample-growth control)

**The mode contrast REINTRODUCES §12.3b's sample-growth confound, so this gate is
not optional.** At `V_pin = V4`: the Strict read admits only horizons ≤ V4, which
covers the **first 1000** verses; the Aware read additionally admits horizons
V5..V8, which covers **all 2000**. The hindsight read therefore sees *more
verses*, exactly the confound §12.3b was built to remove — and the fixed-prefix
restriction is what removes it.

- Assert the two measured vectors have **equal length** and **equal subject
  sequence** after the fold-and-restrict. *Input:* an unfixed implementation
  yields **2000 vs 1000** and the assert fires immediately. §12.3b's confound
  becomes mechanical instead of promised.

## G6 — the fold neither drops nor duplicates

- After `deinterlace` at V8, **every** subject in the fixed prefix must appear
  with **exactly 5** rows for projection A (horizons V4..V8) — `assert_eq!`, not
  `>=` (`CLAUDE.md`: *prefer `== N` over `>= N`*). The fold-to-last must then
  yield **exactly 1000** subjects. *Input:* the real emitted row set; any
  off-by-one in the emission loop or any mis-sorted fold changes the count.

## G7 — the ordering the fold depends on is real, not an input-order accident

- `deinterlace` sorts by `(hlc_tick.unwrap_or(lance_version), lance_version)`
  (`temporal.rs:369-374`); the fold takes the **last** row per subject, so it
  depends on that sort.
- **can-fire:** feed the rows in **deliberately descending horizon order**. The
  fold result must be **identical** to the ascending-input result. A fold that
  relied on input order would differ. Same falsifiability shape as
  `temporal.rs`'s own `layer1_orders_one_owners_chain_by_cast_seq_not_log_order`
  (`:586-611`).

---

# 5. a-priori vs hindsight — what makes them differ, and the identity test

## 5.1 The two reads, off the real surface

```
a-priori  = deinterlace(&rows, &QueryReference::at(V4, 0), &NoDeps)
hindsight = deinterlace(&rows, &QueryReference::at(V8, 0), &NoDeps)
```
then, for each: keep `projection == j`, **fold by subject taking the LAST row**
(the highest horizon ≤ the pin — `deinterlace` already sorted ascending), then
restrict to the fixed prefix.

Nothing is reconstructed from a version; `temporal.rs` is not modified (§12.5
holds). `QueryReference::at` is used as what it is — a **reader pin**, per
§12.3a(4).

**A precision note that must appear in the harness, not just here:** a version
*range* `Vk..Vm` is **not** expressible in one `deinterlace` call. The surface
takes a single `ref_version` and admits the **prefix** `0..=ref_version` under
`Strict`. The lower bound of any "range" is caller-side row selection, not a
`temporal.rs` capability. The `Vk` vs `Vm` contrast below is therefore
**pin-vs-pin over a fixed subject set**, which is what §12.3b's control actually
specifies — but no result line may describe it as "a range read the surface
performed".

## 5.2 What makes them differ — precisely, and only this

**The rank criterion's POOL.** At V4 the pool is verses 0..1000; at V8 it is
0..2000. A verse in the fixed prefix that sat in the top quartile of the first
1000 may not sit in the top quartile of the first 2000, and vice versa.
**The verse's own score never changed** — what changed is the cohort it is read
against. That is the mechanical content of *wirkungsgeschichtliches Bewusstsein*
in this harness, and it is the **only** source of movement: **G2 proves the
plumbing contributes none**, and §1.2 records that `knowable_from` contributes
none either.

## 5.3 The explicit "they are identical — drop the distinction" test

```
Δ(pair, k) = κ_hindsight(k, m) − κ_apriori(k)
```

- If `max over pairs and prefixes |Δ| < 0.01` → print
  **`A-PRIORI/HINDSIGHT DISTINCTION DOES NO WORK — DROPPED`**, and the write-up
  must not narrate it. Verbatim §12.3b.
- **Mandatory companion, and it is not optional:** a κ can be unchanged while the
  underlying verdicts churn — two flips in opposite directions cancel in κ. So
  report `hamming(A@V4, A@V8)` and `hamming(B@V4, B@V8)` **beside** Δκ, **never
  averaged into it**. Collapsing the two would be the exact defect §12.3c retired
  κ for (*"two lenses can agree on a verse for opposite reasons and κ scores that
  as agreement"*), reintroduced one level up.

Three named outcomes, fixed now:

| Δκ | Hamming | outcome |
|---|---|---|
| `< 0.01` | `= 0` | the reads are genuinely identical → **DROP the distinction** |
| `< 0.01` | `> 0` | **CHURN WITHOUT REALIGNMENT** — the reads differ but their agreement does not. **No fusion claim.** |
| `≥ 0.10` | (reported) | movement fires; fusion may be claimed **only if** §3.2 says IN/IN |

The middle row exists so that a Δκ of 0.004 with 180 flipped verses cannot be
written up as either a success or a clean null after the fact.

---

# 6. What is NOT claimed

1. **C3 ceiling (hard gate).** κ / φ between two projections measure **overlap** —
   a **reliability**-class statement. No external criterion is wired, so the
   public claim ceiling is verbatim: *"measurable reliability as a first step
   toward measurable awareness."* It is **NOT** claimed that the later horizon
   reads the verses **better**, **more truly**, or **more completely** — only
   **differently** (§12.3b's tightened ceiling). Validity is D3b and D3b stays
   blocked.
2. **C4 significance — NO p-value is reported, at all.** `jc::stats`' p-values are
   classical **independent-sample** p-values, and verses within one book are
   domain-correlated. `I-NOISE-FLOOR-JIRAK` names the **problem**; `jirak.rs` is a
   fingerprint-specific empirical probe and is **not** the solution (C4, as
   corrected 2026-08-04). D-BLW-3 has **no justified dependence model** for a rank
   statistic over an ordered corpus, so it **states plainly that no significance
   claim is made** rather than borrowing one. `t_test_*` / `anova_one_way` are
   deliberately not called.
3. **C2 naming discipline, binding here for the first time** (this IS the first
   binary-criteria witness): φ is reported as **φ**, never "Pearson"; κ as **κ**,
   never "ICC"; and the **full `BinaryAssociation` table** — `n00/n01/n10/n11`,
   **both** marginals, `p_o`, `p_e` — ships with **every** κ. **Never a bare κ**
   (`stats.rs:602-611` states exactly why). Spearman is omitted as redundant on
   binary data (§C2). **KR-20 is NOT reported**: two projections is below any
   meaningful internal-consistency claim, and pooling Z in would be a category
   error (Z is a control, not an item of the same construct). If a third *item*
   is ever added, KR-20 — not α — is the correct name.
4. **No stance claim.** The four stances are not inputs (§0). Nothing here bears
   on Hegel / Nietzsche / Kant / Wittgenstein.
5. **No §12.6 anchor claim** — not A1, not A2, not A3/A3′. The seed vocabulary is
   A1-derived **for provenance only**; a bloom instrument is exactly what A1
   predicts cannot separate index 55 from 62.
6. **No cross-language claim.** One lane (KJV). The 9 PD lanes exist on disk;
   no instrument for them exists here.
7. **No zero-copy claim.** `deinterlace` `.cloned()`s the admitted rows
   (`temporal.rs:363`). §12.2's precision note binds verbatim: the rows are small
   verdict records, so the cost is a selection over lightweight rows — **not
   absent**.
8. **No durability claim.** `MemWal` is in-process `Mutex`/`Vec`
   (`blw_tenant.rs:405-416`) and its own header says the "versions" are
   **sequence numbers, not Lance versions** (`:44-45`). Every result line using
   the word "version" must carry that qualifier.
9. **No HLC / multi-writer claim** (§1.3).
10. **No parallelism / scale claim** — that is D-BLW-4, whose axis is rows inside
    one owner (§12.3a′).
11. **No fusion claim outside the IN/IN band** (§3.2), and none at all before D3b
    (§12.4).

---

# 7. Honest blockers

**B1 — §12.3's D-BLW-3 as literally written cannot be built.** Its declared input
(four-stance pairwise agreement) is dead: 3 of 4 stances UNREACHABLE as MEASURED
(§12.3a″), the instrument retired (§12.3c) and its rebuild uncompiled, and all of
it runs off-substrate (§12.7 defect 2). **The honest deliverable is smaller than
the plan row describes.** Re-scope to the kanban-plan D3 wording (two projections
of one cohort) with the projections defined over the tenant. Recorded as a
reduction, with reasons — not substituted silently.

**B2 — the D-BLW-1 harness as shipped produces a series over which fusion CANNOT
move.** All verses are seated before V1 (`:615` before `:758`) and the row body is
horizon-independent (`:688` fixed probe over never-rewritten content planes). So
`Δ ≡ 0` **by construction** and the falsifier would be vacuous. **P1 (incremental
seating) and P2 (a horizon-relative criterion) are prerequisites, not
enhancements** (§1.4). *This is the single most important item for the
orchestrator to carry.*

**B3 — the corpus floor and the tenant capacity have exactly 2× headroom.**
`N_CAP = 2048` (type-level const), `DEFAULT_VERSES = 2000`, floor `k ≥ 1000`. If a
degeneracy guard fires and the reflex is "use more verses", that requires growing
`N_CAP` — and the identity planes cost 6,144 B/row (`blw_tenant.rs:62-64`), so
`MailboxSoA<4096>` is ~24 MiB of planes. **Decide before the run**; deciding after
a κ exists is fitting. (`ISS-MAILBOXSOA-ROW-COST-VS-512B-CANON` remains open and
is not resolved here.)

**B4 — the trajectory is short.** 8 sealed cycles = two DAG loops. No trend claim
is licensed; the movement test is a two-point contrast and the intermediate κs
are reported, never fitted. §12.3's word "trajectory" is doing more work than 8
points can support and the write-up must say so.

**B5 — `jc` must be re-added as a dev-dependency.** `jc = { path = "../jc" }`
under `[dev-dependencies]` of `lance-graph-planner`. It is workspace-**excluded**
(root `Cargo.toml` `exclude`, `crates/jc`), which is fine for a path dep. The
planner's own Cargo.toml records the constraints at `:67-77`: **dev-only, never a
production dep, do not modify `jc`, do not invert the edge** — a measure cannot
be its own oracle. Carry all four verbatim.

**B6 — this harness would be the FIRST implementor of `DeinterlaceRow` and the
FIRST caller of `deinterlace` anywhere.** Verified by grep over `crates/`:
the only hits outside `temporal.rs` itself are a doc-comment mention at
`lance-graph/examples/reasoning_loop.rs:51`. That is a genuine contribution and
also a risk — nothing else exercises this surface in anger, so any surprise is
D-BLW-3's to absorb. Gates G1/G3/G6/G7 exist because of it.

**B7 — UNVERIFIED (marked, not asserted):**
- Whether `set_populated` growing mid-run is exercised by any existing test. Its
  doc sanctions caller-managed growth (`mailbox_soa.rs:487-497`); **no test was
  located**. UNVERIFIED.
- Whether `MailboxSoaView::n_rows` picks up a mid-run `set_populated` with no
  caching. The doc says `n_rows` is bound to `populated` since W1c; **the impl was
  not read**. UNVERIFIED.
- Whether `recover_and_apply`'s watermark handling behaves across **8** cycles;
  `blw_tenant.rs` exercises **3**. UNVERIFIED.
- Wall-clock of an 8-cycle × 2000-row run. The rank criterion adds one sort per
  (horizon × projection), `n ≤ 2000` — expected trivial beside the existing bloom
  sweep, but **NOT measured**. (§12.7 records that the *previous* arm blew a
  10-minute budget for an unrelated reason — `stance::stream`'s per-lift
  O(arena) `staunen` scan — which this design does not use at all.)

**B8 — the scope reduction I recommend explicitly.** Do **not** attempt the
four-stance version, the cross-language version, or any §12.6 anchor
reproduction inside D-BLW-3. The honest deliverable is:

> **the first working `DeinterlaceRow` implementor and `deinterlace` caller; two
> rank-based, horizon-relative binary projections over one real sealed series;
> a pre-registered band and movement threshold reused (not re-invented) from
> §12.3a/§12.3b; and an inert control projection that proves the plumbing
> contributes no movement.**

That is smaller than §12.3's row describes, and it is the part that can actually
be falsified.

---

## Seam I stopped at

No code. No compile. The design assumes P1/P2 land as harness changes on a new
example (`examples/blw_fusion.rs` or equivalent) that **consumes**
`blw_tenant.rs`'s shape rather than editing it — but that placement decision is
the orchestrator's, and I did not make it. `temporal.rs`, `crates/jc`, and
`persist_sink.rs` are untouched by this design by construction (§12.5).
