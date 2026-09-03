# rubicon-loco-rung-cognitive-fabric-v1 — one clockwork fabric, or a measured NO

> **Status: PROPOSED — PLAN/BOARD ONLY. SOURCE-FIRST.** No production
> implementation in this PR. Every §A–§J claim below carries `file:line`
> evidence gathered at HEAD `de1d0c2f`; nothing is inferred from a filename.
>
> **⊘ CORRECTED (2026-08-29):** this line originally read *"nothing is …
> quoted from a prior plan"*, and that phrasing caused a real defect — §F was
> written WITHOUT reading `alpha-channel-rung-overlay-v1.md` (76 KB) and
> invented a storage model for something already designed in detail.
> Source-first means **verify** a prior plan against source, never **skip** it:
> a prior plan is mandatory reading and its claims are then re-checked, exactly
> as `CLAUDE.md` § "Consult before you guess" requires. Mandatory reads for
> this plan: `alpha-channel-rung-overlay-v1.md`,
> `hhtl-thinking-tables-le-contract-v1.md` §2.3/§3,
> `unified-soa-rubikon-integration-v1.md`, and the `D-ACR-*` rows of
> `STATUS_BOARD.md`.
> **Operator directive (2026-08-29):** *"The next step is not another bespoke
> cognitive controller… test whether the existing lance-graph / OGAR substrate
> can become one clockwork cognitive fabric."*
> **Arc:** #1074 (plan) + #1075 (triptych) + #1076 (arc) — all MERGED and now
> **historical substrate, not open threads.**

## §0 The thesis

Rungs 1..10 are **horizons/views**, not ownership locations for thinking
styles. `ogar-loco` is the ABI-shaped call membrane. **Frozen** = a hardened
callable atom, never "special Rust outside the substrate". V4/R2IL is the
compositional program representation. **Alpha is the rung-level STORAGE** —
separate thinking tables at the graph's own addresses, one row per rung,
sparsely occupied, two owners (ontology / session) — not a delta inside a
node row (§F). Kanban/Rubicon owns lifecycle. Revision is explicit per
cycle and its receipt seeds the next. `temporal.rs` makes it replayable.

**Success criterion:** *change, version, test, replay and roll back a reasoning
policy without recompiling bespoke `lance-graph-planner` control flow* — unless
the change adds a genuinely new atomic capability or constitutional rule.

---

## §A Current-state ownership map — VERIFIED

| # | concern | owner (file:line) | status |
|---|---|---|---|
| 3 | `ExecTarget` | `contract::kanban` | **see §A.1 — naming debt** |
| 4 | recipe kernels | `contract/src/recipe_kernels.rs` | live; the interpreted layer |
| 5 | `KanbanMove` | `contract/src/kanban.rs:230` | live, typed |
| 6 | owner adapter / `BatchWriter` | `planner/src/owner_adapter.rs` | live, write-on-behalf |
| 9 | Revision route | `contract::kanban::advance_on_revision` | **NEW on main (#1075)** |
| 10 | temporal/versioning | `planner/src/temporal.rs` | live |
| 11 | kanban visibility | `lance-graph-supervisor/src/kanban_actor.rs` | exists — §G |

### §A.1 `ExecTarget::Elixir` names NO Elixir — FINDING, not conjecture

- **Zero `.ex`/`.exs` files** in the workspace. `crates/elixir-template` is Rust.
- `planner/src/strategy/style_strategy.rs:499` states it directly: *"actually
  ran = the interpreted `recipe_kernels` layer = `ExecTarget::Elixir`"*.

**Verdict: naming/architecture debt, not an execution backend to preserve.**
A semantically honest successor names implementation identity
(`Native` / `Frozen` / `Jit` / `R2il`). **Do NOT rename until a measured BUY** —
the variant is load-bearing in `lance-graph-ogar/src/actions.rs:94,103` and the
`owner_adapter`/`persist_sink` paired-move tests.

---

## §B Rung-4 fossil census — the fossil is THIN, and the thesis is already true

`rung.?4|RungLevel::*4|rung_4` sweep over `crates/`: **5 hits** — ⊘ *census
corrected in review (codex P2, re-verified at source): the regex was NOT
exhaustive.* Two live sites use semantic/spaced forms it missed:
`contract/src/proprioception.rs` `ANCHOR_REGISTRY` carries `rung: 4` (a data
entry), and `planner/src/elevation/mod.rs:119-124` matches
`RungLevel::Abstract` (mapping ALL ten rungs to elevation tiers, rung 4 not
privileged). Re-verdict on the corrected evidence: **the fossil conclusion
STANDS** — neither added site is a rung-4 privilege gate — but D-RLR-2's
census MUST include semantic (`RungLevel::Abstract`) and spaced (`rung: 4`)
forms, and must not cite the original five-hit count.

| hit | class |
|---|---|
| `planner/src/temporal.rs:637` | **TEST of a real mechanism** (see below) |
| `planner/src/thinking/sigma_chain.rs:8` | **DOC-ONLY** (Φ = Belief at Rung 4) |
| `planner/examples/probe_style_microcode_frontier.rs:19` | **DOC-ONLY** |
| `contract/src/dispatch_mode.rs:5` | **DOC-ONLY** |
| `cognitive/src/spo/cognitive_codebook.rs:834` | **DOC-ONLY** (a codebook label) |

**No `PHYSICAL REQUIREMENT` hit exists.** Nothing in layout, ABI or dispatch
restricts cognition to rung 4.

**And the one real rung-dependent mechanism already IS the operator's model.**
`planner/src/temporal.rs:87-97`:

```rust
/// Low rungs reason strictly in the present; mid rungs admit hindsight;
/// top rungs may spoiler-read.
pub fn for_rung(rung: u8) -> Self {
    match rung { 0..=4 => Strict, 5..=8 => Aware, _ => Retro }
}
```

…paired with `admits(status)` gating which `TemporalStatus` rows a reader may
dispatch on. **Rung already differs by TEMPORAL HORIZON, not by the right to
think** — and the policy is marked *"(tunable)"*.

**Verdict: the generalization is far cheaper than assumed.** What is missing is
not permission but *demonstration* — no non-rung-4 horizon has been shown
end-to-end. **STOP-gate `F-RLR-1` (§J) is the falsifier.**

---

## §C Atom census — the loco carrier ALREADY EXISTS and is already used

`lance-graph-ogar/src/recipe_vocab.rs:78` imports `ogar_loco::{FnIndex,
DOMAIN_FLOOR}` and provides:

- `op_of(recipe_id: u8) -> Option<FnIndex>` (`:108`)
- `recipe_of(f: FnIndex) -> Option<u8>` (`:117`)
- **`ladder_program() -> Vec<FnIndex>`** (`:126`) — a program already IS a
  byte-addressed atom sequence
- `domain_stack_arity(&self, f: FnIndex) -> Option<u8>` (`:142`)

**§D verdict (loco capability): YES — `classid/vocabulary + u8 FnIndex` already
represents Frozen atoms AND composed programs. A new carrier is NOT justified,
and proposing one is a STOP.**

> **⊘ TENSION RESOLVED 2026-09-03 — no conflict, and a shipped symbol proves
> it.** #1152's §F.6 flagged a possible contradiction between this verdict
> (`classid/vocabulary + u8 FnIndex` already carries Frozen atoms and composed
> programs) and `r2il-machine-semantic-contract-v1` §4 R4 (*"the macro
> vocabulary is a PALETTE, not a `FnIndex` per macro"*). I declined to
> adjudicate it from a partial read. The owner session has, and the answer is
> that R4's target is **one FnIndex per macro** — which would burn the slot
> space — never FnIndex as the addressing mechanism.
>
> `ogar_loco::TERNLOG` settles it concretely: **one `FnIndex`, with the call's
> value byte carrying the 8-bit truth table, so a single address covers all 256
> combinators** (`ogar-loco/src/lib.rs:607`, `vocabulary.rs:92-98`). That is
> palette-as-vocabulary and FnIndex-as-address in one shipped symbol. This
> verdict stands unchanged.


| candidate | status |
|---|---|
| recipe kernels | **EXISTS + CALLABLE** via `FnIndex` |
| Shannon entropy | **EXISTS BUT NOT LOCO-ADDRESSABLE** — ≥6 uncoordinated `entropy()` surfaces |
| EWA covariance | **EXISTS** (`jc::ewa_sandwich`), not loco-addressable |
| masks (∩ ∖ ⊆) | **EXISTS + CALLABLE** (`contract::revision::EvidenceMask`, #1075) |
| fusion / revision | **EXISTS + CALLABLE** (#1075) |
| topology / ReasoningBand | **EXISTS + CALLABLE** (`causal-edge::layout`) |
| counterfactual / intervention | **EXISTS** (`contract::counterfactual`; R2IL V4 probe rows) |
| perspective residual / parallax | **MISSING** (`parallax` = 0 hits) |
| FieldModulation / styles | **EXISTS + CALLABLE** (7 knobs) |
| rung/horizon read | **EXISTS + CALLABLE** (`EpistemicMode::{for_rung, admits}`) |

**No Shannon exemption.** A deterministic read over a hypothesis distribution
is eligible to become a Frozen atom like any other hardened primitive. Native
kernels stay native; only the *address* is added. Do not rewrite good SIMD in
R2IL to claim purity.

---

## §E R2IL convergence — the seam is NARROWER than expected

R2IL is **not** only accumulating in lance-graph-java. Six probes exist here:
`probe_r2il_{slag_boundary, optimization_transfer, frontier_phase2,
real_episodes, bpe_recombination_falsifiers}` + `probe_style_microcode_frontier`
(all `planner/examples/`).

**Verdict:** the smallest membrane is a **shared operator/program semantic
contract**, with per-host adapters — never a lance-graph-specific thinking DSL,
and never an R2IL→pseudo-IR transcode unless the operator contract demands it.
**Measure readiness before designing the membrane** (`D-RLR-4`).

> **⊘ RESOLVED 2026-09-03 by the plan's owner session (cross-session).** The
> storage half of this section — *how a session stores R2IL* — is answered by
> `.claude/plans/r2il-machine-semantic-contract-v1.md` §4, now tracked as
> `D-R2IL-1` (lance-graph #1155). This §E verdict was an independent, thinner
> restatement of it, arrived at because that plan carried **no D-ids at all**,
> so `STATUS_BOARD` had nothing to hold and it was invisible to a
> mandatory-reads pass. It defers to §4; it does not compete with it.
>
> **`D-RLR-4` is re-scoped, NOT retired** — at the owner session's explicit
> request. What survives is the half §4 never touches: **what does the
> lance-graph-java membrane consume**, now that the `0x87..0x8B` loco band is
> retracted and `ogar_loco::TERNLOG` = `FnIndex(0x86)` is minted-but-unconsumed
> (`ogar-loco/src/lib.rs:607`; the reservation at `:596`). That question went
> live with lgj #70, which blocks the `BELNAP_JOIN` mint.


---

## §F Alpha channel = the rung-level STORAGE. It is not written today.

> **⊘ CORRECTED (operator, 2026-08-29).** A first draft of this section read
> Alpha as an abstract non-destructive overlay and asked a **stride-budget**
> question ("ten thin deltas in the 480 B value slab — is there a 10× delta
> reserve?"). That is wrong in KIND, not merely in wording, and it was written
> without reading `.claude/plans/alpha-channel-rung-overlay-v1.md` (76 KB,
> 2026-08-21) — the exact rediscovery tax `CLAUDE.md` § "Consult before you
> guess" forbids. **The alpha channel IS storage: the rung levels live across
> separate thinking TABLES at the same addresses, not as deltas inside one
> row.** There is no stride budget question. The wrong framing is recorded
> rather than deleted, per the append-only convention.

### §F.1 What the design actually is (all quotes verified this session)

The operator's own frame, `alpha-channel-rung-overlay-v1.md` §0, items 3 + 4:

> *"Gedanken 2. Ordnung als thinking about thinking als graph overlay mit der
> gleichen Adresse wie der Graph aber **separate thinking tables**."*
> *"**Rung levels 2–10 als Alpha layer projizieren**."*

Three properties fix the shape, and each rules out the delta reading:

| property | source | consequence |
|---|---|---|
| **same address, different table** | plan §3 | a rung-*n* thought at node `g` is the **same `NodeGuid`**, a **different `(classid, rail)` thinking-table row** resolved by the ClassView |
| **sparse occupancy** | plan §3, verbatim: *"'1:1' names the addressing, never the occupancy"* | only visited addresses materialise — ten rungs do not cost ten copies of anything |
| **two owners, two tables** | plan §2 (⊘-corrected by the operator same day) | ontology thinking table (ontology mailbox, durable) **vs** session overlay (session mailbox, ephemeral) — the contamination boundary is *structural in the table split* |

`contract::attention_facet` states the same rejection independently, and it is
the closest thing to a ruling on the delta framing:

> The tempting alternative was a **matrix**: six *vertical* rows of the same
> 12-byte atom, one per rung/layer/timestep. It is rejected … the vertical
> dimension is **already addressed** — it is the `facet_classid` selecting
> which thinking-table row a focus belongs to. **One atom, one
> `(classid, rail)`; the vertical stack is a set of atoms, not a field inside
> one.**

So: **ten rungs = ten rows, sparsely occupied, at one address.** Not ten
residuals sharing a 480-byte slab. The `10× delta reserve` question is
withdrawn — it measured a budget that nothing spends.

### §F.2 How rung levels are written TODAY: they are not

Measured this session, whole workspace. Every row verified by reading the file.

| surface | what it holds | durable? |
|---|---|---|
| `RungElevator { base, level, block_streak, flow_streak }` — `cognitive_shader.rs:272`, behind `RwLock` in `cognitive-shader-driver/src/driver.rs:132,907` | ONE current level | **no** — process memory |
| `RungLevel` as `u8` on the wire — `grpc.rs:411`, `wire.rs:921` (`RungLevel::from_u8`) | ONE level | **no** — transport |
| `holograph::storage_transport::StorageFlags.rung: u8` (`:65-66`, *"Abstraction rung (0-255)"*); `holograph::width_32k::schema` packs it at word 8 bits 24–31 | ONE byte, ONE current rung, in a 32-byte node header | **yes, but** — a header byte, not a Lance table, and not ten channels |
| every live planner construction — `orchestration_impl.rs:151`, `pipeline.rs:593`, `api.rs:180`, `codec_bridge.rs:109`, `cypher_bridge.rs:130` | hardcoded `RungLevel::Surface` | — |
| `lance-graph-planner/src/persist_sink.rs` | **excludes rung, five times**, verbatim: *"Scope boundary — storage only, no semantic / rung types minted here"*; *"It does NOT mint (or carry) … rung …"*; *"Storage-only: it carries NO rung / projection / branch / semantic tags"* | — |
| `soa_envelope.rs` | zero `rung` hits | — |
| any arrow schema | no `Field::new("rung", …)` anywhere in the workspace | — |

**The atoms shipped; the write path did not.** `D-ACR-1` (`RowFocusMask`,
`contract::attention_facet`), `D-ACR-7` (`contract::band_reading`) and
`D-ACR-8` (`contract::rubicon_witness`) are all **Shipped** on
`STATUS_BOARD.md` — and their only consumers workspace-wide are **four probe
examples** in `lance-graph-planner/examples/`. Nothing persists them.

Two blockers, both already named on the board, and neither is a stride budget:

1. **`D-ACR-2` — the Rung-ladder rail is UNMINTED.** `HTT §2.3` row
   *Rung ladder* reads `*(unassigned)* · see §3 · **unminted, undesigned**`,
   and the board row says *"Queued — gates on operator mint decision
   (HTT §8 Q3)"*. Until a rail is minted there is no `(classid, rail)` for a
   rung row to BE.
2. **`D-ACR-3` — there is no ontology write path to guard.** Board, verbatim:
   *"`SoaEnvelope` has ONE production implementor (`NodeRowPacket`) … and
   `mailbox_owner()` has **zero callers outside its own module**. There is no
   ontology-owned write to trace TO and no session-tagged read to trace FROM."*

### §F.3 What `D-RLR-5` must therefore measure (re-scoped)

The trace is unchanged in spirit and wrong in target. Re-scoped:

- **NOT** "does the 480 B value slab have room for ten rung deltas" — withdrawn,
  measures a budget nothing spends.
- **IS**: (a) name the `(classid, rail)` pair each of rungs 1–10 would occupy,
  and (b) show ONE end-to-end write→read of a single rung row through a real
  owner. One rung proven end-to-end beats ten designed.

`D-RLR-5`'s default verdict stays **HELD**, and the reason is now specific:
*blocked behind `D-ACR-2`'s mint and `D-ACR-3`'s missing write path*, not
"representation existing is not integration".

### §F.4 ⊘ HARD FENCE — `NodeGuid` is NOT an immutable pointer across rungs

*(Unchanged — verified at `canonical_node.rs:349-367`. This fence gets
SHARPER under §F.1, not weaker: if ten rungs share one address by design, the
stability of that address is the whole load-bearing assumption.)*

**Do not key alpha (or anything cross-rung) on an independently RE-MINTED
`NodeGuid`.** ⊘ *Narrowed in review (codex P2): the first wording was a
blanket fence that would have rejected the same-address overlay model itself
(`alpha-channel-rung-overlay-v1.md` requires the overlay to use the same
`NodeGuid`). The safe form, per source: `NodeGuid` derives `Eq`/`Hash` over
its raw `[u8; 16]` and tail variants only REINTERPRET those bytes — so a key
COPIED VERBATIM from the base row stays valid across a registry flip. What is
unsafe is minting a semantically-equal address independently under a
different `tail_variant` and expecting equality. Copy, never re-mint.*

- **The tail reading is registry-resolved, not intrinsic.** Mints go through
  `mint_for(classid_read_mode(c).tail_variant, …)` — *"NEVER by hardcoding
  `new` vs `new_v2`"*. The same 16 bytes read differently per classid.
- **The variant is DESIGNED to flip.** *"Migrating a class's identity to V3 is
  then a one-line flip of its `tail_variant` in the registry, **with zero
  consumer rewrite**."* An existing guid's tail interpretation can change
  underneath a consumer, by design.
- **It is feature-gated.** With `guid-v2-tail` off, `classid_read_mode` returns
  `V1` for every classid — the same source, built differently, reads tails
  differently.
- **The zero-fallback ladder means it is not uniformly a full address:**
  `classid == 0` / `family == 0` are *not consulted*, so in the bootstrap case
  `identity` alone discriminates.

`NodeGuid` is a **content-blind carrier whose reading is late-bound**, not a
pointer with one meaning everywhere. An alpha overlay that assumes otherwise
breaks silently on a registry flip — the worst failure shape, because nothing
errors.

**`F-RLR-9` (STOP):** any alpha/cross-rung design that treats a `NodeGuid` as
an immutable pointer, or that compares guids minted under different
`tail_variant` registrations as though they addressed the same thing.
Whatever identity the overlay keys on must be stated and shown stable under
(a) a `tail_variant` flip and (b) the feature being off.

**`F-RLR-10` (STOP, new):** any statement about how rung levels are stored
that was not read out of `alpha-channel-rung-overlay-v1.md` + `HTT §2.3/§3` +
the `D-ACR-*` board rows. This section's own first draft is the instance: it
invented a delta-budget model for a design that had already been written down
in 76 KB, and would have sent `D-RLR-5` to measure the wrong quantity.

### §F.5 ⊘ CORRECTED (2026-08-29, same day) — the write path is not hypothetical, it already shipped, in a sibling repo

§F.3 said `D-RLR-5` was HELD behind `D-ACR-2` (rail unminted) and `D-ACR-3`
(no ontology-owned write path exists). **The second half of that was checked
in `lance-graph` only** — and it does hold, narrowly, for the reason given
below. But stated as "the write path is unbuilt," it was answered wrong to a
direct question (*"how do you currently write 10 rung levels?"*) without
checking a sibling repo already cloned on local disk. Full verified account:
`.claude/board/EPIPHANIES.md` `E-A-RUNG-WRITE-PATH-ALREADY-SHIPPED-IN-A-SIBLING-REPO-1`.

**What is real, read from merged source, not from a PR description:**
`MedCare-rs`'s `medcare-nodesoa::alpha` (PR #565 merged 2026-08-22, extended
by PR #590 merged 2026-08-26) depends **directly** on
`lance_graph_contract::canonical_node::{NodeGuid, NodeRow}` and defines

```rust
pub struct AlphaStamp { cycle: u32, seq: u32, rung: u8, visits: u16 }
```

— a `rung: u8` at byte offset 8 of a 16-byte value slot, inside the SAME
canonical 512-byte `NodeRow`, through the SAME `FixedSizeBinary(512)` Arrow
column, with an optional `lance` feature that persists it to a real on-disk
Lance dataset (PR #561: *"8 arrow-only Tests plus der On-Disk-Beweis gegen
einen echten Lance-Datensatz"*). Ephemeral by operator ruling — *"ephemer
daneben, verwerfbar"* — no bake-table row, discardable whole, exactly the
`E-EPHEMERAL-DISCARDABLE` shape this plan's §F already argued for on other
grounds.

**Corrected reading of `D-ACR-3`.** `lance-graph` itself has no ontology-owned
write path because `lance-graph` itself owns no live `NodeRow` spine to
attach an overlay to — it is a contract-only crate. `MedCare-rs` has a live
baked spine (the OBO/ontology bake), so it built and Lance-proved the pattern
immediately, on the identical contract types. **The write path is proven,
once, adjacent to real data — not unbuilt in the architecture.** Whatever
`D-ACR-2`/mint work happens in `lance-graph` should read this implementation
first, not design a second one from a blank page.

**Explicit non-claim, so this does not become the next overcorrection.**
`AlphaStamp.rung` is a plain `u8` with domain-local meaning ("which rung of
*attention*"); it does **not** import `RungLevel` (0-9,
Surface..Transcendent) from `contract::cognitive_shader`. Same name, same
operator-brainstorm week, **not proven to be the same vocabulary.** `D-RLR-5`
(re-scoped) should include: is `AlphaStamp.rung` the same ladder `RungLevel`
names, a coarser projection of it, or an unrelated attention-depth scale?

**`F-RLR-11` (STOP, new):** any claim that a mechanism is "unbuilt" or "no
write path exists" that was reached by searching only the repo the session
happens to be in. Before asserting an absence, check every repo in the
session's own scope that plausibly contains the thing — a sibling repo
already cloned to local disk, with its `CLAUDE.md` already loaded into this
session's context, is not an exotic place to have to look.


### §F.6 ⊘ CORRECTED (2026-08-31) — the substrate landed, and "ten rows" was still the wrong noun

The mechanism this section has now been wrong about twice **exists in this
repo as of #1112** (merged 2026-08-31). It is no longer a design to infer from
plan prose; it is source, and the source names itself precisely. Every claim
below is `file:line` against the tree at `cc0046f8`.

**What landed.** `alpha`, `alpha_tunnel` and `rung_schedule` migrated into
`lance-graph-contract`; `rung_horizon` is new in `lance-graph-planner`.

| module | lines | what it is |
|---|---:|---|
| `contract/src/alpha.rs` | 938 | the overlay algebra — `AlphaStamp{cycle,seq,rung,visits}` (`:109`), `AlphaAllocation` (`:339`), `AlphaOverlay` (`:443`), `AlphaMask` (`:224`) |
| `contract/src/alpha_tunnel.rs` | 402 | `AlphaTunnel` (`:73`) — the split tunnel |
| `contract/src/rung_schedule.rs` | 372 | dependency-wave scheduler; `LEVELS = 10` (`:59`) |
| `planner/src/rung_horizon.rs` | 213 | per-rung readers + `claim_admitted` (`:59`) |

#### The correction: ten LANES over ONE reservation, not ten rows

§F.1 closes with *"ten rungs = ten rows, sparsely occupied, at one address."*
**The noun is wrong.** `AlphaTunnel` holds `lanes: Vec<AlphaOverlay<'a>>`
(`alpha_tunnel.rs:73-76`), and its constructor maps `(0..LEVELS)` over
**one** borrowed allocation:

```rust
// alpha_tunnel.rs:82-89
pub fn over(alloc: &'a AlphaAllocation<'a>, cycle: u32) -> Self {
    Self {
        lanes: (0..LEVELS)
            .map(|_| AlphaOverlay::over_shared(alloc, cycle))
            .collect(),
        cycle,
    }
}
```

`LEVELS = 10` (`rung_schedule.rs:59`). One `AlphaAllocation`, ten borrows.
The module states the per-lane cost as one empty `Vec` (`:25-27`), and states
the prohibition directly: ten lanes must **not** mean ten address sets,
because reserving is defined as costing zero rows, and ten copies of the
address set would make "reserve" cost ten times nothing (`:22-24`; the same
argument from the allocation's side at `alpha.rs:454-456`).

So all three readings this plan has carried are now settled against source:

| reading | verdict |
|---|---|
| ten deltas sharing a 480 B value slab (§F original) | wrong — withdrawn in §F.1 |
| ten rows / ten tables (§F.1) | **wrong — the noun is `lane`, and the reservation is ONE** |
| ten lanes over one `AlphaAllocation` | correct (`alpha_tunnel.rs:73-89`) |

#### "Split tunnel" names the read/write path split — not a budget, not a table count

Neither prior reading had this at all. The module's own heading (`:12-18`)
states that reading and writing take different paths: reads go to the baked
spine, shared by all ten lanes without a lock because `&[NodeRow]` may be
shared; writes go to the overlay at the same addresses, and `alpha` makes that
direction a compile-time property rather than a runtime check. **That
asymmetry is what the words "split tunnel" denote.**

#### §F.2 needs refinement, NOT reversal — the persistence half did not migrate

§F.2's heading (*"How rung levels are written TODAY: they are not"*) reads as
superseded, and is not. `alpha.rs:1-5` records that the Arrow/Lance storage
glue (`to_batch`, `key_bytes_at`, the `lance` feature module) **deliberately
stayed with the storage crate**; what migrated is the pure overlay algebra
over contract types. So:

> `lance-graph` can now **express** a rung stamp. It still does not
> **persist** one. The Arrow/Lance write path remains in the consumer.

§F.2's table is therefore still accurate as a statement about *persistence in
this repo*, and its heading should be read as scoped to that.

#### `D-ACR-3`'s blocker SURVIVES #1112 — checked, not assumed

The tempting inference is that a landed write path unblocks `D-ACR-3`. It does
not. `mailbox_owner()` still has **zero callers outside its own module**; the
only occurrence anywhere else in `crates/` is a doc-comment mention at
`alpha_tunnel.rs:33`. The tunnel enforces one-writer **structurally** — each
lane owns its own `&mut`, so parallelism needs no lock (`:29-38`) — rather
than through the mailbox-ownership machinery `D-ACR-3` exists to test. The
`D-RLR-5` board row's HELD reason stands unchanged.

#### The temporal-isolation mechanism, for the record

`rung_horizon::claim_admitted` (`:59`) classifies **before** claiming, and the
ordering is load-bearing: `classify` → `reader.mode.admits(status)` → only
then `lane.claim` (`:66-73`). A refused row therefore leaves no trace in the
lane — *not even a `visits` bump* (`:56-58`), pinned by an assertion that the
refused row reads back as `None` (`:168-173`). Note this is the same `visits`
counter §F.5 identified in the consumer-side original; it now carries a
second duty.

**`F-RLR-12` (STOP, new):** correcting a claim about a mechanism by
substituting a different English noun for it, when the mechanism exists in
source and names its own type. This section replaced "deltas" with "rows" and
was still wrong, because neither word was read off `AlphaTunnel`. A correction
must cite the type's definition, not re-describe its shape.


## §G Kanban / Rubicon verdict

- **Internal string paths: NONE.** No `from_str`, no `as_str`, no column-name
  literals in `contract/src/kanban.rs` or `planner/src/`. **Kanban is already
  typed/binary internally — no cutover candidate exists.** Strings live only at
  membranes.
- **Revision is ALLOWED but not UNAVOIDABLE.** #1075 added
  `advance_on_revision`, and `Evaluation`'s successors are `[Commit, Plan,
  Prune]` — `advance()` still reaches `Commit` directly. Lifecycle topology
  does **not** force Revision on a completed cycle. `D-RLR-6` asks for the
  smallest invariant that would, **without inventing a second Rubicon.**
- `kanban_actor.rs` exists (`lance-graph-supervisor`). **Hard fence retained:
  visibility must not gain mutability**; all progression stays on the #879
  sealed-cycle path.

---

## §H Symbiont quarantine — reachability PROVEN, action recommended

`Cargo.toml:86` places `"crates/symbiont"` in **`exclude`** (not `members`), and
**no crate path-deps it** — every other hit is a comment citing it as a
sibling/precedent (`cognitive-stack`, `lance-graph-ogar` manifests).

**Verdict — ⊘ narrowed in review (codex P2): workspace exclusion + zero
reverse path-deps proves only "unreachable from ROOT-WORKSPACE builds", not
unreachable: `crates/symbiont/Dockerfile` builds a runnable image with
`symbiont` as entrypoint (own manifest + container path). The quarantine
recommendation therefore rests on the OPERATOR ruling, which is stronger than
the reachability inference anyway: symbiont is ⊘ DEPRECATED 2026-08-18,
operator no-go — "dormant excluded crate, never live surface" (CLAUDE.md).** Recommend
the smallest quarantine that preserves archaeology while stopping a future
session reading it as current — a `DEPRECATED-ARCHITECTURE` header + a board
pointer. **`remove Symbiont` ≠ `remove SurrealDB`**; storage/query use is a
separate question this plan does not touch.

---

## §I The ONE smallest first Wave — `D-RLR-1`

**Prove the central thesis end-to-end, once, at a NON-rung-4 horizon:**

**Rung-number mapping (pinned before W1, per review — Major):** every `r` in
this plan is the **`RungLevel` discriminant, 0..=9** (`cognitive_shader.rs:157`;
`from_u8` saturating is THE one u8→rung mapping). Prose that counts "rungs
1..10" is the 1-indexed human ordinal of the SAME ten levels (ordinal n =
discriminant n−1); the wire value is the discriminant. `for_rung(r)` takes the
discriminant: 0..=4 → Strict, 5..=8 → Aware, 9 → Retro. "Rung 4" throughout =
discriminant 4 = `RungLevel::Abstract`. No `(classid, rail)` row may be minted
against an ordinal.

```text
EpistemicMode::for_rung(r ≥ 5)        a horizon that is NOT rung 4
   → invoke ONE existing Frozen atom via ogar_loco::FnIndex
     (recipe_vocab::op_of — already callable, no new atom)
   → read/write ONE alpha surface
   → emit a typed receipt
   → participate in Evaluation → advance_on_revision
   → stay on the canonical #879 sealed-cycle path
   → temporal.rs replay proves what happened
```

Chosen from source, not invention: the atom is `recipe_vocab::op_of`
(**EXISTS + CALLABLE**), the horizon read is `EpistemicMode::for_rung`
(**EXISTS**), the revision route is `advance_on_revision` (**NEW on main**).
**Nothing new is built in W1 except the wiring that proves they compose.**

---

## §J Falsifiers / STOP gates — every BUY has a NO-BUY

| id | gate | NO-BUY when |
|---|---|---|
| `F-RLR-1` | a non-rung-4 horizon completes the §I chain | a **physical/layout/ABI** reason forces cognition to rung 4 — then STOP the generalization and name it precisely |
| `F-RLR-2` | loco addresses a Frozen atom with no new carrier | a new carrier is proposed before `ogar_loco` is proven insufficient — **automatic STOP** |
| `F-RLR-3` | ONE rung row written and read back through a real owner | no `(classid, rail)` is minted for it (`D-ACR-2`) or no owner write path exists (`D-ACR-3`) ⇒ **HELD** — and HELD for that named reason, never for "representation exists" (§F.2) |
| `F-RLR-4` | no source text on the cognition hot path | Elixir/Blockly/Scratch/AST parsed during production cognition, **or** changing the human projection changes execution semantics |
| `F-RLR-5` | Revision unavoidable per completed cycle | it requires a second Rubicon or moves epistemic judgement into MUL |
| `F-RLR-6` | R2IL reuses lance-graph-java semantics | a lance-graph-specific thinking DSL appears |
| `F-RLR-7` | kanban stays typed internally | any `typed → stringify → internal transport → parse → typed` path |
| `F-RLR-8` | band promotion stays typed | a band transition reduces to a threshold on one untyped scalar |
| `F-RLR-9` | alpha keys on a stated, flip-stable identity | a `NodeGuid` is treated as an immutable cross-rung pointer, or guids under different `tail_variant` registrations are compared as one address (§F.4) |
| `F-RLR-10` | every rung-storage claim is read out of `alpha-channel-rung-overlay-v1.md` + HTT §2.3/§3 + the `D-ACR-*` board rows | a storage model for rungs is asserted from the session's own reasoning — §F's own first draft is the instance (§F.1 ⊘) |
| `F-RLR-11` | every "unbuilt" / "no writer" / absence claim checks the SIBLING repos before landing | an absence claim rests on a single-repo (or grep-only) census — the E-A-RUNG-WRITE-PATH sibling-repo correction is the founding instance |

**Constitutional, carried from the merged arc:** ambiguity/entropy/parallax
**never** terminate cognition — only the Rubicon boundary owns stop/commit/veto.
Shannon reduction ≠ evidential increase. EWA covariance ≠ empirical authority.
Counterfactual necessity ≠ observation.

---

## §K What this plan deliberately does NOT do

No second thinking DSL. No scheduler (W8 tests sufficiency only; a future
scheduler may prefetch/queue/wake but **never** decide truth, causality,
independence, band promotion or revision acceptance — **prefetch ≠ belief**).
No new trust scalar. No rename of `ExecTarget::Elixir` before a measured BUY.
No `ClassView`/`VocabularyRegistry` collapse — they share classid discipline
while answering different questions (*what fields does this object expose?* vs
*what do these program bytes mean?*).
