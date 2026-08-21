# D-ACR-0 — `attention_mask` audit against piece E

> **Deliverable:** `alpha-channel-rung-overlay-v1.md` §4, `D-ACR-0`.
> **Scope:** report only, no code — as the deliverable specifies.
> **Falsifier:** *"the audit names a caller, or records EXISTS-UNCALLED."*
> **Verdict: EXISTS-UNCALLED — and the shipped type is a DIFFERENT MECHANISM
> wearing the name.** Both halves matter; the second is the load-bearing one.
>
> Every claim below is a read of the file named, this session. Every "absent"
> is a grep that returned nothing, with the grep stated.

## 1. The question D-ACR-0 asks

Piece E of §1 graded `cognitive-shader-driver/src/attention_mask.rs` +
`attention_mask_actor.rs` as *"shipped; **unaudited for this use**"* — the use
being the operator's **eye-tracking residue carrier**: *"record WHERE the eye
looked, not what it saw"*, an ephemeral overlay whose rows are the addresses a
search visited.

So the audit's question is not "does it work" but **"is the shipped mask a
residue carrier, or something else wearing the name?"**

## 2. Callers — measured

```
grep -rn "AttentionMask|attention_mask" --include=*.rs .   # workspace, minus target/
```

Three hits outside the two files themselves. All three are non-consumers:

| hit | what it actually is |
|---|---|
| `lib.rs:93` `pub mod attention_mask;` | module declaration |
| `lib.rs:94` `pub mod attention_mask_actor;` | module declaration |
| `mailbox_soa.rs:11` | a **doc comment stating the opposite**: *"wrap, **NO AttentionMask/LRU**, NO cross-cycle rollup — those are W6's"* |

Sibling repos, same grep: `MedCare-rs` 0 files · `OGAR` 0 files ·
`ndarray` 0 files.

**No caller exists anywhere.** The one file that names it in code-adjacent
prose does so to declare that it does *not* use it. `EXISTS-UNCALLED` is
recorded, per the falsifier's second branch.

## 3. What the type actually is

`AttentionMaskSoA` (`attention_mask.rs:46`):

```rust
pub struct AttentionMaskSoA {
    pub entries: Vec<AttentionMaskEntry>,   // (mailbox_id, w_slot, active, last_touched_cycle, plasticity_residual)
    pub max_active: usize,
    pub current_cycle: u32,
}
```

Its whole surface is `touch` / `evict_lru` / `tick` / `active_count` /
`is_active` / `entries`. Read plainly: **an LRU admission-control table keyed
by `MailboxId`, holding a 6-bit `w_slot` per entry.**

Three properties decide the verdict:

- **Keyed by mailbox, not by row or address.** The key is `MailboxId`
  (`= u32`, from `contract::collapse_gate`). There is no `NodeGuid`, no
  `NiblePath`, no classid, no row index anywhere in the file.
- **It is not a mask.** `entries: Vec<_>` with `.iter().find(...)` in `touch`,
  `.iter().filter(...).min_by_key(...)` in `evict_lru`, `.iter().filter().count()`
  in `active_count`, `.iter().any()` in `is_active` — **every operation is an
  O(n) linear scan over a `Vec`.** No bitset, no set algebra, no
  union/intersect/andnot.
- **It records occupancy, never a trajectory.** `last_touched_cycle` is
  overwritten on each `touch` (`:88`). The previous value is gone. A residue
  carrier must answer *"where did the eye look"* — this type can only answer
  *"is this mailbox currently claimed, and how stale."*

## 4. The name collision, traced to its origin

The shipped type is not a half-built residue carrier. It is a **complete
implementation of a different design**, and that design is on the board:
`causaledge64-mailbox-rename-soa-v1.md` §4 —

> *"**AttentionMask SoA — the session-ephemeral rename register file.**"*

Its job there is the exact inverse of a residue overlay:

| | rename register file (what shipped) | residue carrier (piece E) |
|---|---|---|
| direction | **wide identity → narrow slot** | **visited address → recorded** |
| why LRU | slots are **scarce** (5-bit G / 6-bit W / 8-bit style) and must be recycled | — nothing is scarce; the overlay is discardable whole |
| keyed by | `MailboxId` | the graph's own address (`NodeGuid`/`NiblePath`) |
| answers | *"which slot currently holds this identity"* | *"where did rung-n look"* |
| history | none — `last_touched_cycle` is overwritten | the point of the thing |

The originating plan is explicit that CE64 needs this because a `u32` OGIT
domain cannot fit in 5 bits: the rename table is what lets a wide identity ride
in a narrow field. **That is a compression concern, not an attention concern.**
The word "attention" in the name refers to *which identities are currently
resident in the scarce slot file* — a cache-occupancy sense, not the
eye-tracking sense §0 uses.

This is the **fourth homonym collision** this arc has had to separate, after
§3a's four "witness" surfaces, §3k's four "nibble" encodings, and §3l's three
"hydration" meanings. Same discipline applies: the shared word is not evidence
of a shared mechanism.

**The dependency map already warned about exactly this**
(`bindspace-mailbox-soa-dependency-map-v1.md:108`, `:191`):

> *"`attention_mask.rs` / `attention_mask_actor.rs` define their OWN
> `AttentionMaskSoA` — share only the `MailboxId`/`w_slot` vocabulary.
> **Do NOT conflate** with `MailboxSoA<N>`."*
> *"independent `AttentionMaskSoA`; do not fold into the migration."*

That warning was aimed at a different conflation (vs `MailboxSoA`), but it
generalises, and it is the reason piece E's grade was *"unaudited"* rather
than *"shipped for this"*.

## 5. Incompleteness even against its OWN spec

Worth recording so a future session does not read "shipped" as "finished":

| the rename-SoA plan §4 specifies | what shipped |
|---|---|
| `g_slots: [Option<u32>; 32]` (5-bit OGIT domain) | **absent** |
| `w_slots: [Option<WitnessId>; 64]` (6-bit witness palette) | present only as a bare `w_slot: u8` field — no `WitnessId`, no slot table |
| `style_slots: [Option<StyleId>; 256]` (8-bit style) | **absent** |
| fixed-size arrays (the point — a register FILE) | a growable `Vec` |
| *"Owned by a singleton ractor actor `AttentionMaskActor`"* | trait scaffold only — `attention_mask_actor.rs:2`: *"concrete ractor binding is sprint-12+ work"* |

Two further dead surfaces, both measured:

- **`plasticity_residual: u8`** — *"reserved for sprint-12+ learning signal"*.
  Grep across the workspace returns exactly two hits: its declaration
  (`:36`) and its initialisation to `0` (`:98`). **It is never read and never
  written non-zero.** A field that only ever holds its zero value carries no
  information — the same shape as the `closed_class_guess` 150/150 defect the
  falsifiability rule names.
- **`AttentionMaskMsg::BindReply`** — its handler is
  `AttentionMaskOutcome::NoOp` (`actor:97`). The variant exists, carries three
  fields, and does nothing. `reply_to: u32` is discarded via `..` in the one
  pattern that matches it.

## 6. The singleton problem — the design predates a ruling that killed it

§4 of the originating plan says the actor is a **singleton** ("one global
instance per session"). That is the shape the V3 mailbox ruling removed:
**no singleton CollapseGate; one mailbox = one kanban board as tenant**
(`CLAUDE.md` ★ V3 entry point; `E-CE64-MB-4` one-writer-per-mailbox).

So the unfinished ractor binding is not merely unfinished — **finishing it as
specified would rebuild a singleton the substrate deliberately eliminated.**
Anyone picking `attention_mask_actor.rs` up should treat "sprint-12+ work" as
superseded, not queued.

## 7. Consequences for `D-ACR-1`

The audit's purpose is to tell `D-ACR-1` what it can reuse. The answer is
**nothing structural**, and that is a cleaner starting position than a partial
fit would have been:

1. **Do not extend `AttentionMaskSoA`.** It is a correct implementation of a
   different contract, uncalled, keyed by the wrong thing, and O(n) per
   operation. Extending it would fold two mechanisms into one type — the
   conflation §3a/§3k/§3l each had to unpick.
2. **Do not reuse the name.** A fifth homonym is avoidable here at zero cost:
   `RowFocusMask` is already the board's own name for the primitive (S3.1b),
   and it says what it is — a mask, over rows.
3. **The governing choice rules out the shipped shape anyway.** §3k's
   operator-stated default is
   `Mask × ClassView/WideFieldMask → Mask`, bulk, *"never per-row, never
   per-frontier-size"*. `AttentionMaskSoA` is a per-entry linear scan — the
   precise shape `lance-graph-java`'s `Predicate.java` doc calls
   *"catastrophic"* (*"64,000 objects and 64,000 crossings for 64,000
   entities"*). It is not a starting point that can be optimised into
   compliance; it is the anti-pattern named.

### A measured constraint `D-ACR-1` must design around

`D-ACR-1`'s scope line says *"composable with `WideFieldMask` per S3.1b"*.
Measured this session (`class_view.rs:221`, `:251`, `:510`):

- `WideFieldMask` positions are **`u8`** — the universe is **capped at 256**,
  with a loud refusal (`WideMaskCapError::UniverseExceedsSocCap`) above it,
  *"never a silent drop or truncation."*
- It has real set algebra (`union`, `intersect`, `with`, `has`,
  `from_positions`, `EMPTY`).
- Its sibling `FieldMask` is `u64` with `MAX_FIELDS = 64`, and positions
  `>= 64` are **silently dropped** by `from_positions` / `with` (`:88`,
  `:100`) — documented as deliberate, but a trap if confused with the wide
  form.

**So "composable" cannot mean "the same type".** `WideFieldMask` addresses
**field positions** (≤ 256 of them); a `RowFocusMask` addresses **rows**
(a population, unbounded). These are two different bases, and composing them
is precisely the latent-third-basis problem the plan already parked:

> §6 **Y2** — *"`RowFocusMask × WideFieldMask` basis collision — the HTT
> **X4** latent-third-basis problem applies verbatim the moment a focus mask
> meets a field mask"* — *"X4 is audited, not solved, deliberately."*

The audit's contribution is to make that concrete rather than latent: the
collision is not a subtle semantic worry, it is a **cardinality mismatch with
a hard, loud cap on one side.** `D-ACR-1` must state which basis it is in and
what the composition operator means, before it composes anything — otherwise
the first `RowFocusMask` over a population > 256 either refuses loudly (if it
borrows `WideFieldMask`'s cap) or drops silently (if it borrows `FieldMask`'s
rule). Neither is a design; both are an inherited accident.

## 8. Verdict

| question | answer |
|---|---|
| Does a caller exist? | **No.** `EXISTS-UNCALLED`, workspace-wide + three sibling repos. |
| Is it a residue carrier? | **No.** It is a **rename register file** (`causaledge64-mailbox-rename-soa-v1.md` §4) — wide identity → scarce narrow slot, LRU because slots are scarce. |
| Is it a mask? | **No.** `Vec` + linear scan; no bitset, no set algebra. |
| Is it complete against its own spec? | **No.** 2 of 3 slot tables absent; `Vec` not fixed array; ractor binding never built; one field write-once-zero; one message variant a NoOp. |
| Can `D-ACR-1` build on it? | **No** — and it should not try. Piece D stays what §1 called it: *"the only one that is a missing primitive."* |

**Piece E's grade is corrected from *"shipped; unaudited for this use"* to
*"shipped for a DIFFERENT use; uncalled; not a basis for piece D."*** Six of
nine pieces still exist or are planned (§1's count is unchanged — E existing
was never the claim that E fits); what changes is that E is now known not to
serve D, so `D-ACR-1` starts clean instead of starting from an assumed reuse.
