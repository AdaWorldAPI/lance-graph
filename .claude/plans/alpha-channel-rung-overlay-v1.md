# Alpha-channel rung overlay — v1

> **Status:** PROPOSED. No code. Register-before-code, per the dialectic-engine
> build order. Every "exists" claim below was verified by reading the file
> named, this session; every "absent" claim by a grep that returned nothing.
>
> **What this plan is:** the design for the ONE empty row of the thinking
> table — `hhtl-thinking-tables-le-contract-v1.md` §2.3, row **Rung ladder**,
> *"(unassigned) · see §3 — two axes · unminted, undesigned"*. It mints no new
> type. It is the scraping of an operator brainstorm (2026-08-21) onto homes
> that already exist, plus a short list of what genuinely does not.

## §0 — The operator's picture, in his own frame

A Photoshop **alpha channel**: an ephemeral layer laid 1:1 over the ontology,
carrying the residue of a search — *"du weißt, dieses residue ist bei der
Suche von Patient X zum Graphen übergesprungen"*. Consequences he named, each
of which turns out to be a constraint rather than a metaphor:

1. **"Alpha channel damit der Graph nicht von Patienten kontaminiert wird"** —
   the overlay is written; the graph is not.
2. **"Eye tracking der Ontologie-Gedanken"** — record WHERE the eye looked,
   not what it saw. *"Müssten wir alles im Patienten reinschreiben, würde es
   nicht mehr mit vertretbarem Aufwand gehen."*
3. **"Gedanken 2. Ordnung als thinking about thinking als graph overlay mit
   der gleichen Adresse wie der Graph aber separate thinking tables."**
4. **"Rung levels 2–10 als Alpha layer projizieren."**
5. **"Eine Maske über die Aktivitäten."**
6. **"Der Rung über dem Rung hat ein Bewusstsein über focus of attention — wo
   die Gedanken davor waren."**
7. **"Eye tracking ist kein Beweis, aber damit löst du das needle-in-a-haystack
   Problem"** — the overlay is a *pruner*, never a proof. (Same grade
   `ONTOLOGY_BAKE_STATE.md` already gives HHTL: *"a sound pruner but not a
   proof"*.)

## §1 — The scraping: nine pieces, six already have homes

| # | Brainstorm piece | Home | State (verified) |
|---|---|---|---|
| A | separate thinking tables at the graph's own addresses | HTT §2.3, row **Rung ladder** | the empty slot — *this plan* |
| B | rungs 2–10 as alpha layers | HTT §3 rung carve (two axes) + dialectic **V3** field elevation (S11) | designed, unbuilt |
| C | **mask over activities** | `PhaseCensus`, `lance-graph-supervisor/src/kanban_actor.rs` | **SHIPPED** — read-only, one `&self` pass |
| D | **focus of attention** | `RowFocusMask` (STATUS_BOARD S3.1b) | **named on the board, ABSENT from code** — grep hits only `STATUS_BOARD.md` + one handover |
| E | eye-tracking residue carrier | `cognitive-shader-driver/src/attention_mask.rs` + `attention_mask_actor.rs` | shipped; **unaudited for this use** |
| F | sudoku elimination as the search | dialectic §3 five tactics + `ReasoningGap`/`GapKind`, `lance-graph-planner/src/nars/tactics.rs` | **SHIPPED** (V1, `E-DIALECTIC-V1-TACTICS-IN-PLANNER-1`) |
| G | contamination boundary (one-way) | — | **design gap**, §2 |
| H | second-order overlay at the same address | HTT §2.3 + G | **design gap**, §3 |
| I | 64k parallel rungs 1–10 | dialectic **V4** (64k SIMT lowering) | explicitly gated: *"only after V0–V3 green at small scale"* |

**Six of nine already exist or are already planned.** The plan's real content
is D, G and H, and D is the only one that is a missing *primitive*.

## §2 — The contamination boundary is the ownership rule, not a new guard

*"Damit der Graph nicht von Patienten kontaminiert wird."* The substrate
already enforces exactly this, and it is worth stating so nobody builds a
second mechanism:

- **One writer per mailbox** (`E-AGENT-LOG-SHARED-SINK-ANTIPATTERN-1`, and the
  runtime original `SoaEnvelope::mailbox_owner` + the write-on-behalf iron
  rule).
- Therefore: the overlay is a tenant whose **owner is the session mailbox**,
  never the ontology mailbox. A patient-derived write to an ontology row is
  not "discouraged" — it has no owner that could perform it.

**The invariant, stated once:** *the overlay reads the graph; the graph never
reads the overlay.* One direction, compile-checkable at the owner, and it is
what makes rung-n+1 safe to compute from rung-n residue without the residue
becoming evidence.

> **Consequence that must not be lost:** this is also why the overlay may be
> **discarded whole**. It is not a cache of derived truth (which would need
> invalidation); it is a *record of where attention went*. Dropping it costs a
> re-search, never a correctness question.

## §3 — Second-order = same address, different table

*"Gleiche Adresse wie der Graph, aber separate thinking tables."* In the
contract's own vocabulary this is not a new addressing system — HTT §2.2
(**⊘ WITHDRAWN**) already ruled that out: *one fabric, several
ClassView-resolved readings, never alternative addressing systems.*

So a second-order thought at node `g` is:

- the **same** `NodeGuid` (the address is shared, that is the point);
- a **different thinking-table row** — a different `(classid, rail)` pairing
  resolved by the ClassView, per §2.3;
- occupancy **sparse** — the eye-tracking argument is exactly that only the
  visited addresses are materialised. "1:1" names the *addressing*, never the
  occupancy.

## §4 — Deliverables

| D-id | Scope | Repo | Falsifier |
|---|---|---|---|
| **D-ACR-0** | Audit `attention_mask.rs` / `attention_mask_actor.rs` against piece E: is the shipped mask a residue carrier, or something else wearing the name? Report, no code. | lance-graph | the audit names a caller, or records EXISTS-UNCALLED |
| **D-ACR-1** | `RowFocusMask` — the one missing primitive (piece D). Mask over rows visited, composable with `WideFieldMask` per S3.1b. | lance-graph | can-fire **and** can-stay-silent on non-trivial input; a focus over 0 rows and over all rows must be distinguishable from a real one |
| **D-ACR-2** | Mint the **Rung ladder** rail (HTT §2.3 row) — gated on §8 Q3 mint decision, NOT on this plan | lance-graph | `rail_carving` gains its first non-default consumer |
| **D-ACR-3** | The one-way invariant as a test, not prose: an overlay write addressed to an ontology-owned row must fail at the owner | lance-graph | the negative case is the test; a passing write is the bug |
| **D-ACR-4** | Second-order row (§3) over D-ACR-1 + D-ACR-2 | lance-graph | a rung-2 read reconstructs where rung-1 looked, on a fixture where the answer is known independently |
| **D-ACR-5** | 64k lowering | lance-graph | **BLOCKED** — dialectic V4's own gate: V0–V3 green at small scale first |

**Order is not negotiable:** D-ACR-0 (audit) → D-ACR-1 (the primitive) →
D-ACR-3 (the boundary) → D-ACR-2/4 (mint + second order) → D-ACR-5. D-ACR-2
and everything after it sit behind an operator mint decision that this plan
does not pre-empt.

## §5 — Non-goals

- **No new address type.** S3.0/PR #973 was closed exactly here — *"CLOSED —
  NOT NEEDED (use `IdentityQuad` / `ClassAddr` / V3 rail)"*, and the ladder's
  empty column was ruled to be **HYDRATION, not ADDRESS**. The overlay is
  hydration over existing addresses.
- **No CE64 bit.** Bits 59..63 are spoken for (`TRUTH_SHIFT` 59–60 /
  `SPARE_SHIFT` 61–63) and the band there is set only by an explicit
  `with_reasoning_band()` call — **nothing derives it**. The overlay must not
  become a fifth derivation path into those bits.
- **No `ENVELOPE_LAYOUT_VERSION` bump.**
- **No proof claim.** Piece 7: a pruner. Any deliverable that starts asserting
  the residue *justifies* a conclusion has left this plan.

## §6 — Deferred (missing integration)

| # | Item | Why deferred |
|---|---|---|
| **Y1** | Whether the rung ladder's **two axes** (HTT §3) are the same two the alpha layers need, or a third pairing. Unmeasured. | §3 is session-measured for the ladder, never for an overlay |
| **Y2** | `RowFocusMask × WideFieldMask` basis collision — the HTT **X4** latent-third-basis problem applies verbatim the moment a focus mask meets a field mask | X4 is audited, not solved, deliberately |
| **Y3** | The residue's **retention policy**. §2 says it may be discarded whole; nothing says when it is. | needs a measured working-set size, which needs D-ACR-1 first |

## §7 — What this plan does NOT claim

It does not claim the overlay improves recall, that eye-tracking residue finds
needles, or that 64k rungs are reachable. It claims one thing: **the empty row
in §2.3 has a design now, and six of its nine parts already exist.** Every
number that would justify the rest has to be measured after D-ACR-1.
