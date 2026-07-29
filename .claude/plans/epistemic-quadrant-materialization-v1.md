# Epistemic quadrant materialization — grey/white, and what may be written

> **Status: PROPOSED.** Operator direction 2026-07-29: *"We can slowly switch to
> the idea of materializing thoughts as a grey matter / white matter explicit
> materialization for testing of known unknowns, unknown unknowns, unknown knowns
> and known knowns."* **"Slowly" is load-bearing** — this doc exists so the
> criterion is settled before any lane is minted. Nothing here is built.
>
> Companion: `.claude/knowledge/zero-copy-lens-law.md` § "The one apparent
> exception" (the rung test) and § "Grey and white" (the anatomy).

## 1. The matrix already exists — under its affective name

`ndarray::hpc::entropy_ladder::Quadrant::classify(entropy, energy)` → four
variants. It is the Rumsfeld matrix, reached from qualia rather than epistemics:

| entropy × energy | affective | epistemic | what it means here |
|---|---|---|---|
| low H, high c | `Wisdom` | **known known** | collapsed and evidenced — the committed lanes |
| high H, low c | `Staunen` | **known unknown** | surprise registered, entropy work not yet paid |
| high H, high c | `Confusion` | **unknown known** | material present, no ClassView elects it |
| low H, low c | `Boredom` | **unknown unknown** | no surprise *because there is no receptor* |

**Do NOT mint a second enum.** Reuse `Quadrant`; if the epistemic reading needs
to be nameable, it is a `#[doc]` mapping or an accessor, never a parallel type —
that is the prior-art failure mode (one insight, two ids, a divided search
surface for every future session).

**The asymmetry that drives the whole design:** `Boredom` and `Wisdom` differ
only by evidence (both are low-entropy). So the unknown-unknown quadrant is
*indistinguishable from settledness from the inside*. It cannot be detected by
any read; it can only be detected by a probe that **fails**. This is why the
operator's framing is "for testing" and not "for storing."

## 2. The trap — and the rule that defuses it

"Materializing thoughts" reads as licence to store everything. Under the law it
is not, and the rung test disqualifies the obvious first move:

> `Quadrant::classify(entropy, energy)` is a **pure function of two scalars that
> are already readable from lanes.** It is therefore recomputable from the lens,
> which by the law's own falsifier makes storing it a **cache with a correctness
> liability, not a memory.**

So the discipline, and it is the whole of this design:

| | recomputable from the lens? | verdict |
|---|---|---|
| a quadrant **assignment** | yes — `classify(H, c)` | **projection. Never stored.** |
| a quadrant **trajectory** | yes — episodic = Lance versions (`QueryReference::at`) | **projection. Never stored.** |
| a probe **outcome** (held-out test failed here) | **no — the world answered** | **observation. Stored.** |
| "no reader elected this structure until sweep S" | **no — it is a fact about the projection surface, which is in no lane** | **elevation. Stored.** |

**Assignment is a projection; the test outcome is an observation.** That single
line is what keeps this from undoing the zero-copy arc. Anything eligible to be
written here earns it by being *evidence the substrate did not already contain* —
never by being expensive to recompute.

## 3. Placement: grey vs white

Per `zero-copy-lens-law.md` § "Grey and white":

- **Grey (content)** — known knowns already live here. The corpus rows. This
  quadrant mints **nothing new**; proposing a "known-knowns tenant" is the
  second-projection anti-pattern with a philosophy hat on.
- **White (connectivity)** — the other three are all statements about the
  *projection surface*, not about content:
  - known unknown → a **marked missing** connection (where a read should have
    landed and did not);
  - unknown known → an **unelected** connection (present, no ClassView reads it);
  - unknown unknown → the **boundary** of the connection space (where the
    codebook has no centroid, residuals spike, the quorum has no witnesses).

That all three land in white matter is a result, not a convenience: it is the
same fence as *"cross-tenant pointers are legitimate; cross-tenant values are
not."*

## 4. Prior art to consume, never re-derive

| surface | where | covers |
|---|---|---|
| `Quadrant::classify` | `ndarray::hpc::entropy_ladder` | the 2×2 itself |
| `DkPosition` (MountStupid → PlateauOfMastery) | `planner/cache/triple_model.rs` | the meta-confidence axis over self/user/impact |
| `curiosity_mul` (D-SCI-4) | planner MUL | the exploration gateway — where to spend entropy work |
| D-SRS-3 basin self-codes + held-out gate | — | "where am I uncertain", already with a held-out falsifier |
| D-SRS-3b evidence-composite basin uncertainty | — | operator-corrected uncertainty compositing |
| D-SRS-4 derivation provenance + confidence delta | — | the self-reference falsifier |
| `Locus::Quorum` / `Locus::Contradiction` | `causal_witness.rs` | the shipped precedent for *stored* entropy work |
| `world_model.rs` / `world_map.rs` | contract | `WorldModelDto` quadrant-snapshot (self / other / …) |

The known-unknown quadrant is **substantially already built** (D-SRS-3/3b +
curiosity_mul). The genuinely new work is the other two.

## 5. Falsifiers — required before any lane is minted

Per the P0 falsifiability rule, each must have both halves:

- **F1 — the unknown-known sweep can FIRE and can STAY SILENT.** On a corpus with
  deliberately unelected structure it must find it; on a corpus where every lane
  has a reader it must return empty. A sweep that reports "latent structure" on
  every input carries exactly as much information as one that never fires.
- **F2 — the unknown-unknown boundary is only credible from a FAILED probe.**
  Assert the held-out set is non-trivial (`kept * 3 < total`-style anti-vacuity),
  and that a probe which succeeds does *not* mark a boundary. A boundary detector
  that fires without a failure is measuring its own prior.
- **F3 — the recompute falsifier, on every stored value.** Recompute it from the
  lens; if it comes back equal, it was a projection and must be deleted. This is
  the mechanical guard against § 2's trap and applies to every field, forever.
- **F4 — quadrant migration must be observable.** Paying entropy work on a
  `Staunen` item must move it (`Staunen → Confusion → Wisdom`), and *not* moving
  under investment is itself the finding. Without this the quadrant is decoration
  (the `heel_threshold` inertness lesson).

## 6. Sequencing (explicitly NOT now)

Ordered behind the declared priorities — W6 antecedent binder, the ZC-2
remainder, W8 hygiene:

1. **Q0 — census, no code.** Map every existing lane to grey/white and to a
   quadrant. The `zero-copy-lens-law.md` grey/white table is flagged *"the
   governing frame, not a census"*; this closes that gap and is the honest
   prerequisite for everything below.
2. **Q1 — the unknown-known sweep** (F1). Highest value, lowest risk: it reads
   the existing surface and mints nothing. It answers "what does the substrate
   already hold that nothing reads?"
3. **Q2 — probe-outcome storage** (F2 + F3). The first genuinely new stored
   evidence. Needs a named source before a lane — the `revisions:` lesson:
   *if you cannot name what the borrow borrows from, you are not ready.*
4. **Q3 — migration accounting** (F4). Only meaningful once Q1/Q2 exist.

**Gate:** no lane is minted before its falsifier is green. If Q1's sweep cannot
demonstrate both halves of F1 on a real corpus, the next deliverable is the
probe, not more design.
