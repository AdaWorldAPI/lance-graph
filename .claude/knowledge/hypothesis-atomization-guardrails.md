# Hypothesis atomization — Kahneman/Tversky guardrails for verdicts

**READ BY:** the main thread and any agent BEFORE giving a verdict on a
hypothesis, plan, proposal or PR; `hypothesis-atomizer` (the gate);
`premise-auditor` (its sibling for questions); both council harnesses
(`/5plus3` Phase 0, `/coresearch` Phase 0).

**DECISION (operator, 2026-10-21):** "They are different hypotheticals in
one. I absolutely despise if you use Kahneman and Tversky blindness turning
separate ideas into one rejectable item." SCOPE: every verdict, review,
finding or board post that judges someone's idea. REVISIT WHEN: never on cost
grounds.

## The failure, measured (2026-10-21, EXO-50 `session_01G22yTnyw/026`)

The question was: *"Can ClassView and the existing ISA/PartOf coordinate
rails make relationships executable through QR folds, without creating
another domain-specific edge implementation?"* It holds at least four
independent claims. The answer:

1. **Bundled them.** It gave one mixed verdict ("yes for hierarchies, no for
   everything else") instead of one verdict per claim.
2. **Substituted an easier question.** It refuted "computing the ISA
   transitive closure with QR mxm is pointless", which no plan proposes. The
   plan proposes the rails as *coordinate addresses* for a bounded QR focus.
3. **Dropped a scope qualifier.** It rejected the HHTL16 zippers because
   "32 B does not fit the 16 B facet". The plan says, in so many words, that
   they are views and "not a claim that two 16-byte zipper arrays are already
   packed into one 16-byte `FacetCascade`".
4. **Ignored a code fact it had just read.** It said the rails "do not carry"
   membership. `facet.rs:30-35` lists `(group:member)` as one sanctioned
   reading of an 8:8 tile.
5. **Anchored on its own work.** "EXO-119 should build on #1659" rated the
   proposal against the answerer's own PR.
6. **Swapped in its own hypothesis.** "ISA ancestry is a prefix mask" is the
   answerer's idea. It was presented as a verdict on the proposer's idea.

Every one of these is System-1 behaviour. The bundle *felt* answerable in one
breath, so it was answered in one breath.

## The mechanisms these guardrails target (stated once, used as checks)

| bias | how it shows up in a verdict |
|---|---|
| **attribute substitution** | a hard question ("does X work as stated?") is answered with an easier one ("is X needed for the obvious case?") |
| **conjunction / disjunction errors** | a bundle A∧B∧C is judged as a unit; one weak conjunct sinks all of them, or one strong conjunct carries all of them |
| **representativeness** | a carrier that *looks like* a hierarchy is judged as only a hierarchy, ignoring readings the code also sanctions |
| **WYSIATI** | the qualifiers and alternatives absent from the verdict-writer's head (view-only, optional, candidate, lab) are treated as absent from the proposal |
| **anchoring** | the verdict-writer's own artifact becomes the yardstick the proposal is measured against |
| **framing** | the proposer's wording ("without creating…") is turned into a pass/fail test of the whole bundle |
| **confirmation** | evidence is gathered for the first verdict formed, not for each claim on its own |

These are *checks to run forward*, not labels to stick on anyone afterwards
(the `CLAUDE.md` MIRROR rule still holds: ask, never diagnose).

## The guardrails (G1-G10), in the order they run

**G1 — Atomize first, verdict later.** Before any verdict, write the atom
table: `H1..Hn`, each **one subject, one predicate, one condition**. Split
on every "and", "with", "through", "without", "via", "so that", and on every
noun that names a separate mechanism. A compound question with N mechanisms
has at least N atoms. No verdict text may come before the table.

**G2 — Quote-anchor every atom.** Each atom carries the proposer's own words
with a location (`file:line`, plan §, comment id). An atom with no quote is
the verdict-writer's own idea. It goes in a separate block headed
`MINE (not the proposer's)` and is never used to judge a proposer's atom.

**G3 — Carry every qualifier.** Copy each atom's scope words verbatim:
*view, candidate, optional, lab, bounded, typed, not persisted, illustrative,
if, only when*. A refutation that does not hold under the qualified reading
is **void**. Test: would the refutation still stand if the proposer wrote the
qualifier in capitals?

**G4 — Steelman before falsifying.** For each atom, state its strongest
reading in the proposer's terms, plus what would have to be true for it to
work. Then attack that reading, never a weaker neighbour.

**G5 — Name the question your evidence answers.** For every piece of
evidence, write in one line the question it actually answers. If that is not
the atom, the evidence does not bear on the atom and the atom stays `OPEN`.
This is the guard against attribute substitution.

**G6 — One verdict per atom, from a fixed vocabulary.**

| verdict | needs |
|---|---|
| `SUPPORTED-IN-CODE` | a cited code fact that makes the atom true or feasible as stated |
| `REFUTED` | a cited code fact, a passed falsifier or a contradiction with a ruling, against the **qualified, steelmanned** atom |
| `OPEN` | neither yet; name the probe that would decide it |
| `NOT-PROPOSED` | the proposer never claimed it (moves to `MINE` or is dropped) |

**There is no bundle verdict.** "Yes for X, no for Y", "partly", "mostly",
"in principle" and "it depends" about the whole are banned. If a summary is
wanted, it is the tuple of atom verdicts, `H1 S · H2 O · H3 O · H4 O`.

**G7 — Declare dependencies; propagate only along them.** Draw the edges
(`H3 needs H2`). A refutation of `Hk` reaches another atom only through a
declared edge, and the edge must say *why*. Siblings are independent by
default. A weak conjunct never sinks the bundle; a strong one never carries
it.

**G8 — Asymmetric burden for REFUTED.** `REFUTED` needs evidence. Absence of
evidence, a missing probe, unfamiliarity or "seems unnecessary" give `OPEN`.
Necessity ("we don't need it") is a separate question from truth ("it does
not work") and gets its own atom if raised.

**G9 — Own-work anchor check.** If the verdict-writer authored an artifact
the verdict cites (a PR, a crate, a plan), mark the line `OWN-WORK` and give
the verdict without it as well. If the two differ, report both.

**G10 — Reread the source after drafting.** Before publishing, reread the
proposer's text against the atom table. Every qualifier must appear in an
atom, and no atom may contradict a sentence in the source. A single miss
blocks publication. This is a mechanical check, not judgement.

## Output contract (every verdict on someone's idea)

1. The atom table: `id | atom (one predicate) | verbatim quote + location |
   qualifiers | depends on`.
2. Per atom: the steelman (G4), the evidence with the question it answers
   (G5), and the verdict (G6).
3. A `MINE` block for the verdict-writer's own hypotheses, labelled as such.
4. The verdict tuple. No prose summary over the bundle.
5. For every `OPEN`, the probe that decides it.

## Can it fire / can it stay silent

- **Fires:** the 2026-10-21 answer above fails G1 (one mixed verdict), G3
  (the "view" qualifier was dropped for HHTL16), G5 (the closure argument
  answered a question nobody asked) and G2 (a `MINE` hypothesis was used as
  the verdict).
- **Stays silent:** a single-predicate question ("does `WideFieldMask::with`
  take a `u8`?") has one atom, one quote and one cited verdict, so every
  guard passes without extra ceremony.
