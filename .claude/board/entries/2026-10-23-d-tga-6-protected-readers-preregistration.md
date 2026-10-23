# D-TGA-6 / Probe D — protected readers of a squash (pre-registration)

**Status:** PRE-REGISTERED 2026-10-23, before any code. Owner: CC1
(session_01VncKoyG6JkpsoqhqTGTv3f), claimed on EXO-50 `/036`; ownership sync
`GPT:gpt-ada-01/20261023-cc1-cc2-ownership-sync-13`.
**Crate:** `crates/thought-git-probe` (lab only, its own workspace).
**Builds on:** D-TGA-5 (`tests/squash_gate.rs`, merged in #1736: strict R1/R2, P8
Commit-only).

## Question

A squash replaces a family of lane chains with one delta. Its readers are
everything that may later ask what happened. Which readers does the D-TGA-5
`Squashed` value (cells, blame, order, seal) answer exactly the way the full
history does? For each reader it cannot answer, what must be retained as a
witness, or must the squash be refused?

Refusal is a valid result. The question is never "can we compact": it is
"which readers survive compaction, at what retained cost".

## Model

- **History.** Lanes land interleaved. Step `i` lands every lane's `i`-th
  thought together in one version, through `oracle::next_version`, starting from
  a prior version `P`. `P` may be non-empty. The history is `v_1 .. v_T`.
- **Squash.** `squash_gate` as merged. If accepted, it is landed once on `P`.

## Readers

| reader | history answer | retained answer |
|---|---|---|
| `Final` | `v_T.read(c)` for every cell | the squash landed on `P`, read at `c` |
| `Intermediate` | `v_k.read(c)` for `1 ≤ k < T` | none: the squash keeps no per-step state |
| `Contradictions` | union of every `v_k.contradictions` | a witness log |
| `Divergences` | union of every `v_k.divergences` | a witness log |
| `Multiplicity` | committed writes per cell, from the thought log | a witness count. Without one, the best the squash has is blame's popcount, which counts lanes and not writes |

Blame and order are already pinned (P6 and the seal test). They are not
re-tested here.

## Fixtures

The (a)–(d) fixtures come from the falsifier co-architect of the 2026-10-23
coresearch.

- (a) Prune, Veto and Plan writes between commits.
- (b) A committed write that contradicts a given, then a write agreeing with it.
- (c) Two lanes diverge at step 1 and agree at step 2.
- (d) The same value written 3× by one lane, against 1× by another lane.
- (e) The clean lanes of D-TGA-5, landed on a NON-EMPTY prior version.

## Predictions (each is a test)

- **D1.** `Final` is preserved on (a)–(e), and the witness is empty.
  Kill: any differing final read.
- **D2 (loss is real).** Without a witness the squash loses `Contradictions`
  on (b), `Divergences` on (c) and `Multiplicity` on (d). Anti-vacuity: each
  history answer is non-empty or greater than 1 where claimed.
- **D3.** When `Contradictions`, `Divergences` and `Multiplicity` are protected,
  the gate retains a witness, and the retained answers equal the history answers
  on (a)–(e). Disables: drop each witness field in turn. Each disable must turn
  its fixture red.
- **D4.** When `Intermediate` is protected and some intermediate read differs
  from the final read, the gate refuses with `Unretainable(Intermediate)`.
  Silent twins: a one-step history, and a multi-step history whose intermediate
  reads all equal the final, are accepted. Disable: remove the intermediate check.
- **D5 (bytes).** Transient bytes grow with total writes. Retained bytes (delta,
  blame, order, seal, witness) are measured separately, with the witness bytes
  given on their own. This is reported, not gated: the witness grows with the
  number of anomalies and the order list grows with the number of thoughts. A
  proof object is not assumed to be constant size.

## Kill conditions

- A protected reader changes silently.
- R1 or R2 relaxes.
- P8 breaks.
- The witness claims a reader it cannot answer.

## Not in scope

- `ProofRef` and the shared reference type (Option B, still being agreed with CC2).
- F5 (a synthesized head).
- F6 (correlated derivations).
- Any change to `oracle.rs`.
