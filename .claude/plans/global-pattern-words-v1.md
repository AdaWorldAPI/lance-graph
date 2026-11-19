# D-GPW-0..6 — Global pattern words: one word with a hole, its two dimensions and its absent-case policy as parameters

> **Draft / documentation only, 2026-11-19.** This file authorizes no Rust
> code, no word in any lens, no crate and no change to DT64, R2IL or Quack.
> Each step below needs its own PR with a parity test and a disable run.
>
> **Origin:** the owner, 2026-11-18 and 2026-11-19, recorded verbatim on
> EXO-329:
> - "SQL WHERE is a polyfill in F### space, never datafusion literals, not if it's our code"
> - "The above are SQL methods, not Codebook entries"
> - "If we own Zammad-rs, Why should we settle with strings instead we should go with methods"
> - "I understand LIKE 'a.%' is similar to MeTTa and spaCy?"
> - "The idea is to introduce these as global patterns and spaCy and MeTTa as 2 dimensions of the same issue"
> - "Similar to NaN handling parameters"
>
> **The reading in §1 is this session's. The owner has not confirmed it.**
> Status of every claim below is marked.

## 0. Problem

A consumer that inherits a text match has three unrelated mechanisms for one
question, and picks among them itself.

MEASURED in zammad-rs (PostgreSQL 16.15 as oracle):

- Zammad asks for "a permission and everything under it" with
  `name = ? OR name LIKE 'x.%'`: 57 of 512 SELECTs of the recorded workload.
  The tree exists only as dots in `permissions.name`.
- zammad-rs #41 replicated it: a SQL fragment reader of its own, and `LIKE`
  matched against the column's strings.
- zammad-rs #45 replaced it: the intake declares the tree as a relation of
  identities (computed with Zammad's own `Permission.with_parents`), and a
  consumer-local condition `under` answers over identities. All 57 answers
  equal PostgreSQL.
- **The two forms disagree in one case.** After the row `a` is deleted,
  `name = 'a' OR name LIKE 'a.%'` still returns `a.b.c` in PostgreSQL; the
  relation form returns nothing, because a name with no row is no member.
  zammad-rs chose the identity reading and states the difference in a test.
  A consumer should not be the place that choice is made.

The same choice appears twice more in the same work:

- A text of a known type either names a member of a declared set or it does
  not (the owner's `ValidateSet`): refuse it, or treat it as no member.
- A comparison against NULL: two-valued or Kleene. This one already has a
  home, `Logic` on a `Guard` (`lance-graph-contract/src/set_algebra.rs:195`).

## 1. Reading (WORKING-MODEL, unconfirmed)

**A pattern is one global word with a hole.** Its dimensions and its handling
of the absent case are parameters of the word, the way rounding and NaN
handling are parameters of the 8-bit codec.

VERIFIED-IN-CODE, the precedent the owner named: `ndarray/src/hpc/codec8.rs`.
`Codec8<T, S, R, N>` is one operation. `R: Rounding` and `N: NanPolicy<S>` are
independent marker parameters. `NoNaN` has no code for NaN and the checked
encoder rejects it; `NaNAtMin` reserves the code -128 to mean NaN. The
operation is not duplicated per policy, and the policy is fixed where the
codec is named, never chosen per value.

| Parameter | What it decides | spelled today as |
| --- | --- | --- |
| **within** a value (the spaCy dimension) | how a value is split into parts, so which spans a hole may cover | `Permission.with_parents` in a Ruby intake; the `%` and `_` of `LIKE` |
| **across** values (the MeTTa dimension) | which relation a hole binds through, so which members a variable may take | a declared relation and a key set in zammad-rs; `Space::query` in Hyperon |
| **the absent case** (the NaN of matching) | what a name with no member means: no code for it, or a reserved one | implicit, and different in each mechanism |

`LIKE 'a.%'` is the flat-string form: it does both dimensions at once without
naming either, and its absent-case policy is "a name needs no member".

## 2. What already exists (searched before proposing)

- VERIFIED-IN-CODE. The global lens: `lance-graph-r2il/src/polyfill.rs`,
  `POLYFILL_LENS = 0xFFFF :: 0xF000`, 17 words. Position binds; labels are
  aliases. One position has a name per language
  (`AliasLanguage::{R2il, Sql, Quack}`). No word has a hole: `=`, `<`, `AND`,
  `OR`, `NOT`, the folds. `LIKE` and `IN` have no position.
- PROPOSAL, D-MTP-0 done 2026-10-14: `.claude/plans/metta-polyfill-v1.md`.
  MeTTa as a front end over the masking-algebra polyfill, attached at
  `Space::query(&Atom) -> BindingsSet`. Proposed crate `lance-graph-metta`,
  not created. This plan builds on it for the across dimension; it does not
  restate it.
- CURRENT-CONTRACT:
  `.claude/plans/d-cgmm-lazy-codegraph-fact-demand-v1.md:125`. A spaCy-like
  `Doc/Token/Span/Matcher` pipeline is "a useful cold annotation interface
  analogy, not a second crate or text-sequence representation". This plan
  keeps that: the within dimension is a parameter of a word, never a second
  representation of text.
- CURRENT-CONTRACT, D-DT64-7 §1b: "A language that only renames the shared
  algebra gets an alias view, not a concern range." So SQL, spaCy and MeTTa
  spellings of one pattern word are alias views.
- VERIFIED-IN-CODE. `Logic::{TwoValued, Kleene}` is already an absent-case
  parameter, for truth, fixed at registration on the `Guard`.

Not searched: whether `lance-graph-nal`, `deepnsm` or
`grammar-constraints-probe` already hold a token or span type that the within
dimension should reuse. D-GPW-1 is that census.

## 3. Rules proposed

- **R1. One word, parameters, no per-language word.** `LIKE`, a spaCy token
  pattern and a MeTTa `match` over the same relation are names for one
  position. A new position is added only when no parameter choice of an
  existing one expresses the operation (the existing ISA rule).
- **R2. The absent-case policy is declared with the class field, at intake.**
  Never at the call, never by the consumer's code path. It joins `Logic` as
  the same kind of declaration ("the injecting intake defines the class
  reading").
- **R3. Text stops at the boundary.** A pattern's constants are members
  (codebook positions) by the time the word runs; splitting a value into
  parts happens at intake. A word that needs the text of a value at run time
  is refused.
- **R4. A consumer does not imitate a missing word.** Until a pattern word
  exists a consumer keeps its current path, named as "no word", and does not
  expand the pattern into other words.
- **R5. Every parameter choice has a differential oracle.** For `LIKE` that
  is PostgreSQL; for a MeTTa match the engines measured in D-MTP-0.

## 4. Deliverables

| D-id | What | Gate |
| --- | --- | --- |
| D-GPW-0 | This plan and the decisions in §5 | owner |
| D-GPW-1 | Census, counted from source with the command per number: every place lance-graph and its consumers match a pattern today (SQL `LIKE`/`IN`, the cube's `reach`/`closure`, `lance-graph-nal`, `deepnsm`, `grammar-constraints-probe`, zammad-rs `under` and its fragment reader), with the within form, the across form and the absent-case behaviour of each | every row read, not grepped; the three zammad-rs rows reproduce the numbers in §0 |
| D-GPW-2 | The absent-case policy as a declared parameter beside `Logic`: its variants and where it is stated | the zammad-rs deleted-parent case answered both ways from one declaration, each equal to its oracle (PostgreSQL for the prefix reading; the relation's own definition for the member reading); a disable run |
| D-GPW-3 | The across dimension: one pattern word over a declared relation, in the lens the owner decides (§5.1), with its SQL and MeTTa alias names | equals zammad-rs `under` on the 57 recorded lookups and equals D-MTP-0's oracle on its join program; a name that is no alias is refused |
| D-GPW-4 | The within dimension: how a value's parts are declared at intake so a hole can cover a span, reusing whatever D-GPW-1 finds | `LIKE 'a.%'`, `LIKE 'a_'` and an escaped pattern answered without text at run time, equal to PostgreSQL |
| D-GPW-5 | zammad-rs: `under` and the fragment reader's `LIKE` replaced by the word; its own paths kept only as the oracle | all 512 SELECTs still equal PostgreSQL |
| D-GPW-6 | The `lance-graph-metta` front end of metta-polyfill-v1 lowering `match` onto the same word | D-MTP's own gates |

## 5. Decisions needed

1. **Which lens.** The hardcoded lens `0xFFFF :: 0xF0__` (every consumer gets
   the word; D-DT64-7 puts SQL there) or a consumer's own
   `Domain::AppId :: 0xF###`. Proposed: the hardcoded lens.
2. **The absent-case variants.** Proposed, by analogy with `NoNaN` /
   `NaNAtMin`: *no member* (an unknown name is refused or matches nothing)
   and *reserved member* (a name with no row still has a position to stand
   on, which is what a prefix match needs). Whether refusal and
   matches-nothing are two variants or one.
3. **Whether `Logic` and the absent-case policy are one parameter or two.**
   Both say what an absent thing means. Proposed: two declarations of one
   kind, kept apart until a case needs them merged.
4. **Call parameters.** A pattern's constants come with the call. A guard
   carries none (EXO-291, with zammad-rs's count: 53 conditions with
   constants for 18 shapes). D-GPW-3 cannot register a pattern once per shape
   without it.

## 6. Not verified by this session

- Hyperon and PeTTa were not run here; §2 quotes metta-polyfill-v1.
- spaCy was not run or read here.
- Whether an absent-case policy can be a compile-time marker, as in
  `Codec8`, or must be a registered value like `Logic`: not examined.
