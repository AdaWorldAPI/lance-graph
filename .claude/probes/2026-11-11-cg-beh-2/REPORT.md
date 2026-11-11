# D-CG-BEH-2: the ON × WHEN × DO contract as separate registered facts

Source `317c4ffc0`. Crate `crates/cg-beh-probe` (lab, own workspace). Read-only: no production
registration is edited, no word is added, nothing is executed. Owner EXO-155.

## Gate A: registered facts

`lab.facts.tsv` holds this crate's own class (three methods; method 1 has ON, WHEN and DO).
`foundry.facts.tsv` holds Foundry's `CanCook` class as `World::new()` registers it: 16 methods over
a 211-field basis. Format: `crates/cg-beh-probe/README.md`.

`CanCook(item 0, q 1)`:

| part | registered |
|---|---|
| ON | 9 positions: `ops:stock/0..2`, `ops:stockKnown/0..2`, `ops:staff/0`, `ops:equipment/0`, `ops:offers/0` |
| reads | the same 9 positions |
| WHEN | two-valued conjunction of 9 leaves; every leaf number is one of the reads |
| DO | none (`DO_NONE`), for all 16 methods |
| binding | `2fb0b32d08e66e2cd2ad070eaf572c82`, signature revision 1 |

ON equals reads because `can_cook_method` builds both from the guard's fields. The store keeps them
as separate masks.

## Gate B: four kinds of change

Each prediction was compared with what the stores answer.

| change | registered fact that moves | binding digest | old reference at the new store | stale |
|---|---|---|---|---|
| ON declaration | ON positions | moves | `BindingMismatch` | reference stamp; `methods_on` answers |
| WHEN guard, other rows | guard witness | does not move | `GuardMismatch` | guard witness |
| WHEN guard, same rows written differently | expression only | does not move | admitted but for the issuer | nothing |
| DO registration (step order) | ordered steps | moves | `BindingMismatch` | reference stamp |
| DO implementation (a dispatch arm) | none | does not move | admitted | executed parity of each word that has a user |
| field basis, same count | ordered predicates | moves for every method of the class | `BindingMismatch` | witnessed `resolve` / lane binding; reference stamp |
| class schema revision, same basis | schema revision | does not move | admitted but for the issuer | `resolve` at the old revision (`SchemaRevisionMismatch`) |

Evidence paths: the first four and the last are `DT64_REGISTRATION` only. The DO implementation row
is `SOURCE_STRUCTURE` (line in arm, arm names opcode, opcode at table position) joined to
`DT64_REGISTRATION` (DO steps on that position). No row is `EXECUTED_PARITY`.

Measured on the real dispatch source: line 977 is in the `INT_S_LESS` arm (976 to 1018), word 5, used
at ordinal 0 by lab method 1. The `INT_ADD | INT_MULT | INT_S_DIV` arm (1145 to 1162) reaches three
words; one has a user in the lab snapshot.

## Gate C: falsifiers

24 tests. `disable_runs.py` removes 16 guards one at a time; each named test fails (16 of 16).

| falsifier | result |
|---|---|
| basis altered, same field count | `resolve` refuses with `PredicateBasisChanged`; no facts are read |
| guard changed, DO unchanged | only `When`; DO rows and binding digest equal |
| independent DO steps reversed | only `Do` |
| dependent DO steps reversed | the store refuses the registration (`UndeclaredRead`) |
| dispatch mapping removed | that word is refused as unknown; the other 16 answer as before |
| reference from a cloned store | `ForeignStore`; no facts |
| reference with its guard witness dropped | `GuardDropped`; no facts |
| unrelated method | `Unchanged` under every variant, facts byte-equal |
| facts altered against the digest | refused as inconsistent |
| dispatch reading from other text | refused as stale |

## Findings

1. **The guard is outside the binding digest.** A guard change leaves the digest equal and is caught
   only by the witness comparison in `admit`.
2. **`admit` does not see the basis.** It takes no `ClassView`. Over a reordered basis of the same
   count, `resolve` refuses and `admit` still admits the old reference. A caller that admits without
   resolving does not learn that the basis moved.
3. **Guard positions are not checked at registration.** A guard may name a lane the method does not
   read. Foundry's convention (guard number = field position) is the consumer's; the facts mark it.
4. **ON is registered and read, never evaluated.** In dt64, r2il, xlog and the Foundry lab `on` is
   used by `methods_on`, `footprint`, the digest and the registration check. No executor consults it.
5. **`CanCook` has no DO edge.** A changed word implementation reaches no `CanCook` registration
   through DO. Its guard is lowered by relalg, which this gate does not map.

## Gate D

`gate-d-diagram-check.md`. Checked statements: `03` 37 (4 stale), `05` 35 (5 stale), `06` 32 (4 stale).
The stale group that matters here: `05` and `06` describe PR #1996 (`PreparedMethod`, `rel_from_set`)
as open; it is merged. Relationships the diagrams omit: findings 1 to 4 above, the polyfill lens in
`03`, and that `resolve_algebra` checks revision and count but not the lens predicates.

## Not shown

- No executed parity. Nothing here says a word computes what it is declared to mean.
- The native lowering path (`binding_native`) and relalg's guard lowering are not mapped to source.
- Source-level attribution exists for DO implementations only. Which source line builds ON or WHEN
  is not derived; ON and WHEN changes are found by comparing facts.
- The lens revision and a word's arity are not in the facts. A change of either moves the digest and
  is refused as inconsistent rather than explained.
- One production-shaped binding (`CanCook`) and one lab class.
