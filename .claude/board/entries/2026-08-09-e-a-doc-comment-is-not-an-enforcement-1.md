## E-A-DOC-COMMENT-IS-NOT-AN-ENFORCEMENT-1 (2026-08-09)

**Status:** FINDING. **Confidence:** high.

`FleetRecovery::foreign_min_cycle` shipped with the correct rule written
in its doc comment: *"the caller must NEVER raise its durable
`after_cycle` bound to or past this cycle."* A rule stated in prose to a
caller who holds a plain `Option<CycleId>` is an instruction, not a
guard — the unsafe checkpoint stays one obvious line away.

`FleetRecovery::checkpoint_bound(recovered_through)` is the same rule as
an API: it returns the bound the caller MAY store, capped strictly below
any foreign landing. The caller can still ignore it, but the safe path is
now the shortest one.

Companion: the same session made the writer's single-owner claim
enforceable rather than narrated (lexical `store_identity` so
`x/./s.lance` cannot claim a second slot beside `x/s.lance`; an RAII
`WriterClaim` taken before the first `.await`, so a cancelled `open`
cannot leak a reservation). Both are the same move — *carry the
invariant in a value, not in a sentence.*

