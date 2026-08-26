## E-A-A-PERMANENT-FAULT-REPORTED-AS-RETRYABLE-IS-AN-INFINITE-LOOP-1 (2026-08-09)

**Status:** FINDING. **Confidence:** high.

A 511-byte artifact payload violates the writer's 512-byte ABI. It was
reported as `CommitError::Io` — the variant whose documented meaning is
"nothing published, safe to REGENERATE". A caller obeying that contract
regenerates the identical malformed batch and fails identically, forever:
the error classification, not the bug, is what makes it unbounded.

`CommitError::InvalidArtifact { row, len }` is permanent by construction.
The taxonomy rule this instance teaches: **an error variant's retry
semantics are part of its contract**, so a producer-side defect must never
borrow a transport-side variant merely because both mean "did not
commit".

Falsifier: `a_malformed_artifact_is_refused_permanently_not_as_retryable_io`
(refused twice, identically, store untouched).

### E-THE-ARTIFACT-WRITE-DECIDES-WHAT-KANBAN-PROGRESS-BECOMES-DURABLE-1

**FINDING (operator-ruled, implemented Phase A).** The persistence question was
being asked backwards. The old contract let the *cycle* decide when to write —
every 550 ms sweep sealed a `DatasetVersion`, so a thought crossing dozens of
cheap Rubicon steps in ~2 s minted dozens of versions and stretched a ~2 s
ladder across ~32-35 s of barriers. The correct rule inverts the initiative:

> **Kanban never decides when to persist. The semantic write that happens
> anyway decides which kanban progress gets a durable anchor.**

Thinking is cheaper than persisting its intermediate control flow. Rubicon
state, `Continue`, `Hold`, held work, the current rung and scheduler progress
stay **transient** — after a crash they are cheaply regenerated from the pinned
sealed inputs, which is exactly the property that makes discarding them safe.

**The mechanical form is smaller than the ruling sounds.** The gate already
existed in the data: a cast's payload. Non-empty = an artifact (grounding
result, reusable walk product, adjudication, conclusion) → persisted.
Empty = intent-only → ephemeral. `restage_held` had been casting empty payloads
since #879 (`cycle_driver.rs:303`), and #911's 512-byte ABI gate contradicted
it — the *fix* was not padding those casts to 512 bytes to satisfy the gate but
recognizing that **the empty payload IS the ephemerality mechanism**. Two
post-merge P1s dissolved at once: intent-only casts can never trip a payload
gate they never reach, and the empty-cycle version disappears because zero
artifact casts means the sink is never called at all.

**The general lesson.** When a gate and a producer contradict each other, the
question "which one do I bend?" often has a third answer: the contradiction is
the system telling you the two things were never in the same category. A
lifecycle transition and a semantic artifact are not the same kind of fact, and
only one of them belongs in durable storage.

### E-A-PUBLISHED-MANIFEST-IS-HISTORY-RECONCILIATION-NOT-ROLLBACK-1

**FINDING (measured against `lance-9.0.0/src/io/commit.rs:914-950`).** #911
tried to make an optimistic fence *effective* after the fact: on detecting that
a foreign writer had shifted the published version, it issued a compensating
`Dataset::delete` scoped to the cycle, then returned a retryable error. Review
found the flaw (the `(cycle, base_version)` predicate can delete a *successful
concurrent writer's* rows — the very race it exists to handle), but the deeper
error is categorical: **`Dataset::delete` creates another version. It is not
rollback.** There is no undo in an MVCC manifest chain.

The measured constraint underneath: **Lance 9 has no atomic expected-version
fence for `Append`** — the conflict rebase runs even on a single-attempt commit;
strict no-rebase mode exists only for `Overwrite`. A read-then-append is not a
compare-and-swap, and dressing it as one produces exactly the
committed-but-reported-failed hole that made the driver regenerate work that
had already landed.

The resolution is to stop trying to make the fence retroactive and make
**reconciliation authoritative** instead: commit the batch's identity
`(cycle, batch_hash)` *in the same commit as its rows*, look it up before
appending, and let a lost acknowledgement resolve by re-submitting the SAME
frozen batch. Same identity + same hash ⇒ `Reconciled` (success, no second
append). Same identity + different hash ⇒ fail closed. Genuinely unknown ⇒ an
`Ambiguous` state that says so, instead of an error that falsely promises
nothing landed.

**The rule this leaves behind:** never return an error meaning "nothing
landed" when failure could have occurred after publication — and never delete
history to restore an expected version number.

