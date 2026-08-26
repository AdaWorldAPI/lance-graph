## E-A-RECONCILED-HEAD-IS-NOT-A-PUBLICATION-1 (2026-08-09)

**Status:** FINDING. **Confidence:** high — falsified both ways.

Renaming a field is not the same as removing the confusion it names.
`CommitOutcome::Reconciled.current_head` was deliberately NOT called
`version` (#912) precisely because it is the store head AT RECONCILIATION
TIME. One layer up, `seal_cycle` then adopted it as `SealedCycle.version`
anyway — so retrying cycle 1 while the head stood at V5 recorded cycle 1
as "sealed into V5". The careful name survived; the meaning did not.

The repair is structural, not documentary: `SealedCycle` now carries
`publication_version: Option<DatasetVersion>` (`Some` ONLY for a fresh
`Committed`) beside `observed_head: Option<DatasetVersion>` (`Some` only
when a sink actually observed one — `None` on `NoChange`, which calls no
sink and whose `head` is the caller's asserted base). A publication
position that was never observed stays **unknown**; the durable identity
is `(cycle, batch_hash)` and the position is an audit-path read.

**The general shape:** when a type distinguishes two things a consumer
will conflate, the distinction has to be *unrepresentable* at the
consumer, not merely *documented* at the producer. A doc comment warning
"this is not X" is evidence that the next layer will use it as X.

Falsifier: `a_reconciled_retry_never_reports_the_current_head_as_publication`
(cycle 1 → V1, cycle 2 → V2, retry cycle 1 → `Reconciled`, observed head
V2, publication `None`, zero extra appends).

