# Post-merge recovery for #1601–#1617 (2026-10-19)

**Status:** RECONCILED (remediation PR from main `47697c77`). The 17 PRs merged with 17 unresolved review threads and no board coverage. They are not reopened; one PR fixes what exists.

## Review-thread reconciliation

| PR | threads | finding | disposition | fixed in |
|---|---|---|---|---|
| #1603 | 1 | relative-pronoun case taken from the antecedent | valid | docs commit: case comes from inside the clause, gender/number agree |
| #1604 | 2 | no rejected outcome; premise deletion said "rederive from scratch" | valid | docs commit: `Rejected` (witnessed); C-09 admits DRed |
| #1605 | 1 | prerequisite named Lance 7 / LanceDB 0.30 | valid | docs commit: Lance 13.* / LanceDB 0.40.*, reuse H14c-5 |
| #1607 | 2 | `assert` stripped under `-O`; musical fields checked only by type | valid | qualia commit: explicit checks, pinned field digest, `-O` negative controls, CI |
| #1608 | 1 | `assert` stripped under `-O` | valid | qualia commit: explicit checks; sidecar compared to the archived values |
| #1612 | 2 | D-SRAM-4 targets the banned d′ driver; negative control would fail a correct optimizer | valid | docs commit: D-SRAM-4 BLOCKED; gate split |
| #1613 | 2 | `jirak` is not a calibration API; total-input row limit missing | valid | docs commit |
| #1614 | 2 | EIG nonidentifiability; underflow vs zero support | outdated: already fixed in the merged text (plan §148; SB-02a/b, SB-24b/c) | none needed |
| #1615 | 1 | candidates not bound to the evaluated snapshot | valid | act-r commit: `Refusal::StaleCandidate`, 10 tests, disable-verified |
| #1617 | 2 | tile key lacks map generation; palette lacks its own generation | valid | d-rmap commit: §6a generations and falsifiers |
| #1616 | 0 | (no review) gather-shaped hop | architectural | mask-risc commit: quarantine; `ISS-GATHER-HOP-1616` |

PRs #1601, #1602, #1606, #1609, #1610, #1611 had no review findings.

## Open

- GATHER-DECISION (which reads are admitted) is still open; the quarantine does not decide it.
- The ACT-R `Snapshot` is a local type; binding it to canonical contract types is future work.
- D-RMAP amortization against direct sorted-index traversal is unmeasured.
