# 2026-10-16 — OpenCog is a temporary validation oracle, not a dependency

**Status:** DECISION (docs only; nothing built, nothing linked).

DECISION: OpenCog (Classic AtomSpace, PLN, ECAN, the pattern miner) is used only
as a temporary oracle, to validate that our MeTTa reading agrees with OpenCog's
on the same inputs. It is harvested for that check and not wired into any
lance-graph crate.

SCOPE: `lance-graph-metta` stays the MeTTa front end and `Space` adapter
(D-MTP-0, `2026-10-14-metta-polyfill-map.md`). Oracle harnesses that run OpenCog
code live outside the workspace, like `datalog-oracle-probe` and `hyperon-probe`.

BASIS: Classic AtomSpace is AGPL, so its code cannot enter an Apache/MIT crate,
and its value here is only as a reference answer.

REVISIT WHEN: the validation is done. Then one of two outcomes is recorded:
OpenCog is redundant (MeTTa via Hyperon/PeTTa already covers it) and the harvest
is dropped, or it earns a separate `lance-graph-cog` crate under its own licence,
possibly in its own repository.

OPEN: which OpenCog pieces the validation needs (PLN rules, ECAN, AtomSpace
queries); none is pinned yet.
