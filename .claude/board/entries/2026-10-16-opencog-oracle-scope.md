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

## Licence check: does an adapter to OpenCog code poison ours? (read 2026-10-16)

Read: `opencog/atomspace` `LICENSE` at `master` (693 lines), the header and the
"Linking Exception" at lines 672-693; `opencog/cogserver` `LICENSE` has the same
header. Not read: PLN, attention (ECAN), the miner, and per-file headers, which
the licence says can override it ("Unless indicated otherwise (a license
reference in file headers, or a LICENSE file in a directory)").

The AtomSpace is AGPL v3 **with a linking exception** of the GPL-classpath kind:
"permission to link this library with independent modules to produce an
executable, regardless of the license terms of these independent modules", where
"An independent module is a module which is not derived from or based on this
library."

| shape | effect on our licence |
|---|---|
| separate process (CogServer, socket, pipe) | none; no linking happens |
| our adapter linking the unmodified library, written against its API | none under the exception; the AtomSpace part stays AGPL and its source must be offered when we distribute it |
| we patch the AtomSpace (e.g. for speed) | our code stays ours; the patched library is AGPL and the exception is optional on it; AGPL §13 applies if users reach it over a network |
| we port or transcode AtomSpace code into Rust | **this is the risk**: a port is "derived from or based on this library", so it is AGPL, and so is any crate it lands in |

So linking an adapter does not poison the licence; copying the code does. A
harvest for validation must keep OpenCog code and our code apart: OpenCog runs as
the oracle, and our side is written from its documented behaviour, never from its
source. This is not legal advice; check with counsel before distributing anything
that links OpenCog.

OPEN: where a 200× speedup would come from has not been identified or measured.
The MeTTa-side engines measured so far are Hyperon (MIT) and PeTTa (MIT); the
local MORK checkout (`ed57c67`) has no licence file and no licence field, so its
licence is unknown.
