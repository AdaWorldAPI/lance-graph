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

### Oracle and calibration use (same reading)

Running unmodified AGPL software and comparing its answers with ours does not
license our code under AGPL; the AGPL grants running it without conditions. Two
things still need care:

- **Fixtures:** OpenCog's own input files (rule libraries, test scripts) stay
  AGPL wherever they are copied. Fetch them at a pinned commit outside the repo
  (the `bench/fetch.sh` pattern in tesseract-rs) instead of committing them. Our
  own generated inputs and the recorded answers (truth values, binding sets,
  counts) are results, not OpenCog code, and can be committed.
- **Calibration:** matching our constants or truth formulas to OpenCog's answers
  is fine. Taking a formula from the PLN papers or documentation is fine.
  Transcribing it from their source code is the porting case above.

### An OpenCog fork in its own repository that imports our crates

Case: fork OpenCog (AtomSpace) in a separate repository and change it to use
ndarray, Datalog over folds, or a storage adapter backed by `lance-graph-quack`.

- **Our crates keep their licences.** Dependencies flow OpenCog → ndarray /
  quack, never back. ndarray is MIT or Apache-2.0, lance-graph is Apache-2.0
  (repository `LICENSE`; quack and mask-risc carry no own field), relalg is
  MIT OR Apache-2.0. Both are compatible with AGPL v3, so a combined AGPL binary
  may include them without changing them.
- **The fork is AGPL**, including every line we write inside it (the storage
  backend, glue, patches), because that code is based on OpenCog. If the fork is
  distributed, or modified and offered to users over a network (§13), its source
  must be published; a public fork satisfies that.
- **The trap is reuse in the other direction.** Glue written in the fork cannot
  later move into lance-graph without bringing AGPL with it. So general logic is
  written in our repositories first, under our licence, and the fork only calls
  it; the fork keeps the thinnest possible adapter.
- The combined binary is distributed under AGPL terms; our crates inside it are
  still usable on their own under their own licences.

OPEN: whether glue we author ourselves could be dual-licensed is a copyright
question for counsel, not settled here.

### DECISION (same day): `lance-graph-cog` is our API for OpenCog to call

DECISION: improving OpenCog for its community is welcome. `lance-graph-cog` is the
crate OpenCog would use: a lance-graph crate under lance-graph's own licence
(Apache-2.0) that exposes the storage, fold-execution and codebook API. The
AGPL side is a separate OpenCog fork that depends on it and keeps only the thin
adapter (for example an AtomSpace storage backend).

SCOPE: `lance-graph-cog` contains no OpenCog code. It is written from OpenCog's
documented API and observed behaviour, never from its source, so it stays
reusable by any caller. This refines the D-MTP-0 naming note
(`2026-10-14-metta-polyfill-map.md`): "cog" there meant pulling OpenCog itself
into the crate, which stays ruled out; here it names the API OpenCog consumes.

REVISIT WHEN: the oracle validation shows which surface OpenCog actually needs;
until then the crate is not created.

**Precedent:** `lance-graph-ontology` (Apache-2.0) already has this shape. It
hydrates other people's formats into our registry and Lance cache without
containing their code: OGIT TTL (`oxttl`), OWL and FIBO RDF/XML (`oxrdfxml`), XSD
schemas (`quick-xml`), plus a Wikidata path, with `soa_bake` and `lance_cache`
as the fast side. Tenants then reach it through thin `NamespaceBridge` views.
`lance-graph-cog` would do the same for OpenCog: read AtomSpace content (Atomese
atoms) as input data, and give the AGPL fork's adapter the same bake-and-cache
surface.
