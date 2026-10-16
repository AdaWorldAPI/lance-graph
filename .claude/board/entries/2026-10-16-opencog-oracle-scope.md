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

### Repositories (added 2026-10-16)

AdaWorldAPI now has forks of `opencog/atomspace`, `atomspace-cog`, `atomspace-rocks`,
`cogserver`, `lg-atomese` and `learn`, plus an empty `AdaWorldAPI/atomspace-lance`, the
landing space for the AGPL-side adapter. Read: the atomspace-rocks fork at `6577a5b`
(`RocksStorage.h`, the key-schema comment in `RocksIO.cc:78-148`); same licence header and
linking exception as the AtomSpace; per-file headers `AGPL-3.0-or-later`.

`atomspace-lance` is the analogue of `RocksStorageNode`: a `StorageNode` subclass (AGPL, it
is based on OpenCog) whose calls go down into `lance-graph-cog` (Apache-2.0). The StorageNode
API it must implement (`RocksStorage.h:142-171`): open/close, `getAtom`, `getLink`,
`fetchIncomingSet` / `fetchIncomingByType`, `storeAtom`, `removeAtom`, `storeValue` /
`updateValue` / `loadValue`, `loadType`, `loadAtomSpace` / `storeAtomSpace`,
`loadFrameDAG` / `storeFrameDAG` / `deleteFrame`, `barrier`.

How Rocks lays out the data, and the candidate mapping on our side (not built, not measured):

| Rocks key | meaning | our side |
|---|---|---|
| `n@satom`, `l@satom` → sid; `a@sid` → satom | atom s-expression ↔ 64-bit id | content → NodeGuid; names via the label codebook |
| `h@shash` → sid list | hash → ids | not needed if the address is content-derived |
| `i@sid:stype-sid` | incoming set (by type) | a hop: preimage over coordinate ranges |
| `k@sid:kid` → sval | values (truth value etc.) per key | lanes on the row |
| `d@`/`f@`/`o@` frames, `z` heights | Kripke frames (layered AtomSpaces), link heights | hypothesis layers / Lance versions |
| prefix scans (`n@(Concept`) | load all atoms of a type | class range over the classid prefix |

OPEN: the atom identity mapping (content → NodeGuid) is the first design item; RocksIO's
s-expression text is the identity Rocks uses, which we would replace with a deterministic
address.

### Positional tables and zipper merge on read work over any KV store

The layout we use — `tableA[index]`, `tableB[index]`: parallel tables addressed by the same
row index, written sparsely, joined by a zipper merge on read — does not need Lance. It
needs only that each table can be read in index order. A KV store with ordered keys
(RocksDB iterates keys lexically) provides that, so the same layout runs on Rocks: one
key prefix per table, keys ending in the row index, and a read that walks the prefixes in
step and merges them in one pass instead of one `Get` per atom per key.

Why this matters here:
- `atomspace-lance` and an improved `atomspace-rocks` can share one layout; Lance is one
  backend for it, not a precondition (the 2026-10-05 storage-portability decision:
  Quack owns the semantics, the backend supplies capabilities).
- Rocks already sorts by prefix (`k@sid:`, `i@sid:`); what changes is that `sid` becomes a
  row index and the per-atom string keys stop being the join.
- It is the same pattern as the alpha layers (sparse dirty writes, merge on read).

Not built, not measured: no timing exists for Rocks with this layout or for the current
`RocksStorageNode`. A first probe would load one AtomSpace both ways and compare reads
per element.
