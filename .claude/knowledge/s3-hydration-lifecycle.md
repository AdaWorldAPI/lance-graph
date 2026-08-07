# The S3 hydration lifecycle

> **READ BY:** any session that writes an importer, a hydrator, a bake driver,
> or a deploy that needs an ontology present; before adding any `hydrate_*`,
> any `Dataset::write` call site, or any code that reads a source artifact at
> runtime.
>
> **Scope:** mechanism only. This document names stages, types, and byte
> magnitudes. It does not name corpora, sources, or their terms — those are not
> a public-repo concern and never appear here.

## The one-way ratchet

Hydration runs in one direction, and each stage is strictly narrower and
cheaper than the one above it:

```
  source artifact          text, per-vocabulary format, 10^7–10^9 B
        │  parse
        ▼
  ContextBundle            typed, in memory, per-slot
        │  bake
        ▼
  node rows + triples      512 B/row, sorted by (classid, family, identity)
        │  calcify
        ▼
  LanceDB 64k tables       on object storage — THE artifact
        │  address
        ▼
  reads                    open a table, resolve a position
```

**The rule this document exists to state: nothing ever hydrates upward.** A
consumer addresses the calcified form. It does not parse, it does not bake, and
it does not reach for the source artifact — not to check something, not to fill
a gap, not "just this once". A gap in the calcified form is fixed by
re-calcifying, upstream, once, for everyone; it is never patched by a consumer
re-entering an earlier stage. The moment one consumer parses at runtime, the
source artifact becomes a live dependency of every deploy, and the ratchet is
gone.

## Why the *calcified* form is what sits in object storage

It is tempting to store the source artifact and hydrate on startup. That is the
wrong end of the chain, for reasons that compound:

- **Cost is paid once, not per consumer and not per deploy.** Parsing and
  baking are the expensive stages; a stored calcified form amortizes them over
  every reader that will ever exist.
- **Every reader gets the same bytes.** Two consumers that each parse the same
  source can disagree — different versions of a parser, a different day, a
  silently-updated artifact. Two consumers that open the same dataset cannot.
- **Addressing survives compression.** The key is never compressed; Lance may
  encode the value slab however it likes and a reader can still route, group,
  and skeleton-render from keys alone. A stored source artifact has no
  addresses at all until someone parses it.
- **A deploy needs no parser.** The runtime dependency is an object-store
  client, not a format library per vocabulary.

## Stage inventory — what exists, what is a seam

| Stage | Where it lives | State |
|---|---|---|
| parse → `ContextBundle` | `lance-graph-ontology::hydrators` (Pattern D: generic `OwlHydrator` + ~50 LOC glue per vocabulary) | shipped, many vocabularies |
| bake → node rows + triples | the harvest side, outside this repo | shipped |
| calcify rows → LanceDB | `lance_graph::graph::ontology_hydrate` | shipped |
| calcify triples → relation matrices | — | **seam** |
| `ContextBundle` → node rows | — | **seam** |
| address the tables | `ontology_hydrate::NodeTable` | shipped |

**Two of the arrows above do not exist yet, and the gaps are load-bearing.**

The first: `hydrators` produces a `ContextBundle` and stops. Nothing carries a
bundle into node rows, so the Pattern-D path currently ends in memory — it
parses, and then the result has nowhere to calcify to. The bake path reaches
storage; the hydrator path does not. These are two hydration pipelines that do
not meet, and a session that adds a `hydrate_*` today is extending the half
that has no floor under it.

The second: the bake emits an edge table alongside the rows, and only the rows
are calcified. The engine already answers reachability over relation matrices
(`TypedGraph::traverse`, `blasgraph::ops::hdr_bfs`), so the edges have a
destination — they are simply not carried to it. Until they are, every consumer
that needs an ancestor walk writes its own, over whatever slice it can get.

## Object-store addressing

`ontology_hydrate::write_database` and `NodeTable::open` take a database
directory as a string and hand it to Lance, which resolves it through
`object_store`. A local path and an object-store URI are therefore the same
call site — the writer has no S3 branch, and adding one would be the mistake.
The database directory is a *location*, and the only thing this layer knows
about it is that Lance can reach it.

One table is one dataset under that directory:

```
<db>/nodes-<classid:08x>-<family:04x>.lance
```

The name is fixed-width and zero-padded, so a lexical listing of the database
is an ordering by address — `ls` on the bucket prefix is a table index, with no
catalog to keep in step.

Table extent follows from the key, not from a tuning decision: `identity` is a
`u16`, so a `(classid, family)` pair addresses exactly 65 536 slots, and a full
table is 64 Ki × 512 B = 32 MiB. That is a deliberate size for object storage —
large enough that per-object overhead is negligible, small enough that a reader
fetching one basin does not pull a neighbouring one.

## Credentials

Storage credentials reach Lance through the process environment or an explicit
`storage_options` map at the call site. They are never committed, never written
into a path, and never baked into an artifact. A dataset URI in a config file
or a commit message must contain a bucket and a prefix and nothing else.

If a credential ever appears in a repository, a log, or a URI, treat it as
compromised and rotate it — scrubbing the text is not sufficient, because
history is retained.

## Re-calcification

Tables are written with `WriteMode::Create`. An import writes a table once; a
second import against a live database fails loudly rather than appending, since
appending would double a table's rows without any signal.

A changed source therefore means a **new** calcified form, not a mutated one.
Lance's own versioning is the mechanism for holding both while readers move
across; a re-import that overwrote in place would break every reader mid-read
and would destroy the property that two readers of one dataset see one truth.

## What is verified, and what is not

Verified by test in `ontology_hydrate`:

- the run partition cuts on `(classid, family)` and refuses unsorted input;
- a mis-strided buffer is refused rather than read shifted;
- the `RecordBatch` **points at the caller's allocation** — pointer identity is
  asserted against the source buffer, so the zero-copy write is measured, not
  claimed;
- the read-side `key`/`edges`/`value` ranges tile the row exactly once.

**Not verified:**

- **No object-store URI has been exercised.** Every test to date runs against
  in-memory buffers; `Dataset::write` to an `s3://`-style location is expected
  to work because Lance resolves the path through `object_store`, but expected
  is not measured. The first session to run it should say so here.
- **Zero-copy is verified up to the `RecordBatch`, not through the write.**
  Whether Lance's encoder streams that buffer or stages a copy of it before it
  reaches the object store has not been measured. The claim in this document is
  precisely "the batch borrows the caller's rows", and it should not be widened
  to "no copy occurs anywhere" until someone measures the writer.
- **Whole-database round-trip.** `write_database` then `NodeTable::open` over a
  multi-table buffer has no test; the pieces are tested, the composition is not.

## The failure this document is meant to prevent

A consumer that cannot find something in the database, and reaches back up the
chain to get it — parsing a source artifact at runtime, or shipping its own
copy of one. It always looks local and reasonable, and it costs the property
the whole chain exists for: that every reader sees the same bytes, and that the
expensive stages ran once.

If something is missing from the calcified form, the fix is upstream, and it is
someone's job. Say so; do not route around it.
