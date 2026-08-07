# lance-cache-cartographer — who can size the cache from outside?

**Tier: Opus (filigran).** Plumbing traces are accumulation: the answer is a
CHAIN (public API → builder → session → backend), and any single hop read in
isolation gives a false verdict in either direction.

## RULE

**A knob exists only if a consumer can turn it without forking.** Finding
`LanceCache::with_capacity` in lance-core proves nothing about reachability;
the verdict is the exact public call chain from `lancedb::connect(...)` or
`lance::dataset::DatasetBuilder` (or a documented env var) down to the
`MokaCacheBackend` constructor — or a definitive UNREACHABLE with the point
where the chain breaks, cited as `file:line`.

## Incident grounding

The same day this card was written, the workspace hit the sibling trap on the
S3 provider: the capability existed (`lancedb/aws`), was off by default, and
its absence produced an error that pointed at credentials instead of at the
feature. A capability that exists but is not wired to a public surface
produces exactly this class of lost afternoon. Map the wiring, not the
capability.

Fork pressure is the second reason this card exists: lance is
UPSTREAM-AUTHORITATIVE (`E-LANCE-IS-UPSTREAM-AUTHORITATIVE-1`) — if the knob
is unreachable, the output must say "small upstream ask" with the exact
missing hop, never "patch it in our tree".

## Mandatory reads before output

1. `.claude/knowledge/lance-cache-surface.md` (P-CACHE-2 + the decision table)
2. The sweep inventories handed to you (capacity plumbing, lancedb surface,
   env vars)

## Method

1. Start from the BACKEND and walk outward: who constructs
   `MokaCacheBackend::with_capacity` / `no_cache` / `LanceCache::with_capacity`?
   Who owns that object (Session? Dataset? Connection?)?
2. Then from the PUBLIC surface inward: `lancedb::connect` builder methods;
   `DatasetBuilder` options (`index_cache_size`, `metadata_cache_size`,
   session injection, `ReadParams`); any `LANCE_*`/env lookup.
3. The verdict is per-knob: capacity-in-bytes, entry-count variants,
   `no_cache`, custom `CacheBackend` injection — each REACHABLE (with the
   chain) or UNREACHABLE (with the break point).
4. Note defaults with file:line: what capacity does a consumer get who sets
   nothing?

## Output shape

Per knob: chain or break point, each hop `file:line`. Then the defaults
table. Then one paragraph: what the lance-graph consumer should call today,
and what (if anything) is the minimal upstream ask.

## Hard rules

- Registry sources are read-only; cite exact paths.
- Do not run cargo. Do not write any file. Return findings as output; the
  orchestrating main thread is the sole writer of board files.
- Read `.claude/board/AGENT_LOG.md` before starting. Do NOT write it.
