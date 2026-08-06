# S3 is the hydration path, never the store — the on-demand Lance dataset lifecycle

> **READ BY:** any session that opens a Lance dataset from a URI, wires an
> object-store backend, sizes a persistent volume, debugs a
> `"No object store provider found for scheme"` error, plans a rebake input, or
> proposes putting a dataset "in S3" as a runtime store. Also the
> `integration-lead` / `layer-boundary-warden` cards when a deployment topology
> question arrives.
>
> **Companions:** `.claude/knowledge/zero-copy-lens-law.md` (why a local
> filesystem is not a preference but the precondition — the lens needs mapped
> bytes, and a network fetch has none to lend) ·
> `.claude/knowledge/ephemeral-warm-cold-lifecycle.md` (the *reasoning* tier
> ladder; this doc is the *bytes-on-disk* ladder and does not touch it) ·
> `docs/DATAFUSION-PERIMETER.md` §11 (the feature-closure half of §3 below).

## The one-line statement

> **The object store hydrates; the local filesystem stores; the volume only
> decides whether hydration repeats.** Three layers, one job each. Collapsing
> any two of them is the failure this doc exists to prevent.

## Evidence status (per the workspace rule: label everything)

| claim | status | evidence |
|---|---|---|
| `lancedb` ships `default = []`; its `aws` feature forwards to `lance/aws` + `lance-io/aws` (+ `object_store/aws` directly in newer releases, transitively via `lance-io` in older ones) | **FINDING** — source-verified in this session | Read directly from the vendored `lancedb` manifests in the local registry, two releases apart; both show `default = []` and the same forwarding shape. |
| `lance-io` carries `aws` in its OWN defaults — so the opt-out is `lancedb`'s layer, not the stack's | **FINDING** — source-verified in this session | Same read: `lance-io`'s `[features] default` includes `aws`. |
| Without the feature, an `s3://` URI fails at provider registration — BEFORE any credential, endpoint or region is consulted | **FINDING** — reported measurement, not re-verified here | Reported by the session that hit it; the error text names the *scheme*, not a credential. Consistent with the manifest facts above (no provider is compiled in). Falsifier: build with the feature off and confirm the same URI fails identically with every env var unset AND with all of them set correctly. |
| Opening the object store as the runtime store makes every read a network fetch into a fresh buffer — no mmap, no page cache | **FINDING** (mechanism), *unbenchmarked here* | Structural: a remote range request has no mapped page to lend. Follows from the zero-copy law. **No A/B benchmark of remote-store vs local-store read paths has been run in this workspace.** |
| Any local directory satisfies zero-copy; a persistent volume is not a correctness requirement | **FINDING** (mechanism) | The store's requirement is a filesystem path, not a durable one. Persistence changes *how often you hydrate*, never *whether reads are zero-copy*. |
| The endpoint characteristics in §5 | **reported measurement, not re-verified in this session** | Measured once, against one S3-compatible endpoint, from one region, at one time. Provider- and region-dependent; treat the *ratios* as the finding and the absolute numbers as a single observation. |
| The flush/rehydrate lifecycle in §4 is the right shape for large single-use datasets | **CONJECTURE** | Argued from §5's ratios, not from a deployed instance. Falsifier stated inline at §4. **No probe has run.** |

Nothing below is promoted past its row here.

## 1. The three layers

| layer | its ONE job | if it is absent |
|---|---|---|
| **object store** (S3-compatible) | **hydration source** — durable, versioned, shared between machines and between builds | fall back to whatever secondary source the consumer already has; the store still works, the dataset just has to come from somewhere else |
| **local directory** | **THE Lance store** — the path the process opens; zero-copy mmap reads, page cache, no network in the read path | **no fallback — always required.** But *any* local path satisfies it |
| **persistent volume** | decides **which** local directory — chosen only because it survives redeploys | hydrate on every boot; still correct, merely slower |

Read the third row twice. The volume is an **optimization on hydration
frequency**, not a component of the store. A design that says "we need a volume
or this doesn't work" has mis-assigned a job: what it needs is a directory.

## 2. Why the object store must not be the store — even though the URI works

Lance opens an `s3://` URI natively. That is exactly what makes this trap
easy to fall into: the wrong architecture **runs**, correctly, and only
degrades.

The reason it is wrong is the same reason the zero-copy law exists one layer
down. A local dataset read is a mapped page — the kernel hands you bytes that
are already resident, and a lens over them costs a cast. A remote object read
is a range request that lands in a **freshly allocated buffer**: one copy per
read, minimum, plus a round trip, and no page cache to make the second read of
the same bytes free.

So the failure is not "S3 is slow." It is that **mounting the object store as
the runtime store deletes the mmap layer from the architecture** — every
downstream zero-copy guarantee is then a claim about buffers that were copied
into existence. The lens has nothing to borrow from.

> **The review question:** *where does the process open its dataset from?* If
> the answer is a URI with a network scheme, the zero-copy story below it is
> already void, regardless of what any type signature promises.

**Corollary — the local directory has no minimum quality.** An ephemeral
container path is functionally correct: mmap works, the page cache works, the
lens works. Losing that directory on redeploy costs a re-hydration, not a
correctness property. This is why §1's third row is an optimization and not a
requirement.

## 3. The feature gate that costs an hour if you don't know it

**`lancedb` ships `default = []`.** Its `aws` feature is what forwards to
`lance/aws` + `lance-io/aws` (+ `object_store/aws`), and *that* forwarding is
what registers the S3 provider.

Two consequences, both non-obvious:

1. **Without the feature, an `s3://` URI fails at provider lookup — before any
   credential, endpoint, or region is read.** The error names the *scheme*.
   That means **no amount of endpoint/region/credential/quoting debugging can
   possibly help**, because none of that code has been reached. Every minute
   spent on env vars is spent on a code path that does not exist in the binary.
2. **`lance-io` DOES carry `aws` in its own defaults.** So the intuition "the
   Lance stack supports S3 by default" is *true one layer down* and false at
   the layer you depend on. `lancedb` is the layer that opts out. That mismatch
   is the whole reason the diagnosis goes wrong: the mental model is correct
   about the wrong crate.

**The diagnostic rule, mechanical:** an object-store error that names a
**scheme** is a *build* problem (a feature is off). An object-store error that
names a **credential, host, bucket, region, or signature** is a *config*
problem. Never debug the second when you are looking at the first. Read the
error's noun before touching an env var.

*(Env var **names** — `AWS_ENDPOINT_URL`, `AWS_REGION`, and the standard
credential pair — are the config surface for the second class only. They are
inert against the first.)*

## 4. The lifecycle — four states, and what each transition costs

The actual operational ask: **large, single-use datasets** (rebake inputs,
one-off derivations, build artifacts) should not occupy local disk permanently.
They hydrate when needed, get pushed back if mutated, and their local copy is
reclaimed.

| state | what exists where | invariant |
|---|---|---|
| **absent** | object store only | reads are impossible; the store is not open |
| **hydrated** | object store + local dir, identical | reads are zero-copy; this is the only readable state |
| **dirty** | local dir has diverged (written/appended/compacted) | **the local copy is now the only truth** — flushing here destroys data |
| **flushed** | object store only, local reclaimed | ≡ *absent*, but reached deliberately after a push |

Transitions and their costs:

| transition | cost | gate |
|---|---|---|
| absent → hydrated | one large sequential read (§5: sustained, amortized) + one connect | none; safe to repeat, it is idempotent |
| hydrated → dirty | a local write; free | — |
| dirty → hydrated | **push back** — the expensive direction (§5: writes are ~½ read throughput and pay per-fragment object overhead) | must complete before flush, or the divergence is lost |
| hydrated → flushed | a local delete; frees disk | **only legal from `hydrated`, never from `dirty`** |
| flushed → hydrated | same as absent → hydrated | — |

**The one rule that matters:** *flush is legal only from `hydrated`, never from
`dirty`.* The state machine exists to make that a checkable condition rather
than an assumption. A `dirty → flushed` edge is data loss with no error.

**Why writes are an ops step and never a boot path:** the push-back direction
pays object-per-fragment overhead on top of raw throughput (§5), so its cost
scales with fragmentation as well as bytes. Hydration is boot-viable; the return
trip is not. Any design that puts a write-back on a startup path has put the
slowest, most fragmentation-sensitive operation in front of the readiness check.

*Falsifier for the CONJECTURE row:* run the full cycle on a representative
dataset and confirm (a) hydrate wall-clock stays inside the boot budget, (b) a
`dirty → flushed` attempt is refused rather than silently accepted, and (c) a
rehydrate after flush reads back byte-identically. Until that runs, §4 is a
design, not a result.

## 5. Measured characteristics of one S3-compatible endpoint

> **Single observation.** One provider, one region, one point in time,
> deliberately unnamed. **Provider- and region-dependent.** The *ratios* are
> what generalize; the absolute numbers do not. Reported by the session that
> measured them; **not re-run here.**

| operation | observed | what it settles |
|---|---|---|
| small-object round trip | **~250 ms** | |
| large sequential read | **~21 MiB/s** | |
| large sequential write | **~11 MiB/s** (≈ ½ the read rate) | |
| store connect | **~730 ms** | |
| cold re-open + full count, dataset in the tens-of-MB range (~69k rows, ~35 MB) | **~1.4 s** | boot-viable |
| write of that same dataset | **~7.7 s** (≈ 4.4 MiB/s effective — below the raw write rate, the gap being object-per-fragment overhead) | ops step, not a boot path |

**The round-trip number is the load-bearing one.** At ~250 ms per small object,
the object store is roughly **~2.5 million×** slower than RAM and **~2500×**
slower than NVMe. Those two ratios are the whole argument:

- **NOT viable as swap.** A page fault backed by a ~250 ms fetch is not a
  memory hierarchy; it is a hang with a progress bar.
- **NOT viable as a page-fault backing store**, for the same reason — and this
  is precisely §2 restated in numbers: mounting it as the runtime store puts
  that latency *under every read*.
- **VIABLE for hydration.** One large sequential transfer amortizes the round
  trip across the whole dataset; the effective cost is the ~21 MiB/s line plus
  one connect.
- **VIABLE for build caches** — same shape: large objects, few of them, latency
  amortized.

The rule that falls out: **the object store is fine when the object count is
small and the objects are large; it is unusable when the access count is large
and the accesses are small.** Every viable/non-viable verdict above is that one
sentence applied twice.

## 6. Consequences for new work

- **Never open a network-scheme URI as the runtime store.** Hydrate to a local
  path, open the local path. If a design opens the remote URI directly, the
  finding is not "this is slow" — it is that the zero-copy layer has been
  removed.
- **Do not require a persistent volume for correctness.** Require a *directory*.
  State the volume as an optimization with a named cost (one hydration per
  boot), so a deployment without one is a known trade rather than a bug report.
- **Gate the object-store feature explicitly, and say so where the URI is
  parsed.** A scheme-named error must lead the reader to the manifest, not to
  the credentials.
- **Never place a write-back on a startup path.** Push-back is an operational
  step with its own trigger.
- **Any flush path must assert `hydrated`, not assume it.** The `dirty → flushed`
  edge fails silently by construction; only an explicit check catches it.

## Cross-refs

`.claude/knowledge/zero-copy-lens-law.md` (the law this doc is the storage-siting
corollary of) · `.claude/knowledge/ephemeral-warm-cold-lifecycle.md` (the
reasoning-tier ladder — orthogonal; do not conflate its cold tier with this
doc's flushed state) · `docs/DATAFUSION-PERIMETER.md` §11 (feature-closure half)
· ADR-022/023 (the Firewall — no serialization in the hot path; a remote read
path is that violation arriving through the storage layer).
