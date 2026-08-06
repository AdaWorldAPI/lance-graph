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
| **This repo's own object-store path goes through `lance` (default features, `aws` ON), not through `lancedb`** — so §3's `lancedb` gate is a *consumer-side* trap, not this crate's | **FINDING** — probe run, recorded in §3a | Raised by review on PR #901 and verified against the manifests + call sites; the probe command, its output and the promotion decision are in §3a. **This corrects the first draft of §3**, which stated the `lancedb` gate as if it were the gate on this repo's `s3://` reads. |

Nothing below is promoted past its row here.

**Probe record for the two manifest rows above** (re-run any time; all three are
read-only and take seconds):

```bash
# P1 — the feature declarations, read from the vendored manifests:
sed -n '/^\[features\]/,/^\[[a-z]/p' ~/.cargo/registry/src/*/lancedb-*/Cargo.toml
sed -n '/^\[features\]/,/^\[[a-z]/p' ~/.cargo/registry/src/*/lance-io-*/Cargo.toml
# P2 — which crate THIS repo opens datasets with, and how it is configured:
grep -nE '^(lance|lancedb) *=' crates/lance-graph/Cargo.toml
```

**Result (2026-08-06):** P1 — `lancedb` `default = []`, `aws = [...]`; `lance-io`
`default = ["aws", "azure", "gcp"]`. P2 — `lance` is a **direct, non-optional**
dependency taken **with default features**, and `lance`'s own
`default` includes `aws`; `lancedb` is `optional = true, default-features =
false` behind a separate feature. **Promotion decision:** the manifest rows stay
FINDING; the *inference* drawn from them in the first draft of §3 is **corrected**
by §3a rather than promoted.

## 1. The three layers

| layer | its ONE job | if it is absent |
|---|---|---|
| **object store** (S3-compatible) | **hydration source** — durable, versioned, shared between machines and between builds | fall back to whatever secondary source the consumer already has; the store still works, the dataset just has to come from somewhere else |
| **local directory** | **THE Lance store** — the path the process opens; zero-copy mmap reads, page cache, no network in the read path | **no fallback — always required.** Any path on a **supported, mmap-capable local filesystem** satisfies it (see the qualification below) |
| **persistent volume** | decides **which** local directory — chosen only because it survives redeploys | hydrate on every boot; still correct, merely slower |

Read the third row twice. The volume is an **optimization on hydration
frequency**, not a component of the store. A design that says "we need a volume
or this doesn't work" has mis-assigned a job: what it needs is a directory.

**The qualification on "any local path" (raised by review, PR #901).** What the
store needs is not merely *a path that is not a URI* — it is a filesystem that
actually delivers the mmap and locking semantics the zero-copy read depends on. A
network filesystem (NFS/EFS-class), a FUSE mount, or an overlay with unusual
caching presents a perfectly ordinary local-looking path while changing page-cache
behaviour, consistency, and lock semantics underneath it. Those are the cases
where "it is a local directory, therefore reads are zero-copy" stops being true.

So the requirement is **a supported, mmap-capable local filesystem**, and the two
axes stay separate:

- **correctness** — mmap-capable filesystem. Not negotiable, and not satisfied by
  path *shape*.
- **hydration frequency** — durability/persistence. Purely an optimization, as
  the third row says.

An ephemeral container path on an ordinary local filesystem satisfies the first
and not the second, which is exactly the intended trade. A network mount may
satisfy the second and *not* the first, which is the trap — and it is the same
trap as §2 one level down, since a network filesystem reintroduces the network
into the read path while still looking like a directory.

## 2. Why the object store must not be the store — even though the URI works

Lance opens an `s3://` URI natively — **given the object-store feature its
provider registration needs** (§3, and §3a for which crate's feature that is in
any given consumer). That is exactly what makes this trap easy to fall into: with
the feature on, the wrong architecture **runs**, correctly, and only degrades. The
feature being off produces a different, louder failure and is §3's subject; this
section is about the case where it works.

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

**Corollary — the local directory has no minimum *durability*.** An ephemeral
container path on an ordinary local filesystem is functionally correct: mmap
works, the page cache works, the lens works. Losing that directory on redeploy
costs a re-hydration, not a correctness property. This is why §1's third row is an
optimization and not a requirement. (It has no minimum durability; it does have a
minimum *filesystem* — see §1's qualification. "No minimum quality", as the first
draft put it, was too strong.)

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

### 3a. …but diagnose the crate that actually opens YOUR uri — this repo's is `lance`

**Correction, raised by review on PR #901 and verified (probe record in
§ Evidence status).** §3 above is true *about `lancedb`*, and the first draft
stated it as though it were the gate on this repository's object-store reads. It
is not.

| | crate | how this repo takes it | `aws` in effect? |
|---|---|---|---|
| what `VersionedGraph::{s3,azure,gcs}` reads through | **`lance`** | direct, **non-optional**, **default features** | **YES** — `lance`'s own `default` includes `aws` |
| the optional SDK surface | `lancedb` | `optional = true`, **`default-features = false`**, behind its own feature | **NO**, unless that feature turns it on |

So the mechanical rule in §3 stands, but its *first step changes*: **resolve
which crate opens the URI before you look at any manifest.** The §3 story — "the
mental model is correct about the wrong crate" — is exactly the trap this
subsection exists to stop this document from itself falling into, one layer
further out.

Restated so it is checkable rather than remembered:

1. Find the call that opens the URI, and name the crate it belongs to.
2. Read **that** crate's feature declarations, and how *this* manifest takes it
   (a `default-features = false` on the dependency line overrides the upstream
   default, and is easy to miss).
3. Only then decide whether a scheme-named error is a build problem here.

A consumer that opens datasets through `lancedb` is squarely in §3's case. A
consumer that opens them through `lance` with default features is not — and for
that consumer, a scheme-named error means something else and the §3 diagnosis
would send it down the wrong path.

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
| absent → hydrated | one large sequential read (§5: sustained, amortized) + one connect | safe to repeat **within the boundary below** — not unconditionally |
| hydrated → dirty | a local write; free | — |
| dirty → hydrated | **push back** — the expensive direction (§5: writes are ~½ read throughput and pay per-fragment object overhead) | must complete before flush, or the divergence is lost |
| hydrated → flushed | a local delete; frees disk | **only legal from `hydrated`, never from `dirty`** |
| flushed → hydrated | same as absent → hydrated | — |

**The one rule that matters:** *flush is legal only from `hydrated`, never from
`dirty`.* The state machine exists to make that a checkable condition rather
than an assumption. A `dirty → flushed` edge is data loss with no error.

### 4a. The idempotency boundary — `absent → hydrated` is NOT unconditionally safe to repeat

**Correction, raised by review on PR #901.** The first draft's "safe to repeat, it
is idempotent" was too strong, and the strength was load-bearing: the eviction
plan leans on that word to argue a lost race costs only a wasted rehydration. A
Lance dataset is a **multi-file directory**, so:

- a hydration that **fails part-way** leaves a partial directory, and a retry
  that treats it as a destination rather than as debris merges two attempts;
- a hydration against a **different source version** than a previous one mixes
  files from two snapshots into one directory — each file individually valid, the
  directory as a whole not a dataset that ever existed;
- a **concurrent** reclaim (§4's `hydrated → flushed`) deleting files from that
  same directory can remove what a hydration just wrote, or expose a reader to a
  directory that is neither complete nor absent.

None of these is prevented by the transfer being repeatable. So the property is
**conditional**, and the conditions are the contract:

> `absent → hydrated` is idempotent **given (a) a pinned source version and (b) a
> destination that is empty and not concurrently mutated.** Outside those two
> conditions it is not idempotent, it is a merge.

**The mechanism that makes both conditions hold — hydrate aside, publish by
rename.** Fetch into a private temporary directory, then make it visible with a
single atomic directory rename; retire by renaming *away* first and deleting the
renamed copy afterwards. A reader therefore only ever resolves a name that is
either absent or a complete dataset, never one mid-assembly or mid-removal. A
failed hydration leaves only an unpublished temporary directory, which is debris a
sweep can delete without consulting anything.

This is a **filesystem-atomicity boundary, not a coordination protocol** — it adds
nothing to the read path, takes no lock, and holds no lease. That distinction
matters because the eviction plan explicitly rejects a lease/refcount protocol;
this requirement is compatible with that rejection, and is what makes its "worst
case is a wasted rehydration" claim actually true. See
`.claude/plans/idle-flush-dataset-eviction-v1.md` §5a.

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

### 5a. "Boot-viable" is a claim about a SIZE, and the size is in the table

**Correction, raised by review on PR #901.** "VIABLE for hydration" above is a
verdict about the *shape* of the access (few, large, sequential). It is **not** a
verdict about any dataset size, and the row it was measured on is in the
tens-of-megabytes range. Carried forward naively it becomes a boot-viability claim
for arbitrary datasets, which the same numbers refute:

| dataset size | implied transfer at the observed sequential rate | plus connect | boot-viable? |
|---|---|---|---|
| the measured ~35 MB | ~1.7 s | + ~0.7 s | yes — matches the observed ~1.4 s |
| ~256 MB | ~12 s | + ~0.7 s | depends entirely on the boot budget |
| ~1 GiB | **~49 s** | + ~0.7 s | **no**, against any ordinary readiness deadline |

The linear term dominates the moment the round trip stops being the cost, which is
almost immediately. So the honest statement is: **hydration is the right *shape*
at any size; whether it fits a boot budget is a size question that must be
answered against the actual dataset and the actual budget**, and only the
tens-of-megabytes case has been measured here.

**What is NOT stated**, and should not be inferred: no RAM or NVMe baseline was
measured in this workspace — the "~2.5 million×" and "~2500×" ratios use
conventional figures for those tiers, not measurements taken here, and they are
order-of-magnitude arguments rather than benchmarks. Neither is the measurement
method recorded (single run vs. best-of-N, cold vs. warm client). Treat the whole
of §5 as one observation with a shape, not as a performance model.

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
- **Hydrate aside and publish by rename** (§4a). "Repeatable transfer" is not
  idempotence over a multi-file directory.

### 6a. Scope — what this doctrine binds, and the shipped API it does NOT invalidate

**Correction, raised by review on PR #901.** The first bullet above was written
categorically, and read that way it declares an existing, tested, public API
architecturally invalid while offering no replacement. That is not what it means,
and the scope belongs in the document rather than in the reader's judgement.

`crates/lance-graph/src/graph/versioned.rs` ships `VersionedGraph::{s3, azure,
gcs}`. Each stores a network URI as `base_path` and the read methods pass it
straight through, so those constructors *are* the pattern §6's first bullet warns
about. They are **not deprecated by this document**, and nothing here removes
them.

**What the rule binds:** the **hot zero-copy substrate** — any read path whose
correctness story includes mapped bytes, a lens over them, or a page-cache
assumption. There, opening a network-scheme URI does not degrade the guarantee, it
**voids** it, and the finding is structural rather than about speed.

**What the rule does not bind:** occasional, non-hot access where no zero-copy
claim is being made — administrative reads, one-off inspection, a version listing,
a small metadata query. The remote constructors remain the correct tool for those,
and calling one is not a violation.

**Where that leaves the constructors:** they are **usable and unmigrated**, which
is a known state rather than a silent one. The missing piece is a hydrating
counterpart (`hydrate_from(remote) → local`) so a caller that *does* have a
zero-copy story has somewhere to go; that does not exist yet and this
documentation-only change does not add it. Tracked as
`.claude/board/ISSUES.md` `ISS-REMOTE-URI-CONSTRUCTORS-PREDATE-THE-HYDRATION-DOCTRINE`.

Until it exists, the honest instruction to a caller is: **choose by read shape,
not by constructor availability.** If your reads are hot and zero-copy, hydrate to
a local path yourself and use `local()`. If they are occasional and you are
claiming nothing about mapped bytes, the remote constructor is fine.

## Cross-refs

`.claude/knowledge/zero-copy-lens-law.md` (the law this doc is the storage-siting
corollary of) · `.claude/knowledge/ephemeral-warm-cold-lifecycle.md` (the
reasoning-tier ladder — orthogonal; do not conflate its cold tier with this
doc's flushed state) · `docs/DATAFUSION-PERIMETER.md` §11 (feature-closure half)
· ADR-022/023 (the Firewall — no serialization in the hot path; a remote read
path is that violation arriving through the storage layer).
