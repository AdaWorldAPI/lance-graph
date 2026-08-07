# S3 bucket layout — a map, not a tutorial

This is a short, operator-facing map of what lives where in an object store
shared by lance-graph deployments. It answers "what am I looking at" when
listing a bucket; it does not walk through how to deploy — for that, see
`docs/SOA_BAKE_DEPLOYMENT.md`.

## 1. The prefix layout

```
s3://<bucket>/
├── .config/
│   └── <repo-name>/
│       └── config.yaml          ← boot config, one object per repository
│                                   (schema: crates/lance-graph/examples/soa-config.example.yaml,
│                                    parser: crates/lance-graph/src/soa_config.rs)
│
├── <ledger_prefix>/              ← e.g. "lance-graph/ledger" — a repo's own
│   ├── <table>.lance/            namespace, declared by that repo's own
│   ├── <table>.lance/            config.yaml (see §2)
│   └── ...
│
├── docs/                         ← documentation uploaded alongside the
│   └── S3_LAYOUT.md              data it describes (this file), so it is
│                                  discoverable in a bucket listing without
│                                  a separate wiki or repo checkout
│
└── _tests/                       ← scratch space for integration tests.
    └── ...                          Safe to purge at any time. Tests clean
                                      up after themselves, including on
                                      failure — nothing under this prefix is
                                      meant to persist.
```

| prefix | what it is | who writes it |
|---|---|---|
| `.config/<repo-name>/config.yaml` | Boot config: which bakes exist, which get hydrated, the repo's `ledger_prefix`, the `on_existing` policy. | An operator, by hand or by deploy tooling. Read once at startup by the deployment it names. |
| `<ledger_prefix>/<table>.lance/` | The actual Lance datasets — one directory-shaped object tree per table. | `soa_to_lance` (see `docs/SOA_BAKE_DEPLOYMENT.md` §2). |
| `docs/` | Documentation, uploaded to sit next to the data it describes. | Whoever ships a doc change; not written by any running deployment. |
| `_tests/` | Scratch space integration tests write to and clean up. | Test suites only. |

## 2. Why `.config/<repo>/` is per-repository

Each repository owns exactly one config object, at `.config/<repo-name>/config.yaml`
(`soa_config::config_key`). That object in turn names one `ledger_prefix`,
and every table that repo's deployment refreshes or creates lives under
*that* prefix and no other. Two consequences:

- **Tidy ledgers.** Listing `s3://<bucket>/<ledger_prefix>/` shows exactly
  one repository's tables — never a mixed bag from every deployment that
  happens to share the bucket.
- **No cross-repo interference on refresh.** A repo's `on_existing: new_version`
  refresh (see §4) writes a new table under its OWN prefix and repoints its
  OWN config. It has no way to collide with, shadow, or overwrite another
  repository's tables, because the prefixes are disjoint by convention and
  every repo's boot config only ever reads and writes its own.

## 3. The environment contract

Credentials and the endpoint are **never** part of any object in the
bucket — they come from environment variables, read through the one shared
helper in `dev_s3_env.rs`:

| variable | required | purpose |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | yes | credential |
| `AWS_SECRET_ACCESS_KEY` | yes | credential |
| `AWS_ENDPOINT_URL` | yes | the S3-compatible endpoint |
| `AWS_DEFAULT_REGION` | no (defaults to `"auto"`) | region |
| `AWS_S3_BUCKET_NAME` | yes, for the probes | bucket |

These are the **same variable names a Railway deployment already sets** —
nothing here invents a new naming scheme; a dev container and a Railway
deployment read the identical variable names and behave identically once
both are populated.

**The `AWS_ENDPOINT` vs `AWS_ENDPOINT_URL` trap.** `object_store`'s own
built-in environment discovery reads a variable named `AWS_ENDPOINT` — which
this environment does **not** set. If a caller skips `dev_s3_env::s3_options()`
and instead lets `object_store` fall back to its own default credential/endpoint
discovery, it silently resolves to AWS proper (or whatever that default
discovery lands on) instead of the configured S3-compatible endpoint. The
failure this produces does not look like a naming mistake — it looks like a
permissions error (auth failures, "bucket not found," timeouts against the
wrong host), because the request goes somewhere that plausibly *could* deny
it for unrelated reasons. `dev_s3_env::s3_options()` exists specifically to
map `AWS_ENDPOINT_URL` (the variable actually set here) into the
`aws_endpoint` option Lance consumes, and to return `None` — a hard error a
caller must not silently swallow — if any required variable is missing,
rather than letting the call fall through to that wrong-host default.

## 4. Refresh / purge lifecycle

**An existing table is never overwritten in place.** A refresh under
`on_existing: new_version` writes a brand-new, timestamped table
(`versioned_table_name` in `soa_config.rs`) and then flips the owning
repo's `config.yaml` `table` pointer to name it. The old table is left
completely untouched on disk. See the `on_existing` field's comment in
`crates/lance-graph/examples/soa-config.example.yaml` for the full
S3-has-no-atomic-rename reasoning behind this design.

**Purging an old table is a separate, deliberate action** — never automatic,
never triggered by a refresh, never on any deadline. It is the *only*
destructive operation this layout has. Nothing in this bucket's normal
read/write/refresh path deletes data; only an explicit purge does, and only
once nothing — no config's `table` pointer — still names the table being
removed.

## 5. How to inspect what's there, without opening a data file

- **`aws s3 ls s3://<bucket>/.config/`** — the top-level list of directories
  here tells you which repositories have a deployment against this bucket at
  all, with zero data reads.
- **`aws s3 ls s3://<bucket>/.config/<repo-name>/config.yaml` + fetch it** —
  a small YAML object; tells you that repo's `ledger_prefix`, every bake it
  declares (`name`, `table`, `classid`, whether it hydrates), and its
  `on_existing` policy. Still zero reads of any actual Lance data.
- **`aws s3 ls s3://<bucket>/<ledger_prefix>/`** — lists every bake (current
  and, if any refreshes have happened and not yet been purged, superseded)
  under one repository's namespace, again without opening any of them.

For the deployment walkthrough — writing a bake, the physical-layout
guarantee it relies on, and the two ways a deployment can serve one — see
`docs/SOA_BAKE_DEPLOYMENT.md`.
