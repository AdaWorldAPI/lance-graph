# Cargo compute substrate — one ndarray, one package identity, one binary

> READ BY: cargo-substrate-architect, integration-lead, simd-savant,
> kernel-membrane-warden, any session that adds, moves, pins, or feature-gates
> an `ndarray` dependency in ANY repo of this fleet, or that is about to write
> a `[patch]`, a `[workspace.dependencies]` entry, or a relative cross-repo
> path dep.
>
> Born 2026-09-20 from the W0B mask-ABI differential, which surfaced a
> domain-local Boolean algebra (`ogar-r2il::CallMask::and/or/xor/and_not/not`)
> sitting beside the one evaluator. Operator ruling the same day: *"ndarray ist
> das Silizium. Die Crates sind nur verschiedene Schaltungen darauf."*
>
> Scope: DEPENDENCY ARCHITECTURE ONLY. This doc never decides what `CallMask`,
> CE64, Moore, or R2IL MEAN. A session using it to redesign a carrier has left
> its scope.

## 0. The distinction the whole doc rests on

**Package/crate boundaries are not runtime boundaries.** Cargo may look like a
wide graph and still produce one statically linked executable:

```text
                        final-app
              /             |              \
           Quack          R2IL            Odoo
             |              |               |
         mask-risc      ogar-loco          OGAR
             \              |              /
              └────────── ndarray ────────┘
                             │
                             ▼
                 target/release/<one binary>
```

No plugins, no DLL layer, no IPC, no second process. Many crates at compile
time, one binary at runtime — hard compile-time modularity with **no runtime
architecture tax**. "Everything must be one binary" is therefore NOT an
argument for fewer crates; if anything it is an argument for more, each with a
harder contract.

## 1. THE LAW

```text
CARGO COMPUTE SUBSTRATE LAW

 1. ndarray is the mandatory compute substrate.
 2. Hot execution crates depend on ndarray directly or through the one
    canonical execution facade; no sibling SIMD/mask implementation.
 3. All repositories use ONE canonical ndarray source coordinate.
 4. Local development may [patch] that coordinate to a local checkout.
 5. Relative cross-repo ndarray paths are forbidden as durable dependencies.
 6. Features may select capabilities/backends, but may not remove ndarray
    from a crate whose contract is compute execution.
 7. Domain-local carriers are allowed:
       CallMask, AlphaMask, Moore128, ...
    Domain-local duplicate execution algebras are not.
 8. One final executable is expected:
       many Rust crates at compile time
       one statically linked binary at runtime.
 9. `cargo tree -d` must not show multiple ndarray package identities.
10. CI proves the rule. Documentation does not.
```

### What rule 1 does NOT say

It does **not** say every crate must import ndarray. A pure DTO / vocabulary /
contract crate (`lance-graph-contract`, `ogar-loco`, `ogar-vocab`) has no
compute contract and coupling it to the substrate would be artificial. The
binding form is:

> Every crate that EXECUTES masking, SIMD, fold, tile, vector, field, or
> numeric hot-path algebra must use the same canonical ndarray package
> identity. No parallel compute algebra beside it. The final product may link
> everything statically into one binary.

### Rule 7, stated as the carrier/algebra split

```text
CallMask.words              domain CARRIER          ALLOWED, permanently
CallMask::and/or/xor/...    domain-local ALGEBRA    transitional only
ndarray (via mask-risc)     execution AUTHORITY     the destination
```

The migration order is not negotiable and is not "delete the duplicate":

```text
differential parity FIRST  ->  then ownership  ->  then migration
```

A duplicate algebra that has not been proven bit-identical must not be
deleted, delegated, or "aligned" — the difference would be the finding.

## 2. MEASURED CURRENT STATE (2026-09-20)

Every number below came from a command, not from a manifest read by eye.

### 2.1 Six coordinate shapes across ten repos

```
grep -rhnE '^[[:space:]]*ndarray[[:space:]]*=' --include=Cargo.toml <repo>
```

| shape | seen in |
|---|---|
| `path = "../../../ndarray"` | lance-graph (majority), tesseract-rs, lance-graph-java |
| `path = "../ndarray"` | stockfish-rs, ladybug-rs, lance-graph root `[patch]` |
| `path = "../../ndarray"` / `"../../../../ndarray"` / `"../../../../../ndarray"` | q2, lance-graph-java |
| `git = ".../ndarray.git", branch = "master"` | lance-graph (`perturbation-sim`, `helix`) |
| `git = ".../ndarray", branch = "master"` | a2ui-rs, MedCare-rs |
| `workspace = true` | MedCare-rs, q2 |

Path depths run from `../ndarray` to `../../../../../ndarray` — **five
different depths**, so "the repos happen to sit next to each other" is
load-bearing in at least five repos at once. That is what rule 5 exists to
retire.

The `.git` suffix difference is COSMETIC: cargo canonicalizes a git URL and
strips a trailing `.git`, so those two forms are one source. Do not "fix" it as
a bug and do not count it as a second identity.

### 2.2 lance-graph today has ONE identity — by accident of exclusion

The root carries `[patch.crates-io] ndarray = { path = "../ndarray" }`. Note
the section: it redirects the REAL crates.io `ndarray` (the upstream numeric
crate this is a fork of) onto the fork, which is what stops a transitive pull
of upstream from becoming a second identity. It does **not** redirect the
AdaWorldAPI git URL — `[patch.crates-io]` cannot.

The two crates that DO use the git coordinate (`perturbation-sim`, `helix`) are
both workspace-EXCLUDED, and `helix` has its own `[workspace]`. So no member
binary sees both identities. **The single identity is a consequence of those
two crates being excluded, not of any rule.** Promote either to a member and
the graph carries two ndarrays.

### 2.3 The recorded blocker against a canonical git coordinate DOES NOT REPRODUCE

`Cargo.toml`'s patch comment says the git form "re-fetched AdaWorldAPI/ndarray
+ its burn submodule on every resolve; burn is outside the session repo scope
(403) and the gitlink rev is unfetchable, so the git form deadlocks offline
sessions."

Measured on master (`e1ef350`):

- **`burn` is not a submodule.** `git ls-tree HEAD` shows no `160000` gitlink
  and no `.gitmodules`. It is `crates/burn`, an in-tree workspace member
  (`members` line 484).
- The out-of-scope dependency is real but differently located:
  `crates/burn/Cargo.toml` git-deps `AdaWorldAPI/burn.git` rev `9b2b671`
  (three crates), and `AdaWorldAPI/elliptic-curves` appears elsewhere in the
  fork.
- **A git coordinate for `ndarray` nonetheless resolves clean.** A throwaway
  crate with `ndarray = { git = ".../ndarray", branch = "master",
  default-features = false, features = ["std"] }` returned
  `cargo metadata` exit 0, `Locking 10 packages`, `ndarray v0.17.2
  (…?branch=master#e1ef350a)`. No burn fetch. No 403. Cargo resolves the
  `ndarray` package, not every member's dependencies.

So **rule 5 is implementable** and the comment's stated cause is stale. The one
real remaining cost is honest and small: a git coordinate needs network on a
fresh resolve where a path coordinate needs none. That is a reason to `[patch]`
locally (rule 4), never a reason to keep the relative path as the durable
contract.

> The comment is corrected in place in the same commit that added this doc.
> Recorded here because a documented reason that no longer holds is worse than
> no reason: it is the only thing standing against rule 5, and it survived
> unchallenged until somebody ran `git ls-tree`.

### 2.4 One canonical source COUPLES THE FLEET'S MSRV — three repos pin below it

ndarray master reports `requires Rust 1.98`.

| toolchain | repos |
|---|---|
| **1.98.1** | lance-graph, OGAR, a2ui-rs, stockfish-rs, MedCare-rs, q2 |
| 1.97.1 | tesseract-rs |
| 1.95 | odoo-rs |
| 1.94.0 | ladybug-rs |

`tesseract-rs` and `ladybug-rs` both path-dep ndarray directly, so on current
master they are measurably unable to build against it. This is rule 3's real
price and it is not optional: **one canonical source means one MSRV floor for
every consumer of it.** Tracked as
`ISS-NDARRAY-CANONICAL-COORDINATE-COUPLES-FLEET-MSRV`; do NOT bump three
toolchains as a side effect of a dependency-unification pass.

### 2.5 One known duplicate algebra, differential GREEN, migration NOT started

`ogar-r2il::CallMask` carries `and`/`or`/`xor`/`and_not`/`not`/`count` over
inline `[u64; 3]`. Proven bit-identical to `lance-graph-mask-risc` over the
same borrowed words (`crates/r2il-mask-abi-probe`, 6/6, every test
disable-verified), and mask-risc's arbitrary ternlog immediate reproduces all
four binary ops — so CallMask's algebra is a SUBSET, not a sibling.

Ownership is deliberately undecided. Under rule 7 the destination is ndarray
via mask-risc; the three live options are tiny wrappers/oracle, a shared lower
primitive, or delegation if the direction is clean. `lance-graph-quack` states
the target shape in its own manifest: *"the masking algebra is reached THROUGH
mask-risc, never beside it."*

## 3. The target dependency geometry

Not this:

```text
lance-graph -> path ../../../ndarray
OGAR        -> git ndarray
odoo-rs     -> crates.io ndarray        # three packages, one name
```

But this:

```text
every repo:  ndarray = { workspace = true }
workspace root:
    [workspace.dependencies]
    ndarray = { git = "<canonical>", rev = "<pinned>" , ... }

local supercheckout only:
    [patch."<canonical>"]
    ndarray = { path = "../ndarray" }
```

Which gives:

```text
CI / standalone clone   ->  pinned git ndarray
local development       ->  same dep, transparently patched to ../ndarray
release binary          ->  exactly ONE resolved ndarray package
```

### The `[patch]` limit that must be understood before using it

A `[patch]` rewrites a **source**, not a dependency declaration. A manifest
that already says

```toml
ndarray = { path = "../../../ndarray" }
```

is **not** redirected by any `[patch]`. For such a crate the path geometry must
be correct, or the dependency has to be moved once onto a patchable coordinate
(a git source, or `workspace = true`). This is precisely why rule 5 is a rule
and not a preference: a relative path is the one form that cannot be
centrally redirected.

Within a SINGLE workspace, ordinary local crate paths stay — they are not
cross-repo and nothing here asks for them to change.

## 4. Falsifiers — rule 10 in practice

Documentation does not prove the law. Each rule gets a mechanical check:

| rule | check |
|---|---|
| 3, 9 | `cargo metadata` → count distinct package ids named `ndarray`; `cargo tree -d \| grep ndarray` must be empty |
| 5 | no `Cargo.toml` in the fleet matches `ndarray *= *{[^}]*path *= *"(\.\./){2,}` |
| 6 | no `ndarray = { … optional = true` in a crate on the compute-contract list |
| 2, 7 | no `core::arch`, `_mm_`, `#[cfg(target_arch` or `target_feature` outside `ndarray` itself and `#[cfg(test)]` oracles |
| 8 | a representative final binary LINKS: Quack, R2IL/OGAR, the Java ABI, the Odoo PoC |

A guard that cannot fail proves nothing: each check needs a disable run
(introduce the violation, watch the check go red) before it is trusted — the
same rule this workspace applies to every other gate.

## 5. Anti-patterns, each with the right shape beside it

| anti-pattern | right shape |
|---|---|
| "ndarray is mandatory, so this DTO crate must import it" | compute contract, not crate count — leave vocabulary crates alone |
| "one binary, so merge the crates" | one binary is a LINK property; keep the crates and their contracts |
| "the duplicate algebra is obviously wrong, delete it" | differential parity first; the difference is the finding |
| "add `optional = true` so the lean build skips ndarray" | a compute crate without its substrate has no contract left (rule 6) |
| "`[patch]` will unify it" — on a crate with a literal `path =` | `[patch]` rewrites sources, not path declarations |
| "`.git` suffix mismatch is a second identity" | cargo canonicalizes; measured, not a duplicate |
| "bump the three lagging toolchains while we are in here" | that is `ISS-NDARRAY-CANONICAL-COORDINATE-COUPLES-FLEET-MSRV`, its own decision |
