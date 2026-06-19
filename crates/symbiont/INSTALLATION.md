# Installation — building the golden image

## Prerequisites

- **Toolchain:** Rust **1.95.0** (pinned in `rust-toolchain.toml`). OGAR uses
  edition 2024; ndarray / lance-graph / OGAR all pin 1.95.0; ractor builds on
  it (MSRV 1.64).
- **The five sibling repos must be present at these paths** (the `Cargo.toml`
  uses absolute paths):
  ```
  /home/user/ndarray
  /home/user/lance-graph
  /home/user/ractor
  /home/user/surrealdb
  /home/user/OGAR
  ```
- **A C toolchain** (`cc`) and **`cmake`** — needed by the native-build crates
  in the closure (see "C/C++ in the image" below). Present by default in the
  sandbox.
- **Disk:** budget ~10–15 GB for `target/` (datafusion + arrow + lance +
  aws-sdk + surrealdb is a large graph). The resolved graph is **912 packages**.
- **Network:** the first resolve fetches the crates.io index plus the
  `AdaWorldAPI/surrealdb` git repo (referenced by OGAR; redirected to the
  local fork by `[patch]`, but cargo still reads its index once). Subsequent
  builds are offline-capable.

## Build

```bash
cd /home/user/symbiont
cargo build              # debug; the binary IS the golden image
# or, for the smaller/faster artifact:
cargo build --release
```

A successful build is the deliverable — the entire unified graph compiled and
linked into `target/debug/symbiont`. **Verified 2026-06-19:** `cargo build`
exit 0, 19m18s, 4.2 MB binary, 912 packages, zero errors. (The current
`main.rs` is a probe `println!`; a real harness is tracked in
`INTEGRATION_PLAN.md`.)

## Feature posture (why these gates)

The image deliberately uses **`surrealdb-core` with `--no-default-features
--features kv-lance`** (wired in `symbiont/Cargo.toml`):

- **drops `kv-rocksdb`** → no RocksDB C++ storage engine,
- **drops `kv-tikv`** → no gRPC/protoc,
- keeps `kv-lance` → pure-Rust columnar store on `lance`/`lancedb`/`arrow`.

`lance-graph`, `ractor`, `ndarray`, and `OGAR` build with their defaults.
`perturbation-sim` is built **scalar** (zero-dep) in the first image; the
`ndarray-simd` acceleration is a later opt-in (see `INTEGRATION_PLAN.md`).

## C/C++ in the image (status: present, accepted)

Removing C++ was a best-effort nice-to-have, not a gate. What pulls native
build tooling, and where from:

| Crate | Language | Pulled by | Removable? |
|---|---|---|---|
| `aws-lc-sys` | **C++** | `jsonwebtoken[aws_lc_rs]` (core JWT auth) **and** `rustls` for S3 TLS | yes, with work (see below) |
| `aws-sdk-*`, `aws-config`, `aws-sigv4` | Rust | `object_store` S3 cloud support (lance/lancedb) | yes — drop S3 cloud features |
| `openssl-sys` | C | object_store / TLS path | yes, with the S3 path |
| `zstd-sys`, `lz4-sys`, `libz-sys`, `bzip2-sys` | C | lance/parquet/arrow compression | not the target — these link system libs, not a fresh compile |

**If a no-C++ image is wanted later** (optional): (1) build lance/lancedb
without S3 cloud object-store features so the `rustls`+`aws-sdk` TLS path
drops, and (2) flip `surrealdb-core`'s `jsonwebtoken` from `aws_lc_rs` to the
pure-Rust `rust_crypto` backend. Neither is required for the golden image to
build (which it now does — exit 0, with the C/C++ above present).

## Verifying the build (the acceptance gate, incrementally)

```bash
# 1. it compiles + links (the golden image itself)
cargo build

# 2. lint clean (part of the win condition)
cargo clippy --all-targets -- -D warnings

# 3. no dead dependencies (part of the win condition)
cargo install cargo-machete   # once
cargo machete

# 4. NaN-free perturbation cascade (the Spain-grid milestone)
#    run the perturbation-sim test workload over the grid fixture:
cargo test -p perturbation-sim
#    (Spain-grid example wiring tracked in INTEGRATION_PLAN.md)
```
