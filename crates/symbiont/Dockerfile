# syntax=docker/dockerfile:1
#
# Golden-image build validation for the Ada stack.
# Compiles the `symbiont` crate — lance-graph + lance =7.0.0 + lancedb =0.30.0
# + ndarray fork + ractor fork + surrealdb (kv-lance) + OGAR — into ONE binary.
# Purpose: PROVE the unified graph compiles+links in a clean container
# (Railway/CI build validation). The binary itself just prints the linked-stack
# line and exits.
#
# Build context = the lance-graph repo ROOT:
#   docker build -f crates/symbiont/Dockerfile -t symbiont .
#
# On Railway: set the service's Dockerfile path to `crates/symbiont/Dockerfile`
# and the root directory to the repo root.
#
# AUTH: a Railway server has access to all private AdaWorldAPI repositories by
# design, so no token wiring is needed. The build only has to make cargo use
# the SYSTEM git (which honors Railway's ambient credentials) rather than its
# built-in libgit2 transport — done via CARGO_NET_GIT_FETCH_WITH_CLI=true below.

# ndarray branch to clone: master, the default every other build and CI job uses.
ARG ADA_BRANCH=master

# ── builder ──────────────────────────────────────────────────────────────────
FROM rust:1.98.1-bookworm AS builder
ARG ADA_BRANCH
# Make cargo fetch git deps via the system `git` CLI so it uses Railway's
# ambient credentials for the private AdaWorldAPI forks.
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true
# Native build deps for the closure: aws-lc-sys (C++), zstd/lz4 (C), protoc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake clang protobuf-compiler pkg-config libssl-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# The lance-graph repo (this build context). symbiont path-deps lance-graph +
# perturbation-sim from inside it.
COPY . /build/lance-graph
# Production target. The repo's .cargo/config.toml defaults to `native`
# (tuned to the build host); production always runs on x86-64-v4, so the
# copied tree's config is replaced by config-v4.toml. That file only applies
# to x86_64 targets, so an arm64 build keeps its own aarch64 baseline (a
# global RUSTFLAGS would hand the aarch64 compiler an x86 CPU name).
RUN cp /build/lance-graph/.cargo/config-v4.toml /build/lance-graph/.cargo/config.toml

# lance-graph's own Cargo.toml path-deps the ndarray fork as a SIBLING
# (`../../../ndarray`). Clone it next to the repo so that path resolves.
RUN git clone --depth 1 --branch "$ADA_BRANCH" \
        https://github.com/AdaWorldAPI/ndarray.git /build/ndarray

WORKDIR /build/lance-graph/crates/symbiont
# `--target <host triple>` keeps the v4 flags on the binary only. Without it
# cargo also compiles build scripts and proc-macros for v4 and runs them
# here, so a build host without AVX-512 would die with SIGILL mid-build.
# With an explicit target they stay at the host baseline. The binaries then
# land under target/<triple>/ and are copied to the path the runtime stage
# reads.
RUN T="$(rustc -vV | sed -n 's/^host: //p')" \
    && cargo build --release --target "$T" \
    && mkdir -p target/release \
    && cp "target/$T/release/symbiont" target/release/

# ── runtime ──────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/lance-graph/crates/symbiont/target/release/symbiont /usr/local/bin/symbiont
ENTRYPOINT ["symbiont"]
