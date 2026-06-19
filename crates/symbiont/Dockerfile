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
# AUTH: the AdaWorldAPI forks must be reachable during the build. If they are
# private, provide a token — e.g. mount a git credential or pass a build secret
# and configure `git config --global url."https://x-access-token:$TOKEN@github.com/".insteadOf "https://github.com/"`.

ARG ADA_BRANCH=claude/jirak-math-theorems-harvest-rfii13

# ── builder ──────────────────────────────────────────────────────────────────
FROM rust:1.95-bookworm AS builder
ARG ADA_BRANCH
# Native build deps for the closure: aws-lc-sys (C++), zstd/lz4 (C), protoc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake clang protobuf-compiler pkg-config libssl-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# The lance-graph repo (this build context). symbiont path-deps lance-graph +
# perturbation-sim from inside it.
COPY . /build/lance-graph

# lance-graph's own Cargo.toml path-deps the ndarray fork as a SIBLING
# (`../../../ndarray`). Clone it next to the repo so that path resolves.
RUN git clone --depth 1 --branch "$ADA_BRANCH" \
        https://github.com/AdaWorldAPI/ndarray.git /build/ndarray

WORKDIR /build/lance-graph/crates/symbiont
RUN cargo build --release

# ── runtime ──────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/lance-graph/crates/symbiont/target/release/symbiont /usr/local/bin/symbiont
ENTRYPOINT ["symbiont"]
