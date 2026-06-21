# syntax=docker/dockerfile:1
#
# Cognitive Compilation golden-image build.
# Compiles `cognitive-stack` — the Elixir-template crates (NEW stack) linked on
# top of every AdaWorldAPI fork (OLD stack): lance-graph + lance =7.0.0 +
# lancedb =0.30.0 + ndarray + ractor + surrealdb (kv-lance, no rocksdb/tikv) +
# OGAR — into ONE binary. Purpose: PROVE the unified runtime compiles+links in a
# clean container (Railway/CI build validation); the binary prints the linked
# stack + the first-slice reflex template and exits. NO LLM is linked (the LLM is
# teacher-only, never in the reflex hot path).
#
# Build context = the lance-graph repo ROOT:
#   docker build -f crates/cognitive-stack/Dockerfile -t cognitive-stack .
#
# On Railway: set the service's Dockerfile path to
# `crates/cognitive-stack/Dockerfile` and the root directory to the repo root.
#
# AUTH: a Railway server has access to all private AdaWorldAPI repos by design;
# CARGO_NET_GIT_FETCH_WITH_CLI=true makes cargo use the system git (ambient creds).

ARG ADA_BRANCH=claude/jirak-math-theorems-harvest-rfii13

# ── builder ──────────────────────────────────────────────────────────────────
FROM rust:1.95-bookworm AS builder
ARG ADA_BRANCH
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true
# Native build deps for the closure: aws-lc-sys (C++), zstd/lz4 (C), and protoc
# (lance-encoding's build script REQUIRES protobuf-compiler — without it the
# build fails with "Could not find `protoc`").
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake clang protobuf-compiler pkg-config libssl-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# The lance-graph repo (this build context). cognitive-stack path-deps the
# in-repo crates (lance-graph, the four template crates, lance-graph-ogar).
COPY . /build/lance-graph

# lance-graph path-deps the ndarray fork as a SIBLING (`../../../ndarray`).
# Clone it next to the repo so that path resolves.
RUN git clone --depth 1 --branch "$ADA_BRANCH" \
        https://github.com/AdaWorldAPI/ndarray.git /build/ndarray

WORKDIR /build/lance-graph/crates/cognitive-stack
RUN cargo build --release

# ── runtime ──────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/lance-graph/crates/cognitive-stack/target/release/cognitive-stack /usr/local/bin/cognitive-stack
ENTRYPOINT ["cognitive-stack"]
