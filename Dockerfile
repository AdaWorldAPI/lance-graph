# lance-graph — Railway compile-test image
# Verifies the workspace builds cleanly (core + bgz17 + planner + contract)
#
# Build: docker build -t lance-graph-test .
# Run:   docker run --rm lance-graph-test

FROM rust:1.85-slim AS builder

WORKDIR /app

# System deps for arrow/lance/protobuf
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev protobuf-compiler cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifest and lock
COPY Cargo.toml Cargo.lock ./

# Copy all crate manifests for dep resolution
COPY crates/lance-graph/Cargo.toml crates/lance-graph/Cargo.toml
COPY crates/lance-graph-catalog/Cargo.toml crates/lance-graph-catalog/Cargo.toml
COPY crates/lance-graph-contract/Cargo.toml crates/lance-graph-contract/Cargo.toml
COPY crates/lance-graph-planner/Cargo.toml crates/lance-graph-planner/Cargo.toml
COPY crates/lance-graph-benches/Cargo.toml crates/lance-graph-benches/Cargo.toml
COPY crates/lance-graph-python/Cargo.toml crates/lance-graph-python/Cargo.toml
COPY crates/bgz17/Cargo.toml crates/bgz17/Cargo.toml

# Copy source
COPY crates/ crates/

# Build bgz17 standalone (zero deps, fast check)
RUN cargo build --release --manifest-path crates/bgz17/Cargo.toml 2>&1 \
    && echo "=== BGZ17 BUILD OK ==="

# Test bgz17 (121 tests)
RUN cargo test --release --manifest-path crates/bgz17/Cargo.toml 2>&1 \
    && echo "=== BGZ17 TESTS OK ==="

# Build workspace (skip python bindings — needs PyO3 runtime)
RUN cargo build --release \
    --workspace \
    --exclude lance-graph-python \
    2>&1 && echo "=== WORKSPACE BUILD OK ==="

# Run workspace tests (skip python)
RUN cargo test --release \
    --workspace \
    --exclude lance-graph-python \
    2>&1 && echo "=== WORKSPACE TESTS OK ==="

# Minimal runtime image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
CMD ["echo", "lance-graph build verified"]
