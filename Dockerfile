# lance-graph — Railway compile-test image (target-cpu=native, the build host)
# Verifies the workspace builds cleanly (core + bgz17 + planner + contract)
# Requires Rust 1.99.0 (LazyLock, modern std APIs)
#
# CPU detection & SIMD dispatch documentation: see Dockerfile.md
# Dockerfile.avx512 is the canonical deploy file (sapphirerapids). This image
# is the fallback for a deploy that does not name that file: it builds for the
# build host's own CPU, so it takes whatever that host has. ndarray's startup
# cpu_guard stops the binary on a run host that lacks a compiled-in feature.
#
# Build: docker build -t lance-graph-test .
# Run:   docker run --rm lance-graph-test

FROM debian:bookworm-slim AS builder

# System deps for arrow/lance/protobuf
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gcc libc6-dev pkg-config libssl-dev \
    protobuf-compiler cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Rust 1.99.0 via rustup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain 1.99.0 --profile minimal \
    && rustc --version | grep -q "1.99.0"

WORKDIR /app

# Copy workspace manifest (no lock — see ISS-STALE-AUTHORITY-LOCKS)
COPY Cargo.toml ./

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

# The build host's own CPU, the same default as the repo's .cargo/config.toml
# (which this image does not copy, so it is set here).
ENV RUSTFLAGS="-C target-cpu=native"

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
CMD ["echo", "lance-graph build verified — Rust 1.99.0"]
