FROM rust:1.82-bookworm AS builder
WORKDIR /app

# Dependencies first (cached layer)
COPY Cargo.toml Cargo.lock ./
COPY crates/thinking-engine/Cargo.toml crates/thinking-engine/
COPY crates/bgz-tensor/Cargo.toml crates/bgz-tensor/
COPY crates/highheelbgz/Cargo.toml crates/highheelbgz/
RUN mkdir -p crates/thinking-engine/src && echo "fn main() {}" > crates/thinking-engine/src/lib.rs \
    && mkdir -p crates/bgz-tensor/src && echo "" > crates/bgz-tensor/src/lib.rs \
    && mkdir -p crates/highheelbgz/src && echo "" > crates/highheelbgz/src/lib.rs \
    && cargo build --release --manifest-path crates/thinking-engine/Cargo.toml 2>/dev/null || true

# Source code
COPY crates/ crates/

# Download codebooks from GitHub Release
ADD https://github.com/AdaWorldAPI/lance-graph/releases/download/v0.2.0-7lane-codebooks/qwen3-vl-embedding-7lane.tar.gz /tmp/
ADD https://github.com/AdaWorldAPI/lance-graph/releases/download/v0.2.0-7lane-codebooks/jina-v5-7lane.tar.gz /tmp/
ADD https://github.com/AdaWorldAPI/lance-graph/releases/download/v0.2.0-7lane-codebooks/jina-reranker-v3-BF16-7lane.tar.gz /tmp/
RUN cd /app/crates/thinking-engine/data && \
    tar xzf /tmp/qwen3-vl-embedding-7lane.tar.gz && \
    tar xzf /tmp/jina-v5-7lane.tar.gz && \
    tar xzf /tmp/jina-reranker-v3-BF16-7lane.tar.gz && \
    rm /tmp/*.tar.gz

# Build thinking engine
RUN cargo build --release --manifest-path crates/thinking-engine/Cargo.toml

# Runtime image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/crates/thinking-engine/target/release/examples/playground /usr/local/bin/thinking-engine
COPY --from=builder /app/crates/thinking-engine/data/*-7lane /app/data/
EXPOSE 8080
ENTRYPOINT ["thinking-engine"]
