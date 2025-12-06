# Risk Manager - Rust Docker Image
# Enterprise-grade risk management

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM rust:1.75-slim as builder

WORKDIR /build

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Build release binary with optimizations
RUN cargo build --release

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /build/target/release/risk_manager /usr/local/bin/risk_manager

# Create non-root user
RUN useradd -m -u 1000 riskmanager

USER riskmanager

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD pgrep -x risk_manager > /dev/null || exit 1

CMD ["risk_manager"]
