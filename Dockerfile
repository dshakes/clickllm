# syntax=docker/dockerfile:1
# Two stages: build and test the Rust core, then ship it beside the Python
# control plane. The final image carries no toolchain — only what runs.

FROM rust:1.93-slim AS core
WORKDIR /build
COPY Cargo.toml ./
COPY clickllm-core ./clickllm-core
# Fail the image build on a lint or test regression rather than shipping it.
RUN cargo test --release --all && cargo build --release

FROM python:3.13-slim AS runtime
LABEL org.opencontainers.image.title="clickllm" \
      org.opencontainers.image.description="Prove which open model can replace your closed one." \
      org.opencontainers.image.source="https://github.com/dshakes/clickllm" \
      org.opencontainers.image.licenses="Apache-2.0"

# Non-root by default: this tool reads hardware and writes a cache, never system state.
RUN useradd --create-home --uid 10001 clickllm

WORKDIR /app
COPY --from=core /build/target/release/libclickllm_core.rlib /app/lib/
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

USER clickllm
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# `fit` is the only subcommand that is useful with no state, so it is the default.
# Hardware detection sees the container's view; pass --gpus all to plan against
# the host's real accelerators.
HEALTHCHECK --interval=30s --timeout=5s CMD ["clickllm", "fit", "--json"]
ENTRYPOINT ["clickllm"]
CMD ["fit"]
