# syntax=docker/dockerfile:1.7

FROM --platform=linux/amd64 swift:6.1-jammy AS build-base
WORKDIR /src

ARG SWIFT_RELEASE_FLAGS="-Xswiftc -O -Xswiftc -wmo -Xswiftc -enforce-exclusivity=unchecked -Xswiftc -gnone -Xswiftc -use-ld=lld"
ARG C_RELEASE_FLAGS="-Xcc -O3 -Xcc -march=x86-64-v3 -Xcc -mtune=haswell -Xcc -mavx2 -Xcc -mfma -Xcc -flto=thin -Xcc -funroll-loops -Xcc -falign-functions=64 -Xcc -falign-loops=64"
ARG LD_RELEASE_FLAGS="-Xlinker --icf=all"
ARG SWIFT_PGO_GENERATE_FLAGS="-Xswiftc -profile-generate"
ARG C_PGO_GENERATE_FLAGS="-Xcc -fprofile-generate"
ARG SWIFT_PGO_USE_FLAGS="-Xswiftc -profile-use=/tmp/pgo/merged.profdata"
ARG C_PGO_USE_FLAGS="-Xcc -fprofile-use=/tmp/pgo/merged.profdata -Xcc -Wno-profile-instr-unprofiled -Xcc -Wno-profile-instr-out-of-date"
ARG PGO_WARMUP_COUNT="50000"
ARG PGO_READY_POLL_COUNT="1200"
ARG RUNTIME_ALLOCATOR="glibc"

COPY Package.swift Package.resolved ./
RUN swift package resolve

COPY Sources ./Sources
COPY Tests ./Tests
COPY resources ./resources

FROM build-base AS build-local
ARG SWIFT_RELEASE_FLAGS
ARG C_RELEASE_FLAGS
ARG LD_RELEASE_FLAGS
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=cache,target=/root/.swiftpm \
    --mount=type=cache,target=/src/.build \
    mkdir -p /out/resources \
    && swift build -c release --product api $SWIFT_RELEASE_FLAGS $C_RELEASE_FLAGS $LD_RELEASE_FLAGS \
    && BIN_DIR=$(dirname "$(find /src/.build -type f -path '*/release/api' | head -n 1)") \
    && cp "$BIN_DIR/api" /out/api \
    && cp /src/resources/references.bin /out/resources/references.bin \
    && cp /src/resources/mcc_risk.json /out/resources/mcc_risk.json \
    && if [ -f /src/resources/references.ivf ]; then cp /src/resources/references.ivf /out/resources/references.ivf; fi \
    && if [ -f /src/resources/references.pq ]; then cp /src/resources/references.pq /out/resources/references.pq; fi

FROM build-base AS build-pgo-profile
ARG SWIFT_RELEASE_FLAGS
ARG C_RELEASE_FLAGS
ARG LD_RELEASE_FLAGS
ARG SWIFT_PGO_GENERATE_FLAGS
ARG C_PGO_GENERATE_FLAGS
ARG PGO_WARMUP_COUNT
ARG PGO_READY_POLL_COUNT
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=cache,target=/root/.swiftpm \
    --mount=type=cache,target=/src/.build \
    rm -rf /src/.build /tmp/pgo 2>/dev/null || true \
    && mkdir -p /tmp/pgo/profraw \
    && swift build -c release --product api $SWIFT_RELEASE_FLAGS $C_RELEASE_FLAGS $LD_RELEASE_FLAGS $SWIFT_PGO_GENERATE_FLAGS $C_PGO_GENERATE_FLAGS \
    && API_BIN=$(find /src/.build -type f -path '*/release/api' | head -n 1) \
    && test -n "$API_BIN" || (echo "ERROR: instrumented api binary not found" && false) \
    && ( \
        LLVM_PROFILE_FILE=/tmp/pgo/profraw/api-%p.profraw \
        REFERENCES_BIN=/src/resources/references.bin \
        MCC_RISK_JSON=/src/resources/mcc_risk.json \
        IVF_BIN=/src/resources/references.ivf \
        IVFPQ_BIN=/src/resources/references.pq \
        WARMUP_COUNT=$PGO_WARMUP_COUNT \
        PORT=9999 \
        "$API_BIN" >/tmp/pgo/api.stdout 2>/tmp/pgo/api.stderr & \
        API_PID=$!; \
        READY=0; \
        for i in $(seq 1 $PGO_READY_POLL_COUNT); do \
            python3 -c 'import sys, urllib.request; sys.exit(0 if 200 <= urllib.request.urlopen("http://127.0.0.1:9999/ready", timeout=1).status < 300 else 1)' >/dev/null 2>&1 \
            && READY=1 \
            && break || true; \
            sleep 0.25; \
        done; \
        test "$READY" = 1; \
        kill "$API_PID"; \
        wait "$API_PID" || true; \
    ) \
    && test -n "$(find /tmp/pgo/profraw -name '*.profraw' -print -quit)" \
    && llvm-profdata merge -output=/tmp/pgo/merged.profdata /tmp/pgo/profraw/*.profraw \
    && test -f /tmp/pgo/merged.profdata

FROM build-base AS build-submission
ARG SWIFT_RELEASE_FLAGS
ARG C_RELEASE_FLAGS
ARG LD_RELEASE_FLAGS
ARG SWIFT_PGO_USE_FLAGS
ARG C_PGO_USE_FLAGS
COPY --from=build-pgo-profile /tmp/pgo/merged.profdata /tmp/pgo/merged.profdata

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=cache,target=/root/.swiftpm \
    --mount=type=cache,target=/src/.build \
    rm -rf /src/.build 2>/dev/null || true \
    && swift build -c release --product api $SWIFT_RELEASE_FLAGS $C_RELEASE_FLAGS $LD_RELEASE_FLAGS $SWIFT_PGO_USE_FLAGS $C_PGO_USE_FLAGS \
    && swift build -c release --product preprocess $SWIFT_RELEASE_FLAGS $C_RELEASE_FLAGS $LD_RELEASE_FLAGS $SWIFT_PGO_USE_FLAGS $C_PGO_USE_FLAGS \
    && cd /src/resources \
    && sha256sum -c references.sha256 \
    && mkdir -p /out/resources \
    && API_BIN=$(find /src/.build -type f -path '*/release/api' | head -n 1) \
    && PREPROCESS_BIN=$(find /src/.build -type f -path '*/release/preprocess' | head -n 1) \
    && test -n "$API_BIN" || (echo "ERROR: api binary not found" && false) \
    && test -n "$PREPROCESS_BIN" || (echo "ERROR: preprocess binary not found" && false) \
    && IVFPQ_BUILD=1 \
       IVFPQ_SUBVECTORS=4 \
       IVFPQ_TRAIN_SAMPLE=131072 \
       IVFPQ_TRAIN_ITERS=8 \
       IVFPQ_SEED=42 \
       "$PREPROCESS_BIN" /src/resources/references.json.gz /out/resources/references.bin \
    && cp "$API_BIN" /out/api \
    && cp /src/resources/mcc_risk.json /out/resources/mcc_risk.json \
    && test -f /out/resources/references.ivf \
    && test -f /out/resources/references.pq

FROM build-base AS build-submission-no-pgo
ARG SWIFT_RELEASE_FLAGS
ARG C_RELEASE_FLAGS
ARG LD_RELEASE_FLAGS

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=cache,target=/root/.swiftpm \
    --mount=type=cache,target=/src/.build \
    rm -rf /src/.build 2>/dev/null || true \
    && swift build -c release --product api $SWIFT_RELEASE_FLAGS $C_RELEASE_FLAGS $LD_RELEASE_FLAGS \
    && swift build -c release --product preprocess $SWIFT_RELEASE_FLAGS $C_RELEASE_FLAGS $LD_RELEASE_FLAGS \
    && cd /src/resources \
    && sha256sum -c references.sha256 \
    && mkdir -p /out/resources \
    && API_BIN=$(find /src/.build -type f -path '*/release/api' | head -n 1) \
    && PREPROCESS_BIN=$(find /src/.build -type f -path '*/release/preprocess' | head -n 1) \
    && test -n "$API_BIN" || (echo "ERROR: api binary not found" && false) \
    && test -n "$PREPROCESS_BIN" || (echo "ERROR: preprocess binary not found" && false) \
    && IVFPQ_BUILD=1 \
       IVFPQ_SUBVECTORS=4 \
       IVFPQ_TRAIN_SAMPLE=131072 \
       IVFPQ_TRAIN_ITERS=8 \
       IVFPQ_SEED=42 \
       "$PREPROCESS_BIN" /src/resources/references.json.gz /out/resources/references.bin \
    && cp "$API_BIN" /out/api \
    && cp /src/resources/mcc_risk.json /out/resources/mcc_risk.json \
    && test -f /out/resources/references.ivf \
    && test -f /out/resources/references.pq

FROM --platform=linux/amd64 swift:6.1-jammy-slim AS runtime-local
WORKDIR /app
ARG RUNTIME_ALLOCATOR
COPY --from=build-local /out/api /app/api
COPY --from=build-local /out/resources/ /app/resources/

RUN if [ "$RUNTIME_ALLOCATOR" = "jemalloc" ]; then \
      apt-get update && apt-get install -y --no-install-recommends libjemalloc2 && rm -rf /var/lib/apt/lists/*; \
    fi

ENV REFERENCES_BIN=/app/resources/references.bin
ENV MCC_RISK_JSON=/app/resources/mcc_risk.json
ENV IVF_BIN=/app/resources/references.ivf
ENV IVFPQ_BIN=/app/resources/references.pq
ENV LD_PRELOAD=""
RUN if [ "$RUNTIME_ALLOCATOR" = "jemalloc" ]; then echo "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2" >/etc/ld.so.preload; fi

ENTRYPOINT ["/app/api"]

FROM --platform=linux/amd64 swift:6.1-jammy-slim AS runtime-submission-no-pgo
WORKDIR /app
ARG RUNTIME_ALLOCATOR
COPY --from=build-submission-no-pgo /out/api /app/api
COPY --from=build-submission-no-pgo /out/resources/ /app/resources/

RUN if [ "$RUNTIME_ALLOCATOR" = "jemalloc" ]; then \
      apt-get update && apt-get install -y --no-install-recommends libjemalloc2 && rm -rf /var/lib/apt/lists/*; \
    fi

ENV REFERENCES_BIN=/app/resources/references.bin
ENV MCC_RISK_JSON=/app/resources/mcc_risk.json
ENV IVF_BIN=/app/resources/references.ivf
ENV IVFPQ_BIN=/app/resources/references.pq
ENV LD_PRELOAD=""
RUN if [ "$RUNTIME_ALLOCATOR" = "jemalloc" ]; then echo "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2" >/etc/ld.so.preload; fi

ENTRYPOINT ["/app/api"]

FROM --platform=linux/amd64 swift:6.1-jammy-slim AS runtime-submission
WORKDIR /app
ARG RUNTIME_ALLOCATOR
COPY --from=build-submission /out/api /app/api
COPY --from=build-submission /out/resources/ /app/resources/

RUN if [ "$RUNTIME_ALLOCATOR" = "jemalloc" ]; then \
      apt-get update && apt-get install -y --no-install-recommends libjemalloc2 && rm -rf /var/lib/apt/lists/*; \
    fi

ENV REFERENCES_BIN=/app/resources/references.bin
ENV MCC_RISK_JSON=/app/resources/mcc_risk.json
ENV IVF_BIN=/app/resources/references.ivf
ENV IVFPQ_BIN=/app/resources/references.pq
ENV LD_PRELOAD=""
RUN if [ "$RUNTIME_ALLOCATOR" = "jemalloc" ]; then echo "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2" >/etc/ld.so.preload; fi

ENTRYPOINT ["/app/api"]
