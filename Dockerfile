# syntax=docker/dockerfile:1.7

FROM --platform=linux/amd64 swift:6.1-jammy AS build-base
WORKDIR /src

COPY Package.swift Package.resolved ./
RUN swift package resolve

COPY Sources ./Sources
COPY Tests ./Tests
COPY resources ./resources

FROM build-base AS build-local
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=cache,target=/root/.swiftpm \
    --mount=type=cache,target=/src/.build \
    mkdir -p /out/resources \
    && swift build -c release --product api \
        -Xswiftc -O \
        -Xswiftc -wmo \
        -Xswiftc -enforce-exclusivity=unchecked \
        -Xswiftc -gnone \
        -Xcc -O3 \
        -Xcc -march=x86-64-v3 \
        -Xcc -mavx2 \
        -Xcc -mfma \
    && BIN_DIR=$(dirname "$(find /src/.build -type f -path '*/release/api' | head -n 1)") \
    && cp "$BIN_DIR/api" /out/api \
    && cp /src/resources/references.bin /out/resources/references.bin \
    && cp /src/resources/mcc_risk.json /out/resources/mcc_risk.json \
    && if [ -f /src/resources/references.ivf ]; then cp /src/resources/references.ivf /out/resources/references.ivf; fi \
    && if [ -f /src/resources/references.pq ]; then cp /src/resources/references.pq /out/resources/references.pq; fi

FROM build-base AS build-submission

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=cache,target=/root/.swiftpm \
    --mount=type=cache,target=/src/.build \
    rm -rf /src/.build 2>/dev/null || true \
    && swift build -c release --product api \
        -Xswiftc -O \
        -Xswiftc -wmo \
        -Xswiftc -enforce-exclusivity=unchecked \
        -Xswiftc -gnone \
        -Xswiftc -use-ld=lld \
        -Xcc -O3 \
        -Xcc -march=x86-64-v3 \
        -Xcc -mtune=haswell \
        -Xcc -mavx2 \
        -Xcc -mfma \
        -Xcc -flto=thin \
        -Xcc -funroll-loops \
        -Xcc -falign-functions=64 \
        -Xcc -falign-loops=64 \
        -Xlinker --icf=all \
    && swift build -c release --product preprocess \
        -Xswiftc -O \
        -Xswiftc -wmo \
        -Xswiftc -enforce-exclusivity=unchecked \
        -Xswiftc -gnone \
        -Xswiftc -use-ld=lld \
        -Xcc -O3 \
        -Xcc -march=x86-64-v3 \
        -Xcc -mtune=haswell \
        -Xcc -mavx2 \
        -Xcc -mfma \
        -Xcc -flto=thin \
        -Xcc -funroll-loops \
        -Xcc -falign-functions=64 \
        -Xcc -falign-loops=64 \
        -Xlinker --icf=all \
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
COPY --from=build-local /out/api /app/api
COPY --from=build-local /out/resources/ /app/resources/

ENV REFERENCES_BIN=/app/resources/references.bin
ENV MCC_RISK_JSON=/app/resources/mcc_risk.json
ENV IVF_BIN=/app/resources/references.ivf
ENV IVFPQ_BIN=/app/resources/references.pq

ENTRYPOINT ["/app/api"]

FROM --platform=linux/amd64 swift:6.1-jammy-slim AS runtime-submission
WORKDIR /app
COPY --from=build-submission /out/api /app/api
COPY --from=build-submission /out/resources/ /app/resources/

ENV REFERENCES_BIN=/app/resources/references.bin
ENV MCC_RISK_JSON=/app/resources/mcc_risk.json
ENV IVF_BIN=/app/resources/references.ivf
ENV IVFPQ_BIN=/app/resources/references.pq

ENTRYPOINT ["/app/api"]
