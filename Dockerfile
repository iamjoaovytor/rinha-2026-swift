# syntax=docker/dockerfile:1.7

FROM --platform=linux/amd64 swift:6.1-jammy AS build-base
WORKDIR /src

COPY Package.swift Package.resolved ./
RUN swift package resolve

COPY Sources ./Sources
COPY Tests ./Tests

FROM build-base AS build-local
COPY resources/references.bin resources/mcc_risk.json ./resources/

RUN mkdir -p /out/resources \
    && swift build -c release \
    && BIN=$(swift build -c release --show-bin-path) \
    && cp "$BIN/api" /out/api \
    && cp /src/resources/references.bin /out/resources/references.bin \
    && cp /src/resources/mcc_risk.json /out/resources/mcc_risk.json

FROM build-base AS build-submission
COPY resources/references.json.gz resources/references.sha256 resources/mcc_risk.json ./resources/

RUN swift build -c release \
    && cd /src/resources \
    && sha256sum -c references.sha256 \
    && mkdir -p /out/resources \
    && BIN=$(swift build -c release --show-bin-path) \
    && "$BIN/preprocess" /src/resources/references.json.gz /out/resources/references.bin \
    && cp "$BIN/api" /out/api \
    && cp /src/resources/mcc_risk.json /out/resources/mcc_risk.json

FROM --platform=linux/amd64 swift:6.1-jammy-slim AS runtime-local
WORKDIR /app
COPY --from=build-local /out/api /app/api
COPY --from=build-local /out/resources/references.bin /app/resources/references.bin
COPY --from=build-local /out/resources/mcc_risk.json /app/resources/mcc_risk.json

ENV REFERENCES_BIN=/app/resources/references.bin
ENV MCC_RISK_JSON=/app/resources/mcc_risk.json

ENTRYPOINT ["/app/api"]

FROM --platform=linux/amd64 swift:6.1-jammy-slim AS runtime-submission
WORKDIR /app
COPY --from=build-submission /out/api /app/api
COPY --from=build-submission /out/resources/references.bin /app/resources/references.bin
COPY --from=build-submission /out/resources/mcc_risk.json /app/resources/mcc_risk.json

ENV REFERENCES_BIN=/app/resources/references.bin
ENV MCC_RISK_JSON=/app/resources/mcc_risk.json

ENTRYPOINT ["/app/api"]
