# syntax=docker/dockerfile:1.7

FROM --platform=linux/amd64 swift:6.1-jammy AS build
WORKDIR /src

COPY Package.swift Package.resolved ./
RUN swift package resolve

COPY Sources ./Sources
RUN swift build -c release --product preprocess --product api

ARG REFERENCES_URL=https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz
ARG MCC_RISK_URL=https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/mcc_risk.json
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gzip \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /artifacts \
    && curl -fSL -o /artifacts/references.json.gz "$REFERENCES_URL" \
    && curl -fSL -o /artifacts/mcc_risk.json "$MCC_RISK_URL" \
    && /src/.build/release/preprocess /artifacts/references.json.gz /artifacts/references.bin \
    && rm /artifacts/references.json.gz

FROM --platform=linux/amd64 swift:6.1-jammy-slim AS runtime
WORKDIR /app
COPY --from=build /src/.build/release/api /app/api
COPY --from=build /artifacts/references.bin /app/resources/references.bin
COPY --from=build /artifacts/mcc_risk.json /app/resources/mcc_risk.json

ENV REFERENCES_BIN=/app/resources/references.bin
ENV MCC_RISK_JSON=/app/resources/mcc_risk.json

ENTRYPOINT ["/app/api"]
