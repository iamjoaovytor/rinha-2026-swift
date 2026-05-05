# syntax=docker/dockerfile:1.7

FROM --platform=linux/amd64 swift:6.1-jammy AS build
WORKDIR /src

COPY Package.swift Package.resolved ./
RUN swift package resolve

COPY Sources ./Sources
COPY Tests ./Tests
RUN swift build -c release --product api \
    && mkdir -p /out \
    && BIN=$(swift build -c release --show-bin-path) \
    && cp "$BIN/api" /out/api

FROM --platform=linux/amd64 swift:6.1-jammy-slim AS runtime
WORKDIR /app
COPY --from=build /out/api /app/api
COPY resources/references.bin /app/resources/references.bin
COPY resources/mcc_risk.json /app/resources/mcc_risk.json

ENV REFERENCES_BIN=/app/resources/references.bin
ENV MCC_RISK_JSON=/app/resources/mcc_risk.json

ENTRYPOINT ["/app/api"]
