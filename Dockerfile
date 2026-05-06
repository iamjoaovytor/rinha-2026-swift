# syntax=docker/dockerfile:1.7

FROM --platform=linux/amd64 swift:6.1-jammy AS build-base
WORKDIR /src

COPY Package.swift Package.resolved ./
RUN swift package resolve

COPY Sources ./Sources
COPY Tests ./Tests
COPY resources ./resources

FROM build-base AS build-local
RUN mkdir -p /out/resources \
    && swift build -c release \
    && BIN=$(swift build -c release --show-bin-path) \
    && cp "$BIN/api" /out/api \
    && cp /src/resources/references.bin /out/resources/references.bin \
    && cp /src/resources/mcc_risk.json /out/resources/mcc_risk.json \
    && if [ -f /src/resources/references.ivf ]; then cp /src/resources/references.ivf /out/resources/references.ivf; fi

FROM build-base AS build-submission

RUN swift build -c release \
    && cd /src/resources \
    && sha256sum -c references.sha256 \
    && mkdir -p /out/resources \
    && BIN=$(swift build -c release --show-bin-path) \
    && "$BIN/preprocess" /src/resources/references.json.gz /out/resources/references.bin \
    && cp "$BIN/api" /out/api \
    && cp /src/resources/mcc_risk.json /out/resources/mcc_risk.json \
    && cp /out/resources/references.ivf /out/resources/references.ivf

FROM --platform=linux/amd64 swift:6.1-jammy-slim AS runtime-local
WORKDIR /app
COPY --from=build-local /out/api /app/api
COPY --from=build-local /out/resources/ /app/resources/

ENV REFERENCES_BIN=/app/resources/references.bin
ENV MCC_RISK_JSON=/app/resources/mcc_risk.json
ENV IVF_BIN=/app/resources/references.ivf

ENTRYPOINT ["/app/api"]

FROM --platform=linux/amd64 swift:6.1-jammy-slim AS runtime-submission
WORKDIR /app
COPY --from=build-submission /out/api /app/api
COPY --from=build-submission /out/resources/ /app/resources/

ENV REFERENCES_BIN=/app/resources/references.bin
ENV MCC_RISK_JSON=/app/resources/mcc_risk.json
ENV IVF_BIN=/app/resources/references.ivf

ENTRYPOINT ["/app/api"]
