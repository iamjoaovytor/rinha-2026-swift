#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
usage: scripts/benchmark_linux_variant.sh <linux-host> <image-tag> [build-arg...]

examples:
  scripts/benchmark_linux_variant.sh iamjoaovytor@192.168.0.13 baseline
  scripts/benchmark_linux_variant.sh iamjoaovytor@192.168.0.13 pgo-75k PGO_WARMUP_COUNT=75000
  scripts/benchmark_linux_variant.sh iamjoaovytor@192.168.0.13 jemalloc RUNTIME_ALLOCATOR=jemalloc ENV:WARMUP_COUNT=75000
  scripts/benchmark_linux_variant.sh iamjoaovytor@192.168.0.13 jemalloc-fast TARGET:runtime-submission-no-pgo RUNTIME_ALLOCATOR=jemalloc

notes:
  - syncs the current repo to ~/Documents/rinha-2026-swift on the Linux host
  - builds the selected target as joaovytor/rinha-2026-swift:<image-tag>
  - spins the stack up through a temporary compose override
  - runs the official k6 full benchmark from ~/Documents/rinha/official-rinha
  - tears the stack down and prints p99/score/fp/fn/http_errors
EOF
  exit 64
fi

HOST="$1"
IMAGE_TAG="$2"
shift 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_REPO_DIR="/home/iamjoaovytor/Documents/rinha-2026-swift"
REMOTE_RUNNER_DIR="/home/iamjoaovytor/Documents/rinha/official-rinha"
REMOTE_OVERRIDE="/tmp/docker-compose.${IMAGE_TAG}.override.yml"
REMOTE_RESULTS="/tmp/${IMAGE_TAG}.results.json"
REMOTE_IMAGE="joaovytor/rinha-2026-swift:${IMAGE_TAG}"

BUILD_ARGS=()
ENV_OVERRIDES=()
BUILD_TARGET="runtime-submission"
for arg in "$@"; do
  if [[ "$arg" == ENV:* ]]; then
    ENV_OVERRIDES+=("${arg#ENV:}")
  elif [[ "$arg" == TARGET:* ]]; then
    BUILD_TARGET="${arg#TARGET:}"
  else
    BUILD_ARGS+=(--build-arg "$arg")
  fi
done
BUILD_ARGS_STR=""
if [[ ${#BUILD_ARGS[@]} -gt 0 ]]; then
  printf -v BUILD_ARGS_STR '%q ' "${BUILD_ARGS[@]}"
fi
ENV_BLOCK=""
if [[ ${#ENV_OVERRIDES[@]} -gt 0 ]]; then
  ENV_BLOCK+="    environment:\n"
  for env_kv in "${ENV_OVERRIDES[@]}"; do
    key="${env_kv%%=*}"
    value="${env_kv#*=}"
    ENV_BLOCK+="      ${key}: \"${value}\"\n"
  done
fi

printf 'syncing repo to %s\n' "$HOST"
rsync -avz --delete \
  --exclude='.git' \
  --exclude='.build' \
  --exclude='.claude' \
  --exclude='.DS_Store' \
  --exclude='ARCHITECTURE.md' \
  "${ROOT_DIR}/" "${HOST}:${REMOTE_REPO_DIR}/"

printf 'building %s on %s\n' "${REMOTE_IMAGE}" "$HOST"
ssh "$HOST" "bash -lc '
  set -euo pipefail
  mkdir -p /home/iamjoaovytor/Documents/rinha
  if [ ! -d $(printf %q "$REMOTE_RUNNER_DIR") ]; then
    git clone --depth 1 https://github.com/zanfranceschi/rinha-de-backend-2026 $(printf %q "$REMOTE_RUNNER_DIR")
  fi
  cd $(printf %q "$REMOTE_REPO_DIR")
  docker compose down -v --remove-orphans >/tmp/${IMAGE_TAG}.cleanup.log 2>&1 || true
  docker buildx build --platform=linux/amd64 --target=$(printf %q "$BUILD_TARGET") ${BUILD_ARGS_STR}-t $(printf %q "$REMOTE_IMAGE") --load . >/tmp/${IMAGE_TAG}.build.log 2>&1
  cat > $(printf %q "$REMOTE_OVERRIDE") <<EOF
services:
  api1:
    image: ${REMOTE_IMAGE}
$(printf '%b' "$ENV_BLOCK")
  api2:
    image: ${REMOTE_IMAGE}
$(printf '%b' "$ENV_BLOCK")
EOF
  docker compose -f docker-compose.yml -f $(printf %q "$REMOTE_OVERRIDE") down -v --remove-orphans >/tmp/${IMAGE_TAG}.down.log 2>&1 || true
  docker compose -f docker-compose.yml -f $(printf %q "$REMOTE_OVERRIDE") up -d >/tmp/${IMAGE_TAG}.up.log 2>&1
  for i in \$(seq 1 900); do
    if curl -fsS http://localhost:9999/ready >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  cd $(printf %q "$REMOTE_RUNNER_DIR")
  rm -f test/results.json
  k6 run test/test.js >/tmp/${IMAGE_TAG}.k6.log 2>&1
  cp test/results.json $(printf %q "$REMOTE_RESULTS")
  docker compose -f $(printf %q "$REMOTE_REPO_DIR")/docker-compose.yml -f $(printf %q "$REMOTE_OVERRIDE") down -v --remove-orphans >/tmp/${IMAGE_TAG}.finaldown.log 2>&1 || true
'"

printf 'result for %s\n' "${REMOTE_IMAGE}"
ssh "$HOST" "jq -r '{p99, score: .scoring.final_score, http_errors: .scoring.breakdown.http_errors, fp: .scoring.breakdown.false_positive_detections, fn: .scoring.breakdown.false_negative_detections}' '$(printf "%s" "$REMOTE_RESULTS")'"
