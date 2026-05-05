#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"

case "$MODE" in
  smoke) SCRIPT="test/smoke.js" ;;
  full) SCRIPT="test/test.js" ;;
  *)
    echo "usage: scripts/run-k6.sh [smoke|full]" >&2
    exit 2
    ;;
esac

if command -v k6 >/dev/null 2>&1; then
  BASE_URL="${BASE_URL:-http://localhost:9999}"
  exec k6 run "$SCRIPT" -e "BASE_URL=${BASE_URL}"
fi

BASE_URL="${BASE_URL:-http://lb:9999}"
NETWORK="${K6_DOCKER_NETWORK:-$(basename "$PWD")_default}"

exec docker run --rm \
  --network "$NETWORK" \
  -e "BASE_URL=${BASE_URL}" \
  -v "$PWD/test:/work/test" \
  -w /work \
  grafana/k6:latest run "$SCRIPT"
