#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
STATS_INTERVAL_SECONDS="${K6_STATS_INTERVAL_SECONDS:-5}"
STATS_LOG_PATH="${K6_STATS_LOG_PATH:-test/${MODE}-docker-stats.log}"
STATS_LOGGER_PID=""

case "$MODE" in
  smoke) SCRIPT="test/smoke.js" ;;
  stress) SCRIPT="test/stress.js" ;;
  full) SCRIPT="test/test.js" ;;
  *)
    echo "usage: scripts/run-k6.sh [smoke|stress|full]" >&2
    exit 2
    ;;
esac

cleanup() {
  if [[ -n "${STATS_LOGGER_PID}" ]]; then
    kill "${STATS_LOGGER_PID}" >/dev/null 2>&1 || true
    wait "${STATS_LOGGER_PID}" 2>/dev/null || true
  fi
}

start_stats_logger() {
  CONTAINER_IDS=()
  while IFS= read -r container_id; do
    if [[ -n "${container_id}" ]]; then
      CONTAINER_IDS+=("${container_id}")
    fi
  done < <(docker compose ps -q 2>/dev/null || true)
  if [[ "${#CONTAINER_IDS[@]}" -eq 0 ]]; then
    return
  fi

  : > "${STATS_LOG_PATH}"
  (
    while true; do
      printf '[%s]\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "${STATS_LOG_PATH}"
      docker stats --no-stream \
        --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}' \
        "${CONTAINER_IDS[@]}" >> "${STATS_LOG_PATH}" 2>/dev/null || true
      printf '\n' >> "${STATS_LOG_PATH}"
      sleep "${STATS_INTERVAL_SECONDS}"
    done
  ) &
  STATS_LOGGER_PID="$!"
  echo "docker stats logger -> ${STATS_LOG_PATH} (${STATS_INTERVAL_SECONDS}s)" >&2
}

trap cleanup EXIT INT TERM
start_stats_logger

if command -v k6 >/dev/null 2>&1; then
  BASE_URL="${BASE_URL:-http://localhost:9999}"
  k6 run "$SCRIPT" -e "BASE_URL=${BASE_URL}"
  exit $?
fi

BASE_URL="${BASE_URL:-http://lb:9999}"
NETWORK="${K6_DOCKER_NETWORK:-$(basename "$PWD")_default}"

docker run --rm \
  --network "$NETWORK" \
  -e "BASE_URL=${BASE_URL}" \
  -v "$PWD/test:/work/test" \
  -w /work \
  grafana/k6:latest run "$SCRIPT"
