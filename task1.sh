#!/usr/bin/env bash
# task1.sh — replaces the Makefile targets with a single bash CLI
# Usage:
#   ./task1.sh <command> [--float f32|f16] [--ver N]
#   FLOAT=f16 VER=2 ./task1.sh run
# Commands: build | run | debug | profile | clean | clean-logs | help

set -euo pipefail

# -------- defaults (can be overridden by env or CLI) --------
FLOAT="${FLOAT:-f32}"
VER="${VER:-1}"

# -------- tiny args parser (flags override env) -------------
cmd="${1:-help}"
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--float) FLOAT="$2"; shift 2;;
    -v|--ver)   VER="$2";   shift 2;;
    -h|--help)  cmd="help"; shift;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

# -------- derived values -----------------------------------
if [[ "$FLOAT" == "f16" ]]; then
  FLOAT_TYPE="float16"
  FLOAT_FLAG="-f16"
else
  FLOAT_TYPE="float32"
  FLOAT_FLAG="-f32"
fi

BUILD_DIR="./build/src"
BINARY_NAME="task1_${FLOAT_TYPE}_v${VER}"
BINARY_PATH="${BUILD_DIR}/${BINARY_NAME}"
LOGS_DIR="logs"

# -------- helpers ------------------------------------------
timestamp() { date +"%Y%m%d_%H%M%S"; }
ensure_exec() {
  # ensure a script is executable; if missing, fail with a clear msg
  local p="$1"
  [[ -f "$p" ]] || { echo "Required script not found: $p" >&2; exit 1; }
  [[ -x "$p" ]] || chmod +x "$p" || true
}

do_build() {
  echo "=== Building with FLOAT=${FLOAT} VERSION=${VER} ==="
  ensure_exec "scripts/build-task1.sh"
  bash scripts/build-task1.sh "${FLOAT_FLAG}" "-v${VER}"
}

do_run() {
  do_build
  echo "=== Running ${BINARY_NAME} ==="
  mkdir -p "${LOGS_DIR}"
  local ts; ts="$(timestamp)"
  local log="${LOGS_DIR}/${BINARY_NAME}_${ts}.log"

  {
    echo "Run started at: $(date)"
    echo "FLOAT_TYPE: ${FLOAT_TYPE}"
    echo "VERSION: ${VER}"
    echo "======================================"
  } > "${log}"

  # run binary, tee to log
  "${BINARY_PATH}" 2>&1 | tee -a "${log}"

  {
    echo "======================================"
    echo "Run completed at: $(date)"
    echo "Log saved to: ${log}"
  } >> "${log}"
}

do_debug() {
  echo "=== Building Debug version with FLOAT=${FLOAT} VERSION=${VER} ==="
  ensure_exec "scripts/build-task1.sh"
  bash scripts/build-task1.sh "${FLOAT_FLAG}" "-v${VER}" "RD"
}

do_profile() {
  do_debug
  echo "=== Profiling ${BINARY_NAME} with Nsight Compute ==="
  mkdir -p "${LOGS_DIR}/profiles"
  ensure_exec "scripts/nsight-profile.sh"
  local ts; ts="$(timestamp)"
  local out="${LOGS_DIR}/profiles/${BINARY_NAME}_${ts}.ncu-rep"
  bash scripts/nsight-profile.sh -t "${BINARY_PATH}" -o "${out}"
  echo "Profile saved to: ${out}"
}

do_clean() {
  rm -rf "${BUILD_DIR}"
  echo "Build directory cleaned"
}

do_clean_logs() {
  rm -rf "${LOGS_DIR}"
  echo "Logs directory cleaned"
}

do_help() {
  cat <<EOF
task1.sh — build & run helper (Bash version of your Makefile)

Usage:
  ./task1.sh <command> [--float f32|f16] [--ver N]
  FLOAT=f16 VER=2 ./task1.sh run

Commands:
  build       Build code with specified FLOAT and VERSION
  run         Build and run code, save output to logs
  debug       Build with debug symbols (RelWithDebInfo via 'RD')
  profile     Run Nsight Compute profiling
  clean       Clean build directory (${BUILD_DIR})
  clean-logs  Clean logs directory (${LOGS_DIR})
  help        Show this help

Parameters:
  --float f32|f16   Data type (default: ${FLOAT})
  --ver   N         Version number (default: ${VER})

Examples:
  ./task1.sh build --float f16 --ver 2
  FLOAT=f32 VER=1 ./task1.sh run
  ./task1.sh debug --float f16 --ver 3
  ./task1.sh profile --float f32 --ver 1
EOF
}

# -------- dispatch -----------------------------------------
case "$cmd" in
  build)      do_build;;
  run)        do_run;;
  debug)      do_debug;;
  profile)    do_profile;;
  clean)      do_clean;;
  clean-logs) do_clean_logs;;
  help|*)     do_help;;
esac
