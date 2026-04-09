#!/bin/bash
# =============================================================================
# INVEREX Overnight Model vs Signal Benchmark — Orchestrator
# =============================================================================
#
# Runs all 7 agents end-to-end:
#   - Date gate: only runs after 2026-04-08 14:00 (Wednesday)
#   - Idempotency gate: skips if completed.flag exists
#   - caffeinate: prevents system sleep during execution
#   - Parallel: agents 1-6 run in parallel, agent 7 runs after they all finish
#
# Designed to be triggered by launchd. Safe to invoke multiple times — the
# date and completion gates make it a no-op outside the intended window.
# =============================================================================

set -u  # error on undefined vars (NOT -e: we want graceful failure handling)

PROJECT_ROOT="/Users/marinaesteban-medina/Desktop/INVEREX/inverex-mvp"
BENCH_DIR="${PROJECT_ROOT}/results/overnight_model_signal_benchmark"
LOG_DIR="${BENCH_DIR}/logs"
FLAG_DIR="${BENCH_DIR}/.flags"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"

# Target trigger time: Wednesday 2026-04-08 14:00 local time
TRIGGER_EPOCH=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-04-08 14:00:00" "+%s")

mkdir -p "${LOG_DIR}" "${FLAG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LAUNCHER_LOG}"
}

log "==============================================================="
log "INVEREX Overnight Benchmark Launcher invoked"
log "==============================================================="

# -----------------------------------------------------------------------------
# Idempotency gate: bail if already completed
# -----------------------------------------------------------------------------
if [[ -f "${FLAG_DIR}/completed.flag" ]]; then
    log "completed.flag exists — benchmark already finished. Exiting."
    exit 0
fi

# -----------------------------------------------------------------------------
# Date gate: only run after the trigger time
# -----------------------------------------------------------------------------
NOW_EPOCH=$(date "+%s")
if [[ "${NOW_EPOCH}" -lt "${TRIGGER_EPOCH}" ]]; then
    log "Current time is before trigger (Wed 2026-04-08 14:00). Exiting."
    log "  now=${NOW_EPOCH}  trigger=${TRIGGER_EPOCH}  delta=$((TRIGGER_EPOCH - NOW_EPOCH)) seconds"
    exit 0
fi

# -----------------------------------------------------------------------------
# Race-condition guard: only one launcher instance at a time
# -----------------------------------------------------------------------------
if [[ -f "${FLAG_DIR}/running.flag" ]]; then
    PID_RUNNING=$(cat "${FLAG_DIR}/running.flag")
    if kill -0 "${PID_RUNNING}" 2>/dev/null; then
        log "Another launcher instance (PID ${PID_RUNNING}) is already running. Exiting."
        exit 0
    else
        log "Stale running.flag found (PID ${PID_RUNNING} dead). Removing."
        rm -f "${FLAG_DIR}/running.flag"
    fi
fi
echo "$$" > "${FLAG_DIR}/running.flag"
trap 'rm -f "${FLAG_DIR}/running.flag"' EXIT

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
cd "${PROJECT_ROOT}" || { log "Failed to cd to ${PROJECT_ROOT}"; exit 1; }

# Ensure pixi is on PATH (launchd has a minimal environment)
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"
if ! command -v pixi &>/dev/null; then
    log "ERROR: pixi not found in PATH"
    exit 1
fi

log "Working directory: $(pwd)"
log "pixi: $(command -v pixi)"
log "Triggered at: $(date)"

# -----------------------------------------------------------------------------
# Run agents 1-6 in parallel under caffeinate
# -----------------------------------------------------------------------------
log "Starting agents 1-6 in parallel under caffeinate..."

# caffeinate -i: prevent idle sleep   -m: prevent disk sleep   -s: prevent system sleep
# We use one caffeinate that wraps the wait for all agents
(
    caffeinate -i -m -s &
    CAFFEINATE_PID=$!
    log "caffeinate PID: ${CAFFEINATE_PID}"

    # Launch all 6 agents in parallel
    declare -A AGENT_PIDS
    for i in 1 2 3 4 5 6; do
        case $i in
            1) script="agent1_lightgbm_sweep.py" ;;
            2) script="agent2_catboost.py" ;;
            3) script="agent3_alt_learners.py" ;;
            4) script="agent4_pathway.py" ;;
            5) script="agent5_stratified.py" ;;
            6) script="agent6_diagnostics.py" ;;
        esac
        log "Launching agent ${i}: ${script}"
        nohup pixi run python "scripts/overnight/${script}" \
            > "${LOG_DIR}/agent${i}_stdout.log" 2>&1 &
        AGENT_PIDS[$i]=$!
        log "  PID ${AGENT_PIDS[$i]}"
        sleep 2  # stagger to avoid simultaneous import contention
    done

    # Wait for all 6 to finish
    log "Waiting for all 6 agents to complete..."
    for i in 1 2 3 4 5 6; do
        if wait "${AGENT_PIDS[$i]}"; then
            log "  Agent ${i} (PID ${AGENT_PIDS[$i]}) completed successfully"
            touch "${FLAG_DIR}/agent${i}_done.flag"
        else
            log "  Agent ${i} (PID ${AGENT_PIDS[$i]}) FAILED with exit $?"
            touch "${FLAG_DIR}/agent${i}_failed.flag"
        fi
    done

    # Stop caffeinate
    log "All agents finished. Stopping caffeinate."
    kill "${CAFFEINATE_PID}" 2>/dev/null || true
)

# -----------------------------------------------------------------------------
# Run Agent 7 (final report) — depends on outputs from agents 1-6
# -----------------------------------------------------------------------------
log ""
log "Running Agent 7 (report generator)..."
if pixi run python scripts/overnight/agent7_report.py \
    > "${LOG_DIR}/agent7_stdout.log" 2>&1; then
    log "Agent 7 completed successfully"
    touch "${FLAG_DIR}/agent7_done.flag"
else
    log "Agent 7 FAILED with exit $?"
    touch "${FLAG_DIR}/agent7_failed.flag"
fi

# -----------------------------------------------------------------------------
# Mark complete
# -----------------------------------------------------------------------------
touch "${FLAG_DIR}/completed.flag"
log "==============================================================="
log "Benchmark complete. Final reports:"
log "  ${BENCH_DIR}/reports/final_overnight_summary.md"
log "  ${BENCH_DIR}/reports/morning_takeaway.txt"
log "==============================================================="
