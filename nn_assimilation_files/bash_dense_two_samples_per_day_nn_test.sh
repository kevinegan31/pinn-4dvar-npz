#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPT="${SCRIPT_DIR}/dense_two_samples_per_day_nn_test.py"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "$LOG_DIR"

LOGFILE="${LOG_DIR}/dense_two_samples_per_day_nn_test_$(date +'%Y%m%d_%H%M%S').log"
PID_FILE="${LOG_DIR}/dense_two_samples_per_day_nn_test_$(date +'%Y%m%d_%H%M%S').pid"

# NN-NPZ configuration
export NUM_LAYERS=5
export NUM_NEURONS=512
export LEARNING_RATE=0.0001

# Assimilation configuration
export NUM_XB0="${NUM_XB0:-10000}"
export NUM_CG="${NUM_CG:-2000}"
export NUM_JOBS="${NUM_JOBS:-4}"
export MIN_GUESS_VAL="${MIN_GUESS_VAL:-0.001}"

start_job() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")

        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Error: assimilation job already running with PID $PID."
            exit 1
        else
            echo "Removing stale PID file."
            rm -f "$PID_FILE"
        fi
    fi

    nohup python "$SCRIPT" > "$LOGFILE" 2>&1 &
    echo $! > "$PID_FILE"

    echo "NN-NPZ assimilation started."
    echo "PID: $(cat "$PID_FILE")"
    echo "Log: $LOGFILE"
}

stop_job() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Error: no PID file found."
        exit 1
    fi

    PID=$(cat "$PID_FILE")

    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping NN-NPZ assimilation (PID $PID)..."
        kill "$PID"
        sleep 2

        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Process did not stop; forcing termination."
            kill -9 "$PID"
        fi
    else
        echo "Process $PID is no longer running."
    fi

    rm -f "$PID_FILE"
}

case "$1" in
    start)
        start_job
        ;;
    stop)
        stop_job
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac
