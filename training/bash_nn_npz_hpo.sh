#!/bin/bash

# Always run from repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1

# Define variables
SCRIPT_NAME="training/nn_npz_hpo.py"

LOG_DIR="training/nn_hpo_log_files"
mkdir -p "$LOG_DIR"

PID_FILE="$LOG_DIR/nn_npz_hpo_training.pid"

DATASET_IDX=$(printf "%02d" ${2:-1})
LOGFILE="$LOG_DIR/nn_training_hpo_${DATASET_IDX}_$(date +'%Y%m%d_%H%M%S').log"

# Use physical GPU 0 only
export CUDA_VISIBLE_DEVICES=0

export ACTIVATION_FUNCTION='gelu'
export CSV_PATH="./data/training_validation_data/npz_training_set.csv"
export DATASET_IDX=$DATASET_IDX

# Function to start the job
start_job() {
    # Check if a previous job is already running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Error: A training job is already running with PID $PID."
            echo "Stop it first using: ./run_pinn.sh stop"
            exit 1
        else
            echo "Warning: Stale PID file found. Removing..."
            rm -f "$PID_FILE"
        fi
    fi

    # Start the script in the background and save the PID
    nohup python "$SCRIPT_NAME" > "$LOGFILE" 2>&1 &
    echo $! > "$PID_FILE"

    echo "Training started in the background."
    echo "Logs: $LOGFILE"
    echo "To stop the process, use: ./run_pinn.sh stop"
}

# Function to stop the job
stop_job() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Error: No PID file found. Is the job running?"
        exit 1
    fi

    PID=$(cat "$PID_FILE")
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping training process (PID: $PID)..."
        kill $PID
        sleep 2  # Give it time to stop

        if ps -p $PID > /dev/null 2>&1; then
            echo "Process did not stop. Forcing termination..."
            kill -9 $PID
        fi

        echo "Training stopped successfully."
        rm -f "$PID_FILE"
    else
        echo "Error: Process $PID not found. Removing stale PID file."
        rm -f "$PID_FILE"
    fi
}

# Main control logic: start or stop the job
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
