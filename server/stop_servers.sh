#!/bin/bash

# Function to stop a process safely
stop_process() {
    local pid_file=$1
    local process_name=$2

    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")

        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo "Stopping $process_name (PID $PID)..."
            kill "$PID"
            sleep 2  # Give time to shut down
            rm "$pid_file"
        else
            echo "$process_name not running, removing stale PID file."
            rm "$pid_file"
        fi
    else
        echo "$process_name not running or PID file missing."
    fi
}

# Stop MLflow and TensorBoard
stop_process "mlflow.pid" "MLflow"
stop_process "tensorboard.pid" "TensorBoard"

echo "Servers stopped."
