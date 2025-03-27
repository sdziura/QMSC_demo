#!/bin/bash

# Set paths for logs
MLFLOW_LOG_DIR="mlruns"
TB_LOG_DIR="tb_logs"

# Ensure log directories exist
mkdir -p $MLFLOW_LOG_DIR
mkdir -p $TB_LOG_DIR

# Start MLflow server in the background and redirect output to a log file
echo "Starting MLflow server..."
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./$MLFLOW_LOG_DIR > mlflow.log 2>&1 &
MLFLOW_PID=$!
echo "MLflow server started with PID $MLFLOW_PID"

# Start TensorBoard server in the background and redirect output to a log file
echo "Starting TensorBoard..."
tensorboard --logdir $TB_LOG_DIR --port 6006 > tensorboard.log 2>&1 &
TB_PID=$!
echo "TensorBoard server started with PID $TB_PID"

# Save PIDs to files for stopping later
echo $MLFLOW_PID > mlflow.pid
echo $TB_PID > tensorboard.pid

# Display URLs
echo "MLflow is running at http://127.0.0.1:8080"
echo "TensorBoard is running at http://localhost:6006"

# Disown processes to keep them running after script exits
disown
