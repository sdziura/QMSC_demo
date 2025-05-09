#!/bin/bash

# Set paths for logs
MLFLOW_LOG_DIR="mlruns"
TB_LOG_DIR="tb_logs"

# Ensure log directories exist
mkdir -p $MLFLOW_LOG_DIR
mkdir -p $TB_LOG_DIR

# Start MLflow server in foreground
echo "Starting MLflow server..."
mlflow server --host 127.0.0.1 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./$MLFLOW_LOG_DIR | tee logs/mlflow.log &

# Wait for MLflow to start
echo "Waiting for MLflow server to start..."
until nc -z 127.0.0.1 8080; do
  sleep 1
done
echo "MLflow server is ready."

# Start TensorBoard in foreground
echo "Waiting for TensorBoard server to start..."
tensorboard --logdir $TB_LOG_DIR --port 6006 | tee logs/tensorboard.log
until nc -z 127.0.0.1 6006; do
  sleep 1
done
echo "TensorBoard server is ready."
