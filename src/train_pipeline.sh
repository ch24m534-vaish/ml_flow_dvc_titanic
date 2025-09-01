#!/bin/bash
set -e

NEW_DATA=$1
OLD_DATA=data/raw/train.csv
MERGED_DATA=data/processed/
MODEL_NAME="TitanicModel"

# Start MLflow
echo "Starting MLflow server..."
mlflow server \
  --backend-store-uri sqlite:///./mlruns_db/mlflow.db \
  --default-artifact-root ./mlruns_db/artifacts \
  --host 0.0.0.0 \
  --port 5000 > mlflow.log 2>&1 &
MLFLOW_PID=$!
echo "MLflow started with PID: $MLFLOW_PID" sleep 5
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Start timer 
START_TIME=$(date +%s)

# First training
if [ ! -f "$OLD_DATA" ]; then
    if [ -z "$NEW_DATA" ]; then
        echo "No data to train."
        kill $MLFLOW_PID
        exit 1
    fi

    echo "⚡ First training..."
    mkdir -p data/raw
    cp "$NEW_DATA" "$OLD_DATA"

    dvc repro --force

    LOGGED_URI=$(cat models/initial/logged_model_uri.txt)
    python3 - <<PY
from utils import register_and_promote_model
register_and_promote_model("$LOGGED_URI", "$MODEL_NAME", validation_passed=True)
PY

    END_TIME=$(date +%s)
    echo "First training done in $((END_TIME - START_TIME)) seconds."
    kill $MLFLOW_PID
    exit 0
fi

# Subsequent training with new data
if [ -z "$NEW_DATA" ]; then
    echo "No new data. Nothing to do."
    kill $MLFLOW_PID
    exit 0
fi

# Drift detection
if python3 src/drift_detector.py --old $OLD_DATA --new $NEW_DATA --params params.yaml; then
    echo "No drift"
fi

# Drift detected → merge and retrain
echo " Drift detected. Merging data and retraining..."
python3 - <<PY
import pandas as pd
old = pd.read_csv("$OLD_DATA")
new = pd.read_csv("$NEW_DATA")
merged = pd.concat([old, new], ignore_index=True)
merged.to_csv("$OLD_DATA", index=False)
PY

dvc repro --force

END_TIME=$(date +%s)
echo "Retraining completed in $((END_TIME - START_TIME)) seconds."

# kill $MLFLOW_PID
# echo "MLflow server stopped."
