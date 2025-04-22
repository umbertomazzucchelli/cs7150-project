#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Preprocessing Step ---
echo "Running WISDM Preprocessing..."
python preprocess_wisdm.py
echo "WISDM Preprocessing complete."

# --- Model Execution Step ---
echo "\nStarting HAR Model Comparison Run..."

# Run the main model comparison script 
# This will execute models on WISDM first, then R1 (wrist & thigh)
python run_har_models.py --max_workers 12 --log_dir "logs_wisdm"

echo "\nHAR Model Comparison Run finished."
