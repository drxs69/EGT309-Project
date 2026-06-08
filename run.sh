#!/usr/bin/env bash
# run.sh
# -------
# Convenience script to execute the ElderGuard Analytics ML pipeline.
# Run from the project root directory.
#
# Usage:
#   ./run.sh                                   # default settings
#   ./run.sh --rf_n_estimators 300             # override RF trees
#   ./run.sh --gb_learning_rate 0.05           # override GB LR
#   ./run.sh --skip_train                      # evaluate saved models only

set -e

echo "========================================="
echo "  ElderGuard Analytics – Running Pipeline"
echo "========================================="

# Activate virtual environment if present
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Install dependencies if requirements.txt is present
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    python.exe -m pip install -r requirements.txt -q
fi

# Run the pipeline
python.exe src/pipeline.py "$@"

echo ""
echo "Pipeline finished. Check saved_model/ for outputs."
