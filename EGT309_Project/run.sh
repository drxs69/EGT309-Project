#!/bin/bash
set -e

echo "========================================"
echo "ElderGuard Week 6 + Week 7 Pipeline"
echo "========================================"

echo "Step 1: Install Python requirements"
pip install -r requirements.txt

echo "Step 2: Run Week 7 ML training pipeline"
python src/main.py --config config.json

echo "Pipeline completed. Check results/, visuals/, and saved_model/."
