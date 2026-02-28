#!/bin/bash
# SEP QA Pipeline — Runs all 4 datasets through generation, NLI, extraction, and probe training.
#
# Usage:
#   nohup bash run_pipeline.sh > pipeline_qa.log 2>&1 &
#
# Or run individual datasets:
#   bash run_pipeline.sh squad
#   bash run_pipeline.sh trivia_qa nq
set -e

PYTHON=/home/anish/miniconda3/envs/se_probes/bin/python
WORKDIR=/home/anish/yaawar/LLM/semantic-entropy-probes
cd "$WORKDIR"

# If specific datasets passed as args, use those; otherwise use all 4
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("squad" "trivia_qa" "nq" "bioasq")
fi

LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "SEP QA Pipeline"
echo "=========================================="
echo "Datasets:   ${DATASETS[*]}"
echo "Python:     $PYTHON"
echo "Start time: $(date)"
echo "=========================================="

# ---- Stage 1: Generation ----
echo ""
echo "=== STAGE 1: Generation ==="
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] Starting generation for $ds..."
    $PYTHON run_qa_generation.py --dataset "$ds" 2>&1 | tee "$LOG_DIR/gen_${ds}.log"
    echo "[$(date)] Finished generation for $ds."
    echo ""
done

# ---- Stage 2: NLI Labels ----
echo "=== STAGE 2: NLI Labels ==="
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] Starting NLI for $ds..."
    $PYTHON compute_nli_labels.py --dataset "$ds" 2>&1 | tee "$LOG_DIR/nli_${ds}.log"
    echo "[$(date)] Finished NLI for $ds."
    echo ""
done

# ---- Stage 3: Feature Extraction ----
echo "=== STAGE 3: Feature Extraction ==="
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] Extracting features for $ds..."
    $PYTHON extract_all_layers.py --dataset "$ds" 2>&1 | tee "$LOG_DIR/extract_${ds}.log"
    echo "[$(date)] Finished extraction for $ds."
done

# ---- Stage 4: ID Evaluation ----
echo ""
echo "=== STAGE 4: In-Distribution Evaluation ==="
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] Training ID probe for $ds..."
    $PYTHON train_probe.py --mode id --dataset "$ds" 2>&1 | tee "$LOG_DIR/probe_id_${ds}.log"
    echo ""
done

# ---- Stage 5: Cross-Dataset OOD Matrix ----
echo "=== STAGE 5: Cross-Dataset AUROC Matrix ==="
echo "[$(date)] Computing cross-dataset AUROC matrix..."
$PYTHON train_probe.py --mode matrix 2>&1 | tee "$LOG_DIR/probe_matrix.log"

echo ""
echo "=========================================="
echo "Pipeline complete at $(date)"
echo "=========================================="
