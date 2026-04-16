#!/bin/bash
#SBATCH --job-name=pick_place
#SBATCH --output=slurm_outputs/pick_place_%j.log
#SBATCH --error=slurm_outputs/pick_place_%j.err
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --partition=all


set -e

MOLMO_DIR="/n/fs/robot-data/molmospaces"
LAP_DIR="/n/fs/robot-data/language-action-pretraining"
CONDA_BASE="/n/fs/robot-data/miniconda3"

echo "[$(date)] Starting eval_lap_pick_place job on $(hostname)"

# ── 1. Launch policy server in background ──────────────────────────────────
echo "[$(date)] Starting policy server..."
cd "$LAP_DIR"
JAX_PLATFORMS=cuda uv run --group cuda scripts/serve_policy.py --env=LAP &
SERVER_PID=$!
echo "Policy server PID: $SERVER_PID"

# ── 2. Wait for server to be ready on port 8000 ───────────────────────────
echo "[$(date)] Waiting for policy server to be ready..."
MAX_WAIT=2000
ELAPSED=0
until nc -z localhost 8000 2>/dev/null; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Policy server did not start within ${MAX_WAIT}s"
        kill "$SERVER_PID" 2>/dev/null
        exit 1
    fi
done
echo "[$(date)] Policy server is ready (waited ${ELAPSED}s)"

# ── 3. Run evaluation ──────────────────────────────────────────────────────
echo "[$(date)] Starting evaluation..."
cd "$MOLMO_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mlspaces


MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python molmo_spaces/evaluation/eval_main.py \
    molmo_spaces.evaluation.configs.evaluation_configs:LAPPolicyEvalConfig \
    --benchmark_dir assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark \
    --num_workers 5 \
    --output_dir "eval_output/LAPPolicyEvalConfig/20260413_092728" \
    --resume
EVAL_EXIT=$?

# ── 4. Cleanup ─────────────────────────────────────────────────────────────
echo "[$(date)] Evaluation finished (exit code: $EVAL_EXIT). Shutting down policy server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo "[$(date)] Done."
exit $EVAL_EXIT
