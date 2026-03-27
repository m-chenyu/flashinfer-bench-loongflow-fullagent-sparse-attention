SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

cd "$SCRIPT_DIR" || exit 1

echo "[LoongFlow] Starting contest task"

python ../../../math_evolve_agent.py \
    --config task_config.yaml \
    --task-file task_prompt.txt \
    --initial-file dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json \
    --eval-file eval_program_modal.py \
    --log-level INFO
