SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

cd "$SCRIPT_DIR" || exit 1

INITIAL_FILE="${INITIAL_FILE:-initial_program.py}"

echo "[LoongFlow] Starting with initial file: $INITIAL_FILE"

python ../math_evolve_agent.py \
  --config task_config.yaml \
  --task-file task_prompt_tilelang.txt \
  --initial-file "$INITIAL_FILE" \
  --eval-file eval_program_with_tilelang.py \
  --log-level INFO
