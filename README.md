## LoongFlow Full-Agent Bundle for Sparse Attention

This bundle is a trimmed submission package for the `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` task.

It keeps three things only:

1. The core LoongFlow full-agent code needed to understand and run the math evolution agent.
2. The task-specific sparse-attention prompt, config, evaluator, and initial JSON definition.
3. The final autonomous iteration-20 artifacts for the three competing models:
   - `gpt-5.4`
   - `claude-sonnet-4-6`
   - `gemini-3.1-pro-preview`

### What Was Removed

- `.venv`
- caches and `__pycache__`
- unrelated task examples
- unrelated FlashInfer tasks (`gdn_decode`, `gdn_prefill`, `moe`)
- bulky raw output directories
- temporary logs and repeated intermediate executor files
- embedded API keys

### Important Note

This bundle is organized around the final autonomous stage that produced the iteration-20 result artifacts.
The original run resumed from an earlier checkpoint and then continued autonomously to iteration 20.
To keep the package small, this bundle preserves the final iteration-20 artifacts rather than the full raw output tree.

If you need a fully replayable resume package from the exact prior checkpoint, you should additionally archive the corresponding upstream checkpoint/output tree from the original project.

### Key Paths

- Agent entrypoint:
  `agents/math_agent/math_evolve_agent.py`
- Sparse-attention task assets:
  `agents/math_agent/cuda_task/flashinfer_mlsys26/sparse_attention/`
- Final iteration-20 artifacts:
  `artifacts/iter20/`

### Best Scores in This Bundle

- `gpt-5.4`: `43.98x`
- `claude-sonnet-4-6`: `39.33x`
- `gemini-3.1-pro-preview`: `39.87x`

### Config Handling

The copied task config has been sanitized. Fill in your own API endpoint and key before running locally.

Recommended approach:

1. Copy `task_config.yaml` to a local untracked file if needed.
2. Fill in your own credentials.
3. Do not commit the filled credentials file.
