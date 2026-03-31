"""
LoongFlow-compatible CUDA Kernel Evaluator using Modal + flashinfer-bench.

Usage:
    1. Deploy Modal function (one-time):
       python -m modal deploy modal_eval_deploy.py

    2. Upload dataset (one-time):
       python -m modal run modal_eval_deploy.py::upload_dataset \
           --local-path /path/to/datasets/mlsys26-contest

    3. Use this file as --eval-file in LoongFlow:
       python ../../math_evolve_agent.py \
         --config task_config.yaml \
         --eval-file eval_program_modal.py \
         --initial-file task_definition.json \
         ...

Requires: modal package installed locally (pip install modal)
"""
import json
import os
import sys
import time
import traceback
from typing import Dict, Any

import modal

# ============================================================================
# CONFIGURATION - modify these per task
# ============================================================================

# Modal app and function name (must match modal_eval_deploy.py)
MODAL_APP_NAME = "loongflow-cuda-eval"
MODAL_FUNCTION_NAME = "remote_eval_kernel"

# Task ID - the problem definition name in the flashinfer-bench dataset
# This must match the "name" field in your JSON task definition
TASK_ID = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"

# Dataset name in the Modal Volume
DATASET_NAME = "mlsys26-contest"

# Backend hint (informational; flashinfer-bench always uses PythonBuilder
# since LLM code is Python that wraps CUDA via load_inline)
BACKEND = "cuda"

# Per-evaluation timeout (seconds) for the benchmark runner.
# Must be large enough for nvcc compilation (~30-60s) + actual kernel run.
BENCH_TIMEOUT = 300


# ============================================================================
# Core evaluate() function - called by LoongFlow framework
# ============================================================================

def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Evaluate a kernel via Modal remote GPU + flashinfer-bench.

    Conforms to the LoongFlow evaluate() interface:
    - Input: program_path (str) - path to file with kernel code (run() function)
    - Output: dict with {status, summary, score, metrics, artifacts}

    Also handles JSON task definitions gracefully (returns score=0.0).
    """
    start = time.time()

    try:
        with open(program_path, 'r') as f:
            content = f.read().strip()

        # If the file is a JSON task definition (not kernel code), skip evaluation
        if content.startswith('{'):
            try:
                task_def = json.loads(content)
                task_name = task_def.get('name', 'unknown')
                return {
                    "status": "success",
                    "summary": f"Task definition loaded: {task_name} (no kernel to evaluate)",
                    "score": 0.0,
                    "metrics": {
                        "correctness": 0.0,
                        "speedup": 0.0,
                        "exec_time_ms": 0.0,
                        "baseline_time_ms": 0.0,
                        "eval_time": time.time() - start,
                    },
                    "artifacts": {
                        "task_name": task_name,
                        "eval_backend": "modal (skipped - task definition)",
                    },
                }
            except json.JSONDecodeError:
                pass  # Not valid JSON, treat as kernel code

        kernel_code = content
        print(f"Starting Modal evaluation: {program_path}")
        print(f"  task_id={TASK_ID}, dataset={DATASET_NAME}, backend={BACKEND}")

        # Look up the deployed Modal function
        try:
            remote_fn = modal.Function.from_name(
                MODAL_APP_NAME, MODAL_FUNCTION_NAME
            )
        except Exception as e:
            return _make_error_result(
                f"Cannot find Modal function '{MODAL_APP_NAME}/{MODAL_FUNCTION_NAME}'. "
                f"Run 'python -m modal deploy modal_eval_deploy.py' first. Error: {e}",
                time.time() - start,
                failure_stage="modal_lookup",
            )

        # Call the remote evaluation function
        print("-> Sending kernel to Modal remote GPU...")
        try:
            result = remote_fn.remote(
                kernel_code=kernel_code,
                task_id=TASK_ID,
                dataset_name=DATASET_NAME,
                backend=BACKEND,
                timeout=BENCH_TIMEOUT,
            )
        except Exception as e:
            return _make_error_result(
                f"Modal remote call failed: {e}",
                time.time() - start,
                failure_stage="modal_remote_call",
            )

        eval_time = time.time() - start
        print(f"<- Modal returned in {eval_time:.2f}s")
        print(f"[Modal raw result] {json.dumps(result, indent=2, default=str)}")

        return _convert_result(result, eval_time)

    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc()
        return {
            "status": "execution_failed",
            "summary": error_msg,
            "score": 0.0,
            "metrics": {"correctness": 0.0, "speedup": 0.0, "eval_time": 0.0},
            "artifacts": {
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "failure_stage": "evaluation_framework",
            },
        }


# ============================================================================
# Result conversion
# ============================================================================

def _convert_result(modal_result: dict, eval_time: float) -> Dict[str, Any]:
    """Convert flashinfer-bench result dict to LoongFlow format."""
    compiled = modal_result.get("compiled", False)
    correct = modal_result.get("correct", False)
    speedup = modal_result.get("speedup", 0.0)
    latency_ms = modal_result.get("latency_ms", 0.0)
    task_id = modal_result.get("task_id", TASK_ID)
    error = modal_result.get("error")
    stats = modal_result.get("stats", {})

    ref_latency_ms = stats.get("reference_latency_ms", 0.0) if stats else 0.0
    correctness_score = 1.0 if (compiled and correct) else 0.0
    combined_score = float(speedup) * 0.1 if speedup and speedup > 0 and correctness_score > 0 else 0.0

    # Determine status and summary
    if error and not compiled:
        status = "execution_failed"
        summary = f"Compilation failed: {error}"
    elif error and not correct:
        status = "validation_failed"
        summary = f"Validation failed: {error}"
    elif error:
        status = "execution_failed"
        summary = f"Evaluation error: {error}"
    elif correct and speedup and speedup >= 1.0:
        status = "success"
        summary = (
            f"Optimization successful! Speedup: {speedup:.2f}x "
            f"(ref: {ref_latency_ms:.3f}ms, kernel: {latency_ms:.3f}ms)"
        )
    elif correct:
        status = "success"
        summary = f"Kernel correct but slower. Speedup: {speedup:.2f}x, Latency: {latency_ms:.3f}ms"
    else:
        status = "execution_failed"
        summary = "Unknown evaluation result"

    # Build artifacts
    artifacts = {
        "eval_backend": "modal+flashinfer-bench",
        "task_id": task_id,
        "execution_time": f"{eval_time:.2f}s",
    }
    if latency_ms:
        artifacts["kernel_latency_ms"] = f"{latency_ms:.3f}ms"
    if ref_latency_ms:
        artifacts["reference_latency_ms"] = f"{ref_latency_ms:.3f}ms"
    if speedup:
        artifacts["speedup"] = f"{speedup:.2f}x"
    if stats:
        artifacts["bench_stats"] = stats
    if compiled and correct:
        artifacts["correctness_check"] = "Passed"
    elif error:
        artifacts["error"] = error

    print(
        f"Evaluation complete: compiled={compiled}, correct={correct}, "
        f"speedup={speedup:.2f}x, score={combined_score:.4f}"
    )

    return {
        "status": status,
        "summary": summary,
        "score": float(combined_score),
        "metrics": {
            "correctness": float(correctness_score),
            "speedup": float(speedup) if speedup else 0.0,
            "exec_time_ms": float(latency_ms) if latency_ms else 0.0,
            "baseline_time_ms": float(ref_latency_ms),
            "compile_time": 0.0,
            "eval_time": float(eval_time),
        },
        "artifacts": artifacts,
    }


def _make_error_result(
    error: str, eval_time: float, failure_stage: str = "unknown"
) -> Dict[str, Any]:
    """Build a LoongFlow-compatible error result."""
    print(f"Error: {error}", file=sys.stderr)
    return {
        "status": "execution_failed",
        "summary": f"Evaluation failed: {error}",
        "score": 0.0,
        "metrics": {
            "correctness": 0.0,
            "speedup": 0.0,
            "exec_time_ms": 0.0,
            "baseline_time_ms": 0.0,
            "compile_time": 0.0,
            "eval_time": float(eval_time),
        },
        "artifacts": {
            "error": error,
            "failure_stage": failure_stage,
            "eval_backend": "modal+flashinfer-bench",
        },
    }


if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else "./test.py"
    if os.path.exists(test_file):
        result = evaluate(test_file)
        print(f"\nStatus: {result['status']}")
        print(f"Summary: {result['summary']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Metrics: {result['metrics']}")
    else:
        print(f"File not found: {test_file}")
