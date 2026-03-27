"""Triton Kernel Optimization Evaluator for LoongFlow - Uses Remote Evaluation Service"""

import os
import sys
import time
import traceback
import requests
import uuid as uuid_lib
from typing import Dict, Any, Optional

# Base code template - PyTorch reference implementation for the task
# Task: Diagonal matrix multiplication C = diag(A) * B
BASE_CODE = '''

YOUR BASE CODE HERE

'''


class TritonKernelEvaluator:
    """Triton Kernel Evaluator using remote evaluation service"""

    def __init__(
        self,
        base_code: str = BASE_CODE,
        server_host: str = "http://10.79.130.248:8995",
        build_dir: str = "/tmp/triton_eval_cache",
        timeout: int = 600,
        gpu_type: str = "",
        perf_trials: int = 100,
        seed: int = 42,
    ):
        """
        Initialize Triton Kernel Evaluator with remote service.

        Args:
            base_code: PyTorch reference implementation (Model class + get_inputs + get_init_inputs)
            server_host: Remote evaluation server address
            build_dir: Compilation cache directory
            timeout: Request timeout (seconds)
            gpu_type: Target GPU type (e.g. "h20", "b200", "" for auto-detect)
            perf_trials: Number of performance measurement trials
            seed: Random seed for reproducibility
        """
        self.base_code = base_code
        self.server_host = server_host.rstrip("/")
        self.build_dir = build_dir
        self.timeout = timeout
        self.gpu_type = gpu_type
        self.perf_trials = perf_trials
        self.seed = seed

        # Available Triton endpoints
        self.endpoints = {
            "eval":        "/eval_triton",        # Full evaluation (correctness + performance)
            "compile":     "/compile_triton",     # Compile only
            "run":         "/run_triton",         # Compile + run
            "correct":     "/correct_triton",     # Compile + run + correctness
            "performance": "/performance_triton", # Compile + run + performance
        }

        self._check_health()

    def _check_health(self) -> bool:
        """Check if evaluation service is available"""
        try:
            resp = requests.get(f"{self.server_host}/health", timeout=10)
            if resp.status_code == 200:
                print(f"✓ Triton evaluation service healthy: {self.server_host}")
                return True
            else:
                print(f"⚠ Evaluation service returned status {resp.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot connect to evaluation service {self.server_host}: {e}")
            return False

    def _send_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send evaluation request to remote server.

        Args:
            endpoint: Logical endpoint name (key in self.endpoints)
            payload: JSON request payload

        Returns:
            Server response as a dict
        """
        if endpoint not in self.endpoints:
            raise ValueError(f"Unsupported endpoint: {endpoint}. "
                             f"Available: {list(self.endpoints.keys())}")

        url = f"{self.server_host}{self.endpoints[endpoint]}"

        try:
            start_time = time.time()
            resp = requests.post(url, json=payload, timeout=self.timeout)
            elapsed = time.time() - start_time

            try:
                data = resp.json()
            except Exception as e:
                return {
                    "success": False,
                    "status_code": resp.status_code,
                    "error": f"Cannot parse JSON response: {e}",
                    "result": None,
                    "client_time": elapsed,
                }

            if not isinstance(data, dict):
                return {
                    "success": False,
                    "status_code": resp.status_code,
                    "error": f"Invalid response type: expected dict, got {type(data)}",
                    "result": None,
                    "client_time": elapsed,
                }

            data["client_time"] = elapsed
            return data

        except requests.exceptions.Timeout:
            print(f"✗ Request timeout ({self.timeout}s): {url}")
            return {
                "success": False,
                "error": f"Request timeout ({self.timeout}s)",
                "result": None,
            }
        except requests.exceptions.RequestException as e:
            print(f"✗ Request error: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": None,
            }

    def eval(
        self,
        kernel_code: str,
        base_code: Optional[str] = None,
        correct_mode: str = "normal",
        solution_class_name: str = "ModelNew",
        base_class_name: str = "Model",
        level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full Triton kernel evaluation: compile + correctness + performance.

        Args:
            kernel_code: Triton kernel source code (must define ModelNew)
            base_code: Optional override for base/reference code
            correct_mode: Correctness mode ("normal", "biased", "contest")
            solution_class_name: Name of the solution class in kernel_code
            base_class_name: Name of the reference class in base_code
            level: Optional task level identifier for cache namespacing

        Returns:
            Server response dict with 'success', 'result', 'server_time', 'client_time'
        """
        task_uuid = str(uuid_lib.uuid4())
        if level is None:
            level = f"triton_eval_{task_uuid[:8]}"

        effective_base_code = base_code if base_code is not None else self.base_code

        payload = {
            "solution_str":        kernel_code,
            "base_str":            effective_base_code,
            "build_dir":           self.build_dir,
            "solution_class_name": solution_class_name,
            "base_class_name":     base_class_name,
            "correct_mode":        correct_mode,
            "perf_trials":         self.perf_trials,
            "seed":                self.seed,
            "num_gpus":            1,
        }

        if self.gpu_type:
            payload["gpu_type"] = self.gpu_type

        print(f"→ Evaluating Triton kernel (level={level}, uuid={task_uuid[:8]}...)")
        result = self._send_request("eval", payload)

        if result.get("success"):
            print(f"✓ Triton evaluation successful (level={level})")
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"✗ Triton evaluation failed (level={level}): {error_msg}")

        return result


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Evaluate a Triton kernel program using the remote evaluation service.
    Conforms to the LoongFlow evaluator interface.

    Args:
        program_path: Path to the Python file containing ModelNew (Triton implementation)

    Returns:
        dict with keys: 'status', 'summary', 'score', 'metrics', 'artifacts'
            status  : 'success' | 'validation_failed' | 'execution_failed'
            summary : Human-readable result description
            score   : float, speedup * 0.1 (correct+compiled), else 0.0
            metrics : dict with correctness, speedup, timing, etc.
            artifacts: dict with raw details, errors, backend info
    """
    timeout_duration = 600  # 10 minutes

    try:
        print(f"Starting Triton remote evaluation: {program_path}")
        start_time = time.time()

        with open(program_path, "r") as f:
            kernel_code = f.read()

        print("\nInitializing Triton remote evaluator...")
        evaluator = TritonKernelEvaluator(
            base_code=BASE_CODE,
            server_host="http://10.79.130.248:8995",
            build_dir="/tmp/triton_eval_cache",
            timeout=timeout_duration,
            gpu_type="",
            perf_trials=100,
            seed=42,
        )

        print("Sending Triton evaluation request to remote server...")
        result = evaluator.eval(
            kernel_code=kernel_code,
            base_code=BASE_CODE,
            correct_mode="normal",
            solution_class_name="ModelNew",
            base_class_name="Model",
        )

        end_time = time.time()
        eval_time = end_time - start_time

        # ------------------------------------------------------------------ #
        # Handle top-level request failure                                     #
        # ------------------------------------------------------------------ #
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error")

            if "timeout" in error_msg.lower():
                return {
                    "status": "execution_failed",
                    "summary": f"Triton remote evaluation timed out: {error_msg}",
                    "score": 0.0,
                    "metrics": {
                        "correctness": 0.0,
                        "speedup": 0.0,
                        "eval_time": float(eval_time),
                    },
                    "artifacts": {
                        "failure_stage": "remote_timeout",
                        "timeout_duration": f"{timeout_duration}s",
                        "error": error_msg,
                        "backend": "triton",
                    },
                }

            return {
                "status": "execution_failed",
                "summary": f"Triton remote kernel evaluation failed: {error_msg}",
                "score": 0.0,
                "metrics": {
                    "correctness": 0.0,
                    "speedup": 0.0,
                    "eval_time": float(eval_time),
                },
                "artifacts": {
                    "error": error_msg,
                    "failure_stage": "remote_evaluation",
                    "raw_result": str(result),
                    "backend": "triton",
                },
            }

        # ------------------------------------------------------------------ #
        # Parse inner result                                                   #
        # ------------------------------------------------------------------ #
        result_data = result.get("result", {})

        # Performance metrics
        performance     = result_data.get("performance", {})
        exec_time_ms    = performance.get("solution_runtime_ms", 0.0)
        baseline_time_ms = performance.get("base_runtime_ms", None)
        speedup         = performance.get("speedup", None)

        # Correctness
        correct_info = result_data.get("correct", {})
        is_correct   = correct_info.get("status", "unknown") == "success"

        # Compile
        compile_info    = result_data.get("compile", {})
        compile_success = compile_info.get("status", "unknown") == "success"
        compile_time    = compile_info.get("compile_time", 0.0)

        # ------------------------------------------------------------------ #
        # Scoring: speedup * 0.1 when correct; 0 otherwise                    #
        # (matches CUDA and TileLang evaluators)                               #
        # ------------------------------------------------------------------ #
        correctness_score = 1.0 if (is_correct and compile_success) else 0.0

        if speedup is not None and speedup > 0 and correctness_score > 0:
            combined_score = float(speedup) * 0.1
        else:
            combined_score = 0.0

        # ------------------------------------------------------------------ #
        # Status & summary                                                     #
        # ------------------------------------------------------------------ #
        if not compile_success:
            status  = "execution_failed"
            compile_error = compile_info.get("error_msg", "Unknown compilation error")
            summary = f"Triton compilation failed: {compile_error}"
        elif not is_correct:
            status  = "validation_failed"
            correct_msg = correct_info.get("error_msg", "Correctness check failed")
            summary = f"Triton correctness check failed: {correct_msg}"
        elif speedup is None or baseline_time_ms is None:
            status  = "success"
            summary = (
                f"Triton kernel compiled and executed successfully. "
                f"Runtime: {exec_time_ms:.3f} ms"
            )
        elif speedup >= 1.0:
            status  = "success"
            summary = (
                f"Triton optimization successful! Speedup: {speedup:.2f}x "
                f"(baseline: {baseline_time_ms:.3f}ms, optimized: {exec_time_ms:.3f}ms)"
            )
        else:
            status  = "success"
            summary = (
                f"Triton kernel runs but slower than baseline. "
                f"Speedup: {speedup:.2f}x "
                f"(baseline: {baseline_time_ms:.3f}ms, optimized: {exec_time_ms:.3f}ms)"
            )

        print(
            f"Triton evaluation completed: correct={is_correct}, "
            f"compile={compile_success}, speedup={speedup}, score={combined_score:.4f}"
        )

        # ------------------------------------------------------------------ #
        # Artifacts                                                            #
        # ------------------------------------------------------------------ #
        artifacts = {
            "backend":          "triton",
            "execution_time":   f"{eval_time:.2f}s",
            "kernel_time_ms":   f"{exec_time_ms:.3f}ms",
            "compile_time":     f"{compile_time:.2f}s",
            "server_time":      f"{result.get('server_time', 0.0):.2f}s",
            "client_time":      f"{result.get('client_time', 0.0):.2f}s",
        }

        if baseline_time_ms is not None:
            artifacts["baseline_time_ms"] = f"{baseline_time_ms:.3f}ms"
        if speedup is not None:
            artifacts["speedup"] = f"{speedup:.2f}x"

        if is_correct and compile_success:
            artifacts["correctness_check"] = "Passed"
        else:
            if not compile_success:
                artifacts["compilation_error"] = compile_info.get("error_msg", "Unknown")
            if not is_correct:
                artifacts["correctness_error"] = correct_info.get("error_msg", "Unknown")

        if speedup and speedup > 1.5:
            artifacts["stdout"] = (
                f"Excellent Triton optimization! {speedup:.1f}x faster than baseline"
            )

        return {
            "status":  status,
            "summary": summary,
            "score":   float(combined_score),
            "metrics": {
                "correctness":      float(correctness_score),
                "speedup":          float(speedup) if speedup is not None else 0.0,
                "exec_time_ms":     float(exec_time_ms),
                "baseline_time_ms": float(baseline_time_ms) if baseline_time_ms is not None else 0.0,
                "compile_time":     float(compile_time),
                "eval_time":        float(eval_time),
            },
            "artifacts": artifacts,
        }

    except Exception as e:
        error_msg = f"Triton evaluation failed: {str(e)}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc()

        return {
            "status":  "execution_failed",
            "summary": error_msg,
            "score":   0.0,
            "metrics": {
                "correctness": 0.0,
                "speedup":     0.0,
                "eval_time":   0.0,
            },
            "artifacts": {
                "error_type":    type(e).__name__,
                "traceback":     traceback.format_exc(),
                "failure_stage": "evaluation_framework",
                "backend":       "triton",
            },
        }


if __name__ == "__main__":
    test_file = "./test.py"
    if os.path.exists(test_file):
        result = evaluate(test_file)
        print("\n=== Triton Evaluation Result ===")
        print(f"Status:   {result['status']}")
        print(f"Summary:  {result['summary']}")
        print(f"Score:    {result['score']:.4f}")
        print(f"Metrics:  {result['metrics']}")
        print(f"Artifacts:{result['artifacts']}")
    else:
        print(f"Test file not found: {test_file}")
