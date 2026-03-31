"""
TileLang Kernel Optimization Evaluator for LoongFlow
Evaluates generated TileLang DSL kernels using remote evaluation service
"""

import os
import sys
import time
import traceback
import requests
import uuid as uuid_lib
from typing import Dict, Any, Optional

# Base code template - embedded directly to avoid path issues when evaluator is copied
BASE_CODE = '''

YOUR BASE CODE HERE

'''


class TileLangKernelEvaluator:
    """TileLang Kernel Evaluator using remote evaluation service"""
    
    def __init__(
        self,
        base_code: str = BASE_CODE,
        server_host: str = "http://10.79.130.248:8995",
        build_dir: str = "/tmp/tilelang_eval_cache",
        timeout: int = 600,
        gpu_arch: str = "9.0",
        num_compile_cpus: int = 2,
    ):
        """
        Initialize TileLang Kernel Evaluator with remote service
        
        Args:
            base_code: Base code template (PyTorch reference implementation)
            server_host: Remote evaluation server address
            build_dir: Compilation cache directory
            timeout: Request timeout (seconds)
            gpu_arch: GPU architecture version
            num_compile_cpus: Number of CPU cores for compilation
        """
        self.base_code = base_code
        self.server_host = server_host.rstrip('/')
        self.build_dir = build_dir
        self.timeout = timeout
        self.gpu_arch = gpu_arch
        self.num_compile_cpus = num_compile_cpus
        
        # Available TileLang endpoints
        self.endpoints = {
            'eval': '/eval_tilelang',           # Full evaluation (compile+run+correctness+performance)
            'compile': '/compile_tilelang',     # Compile only
            'correct': '/correct_tilelang',     # Compile+run+correctness
        }
        
        # Check service health
        self._check_health()
    
    def _check_health(self) -> bool:
        """Check if evaluation service is available"""
        try:
            health_url = f"{self.server_host}/health"
            resp = requests.get(health_url, timeout=10)
            if resp.status_code == 200:
                print(f"✓ TileLang evaluation service is healthy: {health_url}")
                return True
            else:
                print(f"⚠ Evaluation service returned status {resp.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot connect to evaluation service {self.server_host}: {e}")
            return False
    
    def _send_request(
        self,
        endpoint: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send evaluation request to remote server
        
        Args:
            endpoint: API endpoint name
            payload: Request payload
            
        Returns:
            Server response result
        """
        if endpoint not in self.endpoints:
            raise ValueError(f"Unsupported endpoint: {endpoint}")
        
        url = f"{self.server_host}{self.endpoints[endpoint]}"
        
        try:
            start_time = time.time()
            resp = requests.post(url, json=payload, timeout=self.timeout)
            elapsed = time.time() - start_time
            
            # Parse response
            try:
                data = resp.json()
            except Exception as e:
                return {
                    "success": False,
                    "status_code": resp.status_code,
                    "error": f"Cannot parse JSON response: {e}",
                    "result": None,
                    "server_time": None,
                    "client_time": elapsed,
                }
            
            # Ensure dictionary format
            if not isinstance(data, dict):
                return {
                    "success": False,
                    "status_code": resp.status_code,
                    "error": f"Invalid response type: expected dict, got {type(data)}",
                    "result": None,
                    "server_time": None,
                    "client_time": elapsed,
                }
            
            # Add client timing
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
        correct_mode: Optional[str] = None,
        solution_parse_mode: str = "raw",
        level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full evaluation of TileLang kernel (compile+run+correctness+performance)
        
        Args:
            kernel_code: TileLang DSL code to evaluate
            base_code: Base code (optional, uses init base_code if not provided)
            correct_mode: Correctness check mode
            solution_parse_mode: Parse mode (raw for TileLang)
            level: Evaluation level identifier
            
        Returns:
            Evaluation result dictionary
        """
        # Generate unique identifier
        task_uuid = str(uuid_lib.uuid4())
        if level is None:
            level = f"tilelang_eval_{task_uuid[:8]}"
        
        # Use provided base_code or default self.base_code
        effective_base_code = base_code if base_code is not None else self.base_code
        
        # Build request payload for TileLang
        payload = {
            "solution_str": kernel_code,
            "base_str": effective_base_code,
            "build_dir": self.build_dir,
            "level": level,
            "uuid": task_uuid,
            "solution_parse_mode": solution_parse_mode,
            "gpu_arch": self.gpu_arch,
            "num_compile_cpus": self.num_compile_cpus,
            "solution_class_name": "ModelNew",  # TileLang solution class
            "base_class_name": "Model",         # PyTorch reference class
        }
        
        # Add correctness check mode
        if correct_mode:
            payload["correct_mode"] = correct_mode
        
        # Send request
        print(f"→ Evaluating TileLang kernel (level={level}, uuid={task_uuid[:8]}...)")
        result = self._send_request("eval", payload)
        
        # Parse result
        if result.get("success"):
            print(f"✓ TileLang evaluation successful (level={level})")
            return result
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"✗ TileLang evaluation failed (level={level}): {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "exec_time": None,
                "compile_time": None,
                "is_correct": False,
                "performance": {},
                "raw_result": result,
                "uuid": task_uuid,
            }


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Evaluate the TileLang kernel program using remote evaluation service
    
    Args:
        program_path: Path to the program file containing ModelNew (TileLang implementation)
        
    Returns:
        A dictionary with 'status', 'summary', 'score', 'metrics', and 'artifacts'
        conforming to LoongFlow evaluation interface
    """
    timeout_duration = 600  # 10 minutes timeout
    
    try:
        print(f"Starting TileLang remote evaluation of: {program_path}")
        start_time = time.time()
        
        # Read the TileLang kernel code from file
        with open(program_path, 'r') as f:
            kernel_code = f.read()
        
        # Note: For TileLang, we skip the PyTorch forbidden function check
        # because TileLang DSL uses different APIs and may include PyTorch imports
        print("Note: TileLang evaluation - skipping PyTorch forbidden function check")
        
        # Initialize remote TileLang evaluator
        print("\nInitializing TileLang remote evaluator...")
        evaluator = TileLangKernelEvaluator(
            base_code=BASE_CODE,
            server_host="http://10.79.130.248:8995",
            build_dir="/tmp/tilelang_eval_cache",
            timeout=timeout_duration,
            gpu_arch="9.0",
            num_compile_cpus=2,
        )
        
        # Evaluate TileLang kernel using remote service
        print("Sending TileLang evaluation request to remote server...")
        result = evaluator.eval(
            kernel_code=kernel_code,
            base_code=BASE_CODE,
            solution_parse_mode="raw",
        )
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        # Check if evaluation was successful
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown error')
            
            # Check for timeout
            if 'timeout' in error_msg.lower():
                return {
                    "status": "execution_failed",
                    "summary": f"TileLang remote evaluation timed out: {error_msg}",
                    "score": 0.0,
                    "metrics": {
                        "correctness": 0.0,
                        "speedup": 0.0,
                        "eval_time": float(eval_time)
                    },
                    "artifacts": {
                        "failure_stage": "remote_timeout",
                        "timeout_duration": f"{timeout_duration}s",
                        "error": error_msg,
                        "backend": "tilelang"
                    }
                }
            
            return {
                "status": "execution_failed",
                "summary": f"TileLang remote kernel evaluation failed: {error_msg}",
                "score": 0.0,
                "metrics": {
                    "correctness": 0.0,
                    "speedup": 0.0,
                    "eval_time": float(eval_time)
                },
                "artifacts": {
                    "error": error_msg,
                    "failure_stage": "remote_evaluation",
                    "raw_result": str(result),
                    "backend": "tilelang"
                }
            }
        
        # Extract result data
        result_data = result.get('result', {})
        
        # Extract performance metrics
        performance = result_data.get('performance', {})
        exec_time_ms = performance.get('solution_runtime_ms', 0.0)
        baseline_time_ms = performance.get('base_runtime_ms', None)
        speedup = performance.get('speedup', None)
        
        # Extract correctness info
        correct_info = result_data.get('correct', {})
        is_correct = correct_info.get('status', 'unknown') == 'success'
        
        # Extract compile info
        compile_info = result_data.get('compile', {})
        compile_time = compile_info.get('compile_time', 0.0)
        compile_success = compile_info.get('status', 'unknown') == 'success'
        
        # Calculate score
        # Score = speedup * 0.1 (if correct and compiled successfully)
        # This ensures score stays within [0, 1] range for framework compatibility
        correctness_score = 1.0 if (is_correct and compile_success) else 0.0
        
        if speedup is not None and speedup > 0 and correctness_score > 0:
            # Use speedup * 0.1 as score when correct
            # e.g., 2x speedup = 0.2 score, 10x speedup = 1.0 score
            combined_score = float(speedup) * 0.1
        else:
            # If incorrect, not compiled, or no speedup available
            combined_score = 0.0
        
        # Prepare status and summary
        if not compile_success:
            status = "execution_failed"
            compile_error = compile_info.get('error', 'Unknown compilation error')
            summary = f"TileLang compilation failed: {compile_error}"
        elif not is_correct:
            status = "validation_failed"
            correct_msg = correct_info.get('message', 'Correctness check failed')
            summary = f"TileLang correctness check failed: {correct_msg}"
        elif speedup is None or baseline_time_ms is None:
            status = "success"
            summary = f"TileLang kernel compiled and executed successfully. Runtime: {exec_time_ms:.3f} ms"
        elif speedup >= 1.0:
            status = "success"
            summary = (
                f"TileLang optimization successful! Speedup: {speedup:.2f}x "
                f"(baseline: {baseline_time_ms:.3f}ms, optimized: {exec_time_ms:.3f}ms)"
            )
        else:
            status = "success"
            summary = f"TileLang kernel runs but slower than baseline. Speedup: {speedup:.2f}x"
        
        print(
            f"TileLang evaluation completed: correct={is_correct}, "
            f"compile={compile_success}, speedup={speedup}, score={combined_score:.4f}"
        )
        
        # Prepare artifacts
        artifacts = {
            "backend": "tilelang",
            "execution_time": f"{eval_time:.2f}s",
            "kernel_time_ms": f"{exec_time_ms:.3f}ms",
            "compile_time": f"{compile_time:.2f}s"
        }
        
        if baseline_time_ms is not None:
            artifacts["baseline_time_ms"] = f"{baseline_time_ms:.3f}ms"
        
        if speedup is not None:
            artifacts["speedup"] = f"{speedup:.2f}x"
        
        if is_correct and compile_success:
            artifacts["correctness_check"] = "Passed"
        else:
            if not compile_success:
                artifacts["compilation_error"] = compile_info.get('error', 'Unknown')
            if not is_correct:
                artifacts["correctness_error"] = correct_info.get('message', 'Unknown')
        
        if speedup and speedup > 1.5:
            artifacts["stdout"] = f"Excellent TileLang optimization! {speedup:.1f}x faster than baseline"
        
        # Add server metrics
        artifacts["server_time"] = f"{result.get('server_time', 0.0):.2f}s"
        artifacts["client_time"] = f"{result.get('client_time', 0.0):.2f}s"
        
        return {
            "status": status,
            "summary": summary,
            "score": float(combined_score),
            "metrics": {
                "correctness": float(correctness_score),
                "speedup": float(speedup) if speedup is not None else 0.0,
                "exec_time_ms": float(exec_time_ms),
                "baseline_time_ms": float(baseline_time_ms) if baseline_time_ms is not None else 0.0,
                "compile_time": float(compile_time),
                "eval_time": float(eval_time)
            },
            "artifacts": artifacts
        }
    
    except Exception as e:
        error_msg = f"TileLang evaluation failed: {str(e)}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc()
        
        return {
            "status": "execution_failed",
            "summary": error_msg,
            "score": 0.0,
            "metrics": {
                "correctness": 0.0,
                "speedup": 0.0,
                "eval_time": 0.0
            },
            "artifacts": {
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "failure_stage": "evaluation_framework",
                "backend": "tilelang"
            }
        }


if __name__ == "__main__":
    # Test with test.py (should contain TileLang DSL implementation)
    test_file = "./test.py"
    if os.path.exists(test_file):
        result = evaluate(test_file)
        print("\n=== TileLang Evaluation Result ===")
        print(f"Status: {result['status']}")
        print(f"Summary: {result['summary']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Metrics: {result['metrics']}")
        print(f"Artifacts: {result['artifacts']}")
    else:
        print(f"Test file not found: {test_file}")