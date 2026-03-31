"""
Modal remote GPU evaluation service for LoongFlow using flashinfer-bench.

Deploy once:
    python -m modal deploy modal_eval_deploy.py

Ensure dataset is uploaded first (see ensure_dataset_synced or manual upload).
"""

import os

import modal

MODAL_APP_NAME = "loongflow-cuda-eval"
VOLUME_NAME = "flashinfer-trace"
DATASET_PATH = "/data"

app = modal.App(MODAL_APP_NAME)

image = modal.Image.from_registry(
    "nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04", add_python="3.12"
).pip_install("flashinfer-bench", "torch", "triton", "pydantic")

dataset_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="B200",
    volumes={DATASET_PATH: dataset_vol},
    timeout=900,
    serialized=True,
)
def remote_eval_kernel(
    kernel_code: str,
    task_id: str,
    dataset_name: str = "mlsys26-contest",
    backend: str = "cuda",
    timeout: int = 30,
) -> dict:
    """
    Evaluate a kernel on Modal GPU using flashinfer-bench.

    Args:
        kernel_code: Source code of the kernel (must define a `run()` function).
        task_id: Definition/problem name (e.g. "gdn_decode_qk4_v8_d128_k_last").
        dataset_name: Dataset subdirectory name in the Modal Volume.
        backend: "cuda".
        timeout: Timeout in seconds per solution evaluation.

    Returns:
        dict with compiled, correct, speedup, latency_ms, task_id, stats, error.
    """
    import uuid
    import tempfile

    from flashinfer_bench.bench import Benchmark, BenchmarkConfig
    from flashinfer_bench.data import (
        BuildSpec,
        EvaluationStatus,
        Solution,
        SourceFile,
        SupportedLanguages,
        TraceSet,
    )

    # Use a unique torch extensions cache dir per invocation to avoid
    # concurrent load_inline calls conflicting on shared cache paths.
    unique_ext_dir = tempfile.mkdtemp(prefix="torch_ext_")
    os.environ["TORCH_EXTENSIONS_DIR"] = unique_ext_dir

    dataset_root = os.path.join(DATASET_PATH, dataset_name)
    trace_set = TraceSet.from_path(dataset_root)

    solution_name = f"agent_{uuid.uuid4().hex[:8]}"
    # Always use PYTHON: LLM-generated code is Python that wraps
    # CUDA (load_inline). PythonBuilder imports the module and
    # compilation happens at runtime via nvcc.
    language = SupportedLanguages.PYTHON

    solution = Solution(
        name=solution_name,
        definition=task_id,
        author="agent",
        spec=BuildSpec(
            language=language,
            target_hardware=["cuda"],
            entry_point="main.py::run",
            dependencies=[],
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content=kernel_code)],
    )

    trace_set.solutions.setdefault(task_id, []).append(solution)
    trace_set._solution_by_name[solution_name] = solution

    config = BenchmarkConfig(
        warmup_runs=3,
        iterations=5,
        num_trials=1,
        definitions=[task_id],
        solutions=[solution_name],
        timeout_seconds=timeout,
    )

    benchmark = Benchmark(trace_set, config)
    try:
        result_ts = benchmark.run_all(dump_traces=False)
    finally:
        benchmark.close()
        # Clean up the per-invocation torch extensions cache
        import shutil
        shutil.rmtree(unique_ext_dir, ignore_errors=True)

    traces = result_ts.traces.get(task_id, [])

    error_statuses = {
        EvaluationStatus.COMPILE_ERROR,
        EvaluationStatus.RUNTIME_ERROR,
        EvaluationStatus.INCORRECT_SHAPE,
        EvaluationStatus.INCORRECT_NUMERICAL,
        EvaluationStatus.INCORRECT_DTYPE,
        EvaluationStatus.TIMEOUT,
    }
    for trace in traces:
        ev = trace.evaluation
        if ev and ev.status in error_statuses:
            return {
                "compiled": ev.status != EvaluationStatus.COMPILE_ERROR,
                "correct": False,
                "task_id": task_id,
                "error": f"{ev.status.value}: {ev.log}",
            }

    per_workload = []
    for trace in traces:
        ev = trace.evaluation
        if ev and ev.status == EvaluationStatus.PASSED:
            per_workload.append({
                "latency_ms": ev.performance.latency_ms,
                "reference_latency_ms": ev.performance.reference_latency_ms,
                "speedup": ev.performance.speedup_factor,
                "max_relative_error": ev.correctness.max_relative_error,
                "max_absolute_error": ev.correctness.max_absolute_error,
            })

    if not per_workload:
        return {"task_id": task_id, "error": "No evaluation results"}

    n = len(per_workload)
    result = {
        "compiled": True,
        "correct": True,
        "speedup": sum(w["speedup"] for w in per_workload) / n,
        "latency_ms": sum(w["latency_ms"] for w in per_workload) / n,
        "task_id": task_id,
        "stats": {
            "reference_latency_ms": sum(w["reference_latency_ms"] for w in per_workload) / n,
            "max_relative_error": max(w["max_relative_error"] for w in per_workload),
            "max_absolute_error": max(w["max_absolute_error"] for w in per_workload),
            "total_workloads": n,
        },
        "per_workload": per_workload,
    }

    return result


@app.local_entrypoint()
def upload_dataset(local_path: str, dataset_name: str = "mlsys26-contest"):
    """Upload local dataset to Modal Volume.

    Usage:
        python -m modal run modal_eval_deploy.py::upload_dataset \
            --local-path /path/to/datasets/mlsys26-contest \
            --dataset-name mlsys26-contest
    """
    import os

    if not os.path.isdir(local_path):
        raise FileNotFoundError(f"Not a directory: {local_path}")

    try:
        if dataset_vol.listdir(f"/{dataset_name}"):
            print(f"Dataset '{dataset_name}' already exists in Volume, skipping.")
            return
    except Exception:
        pass

    print(f"Uploading '{local_path}' -> Volume:/{dataset_name} ...")
    with dataset_vol.batch_upload() as batch:
        batch.put_directory(local_path, f"/{dataset_name}")
    print("Upload complete.")
