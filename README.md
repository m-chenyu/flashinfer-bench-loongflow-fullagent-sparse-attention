# Sparse Attention Submission V2

This repository is prepared as a FlashInfer contest submission for
`dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`.

The current submission payload is centered on a CUDA implementation exported
through the PyTorch extension path:

- solution name: `ML Team-dsa-sparse-attn-v2`
- build language: `cuda`
- binding backend: `torch`
- entry point: `kernel.cu::dsa_forward`
- intended submission tag: `submission-v2`

## Submission Files

The files that matter for the current submission are:

- `config.toml`
  Declares the solution metadata and build configuration used by evaluation.
- `solution/cuda/kernel.cu`
  Main CUDA implementation and exported `dsa_forward` symbol.
- `solution/cuda/binding.py`
  Auxiliary Python-side binding metadata included with the packed sources.
- `scripts/pack_solution.py`
  Packs the solution directory into `solution.json` for local validation.

The previous Python solution path is no longer the active submission target.
This repo now treats the CUDA implementation as the source of truth.

## Local Packaging

To pack the current solution into `solution.json`:

```bash
python3 scripts/pack_solution.py
```

This reads `config.toml`, collects the files under `solution/cuda/`, and
produces a packed solution JSON at the repository root.

## Tagging Workflow

For contest submission, create a git commit containing the CUDA submission
files, then create and push a submission tag such as:

```bash
python3 /Users/machenyu01/.codex/skills/flashinfer-submission-tagger/scripts/tag_submission.py \
  --repo /Users/machenyu01/Downloads/mlsys/flashinfer-bench-loongflow-fullagent-sparse-attention \
  --tag submission-v2 --create --push
```

The evaluator consumes the tagged commit by checking out the tag and reading
`config.toml`.

## Repository Context

The repository still contains the supporting LoongFlow full-agent assets and
historic artifacts used during autonomous search. They remain useful for
reproducing context, but the active submission is the CUDA code under
`solution/cuda/`.
