# OmniDocBench v1.6 reproducible runner

This directory contains an end-to-end, resumable runner for the **1,651-page
OmniDocBench v1.6 full set**. It uses host Ollama for batch-size-one VLM
inference and the pinned official Docker evaluator for `end2end` scoring.

The current workflow has a hard human approval gate:

1. `./run_smoke.sh` performs preflight, two-model warm-up/GPU checks, the same
   fixed 20-page inference for both models, official smoke evaluation, and
   smoke reporting.
2. It never calls the full runner. A valid smoke run remains
   `SMOKE_REQUIRES_AI_REVIEW` until all 40 outputs have been inspected and a
   signed-by-content `ai_review.json` plus `READY_FOR_FULL_RUN` manifest exist.
3. Only after reviewing that report should the operator manually run
   `./run_full.sh` inside tmux. Re-running the command resumes from EBS
   checkpoints.

No script creates tmux sessions, background jobs, schedules, or approval files.

Under the fixed-configuration comparison policy, model-originated outcomes such
as repetition, empty content, refusal, malformed Markdown, or
`done_reason=length` are preserved and scored. They remain visible as anomaly
flags but do not by themselves block the experiment. API/runtime failures,
wrong inputs or signatures, GPU fallback, missing predictions, and evaluator
failures remain blocking.

## Pinned inputs

- Dataset: `opendatalab/OmniDocBench` revision
  `d386947f7fc3bafdcd756c8485845a2f43a19875` (`add v1.6`)
- Full GT SHA-256:
  `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`
- Expected page/image count: 1,651
- Official evaluation code: last v1.6 main commit
  `147cd5ac9472002f5751221d390bf00abdbc0d2f`
- Runtime image:
  `ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204@sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617`

See `VERSION_PINNING.md` for the important 1,355-page embedded-GT discrepancy.

## Paths

- Code/config: this repository directory
- Dataset: `/mnt/nvme/datasets/OmniDocBench_v1.6/`
- Rebuildable scratch: `/mnt/nvme/scratch/omnidocbench_v1_6/`
- Rebuildable inference cache: `/mnt/nvme/outputs/omnidocbench_v1_6/`
- Checkpoints, raw responses, predictions, evaluator results, reports:
  `/opt/ocf-ai/outputs/omnidocbench_v1_6/<run_id>/`

The runner writes successful raw responses and predictions directly to EBS
before committing the SQLite transaction. `/mnt/nvme` is never the sole copy of
an inference result.

## Dataset preparation

The downloader is deliberately separate from preflight. It downloads only the
exact pinned revision, checks the GT hash and each Hugging Face LFS object hash,
and resumes already verified files:

```bash
python3 scripts/fetch_dataset.py
```

To inspect the remote immutable image manifest without downloading images:

```bash
python3 scripts/fetch_dataset.py --manifest-only
```

## Model approval

`config/models.yaml` is the authoritative logical-to-local mapping. A mapping
must contain an exact local Ollama tag and `supervisor_approved: true`.
Preflight never pulls models and never substitutes a different size, quantized
build, third-party build, or cloud endpoint. Model digest, architecture,
quantization, capabilities and context metadata are collected from the local
Ollama API.

## Commands

```bash
./run_tests.sh
./run_smoke.sh
./run_smoke.sh --preflight-only
./status.sh
./build_report.sh --smoke
./evaluate_only.sh
```

After smoke is explicitly reported as `READY_FOR_FULL_RUN`, the sole full-run
command is:

```bash
./run_full.sh
```

`evaluate_only.sh` never calls Ollama. `build_report.sh` reads EBS checkpoints
and evaluator artifacts without inference.

## Resume and interruption

`results.sqlite` uses `(run_signature, model_id, page_id)` as its key. A page is
reused only when its status is successful, its image hash is identical, both
raw and Markdown files exist, and both hashes still match. SIGINT/SIGTERM set an
interrupting status and stop after the current atomic page. Any config, model
mapping, prompt/options, dataset pin, evaluator pin, or scorer version change
creates a different signature and is refused by the existing checkpoint.

Predictions directories contain only official `.md` files. Raw API objects and
per-page metadata are in sibling directories.

## Common inference contract

Both models use byte-identical prompt text and options from
`config/benchmark.json`: `num_ctx=65536`, `num_predict=16384`,
`temperature=0`, `repeat_penalty=1.0`, `top_k=64`, `top_p=0.95`,
top-level `think=false`, `stream=false`, batch size 1, and fp16 KV cache.
Official page images are passed directly without OCR crops, enhancement, or
re-rasterization.

A response that reaches the common 16,384-token limit is preserved and reported
as a scoreable model-under-common-configuration outcome. The limit is never
raised for only one model.
