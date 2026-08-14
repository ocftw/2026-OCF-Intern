# 生成多模型比較報告

任何有這台機器帳號的人都可以直接執行，**不需要這個 repo 擁有者的登入權限**——資料與程式
碼都已部署在共用路徑，只有讀取權限，不會動到任何正在進行中的評測：

```bash
cd /opt/ocf-ai/outputs/omnidocbench_reports/code

# 先看看目前有哪些模型已經有完整的官方評測結果可以選
python3 tools/build_comparison_report.py --list-models

# 以 gemma4_e2b 的 text_block edit distance 為基準，各抽 5 筆最好/最差/隨機，
# 目前所有已跑完的模型全部並列比較
python3 tools/build_comparison_report.py \
  --baseline gemma4_e2b --best 5 --worst 5 --random 5 \
  --out /opt/ocf-ai/outputs/omnidocbench_reports/generated/<你的名字>_report.html
```

- `generated/` 底下大家都能互相讀寫，產生的報告留在那裡讓其他人也能打開看。
- 打開產生的 HTML 後，畫面最上面可以直接**點欄位標題排序**整體分數表（預設 Overall 由高到
  低），也有一個工具列可以即時**切換排序依據的模型、方向、篩選最佳/最差/隨機、依檔名搜
  尋**——這些都是純前端互動，不需要重新執行指令。往下展開「📖 指標說明」摺疊區塊可以看
  六個子指標(`text_block edit`/`table edit`/`table TEDS`/`formula edit`/`formula CDM`/
  `reading_order edit`)的意義、演算法跟方向。
- 完整參數說明：`python3 tools/build_comparison_report.py --help`。
- 想要舊版「兩個模型並排、依單一模型排序、產生時就固定順序」的簡化版報告，可以用
  `tools/build_visual_report.py`（用法見檔案內 docstring）。

## 共用資料維護

上面那條路徑(`/opt/ocf-ai/outputs/omnidocbench_reports/`)是這個目錄的**唯讀鏡像**，不含任
何一頁圖片或預測資料本身——它讀的其實是 `/mnt/nvme/datasets/OmniDocBench_v1.6/`(GT、圖片)
跟 `/opt/ocf-ai/outputs/omnidocbench_v1_6/`(各模型的預測與官方評測結果)，這兩處資料夾原本
的檔案權限只有擁有者能讀，已經用 `tools/fix_shared_report_permissions.py` 開了 other-read。

**新模型跑完評測後**，要讓其他人也看得到，依序做兩件事：

```bash
# 1. 幫新模型的預測檔案(以及萬一有新圖片)開 other-read 權限(原地生效，不搬移/複製資料，
#    可重複執行，已經開過的檔案會被跳過)
python3 tools/fix_shared_report_permissions.py

# 2. 把程式碼變更同步到共用鏡像(只鏡像程式碼/設定，不含 *.html、__pycache__)
rsync -a --delete \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.html' --exclude='.git/' \
  ./ /opt/ocf-ai/outputs/omnidocbench_reports/code/
chmod -R o+rX /opt/ocf-ai/outputs/omnidocbench_reports/code
chgrp -R ocfai /opt/ocf-ai/outputs/omnidocbench_reports/code
```

（這台機器系統碟空間吃緊，資料本身刻意不複製一份到 `/opt/ocf-ai/outputs`，只鏡像程式碼——
程式碼加設定總共幾百 KB，資料在原地用權限開放讀取即可。）

---

# OmniDocBench v1.6 reproducible runner

This directory contains an end-to-end, resumable runner for the **1,651-page
OmniDocBench v1.6 full set**. It uses host Ollama for batch-size-one VLM
inference and the pinned official Docker evaluator for `end2end` scoring.

The current workflow has a hard human approval gate:

1. `./run_smoke.sh` performs preflight, per-model warm-up/GPU checks, the same
   fixed 20-page inference for every configured model, official smoke
   evaluation, and smoke reporting.
2. It never calls the full runner. A valid smoke run remains
   `SMOKE_REQUIRES_AI_REVIEW` until all outputs have been inspected and a
   signed-by-content `ai_review.json` plus `READY_FOR_FULL_RUN` manifest exist.
3. Only after reviewing that report should the operator manually run
   `./run_full.sh` inside tmux. Re-running the command resumes from EBS
   checkpoints.

No script creates tmux sessions, background jobs, schedules, or approval files.

For a `config_variants/*.yaml` run only (never the main `config/models.yaml`
path), `tools/run_model_variant.py full-inference --skip-review` is an
explicit operator opt-out of this gate: default behavior (no flag) is
unchanged, and the flag lives entirely in `tools/run_model_variant.py`, which
is outside `code_hash`'s scope -- using it never changes any run's
`run_signature`/`run_id`, Gemma or otherwise. Skipping smoke this way also
means `run_official_evaluator()` (which normally writes each model's
`immutable_config/{model_id}_official.yaml` as a side effect of the smoke
evaluation) never runs, so `full-inference` calls
`ensure_official_configs()` itself to write that same, content-identical,
already-hashed file per model -- otherwise `tools/evaluate_model.py` would
fail later with "immutable official config is missing".

Under the fixed-configuration comparison policy, model-originated outcomes such
as repetition, empty content, refusal, malformed Markdown, or
`done_reason=length` are preserved and scored. They remain visible as anomaly
flags but do not by themselves block the experiment. API/runtime failures,
wrong inputs or signatures, GPU fallback, missing predictions, and evaluator
failures remain blocking. Official-evaluator hangs triggered by a specific
page's degenerate model output are handled explicitly and auditably instead --
see `tools/mark_eval_timeout_page.py` below.

## Models

Two independent model sets, each with its own logical-to-local mapping file so
adding one never changes the other's `run_id` (see `tools/run_model_variant.py`):

| Model id | Logical name | Mapping file |
|---|---|---|
| `gemma4_e2b` | Gemma 4 E2B | `config/models.yaml` |
| `gemma4_26b_a4b` | Gemma 4 26B A4B | `config/models.yaml` |
| `gemma4_e4b` | Gemma 4 E4B | `config_variants/models_e4b_12b_31b.yaml` |
| `gemma4_12b` | Gemma 4 12B | `config_variants/models_e4b_12b_31b.yaml` |
| `gemma4_31b` | Gemma 4 31B | `config_variants/models_e4b_12b_31b.yaml` |
| `qwen3_vl_4b` | Qwen3-VL 4B | `config_variants/models_qwen_internvl.yaml` |
| `qwen3_vl_32b` | Qwen3-VL 32B | `config_variants/models_qwen_internvl.yaml` |
| `internvl3_5_4b` | InternVL3.5 4B | `config_variants/models_qwen_internvl.yaml` |
| `internvl3_5_38b` | InternVL3.5 38B | `config_variants/models_qwen_internvl.yaml` |

All nine currently have a complete `official_evaluation_full_1651` result;
`python3 tools/build_comparison_report.py --list-models` always reflects the
live state (and explains, for any model missing a result, whether it's still
running or genuinely failed). The comparison report needs no changes to see a
new model set -- it auto-discovers every model with a complete result across
all run directories, Gemma and non-Gemma alike; see "生成多模型比較報告" at the
top of this file for the exact command.

The four `config_variants/model_<id>.yaml` single-model files (each its own
`run_id`) were the ones used to smoke-test each model individually before the
combined run; `config_variants/models_qwen_internvl.yaml` is the one actually
used for the completed full run above and is the one to reuse or extend.

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
- Shared, other-users-can-read-too mirror for report generation only (see
  above): `/opt/ocf-ai/outputs/omnidocbench_reports/`

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

`config/models.yaml` (and any `config_variants/*.yaml` used via
`tools/run_model_variant.py`) is the authoritative logical-to-local mapping. A
mapping must contain an exact local Ollama tag and `supervisor_approved: true`.
Preflight never pulls models and never substitutes a different size, quantized
build, third-party build, or cloud endpoint. Model digest, architecture,
quantization, capabilities and context metadata are collected from the local
Ollama API.

Not every model has an official Ollama library entry. `internvl3_5_4b` and
`internvl3_5_38b` are imported from community GGUF quantizations (no vision
model from that vendor is in the official library); `tools/fetch_vlm_models.py`
pulls the official Qwen3-VL tags and does the community GGUF + mmproj import
+ `ollama create` for InternVL, with sha256-pinned source files. A mapping can
also declare `expected_min_context` to override the default 65536 floor for a
model whose real native context is lower (InternVL3.5's is 40960) -- an
explicit, recorded acknowledgment of that shortfall, not a bug workaround; see
the `mapping_evidence` field on those two entries for the full trail
(community-GGUF provenance, disk/import issues hit, context caveat).

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

## `tools/` reference

Everything here is read-only with respect to any run's official evaluation
output unless its own docstring says otherwise.

| Script | What it does |
|---|---|
| `evaluate_model.py` | Runs the pinned official evaluator for one model against an existing full inference run; supports resumable batching (`--batch-size`) with automatic corpus-wide merge once every batch is `done`, and `--eval-timeout-as-empty` to substitute a known-hang page with an empty prediction (see `mark_eval_timeout_page.py`). |
| `auto_resume_evaluation.py` | Supervises `evaluate_model.py` across the confirmed degenerate-repetition evaluator hang: retries stalled batches, and only auto-registers a page via `mark_eval_timeout_page.py` when the stall matches the exact known signature -- otherwise it stops for manual review. |
| `mark_eval_timeout_page.py` | Records (or removes) a ground-truth page as a known official-evaluator hang in `evaluation_timeout_pages.json`, with mandatory reason + evidence. Only ever *used* when a run passes `--eval-timeout-as-empty`; the real prediction is never modified. |
| `fix_shared_report_permissions.py` | One-time-per-new-model chmod fix (see "共用資料維護" above): adds other-read on the GT JSON, dataset images, and every model's `predictions/` files so non-owner users can generate reports. Idempotent, never touches any other file. |
| `run_model_variant.py` | Runs preflight/smoke/inference-only for an alternate model set defined in a `config_variants/*.yaml` file living outside `config/`, so it gets its own `run_id` without disturbing any other run's signature. `full-inference` supports `--skip-review` (see the hard-approval-gate section above). |
| `fetch_vlm_models.py` | Fetches/imports the models behind `config_variants/models_qwen_internvl.yaml` and the four single-model `model_<id>.yaml` files: official `ollama pull` for the two Qwen3-VL tags, and download + sha256-verify + two-`FROM`-Modelfile `ollama create` for the two community-GGUF InternVL3.5 imports. `--only <id>` fetches a subset; `--smoke-test` runs a real image through each fetched model afterward and prints the response. |
| `full_inference_only.py` | Runs the full 1,651-page inference pass without the built-in evaluator call at the end (evaluation is done afterwards via `evaluate_model.py`/`auto_resume_evaluation.py`, which have stall protection the built-in call lacks). |
| `evaluation_status.py` / `inference_status.py` | Read-only progress snapshots (single-shot or `--watch`) for an in-progress official evaluation / inference run. |
| `stop_evaluation.py` | Stops any evaluator container left running for a given run/model. |
| `build_comparison_report.py` | Self-contained, interactively-sortable HTML report: every model with a complete result side by side, page image + ground truth + best/worst/random subset by one baseline model's score, re-sortable in the browser by any model or by any summary-table metric. This is the tool documented at the top of this file. |
| `build_visual_report.py` | Older, simpler two-model-at-a-time visual report (image + GT + predictions), fixed sort order set at generation time. |
| `_container_ops.py`, `_merge_driver.py` | Internal helpers (docker container lookup; batch-merge logic run inside the pinned evaluator image) used by the scripts above, not meant to be invoked directly. |

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

Every model uses byte-identical prompt text and options from
`config/benchmark.json`: `num_ctx=65536`, `num_predict=16384`,
`temperature=1.0`, `repeat_penalty=1.0`, `top_k=64`, `top_p=0.95`,
`seed=20260731`, top-level `think=false`, `stream=false`, batch size 1, and
fp16 KV cache. Official page images are passed directly without OCR crops,
enhancement, or re-rasterization.

A response that reaches the common 16,384-token limit is preserved and reported
as a scoreable model-under-common-configuration outcome. The limit is never
raised for only one model.
