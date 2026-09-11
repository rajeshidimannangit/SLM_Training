# SLM Fine-Tuning (Local / Enterprise)

Local Unsloth-based small language model fine-tuning with **LoRA** and **QLoRA**, separated configs for hyperparameter tuning, isolated **base** vs **finetuned** model folders, and an enterprise metrics store (training + evaluation + business/use-case KPIs) with a CLI.

## Layout

```text
.
├── configs/
│   ├── model/          # base + finetuned paths, seq length, dtype
│   ├── lora/           # lora.yaml | qlora.yaml
│   ├── training/       # learning rate, epochs, batch, scheduler, …
│   │                   # default.yaml (local) + colab.yaml (CUDA)
│   ├── data/           # dataset paths + template
│   └── metrics/        # metric catalog + SQLite/MLflow store
├── notebooks/
│   └── slm_finetune_colab.ipynb   # Google Colab cloud training
├── data/
│   ├── raw/
│   ├── processed/      # train.jsonl, eval.jsonl
│   └── evaluation/     # usecase_eval.jsonl (business KPIs)
├── models/
│   ├── base/           # base model metadata / snapshots
│   └── finetuned/      # adapters per run_id
├── experiments/        # per-run summaries
├── artifacts/
│   ├── mlruns/         # MLflow tracking (optional)
│   ├── metrics/        # SQLite metrics DB
│   └── reports/        # resolved configs / reports
└── src/slm_finetune/   # training, evaluation, metrics, chat UI, CLI
```

## Setup

```bash
cd "SLM Finetuning"
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
# Or: pip install -r requirements.txt && pip install -e .
```

Unsloth on **Apple Silicon** uses the **MLX** backend (`UnslothTrainer`). On **CUDA** (including Google Colab) it uses TRL `SFTTrainer`. Stock TRL cannot train MLX models (you would see `AttributeError: 'Model' object has no attribute 'config'`).

## Google Colab (cloud GPU)

1. Open [`notebooks/slm_finetune_colab.ipynb`](notebooks/slm_finetune_colab.ipynb) in [Google Colab](https://colab.research.google.com/).
2. **Runtime → Change runtime type → GPU**.
3. Get the project onto Colab via one of:
   - **Clone** your GitHub repo, or
   - **Mount Drive** to the project folder, or
   - **Upload a zip** created with:
     ```bash
     bash scripts/pack_for_colab.sh
     ```
4. Run the install + train cells. Colab uses:
   - `configs/model/colab.yaml`
   - `configs/training/colab.yaml`
5. Download the adapter zip from the last cell into local `models/finetuned/`.

```bash
# Equivalent CLI inside Colab after install:
slm train \
  --method qlora \
  --model-config configs/model/colab.yaml \
  --training-config configs/training/colab.yaml \
  --run-name card-ops-colab
```

| Profile | Config | Backend |
|---------|--------|---------|
| Local Mac | `configs/training/default.yaml` | MLX |
| Colab / CUDA | `configs/training/colab.yaml` | CUDA |

## Configure hyperparameters

Edit only `configs/training/default.yaml` for LR, epochs, batch size, warmup, scheduler, etc. LoRA vs QLoRA lives in `configs/lora/`. Model paths live in `configs/model/default.yaml`.

## Train

```bash
# QLoRA (default)
slm train --method qlora

# LoRA
slm train --method lora

# Override LR / epochs without editing YAML
slm train --method qlora --lr 1e-4 --epochs 2 --run-name support-bot-v1
```

Adapters are written to `models/finetuned/<run_id>/`. Base model config/metadata uses `models/base/`.

## Compare chat (base vs fine-tuned)

Side-by-side Gradio UI: one message in, answers from **base** and **fine-tuned** models.

```bash
pip install gradio   # if not already installed
slm chat --model-dir models/finetuned/card-ops-fast3
# open http://127.0.0.1:7860
```

Click **Load models**, then send a customer message. Optional: `--port 7861`, `--share`.

## Use-case / business metrics

```bash
slm eval-usecase --model-dir models/finetuned/<run_id>
```

KPIs (configurable in `configs/metrics/default.yaml`):

| Category | Examples |
|----------|----------|
| Training | loss, learning_rate, throughput |
| Evaluation | eval_loss, perplexity, ROUGE, BLEU |
| Use-case | task_success_rate, latency_p50/p95, format_compliance, hallucination_flag_rate |
| Business | business_score (weighted composite) |

## Metrics CLI

```bash
slm metrics list-runs
slm metrics show <run_id>
slm metrics show <run_id> --category training
slm metrics summary <run_id>
slm metrics latest
slm metrics compare <run_a> <run_b>
slm metrics export <run_id> -o artifacts/reports/<run_id>_metrics.json

# Base SLM vs Fine-tuned SLM enterprise report
slm metrics report --finetuned-model-dir models/finetuned/<run_id>
# Uses held-out questions in data/evaluation/heldout_comparison_eval.jsonl
# (not part of train.jsonl). Prints metrics + query/answer for base vs FT.
slm metrics report --demo                          # sample table without GPU
slm metrics report --from-json artifacts/reports/<id>/base_vs_finetuned_report.json
```

Report shape:

| Metric | Base SLM | Fine-tuned SLM |
|--------|----------|----------------|
| Classification accuracy | … | … |
| Precision / Recall / F1 | … | … |
| Reason-code accuracy | … | … |
| Customer reply | … | … |
| Structured output | … | … |
| Consistency | … | … |
| Hallucination | … | … |
| Latency | … | … |

## Data format

`data/processed/*.jsonl` — Alpaca-style by default:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

`data/evaluation/usecase_eval.jsonl` — business checks:

```json
{
  "prompt": "...",
  "expected": "...",
  "must_contain": ["..."],
  "must_not_contain": ["..."],
  "format_regex": "..."
}
```

## Design notes

- **Configs are the control plane** — change tuning parameters in YAML; CLI flags only override.
- **SQLite is the system of record** for metrics (`artifacts/metrics/metrics.db`); MLflow mirrors when enabled.
- **Run isolation** — each run gets its own finetuned folder, experiment summary, and metric rows keyed by `run_id`.
