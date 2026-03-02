# AGENTS.md — keen-echo

## Project Overview

keen-echo is a personal ML/AI fine-tuning pipeline that creates a "digital twin" LLM
mimicking a specific writing style. It fine-tunes Qwen3-8B on personal writing (e.g.,
Obsidian Daily Notes) using LoRA via the Unsloth framework, then deploys via vLLM on
GCP Cloud Run or locally via Ollama.

This is a data/ML pipeline project, not a traditional software library. It consists
of standalone Python scripts, JSONL datasets, Bash deployment scripts, and a
single-file HTML chat UI.

## Repository Layout

```
keen-echo/
  train.py                       # Main training script (runs on RunPod GPU)
  export_gguf.py                 # Standalone GGUF export helper (CLI args)
  deploy_gcp.sh                  # GCP Cloud Run deployment (bash)
  index.html                     # Browser chat UI (HTML/CSS/JS, zero dependencies)
  refine_mannerisms.py           # Text cleaning: lowercasing, period removal, style fixes
  validate_dataset.py            # Dataset quality checks (JSON validity, duplicates, stats)
  .env.example                   # Environment config template (GCP project, bucket, etc.)
  README.md                      # Project README
  AGENTS.md                      # This file
  .gitignore                     # Excludes personal data, model outputs, legacy dirs
  docs/
    01-runpod-training.md        # RunPod setup guide
    02-gcp-deployment.md         # GCP deployment guide
    03-local-ollama.md           # Local Ollama guide
  examples/
    sample_pairs.jsonl           # Example training data format (5 synthetic pairs)
```

### Files NOT in the repo (gitignored)

These stay on disk but are excluded from version control:
- `keenan_corpus.txt` — Raw extracted text from Obsidian Daily Notes
- `extraction_debug.jsonl` — Line-by-line labeling of source notes
- `extraction_stats.json` — Corpus extraction statistics
- `training_pairs.jsonl` — Initial instruction/output pairs
- `augmented_pairs.jsonl` — Expanded pairs
- `refined_pairs.jsonl` — Style-corrected pairs
- `test_prompt.md` / `test_prompt_refined.md` — Personal test prompts
- `validation_report.md` — Multi-agent validation report (kept locally)
- `chat_artifacts/` — Legacy directory (contents moved to root + docs/)
- `training_setup(gemini3gen)/` — Earlier experiment (superseded)
- `*.gguf` and model output directories

## Build / Run / Test Commands

There is no formal build system, package manager, or test framework. Scripts are
run directly with Python. There is no `requirements.txt` or `pyproject.toml`.

### Data Pipeline

```bash
# Refine training pairs (apply style cleaning to outputs)
python refine_mannerisms.py input.jsonl output.jsonl

# Validate a dataset file (checks JSON, required fields, duplicates, stats)
python validate_dataset.py augmented_pairs.jsonl
```

### Training (on RunPod with GPU)

```bash
pip install unsloth
pip install --upgrade transformers trl datasets
python train.py
```

Training runs 3 built-in test prompts automatically at the end (Step 9 in the script).

### Deployment

```bash
# GCP Cloud Run deployment
bash deploy_gcp.sh

# Local Ollama
ollama create my-model -f Modelfile
ollama run my-model "your prompt here"
```

### Testing

There are no pytest/unittest tests. Validation is done via:
1. `python validate_dataset.py <file>` for dataset quality
2. Built-in inference tests in `train.py` (runs after training)
3. Manual prompt testing

## Python Dependencies

Installed manually (no lockfile):
- `unsloth` - LoRA fine-tuning framework
- `transformers` - HuggingFace
- `trl` - SFTTrainer
- `datasets` - HuggingFace datasets
- `torch` - PyTorch
- `peft` - Parameter-Efficient Fine-Tuning
- `bitsandbytes` - Quantization
- Standard library: `json`, `re`, `sys`, `os`, `collections`, `datetime`

## Code Style Guidelines

### Language and Structure

- Python 3. No type hints are used.
- Scripts are standalone and self-contained (no shared modules or library imports
  between project files).
- No classes; use functions and procedural/script-level code.
- Main training script uses `if __name__ == "__main__": main()` guard.
  Smaller utility scripts may omit the guard.

### Imports

- Standard library imports first, then third-party. No blank line separator enforced.
- Use simple `import` and `from X import Y`. No `from __future__` imports.
- No import sorting tools. Keep imports at the top of the file.

```python
import os
import json
import torch
from datetime import datetime
```

### Naming Conventions

- Functions: `snake_case` (e.g., `style_cleaner`, `validate_dataset`, `format_conversation`)
- Variables: `snake_case` (e.g., `processed_count`, `valid_json_count`)
- Module-level constants/config: `UPPER_CASE` (e.g., `CONFIG`, `SYSTEM_PROMPT`, `EOS_TOKEN`)
- No classes in the codebase.

### Formatting

- No formatter (black, ruff, etc.) is configured.
- 4-space indentation.
- Use section separator comments in longer scripts:
  ```python
  # ============================================================
  # SECTION NAME
  # ============================================================
  ```
- Use numbered step comments within `main()`:
  ```python
  # Step 1: Install dependencies (if needed)
  ```

### Error Handling

- Minimal, pragmatic error handling. No custom exception classes.
- `try/except FileNotFoundError` for file I/O.
- `try/except json.JSONDecodeError` for JSONL parsing.
- `try/except ImportError` with inline `pip install` fallback for optional deps.
- Print errors directly; no logging framework is used.

### Output and Logging

- All output via `print()`. No `logging` module.
- Use f-strings for formatted output.
- Print before/after samples when processing data.

### Data Format

- Training data is JSONL with schema: `{"instruction": "...", "input": "", "output": "..."}`
- The `input` field is always an empty string (Alpaca format compatibility).
- File encoding: UTF-8 with `encoding="utf-8"` in `open()` calls.

### Bash Scripts

- Use `set -e` for fail-fast behavior.
- Use `set -o pipefail` when piping commands.

### HTML/JS (Chat UI)

- Single-file architecture: all HTML, CSS, and JS in one `index.html`.
- Zero external dependencies.
- Dark theme styling.

## Key Technical Decisions

- Model: `unsloth/Qwen3-8B-unsloth-bnb-4bit` (Dynamic 4-bit quantization).
  Note: There is NO Qwen3-7B. Valid sizes are 0.6B, 1.7B, 4B, 8B, 14B, 32B.
- Thinking mode is DISABLED (`enable_thinking=False`) for persona fine-tuning.
- LoRA config: r=16, alpha=16, dropout=0, targets all attention + MLP projections.
- Training: 3 epochs, batch_size=2, gradient_accumulation=4, lr=2e-4, adamw_8bit.
- GGUF export uses `q4_k_m` quantization.

## Agent Notes

- When modifying Python scripts, maintain the existing style: no type hints,
  procedural code, `print()` for output, minimal error handling.
- When modifying training config, edit the `CONFIG` dict at the top of `train.py`.
- Personal data files are gitignored — never commit them.
- The `chat_artifacts/` directory is the legacy location; all active files are at
  root level or in `docs/` and `examples/`.
- No CI/CD, no linting, no formatting tools, no pre-commit hooks.
