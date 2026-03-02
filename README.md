# keen-echo

A fine-tuning pipeline that creates a "digital twin" LLM mimicking your writing style. Train on your personal notes, deploy anywhere.

Built by [Keenan](https://github.com/keentechcodes) as a way to clone his own voice from Obsidian Daily Notes but designed so anyone can fork it and train their own. The trained model is available on [HuggingFace](https://huggingface.co/keenanthekeen/keenan-qwen3-8b).

## Overview

| Aspect | Details |
|--------|---------|
| **Model** | Qwen3-8B (LoRA fine-tuned via Unsloth) |
| **Training Data** | Your personal writing (Obsidian notes, journals, etc.) |
| **Deployment** | GCP Cloud Run (serverless GPU) or local via Ollama |
| **Model Hosting** | HuggingFace (recommended) or GCS bucket |
| **Training Cost** | ~$0.25-0.40 one-time (RunPod) |
| **Running Cost** | $0 idle / ~$0.50/hr active (cloud), $0 (local) |

## The Pipeline

```
  PREPARE               TRAIN                DEPLOY              CHAT
  ┌─────────┐          ┌─────────┐          ┌─────────┐        ┌─────────┐
  │ Your    │          │ RunPod  │          │  GCP    │        │  Web    │
  │ notes   │ ──────▶  │ ~30min  │ ──────▶  │ Cloud   │ ────▶  │  UI     │
  │ → JSONL │          │ ~$0.30  │          │  Run    │        │  Free   │
  └─────────┘          └─────────┘          └─────────┘        └─────────┘
  refine + validate    Unsloth + Qwen3-8B   vLLM + L4 GPU      index.html
                       LoRA fine-tuning      or local Ollama    (zero deps)
```

## Project Structure

```
keen-echo/
├── train.py                     # Main training script (RunPod GPU)
├── export_gguf.py               # GGUF export helper (for Ollama)
├── deploy_gcp.sh                # GCP Cloud Run deployment
├── index.html                   # Browser chat UI (zero dependencies)
├── refine_mannerisms.py         # Style cleaning for training data
├── validate_dataset.py          # Dataset quality checks
├── .env.example                 # Environment config template
│
├── docs/
│   ├── 01-runpod-training.md    # RunPod setup guide
│   ├── 02-gcp-deployment.md     # GCP Cloud Run guide
│   └── 03-local-ollama.md       # Local Ollama guide
│
└── examples/
    └── sample_pairs.jsonl       # Example training data format
```

## Quick Start

### 1. Prepare Your Data

Create a JSONL file with instruction/output pairs extracted from your writing:

```json
{"instruction": "what motivates you to keep learning?", "input": "", "output": "honestly its just curiosity..."}
{"instruction": "how do you deal with burnout?", "input": "", "output": "i step back and do literally nothing productive for a day..."}
```

See `examples/sample_pairs.jsonl` for more examples. Use `refine_mannerisms.py` and `validate_dataset.py` to clean and validate your data.

### 2. Train on RunPod (~30 minutes)

1. Create a RunPod GPU pod (RTX 4090, A10, or A100)
2. Upload `train.py` and your training data JSONL
3. Edit the `CONFIG` dict and `SYSTEM_PROMPT` in `train.py` to match your persona
4. Run:

```bash
pip install unsloth
pip install --upgrade transformers trl datasets
python train.py
```

Outputs:
- `your-model-merged/` — Full model for Cloud Run (vLLM)
- `your-model-gguf/` — Quantized GGUF for local Ollama

Detailed guide: [docs/01-runpod-training.md](docs/01-runpod-training.md)

### 3. Deploy

After training, upload your model to [HuggingFace](https://huggingface.co) for easy access:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload username/your-model ./your-model-merged/
```

**Option A: GCP Cloud Run** (serverless, scales to zero)
```bash
cp .env.example .env   # set HF_MODEL_ID or GCS bucket config
bash deploy_gcp.sh
```
Detailed guide: [docs/02-gcp-deployment.md](docs/02-gcp-deployment.md)

**Option B: Local Ollama** (free, private, always available)
```bash
ollama create my-model -f Modelfile
ollama run my-model "hello, how are you?"
```
Detailed guide: [docs/03-local-ollama.md](docs/03-local-ollama.md)

### 4. Chat

Open `index.html` in your browser, enter your API endpoint, and start chatting.

## Customization

### Persona & System Prompt

Edit the `SYSTEM_PROMPT` in `train.py` to describe the writing style you want the model to learn. The same prompt should be used in `index.html` for inference.

### Model Size

```python
# In train.py CONFIG dict:
"model_name": "unsloth/Qwen3-4B-unsloth-bnb-4bit"   # Smaller, faster
"model_name": "unsloth/Qwen3-8B-unsloth-bnb-4bit"   # Default (recommended)
"model_name": "unsloth/Qwen3-14B-unsloth-bnb-4bit"  # Larger, more capable
```

Note: There is no Qwen3-7B. Valid dense sizes are 0.6B, 1.7B, 4B, 8B, 14B, 32B.

### Training Data

The more authentic writing samples you provide, the better the output. Sources that work well:
- Daily journal entries
- Personal notes (Obsidian, Notion, etc.)
- Chat logs / messages
- Social media posts
- Blog drafts

## Data Pipeline Scripts

```bash
# Clean up style quirks in your training data
python refine_mannerisms.py input.jsonl output.jsonl

# Validate a dataset (checks JSON, required fields, duplicates, stats)
python validate_dataset.py your_pairs.jsonl
```

## Cost Breakdown

| Component | Cost |
|-----------|------|
| Training (RunPod, one-time) | ~$0.30 |
| Storage (GCS, monthly) | ~$0.30 |
| Cloud Run (per hour active) | ~$0.50-0.80 |
| Cloud Run (idle) | $0 |
| Local Ollama | $0 |

Typical monthly cost for light usage: $1-5.

## Technical Details

- **Base model:** Qwen3-8B with Unsloth Dynamic 4-bit quantization
- **Fine-tuning:** LoRA (r=16, alpha=16), targeting all attention + MLP projections
- **Training:** 3 epochs, batch_size=2, gradient_accumulation=4, lr=2e-4, adamw_8bit
- **Thinking mode:** Disabled (`enable_thinking=False`) (required for persona fine-tuning)
- **GGUF export:** Q4_K_M quantization (~4.5GB)
- **Chat template:** ChatML (`<|im_start|>/<|im_end|>`)
- **Inference params:** temp=0.7, top_p=0.8, top_k=20 (Qwen3 non-thinking recommendations)

## Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Unsloth Qwen3 Guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Cloud Run GPU Guide](https://cloud.google.com/run/docs/configuring/services/gpu)
- [Ollama Documentation](https://ollama.com/docs)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-8B)

## License

This project is for personal/educational use. The fine-tuned model inherits Qwen's Apache 2.0 license.
