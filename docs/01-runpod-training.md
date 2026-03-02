# RunPod Training Setup

Quick guide to run the training script on RunPod.

**Model:** Qwen3-8B (via `unsloth/Qwen3-8B-unsloth-bnb-4bit`)
**VRAM Required:** ~5GB with QLoRA (4-bit)

## Step 1: Create a Pod

1. Go to [runpod.io](https://runpod.io) and log in
2. Click **"+ Deploy"** → **"GPU Pod"**
3. Select a GPU:
   - **Recommended:** RTX 4090 (24GB) - ~$0.39/hr
   - **Alternative:** A10 (24GB) - ~$0.32/hr
   - **Overkill:** A100 (80GB) - ~$1.89/hr
4. Select template: **"Unsloth"** (preferred) or **"RunPod Pytorch 2.1"**
5. Set volume: **100GB** (merged model ~16GB + GGUF ~5GB + intermediates)
6. Click **"Deploy"**

## Step 2: Connect to Pod

Once the pod is running:
1. Click **"Connect"**
2. Choose **"Start Web Terminal"** or use SSH

## Step 3: Upload Files

### Option A: Using the Web UI
1. Click **"File Manager"** in RunPod
2. Upload these files to `/workspace/`:
   - `train.py`
   - `augmented_pairs.jsonl` (your training data)

### Option B: Using wget (if hosted)
```bash
cd /workspace
# Upload your files to a temporary host first, then:
wget https://your-host.com/train.py
wget https://your-host.com/augmented_pairs.jsonl
```

### Option C: Copy-paste the script
```bash
cd /workspace
nano train.py
# Paste the script content, then Ctrl+X, Y, Enter
```

## Step 4: Install Dependencies

```bash
pip install unsloth
pip install --upgrade transformers trl datasets
```

## Step 5: Run Training

```bash
cd /workspace
python train.py
```

### Expected Output:
```
============================================================
keen-echo — Digital Twin Training
============================================================
Started at: 2026-01-13 12:00:00
Model: unsloth/Qwen3-8B-unsloth-bnb-4bit

📦 Loading model...
✓ Model loaded

🔧 Adding LoRA adapters...
✓ LoRA adapters added (r=16, alpha=16)

📊 Loading dataset...
✓ Loaded 304 training pairs

🚀 Starting training...
[Training progress bar...]

✅ Training Complete!
Training time: 1200.00 seconds
Final loss: 0.8234

💾 Saving model...
✓ LoRA adapters saved

🔀 Merging and saving full model...
✓ Merged model saved

📦 Exporting to GGUF...
✓ GGUF exported
```

## Step 6: Download Outputs

After training completes, you'll have these folders (names match `output_dir` in CONFIG):
```
/workspace/
├── my-model/           # LoRA adapters only
├── my-model-merged/    # Full model (for Cloud Run)
└── my-model-gguf/      # GGUF file (for Ollama)
```

### Download merged model for Cloud Run:
```bash
# Compress the merged model
cd /workspace
tar -czvf my-model.tar.gz my-model-merged/

# Download via RunPod file manager or:
# Use runpodctl to sync files
```

### Download GGUF for local Ollama:
```bash
# The GGUF file is in my-model-gguf/
# Download just this file (~4-5GB)
```

## Step 7: (Optional) Push to HuggingFace

Edit the script to set your HF username:
```python
CONFIG = {
    ...
    "hub_model_id": "your-username/your-model-name",
}
```

Then set your token:
```bash
export HF_TOKEN="your_huggingface_token"
python train.py
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` to 1
- Reduce `max_seq_length` to 1024

### Training is slow
- Make sure you're on a GPU pod (not CPU)
- Check GPU usage: `nvidia-smi`

### Import errors
```bash
pip install unsloth --upgrade
pip install transformers trl datasets --upgrade
```

## Cost Estimate

| GPU | Time | Cost |
|-----|------|------|
| RTX 4090 | ~30 min | ~$0.20 |
| A10 | ~40 min | ~$0.21 |
| A100 80GB | ~20 min | ~$0.63 |

**Total estimated cost: $0.20 - $0.65**
