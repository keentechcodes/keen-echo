# Local Deployment with Ollama

Run your fine-tuned model locally on your machine. No cloud costs, always available.

## Hardware Requirements

Ollama runs inference on CPU by default. Discrete NVIDIA GPUs are supported for
acceleration, but integrated GPUs (Intel/AMD iGPU) typically won't help.

**Minimum:** 16GB RAM, any modern x86_64 CPU
**Recommended:** 32GB RAM, 8+ core CPU (or discrete NVIDIA GPU with 8GB+ VRAM)

**Expected CPU performance:** 5-10 tokens/second with Q4_K_M quantization.

## Step 1: Install Ollama

### On Linux / WSL2
```bash
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve
```

### On macOS
```bash
brew install ollama
ollama serve
```

### On Windows
Download from [ollama.com/download](https://ollama.com/download)

## Step 2: Get the GGUF File

You need the quantized `.gguf` file from training. There are several ways to get it.

### From HuggingFace (easiest)

If your model is hosted on HuggingFace with a GGUF file:

```bash
pip install huggingface_hub

# Download just the GGUF file (not the full model)
# Customize: replace with your HuggingFace model ID and GGUF filename
huggingface-cli download username/your-model your-model-Q4_K_M.gguf \
    --local-dir ~/models/my-model
```

### From RunPod (after training)

After training on RunPod, you'll have a GGUF directory:
```
your-model-gguf/
└── your-model-Q4_K_M.gguf  (~4.5GB)
```

Download this file to your local machine.

**Using RunPod File Manager:**
1. Navigate to your GGUF output directory in the RunPod file browser
2. Right-click the `.gguf` file -> Download

**Using SCP (if SSH enabled):**
```bash
scp root@your-pod-ip:/workspace/your-model-gguf/*.gguf ~/models/
```

## Step 3: Create Modelfile

Create a file called `Modelfile` (no extension):

```bash
mkdir -p ~/models/my-model
cd ~/models/my-model

# Move your GGUF here (customize the filename to match yours)
mv ~/Downloads/your-model-Q4_K_M.gguf .

# Create Modelfile
# Note: Qwen3 recommended params for non-thinking mode: temp=0.7, top_p=0.8, top_k=20
cat > Modelfile << 'EOF'
FROM ./your-model-Q4_K_M.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER repeat_penalty 1.1

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """You are a helpful assistant. Customize this system prompt to match your persona and writing style."""
EOF
```

## Step 4: Create the Model in Ollama

```bash
cd ~/models/my-model
ollama create my-model -f Modelfile
```

Expected output:
```
transferring model data
creating model layer
writing manifest
success
```

## Step 5: Run It!

### Interactive Chat
```bash
ollama run my-model
```

### Single Query
```bash
ollama run my-model "hello, how are you?"
```

### Via API
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "my-model",
  "messages": [
    {"role": "user", "content": "hello, how are you?"}
  ],
  "stream": false
}'
```

## Step 6: Connect a Client

Ollama serves a local API at `http://localhost:11434`. You can use any compatible client:

- **Ollama CLI** (built-in, see examples above)
- **[Open WebUI](https://github.com/open-webui/open-webui)** (recommended for a full chat experience):

```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

Then open `http://localhost:3000`

## Performance Tuning

### For faster inference

**Increase threads:**
```bash
OLLAMA_NUM_THREADS=8 ollama serve
```

**Use Q4_K_S quantization (smaller, faster):**
In the training script, change:
```python
quantization_method="q4_k_s"  # instead of q4_k_m
```

### For better quality

**Use Q5_K_M or Q6_K quantization:**
```python
quantization_method="q5_k_m"  # ~5.5GB, better quality
```

**Use Q8_0 (best quality, 32GB RAM required):**
```python
quantization_method="q8_0"  # ~8GB, near-original quality
```

## Expected Speed

| Response Length | Tokens | Time (CPU) |
|-----------------|--------|------------|
| Short reply | ~50 | 5-10 sec |
| Medium reply | ~100 | 10-20 sec |
| Long reply | ~200 | 20-40 sec |

Totally usable for a personal model you query occasionally.

## Ollama Commands Cheat Sheet

```bash
# List models
ollama list

# Show model info
ollama show my-model

# Remove model
ollama rm my-model

# Pull a base model (for comparison)
ollama pull qwen3:8b

# Check running models
ollama ps

# Stop Ollama
ollama stop
```

## Troubleshooting

### "model not found"
```bash
# Make sure you're in the right directory
cd ~/models/my-model
ls -la  # Should see Modelfile and .gguf

# Recreate
ollama create my-model -f Modelfile
```

### Slow performance
- Close other applications
- Increase threads: `OLLAMA_NUM_THREADS=8 ollama serve`
- Use smaller quantization (Q4_K_S)

### Out of memory
- Close browsers and other RAM-heavy apps
- Use Q4_K_S quantization
- Reduce context length in Modelfile:
  ```
  PARAMETER num_ctx 1024
  ```

### Garbled output
- Check the TEMPLATE format matches Qwen3's ChatML template (`<|im_start|>/<|im_end|>`)
- Ensure the GGUF was exported correctly
- Make sure thinking mode wasn't enabled during training (should use `enable_thinking=False`)
