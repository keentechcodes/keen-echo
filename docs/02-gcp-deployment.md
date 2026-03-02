# GCP Cloud Run Deployment

Deploy your fine-tuned model to Google Cloud Run with vLLM.

## Prerequisites

1. **Google Cloud Account** with billing enabled
   - New users get $300 free credit
   - [console.cloud.google.com](https://console.cloud.google.com)

2. **gcloud CLI** installed
   - [Install guide](https://cloud.google.com/sdk/docs/install)

3. **GPU Quota** (you may need to request this)
   - Go to: IAM & Admin → Quotas
   - Search: "NVIDIA L4"
   - Request increase for your region

## Step 0: Configure .env

The deployment script reads configuration from a `.env` file. Create one from the example:

```bash
cp .env.example .env
```

Then edit `.env`. You need to choose a model source — HuggingFace or GCS:

**Option A: HuggingFace (simpler)**
```bash
GCP_PROJECT_ID="your-gcp-project-id"
GCP_REGION="us-central1"
SERVICE_NAME="your-bot"
HF_MODEL_ID="username/your-model"   # vLLM pulls directly from HuggingFace
```

**Option B: GCS bucket (self-hosted)**
```bash
GCP_PROJECT_ID="your-gcp-project-id"
GCP_REGION="us-central1"
SERVICE_NAME="your-bot"
HF_MODEL_ID=""                       # Leave empty to use GCS
GCS_BUCKET_NAME="your-model-bucket"
MODEL_PATH="my-model-merged"
```

HuggingFace is simpler — no bucket setup, no upload step. GCS gives you more control and avoids downloading the model on every cold start.

## Step 1: Initial Setup

### Install gcloud CLI (if not installed)

**Linux/WSL:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

**Or via apt:**
```bash
sudo apt-get install google-cloud-cli
```

### Authenticate
```bash
gcloud auth login
gcloud auth application-default login
```

### Create a new project
```bash
# Customize: replace with your project name
gcloud projects create your-project-id --name="Your Project Name"
gcloud config set project your-project-id

# Enable billing (required for GPU)
# Do this in the console: https://console.cloud.google.com/billing
```

### Enable required APIs
```bash
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com \
    compute.googleapis.com
```

## Step 2: Make Your Model Available

Choose the option matching your `.env` configuration.

### Option A: HuggingFace

If you set `HF_MODEL_ID` in your `.env`, your model is already available. vLLM will
pull it directly at startup. Skip to Step 3.

To upload a model to HuggingFace (if you haven't already):

```bash
pip install huggingface_hub

# Login (get token from https://huggingface.co/settings/tokens)
huggingface-cli login

# Upload your merged model directory
huggingface-cli upload username/your-model ./your-model-merged/
```

> **Note:** First cold start with HuggingFace will be slower (model downloads ~16GB).
> Subsequent cold starts use the cached download.

### Option B: Google Cloud Storage

If you left `HF_MODEL_ID` empty and set `GCS_BUCKET_NAME` + `MODEL_PATH`:

#### Create a bucket
```bash
BUCKET_NAME="your-model-bucket-$(date +%s)"
REGION="us-central1"

gsutil mb -l $REGION gs://$BUCKET_NAME
```

#### Upload your model

From your local machine (after downloading from RunPod):
```bash
# Extract if compressed
tar -xzvf your-model.tar.gz

# Upload to GCS (this may take 10-20 minutes)
# Customize: replace with your model directory name from training
gsutil -m cp -r your-qwen3-8b-merged gs://$BUCKET_NAME/

# Verify upload
gsutil ls gs://$BUCKET_NAME/your-qwen3-8b-merged/
```

Expected files:
```
gs://your-model-bucket/your-qwen3-8b-merged/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
└── model.safetensors.index.json
```

## Step 3: Create Service Account

The deploy script handles this automatically, but if running manually:

```bash
SA_NAME="model-service-sa"
PROJECT_ID=$(gcloud config get-value project)

# Create service account
gcloud iam service-accounts create $SA_NAME \
    --display-name="Model Service Account"

# Grant access to the bucket (GCS mode only)
gsutil iam ch \
    serviceAccount:$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com:objectViewer \
    gs://$BUCKET_NAME
```

## Step 4: Deploy to Cloud Run

The easiest way is to run the deploy script:

```bash
bash deploy_gcp.sh
```

The script detects your model source from `.env` and runs the right `gcloud run deploy` command. If you prefer to run it manually:

### HuggingFace mode

```bash
SERVICE_NAME="your-model-service"
REGION="us-central1"
HF_MODEL_ID="username/your-model"  # Customize: your HuggingFace model ID

# Google's pre-built vLLM image (Vertex AI)
VLLM_IMAGE="us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250312_0916_RC01"

gcloud run deploy $SERVICE_NAME \
    --image $VLLM_IMAGE \
    --region $REGION \
    --port 8000 \
    --cpu 8 \
    --memory 32Gi \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --max-instances 1 \
    --min-instances 0 \
    --timeout 300 \
    --service-account $SA_NAME@$PROJECT_ID.iam.gserviceaccount.com \
    --command python3 \
    --args="-m,vllm.entrypoints.openai.api_server,--model,$HF_MODEL_ID,--tensor-parallel-size,1,--max-model-len,2048,--trust-remote-code" \
    --no-gpu-zonal-redundancy \
    --allow-unauthenticated
```

### GCS mode

```bash
SERVICE_NAME="your-model-service"
REGION="us-central1"
MODEL_PATH="your-qwen3-8b-merged"  # Customize: must match the directory name in your bucket

# Google's pre-built vLLM image (Vertex AI)
# Check for latest tags at: https://console.cloud.google.com/artifacts/docker/vertex-ai/us/vertex-vision-model-garden-dockers
VLLM_IMAGE="us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250312_0916_RC01"

gcloud run deploy $SERVICE_NAME \
    --image $VLLM_IMAGE \
    --region $REGION \
    --port 8000 \
    --cpu 8 \
    --memory 32Gi \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --max-instances 1 \
    --min-instances 0 \
    --timeout 300 \
    --service-account $SA_NAME@$PROJECT_ID.iam.gserviceaccount.com \
    --set-env-vars "HF_HUB_OFFLINE=1" \
    --add-volume name=model-volume,type=cloud-storage,bucket=$BUCKET_NAME \
    --add-volume-mount volume=model-volume,mount-path=/model \
    --command python3 \
    --args="-m,vllm.entrypoints.openai.api_server,--model,/model/$MODEL_PATH,--tensor-parallel-size,1,--max-model-len,2048,--trust-remote-code" \
    --no-gpu-zonal-redundancy \
    --allow-unauthenticated
```

### What this does:
- Deploys vLLM as a serverless container
- **HuggingFace mode:** vLLM downloads the model from HuggingFace at startup
- **GCS mode:** Mounts your GCS bucket at `/model` and serves from there
- Starts an OpenAI-compatible API server
- Scales to 0 when idle (no cost)
- Allows unauthenticated access (for testing)

## Step 5: Get Service URL

```bash
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --region $REGION \
    --format 'value(status.url)')

echo "Your API endpoint: $SERVICE_URL"
```

## Step 6: Test the Deployment

### Health check
```bash
curl $SERVICE_URL/health
```

### List models
```bash
curl $SERVICE_URL/v1/models
```

### Chat completion
```bash
# The "model" value depends on your source:
#   HuggingFace: use your HF_MODEL_ID (e.g., "username/your-model")
#   GCS: use your MODEL_PATH (e.g., "your-qwen3-8b-merged")
curl $SERVICE_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-qwen3-8b-merged",
    "messages": [
      {
        "role": "system",
        "content": "Your system prompt here. Customize this to match your persona."
      },
      {
        "role": "user",
        "content": "hello, how are you?"
      }
    ],
    "max_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20
  }'
```

> **Note:** Qwen3 recommended params for non-thinking mode: temp=0.7, top_p=0.8, top_k=20

**Note:** First request after idle will take 30-60 seconds (cold start).

## Step 7: Connect the Chat UI

1. Open `index.html` in your browser
2. Enter your service URL in the "API Endpoint" field
3. Start chatting!

Or host it:
```bash
# Simple local server
python -m http.server 8080
# Open http://localhost:8080
```

## Cost Management

### Check current costs
```bash
# View Cloud Run metrics
gcloud run services describe $SERVICE_NAME --region $REGION
```

### Scale to zero (automatic)
The service automatically scales to 0 after ~15 minutes of no requests.

### Delete when done
```bash
# Delete the service
gcloud run services delete $SERVICE_NAME --region $REGION

# Delete the bucket (optional, keeps model for later)
gsutil rm -r gs://$BUCKET_NAME
```

## Pricing Estimate

| State | Cost |
|-------|------|
| Idle (scaled to 0) | $0/hr |
| Active (L4 GPU) | ~$0.50-0.80/hr |
| Storage (15GB model) | ~$0.30/month |

**Typical usage:** 2 hours/month = ~$1-2/month

## Troubleshooting

### "Quota exceeded" error
- Request L4 GPU quota in your region
- Go to: IAM & Admin → Quotas → Search "NVIDIA L4"

### Cold start too slow
- Set `--min-instances 1` (keeps one warm, costs ~$0.50/hr)

### Out of memory
- Reduce `--max-model-len` to 1024
- Or use a smaller quantized model

### Can't connect from UI
- Check CORS: The vLLM container should allow all origins
- Verify the URL includes `https://`
- Check browser console for errors

### Model not loading
```bash
# Check logs
gcloud run services logs read $SERVICE_NAME --region $REGION

# Verify model files exist (GCS mode)
gsutil ls gs://$BUCKET_NAME/$MODEL_PATH/
```

### HuggingFace download slow or failing
- First cold start downloads the full model (~16GB) — this takes several minutes
- Make sure the HuggingFace repo is public, or set `HF_TOKEN` as an env var in the deploy command
- Check logs for download progress: `gcloud run services logs read $SERVICE_NAME --region $REGION`
