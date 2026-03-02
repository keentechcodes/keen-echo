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

Then edit `.env` with your values:
```bash
GCP_PROJECT_ID="your-gcp-project-id"
GCP_REGION="us-central1"
GCS_BUCKET_NAME="your-model-bucket"
SERVICE_NAME="your-bot"
MODEL_PATH="my-model-merged"
```

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

## Step 2: Upload Model to Cloud Storage

### Create a bucket
```bash
BUCKET_NAME="your-model-bucket-$(date +%s)"
REGION="us-central1"

gsutil mb -l $REGION gs://$BUCKET_NAME
```

### Upload your model

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

```bash
SA_NAME="model-service-sa"
PROJECT_ID=$(gcloud config get-value project)

# Create service account
gcloud iam service-accounts create $SA_NAME \
    --display-name="Model Service Account"

# Grant access to the bucket
gsutil iam ch \
    serviceAccount:$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com:objectViewer \
    gs://$BUCKET_NAME
```

## Step 4: Deploy to Cloud Run

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
- Mounts your GCS bucket at `/model`
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

# Verify model files exist
gsutil ls gs://$BUCKET_NAME/$MODEL_PATH/
```
