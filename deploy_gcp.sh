#!/bin/bash
# ============================================================
# keen-echo — GCP Cloud Run Deployment Script
# ============================================================
#
# Deploys a fine-tuned LLM to GCP Cloud Run with GPU (vLLM).
#
# Prerequisites:
#   1. Google Cloud account with billing enabled
#   2. gcloud CLI installed and authenticated
#   3. Your fine-tuned model uploaded to GCS
#
# Usage:
#   chmod +x deploy_gcp.sh
#   ./deploy_gcp.sh
#
# ============================================================

set -e  # Exit on error

# ============================================================
# CONFIGURATION — Loaded from .env
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "Error: .env file not found."
    echo "Copy the example and fill in your values:"
    echo "  cp .env.example .env"
    exit 1
fi

source "$SCRIPT_DIR/.env"

PROJECT_ID="$GCP_PROJECT_ID"
REGION="$GCP_REGION"
BUCKET_NAME="$GCS_BUCKET_NAME"

# ============================================================
# COLORS FOR OUTPUT
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================================"
echo "keen-echo — GCP Cloud Run Deployment"
echo "============================================================"
echo -e "${NC}"

# ============================================================
# STEP 1: Authenticate and set project
# ============================================================

echo -e "${YELLOW}[1/7] Setting up GCP project...${NC}"

gcloud config set project $PROJECT_ID

echo -e "${GREEN}✓ Project set to $PROJECT_ID${NC}"

# ============================================================
# STEP 2: Enable required APIs
# ============================================================

echo -e "${YELLOW}[2/7] Enabling required APIs...${NC}"

gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com

echo -e "${GREEN}✓ APIs enabled${NC}"

# ============================================================
# STEP 3: Create GCS bucket (if not exists)
# ============================================================

echo -e "${YELLOW}[3/7] Creating Cloud Storage bucket...${NC}"

if gsutil ls -b gs://$BUCKET_NAME 2>/dev/null; then
    echo -e "${GREEN}✓ Bucket already exists${NC}"
else
    gsutil mb -l $REGION gs://$BUCKET_NAME
    echo -e "${GREEN}✓ Bucket created: gs://$BUCKET_NAME${NC}"
fi

# ============================================================
# STEP 4: Upload model (if not already uploaded)
# ============================================================

echo -e "${YELLOW}[4/7] Checking model upload...${NC}"

if gsutil ls gs://$BUCKET_NAME/$MODEL_PATH/ 2>/dev/null; then
    echo -e "${GREEN}✓ Model already uploaded${NC}"
else
    echo -e "${YELLOW}Model not found in bucket.${NC}"
    echo -e "${YELLOW}Please upload your model first:${NC}"
    echo ""
    echo "  gsutil -m cp -r /path/to/your-model-merged gs://$BUCKET_NAME/$MODEL_PATH/"
    echo ""
    echo -e "${RED}Exiting. Run this script again after uploading.${NC}"
    exit 1
fi

# ============================================================
# STEP 5: Create service account
# ============================================================

echo -e "${YELLOW}[5/7] Setting up service account...${NC}"

SA_NAME="${SERVICE_NAME}-sa"
SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

if gcloud iam service-accounts describe $SA_EMAIL 2>/dev/null; then
    echo -e "${GREEN}✓ Service account already exists${NC}"
else
    gcloud iam service-accounts create $SA_NAME \
        --display-name="$SERVICE_NAME Service Account"
    echo -e "${GREEN}✓ Service account created${NC}"
fi

# Grant GCS access
gsutil iam ch serviceAccount:$SA_EMAIL:objectViewer gs://$BUCKET_NAME

echo -e "${GREEN}✓ Service account configured${NC}"

# ============================================================
# STEP 6: Deploy to Cloud Run
# ============================================================

echo -e "${YELLOW}[6/7] Deploying to Cloud Run...${NC}"

# Using Google's pre-built vLLM container
# Check for latest tags: https://console.cloud.google.com/artifacts/docker/vertex-ai/us/vertex-vision-model-garden-dockers
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
    --service-account $SA_EMAIL \
    --set-env-vars "MODEL_ID=/model,HF_HUB_OFFLINE=1" \
    --add-volume name=model-volume,type=cloud-storage,bucket=$BUCKET_NAME \
    --add-volume-mount volume=model-volume,mount-path=/model \
    --command python3 \
    --args="-m,vllm.entrypoints.openai.api_server,--model,/model/$MODEL_PATH,--tensor-parallel-size,1,--max-model-len,2048,--trust-remote-code" \
    --no-gpu-zonal-redundancy \
    --allow-unauthenticated

echo -e "${GREEN}✓ Deployed to Cloud Run${NC}"

# ============================================================
# STEP 7: Get service URL and test
# ============================================================

echo -e "${YELLOW}[7/7] Getting service URL...${NC}"

SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --region $REGION \
    --format 'value(status.url)')

echo -e "${GREEN}"
echo "============================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "============================================================"
echo -e "${NC}"

echo -e "Service URL: ${BLUE}$SERVICE_URL${NC}"
echo ""
echo "Test your deployment:"
echo ""
echo "curl $SERVICE_URL/v1/chat/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo '    "model": "'$MODEL_PATH'",'
echo '    "messages": ['
echo '      {"role": "system", "content": "You are a digital twin. Respond in the persona's style."},'
echo '      {"role": "user", "content": "thoughts on learning?"}'
echo '    ],'
echo '    "max_tokens": 150,'
echo '    "temperature": 0.7'
echo "  }'"
echo ""
echo "============================================================"
echo "IMPORTANT NOTES:"
echo "============================================================"
echo ""
echo "1. Cold start takes 30-60 seconds (model loading)"
echo "2. After that, responses are fast (~1-2 seconds)"
echo "3. Service scales to 0 when idle (no cost)"
echo "4. Cost when active: ~\$0.50-0.80/hour"
echo ""
echo "To delete the service:"
echo "  gcloud run services delete $SERVICE_NAME --region $REGION"
echo ""
