#!/bin/bash
set -e

PROJECT_ID="drone-detection-mlops"
SERVICE_ACCOUNT="github-actions-ci@drone-detection-mlops.iam.gserviceaccount.com"

echo "Setting up service account permissions for CI/CD"
echo "---------------------------------------------------------------"

# Cloud Build Editor - allows triggering builds
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/cloudbuild.builds.editor" \
  --quiet

# Storage Admin - allows pushing Docker images to GCR/Artifact Registry + DVC
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/storage.admin" \
  --quiet

# Cloud Run Admin - allows deploying services
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/run.admin" \
  --quiet

# Service Account User - allows acting as other service accounts during deployment
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/iam.serviceAccountUser" \
  --quiet

# Artifact Registry Writer - for pushing container images (if using Artifact Registry)
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/artifactregistry.writer" \
  --quiet

echo "Service account configured successfully!"
