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

# Storage Admin - allows pushing Docker images to GCR/Artifact Registry
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/storage.admin" \
  --quiet

echo "Service account configured successfully!"
