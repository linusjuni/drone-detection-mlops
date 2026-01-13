#!/bin/bash
set -e

PROJECT_ID="drone-detection-mlops"
MEMBERS_FILE="$(dirname "$0")/cloud_members.txt"

if [ ! -f "$MEMBERS_FILE" ]; then
    echo "Error: $MEMBERS_FILE not found"
    exit 1
fi

echo "Setting up IAM permissions for GCP project: $PROJECT_ID"
echo ""

while IFS= read -r MEMBER || [ -n "$MEMBER" ]; do
    # Skip empty lines and comments
    [[ -z "$MEMBER" || "$MEMBER" =~ ^#.*$ ]] && continue

    echo "Configuring permissions for: $MEMBER"

    # Project-level roles
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="$MEMBER" \
        --role="roles/iam.serviceAccountUser" \
        --quiet

    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="$MEMBER" \
        --role="roles/aiplatform.user" \
        --quiet

    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="$MEMBER" \
        --role="roles/artifactregistry.writer" \
        --quiet

    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="$MEMBER" \
        --role="roles/cloudbuild.builds.editor" \
        --quiet

    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="$MEMBER" \
        --role="roles/monitoring.viewer" \
        --quiet

    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="$MEMBER" \
        --role="roles/logging.viewer" \
        --quiet

    # Bucket permissions
    gsutil iam ch "$MEMBER:roles/storage.objectAdmin" \
        gs://drone-detection-mlops-data

    gsutil iam ch "$MEMBER:roles/storage.objectAdmin" \
        gs://drone-detection-mlops-models

    echo "  Permissions granted"
    echo ""
done < "$MEMBERS_FILE"
