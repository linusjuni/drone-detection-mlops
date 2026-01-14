#!/bin/bash
set -e

# Project configuration
PROJECT_ID="drone-detection-mlops"
MEMBERS_FILE="$(dirname "$0")/cloud_members.txt"

# Check if members file exists
if [ ! -f "$MEMBERS_FILE" ]; then
    echo "Error: $MEMBERS_FILE not found"
    exit 1
fi

echo "Granting Full Technical Admin (Editor) for Project: $PROJECT_ID"
echo "---------------------------------------------------------------"

while IFS= read -r MEMBER || [ -n "$MEMBER" ]; do
    # Skip empty lines and comments
    [[ -z "$MEMBER" || "$MEMBER" =~ ^#.*$ ]] && continue

    echo "Configuring Full Admin for: $MEMBER"

    # 1. EDITOR ROLE: Full power to build, train, and manage all resources.
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="$MEMBER" \
        --role="roles/editor" \
        --quiet

    # 2. SERVICE USAGE CONSUMER: Essential for running gcloud commands.
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="$MEMBER" \
        --role="roles/serviceusage.serviceUsageConsumer" \
        --quiet

    # 3. SERVICE ACCOUNT USER: Allows team members to run jobs AS the service accounts.
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="$MEMBER" \
        --role="roles/iam.serviceAccountUser" \
        --quiet

    echo "  Done."
    echo ""
done < "$MEMBERS_FILE"

echo "Success: The team now has equal admin power over all project resources."
