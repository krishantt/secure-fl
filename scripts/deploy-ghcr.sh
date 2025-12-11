#!/bin/bash

# One-time deployment script for GHCR
# Usage: ./deploy-now.sh [TAG]

set -euo pipefail

TAG="${1:-latest}"
REGISTRY="ghcr.io"

# Get repo info
REPO_URL=$(git config --get remote.origin.url)
if [[ "$REPO_URL" =~ github\.com[:/](.+)/(.+) ]]; then
    OWNER="${BASH_REMATCH[1]}"
    REPO="${BASH_REMATCH[2]%.git}"
else
    echo "Error: Could not parse GitHub repository"
    exit 1
fi

IMAGE_NAME=$(echo "$OWNER/$REPO" | tr '[:upper:]' '[:lower:]')
FULL_IMAGE="$REGISTRY/$IMAGE_NAME"

echo "🚀 Deploying to GHCR..."
echo "📦 Image: $FULL_IMAGE:$TAG"

# Check token
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo "❌ GITHUB_TOKEN required. Set with:"
    echo "   export GITHUB_TOKEN=your_token_here"
    exit 1
fi

# Login
echo "🔑 Logging in..."
echo "$GITHUB_TOKEN" | docker login "$REGISTRY" -u "$OWNER" --password-stdin

# Setup buildx for multi-platform
echo "🔧 Setting up buildx..."
docker buildx create --name multiarch --driver docker-container --use 2>/dev/null || docker buildx use multiarch

# Build and push
echo "🔨 Building and pushing..."
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag "$FULL_IMAGE:$TAG" \
    --tag "$FULL_IMAGE:latest" \
    --push \
    .

echo "✅ Deployed successfully!"
echo "📋 Image: $FULL_IMAGE:$TAG"
echo "🔄 Pull: docker pull $FULL_IMAGE:$TAG"
echo "▶️  Run:  docker run --rm $FULL_IMAGE:$TAG"
echo "🌐 Make public: https://github.com/$OWNER/$REPO/pkgs/container/$REPO"
