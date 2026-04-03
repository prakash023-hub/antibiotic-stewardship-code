#!/usr/bin/env bash
# deploy.sh — push this environment to a HuggingFace Space
#
# Prerequisites:
#   1. huggingface-hub installed:  pip install huggingface_hub
#   2. Logged in:                  huggingface-cli login
#   3. A Docker Space created at   https://huggingface.co/new-space
#      (SDK = Docker, visibility = Public)
#
# Usage:
#   chmod +x deploy.sh
#   HF_REPO="your-username/your-space-name" ./deploy.sh

set -e

HF_REPO="${HF_REPO:-}"

if [ -z "$HF_REPO" ]; then
  echo "❌  Set HF_REPO before running, e.g.:"
  echo "    HF_REPO=myusername/antibiotic-stewardship ./deploy.sh"
  exit 1
fi

echo "🚀  Deploying to HuggingFace Space: $HF_REPO"

TMPDIR=$(mktemp -d)
git clone "https://huggingface.co/spaces/$HF_REPO" "$TMPDIR/space"

rsync -av --exclude='__pycache__' --exclude='.git' --exclude='deploy.sh' \
  "$(dirname "$0")/" "$TMPDIR/space/"

cd "$TMPDIR/space"
git add -A
git commit -m "Deploy antibiotic stewardship OpenEnv"
git push

echo ""
echo "✅  Done! Your space will be live in ~2 minutes at:"
echo "    https://huggingface.co/spaces/$HF_REPO"
echo ""
echo "Test it with:"
echo "    curl https://${HF_REPO/\//--}.hf.space/health"
echo ""
echo "Run inference with:"
echo "    export API_BASE_URL=https://${HF_REPO/\//--}.hf.space"
echo "    export OPENAI_API_KEY=sk-..."
echo "    python inference.py"
