#!/usr/bin/env bash
# RouteIQ AWS deployment script
# Usage: ./infra/deploy.sh
set -euo pipefail

STACK_NAME="routeiq"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

# Load .env if present (won't override existing env vars)
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?ERROR: OPENROUTER_API_KEY must be set in .env or environment}"

echo "==> RouteIQ Deployment"
echo "    Stack:  $STACK_NAME"
echo "    Region: $REGION"
echo ""

# ----------------------------------------------------------------
# 1. Ensure SAM CLI is available
# ----------------------------------------------------------------
if ! command -v sam &>/dev/null; then
  echo "ERROR: AWS SAM CLI not found."
  echo "       Install: brew install aws-sam-cli"
  exit 1
fi

# ----------------------------------------------------------------
# 2. SAM build (uses Docker to match Lambda runtime)
# ----------------------------------------------------------------
echo "==> Building with SAM..."
sam build \
  --template infra/template.yaml \
  --use-container

# ----------------------------------------------------------------
# 3. SAM deploy
# ----------------------------------------------------------------
echo "==> Deploying to AWS..."
sam deploy \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --parameter-overrides \
    "OpenRouterApiKey=${OPENROUTER_API_KEY}" \
    "SkipAuth=false" \
  --no-confirm-changeset \
  --no-fail-on-empty-changeset

# ----------------------------------------------------------------
# 4. Print the deployed API URL
# ----------------------------------------------------------------
echo ""
echo "==> Deployment complete!"
API_URL=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='RouteIQApiUrl'].OutputValue" \
  --output text)

echo ""
echo "    RouteIQ API URL: $API_URL"
echo ""
echo "Test with:"
echo "  curl -s -X POST $API_URL/v1/chat/completions \\"
echo "    -H 'Authorization: Bearer <your-routeiq-key>' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"gpt-5.4\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'"
