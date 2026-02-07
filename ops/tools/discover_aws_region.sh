#!/usr/bin/env bash
# Script to discover AWS region from existing resources

set -euo pipefail

echo "=== AWS Region Discovery ==="
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &>/dev/null; then
    echo "❌ AWS credentials not configured yet."
    echo "   Run 'aws configure' first with your access key and secret key."
    echo "   For region, you can use 'us-east-1' as a temporary default."
    exit 1
fi

echo "✓ AWS credentials configured"
echo ""

# Try to get region from S3 bucket
echo "Checking S3 bucket location..."
if aws s3api get-bucket-location --bucket procedure-suite-models 2>/dev/null; then
    echo "✓ Found region from S3 bucket"
    echo ""
fi

# Try to list EC2 instances across common regions
echo "Checking EC2 instances in common regions..."
REGIONS=("us-east-1" "us-west-2" "us-west-1" "eu-west-1" "ap-southeast-1" "ap-northeast-1")

for region in "${REGIONS[@]}"; do
    echo -n "Checking $region... "
    count=$(aws ec2 describe-instances --region "$region" --query 'length(Reservations[*].Instances[*])' --output text 2>/dev/null || echo "0")
    if [ "$count" != "0" ] && [ "$count" != "None" ]; then
        echo "✓ Found $count instance(s)"
        echo "  → Your region is likely: $region"
    else
        echo "no instances"
    fi
done

echo ""
echo "=== Recommendation ==="
echo "If you found instances above, use that region."
echo "Otherwise, common defaults are:"
echo "  - us-east-1 (N. Virginia) - Most common"
echo "  - us-west-2 (Oregon) - Common for West Coast"
echo ""
echo "You can configure with: aws configure set region <region-name>"
