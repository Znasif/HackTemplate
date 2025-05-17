#!/bin/bash
# Script to deploy the client application manually

set -e

# Configuration
REGION="us-east-1"
STACK_NAME="vidserver-infra"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Get S3 bucket name and CloudFront distribution ID from CloudFormation stack
S3_BUCKET_NAME=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION --query "Stacks[0].Outputs[?ExportName=='production-S3BucketName'].OutputValue" --output text)
CLOUDFRONT_DISTRIBUTION_ID=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION --query "Stacks[0].Outputs[?ExportName=='production-CloudFrontDistributionId'].OutputValue" --output text)

if [ -z "$S3_BUCKET_NAME" ] || [ -z "$CLOUDFRONT_DISTRIBUTION_ID" ]; then
    echo "Could not retrieve S3 bucket name or CloudFront distribution ID. Make sure the CloudFormation stack is deployed."
    exit 1
fi

# Deploy client files to S3
echo "Deploying client files to S3..."
cd ../client
aws s3 sync ./ s3://$S3_BUCKET_NAME --delete

# Invalidate CloudFront cache
echo "Invalidating CloudFront cache..."
aws cloudfront create-invalidation --distribution-id $CLOUDFRONT_DISTRIBUTION_ID --paths "/*"

echo "Client application deployed successfully to S3 bucket: $S3_BUCKET_NAME"
echo "CloudFront cache invalidation initiated for distribution: $CLOUDFRONT_DISTRIBUTION_ID"
