#!/bin/bash
# Script to set up GitHub repository secrets for CI/CD

set -e

# Configuration
REPO_OWNER="your-github-username"
REPO_NAME="vidServer"
REGION="us-east-1"
STACK_NAME="vidserver-infra"

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI is not installed. Please install it first."
    exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Make sure the user is logged in to GitHub CLI
gh auth status || gh auth login

# Get outputs from CloudFormation stack
echo "Getting outputs from CloudFormation stack..."
ECR_REPOSITORY_URI=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION --query "Stacks[0].Outputs[?ExportName=='production-ECRRepositoryUri'].OutputValue" --output text)
S3_BUCKET_NAME=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION --query "Stacks[0].Outputs[?ExportName=='production-S3BucketName'].OutputValue" --output text)
CLOUDFRONT_DISTRIBUTION_ID=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION --query "Stacks[0].Outputs[?ExportName=='production-CloudFrontDistributionId'].OutputValue" --output text)
ECS_CLUSTER_NAME="production-vidserver-cluster"
ECS_SERVICE_NAME="production-vidserver-service"

# Get ECR repository name from the URI
ECR_REPOSITORY_SERVER=$(echo $ECR_REPOSITORY_URI | cut -d'/' -f2)

# Set GitHub secrets
echo "Setting GitHub secrets..."

# AWS Credentials (these should be set manually through GitHub UI for security)
echo "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY manually in the GitHub repository settings."

# AWS Region
gh secret set AWS_REGION --body "$REGION" --repo "$REPO_OWNER/$REPO_NAME"

# ECR Repository
gh secret set ECR_REPOSITORY_SERVER --body "$ECR_REPOSITORY_SERVER" --repo "$REPO_OWNER/$REPO_NAME"

# S3 Bucket
gh secret set S3_BUCKET --body "$S3_BUCKET_NAME" --repo "$REPO_OWNER/$REPO_NAME"

# CloudFront Distribution ID
gh secret set CLOUDFRONT_DISTRIBUTION_ID --body "$CLOUDFRONT_DISTRIBUTION_ID" --repo "$REPO_OWNER/$REPO_NAME"

# ECS Cluster and Service
gh secret set ECS_CLUSTER --body "$ECS_CLUSTER_NAME" --repo "$REPO_OWNER/$REPO_NAME"
gh secret set ECS_SERVICE_SERVER --body "$ECS_SERVICE_NAME" --repo "$REPO_OWNER/$REPO_NAME"

echo "GitHub secrets have been set up successfully."
