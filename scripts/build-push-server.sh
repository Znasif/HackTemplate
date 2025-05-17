#!/bin/bash
# Script to build and push server Docker image manually

set -e

# Configuration
REGION="us-east-1"
STACK_NAME="vidserver-infra"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install it first."
    exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Get ECR repository URI from CloudFormation stack
ECR_REPOSITORY_URI=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION --query "Stacks[0].Outputs[?ExportName=='production-ECRRepositoryUri'].OutputValue" --output text)

if [ -z "$ECR_REPOSITORY_URI" ]; then
    echo "Could not retrieve ECR repository URI. Make sure the CloudFormation stack is deployed."
    exit 1
fi

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $(echo $ECR_REPOSITORY_URI | cut -d'/' -f1)

# Build Docker image
echo "Building Docker image..."
cd ../server
docker build -t $ECR_REPOSITORY_URI:latest .

# Push Docker image to ECR
echo "Pushing Docker image to ECR..."
docker push $ECR_REPOSITORY_URI:latest

echo "Docker image pushed to ECR successfully: $ECR_REPOSITORY_URI:latest"
