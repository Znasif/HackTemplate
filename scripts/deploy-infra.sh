#!/bin/bash
# Script to deploy CloudFormation stack for VidServer infrastructure

set -e

# Configuration
STACK_NAME="vidserver-infra"
REGION="us-east-1"
TEMPLATE_FILE="../infrastructure/cloudformation.yml"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if the stack exists
if aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION &> /dev/null; then
    # Update the stack
    echo "Updating the CloudFormation stack..."
    aws cloudformation update-stack \
        --stack-name $STACK_NAME \
        --template-body file://$TEMPLATE_FILE \
        --capabilities CAPABILITY_IAM \
        --region $REGION
else
    # Create the stack
    echo "Creating the CloudFormation stack..."
    aws cloudformation create-stack \
        --stack-name $STACK_NAME \
        --template-body file://$TEMPLATE_FILE \
        --capabilities CAPABILITY_IAM \
        --region $REGION
fi

# Wait for the stack to complete
echo "Waiting for stack operation to complete..."
aws cloudformation wait stack-create-complete --stack-name $STACK_NAME --region $REGION || \
aws cloudformation wait stack-update-complete --stack-name $STACK_NAME --region $REGION

# Get the outputs
echo "Stack deployment completed. Outputs:"
aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query "Stacks[0].Outputs" \
    --region $REGION
