#!/bin/bash
# Deploy Kalshi MM Trading Bot to AWS us-east-2
# Phase 1: Live Trading with 8 Categories

set -e

REGION="us-east-2"
INSTANCE_TYPE="t3.small"
AMI_ID="ami-0772d6acfbccb1275"  # Amazon Linux 2023 in us-east-2
KEY_NAME="kalshi-trading-bot"
SECURITY_GROUP="kalshi-bot-sg"

echo "=========================================="
echo "Deploying Kalshi Trading Bot to AWS"
echo "Region: $REGION"
echo "Mode: LIVE TRADING (REAL MONEY)"
echo "Categories: 8"
echo "=========================================="
echo

# Check for AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS credentials not configured"
    echo "Run: aws configure"
    exit 1
fi

# Create security group if needed
echo "Setting up security group..."
if ! aws ec2 describe-security-groups --region $REGION --group-names $SECURITY_GROUP &>/dev/null; then
    echo "Creating security group: $SECURITY_GROUP"
    SG_ID=$(aws ec2 create-security-group \
        --region $REGION \
        --group-name $SECURITY_GROUP \
        --description "Kalshi Trading Bot Security Group" \
        --query 'GroupId' --output text)

    # Allow SSH (for monitoring)
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0

    echo "Security group created: $SG_ID"
else
    echo "Security group already exists: $SECURITY_GROUP"
fi

# Create EC2 key pair if needed
if ! aws ec2 describe-key-pairs --region $REGION --key-names $KEY_NAME &>/dev/null; then
    echo "Creating EC2 key pair: $KEY_NAME"
    aws ec2 create-key-pair \
        --region $REGION \
        --key-name $KEY_NAME \
        --query 'KeyMaterial' \
        --output text > ~/.ssh/${KEY_NAME}.pem
    chmod 400 ~/.ssh/${KEY_NAME}.pem
    echo "Key pair saved to: ~/.ssh/${KEY_NAME}.pem"
else
    echo "Key pair already exists: $KEY_NAME"
fi

# Create deployment package
echo
echo "Creating deployment package..."
cd "$(dirname "$0")/.."
tar czf /tmp/kalshi-bot-deploy.tar.gz \
    rl_bot/*.py \
    rl_bot/mm_checkpoints/*.zip \
    .env \
    requirements.txt \
    deploy/start_live_trading.sh

echo "Deployment package created: /tmp/kalshi-bot-deploy.tar.gz"
echo "Size: $(du -h /tmp/kalshi-bot-deploy.tar.gz | cut -f1)"

# Launch EC2 instance
echo
echo "Launching EC2 instance..."
echo "Instance type: $INSTANCE_TYPE"
echo "Region: $REGION"

INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=kalshi-trading-bot}]" \
    --user-data file://deploy/user_data.sh \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --region $REGION \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance running at: $PUBLIC_IP"
echo "Waiting 60 seconds for instance initialization..."
sleep 60

# Copy deployment package
echo
echo "Deploying code to instance..."
scp -i ~/.ssh/${KEY_NAME}.pem \
    -o StrictHostKeyChecking=no \
    /tmp/kalshi-bot-deploy.tar.gz \
    ec2-user@${PUBLIC_IP}:~/

# Connect and setup
echo "Setting up trading bot..."
ssh -i ~/.ssh/${KEY_NAME}.pem \
    -o StrictHostKeyChecking=no \
    ec2-user@${PUBLIC_IP} << 'EOF'
# Extract deployment
cd ~
tar xzf kalshi-bot-deploy.tar.gz
rm kalshi-bot-deploy.tar.gz

# Install Python and dependencies
sudo yum update -y
sudo yum install -y python3 python3-pip git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up live trading environment
sed -i 's/PAPER_MODE=true/PAPER_MODE=false/' .env

# Start trading bot in background
chmod +x deploy/start_live_trading.sh
nohup ./deploy/start_live_trading.sh > trading.log 2>&1 &

echo "Trading bot started!"
echo "Logs: ~/trading.log"
EOF

echo
echo "=========================================="
echo "DEPLOYMENT COMPLETE"
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "Region: $REGION"
echo
echo "Monitor logs:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo "  tail -f trading.log"
echo
echo "Stop trading:"
echo "  pkill -f live_trader_v2.py"
echo
echo "Terminate instance:"
echo "  aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"
echo
echo "⚠️  WARNING: LIVE TRADING WITH REAL MONEY IS NOW ACTIVE"
echo "=========================================="
