#!/usr/bin/env bash
# deploy.sh — Set up a Kalshi data collector on an EC2 on-demand instance.
#
# Prerequisites:
#   - AWS CLI configured with credentials (aws configure)
#   - An S3 bucket created:  aws s3 mb s3://kalshi-data-YOUR_ACCOUNT
#   - Your Kalshi RSA key and .env ready to copy
#
# Usage:
#   chmod +x data_collector/deploy.sh
#   ./data_collector/deploy.sh
#
# Cost: ~$3-5/month (t3.micro on-demand + S3 Standard-IA)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------
INSTANCE_TYPE="t3.small"
REGION="us-east-2"
AMI="ami-08e3f17ecdd66f6c8"  # Amazon Linux 2023 (us-east-2, 2026-06-22)
KEY_NAME="kalshi-collector"   # your EC2 key pair name
SECURITY_GROUP="kalshi-collector-sg"
S3_BUCKET="${AWS_S3_BUCKET:-kalshi-data-prod}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Kalshi Data Collector Deployment ==="
echo "Region:   $REGION"
echo "Instance: $INSTANCE_TYPE (on-demand)"
echo "Bucket:   $S3_BUCKET"
echo ""

# ---------------------------------------------------------------------------
# 1. Create security group (if it doesn't exist)
# ---------------------------------------------------------------------------
echo "[1/5] Security group..."
SG_ID=$(aws ec2 describe-security-groups \
    --region "$REGION" \
    --group-names "$SECURITY_GROUP" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || true)

if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    SG_ID=$(aws ec2 create-security-group \
        --region "$REGION" \
        --group-name "$SECURITY_GROUP" \
        --description "Kalshi data collector - SSH only" \
        --query 'GroupId' --output text)
    # Allow SSH from anywhere (restrict to your IP in production)
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0
    echo "  Created SG: $SG_ID"
else
    echo "  Using existing SG: $SG_ID"
fi

# ---------------------------------------------------------------------------
# 2. Create IAM role for S3 access (if it doesn't exist)
# ---------------------------------------------------------------------------
echo "[2/5] IAM role..."
ROLE_NAME="kalshi-collector-role"
INSTANCE_PROFILE="kalshi-collector-profile"

if ! aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
    # Trust policy: allow EC2 to assume this role
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }'
    # Attach S3 write policy scoped to our bucket
    aws iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name "s3-write" \
        --policy-document "{
            \"Version\": \"2012-10-17\",
            \"Statement\": [{
                \"Effect\": \"Allow\",
                \"Action\": [\"s3:PutObject\", \"s3:GetObject\", \"s3:ListBucket\"],
                \"Resource\": [
                    \"arn:aws:s3:::${S3_BUCKET}\",
                    \"arn:aws:s3:::${S3_BUCKET}/*\"
                ]
            }]
        }"
    echo "  Created role: $ROLE_NAME"
else
    echo "  Using existing role: $ROLE_NAME"
fi

# Create instance profile if needed
if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" &>/dev/null; then
    aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE"
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE" \
        --role-name "$ROLE_NAME"
    echo "  Created instance profile: $INSTANCE_PROFILE"
    echo "  Waiting 10s for IAM propagation..."
    sleep 10
fi

# ---------------------------------------------------------------------------
# 3. Build user-data script (runs on first boot)
# ---------------------------------------------------------------------------
echo "[3/5] Preparing user-data..."

USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -euo pipefail

# Install Python and dependencies
dnf install -y python3.11 python3.11-pip git
python3.11 -m pip install --upgrade pip

# Create app directory
mkdir -p /opt/kalshi
cd /opt/kalshi

# Install Python packages
python3.11 -m pip install \
    websockets \
    cryptography \
    polars \
    python-dotenv \
    boto3 \
    requests

# The .env and RSA key must be copied separately (see step 5 below)

# Create systemd service
cat > /etc/systemd/system/kalshi-collector.service <<'EOF'
[Unit]
Description=Kalshi Data Collector
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/kalshi
ExecStart=/usr/bin/python3.11 -m data_collector.collect --bucket BUCKET_PLACEHOLDER --region us-east-2 --flush-interval 300
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

# Graceful shutdown: send SIGTERM, wait 60s for flush
KillSignal=SIGTERM
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
EOF

echo "Setup complete. Copy code + credentials, then: systemctl enable --now kalshi-collector"
USERDATA
)

# Replace bucket placeholder
USER_DATA="${USER_DATA//BUCKET_PLACEHOLDER/$S3_BUCKET}"

# ---------------------------------------------------------------------------
# 4. Launch on-demand instance
# ---------------------------------------------------------------------------
echo "[4/5] Launching on-demand instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile Name="$INSTANCE_PROFILE" \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=kalshi-collector}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "  Instance: $INSTANCE_ID"
echo "  Waiting for public IP..."

aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "  Public IP: $PUBLIC_IP"

# ---------------------------------------------------------------------------
# 5. Copy code and credentials
# ---------------------------------------------------------------------------
echo "[5/5] Copy code to instance..."
echo ""
echo "Run the following commands once the instance is ready (~60s):"
echo ""
echo "  # Copy project files"
echo "  scp -i ~/.ssh/${KEY_NAME}.pem -r ${PROJECT_DIR}/{data_collector,authentication_to_kalshi,config.py,.env,rsa_keys} ec2-user@${PUBLIC_IP}:/opt/kalshi/"
echo ""
echo "  # SSH in and start the service"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo "  sudo systemctl enable --now kalshi-collector"
echo ""
echo "=== Monitoring ==="
echo "  # View logs"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP} 'sudo journalctl -u kalshi-collector -f'"
echo ""
echo "  # Check S3 data"
echo "  aws s3 ls s3://${S3_BUCKET}/trades/ --recursive"
echo "  aws s3 ls s3://${S3_BUCKET}/orderbooks/ --recursive"
echo ""
echo "  # Stop collector"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP} 'sudo systemctl stop kalshi-collector'"
echo ""
echo "  # Terminate instance"
echo "  aws ec2 terminate-instances --region ${REGION} --instance-ids ${INSTANCE_ID}"
