#!/bin/bash
# EC2 User Data Script - Initialize instance for trading bot

# Update system
yum update -y

# Install Python 3.9+
amazon-linux-extras install python3.8 -y
yum install -y python3-pip git

# Create trading user
useradd -m trader
su - trader

# Set up environment
echo "Instance initialized for Kalshi Trading Bot"
echo "Ready for deployment"
