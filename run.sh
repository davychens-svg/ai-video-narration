#!/bin/bash

# Inerbee Video Narrator - Quick Start Script

echo "🚀 Starting Inerbee Video Narrator..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel -q

# Install/update dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create logs directory if it doesn't exist
mkdir -p logs

# Start server
echo "Starting server on http://localhost:8001"
echo "Press Ctrl+C to stop"
echo ""

cd server
python main.py
