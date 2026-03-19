#!/bin/bash
# Raspberry Pi / Edge System Setup Script for YOLOv8 Edge AI Vision
set -e

echo "[INFO] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

echo "[INFO] Installing necessary system libraries for OpenCV and ONNX..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libopencv-dev \
    libatlas-base-dev \
    git \
    wget

echo "[INFO] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[INFO] Upgrading pip..."
pip install --upgrade pip

echo "[INFO] Installing Python dependencies..."
# For Raspberry Pi, we install standard onnxruntime which natively uses ARM NEON/CPU
pip install -r ../requirements.txt

echo "[INFO] Setup Complete! Activate your environment via: source venv/bin/activate"
