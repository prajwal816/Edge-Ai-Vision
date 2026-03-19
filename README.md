# Edge AI Computer Vision — Environment Understanding

An optimized real-time object detection system designed for edge devices (such as Raspberry Pi) using YOLOv8.

## Features
- **YOLOv8 Training**: Fine-tune on custom datasets with automated configuration simulation.
- **Optimization**: Export trained models to ONNX with INT8 quantization for superior edge performance.
- **OpenCV Pipeline**: A real-time video inference and frame streaming pipeline for testing model capabilities.
- **Edge Deployment**: Scripts to easily simulate and deploy to Raspberry Pi CPU.
- **Performance Tracking**: Measure FPS, latency, and evaluate bounds with specialized benchmarking tools.

## Project Structure
```text
edge-ai-vision/
├── benchmarks/         # FPS and Latency benchmarking
├── configs/            # YAML configurations (e.g., config.yaml)
├── data/               # Dataset storage (simulated 8K images)
├── deployment/         # Edge devices (Pi) setup scripts
├── src/                # Core implementation logic
│   ├── inference/      # ONNX Runtime CPU engine wrapper
│   ├── models/         # YOLOv8 wrappers
│   ├── optimization/   # ONNX and INT8 export handlers
│   ├── pipeline/       # OpenCV streaming logic
│   └── utils/          # Standard logging
├── main.py             # CLI interface orchestrator
├── README.md           # This file
└── requirements.txt    # Project dependencies
```

## Setup & Installation

**Prerequisites:** Python 3.9+ 

1. **Clone the repository:**
   ```bash
   git clone <repo>
   cd Edge-Ai-Vision
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Info
For this project, we simulated an 8K image environment. The data loader requires image paths configured in `configs/config.yaml`. The schema defaults to 80 COCO classes, configurable directly in the YAML file.

## Usage Guide
The project relies on a centralized CLI via `main.py`.

### 1. Training Steps
Train the model on your custom dataset defined in `config.yaml`.
```bash
python main.py train --config configs/config.yaml
```

### 2. Optimization Steps
Export the model to ONNX with CPU-optimized INT8 quantization for standard edge deployment.
```bash
python main.py export --config configs/config.yaml --weights runs/detect/train/weights/best.pt
```

### 3. Edge Deployment Guide
For deploying onto a Raspberry Pi (or similar ARM-based hardware):
1. Copy the repository to the device.
2. Run the deployment script to prepare the environment:
   ```bash
   cd deployment
   chmod +x pi_setup.sh
   ./pi_setup.sh
   ```
3. Source the environment and run inference streams natively via CPU.

### 4. Real-time Inference Pipeline
To run a real-time OpenCV pipeline locally (or on the edge device). Replace `source` in `config.yaml` to `"0"` for webcams or an RTSP stream link.
```bash
python main.py stream --config configs/config.yaml
```

### 5. Performance Results & Benchmarking
Evaluate CPU speed, exact latency, and frames per second limit.
```bash
python main.py benchmark --weights runs/detect/train/weights/best.onnx --iters 100
```
