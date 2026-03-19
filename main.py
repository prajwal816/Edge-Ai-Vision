import argparse
import sys
from src.utils.logger import logger
from src.training.train import run_training
from src.optimization.export import run_export
from src.pipeline.stream import run_pipeline
from benchmarks.benchmark import benchmark_inference

def main():
    parser = argparse.ArgumentParser(
        description="Edge AI Computer Vision - Environment understanding system."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the YOLOv8 model")
    train_parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX with INT8 quantization")
    export_parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    export_parser.add_argument("--weights", type=str, default=None, help="Weights to export (default: best.pt)")
    
    # Infer/Stream command
    stream_parser = subparsers.add_parser("stream", help="Run real-time inference pipeline")
    stream_parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run FPS and latency benchmarks")
    bench_parser.add_argument("--weights", type=str, default="runs/detect/train/weights/best.onnx", help="Path to model")
    bench_parser.add_argument("--iters", type=int, default=100, help="Number of benchmark iterations")

    args = parser.parse_args()

    if args.command == "train":
        logger.info("Executing training workflow...")
        run_training(args.config)
        
    elif args.command == "export":
        logger.info("Executing optimization and export workflow...")
        run_export(args.config, args.weights)
        
    elif args.command == "stream":
        logger.info("Executing streaming pipeline...")
        run_pipeline(args.config)
        
    elif args.command == "benchmark":
        logger.info("Executing benchmark...")
        benchmark_inference(args.weights, args.iters)
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
