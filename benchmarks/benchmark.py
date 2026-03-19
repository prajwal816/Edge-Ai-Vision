import time
import numpy as np
import cv2
from pathlib import Path
from src.inference.engine import InferenceEngine
from src.utils.logger import logger

def benchmark_inference(model_path: str, iterations: int = 100):
    """
    Benchmarks raw inference speed (FPS and Latency) for the edge model.
    """
    if not Path(model_path).exists():
        logger.error(f"Cannot benchmark. Model not found at {model_path}")
        return

    logger.info(f"Starting benchmark on model: {model_path} for {iterations} iterations")
    engine = InferenceEngine(model_path)
    
    # Create dummy input frame
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warmup
    logger.info("Warming up model...")
    for _ in range(10):
        engine.infer(dummy_frame)
        
    logger.info("Running iterations...")
    latencies = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        engine.infer(dummy_frame)
        end = time.perf_counter()
        latencies.append(end - start)
        
    avg_latency = np.mean(latencies) * 1000  # ms
    p99_latency = np.percentile(latencies, 99) * 1000 # ms
    fps = 1.0 / np.mean(latencies)
    
    logger.info("=== Benchmark Results ===")
    logger.info(f"Total Iterations : {iterations}")
    logger.info(f"Average Latency  : {avg_latency:.2f} ms")
    logger.info(f"P99 Latency      : {p99_latency:.2f} ms")
    logger.info(f"Average FPS      : {fps:.2f}")
    logger.info("=========================")

if __name__ == "__main__":
    benchmark_inference("runs/detect/train/weights/best.onnx")
