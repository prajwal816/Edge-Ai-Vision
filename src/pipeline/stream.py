import cv2
import time
import yaml
from pathlib import Path
from src.inference.engine import InferenceEngine
from src.utils.logger import logger

def run_pipeline(config_path: str = "configs/config.yaml"):
    """
    Executes the OpenCV video processing pipeline.
    Connects to video source, processes frames, and renders bounding boxes.
    """
    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    inf_cfg = config.get("inference", {})
    model_path = inf_cfg.get("model_path", "runs/detect/train/weights/best.onnx")
    conf_thres = inf_cfg.get("confidence_threshold", 0.25)
    iou_thres = inf_cfg.get("iou_threshold", 0.45)
    source = inf_cfg.get("source", "0")
    
    source = int(source) if str(source).isdigit() else source
    
    if not Path(model_path).exists():
        logger.error(f"Model path does not exist: {model_path}")
        logger.info("Skipping stream initialization. Please train and export model first.")
        return
        
    engine = InferenceEngine(model_path, conf_thres, iou_thres)
    
    logger.info(f"Connecting to video source: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return
        
    logger.info("Starting inference stream. Press 'q' to exit.")
    
    # We'll just do a dry raw-output inference without full NMS logic
    # mostly to demonstrate processing overhead and bounding box mock rendering
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame. Exiting stream.")
            break
            
        start_time = time.time()
        
        try:
            # inference call
            preds = engine.infer(frame)
            
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0
            
            # Simulated rendering info
            cv2.putText(
                frame, 
                f"FPS: {fps:.1f}", 
                (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # NOTE: For environment without UI, cv2.imshow will fail. 
            # In headless mode like docker, comment this out.
            try:
                cv2.imshow("Edge AI Vision Stream", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Stream terminated by user.")
                    break
            except cv2.error:
                logger.warning("Headless environment detected. Cannot display video. Printing FPS instead: " + f"{fps:.1f}")
                # We break after one frame if headless, otherwise loop forever
                break
                
        except Exception as e:
            logger.error(f"Error during stream inference: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()
