import yaml
from pathlib import Path
from src.models.yolo_model import EdgeYOLOModel
from src.utils.logger import logger

def run_export(config_path: str = "configs/config.yaml", model_weights: str = None):
    """
    Export the YOLOv8 model according to the optimization configurations.
    Default export format is ONNX with INT8 quantization for edge devices.
    """
    logger.info(f"Loading optimization config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    opt_cfg = config.get("optimization", {})
    export_format = opt_cfg.get("export_format", "onnx")
    half = opt_cfg.get("half", False)
    int8 = opt_cfg.get("int8", True)
    
    # Use best.pt if no weights are explicitly provided
    if not model_weights:
        # Check standard YOLO output path
        default_weights = Path("runs/detect/train/weights/best.pt")
        if default_weights.exists():
            model_weights = str(default_weights)
        else:
            logger.warning("best.pt not found, defaulting to yolov8n.pt")
            model_weights = "yolov8n.pt"
            
    logger.info(f"Initializing model with weights: {model_weights}")
    model = EdgeYOLOModel(weights=model_weights)
    
    logger.info("Starting model export process...")
    try:
        exported_path = model.export(format=export_format, int8=int8, half=half)
        logger.info(f"Model successfully exported to {exported_path}")
        return exported_path
    except Exception as e:
        logger.error(f"Failed to export model: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    run_export()
