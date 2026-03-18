from ultralytics import YOLO
from src.utils.logger import logger

class EdgeYOLOModel:
    def __init__(self, weights="yolov8n.pt"):
        """Initialize YOLOv8 model for Edge AI project."""
        logger.info(f"Loading YOLO model with weights: {weights}")
        self.model = YOLO(weights)
    
    def train(self, data_yaml: str, epochs: int = 50, batch_size: int = 16, imgsz: int = 640, device: str = "cpu"):
        """Fine-tune the model on custom dataset."""
        logger.info(f"Starting training on {data_yaml} for {epochs} epochs.")
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            project="runs/detect",
            name="train"
        )
        logger.info("Training completed.")
        return results

    def export(self, format="onnx", int8=True, half=False, workspace=4):
        """Export the model to the target format with optimizations."""
        logger.info(f"Exporting model to {format}. INT8: {int8}, Half: {half}")
        path = self.model.export(format=format, int8=int8, half=half, workspace=workspace)
        logger.info(f"Model exported to {path}")
        return path

    def predict(self, source, conf=0.25, iou=0.45):
        """Run inference."""
        return self.model.predict(source, conf=conf, iou=iou)
