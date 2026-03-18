import onnxruntime as ort
import numpy as np
import cv2
from src.utils.logger import logger

class InferenceEngine:
    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45):
        """Initializes ONNX Runtime session for CPU execution."""
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        logger.info(f"Loading ONNX model from {model_path} for CPU inference")
        
        # Configure session options for CPU optimization
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4  # Typical for RPi 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.session = ort.InferenceSession(
                model_path, 
                sess_options, 
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info("ONNX Runtime session initialized successfully")
        except Exception as e:
            logger.error(f"Failed to load ONNX model. Error: {e}")
            raise

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Prepares image for YOLOv8 ONNX model."""
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1)) # HWC to CHW
        img_batch = np.expand_dims(img_chw, axis=0) # Add batch dimension
        return img_batch

    def infer(self, img: np.ndarray) -> np.ndarray:
        """Runs inference via ONNX Runtime."""
        input_tensor = self.preprocess(img)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return outputs[0]
