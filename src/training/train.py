import yaml
import os
from pathlib import Path
from src.models.yolo_model import EdgeYOLOModel
from src.utils.logger import logger

def simulate_dataset(config_path: str):
    """
    Simulates the existence of an 8K image dataset by generating a valid data.yaml 
    for Ultralytics YOLO if the paths don't exist.
    """
    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    dataset_cfg = config.get("dataset", {})
    
    # We create a generic dataset.yaml for YOLO consumption
    data_yaml_path = Path("data/dataset.yaml")
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    nc = dataset_cfg.get("nc", 80)
    names = dataset_cfg.get("names", [f"class_{i}" for i in range(nc)])
    
    # Using absolute paths as YOLO expects
    yolo_data = {
        "train": str(Path(dataset_cfg.get("train_images", "data/train/images")).absolute()),
        "val": str(Path(dataset_cfg.get("val_images", "data/val/images")).absolute()),
        "nc": nc,
        "names": names
    }
    
    with open(data_yaml_path, "w") as f:
        yaml.dump(yolo_data, f)
        
    logger.info(f"Generated dataset config at {data_yaml_path}")
    return str(data_yaml_path), config

def run_training(config_path: str = "configs/config.yaml"):
    data_yaml_path, config = simulate_dataset(config_path)
    
    train_cfg = config.get("training", {})
    weights = train_cfg.get("weights", "yolov8n.pt")
    epochs = train_cfg.get("epochs", 50)
    batch_size = train_cfg.get("batch_size", 16)
    img_size = train_cfg.get("img_size", 640)
    device = train_cfg.get("device", "cpu")
    
    logger.info(f"Initializing training with model: {weights}")
    model = EdgeYOLOModel(weights=weights)
    
    model.train(
        data_yaml=data_yaml_path,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=img_size,
        device=device
    )

if __name__ == "__main__":
    run_training()
