import logging
import sys
from pathlib import Path

def setup_logger(name: str = "EdgeAIVision", log_file: str = "edge_vision.log") -> logging.Logger:
    """Sets up a standardized logger for the project."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Default logger instance
logger = setup_logger()
