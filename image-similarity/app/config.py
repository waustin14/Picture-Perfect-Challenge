"""
Configuration constants for the image similarity service.
"""
import os

# Model settings
MODEL_TYPE = os.getenv("MODEL_TYPE", "dino_vitb16")  # Single-backbone for speed
DEVICE = os.getenv("DEVICE", "cuda")  # "cuda" or "cpu"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")

# Image settings
INPUT_SIZE = (224, 224)  # DreamSim expected input size
ORIGINAL_SIZE = (1024, 1024)  # Expected input image size

# API settings
MAX_PAIRS_PER_REQUEST = int(os.getenv("MAX_PAIRS_PER_REQUEST", "128"))
BATCH_SIZE_LIMIT = int(os.getenv("BATCH_SIZE_LIMIT", "64"))  # For chunking large batches

# Request timeout (seconds)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# URL fetch settings
URL_FETCH_TIMEOUT = int(os.getenv("URL_FETCH_TIMEOUT", "10"))
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "20"))
