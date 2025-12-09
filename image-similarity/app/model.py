"""
Model module for DreamSim loading and similarity computation.

Responsibilities:
- Load DreamSim model at startup
- Keep model in GPU memory for the lifetime of the process
- Provide batched compute_distance function
"""
import logging
from typing import Tuple, Callable, Optional
import torch

from .config import MODEL_TYPE, DEVICE, MODEL_CACHE_DIR

logger = logging.getLogger(__name__)


class DreamSimModel:
    """Wrapper for DreamSim model with batched inference support."""
    
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.device = None
        self._initialized = False
    
    def initialize(self, device: Optional[str] = None) -> None:
        """
        Initialize and load the DreamSim model.
        
        Args:
            device: Device to load model on ("cuda" or "cpu"). Defaults to config.DEVICE.
        """
        if self._initialized:
            logger.warning("Model already initialized, skipping re-initialization")
            return
        
        from dreamsim import dreamsim
        
        self.device = device or DEVICE
        
        # Check if CUDA is available when requested
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        
        logger.info(f"Loading DreamSim model (type={MODEL_TYPE}) on {self.device}...")
        
        # Load model with single-backbone variant for speed
        self.model, self.preprocess = dreamsim(
            pretrained=True,
            dreamsim_type=MODEL_TYPE,
            device=self.device,
            cache_dir=MODEL_CACHE_DIR
        )
        
        # Set model to eval mode
        self.model.eval()
        
        self._initialized = True
        logger.info(f"DreamSim model loaded successfully on {self.device}")
    
    def get_preprocess(self) -> Callable:
        """Get the preprocessing function for images."""
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return self.preprocess
    
    @torch.no_grad()
    def compute_distance(
        self, 
        ref_batch: torch.Tensor, 
        gen_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual distance between reference and generated image batches.
        
        Args:
            ref_batch: (N, 3, 224, 224) tensor of reference images on correct device
            gen_batch: (N, 3, 224, 224) tensor of generated images on correct device
            
        Returns:
            (N,) tensor of distances, where 0 = identical, larger = more different
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # Ensure tensors are on the correct device
        ref_batch = ref_batch.to(self.device)
        gen_batch = gen_batch.to(self.device)
        
        # DreamSim model computes pairwise distance
        # The model expects (N, 3, 224, 224) inputs and returns (N,) distances
        distances = self.model(ref_batch, gen_batch)
        
        # Ensure we return a 1D tensor
        if distances.dim() == 0:
            distances = distances.unsqueeze(0)
        
        return distances.cpu()
    
    @torch.no_grad()
    def compute_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute embeddings for a batch of images.
        
        Args:
            batch: (N, 3, 224, 224) tensor of images
            
        Returns:
            (N, D) tensor of embeddings
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        batch = batch.to(self.device)
        embeddings = self.model.embed(batch)
        return embeddings.cpu()
    
    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._initialized
    
    def get_device(self) -> str:
        """Get the device the model is on."""
        return self.device


def distance_to_similarity(d: torch.Tensor) -> torch.Tensor:
    """
    Convert distance to similarity score.
    
    Args:
        d: Distance values (0 = identical, larger = more different)
        
    Returns:
        Similarity scores in (0, 1], where higher is more similar
    """
    return 1.0 / (1.0 + d)


def similarity_to_distance(s: torch.Tensor) -> torch.Tensor:
    """
    Convert similarity to distance score.
    
    Args:
        s: Similarity values in (0, 1]
        
    Returns:
        Distance values (0 = identical, larger = more different)
    """
    return (1.0 / s) - 1.0


# Global model instance
_model_instance: Optional[DreamSimModel] = None


def get_model() -> DreamSimModel:
    """Get or create the global model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = DreamSimModel()
    return _model_instance


def init_model(device: Optional[str] = None) -> DreamSimModel:
    """Initialize the global model instance."""
    model = get_model()
    if not model.is_initialized():
        model.initialize(device)
    return model
