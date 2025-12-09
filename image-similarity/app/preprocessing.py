"""
Preprocessing module for image loading and transformations.

Handles:
- Input validation
- Image decoding (PNG/JPEG → RGB tensor)
- Resize to 224x224
- Normalization according to DreamSim's transforms
"""
import base64
import io
from typing import Optional, Callable
import httpx
from PIL import Image
import torch

from .config import INPUT_SIZE, URL_FETCH_TIMEOUT, MAX_IMAGE_SIZE_MB


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or processed."""
    pass


def is_url(s: str) -> bool:
    """Check if string is a URL."""
    return s.startswith(("http://", "https://"))


def decode_base64_image(b64_string: str) -> Image.Image:
    """
    Decode a base64-encoded image string to a PIL Image.
    
    Args:
        b64_string: Base64-encoded PNG/JPEG image data
        
    Returns:
        PIL.Image in RGB mode
        
    Raises:
        ImageLoadError: If decoding fails
    """
    try:
        # Handle data URI format if present
        if "," in b64_string and b64_string.startswith("data:"):
            b64_string = b64_string.split(",", 1)[1]
        
        image_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (handles RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return image
    except Exception as e:
        raise ImageLoadError(f"Failed to decode base64 image: {e}")


async def fetch_image_from_url(url: str, client: Optional[httpx.AsyncClient] = None) -> Image.Image:
    """
    Fetch an image from a URL.
    
    Args:
        url: HTTP(S) URL to fetch image from
        client: Optional async HTTP client (will create one if not provided)
        
    Returns:
        PIL.Image in RGB mode
        
    Raises:
        ImageLoadError: If fetch or decoding fails
    """
    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=URL_FETCH_TIMEOUT)
        close_client = True
    
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        
        # Check content length if available
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise ImageLoadError(f"Image too large: {int(content_length) / (1024*1024):.1f}MB > {MAX_IMAGE_SIZE_MB}MB limit")
        
        image = Image.open(io.BytesIO(response.content))
        
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return image
    except httpx.HTTPError as e:
        raise ImageLoadError(f"Failed to fetch image from URL: {e}")
    except Exception as e:
        if isinstance(e, ImageLoadError):
            raise
        raise ImageLoadError(f"Failed to load image from URL: {e}")
    finally:
        if close_client:
            await client.aclose()


async def load_image(source: str, client: Optional[httpx.AsyncClient] = None) -> Image.Image:
    """
    Load an image from either a URL or base64 string.
    
    Args:
        source: Either a URL (http:// or https://) or base64-encoded image data
        client: Optional async HTTP client for URL fetching
        
    Returns:
        PIL.Image in RGB mode
        
    Raises:
        ImageLoadError: If loading fails
    """
    if is_url(source):
        return await fetch_image_from_url(source, client)
    else:
        return decode_base64_image(source)


def preprocess_image(image: Image.Image, preprocess_fn: Callable) -> torch.Tensor:
    """
    Apply DreamSim preprocessing to a PIL Image.
    
    Args:
        image: PIL.Image in RGB mode
        preprocess_fn: DreamSim's preprocess function
        
    Returns:
        Tensor of shape (1, 3, 224, 224), dtype float32
    """
    # DreamSim's preprocess handles resize and normalization
    # It returns a tensor of shape (1, 3, 224, 224)
    return preprocess_fn(image)


def preprocess_image_manual(image: Image.Image) -> torch.Tensor:
    """
    Manually preprocess a PIL Image (fallback if DreamSim preprocess unavailable).
    
    Args:
        image: PIL.Image in RGB mode
        
    Returns:
        Tensor of shape (3, 224, 224), dtype float32
    """
    import torchvision.transforms as T
    
    transform = T.Compose([
        T.Resize(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)


def validate_image(image: Image.Image) -> None:
    """
    Validate that an image meets requirements.
    
    Args:
        image: PIL.Image to validate
        
    Raises:
        ImageLoadError: If validation fails
    """
    if image.mode != "RGB":
        raise ImageLoadError(f"Image must be RGB, got {image.mode}")
    
    # We accept any size as input since we'll resize anyway
    # but log a warning if it's not 1024x1024 as expected
    # (actual validation can be added if strict size checking is needed)
