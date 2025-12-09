"""
Tests for preprocessing module.
"""
import base64
import io
import pytest
from PIL import Image
import torch


def create_test_image(width: int = 224, height: int = 224, color: tuple = (255, 0, 0)) -> Image.Image:
    """Create a simple test image."""
    img = Image.new("RGB", (width, height), color)
    return img


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class TestDecodeBase64Image:
    """Tests for base64 image decoding."""
    
    def test_decode_png(self):
        """Test decoding a PNG image."""
        from app.preprocessing import decode_base64_image
        
        img = create_test_image()
        b64 = image_to_base64(img, "PNG")
        
        decoded = decode_base64_image(b64)
        
        assert decoded.mode == "RGB"
        assert decoded.size == (224, 224)
    
    def test_decode_jpeg(self):
        """Test decoding a JPEG image."""
        from app.preprocessing import decode_base64_image
        
        img = create_test_image()
        b64 = image_to_base64(img, "JPEG")
        
        decoded = decode_base64_image(b64)
        
        assert decoded.mode == "RGB"
        assert decoded.size == (224, 224)
    
    def test_decode_with_data_uri(self):
        """Test decoding base64 with data URI prefix."""
        from app.preprocessing import decode_base64_image
        
        img = create_test_image()
        b64 = image_to_base64(img, "PNG")
        data_uri = f"data:image/png;base64,{b64}"
        
        decoded = decode_base64_image(data_uri)
        
        assert decoded.mode == "RGB"
    
    def test_decode_rgba_converts_to_rgb(self):
        """Test that RGBA images are converted to RGB."""
        from app.preprocessing import decode_base64_image
        
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        b64 = image_to_base64(img, "PNG")
        
        decoded = decode_base64_image(b64)
        
        assert decoded.mode == "RGB"
    
    def test_decode_invalid_base64(self):
        """Test that invalid base64 raises error."""
        from app.preprocessing import decode_base64_image, ImageLoadError
        
        with pytest.raises(ImageLoadError):
            decode_base64_image("not_valid_base64!!!")


class TestIsUrl:
    """Tests for URL detection."""
    
    def test_http_url(self):
        from app.preprocessing import is_url
        assert is_url("http://example.com/image.png")
    
    def test_https_url(self):
        from app.preprocessing import is_url
        assert is_url("https://example.com/image.png")
    
    def test_base64_string(self):
        from app.preprocessing import is_url
        assert not is_url("iVBORw0KGgoAAAANSUhEUg...")
    
    def test_data_uri(self):
        from app.preprocessing import is_url
        assert not is_url("data:image/png;base64,iVBORw...")


class TestPreprocessImageManual:
    """Tests for manual preprocessing (without DreamSim)."""
    
    def test_output_shape(self):
        """Test that output tensor has correct shape."""
        from app.preprocessing import preprocess_image_manual
        
        img = create_test_image(1024, 1024)
        tensor = preprocess_image_manual(img)
        
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
    
    def test_output_range(self):
        """Test that output values are in reasonable range after normalization."""
        from app.preprocessing import preprocess_image_manual
        
        img = create_test_image(512, 512)
        tensor = preprocess_image_manual(img)
        
        # After ImageNet normalization, values should be roughly in [-3, 3]
        assert tensor.min() >= -4.0
        assert tensor.max() <= 4.0
    
    def test_different_input_sizes(self):
        """Test that different input sizes all produce 224x224 output."""
        from app.preprocessing import preprocess_image_manual
        
        sizes = [(100, 100), (1024, 1024), (500, 300), (224, 224)]
        
        for w, h in sizes:
            img = create_test_image(w, h)
            tensor = preprocess_image_manual(img)
            assert tensor.shape == (3, 224, 224), f"Failed for size {w}x{h}"


class TestLoadImageSync:
    """Tests for synchronous image loading helpers."""
    
    def test_decode_base64_preserves_content(self):
        """Test that decoding preserves image content."""
        from app.preprocessing import decode_base64_image
        
        # Create image with specific pixel values
        img = Image.new("RGB", (10, 10), (123, 45, 67))
        b64 = image_to_base64(img, "PNG")
        
        decoded = decode_base64_image(b64)
        
        # Check a pixel
        pixel = decoded.getpixel((5, 5))
        assert pixel == (123, 45, 67)
