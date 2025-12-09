"""
Tests for API endpoints.
"""
import base64
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient


def create_test_image(width: int = 224, height: int = 224, color: tuple = (255, 0, 0)) -> Image.Image:
    """Create a simple test image."""
    return Image.new("RGB", (width, height), color)


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture(scope="module")
def client():
    """Create test client with app."""
    from app.api import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self, client):
        """Health response should have expected fields."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "model_type" in data


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_info(self, client):
        """Root endpoint should return service info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data


@pytest.mark.slow
class TestSimilarityEndpoint:
    """Tests for similarity computation endpoint."""
    
    def test_single_pair_identical_images(self, client):
        """Identical images should have low distance, high similarity."""
        img = create_test_image(color=(128, 128, 128))
        b64 = image_to_base64(img)
        
        response = client.post("/similarity", json={
            "pairs": [{
                "reference_image": b64,
                "generated_image": b64,
            }]
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["scores"]) == 1
        assert data["scores"][0]["distance"] < 0.01
        assert data["scores"][0]["similarity"] > 0.99
        assert "processing_time_ms" in data
    
    def test_single_pair_different_images(self, client):
        """Different images should have positive distance, lower similarity."""
        img1 = create_test_image(color=(0, 0, 0))  # Black
        img2 = create_test_image(color=(255, 255, 255))  # White
        
        response = client.post("/similarity", json={
            "pairs": [{
                "reference_image": image_to_base64(img1),
                "generated_image": image_to_base64(img2),
            }]
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["scores"][0]["distance"] > 0
        assert data["scores"][0]["similarity"] < 1.0
    
    def test_multiple_pairs(self, client):
        """Should handle multiple pairs in one request."""
        pairs = []
        for i in range(5):
            img1 = create_test_image(color=(i * 50, 0, 0))
            img2 = create_test_image(color=(0, i * 50, 0))
            pairs.append({
                "reference_image": image_to_base64(img1),
                "generated_image": image_to_base64(img2),
                "pair_id": f"pair_{i}",
            })
        
        response = client.post("/similarity", json={"pairs": pairs})
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["scores"]) == 5
        for i, score in enumerate(data["scores"]):
            assert score["pair_id"] == f"pair_{i}"
            assert score["distance"] >= 0
            assert 0 < score["similarity"] <= 1
    
    def test_empty_pairs_rejected(self, client):
        """Empty pairs list should be rejected."""
        response = client.post("/similarity", json={"pairs": []})
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_base64_handled(self, client):
        """Invalid base64 should return error or infinity distance."""
        valid_img = create_test_image()
        
        response = client.post("/similarity", json={
            "pairs": [{
                "reference_image": image_to_base64(valid_img),
                "generated_image": "not_valid_base64!!!",
            }]
        })
        
        # Should either return 400 or handle gracefully with infinity
        assert response.status_code in [200, 400]
    
    def test_jpeg_images(self, client):
        """Should handle JPEG format."""
        img = create_test_image()
        b64 = image_to_base64(img, format="JPEG")
        
        response = client.post("/similarity", json={
            "pairs": [{
                "reference_image": b64,
                "generated_image": b64,
            }]
        })
        
        assert response.status_code == 200
    
    def test_data_uri_format(self, client):
        """Should handle data URI format."""
        img = create_test_image()
        b64 = image_to_base64(img)
        data_uri = f"data:image/png;base64,{b64}"
        
        response = client.post("/similarity", json={
            "pairs": [{
                "reference_image": data_uri,
                "generated_image": b64,
            }]
        })
        
        assert response.status_code == 200
    
    def test_large_images_resized(self, client):
        """Large images should be handled (resized to 224x224)."""
        img = create_test_image(width=1024, height=1024)
        b64 = image_to_base64(img)
        
        response = client.post("/similarity", json={
            "pairs": [{
                "reference_image": b64,
                "generated_image": b64,
            }]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["scores"][0]["distance"] < 0.01  # Same image, low distance
    
    def test_response_times_included(self, client):
        """Response should include processing time."""
        img = create_test_image()
        b64 = image_to_base64(img)
        
        response = client.post("/similarity", json={
            "pairs": [{
                "reference_image": b64,
                "generated_image": b64,
            }]
        })
        
        data = response.json()
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], (int, float))
        assert data["processing_time_ms"] > 0


@pytest.mark.slow  
class TestBatchPerformance:
    """Performance tests for batched processing."""
    
    def test_30_pairs_under_30_seconds(self, client):
        """30 pairs should complete in under 30 seconds."""
        import time
        
        # Create 30 image pairs
        pairs = []
        for i in range(30):
            img1 = create_test_image(width=1024, height=1024, color=(i * 8, 0, 0))
            img2 = create_test_image(width=1024, height=1024, color=(0, i * 8, 0))
            pairs.append({
                "reference_image": image_to_base64(img1),
                "generated_image": image_to_base64(img2),
            })
        
        start = time.time()
        response = client.post("/similarity", json={"pairs": pairs})
        elapsed = time.time() - start
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["scores"]) == 30
        assert elapsed < 30, f"Request took {elapsed:.1f}s, exceeds 30s limit"
        
        print(f"30 pairs processed in {elapsed:.2f}s ({data['processing_time_ms']:.1f}ms reported)")
