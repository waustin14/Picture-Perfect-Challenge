"""
Tests for model module.

Note: These tests require the DreamSim model to be available.
Some tests are marked as slow and may be skipped in CI.
"""
import pytest
import torch


class TestDistanceToSimilarity:
    """Tests for distance/similarity conversion functions."""
    
    def test_zero_distance_gives_one_similarity(self):
        """Distance 0 should give similarity 1."""
        from app.model import distance_to_similarity
        
        d = torch.tensor([0.0])
        s = distance_to_similarity(d)
        
        assert torch.allclose(s, torch.tensor([1.0]))
    
    def test_large_distance_gives_low_similarity(self):
        """Large distance should give similarity close to 0."""
        from app.model import distance_to_similarity
        
        d = torch.tensor([100.0])
        s = distance_to_similarity(d)
        
        assert s.item() < 0.1
        assert s.item() > 0.0
    
    def test_similarity_in_valid_range(self):
        """Similarity should always be in (0, 1]."""
        from app.model import distance_to_similarity
        
        distances = torch.tensor([0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
        similarities = distance_to_similarity(distances)
        
        assert (similarities > 0).all()
        assert (similarities <= 1).all()
    
    def test_monotonically_decreasing(self):
        """Similarity should decrease as distance increases."""
        from app.model import distance_to_similarity
        
        distances = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        similarities = distance_to_similarity(distances)
        
        for i in range(len(similarities) - 1):
            assert similarities[i] > similarities[i + 1]
    
    def test_batch_processing(self):
        """Should handle batch inputs."""
        from app.model import distance_to_similarity
        
        d = torch.rand(100)
        s = distance_to_similarity(d)
        
        assert s.shape == d.shape


class TestSimilarityToDistance:
    """Tests for reverse conversion."""
    
    def test_roundtrip(self):
        """Converting distance -> similarity -> distance should be identity."""
        from app.model import distance_to_similarity, similarity_to_distance
        
        original = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0])
        sim = distance_to_similarity(original)
        recovered = similarity_to_distance(sim)
        
        assert torch.allclose(original, recovered, atol=1e-6)


@pytest.mark.slow
class TestDreamSimModel:
    """
    Tests for DreamSim model loading and inference.
    
    These tests require GPU/model weights and are marked as slow.
    """
    
    @pytest.fixture(scope="class")
    def model(self):
        """Load model once for all tests in class."""
        from app.model import DreamSimModel
        
        m = DreamSimModel()
        m.initialize(device="cpu")  # Use CPU for testing
        return m
    
    def test_model_initializes(self, model):
        """Model should initialize successfully."""
        assert model.is_initialized()
        assert model.get_device() in ["cpu", "cuda"]
    
    def test_compute_distance_shape(self, model):
        """compute_distance should return correct shape."""
        preprocess = model.get_preprocess()
        
        # Create dummy images
        from PIL import Image
        img1 = Image.new("RGB", (224, 224), (255, 0, 0))
        img2 = Image.new("RGB", (224, 224), (0, 255, 0))
        
        # Preprocess
        t1 = preprocess(img1)
        t2 = preprocess(img2)
        
        # Compute distance
        distances = model.compute_distance(t1, t2)
        
        assert distances.shape == (1,)
        assert distances.dtype == torch.float32
    
    def test_identical_images_low_distance(self, model):
        """Identical images should have very low distance."""
        preprocess = model.get_preprocess()
        
        from PIL import Image
        img = Image.new("RGB", (224, 224), (128, 128, 128))
        
        t1 = preprocess(img)
        t2 = preprocess(img)
        
        distances = model.compute_distance(t1, t2)
        
        # Distance should be very close to 0
        assert distances.item() < 0.01
    
    def test_different_images_positive_distance(self, model):
        """Very different images should have positive distance."""
        preprocess = model.get_preprocess()
        
        from PIL import Image
        img1 = Image.new("RGB", (224, 224), (0, 0, 0))  # Black
        img2 = Image.new("RGB", (224, 224), (255, 255, 255))  # White
        
        t1 = preprocess(img1)
        t2 = preprocess(img2)
        
        distances = model.compute_distance(t1, t2)
        
        # Distance should be positive
        assert distances.item() > 0.0
    
    def test_batch_processing(self, model):
        """Should handle batched inputs."""
        preprocess = model.get_preprocess()
        
        from PIL import Image
        
        # Create batch of images
        batch_size = 5
        imgs1 = [Image.new("RGB", (224, 224), (i * 50, 0, 0)) for i in range(batch_size)]
        imgs2 = [Image.new("RGB", (224, 224), (0, i * 50, 0)) for i in range(batch_size)]
        
        # Preprocess and stack
        t1 = torch.cat([preprocess(img) for img in imgs1], dim=0)
        t2 = torch.cat([preprocess(img) for img in imgs2], dim=0)
        
        distances = model.compute_distance(t1, t2)
        
        assert distances.shape == (batch_size,)
        assert (distances >= 0).all()
    
    def test_distance_non_negative(self, model):
        """Distances should always be non-negative."""
        preprocess = model.get_preprocess()
        
        from PIL import Image
        import random
        
        # Test with random images
        for _ in range(5):
            color1 = tuple(random.randint(0, 255) for _ in range(3))
            color2 = tuple(random.randint(0, 255) for _ in range(3))
            
            img1 = Image.new("RGB", (224, 224), color1)
            img2 = Image.new("RGB", (224, 224), color2)
            
            t1 = preprocess(img1)
            t2 = preprocess(img2)
            
            distances = model.compute_distance(t1, t2)
            
            assert distances.item() >= 0
