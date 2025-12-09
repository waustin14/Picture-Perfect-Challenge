# Image Similarity Service

A microservice for computing perceptual image similarity using [DreamSim](https://github.com/ssundaram21/dreamsim).

DreamSim is a perceptual metric that bridges the gap between "low-level" metrics (LPIPS, PSNR, SSIM) and "high-level" measures (CLIP), achieving better alignment with human similarity judgments.

## Features

- Fast batch processing of image pairs (30+ pairs in <10 seconds on GPU)
- Support for both base64-encoded images and URLs
- Single GPU-optimized deployment
- RESTful API with FastAPI
- Docker support with NVIDIA GPU passthrough

## Requirements

### Hardware
- 1× NVIDIA GPU (≥8 GB VRAM, CUDA capable)
- 4-8 vCPUs
- ≥16 GB RAM

### Software
- Python 3.10+
- CUDA 12.1+ (for GPU acceleration)
- Docker with NVIDIA Container Toolkit (for containerized deployment)

## Quick Start

### Local Development (without Docker)

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or: venv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the service:**
   ```bash
   uvicorn app.api:app --host 0.0.0.0 --port 8080
   ```

### Docker Deployment

1. **Build the image:**
   ```bash
   docker build -t image-similarity-service:latest .
   ```

2. **Run with GPU:**
   ```bash
   docker run --gpus all -p 8080:8080 image-similarity-service:latest
   ```

   Or using docker-compose:
   ```bash
   docker-compose up -d
   ```

3. **Verify it's running:**
   ```bash
   curl http://localhost:8080/health
   ```

## API Reference

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "model_type": "dino_vitb16"
}
```

### Compute Similarity

```
POST /similarity
Content-Type: application/json
```

**Request Body:**
```json
{
  "pairs": [
    {
      "reference_image": "<base64 or URL>",
      "generated_image": "<base64 or URL>",
      "pair_id": "optional-identifier"
    }
  ]
}
```

**Image Input Formats:**
- Base64-encoded PNG/JPEG: `"iVBORw0KGgoAAAANS..."`
- Data URI: `"data:image/png;base64,iVBORw0KGgo..."`
- URL: `"https://example.com/image.png"`

**Response:**
```json
{
  "scores": [
    {
      "distance": 0.1234,
      "similarity": 0.8901,
      "pair_id": "optional-identifier"
    }
  ],
  "processing_time_ms": 245.67
}
```

**Score Interpretation:**
- `distance`: Perceptual distance (0 = identical, larger = more different)
- `similarity`: Similarity score (0-1, higher = more similar)
  - Formula: `similarity = 1 / (1 + distance)`

## Examples

### Python Client

```python
import base64
import requests
from PIL import Image
import io

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Using base64-encoded images
response = requests.post(
    "http://localhost:8080/similarity",
    json={
        "pairs": [
            {
                "reference_image": image_to_base64("reference.png"),
                "generated_image": image_to_base64("generated.png"),
                "pair_id": "test-1"
            }
        ]
    }
)

result = response.json()
print(f"Distance: {result['scores'][0]['distance']:.4f}")
print(f"Similarity: {result['scores'][0]['similarity']:.4f}")
```

### Using URLs

```python
response = requests.post(
    "http://localhost:8080/similarity",
    json={
        "pairs": [
            {
                "reference_image": "https://example.com/ref.png",
                "generated_image": "https://example.com/gen.png"
            }
        ]
    }
)
```

### cURL with Base64

```bash
# First, encode your images
REF_B64=$(base64 -w0 reference.png)
GEN_B64=$(base64 -w0 generated.png)

# Send request
curl -X POST http://localhost:8080/similarity \
  -H "Content-Type: application/json" \
  -d "{
    \"pairs\": [
      {
        \"reference_image\": \"$REF_B64\",
        \"generated_image\": \"$GEN_B64\"
      }
    ]
  }"
```

### cURL with URLs

```bash
curl -X POST http://localhost:8080/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "pairs": [
      {
        "reference_image": "https://example.com/ref.png",
        "generated_image": "https://example.com/gen.png"
      }
    ]
  }'
```

### Batch Processing (30 pairs)

```python
import requests
import base64
from pathlib import Path

# Prepare 30 image pairs
pairs = []
for i in range(30):
    pairs.append({
        "reference_image": base64.b64encode(
            Path(f"refs/ref_{i}.png").read_bytes()
        ).decode(),
        "generated_image": base64.b64encode(
            Path(f"gens/gen_{i}.png").read_bytes()
        ).decode(),
        "pair_id": f"pair_{i}"
    })

response = requests.post(
    "http://localhost:8080/similarity",
    json={"pairs": pairs},
    timeout=30
)

result = response.json()
print(f"Processed {len(result['scores'])} pairs in {result['processing_time_ms']:.1f}ms")
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Device to run model on (`cuda` or `cpu`) |
| `MODEL_TYPE` | `dino_vitb16` | DreamSim model variant |
| `MODEL_CACHE_DIR` | `/app/models` | Directory for model weights |
| `MAX_PAIRS_PER_REQUEST` | `128` | Maximum pairs per request |
| `BATCH_SIZE_LIMIT` | `64` | Internal batch chunk size |
| `URL_FETCH_TIMEOUT` | `10` | Timeout for fetching URLs (seconds) |
| `MAX_IMAGE_SIZE_MB` | `20` | Maximum image size when fetching URLs |

## Performance

### Benchmarks (NVIDIA A100)

| Batch Size | Processing Time | Per-Pair Time |
|------------|-----------------|---------------|
| 1          | ~150ms          | 150ms         |
| 10         | ~400ms          | 40ms          |
| 30         | ~800ms          | 27ms          |
| 100        | ~2.5s           | 25ms          |

**Target:** ≤10 seconds for 30 pairs  
**Hard limit:** ≤30 seconds for 30 pairs

## Testing

### Run unit tests (no model required)
```bash
pytest tests/test_preprocessing.py -v
```

### Run all tests (requires model)
```bash
pytest -v
```

### Skip slow tests
```bash
pytest -v -m "not slow"
```

### Run performance benchmark
```bash
# Start the service first, then:
python tests/test_e2e.py --url http://localhost:8080 --benchmark
```

## Project Structure

```
image-similarity-service/
├── app/
│   ├── __init__.py
│   ├── api.py            # FastAPI endpoints
│   ├── model.py          # DreamSim loading and scoring
│   ├── preprocessing.py  # Image loading + transforms
│   ├── schemas.py        # Pydantic request/response models
│   └── config.py         # Configuration constants
├── tests/
│   ├── test_api.py       # API endpoint tests
│   ├── test_model.py     # Model unit tests
│   ├── test_preprocessing.py  # Preprocessing tests
│   └── test_e2e.py       # End-to-end performance tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── pytest.ini
└── README.md
```

## Model Information

This service uses DreamSim with the `dino_vitb16` backbone by default. DreamSim was trained on human perceptual judgments and achieves better alignment with human similarity perception than existing metrics.

**Paper:** [DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data](https://arxiv.org/abs/2306.09344) (NeurIPS 2023 Spotlight)

## Troubleshooting

### CUDA out of memory
- Reduce `BATCH_SIZE_LIMIT` environment variable
- Ensure no other GPU processes are running

### Model download fails
- Check internet connectivity
- Verify write permissions for `MODEL_CACHE_DIR`
- Try pre-downloading: `python -c "from dreamsim import dreamsim; dreamsim(pretrained=True)"`

### Slow first request
- First request downloads and loads model (~10-30s)
- Subsequent requests are fast
- Use health check to verify model is loaded before sending requests

### URL fetch timeout
- Increase `URL_FETCH_TIMEOUT` for slow external servers
- Prefer base64 encoding for reliable internal services

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [DreamSim](https://github.com/ssundaram21/dreamsim) by Stephanie Fu et al.
- Built with [FastAPI](https://fastapi.tiangolo.com/) and [PyTorch](https://pytorch.org/)
