# Picture Perfect Challenge

A competitive multiplayer game where players generate AI images to match a reference image as closely as possible.

## How It Works

1. An admin creates a game and players join via a game code
2. Each round, a reference image is shown to all players
3. Players write text prompts to generate images using AI (Stable Diffusion XL or FLUX)
4. Generated images are scored based on perceptual similarity to the reference
5. Players earn points based on how closely their images match
6. Leaderboards track scores across multiple rounds

## Architecture

The project uses a microservices architecture:

| Service | Technology | Description |
|---------|------------|-------------|
| **Web** | React, TypeScript, Vite | Frontend SPA served via Nginx |
| **Logic Service** | Python, FastAPI | Game orchestration, REST API, WebSocket server |
| **Image Generation** | Python, Stable Diffusion XL | GPU-accelerated text-to-image generation |
| **Image Generation (FLUX)** | Python, FLUX.1-schnell | Alternative faster image generation |
| **Image Similarity** | Python, DreamSim | Perceptual similarity scoring |
| **PostgreSQL** | PostgreSQL 16 | Game state persistence |
| **MinIO** | S3-compatible storage | Image storage |
| **Jaeger** | Distributed tracing | Observability |

## Prerequisites

- Docker and Docker Compose v2.0+
- NVIDIA GPU with CUDA 12.4+ (for image generation)
- 10+ GB free disk space (for AI models)
- 8+ GB GPU VRAM (16+ GB recommended for FLUX)

## Quick Start

```bash
# Build all services
docker compose build

# Start all services
docker compose up -d

# View logs
docker compose logs -f
```

**Startup times:**
- First run: 5-15 minutes (downloads ~6-8GB of AI models)
- Subsequent runs: 1-2 minutes (loads cached models)

## Access Points

| Service | URL |
|---------|-----|
| Web UI | http://localhost:3000 |
| Logic API | http://localhost:8000 |
| Image Gen API | http://localhost:8080 |
| Image Similarity API | http://localhost:8081 |
| FLUX Gen API | http://localhost:8082 |
| MinIO Console | http://localhost:9001 |
| Jaeger UI | http://localhost:16686 |

## Game Features

### Player Features
- Join games via code
- Generate images with text prompts
- Use negative prompts to refine generation
- Up to 6 generation attempts per round
- Submit best image for scoring
- View live timer and results
- Track scores on leaderboard

### Admin Features
- Create and manage games
- Upload reference images
- Configure round duration (30s - 5 minutes)
- Start/end rounds manually
- View all player submissions

## Configuration

Key environment variables can be configured in `docker-compose.yaml`:

### Image Generation
```yaml
GEN_MODE: local          # "local", "mock", or "stability"
MODEL_ID: stabilityai/stable-diffusion-xl-base-1.0
ENABLE_BATCHING: true    # Batch requests for better GPU throughput
MAX_BATCH_SIZE: 8        # Auto-detected based on VRAM
```

### Image Similarity
```yaml
DEVICE: cpu              # "cpu" or "cuda"
MODEL_TYPE: dino_vitb16  # DreamSim model variant
```

### Multi-GPU Support
```yaml
GPU_IDS: 0,1,2
GPU_LOAD_BALANCE_STRATEGY: queue-depth  # or "round-robin"
```

## Development

### Running Individual Services

```bash
# Start just the database and MinIO
docker compose up -d db minio

# Run logic service locally
cd logic-service
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Run frontend in dev mode
cd web
npm install
npm run dev
```

### Project Structure

```
.
├── web/                    # React frontend
├── logic-service/          # Game logic API
├── image-gen/              # Stable Diffusion service
├── image-gen-flux-schnell/ # FLUX model service
├── image-similarity/       # Similarity scoring service
├── data/                   # Reference images and prompts
└── docker-compose.yaml     # Service orchestration
```

## Troubleshooting

### Image generation is slow
- Ensure GPU is properly detected: `nvidia-smi`
- Check batching is enabled in logs
- First requests are slower due to model compilation

### 413 Error on image upload
- The nginx config limits upload size
- Adjust `client_max_body_size` in `web/nginx.conf`

### Models not loading
- Check disk space for model cache
- Verify `HF_TOKEN` is set for gated models (SD 3.5, etc.)
- Check logs: `docker compose logs image-gen`

### WebSocket disconnections
- Ensure `proxy_read_timeout` is set high enough in nginx
- Check browser console for connection errors

## License

MIT
