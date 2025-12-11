# Model Caching Guide

## How Caching Works

The service now **automatically caches models** to prevent re-downloading on every container rebuild.

### Cache Location

Models are stored in the Docker volume:
```
Docker Volume: image_data
Mount Point:   /data
Cache Dir:     /data/huggingface/hub
```

### What Gets Cached

- SDXL model files (~6-8GB)
- Tokenizer files
- Model configuration
- VAE weights
- All HuggingFace transformers cache

---

## First Run vs Subsequent Runs

### First Run (Initial Download)

```bash
docker-compose up -d image-gen
```

**Timeline:**
- Container starts: 0s
- Model download begins: 0s
- Download completes: 5-15 minutes (depending on network)
- Model loaded to GPU: +30 seconds
- Service ready: **Total ~5-16 minutes**

**Logs you'll see:**
```
[image-gen] Loading model from Hugging Face...
[image-gen] NOTE: First run will download ~6-8GB model files - this can take 5-15 minutes!
Downloading (…)ain/model_index.json: 100%|██████| 543/543 [00:00<00:00, 5.43kB/s]
Downloading (…)_model/model.safetensors: 100%|██████| 6.94G/6.94G [12:34<00:00, 9.18MB/s]
...
[image-gen] Model loaded, moving to device cuda...
[image-gen] Model ready on cuda
```

### Subsequent Runs (Cache Hit)

```bash
# Rebuild container
docker-compose up -d --build image-gen

# Or restart
docker-compose restart image-gen
```

**Timeline:**
- Container starts: 0s
- **Cache found - no download!** ✅
- Model loaded from cache: 30 seconds
- Service ready: **Total ~30-60 seconds**

**Logs you'll see:**
```
[image-gen] Loading model from Hugging Face...
[image-gen] Model loaded, moving to device cuda...  ← Fast! No download
[image-gen] Model ready on cuda
```

---

## Verifying Cache is Working

### Check if Model is Cached

```bash
# Check cache size (should be ~6-8GB after first run)
docker exec image-game-image-gen du -sh /data/huggingface

# List cached models
docker exec image-game-image-gen ls -lh /data/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/
```

### Check Volume Size

```bash
# See volume usage
docker volume ls
docker volume inspect image_data

# Check what's in the volume
docker run --rm -v Picture-Perfect-Challenge_image_data:/data alpine ls -lh /data/
```

---

## Cache Management

### Clear Cache (Force Re-Download)

**Option 1: Remove specific model**
```bash
docker exec image-game-image-gen rm -rf /data/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0
docker-compose restart image-gen  # Will re-download
```

**Option 2: Clear entire cache**
```bash
docker exec image-game-image-gen rm -rf /data/huggingface/*
docker-compose restart image-gen  # Will re-download
```

**Option 3: Delete volume (nuclear option)**
```bash
docker-compose down
docker volume rm Picture-Perfect-Challenge_image_data
docker-compose up -d  # Will re-download and recreate
```

### Backup Cache

```bash
# Create backup of cached models
docker run --rm -v Picture-Perfect-Challenge_image_data:/data \
  -v $(pwd):/backup alpine \
  tar czf /backup/model-cache-backup.tar.gz /data/huggingface
```

### Restore Cache

```bash
# Restore from backup
docker run --rm -v Picture-Perfect-Challenge_image_data:/data \
  -v $(pwd):/backup alpine \
  tar xzf /backup/model-cache-backup.tar.gz -C /
```

---

## Pre-Download Model During Build (Optional)

To bake the model into the Docker image (eliminates runtime download completely):

### 1. Edit Dockerfile

Uncomment these lines in `image-gen/Dockerfile`:
```dockerfile
# Optional: Pre-download model during build (uncomment to enable)
ARG MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
RUN python -c "from diffusers import StableDiffusionXLPipeline; StableDiffusionXLPipeline.from_pretrained('${MODEL_ID}')"
```

### 2. Rebuild Image

```bash
docker-compose build image-gen
```

**Pros:**
- ✅ No download at container start
- ✅ Works offline
- ✅ Fastest startup (~30 seconds)

**Cons:**
- ❌ Docker image becomes 8-10GB larger
- ❌ Slower Docker builds (download during build)
- ❌ Less flexible (changing models requires rebuild)

**Recommended for:**
- Production deployments
- Air-gapped environments
- Deployments with slow internet

---

## Environment Variables

All set in `docker-compose.yaml`:

```yaml
# Model cache location
HF_HOME: /data/huggingface
TRANSFORMERS_CACHE: /data/huggingface/hub

# Model to use
MODEL_ID: stabilityai/stable-diffusion-xl-base-1.0
```

To use a different model:
```yaml
MODEL_ID: runwayml/stable-diffusion-v1-5  # Smaller, faster model
```

---

## Troubleshooting

### Cache Not Working (Re-Downloads Every Time)

**Symptom:** Every container restart downloads the model again

**Causes:**
1. Volume not mounted properly
2. Environment variables not set
3. Cache directory doesn't match

**Fix:**
```bash
# 1. Verify volume is mounted
docker inspect image-game-image-gen | grep -A 5 Mounts

# 2. Check environment variables
docker exec image-game-image-gen env | grep HF

# 3. Verify cache directory
docker exec image-game-image-gen ls -lh /data/huggingface/
```

### Volume Full

**Symptom:** Disk space errors during model download

**Fix:**
```bash
# Check volume size
docker system df -v

# Clean up unused volumes
docker volume prune

# Or increase Docker Desktop disk allocation
```

### Wrong Model Cached

**Symptom:** Changed MODEL_ID but old model still loading

**Fix:**
```bash
# Clear cache
docker exec image-game-image-gen rm -rf /data/huggingface/*

# Restart
docker-compose restart image-gen
```

---

## Performance Comparison

| Scenario | Download Time | Load Time | Total Startup |
|----------|---------------|-----------|---------------|
| **First run** (no cache) | 5-15 min | 30 sec | **5-16 min** |
| **Cached** (volume) | 0 sec ✅ | 30 sec | **30-60 sec** |
| **Baked in image** | 0 sec ✅ | 30 sec | **30 sec** |

---

## Summary

✅ **Cache is now enabled by default**
- Volume mount: `/data/huggingface`
- First run: Downloads model (~5-15 min)
- Subsequent runs: Uses cache (~30 sec)
- Persists across container rebuilds

✅ **No configuration needed**
- Just `docker-compose up` and it works
- Model cached automatically

✅ **Optional: Bake model into image**
- Uncomment lines in Dockerfile
- Even faster startup
- Good for production

**You're all set!** The model will download once and be reused forever (until you explicitly clear the cache).
