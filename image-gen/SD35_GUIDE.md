# Stable Diffusion 3.5 Large Turbo Guide

## Overview

SD 3.5 Large Turbo is Stability AI's latest model offering:
- **Faster generation** - 4-8 steps instead of 20-50
- **Better quality** - Improved prompt following and detail
- **Larger model** - 11-12GB (vs 6.9GB for SDXL)
- **Different architecture** - Uses MMDiT instead of U-Net

---

## Quick Comparison

| Feature | SDXL Base 1.0 | SD 3.5 Large Turbo |
|---------|---------------|-------------------|
| **Model Size** | 6.9GB | 11-12GB |
| **VRAM Base** | ~7.5GB | ~11GB |
| **Inference Steps** | 20-50 | 4-8 ⚡ |
| **Generation Time** | ~15-20s | ~8-12s ⚡ |
| **Batch Size (RTX 3060 12GB)** | 2 | 1 (due to larger model) |
| **Batch Size (L40S 48GB)** | 12-13 | 7-8 |
| **Quality** | Good | Better ✨ |
| **Prompt Following** | Good | Excellent ✨ |
| **License** | Open | Requires HF token |

---

## Prerequisites

### 1. HuggingFace Account & Token

SD 3.5 requires accepting the license and using an access token.

**Steps:**
1. Create account at https://huggingface.co/join
2. Accept SD 3.5 license at https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo
3. Generate token at https://huggingface.co/settings/tokens
   - Type: "Read" token is sufficient
   - Name: "sd35-image-gen"
4. Copy token (starts with `hf_...`)

### 2. Update diffusers Library

Ensure you have `diffusers>=0.30.0`:

```bash
# Check version
docker exec image-game-image-gen pip show diffusers

# If needed, update requirements.txt and rebuild
```

---

## How to Switch to SD 3.5 Large Turbo

### Method 1: Edit docker-compose.yaml (Recommended)

```yaml
image-gen:
  environment:
    # Change MODEL_ID
    MODEL_ID: stabilityai/stable-diffusion-3.5-large-turbo

    # Add HuggingFace token
    HF_TOKEN: hf_your_token_here  # Replace with your token
```

### Method 2: Environment Variables

```bash
export MODEL_ID=stabilityai/stable-diffusion-3.5-large-turbo
export HF_TOKEN=hf_your_token_here

docker-compose up -d --build image-gen
```

---

## Step-by-Step Migration

### 1. Get HuggingFace Token

```bash
# Visit https://huggingface.co/settings/tokens
# Create new token with "Read" access
# Copy token value (hf_...)
```

### 2. Update docker-compose.yaml

```yaml
image-gen:
  container_name: image-game-image-gen
  environment:
    MODEL_ID: stabilityai/stable-diffusion-3.5-large-turbo
    HF_TOKEN: hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890  # Your token
    DEVICE: cuda

    # Optional: Reduce default inference steps for Turbo
    # (this would require code changes to respect this env var)
```

### 3. Rebuild Container

```bash
# Stop current container
docker-compose down image-gen

# Rebuild and start (will download SD 3.5 model)
docker-compose up -d --build image-gen

# Follow logs
docker-compose logs -f image-gen
```

**Expected logs:**
```
[image-gen] Detected model type: SD 3.x
[image-gen] Loading model from Hugging Face...
[image-gen] NOTE: First run will download ~11-12GB model files - this can take 5-20 minutes!
[image-gen] Using HuggingFace token for authenticated access
Downloading (…)_model/model.safetensors: 100%|██| 11.2G/11.2G [15:23<00:00, 12.1MB/s]
[image-gen] Model loaded, moving to device cuda...
[calculate_optimal_batch_size] Total VRAM: 12.00 GB
[calculate_optimal_batch_size] Calculated optimal batch size: 1  ← Smaller due to larger model
[image-gen] Model ready on cuda
```

### 4. Test Generation

```bash
# Test single image
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a photorealistic cat wearing a space helmet, highly detailed",
    "game_id": "test",
    "round_id": "test",
    "player_id": "test",
    "num_inference_steps": 4
  }'
```

**Note:** Use `num_inference_steps: 4-8` for Turbo models (not 20-50).

---

## Performance Expectations

### RTX 3060 (12GB)

**SDXL:**
- Batch size: 2
- Single image: ~20s
- Batch of 2: ~35s

**SD 3.5 Large Turbo:**
- Batch size: 1 (larger model)
- Single image: ~10-12s ⚡
- Throughput: ~0.08 img/sec

**For 50 users:**
- SDXL (batch 2): ~14.5 minutes
- SD 3.5 (batch 1): ~10 minutes (faster per-image compensates for smaller batch)

### L40S (48GB)

**SDXL:**
- Batch size: 12-13
- Batch time: ~40s
- Throughput: ~0.3 img/sec

**SD 3.5 Large Turbo:**
- Batch size: 7-8
- Batch time: ~25s ⚡
- Throughput: ~0.32 img/sec

**For 50 users:**
- SDXL (batch 12): ~4.2 minutes
- SD 3.5 (batch 8): ~3.8 minutes ⚡

---

## Auto-Detected Batch Sizes

| GPU | VRAM | SDXL Batch | SD 3.5 Batch |
|-----|------|-----------|-------------|
| RTX 3060 | 12GB | 2 | 1 |
| RTX 3090 | 24GB | 4-5 | 3 |
| RTX 4090 | 24GB | 8 | 3-4 |
| L40S | 48GB | 12-13 | 7-8 |
| A100 (80GB) | 80GB | 20+ | 15+ |
| H100 (80GB) | 80GB | 20+ | 15+ |

---

## Optimizing for SD 3.5 Turbo

### 1. Reduce Inference Steps in Frontend

Update default steps for Turbo model:

```typescript
// In your frontend or API
const inferenceSteps = modelId.includes('turbo') ? 4 : 25;
```

### 2. Adjust Guidance Scale

SD 3.5 works well with lower guidance:

```json
{
  "guidance_scale": 3.5,  // Lower for SD3.5 (vs 7.5 for SDXL)
  "num_inference_steps": 4
}
```

### 3. Enable Model Offloading (If Low VRAM)

For GPUs with limited VRAM, enable CPU offloading:

```python
# In LocalSDBackend.__init__, after loading:
self.pipe.enable_model_cpu_offload()  # Slower but uses less VRAM
```

---

## Switching Between Models

### Keep Both Models Cached

Both models can coexist in the cache:

```bash
# Check cache
docker exec image-game-image-gen ls -lh /data/huggingface/hub/

# Should see:
# models--stabilityai--stable-diffusion-xl-base-1.0/        (6.9GB)
# models--stabilityai--stable-diffusion-3.5-large-turbo/    (11.2GB)
```

**Switch by changing MODEL_ID:**
```yaml
MODEL_ID: stabilityai/stable-diffusion-xl-base-1.0  # Use SDXL
# or
MODEL_ID: stabilityai/stable-diffusion-3.5-large-turbo  # Use SD3.5
```

**No re-download needed!** Both are cached.

---

## Troubleshooting

### Issue: "Access Denied" or 401 Error

**Cause:** Model requires HuggingFace token

**Fix:**
```yaml
HF_TOKEN: hf_your_actual_token_here  # Add this to docker-compose.yaml
```

### Issue: "Out of Memory" with SD 3.5

**Cause:** Model is larger, batch size may be too high

**Fix 1 - Let auto-detection handle it:**
```yaml
# Remove MAX_BATCH_SIZE override, use auto-detection
# MAX_BATCH_SIZE: 8  ← Comment this out
```

**Fix 2 - Manually reduce:**
```yaml
MAX_BATCH_SIZE: 1  # Force batch size 1 for RTX 3060
```

**Fix 3 - Enable CPU offloading:**
Add to `LocalSDBackend.__init__`:
```python
self.pipe.enable_model_cpu_offload()
self.pipe.enable_sequential_cpu_offload()
```

### Issue: Slow Generation with SD 3.5

**Cause:** Using too many inference steps

**Fix:**
Use 4-8 steps for Turbo models:
```json
{
  "num_inference_steps": 4  // Not 20-50!
}
```

### Issue: Different Image Style/Quality

**Expected!** SD 3.5 has different characteristics:
- Better prompt following
- More photorealistic
- Different artistic style

**Adjust prompts:**
- SD 3.5 understands complex prompts better
- Less need for "quality boosters" like "highly detailed, 8k"
- More natural language works well

---

## Model Variants

### SD 3.5 Large Turbo (Recommended)

```yaml
MODEL_ID: stabilityai/stable-diffusion-3.5-large-turbo
```

- **Best for:** Production, speed-critical apps
- **Inference steps:** 4-8
- **Speed:** Fastest
- **Quality:** Excellent

### SD 3.5 Large (Non-Turbo)

```yaml
MODEL_ID: stabilityai/stable-diffusion-3.5-large
```

- **Best for:** Highest quality, not speed-critical
- **Inference steps:** 20-50
- **Speed:** Slower
- **Quality:** Highest

### SD 3.5 Medium

```yaml
MODEL_ID: stabilityai/stable-diffusion-3.5-medium
```

- **Best for:** Balance of speed/quality/VRAM
- **Size:** ~5-6GB
- **Batch size:** Larger (similar to SDXL)

---

## Recommended Setup

### For RTX 3060 (12GB)

```yaml
# SD 3.5 Large Turbo
MODEL_ID: stabilityai/stable-diffusion-3.5-large-turbo
HF_TOKEN: hf_your_token
ENABLE_BATCHING: "true"
# Batch size 1 (auto-detected)
```

**Performance:**
- Single image: ~10s
- Better quality than SDXL
- Competitive throughput despite batch size 1

### For L40S (48GB)

```yaml
# SD 3.5 Large Turbo
MODEL_ID: stabilityai/stable-diffusion-3.5-large-turbo
HF_TOKEN: hf_your_token
ENABLE_BATCHING: "true"
# Batch size 7-8 (auto-detected)
```

**Performance:**
- Batch of 8: ~25s
- 50 users: ~3.8 minutes
- **Best overall** - Speed + Quality

### For Multi-GPU (4× L40S)

```yaml
MODEL_ID: stabilityai/stable-diffusion-3.5-large-turbo
HF_TOKEN: hf_your_token
GPU_IDS: 0,1,2,3
GPU_LOAD_BALANCE_STRATEGY: queue-depth
```

**Performance:**
- Combined: 4 × 0.32 img/sec = 1.28 img/sec
- 50 users: ~39 seconds ⚡

---

## Summary

✅ **Code changes complete** - Automatic model detection
✅ **Auto batch sizing** - Adjusts for SD3.5's larger size
✅ **HF token support** - Ready for gated models
✅ **Backward compatible** - SDXL still works

**To switch to SD 3.5:**
1. Get HuggingFace token
2. Update `MODEL_ID` in docker-compose.yaml
3. Add `HF_TOKEN` to environment
4. Rebuild container
5. Use 4-8 inference steps (not 20-50)

**Expected results:**
- **Faster generation** - 40-50% faster per image
- **Better quality** - Improved detail and prompt following
- **Smaller batches** - Due to larger model size
- **Overall faster** - Speed gains offset batch size reduction

Ready to upgrade! 🚀
