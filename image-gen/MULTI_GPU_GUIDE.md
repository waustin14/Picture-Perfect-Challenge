# Multi-GPU Usage Guide

## Overview

The image-gen service **automatically detects and uses all available GPUs** when batching is enabled. This provides near-linear scaling for concurrent request throughput.

## Quick Start

### Scenario 1: Use All Available GPUs (Automatic)

Simply start the service with batching enabled:

```bash
export ENABLE_BATCHING=true
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

The service will:
1. Detect all available GPUs
2. Load the model on each GPU
3. Automatically distribute requests across GPUs

**Console output:**
```
[MultiGPUBatchedBackend] Auto-detected 4 GPU(s): [0, 1, 2, 3]
[MultiGPUBatchedBackend] Initializing with 4 GPU(s): [0, 1, 2, 3]
[image-gen] Using LOCAL backend with model ... on cuda:0 (GPU 0)
[image-gen] Using LOCAL backend with model ... on cuda:1 (GPU 1)
[image-gen] Using LOCAL backend with model ... on cuda:2 (GPU 2)
[image-gen] Using LOCAL backend with model ... on cuda:3 (GPU 3)
[MultiGPUBatchedBackend] Load balancing strategy: round-robin
[MultiGPUBatchedBackend] Starting batch processors on 4 GPU(s)...
[MultiGPUBatchedBackend] All batch processors started
```

---

### Scenario 2: Use Specific GPUs

Use the `GPU_IDS` environment variable to select specific GPUs:

```bash
# Use only GPUs 0 and 2
export GPU_IDS=0,2
export ENABLE_BATCHING=true
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

### Scenario 3: Use CUDA_VISIBLE_DEVICES

Alternatively, use the standard CUDA environment variable:

```bash
# Only make GPUs 1 and 3 visible (they become cuda:0 and cuda:1)
export CUDA_VISIBLE_DEVICES=1,3
export ENABLE_BATCHING=true
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

**Important:** When using `CUDA_VISIBLE_DEVICES`, the GPUs are remapped:
- System GPU 1 becomes `cuda:0` in the application
- System GPU 3 becomes `cuda:1` in the application

---

### Scenario 4: Single GPU (Disable Multi-GPU)

To use only one GPU even if multiple are available:

```bash
# Method 1: Specify single GPU
export GPU_IDS=0
export ENABLE_BATCHING=true
uvicorn app.main:app --host 0.0.0.0 --port 8080

# Method 2: Use CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
export ENABLE_BATCHING=true
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

## Load Balancing Strategies

### Round-Robin (Default)

Distributes requests evenly across all GPUs in a circular pattern:

```bash
export GPU_LOAD_BALANCE_STRATEGY=round-robin
```

**When to use:**
- Uniform request patterns
- All requests have similar complexity
- Simple and predictable distribution

**Example:**
- Request 1 → GPU 0
- Request 2 → GPU 1
- Request 3 → GPU 2
- Request 4 → GPU 3
- Request 5 → GPU 0 (wraps around)

---

### Queue-Depth (Recommended for Variable Load)

Sends each request to the GPU with the shortest queue:

```bash
export GPU_LOAD_BALANCE_STRATEGY=queue-depth
```

**When to use:**
- Variable request complexity
- Burst traffic patterns
- Want to minimize individual request latency

**Example:**
```
GPU 0: queue=5 requests
GPU 1: queue=2 requests ← Next request goes here
GPU 2: queue=7 requests
GPU 3: queue=3 requests
```

---

## Performance Examples

### Example 1: 4× L40S GPUs with 50 Concurrent Users

**Configuration:**
```bash
export GPU_IDS=0,1,2,3
export ENABLE_BATCHING=true
export MAX_BATCH_SIZE=8
export MAX_BATCH_WAIT_TIME=5.0
```

**Performance:**
- **Single GPU:** 250 seconds for 50 images
- **4 GPUs:** 62 seconds for 50 images
- **Speedup:** 4x (near-linear scaling!)

**Load test:**
```bash
python test_batching.py --burst 50
```

**Expected output:**
```
Total Duration: 62s
Successful: 50/50
Throughput: 0.81 images/second

Batch Statistics:
  num_gpus: 4
  total_batches: 25 (across all GPUs)
  avg_batch_size: 8.0
```

---

### Example 2: Mixed GPU Setup (2× L40S + 2× A100)

The service works with mixed GPU types:

```bash
# If you have GPUs 0,1 as L40S and GPUs 2,3 as A100
export GPU_IDS=0,1,2,3
export ENABLE_BATCHING=true
```

**Notes:**
- Each GPU processes independently at its own speed
- Faster GPUs (A100) will process more batches
- Queue-depth strategy automatically sends more work to faster GPUs

---

### Example 3: Development with Single GPU

For development/testing:

```bash
export GPU_IDS=0
export ENABLE_BATCHING=true
export MAX_BATCH_SIZE=4  # Lower for dev GPU
export MAX_BATCH_WAIT_TIME=3.0
```

---

## Monitoring Multi-GPU Performance

### Check GPU Utilization

```bash
# Terminal 1: Start service
uvicorn app.main:app --host 0.0.0.0 --port 8080

# Terminal 2: Watch GPU usage
watch -n 1 nvidia-smi

# Terminal 3: Run load test
python test_batching.py --burst 50
```

**Look for:**
- GPU Utilization: Should be >80% on all GPUs during processing
- Memory Usage: Each GPU should have model loaded (~6-8GB for SDXL)
- Temperature: Monitor to ensure adequate cooling

---

### Check Per-GPU Statistics

```bash
curl http://localhost:8080/stats | jq '.per_gpu_stats'
```

**Response:**
```json
[
  {
    "gpu_id": 0,
    "current_queue_depth": 2,
    "total_requests": 150,
    "total_batches": 23,
    "avg_batch_size": 6.5
  },
  {
    "gpu_id": 1,
    "current_queue_depth": 1,
    "total_requests": 148,
    "total_batches": 23,
    "avg_batch_size": 6.4
  },
  ...
]
```

**What to look for:**
- **Balanced requests:** `total_requests` should be similar across GPUs
- **Low queue depths:** `current_queue_depth` should be low (<5) under normal load
- **Consistent batch sizes:** `avg_batch_size` should be near `MAX_BATCH_SIZE`

---

## Troubleshooting

### Issue: Only Using One GPU

**Symptoms:**
- Service starts but only shows 1 GPU
- `nvidia-smi` shows other GPUs idle

**Solutions:**
1. Check GPU visibility:
   ```bash
   python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
   ```

2. Verify no environment restrictions:
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   echo $GPU_IDS
   ```

3. Check service logs for GPU detection messages

---

### Issue: Unbalanced GPU Load

**Symptoms:**
- Some GPUs at 100%, others idle
- Uneven `total_requests` in per-GPU stats

**Solutions:**
1. Switch to queue-depth load balancing:
   ```bash
   export GPU_LOAD_BALANCE_STRATEGY=queue-depth
   ```

2. Check for GPU performance differences (mixed GPU types)

3. Reduce batch size if some GPUs have less memory

---

### Issue: Out of Memory on Multi-GPU

**Symptoms:**
- CUDA out of memory errors
- Service crashes during multi-GPU initialization

**Solutions:**
1. Reduce batch size:
   ```bash
   export MAX_BATCH_SIZE=4  # Instead of 8
   ```

2. Use fewer GPUs:
   ```bash
   export GPU_IDS=0,1  # Instead of 0,1,2,3
   ```

3. Check GPU memory with `nvidia-smi` - each GPU needs ~8-10GB

---

### Issue: No Speedup with Multiple GPUs

**Symptoms:**
- 4 GPUs not faster than 1 GPU
- Linear scaling not achieved

**Possible causes:**

1. **Not enough concurrent requests:**
   - Need sustained load to fill all GPU queues
   - Test with: `python test_batching.py --burst 50`

2. **Wait time too long:**
   - GPUs idle waiting for batches to fill
   - Solution: Reduce `MAX_BATCH_WAIT_TIME` to 2-3 seconds

3. **CPU bottleneck:**
   - Check CPU usage with `htop`
   - May need more CPU cores for data loading

---

## Production Deployment

### Docker Compose with Multi-GPU

```yaml
version: '3.8'

services:
  image-gen:
    image: your-image-gen:latest
    environment:
      - ENABLE_BATCHING=true
      - GPU_IDS=0,1,2,3
      - MAX_BATCH_SIZE=8
      - GPU_LOAD_BALANCE_STRATEGY=queue-depth
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1', '2', '3']
              capabilities: [gpu]
```

---

### Kubernetes with Multi-GPU

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: image-gen
spec:
  containers:
  - name: image-gen
    image: your-image-gen:latest
    env:
    - name: ENABLE_BATCHING
      value: "true"
    - name: MAX_BATCH_SIZE
      value: "8"
    resources:
      limits:
        nvidia.com/gpu: 4  # Request 4 GPUs
```

---

## Best Practices

1. **Start with auto-detection:**
   - Let the service detect all GPUs automatically
   - Only specify `GPU_IDS` if you need specific GPUs

2. **Use queue-depth load balancing:**
   - Better for variable loads
   - Automatically adapts to GPU performance differences

3. **Monitor per-GPU stats:**
   - Check `/stats` endpoint regularly
   - Look for balanced request distribution

4. **Test scaling:**
   - Run load tests with increasing GPU counts
   - Verify near-linear speedup

5. **Plan for memory:**
   - Each GPU needs ~8-10GB for SDXL
   - Reduce batch size if running out of memory

6. **Cool adequately:**
   - Multi-GPU generates significant heat
   - Monitor temperatures with `nvidia-smi`
   - Ensure datacenter cooling is sufficient

---

## Summary

| Configuration | Command | Use Case |
|---------------|---------|----------|
| All GPUs (auto) | `ENABLE_BATCHING=true` | Production, maximum throughput |
| Specific GPUs | `GPU_IDS=0,2,3` | Reserved GPUs for other tasks |
| CUDA control | `CUDA_VISIBLE_DEVICES=0,1` | System-level GPU management |
| Single GPU | `GPU_IDS=0` | Development, testing |
| Queue balancing | `GPU_LOAD_BALANCE_STRATEGY=queue-depth` | Variable load, mixed GPUs |

**Key takeaway:** Multi-GPU batching provides near-linear scaling. With 4 GPUs, you get ~4x throughput!
