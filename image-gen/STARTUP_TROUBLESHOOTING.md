# Startup Troubleshooting Guide

## Expected Startup Behavior

The image-gen service loads the SDXL model **before** it starts listening on the port. This means:

1. **First Run:** 5-15 minutes (downloading ~6-8GB model)
2. **Subsequent Runs:** 30-60 seconds (loading cached model into GPU)

You won't see "Listening on 0.0.0.0:8080" until the model is fully loaded.

---

## Is It Stuck or Just Slow?

### Check 1: Look for These Log Messages

**Expected startup sequence:**
```
[image-gen] Initializing StableDiffusionService...
[image-gen] Using LOCAL backend with BATCHING enabled (single GPU).
[image-gen] Using LOCAL backend with model stabilityai/stable-diffusion-xl-base-1.0 on cuda
[image-gen] Loading model from Hugging Face...
[image-gen] NOTE: First run will download ~6-8GB model files - this can take 5-15 minutes!
[image-gen] Subsequent runs will use cached model and start in ~30 seconds

← YOU ARE HERE (waiting for download/load) ←

[image-gen] Model loaded, moving to device cuda...
[image-gen] Model ready on cuda
[image-gen] Attention slicing enabled
[BatchedLocalSDBackend] Initialized with:
  - Max batch size: 8
  - Min batch size: 1
  - Max wait time: 5.0s
  - Min wait time: 0.5s
  - Adaptive batching: true
[image-gen] StableDiffusionService initialized
[image-gen] Running startup event...
[BatchedLocalSDBackend] Batch processor started
[image-gen] Batch processor started successfully
[image-gen] Application startup complete
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

**If stuck at "Loading model from Hugging Face":**
- Model is downloading or loading (this is normal!)
- Check disk I/O and network activity (see below)

---

### Check 2: Monitor Download Progress

**Check disk I/O (model downloading):**
```bash
# In another terminal
docker exec -it <container-name> bash

# Check if files are being downloaded
watch -n 2 'du -sh /root/.cache/huggingface/'

# Or check network activity
docker stats <container-name>
```

**Expected cache size:**
- Starting: ~100MB
- During download: Growing to ~6-8GB
- Complete: ~6-8GB

---

### Check 3: Check GPU Availability

```bash
# On host machine
nvidia-smi

# Check if GPU is visible to container
docker exec <container-name> nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.x.xx    Driver Version: 525.x.xx    CUDA Version: 12.x     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA L40S        Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   30C    P0    70W / 350W |   6500MiB / 48000MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

**Look for:**
- GPU is detected
- Memory usage increases as model loads (~6-8GB)
- No "GPU not available" errors

---

## Common Issues

### Issue 1: Stuck for >30 Minutes

**Cause:** Network issue or disk full

**Check:**
```bash
# Check disk space
docker exec <container-name> df -h

# Check network connectivity
docker exec <container-name> ping -c 3 huggingface.co

# Check download logs
docker logs <container-name> 2>&1 | grep -i download
```

**Solution:**
- Ensure sufficient disk space (need 10GB+ free)
- Check network/firewall allows access to huggingface.co
- Try restarting container

---

### Issue 2: "CUDA out of memory" During Startup

**Logs show:**
```
RuntimeError: CUDA out of memory
```

**Cause:** GPU doesn't have enough memory or other processes using GPU

**Check:**
```bash
nvidia-smi  # Check GPU memory usage
```

**Solution:**
```bash
# Kill other processes using GPU
# Or reduce batch size
export MAX_BATCH_SIZE=4  # Instead of 8
docker-compose restart image-gen
```

---

### Issue 3: "No GPU Found"

**Logs show:**
```
[image-gen] Using LOCAL backend with model ... on cpu
```

**Cause:** Docker can't access GPU

**Solution:**
```bash
# Verify Docker has GPU support
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check docker-compose.yml has GPU config
# Should have:
#   deploy:
#     resources:
#       reservations:
#         devices:
#           - capabilities: [gpu]
```

---

### Issue 4: Startup Event Never Runs

**Logs show model loaded but no "Running startup event"**

**Cause:** FastAPI startup event issue

**Check:**
```bash
# Look for this log line
docker logs <container-name> | grep "Running startup event"
```

**Solution:**
- Update FastAPI: `pip install --upgrade fastapi uvicorn`
- Or check for exceptions in logs

---

## Quick Diagnostic Script

Create `check_startup.sh`:

```bash
#!/bin/bash

CONTAINER_NAME="image-gen"

echo "=== Checking image-gen startup ==="
echo

echo "1. Container status:"
docker ps -a | grep $CONTAINER_NAME
echo

echo "2. Recent logs:"
docker logs --tail 20 $CONTAINER_NAME
echo

echo "3. GPU availability:"
docker exec $CONTAINER_NAME nvidia-smi 2>/dev/null || echo "GPU not accessible"
echo

echo "4. Cache size (model download progress):"
docker exec $CONTAINER_NAME du -sh /root/.cache/huggingface/ 2>/dev/null || echo "Cache not found"
echo

echo "5. Disk space:"
docker exec $CONTAINER_NAME df -h / | grep -v Filesystem
echo

echo "6. Network test:"
docker exec $CONTAINER_NAME ping -c 2 huggingface.co 2>/dev/null || echo "Network issue"
echo

echo "=== If stuck at model loading, this is NORMAL on first run ==="
echo "Check cache size - should be growing to ~6-8GB"
echo "Wait 10-15 minutes for first-time download"
```

**Run:**
```bash
chmod +x check_startup.sh
./check_startup.sh
```

---

## Force Fresh Start

If startup is truly stuck (not just slow):

```bash
# Stop and remove container
docker-compose down

# Clear model cache (ONLY if corrupted)
# WARNING: This will re-download 6-8GB!
docker volume rm <volume-name>  # If using volume
# Or: rm -rf /path/to/cache

# Restart
docker-compose up -d

# Watch logs
docker-compose logs -f image-gen
```

---

## Expected Timeline

| Event | First Run | Subsequent Runs |
|-------|-----------|-----------------|
| Container starts | 0s | 0s |
| Service init | 0s | 0s |
| Model download | 2-10 min | 0s (cached) |
| Model load to memory | 30s | 30s |
| Model move to GPU | 20s | 20s |
| Batch processor start | 1s | 1s |
| Server listening | **3-15 min** | **~1 min** |

---

## Success Indicators

✅ **Service is ready when you see:**
```
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

✅ **Test the service:**
```bash
curl http://localhost:8080/healthz
# Should return: {"status":"ok"}

curl http://localhost:8080/stats
# Should return batch statistics
```

---

## Still Having Issues?

1. **Share full logs:**
   ```bash
   docker logs <container-name> > startup_logs.txt
   ```

2. **Share GPU info:**
   ```bash
   nvidia-smi > gpu_info.txt
   ```

3. **Share disk/memory:**
   ```bash
   docker stats --no-stream > docker_stats.txt
   df -h > disk_space.txt
   ```

4. **Check Docker version:**
   ```bash
   docker --version
   docker compose version
   ```
