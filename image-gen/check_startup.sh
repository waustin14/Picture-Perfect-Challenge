#!/bin/bash

# Diagnostic script for image-gen startup issues

CONTAINER_NAME="${1:-image-gen}"

echo "========================================"
echo "Image-Gen Startup Diagnostic"
echo "========================================"
echo

echo "📦 Container: $CONTAINER_NAME"
echo

# Check if container exists
if ! docker ps -a | grep -q $CONTAINER_NAME; then
    echo "❌ Container '$CONTAINER_NAME' not found!"
    echo "Available containers:"
    docker ps -a --format "table {{.Names}}\t{{.Status}}"
    exit 1
fi

echo "1️⃣  Container Status:"
echo "─────────────────────"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "NAMES|$CONTAINER_NAME"
echo

echo "2️⃣  Recent Logs (last 30 lines):"
echo "─────────────────────────────────"
docker logs --tail 30 $CONTAINER_NAME 2>&1
echo

echo "3️⃣  GPU Availability:"
echo "────────────────────"
if docker exec $CONTAINER_NAME nvidia-smi 2>/dev/null; then
    echo "✅ GPU accessible"
else
    echo "❌ GPU not accessible or nvidia-smi not available"
fi
echo

echo "4️⃣  Model Cache Size (Download Progress):"
echo "──────────────────────────────────────────"
CACHE_SIZE=$(docker exec $CONTAINER_NAME du -sh /root/.cache/huggingface/ 2>/dev/null | awk '{print $1}')
if [ -n "$CACHE_SIZE" ]; then
    echo "Cache size: $CACHE_SIZE"
    if [ "$CACHE_SIZE" = "0" ] || [ -z "$CACHE_SIZE" ]; then
        echo "⏳ Model download starting..."
    else
        echo "📥 Downloading/loading... (target: ~6-8GB)"
    fi
else
    echo "⚠️  Cache directory not found (normal on first run)"
fi
echo

echo "5️⃣  Disk Space:"
echo "──────────────"
docker exec $CONTAINER_NAME df -h / 2>/dev/null | grep -E "Filesystem|/" | grep -v "Filesystem"
echo

echo "6️⃣  Network Connectivity:"
echo "────────────────────────"
if docker exec $CONTAINER_NAME ping -c 2 -W 2 huggingface.co >/dev/null 2>&1; then
    echo "✅ Can reach huggingface.co"
else
    echo "❌ Cannot reach huggingface.co (check network/firewall)"
fi
echo

echo "7️⃣  Process Check:"
echo "─────────────────"
docker exec $CONTAINER_NAME ps aux 2>/dev/null | grep -E "python|uvicorn" | grep -v grep
echo

echo "========================================"
echo "Status Summary"
echo "========================================"
echo

# Check for specific log patterns
if docker logs $CONTAINER_NAME 2>&1 | grep -q "Uvicorn running"; then
    echo "✅ SERVICE IS RUNNING!"
    echo
    echo "Test with:"
    PORT=$(docker port $CONTAINER_NAME 2>/dev/null | grep 8080 | cut -d: -f2)
    if [ -n "$PORT" ]; then
        echo "  curl http://localhost:$PORT/healthz"
    else
        echo "  curl http://localhost:8080/healthz"
    fi
elif docker logs $CONTAINER_NAME 2>&1 | grep -q "Loading model from Hugging Face"; then
    echo "⏳ MODEL IS LOADING..."
    echo
    echo "This is normal! Expected wait times:"
    echo "  • First run: 5-15 minutes (downloading ~6-8GB)"
    echo "  • Subsequent runs: 30-60 seconds"
    echo
    echo "Current cache size: $CACHE_SIZE"
    echo
    echo "💡 To monitor progress:"
    echo "  watch -n 5 'docker exec $CONTAINER_NAME du -sh /root/.cache/huggingface/'"
elif docker logs $CONTAINER_NAME 2>&1 | grep -q "CUDA out of memory"; then
    echo "❌ GPU OUT OF MEMORY"
    echo
    echo "Solutions:"
    echo "  1. Reduce batch size: export MAX_BATCH_SIZE=4"
    echo "  2. Check other processes: nvidia-smi"
    echo "  3. Restart container: docker-compose restart $CONTAINER_NAME"
elif docker logs $CONTAINER_NAME 2>&1 | grep -q "Error"; then
    echo "❌ ERROR DETECTED IN LOGS"
    echo
    echo "Last error:"
    docker logs $CONTAINER_NAME 2>&1 | grep -i error | tail -3
    echo
    echo "View full logs:"
    echo "  docker logs $CONTAINER_NAME"
else
    echo "⏳ STARTING UP..."
    echo
    echo "Wait a few moments and run this script again:"
    echo "  ./check_startup.sh $CONTAINER_NAME"
fi

echo
echo "========================================"
echo

# Offer to tail logs
read -p "📜 Tail logs in real-time? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Following logs (Ctrl+C to stop)..."
    echo
    docker logs -f $CONTAINER_NAME
fi
