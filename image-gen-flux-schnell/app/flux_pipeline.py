import os
import textwrap
import time
import uuid
import asyncio
import gc
from io import BytesIO
from pathlib import Path
from typing import Optional, Protocol, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

import torch
from PIL import Image, ImageDraw, ImageFont

from .schemas import GenerateRequest, GenerateResponse
from .storage import storage


def image_to_bytes(image: Image.Image) -> bytes:
    """Convert a PIL Image to PNG bytes."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def clear_gpu_memory(gpu_id: Optional[int] = None):
    """
    Aggressively clear GPU memory to prevent memory leaks.

    Call this after each batch to ensure memory is released.
    """
    # Force garbage collection of Python objects
    gc.collect()

    # Clear PyTorch's CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        # Synchronize to ensure operations are complete
        if gpu_id is not None:
            torch.cuda.synchronize(f"cuda:{gpu_id}")
        else:
            torch.cuda.synchronize()


def log_gpu_memory(prefix: str = "", gpu_id: Optional[int] = None):
    """Log current GPU memory usage for debugging."""
    if not torch.cuda.is_available():
        return

    device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda:0"
    allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB

    print(f"[GPU Memory{f' GPU{gpu_id}' if gpu_id is not None else ''}] {prefix} Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


# TensorCore-aligned batch sizes for optimal GPU utilization
# Multiples of 8 ensure proper alignment for FP16/BF16 operations
# FLUX supports up to 32 on high-VRAM GPUs
TENSORCORE_ALIGNED_BATCH_SIZES = [1, 8, 16, 24, 32]


def get_tensorcore_aligned_batch_size(raw_batch_size: int) -> int:
    """
    Round down to the nearest TensorCore-aligned batch size.

    TensorCores are most efficient when batch dimensions are multiples of 8 (FP16/BF16)
    or 16 (INT8). Non-aligned sizes cause tile quantization and reduced GPU utilization.

    Args:
        raw_batch_size: The calculated batch size before alignment

    Returns:
        The next lowest TensorCore-aligned batch size
    """
    for size in reversed(TENSORCORE_ALIGNED_BATCH_SIZES):
        if size <= raw_batch_size:
            return size
    return 1


def calculate_optimal_batch_size(gpu_id: Optional[int] = None) -> int:
    """
    Calculate optimal batch size based on available GPU memory.

    Returns a TensorCore-aligned batch size (multiple of 8) for optimal GPU utilization.
    Non-aligned batch sizes cause tile quantization and significantly higher latency.

    FLUX Schnell is a larger model than SDXL, requiring more conservative batch sizes.

    Memory estimates:
    - FLUX Schnell: ~12GB model + ~2.5GB per image
    - Safety margin: 15% of total VRAM

    Args:
        gpu_id: Specific GPU to check, or None for cuda:0

    Returns:
        TensorCore-aligned batch size (1, 8, 16, 24, or 32)
    """
    # Check if manual override is set
    manual_batch_size = os.getenv("MAX_BATCH_SIZE")
    if manual_batch_size:
        try:
            raw_size = max(1, min(32, int(manual_batch_size)))
            aligned_size = get_tensorcore_aligned_batch_size(raw_size)
            print(f"[calculate_optimal_batch_size] Manual override: {manual_batch_size} -> {aligned_size} (TensorCore-aligned)")
            return aligned_size
        except ValueError:
            pass

    # Default if no GPU or can't detect
    default_batch_size = 8  # Aligned default

    if not torch.cuda.is_available():
        print("[calculate_optimal_batch_size] No CUDA available, using default batch size: 1")
        return 1

    try:
        device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda:0"

        # Get total GPU memory in GB
        total_memory_bytes = torch.cuda.get_device_properties(device).total_memory
        total_memory_gb = total_memory_bytes / (1024 ** 3)

        gpu_name = torch.cuda.get_device_name(device)

        print(f"[calculate_optimal_batch_size] GPU {gpu_id if gpu_id is not None else 0}: {gpu_name}")
        print(f"[calculate_optimal_batch_size] Total VRAM: {total_memory_gb:.2f} GB")

        # FLUX Schnell model memory requirements
        # FLUX is a larger model (~12GB for the base model in BF16)
        model_memory_gb = 12.0  # Conservative estimate for FLUX Schnell in BF16
        per_image_memory_gb = 2.5  # FLUX uses more memory per image due to larger latent space

        # Safety margin (15% of total memory)
        safety_margin_gb = total_memory_gb * 0.15

        # Calculate available memory for batching
        available_for_batching = total_memory_gb - model_memory_gb - safety_margin_gb

        if available_for_batching <= 0:
            print(f"[calculate_optimal_batch_size] WARNING: Insufficient VRAM for batching, using batch size 1")
            return 1

        # Calculate raw batch size from memory
        raw_batch_size = int(available_for_batching / per_image_memory_gb)

        # Clamp to maximum of 32 (FLUX is memory intensive)
        raw_batch_size = max(1, min(32, raw_batch_size))

        # Align to TensorCore-friendly size (round down to nearest multiple of 8)
        optimal_batch_size = get_tensorcore_aligned_batch_size(raw_batch_size)

        print(f"[calculate_optimal_batch_size] Raw batch size: {raw_batch_size} -> {optimal_batch_size} (TensorCore-aligned)")
        print(f"[calculate_optimal_batch_size]   Model: {model_memory_gb:.1f}GB")
        print(f"[calculate_optimal_batch_size]   Batching: {available_for_batching:.1f}GB ({optimal_batch_size} x {per_image_memory_gb:.1f}GB)")
        print(f"[calculate_optimal_batch_size]   Safety margin: {safety_margin_gb:.1f}GB")

        return optimal_batch_size

    except Exception as e:
        print(f"[calculate_optimal_batch_size] Error detecting GPU memory: {e}")
        print(f"[calculate_optimal_batch_size] Using default batch size: {default_batch_size}")
        return default_batch_size


@dataclass
class BatchStats:
    """Statistics for monitoring batch processing performance."""
    total_requests: int = 0
    total_batches: int = 0
    current_queue_depth: int = 0
    avg_batch_size: float = 0.0
    avg_wait_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    max_queue_depth: int = 0
    recent_batch_sizes: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_wait_times: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_gen_times: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class QueuedRequest:
    """Represents a request waiting in the batch queue."""
    request: GenerateRequest
    future: asyncio.Future
    queued_at: datetime
    position: int = 0


class ImageGenBackend(Protocol):
    def generate(self, req: GenerateRequest) -> GenerateResponse:
        ...


class LocalFluxBackend:
    """Local FLUX Schnell backend using the diffusers library."""

    def __init__(self, output_dir: Path, public_base_url: str, gpu_id: Optional[int] = None):
        self.output_dir = output_dir
        self.public_base_url = public_base_url

        self.model_id = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
        device_env = os.getenv("DEVICE", "auto")  # "auto" | "cuda" | "cpu"

        # Determine device with optional GPU ID
        if device_env == "cpu":
            device = "cpu"
        elif gpu_id is not None:
            # Specific GPU requested
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                device = f"cuda:{gpu_id}"
            else:
                print(f"[WARNING] GPU {gpu_id} not available, falling back to cuda:0 or cpu")
                device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_env == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:  # auto
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.gpu_id = gpu_id

        # FLUX Schnell works best with bfloat16 on supported GPUs
        if "cuda" in str(self.device):
            # Check if GPU supports bfloat16
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
                print(f"[image-gen] Using bfloat16 precision (optimal for FLUX)")
            else:
                torch_dtype = torch.float16
                print(f"[image-gen] GPU doesn't support bfloat16, using float16")
        else:
            torch_dtype = torch.float32

        self.torch_dtype = torch_dtype

        gpu_info = f" (GPU {gpu_id})" if gpu_id is not None else ""
        print(f"[image-gen] Using LOCAL FLUX backend with model {self.model_id} on {self.device}{gpu_info}")

        # Set FLUX-specific defaults
        self._set_model_defaults()

        # Import FLUX pipeline from diffusers
        from diffusers import FluxPipeline

        model_size = "~12GB"
        print(f"[image-gen] Detected model type: FLUX Schnell")
        print(f"[image-gen] Loading model from Hugging Face...")
        print(f"[image-gen] NOTE: First run will download {model_size} model files - this can take 5-20 minutes!")
        print(f"[image-gen] Subsequent runs will use cached model and start in ~30-60 seconds")

        import sys
        sys.stdout.flush()  # Force flush to ensure logs appear

        # Load model - FLUX may require HuggingFace token
        hf_token = os.getenv("HF_TOKEN")
        load_kwargs = {
            "torch_dtype": torch_dtype,
        }
        if hf_token:
            load_kwargs["token"] = hf_token
            print(f"[image-gen] Using HuggingFace token for authenticated access")

        self.pipe = FluxPipeline.from_pretrained(
            self.model_id,
            **load_kwargs
        )
        print(f"[image-gen] Model loaded, moving to device {self.device}...")
        self.pipe.to(self.device)
        print(f"[image-gen] Model ready on {self.device}")

        # Enable memory optimizations if available
        try:
            self.pipe.enable_attention_slicing()
            print(f"[image-gen] Attention slicing enabled")
        except Exception as e:
            print(f"[image-gen] Could not enable attention slicing: {e}")

    def _set_model_defaults(self):
        """Set default parameters for FLUX Schnell."""
        # FLUX Schnell is optimized for speed with few steps
        # It uses guidance-free generation (guidance_scale is often 0 or very low)
        self.default_guidance_scale = 0.0  # FLUX Schnell doesn't use CFG
        self.default_num_steps = 4  # FLUX Schnell is optimized for 4 steps
        print(f"[image-gen] Model defaults: FLUX Schnell (guidance_scale=0.0, steps=4)")

    def _apply_request_defaults(self, req: GenerateRequest) -> GenerateRequest:
        """Apply model-specific defaults to request if not explicitly set."""
        if req.guidance_scale is None:
            req.guidance_scale = self.default_guidance_scale
        if req.num_inference_steps is None:
            req.num_inference_steps = self.default_num_steps
        return req

    def _save_image(self, image: Image.Image, req: GenerateRequest, seed: int, duration_ms: int) -> GenerateResponse:
        image_id = str(uuid.uuid4())

        # Upload to MinIO
        image_bytes = image_to_bytes(image)
        image_url = storage.upload_image(image_bytes, req.game_id, req.round_id, image_id)

        # Fallback to local storage if MinIO upload fails
        if not image_url:
            filename = f"{image_id}.png"
            game_dir = self.output_dir / req.game_id / req.round_id
            game_dir.mkdir(parents=True, exist_ok=True)
            filepath = game_dir / filename
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(filepath, format="PNG")
            rel_path = filepath.relative_to(self.output_dir.parent)
            image_url = f"{self.public_base_url}/{rel_path.as_posix()}"

        return GenerateResponse(
            image_id=image_id,
            image_url=image_url,
            game_id=req.game_id,
            round_id=req.round_id,
            player_id=req.player_id,
            model_id=self.model_id,
            seed=seed,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            duration_ms=duration_ms,
        )

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        # Apply model-specific defaults
        req = self._apply_request_defaults(req)

        start = time.time()

        if req.seed is not None:
            seed = req.seed
        else:
            seed = torch.randint(0, 2**31 - 1, (1,)).item()

        if "cuda" in self.device:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)

        with torch.inference_mode():
            # FLUX Schnell generation
            # Note: FLUX Schnell doesn't use negative prompts effectively
            image = self.pipe(
                prompt=req.prompt,
                width=req.width,
                height=req.height,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                generator=generator,
            ).images[0]

        duration_ms = int((time.time() - start) * 1000)
        return self._save_image(image, req, seed, duration_ms)


class BatchedLocalFluxBackend:
    """
    Batched wrapper around LocalFluxBackend that intelligently groups requests
    for concurrent GPU inference, dramatically improving throughput.

    Features:
    - Dynamic batch formation with configurable window
    - Adaptive batch sizing based on queue depth
    - Comprehensive performance monitoring
    - Thread-safe concurrent request handling
    """

    def __init__(self, output_dir: Path, public_base_url: str, gpu_id: Optional[int] = None):
        # Initialize the underlying backend
        self.backend = LocalFluxBackend(output_dir, public_base_url, gpu_id=gpu_id)
        self.gpu_id = gpu_id

        # Batch configuration - automatically detect optimal batch size based on GPU VRAM
        # FLUX is more memory intensive, so we use smaller batch sizes
        self.max_batch_size = calculate_optimal_batch_size(gpu_id)
        self.min_batch_size = int(os.getenv("MIN_BATCH_SIZE", "1"))
        self.max_wait_time = float(os.getenv("MAX_BATCH_WAIT_TIME", "5.0"))  # seconds
        self.min_wait_time = float(os.getenv("MIN_BATCH_WAIT_TIME", "0.5"))  # seconds

        # Adaptive batching: reduce wait time when queue is deep
        self.enable_adaptive_batching = os.getenv("ENABLE_ADAPTIVE_BATCHING", "true").lower() == "true"
        self.adaptive_threshold = int(os.getenv("ADAPTIVE_BATCH_THRESHOLD", "10"))

        # Queue and synchronization
        self._pending_requests: List[QueuedRequest] = []
        self._queue_lock = asyncio.Lock()
        self._processing = False
        self._processor_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = BatchStats()
        self._stats_lock = asyncio.Lock()

        gpu_info = f" (GPU {gpu_id})" if gpu_id is not None else ""
        print(f"[BatchedLocalFluxBackend{gpu_info}] Initialized with:")
        print(f"  - Max batch size: {self.max_batch_size}")
        print(f"  - Min batch size: {self.min_batch_size}")
        print(f"  - Max wait time: {self.max_wait_time}s")
        print(f"  - Min wait time: {self.min_wait_time}s")
        print(f"  - Adaptive batching: {self.enable_adaptive_batching}")

    async def start_processor(self):
        """Start the background batch processor."""
        if self._processor_task is None or self._processor_task.done():
            self._processor_task = asyncio.create_task(self._batch_processor())
            print("[BatchedLocalFluxBackend] Batch processor started")

    async def _batch_processor(self):
        """Background task that continuously processes batches."""
        while True:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms

                async with self._queue_lock:
                    if not self._pending_requests or self._processing:
                        continue

                    # Calculate dynamic wait time based on queue depth
                    wait_time = self._calculate_wait_time()

                    # Get the oldest request's age
                    oldest_request = self._pending_requests[0]
                    request_age = (datetime.now() - oldest_request.queued_at).total_seconds()

                    # Decide if we should process now
                    should_process = (
                        len(self._pending_requests) >= self.max_batch_size or
                        (len(self._pending_requests) >= self.min_batch_size and request_age >= wait_time)
                    )

                    if should_process:
                        self._processing = True
                        batch = self._pending_requests[:self.max_batch_size]
                        self._pending_requests = self._pending_requests[self.max_batch_size:]

                        # Update queue depth stats
                        async with self._stats_lock:
                            self.stats.current_queue_depth = len(self._pending_requests)

                # Process batch outside the lock to allow new requests
                if should_process:
                    await self._process_batch(batch)
                    async with self._queue_lock:
                        self._processing = False

            except Exception as e:
                print(f"[BatchedLocalFluxBackend] Error in batch processor: {e}")
                async with self._queue_lock:
                    self._processing = False
                await asyncio.sleep(1)

    def _calculate_wait_time(self) -> float:
        """Calculate adaptive wait time based on queue depth."""
        if not self.enable_adaptive_batching:
            return self.max_wait_time

        queue_depth = len(self._pending_requests)

        # If queue is deep, reduce wait time to process faster
        if queue_depth >= self.adaptive_threshold:
            # Linear interpolation between min and max wait time
            # More items in queue = shorter wait time
            ratio = min(1.0, queue_depth / (self.adaptive_threshold * 2))
            wait_time = self.max_wait_time - (ratio * (self.max_wait_time - self.min_wait_time))
            return max(self.min_wait_time, wait_time)

        return self.max_wait_time

    async def _process_batch(self, batch: List[QueuedRequest]):
        """Process a batch of requests using the GPU."""
        if not batch:
            return

        batch_start = time.time()
        batch_size = len(batch)

        print(f"[BatchedLocalFluxBackend] Processing batch of {batch_size} requests")

        # Log GPU memory before batch
        log_gpu_memory("Before batch:", self.gpu_id)

        try:
            # Extract requests and calculate wait times
            requests = [item.request for item in batch]
            wait_times = [(datetime.now() - item.queued_at).total_seconds() * 1000 for item in batch]

            # Generate all images in one batch
            results = await self._generate_batch(requests)

            # Distribute results to futures
            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)

            # Update statistics
            gen_time_ms = (time.time() - batch_start) * 1000
            async with self._stats_lock:
                self.stats.total_requests += batch_size
                self.stats.total_batches += 1
                self.stats.recent_batch_sizes.append(batch_size)
                self.stats.recent_wait_times.extend(wait_times)
                self.stats.recent_gen_times.append(gen_time_ms)

                # Update averages
                if self.stats.recent_batch_sizes:
                    self.stats.avg_batch_size = sum(self.stats.recent_batch_sizes) / len(self.stats.recent_batch_sizes)
                if self.stats.recent_wait_times:
                    self.stats.avg_wait_time_ms = sum(self.stats.recent_wait_times) / len(self.stats.recent_wait_times)
                if self.stats.recent_gen_times:
                    self.stats.avg_generation_time_ms = sum(self.stats.recent_gen_times) / len(self.stats.recent_gen_times)

            print(f"[BatchedLocalFluxBackend] Batch completed in {gen_time_ms:.0f}ms (avg wait: {sum(wait_times)/len(wait_times):.0f}ms)")

        except Exception as e:
            print(f"[BatchedLocalFluxBackend] Error processing batch: {e}")
            import traceback
            traceback.print_exc()

            # Set exception on all futures
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)

        finally:
            # CRITICAL: Clean up GPU memory after batch
            # This prevents memory leaks between batches
            log_gpu_memory("After batch (before cleanup):", self.gpu_id)
            clear_gpu_memory(self.gpu_id)
            log_gpu_memory("After cleanup:", self.gpu_id)

    async def _generate_batch(self, requests: List[GenerateRequest]) -> List[GenerateResponse]:
        """Generate multiple images in a single batch using the GPU."""
        if len(requests) == 1:
            # Single request - use original backend
            result = await asyncio.to_thread(self.backend.generate, requests[0])
            return [result]

        # Multi-request batch processing
        start = time.time()

        # Apply model-specific defaults to all requests
        requests = [self.backend._apply_request_defaults(req) for req in requests]

        # Collect all prompts
        prompts = [req.prompt for req in requests]

        # Use parameters from first request (assume all similar in a batch)
        base_req = requests[0]

        # Generate seeds for each request
        seeds = []
        for req in requests:
            if req.seed is not None:
                seeds.append(req.seed)
            else:
                seeds.append(torch.randint(0, 2**31 - 1, (1,)).item())

        # Create generator for the first seed (batch generation uses one seed)
        if "cuda" in self.backend.device:
            generator = torch.Generator(device=self.backend.device).manual_seed(seeds[0])
        else:
            generator = torch.Generator().manual_seed(seeds[0])

        # Run batch inference on GPU
        def _batch_inference():
            with torch.inference_mode():
                # FLUX batch inference
                images = self.backend.pipe(
                    prompt=prompts,
                    width=base_req.width,
                    height=base_req.height,
                    num_inference_steps=base_req.num_inference_steps,
                    guidance_scale=base_req.guidance_scale,
                    generator=generator,
                ).images
            return images

        images = await asyncio.to_thread(_batch_inference)

        # Save each image and create responses
        responses = []
        for i, (image, req, seed) in enumerate(zip(images, requests, seeds)):
            duration_ms = int((time.time() - start) * 1000)
            response = self.backend._save_image(image, req, seed, duration_ms)
            responses.append(response)

        # Explicitly delete large objects to free memory immediately
        del images
        del generator
        del prompts

        return responses

    async def generate(self, req: GenerateRequest) -> GenerateResponse:
        """
        Queue a generation request and wait for it to be processed in a batch.

        This is the main entry point for async request handling.
        """
        # Create future for this request
        future = asyncio.Future()
        queued_request = QueuedRequest(
            request=req,
            future=future,
            queued_at=datetime.now()
        )

        # Add to queue
        async with self._queue_lock:
            position = len(self._pending_requests)
            queued_request.position = position
            self._pending_requests.append(queued_request)

            # Update max queue depth
            async with self._stats_lock:
                self.stats.current_queue_depth = len(self._pending_requests)
                if self.stats.current_queue_depth > self.stats.max_queue_depth:
                    self.stats.max_queue_depth = self.stats.current_queue_depth

        print(f"[BatchedLocalFluxBackend] Request queued at position {position} (queue depth: {position + 1})")

        # Wait for result
        return await future

    async def get_stats(self) -> dict:
        """Get current batch processing statistics."""
        async with self._stats_lock:
            return {
                "total_requests": self.stats.total_requests,
                "total_batches": self.stats.total_batches,
                "current_queue_depth": self.stats.current_queue_depth,
                "max_queue_depth": self.stats.max_queue_depth,
                "avg_batch_size": round(self.stats.avg_batch_size, 2),
                "avg_wait_time_ms": round(self.stats.avg_wait_time_ms, 2),
                "avg_generation_time_ms": round(self.stats.avg_generation_time_ms, 2),
                "config": {
                    "max_batch_size": self.max_batch_size,
                    "min_batch_size": self.min_batch_size,
                    "max_wait_time": self.max_wait_time,
                    "min_wait_time": self.min_wait_time,
                    "adaptive_batching": self.enable_adaptive_batching,
                    "adaptive_threshold": self.adaptive_threshold,
                }
            }


class MultiGPUBatchedBackend:
    """
    Multi-GPU batching backend that distributes requests across multiple GPUs.

    Features:
    - Automatic GPU detection or manual GPU selection via env var
    - Load balancing across GPUs using round-robin or queue-based strategy
    - Independent batch processing per GPU for maximum parallelism
    - Aggregated statistics across all GPUs
    """

    def __init__(self, output_dir: Path, public_base_url: str):
        self.output_dir = output_dir
        self.public_base_url = public_base_url

        # Determine which GPUs to use
        self.gpu_ids = self._detect_gpus()

        if not self.gpu_ids:
            raise RuntimeError("No GPUs available for MultiGPUBatchedBackend")

        print(f"[MultiGPUBatchedBackend] Initializing with {len(self.gpu_ids)} GPU(s): {self.gpu_ids}")

        # Create one BatchedLocalFluxBackend per GPU
        self.backends: List[BatchedLocalFluxBackend] = []
        for gpu_id in self.gpu_ids:
            backend = BatchedLocalFluxBackend(output_dir, public_base_url, gpu_id=gpu_id)
            self.backends.append(backend)

        # Load balancing strategy
        self.strategy = os.getenv("GPU_LOAD_BALANCE_STRATEGY", "round-robin")  # "round-robin" or "queue-depth"
        self._round_robin_index = 0
        self._round_robin_lock = asyncio.Lock()

        print(f"[MultiGPUBatchedBackend] Load balancing strategy: {self.strategy}")

    def _detect_gpus(self) -> List[int]:
        """
        Detect available GPUs from environment or system.
        """
        # Check GPU_IDS env var first
        gpu_ids_env = os.getenv("GPU_IDS")
        if gpu_ids_env:
            try:
                gpu_ids = [int(x.strip()) for x in gpu_ids_env.split(",")]
                print(f"[MultiGPUBatchedBackend] Using GPUs from GPU_IDS env var: {gpu_ids}")
                return gpu_ids
            except ValueError as e:
                print(f"[WARNING] Invalid GPU_IDS format: {gpu_ids_env}, {e}")

        # Check CUDA_VISIBLE_DEVICES
        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible != "":
            try:
                gpu_ids = [i for i in range(len(cuda_visible.split(",")))]
                print(f"[MultiGPUBatchedBackend] Using GPUs from CUDA_VISIBLE_DEVICES: {gpu_ids} (mapped)")
                return gpu_ids
            except Exception as e:
                print(f"[WARNING] Error parsing CUDA_VISIBLE_DEVICES: {e}")

        # Auto-detect all available GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_ids = list(range(num_gpus))
            print(f"[MultiGPUBatchedBackend] Auto-detected {num_gpus} GPU(s): {gpu_ids}")
            return gpu_ids

        return []

    async def start_processor(self):
        """Start batch processors on all GPUs."""
        print(f"[MultiGPUBatchedBackend] Starting batch processors on {len(self.backends)} GPU(s)...")
        tasks = [backend.start_processor() for backend in self.backends]
        await asyncio.gather(*tasks)
        print(f"[MultiGPUBatchedBackend] All batch processors started")

    async def _select_backend(self) -> BatchedLocalFluxBackend:
        """Select a backend based on load balancing strategy."""
        if self.strategy == "queue-depth":
            # Choose GPU with shortest queue
            min_queue_depth = float('inf')
            selected_backend = self.backends[0]

            for backend in self.backends:
                async with backend._stats_lock:
                    if backend.stats.current_queue_depth < min_queue_depth:
                        min_queue_depth = backend.stats.current_queue_depth
                        selected_backend = backend

            return selected_backend
        else:
            # Round-robin (default)
            async with self._round_robin_lock:
                backend = self.backends[self._round_robin_index]
                self._round_robin_index = (self._round_robin_index + 1) % len(self.backends)
                return backend

    async def generate(self, req: GenerateRequest) -> GenerateResponse:
        """
        Generate an image using the least-loaded GPU.
        """
        backend = await self._select_backend()
        return await backend.generate(req)

    async def get_stats(self) -> dict:
        """Get aggregated statistics across all GPUs."""
        backend_stats = await asyncio.gather(*[b.get_stats() for b in self.backends])

        # Aggregate totals
        total_requests = sum(s["total_requests"] for s in backend_stats)
        total_batches = sum(s["total_batches"] for s in backend_stats)
        max_queue_depth_overall = max(s["max_queue_depth"] for s in backend_stats)

        # Calculate weighted averages
        if total_batches > 0:
            avg_batch_size = sum(
                s["avg_batch_size"] * s["total_batches"] for s in backend_stats
            ) / total_batches
            avg_wait_time_ms = sum(
                s["avg_wait_time_ms"] * s["total_requests"] for s in backend_stats
            ) / total_requests if total_requests > 0 else 0
            avg_generation_time_ms = sum(
                s["avg_generation_time_ms"] * s["total_batches"] for s in backend_stats
            ) / total_batches
        else:
            avg_batch_size = 0
            avg_wait_time_ms = 0
            avg_generation_time_ms = 0

        return {
            "num_gpus": len(self.backends),
            "gpu_ids": self.gpu_ids,
            "load_balance_strategy": self.strategy,
            "total_requests": total_requests,
            "total_batches": total_batches,
            "max_queue_depth": max_queue_depth_overall,
            "avg_batch_size": round(avg_batch_size, 2),
            "avg_wait_time_ms": round(avg_wait_time_ms, 2),
            "avg_generation_time_ms": round(avg_generation_time_ms, 2),
            "per_gpu_stats": [
                {
                    "gpu_id": self.gpu_ids[i],
                    "current_queue_depth": s["current_queue_depth"],
                    "total_requests": s["total_requests"],
                    "total_batches": s["total_batches"],
                    "avg_batch_size": s["avg_batch_size"],
                }
                for i, s in enumerate(backend_stats)
            ],
            "config": backend_stats[0]["config"] if backend_stats else {}
        }


class MockBackend:
    """
    Lightweight fallback that creates a placeholder image when real generation is unavailable.
    """

    def __init__(self, output_dir: Path, public_base_url: str):
        self.output_dir = output_dir
        self.public_base_url = public_base_url
        self.model_id = "mock-placeholder"

    def _save_image(self, image: Image.Image, req: GenerateRequest, duration_ms: int) -> GenerateResponse:
        image_id = str(uuid.uuid4())

        # Upload to MinIO
        image_bytes = image_to_bytes(image)
        image_url = storage.upload_image(image_bytes, req.game_id, req.round_id, image_id)

        # Fallback to local storage if MinIO upload fails
        if not image_url:
            filename = f"{image_id}.png"
            game_dir = self.output_dir / req.game_id / req.round_id
            game_dir.mkdir(parents=True, exist_ok=True)
            filepath = game_dir / filename
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(filepath, format="PNG")
            rel_path = filepath.relative_to(self.output_dir.parent)
            image_url = f"{self.public_base_url}/{rel_path.as_posix()}"

        return GenerateResponse(
            image_id=image_id,
            image_url=image_url,
            game_id=req.game_id,
            round_id=req.round_id,
            player_id=req.player_id,
            model_id=self.model_id,
            seed=req.seed or 0,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps or 4,
            guidance_scale=req.guidance_scale or 0.0,
            duration_ms=duration_ms,
        )

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        start = time.time()

        # Build a simple placeholder image
        image = Image.new("RGB", (req.width, req.height), color=(32, 32, 32))
        draw = ImageDraw.Draw(image)
        text = f"FLUX Schnell (Mock)\n\nPrompt:\n{textwrap.fill(req.prompt, width=40)}"
        font = ImageFont.load_default()
        draw.multiline_text((20, 20), text, fill=(255, 255, 255), font=font, spacing=4)

        duration_ms = int((time.time() - start) * 1000)
        return self._save_image(image, req, duration_ms)


class FluxSchnellService:
    """
    Facade that picks a backend based on GEN_MODE.
    Supports batched and non-batched modes for local generation.
    Automatically uses multi-GPU backend when multiple GPUs are available.
    """

    def __init__(self) -> None:
        output_dir = Path(os.getenv("OUTPUT_DIR", "/data/images"))
        output_dir.mkdir(parents=True, exist_ok=True)
        public_base_url = os.getenv("PUBLIC_BASE_URL", "http://image-gen-flux:8080")

        gen_mode = os.getenv("GEN_MODE", "local").lower()
        enable_batching = os.getenv("ENABLE_BATCHING", "true").lower() == "true"
        force_multi_gpu = os.getenv("FORCE_MULTI_GPU", "false").lower() == "true"
        mock_backend = MockBackend(output_dir, public_base_url)

        self.is_async = False  # Track if backend supports async

        if gen_mode == "mock":
            print("[image-gen-flux] Using MOCK backend (placeholder images).")
            self.backend: ImageGenBackend = mock_backend
        else:
            # default to local with optional batching and multi-GPU support
            if enable_batching:
                # Detect if we should use multi-GPU mode
                num_gpus = self._detect_gpu_count()
                use_multi_gpu = force_multi_gpu or (num_gpus > 1)

                if use_multi_gpu and num_gpus > 0:
                    print(f"[image-gen-flux] Using LOCAL FLUX backend with MULTI-GPU BATCHING ({num_gpus} GPUs).")
                    self.backend = MultiGPUBatchedBackend(output_dir, public_base_url)
                    self.is_async = True
                else:
                    print("[image-gen-flux] Using LOCAL FLUX backend with BATCHING enabled (single GPU).")
                    self.backend = BatchedLocalFluxBackend(output_dir, public_base_url)
                    self.is_async = True
            else:
                print("[image-gen-flux] Using LOCAL FLUX backend (batching disabled).")
                self.backend = LocalFluxBackend(output_dir, public_base_url)

    def _detect_gpu_count(self) -> int:
        """Detect number of available GPUs for this service."""
        # Check GPU_IDS env var first
        gpu_ids_env = os.getenv("GPU_IDS")
        if gpu_ids_env:
            try:
                gpu_ids = [int(x.strip()) for x in gpu_ids_env.split(",")]
                return len(gpu_ids)
            except ValueError:
                pass

        # Check CUDA_VISIBLE_DEVICES
        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible != "":
            try:
                return len(cuda_visible.split(","))
            except Exception:
                pass

        # Auto-detect
        if torch.cuda.is_available():
            return torch.cuda.device_count()

        return 0

    async def start_processor(self):
        """Start the batch processor if using batched backend."""
        if self.is_async and hasattr(self.backend, 'start_processor'):
            await self.backend.start_processor()

    async def generate_async(self, req: GenerateRequest) -> GenerateResponse:
        """Async generate method for batched backends."""
        result = self.backend.generate(req)

        # If the result is a coroutine, await it
        if asyncio.iscoroutine(result):
            result = await result

        return result

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        """Sync generate method (deprecated - use generate_async)."""
        return self.backend.generate(req)

    async def get_stats(self) -> dict:
        """Get batch processing statistics if available."""
        if hasattr(self.backend, 'get_stats'):
            return await self.backend.get_stats()
        return {"error": "Stats not available for this backend"}
