"""
FastAPI application for image similarity service.

Endpoints:
- POST /similarity: Compute similarity for image pairs
- GET /health: Health check endpoint
"""
import asyncio
import logging
import os
import socket
import time
from contextlib import asynccontextmanager
from typing import List, Tuple
from urllib.parse import urlparse

import httpx
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .config import (
    MODEL_TYPE,
    DEVICE,
    MAX_PAIRS_PER_REQUEST,
    BATCH_SIZE_LIMIT,
    URL_FETCH_TIMEOUT,
)
from .model import init_model, get_model, distance_to_similarity
from .preprocessing import load_image, preprocess_image, ImageLoadError
from .schemas import (
    SimilarityRequest,
    SimilarityResponse,
    SimilarityScore,
    HealthResponse,
    ErrorResponse,
    ErrorDetail,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_tracing_initialized = False


def _normalized_otlp_endpoint() -> str:
    """Build an OTLP HTTP endpoint from env vars."""
    base = (
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        or "http://otel-collector:4318"
    ).rstrip("/")
    if not base.endswith("/v1/traces"):
        base = f"{base}/v1/traces"
    return base


def _wait_for_collector(endpoint: str, timeout: int = 30) -> bool:
    """Wait for the OTEL collector to be reachable."""
    parsed = urlparse(endpoint)
    host = parsed.hostname or "localhost"
    port = parsed.port or 4318

    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.create_connection((host, port), timeout=2)
            sock.close()
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(1)
    return False


def init_tracing(app: FastAPI):
    """Initialize OpenTelemetry tracing."""
    global _tracing_initialized
    if _tracing_initialized:
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "image-sim")
    otlp_endpoint = _normalized_otlp_endpoint()

    collector_base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
    if not _wait_for_collector(collector_base):
        logger.warning(f"[otel] Collector not reachable at {collector_base}, tracing disabled")
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    try:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        span_processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()
        _tracing_initialized = True
        logger.info(f"[otel] Tracing initialized for {service_name} -> {otlp_endpoint}")
    except Exception as exc:
        logger.error(f"[otel] Failed to initialize tracing: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - load model at startup."""
    logger.info("Starting up image similarity service...")

    # Initialize tracing
    init_tracing(app)

    try:
        init_model(DEVICE)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    logger.info("Shutting down image similarity service...")


app = FastAPI(
    title="Image Similarity Service",
    description="Microservice for computing perceptual image similarity using DreamSim",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model = get_model()
    return HealthResponse(
        status="healthy" if model.is_initialized() else "degraded",
        model_loaded=model.is_initialized(),
        device=model.get_device() if model.is_initialized() else None,
        model_type=MODEL_TYPE,
    )


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "image-similarity-service",
        "version": "1.0.0",
        "endpoints": {
            "POST /similarity": "Compute similarity for image pairs",
            "GET /health": "Health check",
        }
    }


async def load_and_preprocess_pair(
    pair_idx: int,
    reference_source: str,
    generated_source: str,
    preprocess_fn,
    client: httpx.AsyncClient,
) -> Tuple[int, torch.Tensor, torch.Tensor, str]:
    """
    Load and preprocess an image pair.
    
    Returns:
        Tuple of (pair_idx, ref_tensor, gen_tensor, error_message)
        If successful, error_message will be empty string.
    """
    try:
        # Load images (handles both URL and base64)
        ref_image = await load_image(reference_source, client)
        gen_image = await load_image(generated_source, client)
        
        # Preprocess images
        ref_tensor = preprocess_image(ref_image, preprocess_fn)
        gen_tensor = preprocess_image(gen_image, preprocess_fn)
        
        return (pair_idx, ref_tensor, gen_tensor, "")
    except ImageLoadError as e:
        return (pair_idx, None, None, str(e))
    except Exception as e:
        return (pair_idx, None, None, f"Unexpected error: {e}")


@app.post(
    "/similarity",
    response_model=SimilarityResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)
async def compute_similarity(request: SimilarityRequest):
    """
    Compute perceptual similarity for one or more image pairs.
    
    Each pair consists of a reference image and a generated image.
    Images can be provided as base64-encoded strings or URLs.
    
    Returns distance (0 = identical, larger = more different) and
    similarity (0-1, higher = more similar) for each pair.
    """
    start_time = time.perf_counter()
    
    model = get_model()
    if not model.is_initialized():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    n_pairs = len(request.pairs)
    logger.info(f"Processing {n_pairs} image pairs")
    
    preprocess_fn = model.get_preprocess()
    
    # Load and preprocess all images concurrently
    errors: List[ErrorDetail] = []
    results: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    pair_ids: List[str] = []
    
    async with httpx.AsyncClient(timeout=URL_FETCH_TIMEOUT) as client:
        tasks = [
            load_and_preprocess_pair(
                i,
                pair.reference_image,
                pair.generated_image,
                preprocess_fn,
                client,
            )
            for i, pair in enumerate(request.pairs)
        ]
        
        # Process all pairs concurrently
        loaded_pairs = await asyncio.gather(*tasks)
    
    # Separate successful loads from errors
    for pair_idx, ref_tensor, gen_tensor, error_msg in loaded_pairs:
        pair = request.pairs[pair_idx]
        if error_msg:
            errors.append(ErrorDetail(
                message=error_msg,
                pair_index=pair_idx,
                pair_id=pair.pair_id,
            ))
        else:
            results.append((pair_idx, ref_tensor, gen_tensor))
            pair_ids.append(pair.pair_id)
    
    # If all pairs failed, return error
    if not results:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error="All image pairs failed to load",
                details=errors,
            ).model_dump()
        )
    
    # Log if some pairs failed
    if errors:
        logger.warning(f"{len(errors)} pairs failed to load")
    
    # Stack tensors into batches
    # Sort by original index to maintain order
    results.sort(key=lambda x: x[0])
    
    ref_tensors = torch.cat([r[1] for r in results], dim=0)
    gen_tensors = torch.cat([r[2] for r in results], dim=0)
    original_indices = [r[0] for r in results]
    
    # Process in chunks if batch is large
    all_distances = []
    
    for i in range(0, len(results), BATCH_SIZE_LIMIT):
        batch_ref = ref_tensors[i:i + BATCH_SIZE_LIMIT]
        batch_gen = gen_tensors[i:i + BATCH_SIZE_LIMIT]
        
        distances = model.compute_distance(batch_ref, batch_gen)
        all_distances.append(distances)
    
    # Concatenate all distances
    distances = torch.cat(all_distances, dim=0)
    similarities = distance_to_similarity(distances)
    
    # Build response, maintaining original order and handling errors
    scores = []
    result_idx = 0
    
    for i in range(n_pairs):
        pair = request.pairs[i]
        
        # Check if this pair had an error
        error_for_pair = next((e for e in errors if e.pair_index == i), None)
        
        if error_for_pair:
            # Return NaN-like values for failed pairs
            scores.append(SimilarityScore(
                distance=float('inf'),
                similarity=0.0,
                pair_id=pair.pair_id,
            ))
        else:
            # Find the result for this index
            scores.append(SimilarityScore(
                distance=float(distances[result_idx].item()),
                similarity=float(similarities[result_idx].item()),
                pair_id=pair.pair_id,
            ))
            result_idx += 1
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"Processed {n_pairs} pairs in {elapsed_ms:.1f}ms")
    
    return SimilarityResponse(
        scores=scores,
        processing_time_ms=elapsed_ms,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "details": str(exc)}
    )
