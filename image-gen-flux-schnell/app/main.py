import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .schemas import GenerateRequest, GenerateResponse
from .flux_pipeline import FluxSchnellService

app = FastAPI(title="Image Generation Service (FLUX Schnell)")

_tracing_initialized = False


def _normalized_otlp_endpoint() -> str:
    """
    Build an OTLP HTTP endpoint from env vars.
    Prefers OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, then OTEL_EXPORTER_OTLP_ENDPOINT,
    and ensures it ends with /v1/traces for the HTTP exporter.
    """
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
    import socket
    from urllib.parse import urlparse

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


def init_tracing():
    global _tracing_initialized
    if _tracing_initialized:
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "image-gen-flux-schnell")
    otlp_endpoint = _normalized_otlp_endpoint()

    # Wait for collector to be available before initializing
    collector_base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
    if not _wait_for_collector(collector_base):
        print(f"[otel] Collector not reachable at {collector_base}, tracing disabled")
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
        print(f"[otel] Tracing initialized for {service_name} -> {otlp_endpoint}")
    except Exception as exc:
        print(f"[otel] Failed to initialize tracing: {exc}")


init_tracing()

# CORS so your web app can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize FLUX pipeline at startup
print("[image-gen-flux] Initializing FluxSchnellService...")
flux_service = FluxSchnellService()
print("[image-gen-flux] FluxSchnellService initialized")

# Serve images from /data/images as /images/...
app.mount(
    "/images",
    StaticFiles(directory="/data/images", html=False),
    name="images",
)


@app.on_event("startup")
async def startup_event():
    """Start the batch processor on application startup."""
    print("[image-gen-flux] Running startup event...")
    try:
        await flux_service.start_processor()
        print("[image-gen-flux] Batch processor started successfully")
    except Exception as e:
        print(f"[image-gen-flux] Error starting batch processor: {e}")
        import traceback
        traceback.print_exc()
    print("[image-gen-flux] Application startup complete")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": "FLUX.1-schnell"}


@app.get("/stats")
async def get_stats():
    """Get batch processing statistics and performance metrics."""
    return await flux_service.get_stats()


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    Generate an image from a text prompt using FLUX Schnell.

    Requests are automatically batched when using local backend with batching enabled.
    This significantly improves throughput for concurrent requests.

    Note: FLUX Schnell is optimized for speed and typically uses:
    - 4 inference steps (vs 20-50 for other models)
    - guidance_scale of 0.0 (guidance-free generation)
    """
    import asyncio
    try:
        print(f"[generate] Calling generate_async, backend type: {type(flux_service.backend)}")
        print(f"[generate] is_async flag: {flux_service.is_async}")

        result = flux_service.generate_async(req)
        print(f"[generate] Result type after call: {type(result)}")

        # Ensure we await if it's a coroutine
        if asyncio.iscoroutine(result):
            print(f"[generate] Awaiting coroutine...")
            result = await result
            print(f"[generate] Result type after await: {type(result)}")

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
