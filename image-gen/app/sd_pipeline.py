import os
import textwrap
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional, Protocol

import torch
from PIL import Image, ImageDraw, ImageFont

from .schemas import GenerateRequest, GenerateResponse
from .storage import storage

# Local SD imports
from diffusers import StableDiffusionXLPipeline

# Cloud backend imports
import httpx


def image_to_bytes(image: Image.Image) -> bytes:
    """Convert a PIL Image to PNG bytes."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ImageGenBackend(Protocol):
    def generate(self, req: GenerateRequest) -> GenerateResponse:
        ...


class LocalSDBackend:
    def __init__(self, output_dir: Path, public_base_url: str):
        self.output_dir = output_dir
        self.public_base_url = public_base_url

        self.model_id = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
        device_env = os.getenv("DEVICE", "auto")  # "auto" | "cuda" | "cpu"

        if device_env == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_env == "cpu":
            device = "cpu"
        else:  # auto
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[image-gen] Using LOCAL backend with model {self.model_id} on {self.device}")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        )
        self.pipe.to(self.device)
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

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
        start = time.time()

        if req.seed is not None:
            seed = req.seed
        else:
            seed = torch.randint(0, 2**31 - 1, (1,)).item()

        if self.device == "cuda":
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)

        with torch.inference_mode():
            image = self.pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                generator=generator,
            ).images[0]

        duration_ms = int((time.time() - start) * 1000)
        return self._save_image(image, req, seed, duration_ms)


class StabilityBackend:
    """
    Uses Stability AI's text-to-image API for SDXL 1.0.
    """

    def __init__(self, output_dir: Path, public_base_url: str, fallback_backend: Optional["MockBackend"] = None):
        self.output_dir = output_dir
        self.public_base_url = public_base_url
        self.fallback_backend = fallback_backend

        self.api_key = os.getenv("STABILITY_API_KEY")
        self.api_host = os.getenv("STABILITY_API_HOST", "https://api.stability.ai")
        self.engine_id = os.getenv("STABILITY_ENGINE_ID", "stable-diffusion-xl-1024-v1-0")
        self.allow_fallback = os.getenv("ALLOW_STABILITY_FALLBACK", "true").lower() == "true"
        self.use_mock_only = False

        if not self.api_key:
            if self.allow_fallback and self.fallback_backend:
                print("[image-gen] STABILITY_API_KEY missing; using mock backend instead.")
                self.use_mock_only = True
            else:
                raise RuntimeError("STABILITY_API_KEY is required for Stability backend")

        print(f"[image-gen] Using STABILITY backend with engine {self.engine_id}")

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        if self.use_mock_only and self.fallback_backend:
            return self.fallback_backend.generate(req)

        start = time.time()

        # Stability expects 1024x1024 for SDXL engine; we can clamp/override here if needed.
        width = req.width
        height = req.height

        seed = req.seed if req.seed is not None else 0  # 0 lets API choose

        # JSON text-to-image endpoint (Stability provides both JSON + binary)
        url = f"{self.api_host}/v1/generation/{self.engine_id}/text-to-image"

        payload = {
            "text_prompts": [
                {"text": req.prompt},
            ],
            "cfg_scale": req.guidance_scale,
            "height": height,
            "width": width,
            "samples": 1,
            "steps": req.num_inference_steps,
        }

        if req.negative_prompt:
            payload["text_prompts"].append({"text": req.negative_prompt, "weight": -1.0})

        if seed is not None and seed != 0:
            payload["seed"] = seed

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            if self.allow_fallback and self.fallback_backend:
                print(f"[image-gen] Stability call failed ({exc}); using mock fallback.")
                return self.fallback_backend.generate(req)
            raise

        # Expect first artifact as base64 image
        artifacts = data.get("artifacts") or []
        if not artifacts:
            raise RuntimeError("No image artifacts returned from Stability API")

        art = artifacts[0]
        b64 = art.get("base64")
        if not b64:
            raise RuntimeError("No base64 image in Stability response")

        import base64
        from io import BytesIO

        img_bytes = base64.b64decode(b64)
        image = Image.open(BytesIO(img_bytes))

        duration_ms = int((time.time() - start) * 1000)

        # Upload to MinIO
        image_id = str(uuid.uuid4())
        image_png_bytes = image_to_bytes(image)
        image_url = storage.upload_image(image_png_bytes, req.game_id, req.round_id, image_id)

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

        # Stability doesn't "have" a local model_id, but we can echo engine_id
        model_id = self.engine_id

        # Stability may override or choose a different seed; if it returns one, prefer that
        # (the JSON schema has a `seed` per artifact in many examples)
        if "seed" in art:
            seed = art["seed"]

        return GenerateResponse(
            image_id=image_id,
            image_url=image_url,
            game_id=req.game_id,
            round_id=req.round_id,
            player_id=req.player_id,
            model_id=model_id,
            seed=seed if isinstance(seed, int) else 0,
            width=width,
            height=height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            duration_ms=duration_ms,
        )


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
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            duration_ms=duration_ms,
        )

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        start = time.time()

        # Build a simple placeholder image
        image = Image.new("RGB", (req.width, req.height), color=(32, 32, 32))
        draw = ImageDraw.Draw(image)
        text = f"Prompt:\n{textwrap.fill(req.prompt, width=40)}"
        font = ImageFont.load_default()
        draw.multiline_text((20, 20), text, fill=(255, 255, 255), font=font, spacing=4)

        duration_ms = int((time.time() - start) * 1000)
        return self._save_image(image, req, duration_ms)


class StableDiffusionService:
    """
    Facade that picks a backend based on GEN_MODE.
    """

    def __init__(self) -> None:
        output_dir = Path(os.getenv("OUTPUT_DIR", "/data/images"))
        output_dir.mkdir(parents=True, exist_ok=True)
        public_base_url = os.getenv("PUBLIC_BASE_URL", "http://image-gen:8080")

        gen_mode = os.getenv("GEN_MODE", "local").lower()
        mock_backend = MockBackend(output_dir, public_base_url)

        if gen_mode == "stability":
            try:
                self.backend: ImageGenBackend = StabilityBackend(
                    output_dir,
                    public_base_url,
                    fallback_backend=mock_backend,
                )
            except Exception as exc:
                print(f"[image-gen] Failed to init stability backend ({exc}); using mock.")
                self.backend = mock_backend
        elif gen_mode == "mock":
            print("[image-gen] Using MOCK backend (placeholder images).")
            self.backend = mock_backend
        else:
            # default to local
            self.backend = LocalSDBackend(output_dir, public_base_url)

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        return self.backend.generate(req)
