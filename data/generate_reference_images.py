#!/usr/bin/env python3
"""
Generate reference images from prompts.json using Replicate (SD 3.5 Large Turbo),
with explicit 6 requests/min rate limiting and retries.

prompts.json schema:
[
  {"prompt": "...", "negative_prompt": "..."},
  ...
]

Outputs:
  output/sd3_5_large_turbo_img_X.png  (forced 1024x1024)
  output/prompt_X.txt
  output/negative_prompt_X.txt

Prereqs:
  pip install replicate pillow
  export REPLICATE_API_TOKEN="YOUR_TOKEN"
"""

import io
import json
import sys
import time
from pathlib import Path

import replicate
from PIL import Image


OUTPUT_DIR = Path("output")
PROMPTS_FILE = Path("prompts.json")
MODEL = "stability-ai/stable-diffusion-3.5-large-turbo"

# Rate-limit settings (6 requests per minute)
REQUESTS_PER_MINUTE = 6
MIN_SECONDS_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE  # 10 seconds
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 10


def read_prompt_sets(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("prompts.json must contain a JSON array.")
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict) or "prompt" not in item or "negative_prompt" not in item:
            raise ValueError(f"Item #{idx} must be an object with 'prompt' and 'negative_prompt' fields.")
        if not isinstance(item["prompt"], str) or not isinstance(item["negative_prompt"], str):
            raise ValueError(f"Item #{idx} fields 'prompt' and 'negative_prompt' must be strings.")
    return data


def save_png_1024(img_bytes: bytes, out_path: Path) -> None:
    """Convert output to PNG and force 1024x1024 for consistent gameplay assets."""
    im = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    if im.size != (1024, 1024):
        im = im.resize((1024, 1024), resample=Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="PNG", optimize=True)


def run_with_retries(prompt: str, negative_prompt: str) -> bytes:
    """
    Run inference with retry + backoff for rate limiting/transient errors.
    Note: Replicate exceptions vary; we treat failures as retryable by default.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            output = replicate.run(
                MODEL,
                input={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "aspect_ratio": "1:1",
                },
            )
            return output.read()

        except Exception as e:
            last_error = e
            wait_time = BACKOFF_BASE_SECONDS * attempt
            print(f"⚠️  Attempt {attempt}/{MAX_RETRIES} failed (possible rate limit). Retrying in {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_error}")


def main() -> int:
    try:
        prompt_sets = read_prompt_sets(PROMPTS_FILE)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    last_request_time = 0.0

    total = len(prompt_sets)

    for i, item in enumerate(prompt_sets, start=1):
        prompt = item["prompt"].strip()
        negative_prompt = item["negative_prompt"].strip()

        (OUTPUT_DIR / f"prompt_{i}.txt").write_text(prompt, encoding="utf-8")
        (OUTPUT_DIR / f"negative_prompt_{i}.txt").write_text(negative_prompt, encoding="utf-8")

        # Enforce minimum spacing between requests (6/min)
        elapsed = time.time() - last_request_time
        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)

        img_path = OUTPUT_DIR / f"sd3_5_large_turbo_img_{i}.png"
        print(f"[{i}/{total}] Generating -> {img_path.name}")

        try:
            img_bytes = run_with_retries(prompt, negative_prompt)
            save_png_1024(img_bytes, img_path)
            last_request_time = time.time()

        except Exception as e:
            print(f"❌ ERROR on item {i}: {e}", file=sys.stderr)
            # Continue to next prompt set

    print(f"\n✅ Done. Outputs saved to: {OUTPUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
