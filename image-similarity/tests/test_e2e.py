"""
End-to-end performance tests.

Run these against a running server to verify performance requirements.
"""
import argparse
import base64
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import requests


def create_test_image(width: int = 1024, height: int = 1024, seed: int = 0) -> Image.Image:
    """Create a test image with some variation."""
    import random
    random.seed(seed)
    
    # Create image with random color regions
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    
    # Fill with gradient + noise for realistic test
    for y in range(height):
        for x in range(width):
            r = int((x / width) * 255) % 256
            g = int((y / height) * 255) % 256
            b = (seed * 37) % 256
            pixels[x, y] = (r, g, b)
    
    return img


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_similarity_test(
    base_url: str,
    num_pairs: int,
    image_size: int = 1024,
) -> dict:
    """
    Run a similarity test with the specified number of pairs.
    
    Returns dict with timing and results info.
    """
    print(f"Creating {num_pairs} image pairs ({image_size}x{image_size})...")
    
    # Generate image pairs
    pairs = []
    for i in range(num_pairs):
        img1 = create_test_image(image_size, image_size, seed=i)
        img2 = create_test_image(image_size, image_size, seed=i + 1000)
        pairs.append({
            "reference_image": image_to_base64(img1),
            "generated_image": image_to_base64(img2),
            "pair_id": f"test_{i}",
        })
    
    print(f"Sending request to {base_url}/similarity...")
    
    start_time = time.time()
    response = requests.post(
        f"{base_url}/similarity",
        json={"pairs": pairs},
        timeout=60,
    )
    total_time = time.time() - start_time
    
    result = {
        "num_pairs": num_pairs,
        "image_size": image_size,
        "status_code": response.status_code,
        "total_time_seconds": total_time,
    }
    
    if response.status_code == 200:
        data = response.json()
        result["processing_time_ms"] = data.get("processing_time_ms")
        result["num_scores"] = len(data.get("scores", []))
        
        # Basic stats on distances
        distances = [s["distance"] for s in data["scores"]]
        similarities = [s["similarity"] for s in data["scores"]]
        
        result["distance_min"] = min(distances)
        result["distance_max"] = max(distances)
        result["distance_mean"] = sum(distances) / len(distances)
        result["similarity_min"] = min(similarities)
        result["similarity_max"] = max(similarities)
        result["similarity_mean"] = sum(similarities) / len(similarities)
    else:
        result["error"] = response.text
    
    return result


def run_benchmark(base_url: str, sizes: list = None):
    """Run benchmark with different batch sizes."""
    if sizes is None:
        sizes = [1, 5, 10, 30, 50, 100]
    
    print("=" * 60)
    print("Image Similarity Service Benchmark")
    print("=" * 60)
    
    # Check health
    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health check: {health.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    print()
    
    results = []
    for size in sizes:
        print(f"\n--- Testing {size} pairs ---")
        try:
            result = run_similarity_test(base_url, size)
            results.append(result)
            
            if result["status_code"] == 200:
                print(f"  Status: OK")
                print(f"  Total time: {result['total_time_seconds']:.2f}s")
                print(f"  Processing time: {result['processing_time_ms']:.1f}ms")
                print(f"  Per-pair: {result['total_time_seconds']/size*1000:.1f}ms")
                print(f"  Distance range: [{result['distance_min']:.4f}, {result['distance_max']:.4f}]")
                print(f"  Similarity range: [{result['similarity_min']:.4f}, {result['similarity_max']:.4f}]")
            else:
                print(f"  Status: FAILED ({result['status_code']})")
                print(f"  Error: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"  Exception: {e}")
            results.append({"num_pairs": size, "error": str(e)})
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    # Check if 30 pairs meets requirements
    result_30 = next((r for r in results if r["num_pairs"] == 30), None)
    if result_30 and result_30.get("status_code") == 200:
        time_30 = result_30["total_time_seconds"]
        if time_30 <= 10:
            print(f"✓ 30 pairs in {time_30:.2f}s (TARGET: ≤10s)")
        elif time_30 <= 30:
            print(f"⚠ 30 pairs in {time_30:.2f}s (HARD LIMIT: ≤30s, TARGET: ≤10s)")
        else:
            print(f"✗ 30 pairs in {time_30:.2f}s (EXCEEDS HARD LIMIT: 30s)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark image similarity service")
    parser.add_argument(
        "--url", "-u",
        default="http://localhost:8080",
        help="Base URL of the service (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--pairs", "-p",
        type=int,
        default=30,
        help="Number of pairs to test (default: 30)"
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run full benchmark with multiple sizes"
    )
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(args.url)
    else:
        result = run_similarity_test(args.url, args.pairs)
        print(json.dumps(result, indent=2))
        
        if result.get("status_code") != 200:
            sys.exit(1)


if __name__ == "__main__":
    main()
