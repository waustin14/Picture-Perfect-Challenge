#!/usr/bin/env python3
"""
Load testing script for FLUX Schnell image generation batching.

This script simulates multiple concurrent users submitting image generation
requests to validate batching performance and throughput improvements.

FLUX Schnell is optimized for speed with:
- 4 inference steps (vs 20-50 for other models)
- guidance_scale of 0.0 (guidance-free generation)

Usage:
    python test_batching.py --users 10 --requests-per-user 2
    python test_batching.py --burst 50  # Simulate 50 simultaneous requests
"""

import asyncio
import httpx
import time
import statistics
from typing import List, Dict
import argparse
from datetime import datetime


BASE_URL = "http://localhost:8081"


async def generate_image(client: httpx.AsyncClient, user_id: int, request_num: int) -> Dict:
    """Send a single image generation request."""
    prompt = f"A beautiful landscape with mountains and sunset, user {user_id} request {request_num}"

    payload = {
        "prompt": prompt,
        "game_id": "test-game",
        "round_id": "test-round",
        "player_id": f"player-{user_id}",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 4,  # FLUX Schnell default
        "guidance_scale": 0.0  # FLUX Schnell uses guidance-free generation
    }

    start = time.time()
    try:
        response = await client.post(f"{BASE_URL}/generate", json=payload, timeout=300.0)
        response.raise_for_status()
        duration = time.time() - start

        data = response.json()
        return {
            "user_id": user_id,
            "request_num": request_num,
            "success": True,
            "duration_ms": duration * 1000,
            "generation_ms": data.get("duration_ms", 0),
            "image_url": data.get("image_url"),
            "error": None
        }
    except Exception as e:
        duration = time.time() - start
        return {
            "user_id": user_id,
            "request_num": request_num,
            "success": False,
            "duration_ms": duration * 1000,
            "error": str(e)
        }


async def get_stats(client: httpx.AsyncClient) -> Dict:
    """Fetch current batch processing statistics."""
    try:
        response = await client.get(f"{BASE_URL}/stats")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def print_stats(stats: Dict, prefix: str = ""):
    """Print batch processing statistics, handling both single and multi-GPU formats."""
    if "error" in stats:
        print(f"{prefix}Error fetching stats: {stats['error']}")
        return

    print(f"{prefix}Batch Processing Statistics:")
    print(f"{prefix}  Total batches: {stats.get('total_batches', 'N/A')}")
    print(f"{prefix}  Total requests: {stats.get('total_requests', 'N/A')}")
    print(f"{prefix}  Avg batch size: {stats.get('avg_batch_size', 'N/A')}")

    avg_wait = stats.get('avg_wait_time_ms', 0)
    if isinstance(avg_wait, (int, float)):
        print(f"{prefix}  Avg wait time: {avg_wait:.0f}ms")
    else:
        print(f"{prefix}  Avg wait time: {avg_wait}")

    avg_gen = stats.get('avg_generation_time_ms', 0)
    if isinstance(avg_gen, (int, float)):
        print(f"{prefix}  Avg generation time: {avg_gen:.0f}ms")
    else:
        print(f"{prefix}  Avg generation time: {avg_gen}")

    print(f"{prefix}  Max queue depth: {stats.get('max_queue_depth', 'N/A')}")

    # Multi-GPU specific stats
    if "num_gpus" in stats:
        print(f"\n{prefix}Multi-GPU Configuration:")
        print(f"{prefix}  Number of GPUs: {stats.get('num_gpus', 'N/A')}")
        print(f"{prefix}  GPU IDs: {stats.get('gpu_ids', 'N/A')}")
        print(f"{prefix}  Load balance strategy: {stats.get('load_balance_strategy', 'N/A')}")

        per_gpu_stats = stats.get('per_gpu_stats', [])
        if per_gpu_stats:
            print(f"\n{prefix}Per-GPU Statistics:")
            for gpu_stat in per_gpu_stats:
                gpu_id = gpu_stat.get('gpu_id', '?')
                print(f"{prefix}  GPU {gpu_id}:")
                print(f"{prefix}    Queue depth: {gpu_stat.get('current_queue_depth', 'N/A')}")
                print(f"{prefix}    Requests: {gpu_stat.get('total_requests', 'N/A')}")
                print(f"{prefix}    Batches: {gpu_stat.get('total_batches', 'N/A')}")
                print(f"{prefix}    Avg batch size: {gpu_stat.get('avg_batch_size', 'N/A')}")

    config = stats.get('config', {})
    if config:
        print(f"\n{prefix}Batch Configuration:")
        print(f"{prefix}  Max batch size: {config.get('max_batch_size', 'N/A')}")
        print(f"{prefix}  Min batch size: {config.get('min_batch_size', 'N/A')}")
        print(f"{prefix}  Wait time: {config.get('min_wait_time', 'N/A')}s - {config.get('max_wait_time', 'N/A')}s")
        print(f"{prefix}  Adaptive batching: {config.get('adaptive_batching', 'N/A')}")
        if config.get('adaptive_batching'):
            print(f"{prefix}  Adaptive threshold: {config.get('adaptive_threshold', 'N/A')}")


async def run_burst_test(num_requests: int):
    """Simulate a burst of simultaneous requests."""
    print(f"\n{'='*60}")
    print(f"FLUX SCHNELL BURST TEST: {num_requests} simultaneous requests")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient() as client:
        # Get initial stats
        print("Fetching initial stats...")
        initial_stats = await get_stats(client)
        print(f"Initial queue depth: {initial_stats.get('current_queue_depth', 'N/A')}")
        print(f"Total requests processed: {initial_stats.get('total_requests', 'N/A')}\n")

        # Launch all requests simultaneously
        print(f"Launching {num_requests} requests at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}...")
        start_time = time.time()

        tasks = [
            generate_image(client, user_id=i, request_num=0)
            for i in range(num_requests)
        ]

        results = await asyncio.gather(*tasks)

        total_duration = time.time() - start_time

        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}\n")

        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Successful: {len(successful)}/{num_requests}")
        print(f"Failed: {len(failed)}/{num_requests}\n")

        if successful:
            durations = [r["duration_ms"] for r in successful]
            print(f"Request Duration (total wait + generation):")
            print(f"  Min: {min(durations):.0f}ms")
            print(f"  Max: {max(durations):.0f}ms")
            print(f"  Mean: {statistics.mean(durations):.0f}ms")
            print(f"  Median: {statistics.median(durations):.0f}ms\n")

            gen_times = [r["generation_ms"] for r in successful]
            print(f"Generation Time (backend processing):")
            print(f"  Min: {min(gen_times):.0f}ms")
            print(f"  Max: {max(gen_times):.0f}ms")
            print(f"  Mean: {statistics.mean(gen_times):.0f}ms\n")

        if failed:
            print(f"Errors:")
            for r in failed[:5]:  # Show first 5 errors
                print(f"  User {r['user_id']}: {r['error']}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more errors\n")

        # Get final stats
        print("\nFetching final stats...")
        final_stats = await get_stats(client)
        print_stats(final_stats, prefix="")

        # Calculate throughput
        if total_duration > 0 and len(successful) > 0:
            throughput = len(successful) / total_duration
            print(f"\nThroughput: {throughput:.2f} images/second")
            print(f"Effective time per image: {total_duration / len(successful):.2f}s")


async def run_sustained_test(num_users: int, requests_per_user: int):
    """Simulate sustained load with multiple users making multiple requests."""
    print(f"\n{'='*60}")
    print(f"FLUX SCHNELL SUSTAINED TEST: {num_users} users x {requests_per_user} requests")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient() as client:
        print("Fetching initial stats...")
        initial_stats = await get_stats(client)
        print(f"Initial total requests: {initial_stats.get('total_requests', 'N/A')}\n")

        start_time = time.time()

        # Create tasks for all users and requests
        tasks = [
            generate_image(client, user_id=user, request_num=req)
            for user in range(num_users)
            for req in range(requests_per_user)
        ]

        total_requests = len(tasks)
        print(f"Launching {total_requests} total requests...")

        results = await asyncio.gather(*tasks)

        total_duration = time.time() - start_time

        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}\n")

        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Successful: {len(successful)}/{total_requests}")
        print(f"Failed: {len(failed)}/{total_requests}\n")

        if successful:
            durations = [r["duration_ms"] for r in successful]
            print(f"Request Duration:")
            print(f"  Min: {min(durations):.0f}ms ({min(durations)/1000:.1f}s)")
            print(f"  Max: {max(durations):.0f}ms ({max(durations)/1000:.1f}s)")
            print(f"  Mean: {statistics.mean(durations):.0f}ms ({statistics.mean(durations)/1000:.1f}s)")
            print(f"  Median: {statistics.median(durations):.0f}ms ({statistics.median(durations)/1000:.1f}s)\n")

            gen_times = [r["generation_ms"] for r in successful]
            print(f"Generation Time:")
            print(f"  Min: {min(gen_times):.0f}ms")
            print(f"  Max: {max(gen_times):.0f}ms")
            print(f"  Mean: {statistics.mean(gen_times):.0f}ms\n")

        if failed:
            print(f"Errors:")
            for r in failed[:5]:
                print(f"  User {r['user_id']}: {r['error']}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more errors\n")

        # Get final stats
        final_stats = await get_stats(client)
        print_stats(final_stats, prefix="")

        if total_duration > 0 and len(successful) > 0:
            throughput = len(successful) / total_duration
            print(f"\nOverall Throughput: {throughput:.2f} images/second")


async def run_health_check():
    """Check if the service is healthy."""
    print("Checking service health...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/healthz", timeout=10.0)
            response.raise_for_status()
            data = response.json()
            print(f"Service is healthy: {data}")
            return True
        except Exception as e:
            print(f"Service health check failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Load test FLUX Schnell image generation batching")
    parser.add_argument("--burst", type=int, help="Number of simultaneous requests in burst test")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--requests-per-user", type=int, default=2, help="Requests per user")
    parser.add_argument("--url", type=str, default="http://localhost:8081", help="Base URL of image-gen-flux-schnell service")
    parser.add_argument("--health", action="store_true", help="Just run a health check")
    parser.add_argument("--stats", action="store_true", help="Just fetch and display current stats")

    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url

    if args.health:
        asyncio.run(run_health_check())
    elif args.stats:
        async def show_stats():
            async with httpx.AsyncClient() as client:
                stats = await get_stats(client)
                print_stats(stats)
        asyncio.run(show_stats())
    elif args.burst:
        asyncio.run(run_burst_test(args.burst))
    else:
        asyncio.run(run_sustained_test(args.users, args.requests_per_user))


if __name__ == "__main__":
    main()
