#!/usr/bin/env python3
"""
Load testing script for image generation batching.

This script simulates multiple concurrent users submitting image generation
requests to validate batching performance and throughput improvements.

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


BASE_URL = "http://localhost:8080"


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
        "num_inference_steps": 20,
        "guidance_scale": 7.5
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


async def run_burst_test(num_requests: int):
    """Simulate a burst of simultaneous requests."""
    print(f"\n{'='*60}")
    print(f"BURST TEST: {num_requests} simultaneous requests")
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

        print(f"\nBatch Processing Statistics:")
        print(f"  Total batches: {final_stats.get('total_batches', 'N/A')}")
        print(f"  Avg batch size: {final_stats.get('avg_batch_size', 'N/A')}")
        print(f"  Avg wait time: {final_stats.get('avg_wait_time_ms', 'N/A'):.0f}ms")
        print(f"  Avg generation time: {final_stats.get('avg_generation_time_ms', 'N/A'):.0f}ms")
        print(f"  Max queue depth: {final_stats.get('max_queue_depth', 'N/A')}")

        config = final_stats.get('config', {})
        print(f"\nBatch Configuration:")
        print(f"  Max batch size: {config.get('max_batch_size', 'N/A')}")
        print(f"  Wait time: {config.get('min_wait_time', 'N/A')}s - {config.get('max_wait_time', 'N/A')}s")
        print(f"  Adaptive batching: {config.get('adaptive_batching', 'N/A')}")

        # Calculate throughput
        if total_duration > 0:
            throughput = len(successful) / total_duration
            print(f"\nThroughput: {throughput:.2f} images/second")
            print(f"Effective time per image: {total_duration / len(successful):.2f}s")


async def run_sustained_test(num_users: int, requests_per_user: int):
    """Simulate sustained load with multiple users making multiple requests."""
    print(f"\n{'='*60}")
    print(f"SUSTAINED TEST: {num_users} users × {requests_per_user} requests")
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

        # Get final stats
        final_stats = await get_stats(client)

        print(f"Batch Statistics:")
        print(f"  Total batches: {final_stats.get('total_batches', 'N/A')}")
        print(f"  Avg batch size: {final_stats.get('avg_batch_size', 'N/A')}")
        print(f"  Avg wait time: {final_stats.get('avg_wait_time_ms', 'N/A'):.0f}ms")

        if total_duration > 0:
            throughput = len(successful) / total_duration
            print(f"\nOverall Throughput: {throughput:.2f} images/second")


def main():
    parser = argparse.ArgumentParser(description="Load test image generation batching")
    parser.add_argument("--burst", type=int, help="Number of simultaneous requests in burst test")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--requests-per-user", type=int, default=2, help="Requests per user")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="Base URL of image-gen service")

    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url

    if args.burst:
        asyncio.run(run_burst_test(args.burst))
    else:
        asyncio.run(run_sustained_test(args.users, args.requests_per_user))


if __name__ == "__main__":
    main()
