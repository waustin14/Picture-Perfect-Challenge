#!/usr/bin/env python3
"""
Performance benchmark script for the image similarity service.

Usage:
    python scripts/benchmark.py                    # Default: test 30 pairs
    python scripts/benchmark.py --pairs 100        # Test 100 pairs
    python scripts/benchmark.py --benchmark        # Full benchmark suite
    python scripts/benchmark.py --url http://host:port  # Custom URL
"""
import sys
import os

# Add parent directory to path so we can import from tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_e2e import main

if __name__ == "__main__":
    main()
