"""Measure warmed serving latency and throughput without mistaking timings for relevance."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np


async def benchmark(url: str, queries: list[str], mode: str, requests: int, concurrency: int) -> dict:
    if not queries or any(not query.strip() for query in queries) or requests < 1 or concurrency < 1:
        raise ValueError("Supply nonempty queries and positive requests/concurrency")
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        for query in queries:
            response = await client.post(url + "/search", json={"query": query, "mode": mode})
            response.raise_for_status()

        async def one(number: int) -> float:
            async with semaphore:
                start = time.perf_counter()
                response = await client.post(url + "/search", json={
                    "query": queries[number % len(queries)], "mode": mode,
                })
                response.raise_for_status()
                return (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        timings = await asyncio.gather(*(one(number) for number in range(requests)))
        seconds = time.perf_counter() - start
    return {"mode": mode, "requests": requests, "concurrency": concurrency,
            "p50_ms": float(np.percentile(timings, 50)), "p95_ms": float(np.percentile(timings, 95)),
            "requests_per_second": requests / seconds, "cache": "warm repeated query set"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--queries", type=Path, required=True, help="One demo query per line; not research labels")
    parser.add_argument("--mode", choices=("bm25", "vector", "hybrid", "rerank"), default="hybrid")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    queries = [line.strip() for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not queries or args.requests < 1 or args.concurrency < 1:
        parser.error("Supply nonempty queries and positive requests/concurrency")
    print(json.dumps(asyncio.run(benchmark(args.url.rstrip("/"), queries, args.mode,
                                          args.requests, args.concurrency)), indent=2))
