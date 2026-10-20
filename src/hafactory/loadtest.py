"""In-process timings for the local detector.

The numbers describe this CPU process. They are an EXAMPLE of a load-test
record. They are not Triton latency, GPU throughput, or GPU utilization,
and they must not be copied into benchmarks/gpu-evidence.json.
"""

from __future__ import annotations

import time

from hafactory.detector import detect
from hafactory.phantom import phantom_uint8
from hafactory.preprocess import preprocess


def run_local_load(iterations: int = 8) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    image = preprocess(phantom_uint8())
    samples: list[float] = []
    label = ""
    for _ in range(iterations):
        started = time.perf_counter()
        result = detect(image)
        samples.append(time.perf_counter() - started)
        label = result["label"]
    samples.sort()
    midpoint = samples[(len(samples) - 1) // 2]
    return {
        "evidence_class": "EXAMPLE",
        "runtime": "local-cpu",
        "triton_executed": False,
        "gpu_utilization": None,
        "iterations": iterations,
        "label": label,
        "latency_seconds_p50": midpoint,
        "latency_seconds_max": samples[-1],
        "note": (
            "Local CPU timings. Do not file these numbers as triton-throughput, "
            "triton-latency, or gpu-utilization."
        ),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_local_load(), indent=2))
