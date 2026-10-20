"""Modeled cost per inference.

The GPU-hour anchor matches the companion infrastructure repository's
planning assumption dated 2026-10-19. Throughput is an assumption, not a
measurement from an H100. Local CPU timings must not be substituted here.
"""

from __future__ import annotations

PRICE_AS_OF = "2026-10-19"
GPU_USD_PER_HOUR = 3.50
ASSUMED_INFERENCES_PER_GPU_HOUR = 18_000


def cost_per_inference(
    gpu_usd_per_hour: float = GPU_USD_PER_HOUR,
    inferences_per_gpu_hour: float = ASSUMED_INFERENCES_PER_GPU_HOUR,
) -> dict:
    if gpu_usd_per_hour <= 0 or inferences_per_gpu_hour <= 0:
        raise ValueError("rates must be positive")
    cost = gpu_usd_per_hour / inferences_per_gpu_hour
    return {
        "evidence_class": "ASSUMPTION",
        "measured_gpu_results": 0,
        "price_as_of": PRICE_AS_OF,
        "gpu_usd_per_hour": gpu_usd_per_hour,
        "assumed_inferences_per_gpu_hour": inferences_per_gpu_hour,
        "usd_per_inference": cost,
        "gpu_utilization": None,
        "note": (
            "Utilization is unmeasured. Do not multiply this cost by a guessed "
            "utilization and present it as observed GPU economics."
        ),
    }


def render_dashboard(model: dict | None = None) -> str:
    model = model or cost_per_inference()
    micros = model["usd_per_inference"] * 1_000_000
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Modeled GPU inference cost</title>
  <style>
    body {{ margin: 0; background: #f6f1e7; color: #1b3a4b; font: 18px/1.45 Georgia, serif; }}
    main {{ max-width: 760px; margin: 0 auto; padding: 32px 20px 48px; }}
    .banner {{ background: #1b3a4b; color: #f6f1e7; padding: 12px 20px; font: 600 13px/1.4 "Segoe UI", sans-serif; letter-spacing: 0.04em; text-transform: uppercase; }}
    h1 {{ font-weight: 500; font-size: 32px; }}
    .figure {{ font: 600 40px/1.1 "Segoe UI", sans-serif; margin: 0; }}
    p {{ margin: 0 0 12px; }}
  </style>
</head>
<body>
  <p class="banner">Modeled — not a GPU measurement. Anchor dated {model["price_as_of"]}.</p>
  <main>
    <h1>Cost per inference</h1>
    <p class="figure">${model["usd_per_inference"]:.6f}</p>
    <p>{micros:.2f} microdollars at ${model["gpu_usd_per_hour"]:.2f} per GPU-hour and {model["assumed_inferences_per_gpu_hour"]:,.0f} assumed inferences per GPU-hour.</p>
    <p>GPU utilization published with this page: none. {model["note"]}</p>
    <p>Measured GPU results in this file: {model["measured_gpu_results"]}.</p>
  </main>
</body>
</html>
"""
