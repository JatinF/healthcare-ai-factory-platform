"""Synthetic phantom detector.

The score is the fraction of pixels above 0.5 after preprocessing. The
synthetic sphere is bright enough to cross the presence threshold. This is a
fixture model for the platform path. It is not a diagnostic model.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "phantom-detector"
MODEL_ROLE = "synthetic geometric detector, not a diagnostic device"
PRESENCE_THRESHOLD = 0.05
PIXEL_THRESHOLD = 0.5


def probability(image_chw: np.ndarray) -> float:
    if image_chw.shape != (1, 64, 64):
        raise ValueError("detector expects shape (1, 64, 64)")
    bright = image_chw[0] > PIXEL_THRESHOLD
    return float(bright.mean())


def detect(image_chw: np.ndarray) -> dict:
    score = probability(image_chw)
    label = "phantom-present" if score >= PRESENCE_THRESHOLD else "no-phantom"
    return {
        "model": MODEL_NAME,
        "model_role": MODEL_ROLE,
        "label": label,
        "probability": score,
        "runtime": "local-cpu",
    }
