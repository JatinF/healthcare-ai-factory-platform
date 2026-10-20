"""Geometric phantom used by the synthetic DICOM writer and the detector.

The picture is a bright sphere on a dark background. It is not anatomy.
"""

from __future__ import annotations

import numpy as np

HEIGHT = 64
WIDTH = 64


def phantom_uint8(height: int = HEIGHT, width: int = WIDTH, seed: int = 1) -> np.ndarray:
    if min(height, width) < 8:
        raise ValueError("phantom axes must be at least 8")
    cy = (height - 1) / 2
    cx = (width - 1) / 2
    radius = min(height, width) * 0.28
    yy, xx = np.ogrid[:height, :width]
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    background = (np.arange(height)[:, None] * 3 + np.arange(width)[None, :] * 5 + seed) % 40
    image = np.where(inside, 180, background).astype(np.uint8)
    return image
