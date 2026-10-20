"""MONAI preprocessing for one synthetic slice.

The output is one channel, 64 by 64, intensities scaled to 0..1.
MONAI is required. A missing install fails closed rather than silently
substituting a different transform stack.
"""

from __future__ import annotations

import numpy as np

SPATIAL_SIZE = (64, 64)


def preprocess(image: np.ndarray) -> np.ndarray:
    try:
        from monai.transforms import Compose, EnsureChannelFirst, Resize, ScaleIntensity
    except ImportError as exc:
        raise RuntimeError(
            "MONAI is required for preprocessing. Refusing to run an untracked transform path."
        ) from exc

    if image.ndim != 2:
        raise ValueError("preprocess expects one HxW slice")
    transform = Compose(
        [
            EnsureChannelFirst(channel_dim="no_channel"),
            ScaleIntensity(minv=0.0, maxv=1.0),
            Resize(spatial_size=SPATIAL_SIZE),
        ]
    )
    output = transform(np.ascontiguousarray(image, dtype=np.float32))
    array = np.asarray(output, dtype=np.float32)
    if array.shape != (1, 64, 64):
        raise RuntimeError(f"Unexpected preprocess shape {array.shape}")
    return array
