"""Triton Python backend for the synthetic phantom detector.

The score matches hafactory.detector.probability. This file does not import
the application package, so the Triton image can load it on its own.
"""

from __future__ import annotations

import numpy as np

PIXEL_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.05


def probability(image_chw: np.ndarray) -> float:
    if image_chw.shape != (1, 64, 64):
        raise ValueError("INPUT must be shaped (1, 64, 64)")
    return float((image_chw[0] > PIXEL_THRESHOLD).mean())


class TritonPythonModel:
    def execute(self, requests):
        import triton_python_backend_utils as pb_utils

        responses = []
        for request in requests:
            tensor = pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy()
            if tensor.ndim == 3:
                tensor = tensor[None, ...]
            scores = np.asarray([[probability(item)] for item in tensor], dtype=np.float32)
            output = pb_utils.Tensor("PROBABILITY", scores)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
        return responses
