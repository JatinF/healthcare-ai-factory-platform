"""DICOM slice to a detector result, with the de-identification gate in front."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydicom import dcmread

from hafactory.deid import assert_deidentified
from hafactory.detector import detect
from hafactory.preprocess import preprocess


def pixels_from_dataset(dataset) -> np.ndarray:
    array = dataset.pixel_array
    if array.ndim != 2:
        raise ValueError("This slice pipeline accepts one 2D frame")
    return np.asarray(array)


def run_slice(path: Path) -> dict:
    dataset = dcmread(path)
    assert_deidentified(dataset)
    image = preprocess(pixels_from_dataset(dataset))
    result = detect(image)
    result["study_id"] = str(getattr(dataset, "StudyID", ""))
    result["patient_records"] = 0
    return result
