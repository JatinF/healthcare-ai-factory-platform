"""Write a synthetic secondary-capture DICOM slice.

Study identifiers look like SYN-0001. The file has no accession number,
birth date, or institution. Pixel data is the geometric phantom.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from hafactory.phantom import phantom_uint8


def write_synthetic_slice(
    path: Path,
    study_id: str = "SYN-0001",
    seed: int = 1,
    poison: dict[str, str] | None = None,
) -> Path:
    if not study_id.startswith("SYN-"):
        raise ValueError("Synthetic study ids start with SYN-")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = phantom_uint8(seed=seed)
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.PatientName = "SYNTHETIC"
    dataset.PatientID = study_id
    dataset.StudyID = study_id
    dataset.StudyDescription = "Synthetic geometric phantom"
    dataset.Modality = "OT"
    dataset.SeriesNumber = 1
    dataset.InstanceNumber = 1
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows = int(pixels.shape[0])
    dataset.Columns = int(pixels.shape[1])
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.ascontiguousarray(pixels).tobytes()
    for keyword, value in (poison or {}).items():
        setattr(dataset, keyword, value)
    dataset.save_as(path, implicit_vr=False, little_endian=True, enforce_file_format=True)
    return path
