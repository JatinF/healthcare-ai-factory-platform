"""De-identification gate for synthetic DICOM.

The gate fail-closes. A dataset is accepted only when every identifier field
is absent or uses the synthetic vocabulary. Real-looking values are rejected.
This is a repository control, not a certification that a file is de-identified
under HIPAA.
"""

from __future__ import annotations

from pydicom.dataset import Dataset

SYNTHETIC_NAME = "SYNTHETIC"
STUDY_PREFIX = "SYN-"

# Tags that must not carry a real identifier. Keyword names match pydicom.
FORBIDDEN_IF_PRESENT = (
    "AccessionNumber",
    "PatientBirthDate",
    "PatientSex",
    "PatientAddress",
    "OtherPatientIDs",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "MedicalRecordNumber",
)


class DeidError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def _text(dataset: Dataset, keyword: str) -> str:
    value = dataset.get(keyword, None)
    if value is None:
        return ""
    return str(value).strip()


def deid_problems(dataset: Dataset) -> list[str]:
    problems: list[str] = []
    name = _text(dataset, "PatientName")
    if name and name != SYNTHETIC_NAME:
        problems.append("PatientName is not the synthetic marker")
    patient_id = _text(dataset, "PatientID")
    if patient_id and not patient_id.startswith(STUDY_PREFIX):
        problems.append("PatientID must start with SYN-")
    study_id = _text(dataset, "StudyID")
    if study_id and not study_id.startswith(STUDY_PREFIX):
        problems.append("StudyID must start with SYN-")
    for keyword in FORBIDDEN_IF_PRESENT:
        if _text(dataset, keyword):
            problems.append(f"{keyword} must be empty")
    if "PixelData" not in dataset:
        problems.append("PixelData is missing")
    return problems


def assert_deidentified(dataset: Dataset) -> None:
    problems = deid_problems(dataset)
    if problems:
        raise DeidError(problems)
