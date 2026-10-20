import tempfile
import unittest
from pathlib import Path

from pydicom import dcmread

from hafactory.deid import DeidError, assert_deidentified, deid_problems
from hafactory.dicom_synth import write_synthetic_slice
from hafactory.pipeline import run_slice
from hafactory.preprocess import preprocess
from hafactory.phantom import phantom_uint8


class PipelineTests(unittest.TestCase):
    def test_synthetic_slice_passes_and_detects_the_sphere(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_synthetic_slice(Path(tmp) / "slice.dcm")
            dataset = dcmread(path)
            self.assertEqual(deid_problems(dataset), [])
            result = run_slice(path)
        self.assertEqual(result["label"], "phantom-present")
        self.assertEqual(result["study_id"], "SYN-0001")
        self.assertEqual(result["patient_records"], 0)
        self.assertEqual(result["runtime"], "local-cpu")
        self.assertGreater(result["probability"], 0.05)

    def test_accession_number_fail_closes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_synthetic_slice(
                Path(tmp) / "poison.dcm",
                poison={"AccessionNumber": "ACC100", "PatientBirthDate": "19700101"},
            )
            with self.assertRaises(DeidError) as caught:
                assert_deidentified(dcmread(path))
        self.assertTrue(any("AccessionNumber" in item for item in caught.exception.problems))
        self.assertTrue(any("PatientBirthDate" in item for item in caught.exception.problems))

    def test_real_looking_patient_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_synthetic_slice(Path(tmp) / "slice.dcm", poison={"PatientID": "MRN-100"})
            problems = deid_problems(dcmread(path))
        self.assertIn("PatientID must start with SYN-", problems)

    def test_preprocess_shape_uses_monai(self) -> None:
        image = phantom_uint8()
        output = preprocess(image)
        self.assertEqual(output.shape, (1, 64, 64))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
