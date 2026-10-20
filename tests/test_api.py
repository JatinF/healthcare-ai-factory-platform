import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from hafactory.api import create_app
from hafactory.dicom_synth import write_synthetic_slice


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.audit = Path(self.tmp.name) / "audit.jsonl"
        self.client = TestClient(create_app(self.audit, quota=2))
        self.slice = write_synthetic_slice(Path(self.tmp.name) / "slice.dcm")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _auth(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def test_missing_token_is_unauthorized_and_audited(self) -> None:
        response = self.client.post(
            "/v1/inference",
            files={"dicom": ("slice.dcm", self.slice.read_bytes(), "application/dicom")},
        )
        self.assertEqual(response.status_code, 401)
        text = self.audit.read_text(encoding="utf-8")
        self.assertIn("auth", text)
        self.assertNotIn("PixelData", text)

    def test_inference_and_tenant_quota(self) -> None:
        headers = self._auth("dev-reader-a")
        files = {"dicom": ("slice.dcm", self.slice.read_bytes(), "application/dicom")}
        first = self.client.post("/v1/inference", files=files, headers=headers)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["label"], "phantom-present")
        self.assertEqual(first.json()["tenant"], "tenant-a")
        second_files = {"dicom": ("slice.dcm", self.slice.read_bytes(), "application/dicom")}
        self.assertEqual(self.client.post("/v1/inference", files=second_files, headers=headers).status_code, 200)
        third_files = {"dicom": ("slice.dcm", self.slice.read_bytes(), "application/dicom")}
        self.assertEqual(self.client.post("/v1/inference", files=third_files, headers=headers).status_code, 429)
        metrics = self.client.get("/metrics").text
        self.assertIn("hafactory_inference_requests_total 2", metrics)
        self.assertIn("Not GPU telemetry", metrics)

    def test_poisoned_dicom_is_rejected(self) -> None:
        poisoned = write_synthetic_slice(
            Path(self.tmp.name) / "bad.dcm",
            poison={"InstitutionName": "Example Hospital"},
        )
        response = self.client.post(
            "/v1/inference",
            files={"dicom": ("bad.dcm", poisoned.read_bytes(), "application/dicom")},
            headers=self._auth("dev-reader-a"),
        )
        self.assertEqual(response.status_code, 422)

    def test_export_requires_same_tenant_approver(self) -> None:
        denied = self.client.post(
            "/v1/actions/export",
            json={"study_id": "SYN-0001", "approval_id": "APR-1"},
            headers=self._auth("dev-reader-a"),
        )
        self.assertEqual(denied.status_code, 403)
        reader_cannot_approve = self.client.post(
            "/v1/approvals",
            json={"study_id": "SYN-0001", "approval_id": "APR-1"},
            headers=self._auth("dev-reader-a"),
        )
        self.assertEqual(reader_cannot_approve.status_code, 403)
        granted = self.client.post(
            "/v1/approvals",
            json={"study_id": "SYN-0001", "approval_id": "APR-1"},
            headers=self._auth("dev-approver-a"),
        )
        self.assertEqual(granted.status_code, 200)
        other_tenant = self.client.post(
            "/v1/actions/export",
            json={"study_id": "SYN-0001", "approval_id": "APR-1"},
            headers=self._auth("dev-reader-b"),
        )
        self.assertEqual(other_tenant.status_code, 403)
        allowed = self.client.post(
            "/v1/actions/export",
            json={"study_id": "SYN-0001", "approval_id": "APR-1"},
            headers=self._auth("dev-reader-a"),
        )
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.json()["destination"], "site-record-system-placeholder")

    def test_triton_outage_fail_closes(self) -> None:
        client = TestClient(create_app(self.audit, force_triton_down=True))
        response = client.post(
            "/v1/inference",
            files={"dicom": ("slice.dcm", self.slice.read_bytes(), "application/dicom")},
            headers=self._auth("dev-reader-a"),
        )
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
