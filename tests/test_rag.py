import unittest

from fastapi.testclient import TestClient

from hafactory.rag import GroundednessError, assert_grounded, grounded_answer, load_notes
from hafactory.api import create_app
import tempfile
from pathlib import Path


class RagTests(unittest.TestCase):
    def test_corpus_has_no_patient_records(self) -> None:
        notes = load_notes()
        blob = " ".join(note["text"] for note in notes)
        self.assertNotIn("PatientName", blob)
        self.assertTrue(all(note["id"].startswith("SYN-NOTE-") for note in notes))

    def test_answer_cites_a_note_and_rejects_an_invented_sentence(self) -> None:
        notes = load_notes()
        result = grounded_answer("Which PatientName does the de-identification rule accept?", notes)
        self.assertEqual(result["status"], "answered")
        self.assertTrue(result["citations"])
        with self.assertRaises(GroundednessError):
            assert_grounded("The patient was diagnosed on Tuesday.", {"SYN-NOTE-003": notes[-1]["text"]})

    def test_high_risk_question_requires_approval(self) -> None:
        result = grounded_answer("Please prescribe a treatment and write the diagnosis.", load_notes())
        self.assertEqual(result["status"], "approval-required")
        self.assertEqual(result["answer"], "")

    def test_unknown_question_fail_closes(self) -> None:
        result = grounded_answer("quantum lattice coolant viscosity", load_notes())
        self.assertEqual(result["status"], "refused")

    def test_api_does_not_log_the_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.jsonl"
            client = TestClient(create_app(audit))
            response = client.post(
                "/v1/ask",
                json={"question": "What does the export rule require before export?"},
                headers={"Authorization": "Bearer dev-reader-a"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("SYN-NOTE-002", str(response.json()["citations"]))
            log = audit.read_text(encoding="utf-8")
        self.assertNotIn("export rule", log.lower())


if __name__ == "__main__":
    unittest.main()
