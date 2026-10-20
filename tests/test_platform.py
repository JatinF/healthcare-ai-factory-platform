import importlib.util
import unittest
from pathlib import Path

import yaml

from hafactory.acceptance_report import build_report, render_markdown, software_checks
from hafactory.detector import probability
from hafactory.economics import cost_per_inference, render_dashboard
from hafactory.phantom import phantom_uint8
from hafactory.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[1]


class PlatformTests(unittest.TestCase):
    def test_triton_model_matches_the_library_score(self) -> None:
        model_path = ROOT / "models" / "phantom-detector" / "1" / "model.py"
        spec = importlib.util.spec_from_file_location("triton_phantom", model_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        image = preprocess(phantom_uint8())
        self.assertAlmostEqual(module.probability(image), probability(image))

    def test_cost_model_is_an_assumption(self) -> None:
        model = cost_per_inference()
        self.assertEqual(model["evidence_class"], "ASSUMPTION")
        self.assertEqual(model["measured_gpu_results"], 0)
        self.assertIsNone(model["gpu_utilization"])
        self.assertAlmostEqual(model["usd_per_inference"], 3.50 / 18_000)
        page = (ROOT / "observability" / "gpu-economics.html").read_text(encoding="utf-8")
        self.assertEqual(page, render_dashboard(model))
        self.assertIn("not a GPU measurement", page)

    def test_repo_ships_without_gpu_evidence(self) -> None:
        self.assertFalse((ROOT / "benchmarks" / "gpu-evidence.json").exists())

    def test_published_acceptance_is_no_go(self) -> None:
        self.assertTrue(software_checks())
        report = build_report(True)
        self.assertEqual(report["decision"], "NO-GO")
        text = (ROOT / "acceptance" / "deployment-acceptance.md").read_text(encoding="utf-8")
        self.assertEqual(text, render_markdown(report))

    def test_measured_gpu_evidence_can_flip_the_decision(self) -> None:
        import json
        import tempfile

        records = [
            {
                "test_id": test_id,
                "evidence_class": "MEASURED",
                "result": "pass",
                "serials": ["SN-1"],
                "log": "log.txt",
            }
            for test_id in ("triton-latency", "triton-throughput", "gpu-utilization", "tensorrt-or-onnx-parity")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu-evidence.json"
            path.write_text(json.dumps({"records": records}), encoding="utf-8")
            report = build_report(True, path)
        self.assertEqual(report["decision"], "GO")

    def test_manifests_parse(self) -> None:
        paths = list((ROOT / "deploy" / "kubernetes").glob("*.yaml"))
        paths += list((ROOT / "deploy" / "prometheus").glob("*.yaml"))
        paths.append(ROOT / "deploy" / "helm" / "healthcare-ai-factory" / "values.yaml")
        self.assertGreaterEqual(len(paths), 4)
        for path in paths:
            loaded = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            self.assertTrue(any(document for document in loaded))
            blob = path.read_text(encoding="utf-8")
            self.assertNotIn("BEGIN PRIVATE", blob)
            self.assertNotIn("password:", blob.lower())


if __name__ == "__main__":
    unittest.main()
