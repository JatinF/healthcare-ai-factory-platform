import unittest

from hafactory.loadtest import run_local_load


class LoadTests(unittest.TestCase):
    def test_local_load_stays_an_example(self) -> None:
        report = run_local_load(iterations=4)
        self.assertEqual(report["evidence_class"], "EXAMPLE")
        self.assertEqual(report["runtime"], "local-cpu")
        self.assertFalse(report["triton_executed"])
        self.assertIsNone(report["gpu_utilization"])
        self.assertEqual(report["label"], "phantom-present")
        self.assertGreaterEqual(report["latency_seconds_p50"], 0.0)
        self.assertGreaterEqual(report["latency_seconds_max"], report["latency_seconds_p50"])
        self.assertIn("Local CPU timings", report["note"])


if __name__ == "__main__":
    unittest.main()
