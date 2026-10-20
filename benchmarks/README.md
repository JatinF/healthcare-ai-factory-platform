# GPU evidence

`gpu-evidence.json` is the only file that can move a GPU gate from `not-run` to `pass`. It is absent in this repository on purpose. The published decision stays **NO-GO** until that file exists on the chassis under test and every required row is a `MEASURED` pass.

Do not commit a filled copy from a laptop, from the local CPU load test, or from the modeled cost page. `tests/test_platform.py` fails if `benchmarks/gpu-evidence.json` is present, and it fails if the acceptance report says GO while this file is missing.

## Record shape

```json
{
  "records": [
    {
      "test_id": "triton-latency",
      "evidence_class": "MEASURED",
      "result": "pass",
      "serials": ["chassis serial"],
      "log": "path or artifact name of the Triton log"
    }
  ]
}
```

Required `test_id` values:

| test_id | What a pass means |
| --- | --- |
| `triton-latency` | Latency measured against the Triton endpoint on the GPU under test |
| `triton-throughput` | Throughput measured against that same endpoint |
| `gpu-utilization` | Utilization sampled on that GPU during the run |
| `tensorrt-or-onnx-parity` | Numeric parity between the Triton result and a TensorRT or ONNX Runtime build of the same model |

Each passing row needs `evidence_class` = `MEASURED`, `result` = `pass`, a non-empty `serials` list, and a `log`. A row that is present and fails any of those checks is a fail. A missing row is `not-run`. Either one keeps the decision at NO-GO.

`src/hafactory/loadtest.py` writes a different record. Its `evidence_class` is `EXAMPLE` and `triton_executed` is false. Those timings stay out of this directory.
