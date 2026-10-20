# Service objectives

Two kinds of objective live here. Software objectives are checks this repository executes on every test run. GPU objectives are gates. A gate has no numeric result until `benchmarks/gpu-evidence.json` carries a `MEASURED` pass with serials and a log. A CI run on a GitHub-hosted CPU runner is not an H100 service objective.

## Software objectives exercised by pytest

| Objective | Pass condition in this repo |
| --- | --- |
| Synthetic slice | Clean `SYN-0001` returns `phantom-present`, `patient_records` 0, runtime `local-cpu` |
| De-identification | Accession number, birth date, institution name, or a non-`SYN-` patient id fails closed |
| Authentication | Missing bearer returns 401 and an audit line with no pixel data |
| Tenant quota | The third call at quota 2 returns 429 |
| Tenant isolation | An approval in tenant-a does not authorize tenant-b |
| Human approval | A reader cannot grant. Export without a matching approval returns 403 |
| Groundedness | An invented sentence raises `GroundednessError`. An unsupported question returns `refused` |
| High-risk assistant | Prescribe or diagnosis wording returns an empty answer and `approval-required` |
| Question privacy | The audit log for `/v1/ask` does not contain the question text |
| Cost honesty | The cost model stays `ASSUMPTION`, utilization `None`, measured GPU results 0 |
| Release honesty | The committed acceptance report matches a fresh render and the decision is NO-GO |
| Manifest hygiene | Kubernetes YAML parses, and the files contain no private key and no password field |

These checks protect the repository. They do not measure a cluster.

## GPU gates

| Gate id | What must be true before production serving | Status in this repo |
| --- | --- | --- |
| `triton-latency` | Latency recorded against Triton on the GPU under test | not-run |
| `triton-throughput` | Throughput recorded against that endpoint | not-run |
| `gpu-utilization` | Utilization sampled during that run | not-run |
| `tensorrt-or-onnx-parity` | The served model matches a TensorRT or ONNX Runtime build | not-run |

The numeric threshold for latency and throughput is chosen in the acceptance window and written into the evidence log. This repository does not publish a p95 millisecond target as if it had been measured. Local CPU timings from `hafactory.loadtest` are an `EXAMPLE`. They do not fill these rows, and they do not feed the cost page.

## Alerting objectives

| Alert | Expression intent | Severity | Runbook |
| --- | --- | --- | --- |
| `InferenceApiDown` | `up{job="healthcare-inference-api"} == 0` for 5 minutes | page | `docs/runbooks/inference-down.md` |
| `InferenceBackendErrors` | API up and no inference traffic for 30 minutes | ticket | `docs/runbooks/inference-down.md` |

No sample series is committed. The rules are not evaluated here. The metrics the API exposes are process counters. Their text says they are not GPU telemetry.

## Error budget while the decision is NO-GO

There is no production error budget. The service is not accepted for production traffic. Scaling the Triton Deployment, enabling NIM, or copying the modeled microdollar figure into a capacity plan does not create one. The budget starts after every GPU gate is a measured pass on the chassis that will run the service, and after the site identity provider has replaced the dev token map.
