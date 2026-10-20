# Enterprise Healthcare AI Factory

Production-oriented healthcare AI factory reference implementation combining NVIDIA GPU inference, medical imaging workflows, private clinical GenAI, Kubernetes orchestration, healthcare security controls, AI evaluation, observability, and deployment validation using synthetic or publicly licensed data.

The executable slice is:

synthetic DICOM → de-identification gate → MONAI preprocess → phantom detector → authenticated API → audit and metrics → acceptance report.

Private retrieval and the GPU cost model sit on that slice. Pixel data in this repository is a 64×64 geometric sphere. Study ids start with `SYN-`. The corpus declares `patient_records: 0`. The detector is a fixture. It is not a diagnostic device, and it does not diagnose.

**Deployment decision: NO-GO.** GPU latency, throughput, utilization, and TensorRT or ONNX parity have not been measured. The report is [`acceptance/deployment-acceptance.md`](acceptance/deployment-acceptance.md).

| | |
| --- | --- |
| Package | `hafactory` |
| Imaging | Synthetic secondary-capture DICOM, MONAI, bright-pixel detector |
| Serving | Triton model repository checked in; server not started |
| API | FastAPI, bearer token, tenant quota, approval gate |
| Assistant | Extractive RAG over three synthetic notes, no network call |
| Cluster | Kubernetes manifests and a Helm chart, Triton replicas 0, NIM off |
| Economics | $3.50 per GPU-hour assumption dated 2026-10-19 |

## How this sits next to the GPU factory

| | [NVIDIA H100 GPU Factory](https://github.com/JatinF/h100-ai-factory-reference-architecture) | This repository |
| --- | --- | --- |
| Focus | Physical infrastructure and cluster acceptance | Healthcare workloads and AI platform delivery |
| Core technologies | H100, NVLink, InfiniBand, NCCL, Slurm | Triton, MONAI, Kubernetes, retrieval, evaluation |
| Primary outcome | Validated GPU infrastructure design | Reproducible, secure AI service deployment |

The infrastructure repository is the design and the acceptance plan for the machines. This repository is the workload and the controls that run after those machines have measured evidence. The shared planning rate is $3.50 per GPU-hour, dated 2026-10-19. Both decisions stay NO-GO until the chassis under test supplies `MEASURED` rows. DGX and H100 specification figures are not results of this platform.

## Evidence

| Tag | Meaning in this repo |
| --- | --- |
| `ASSUMPTION` | $3.50 per GPU-hour and 18,000 inferences per GPU-hour |
| `DERIVED` | Dollars per inference = 3.50 / 18,000 |
| `EXAMPLE` | Synthetic notes, and the local CPU load-test record |
| `GATE` | Triton latency, throughput, utilization, accelerator parity |
| `MEASURED` | A row in `benchmarks/gpu-evidence.json` with serials and a log |

`SPEC` figures are not claimed here. Local CPU timings stay in the `EXAMPLE` record from `python -m hafactory.loadtest`. The cost page says utilization is unmeasured. Schema and the rule against committing a filled evidence file: [`benchmarks/README.md`](benchmarks/README.md).

## Run the software slice

Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Pytest runs the imaging pipeline, the API, the assistant, the local load record, manifest parsing, and a check that the committed dashboard and acceptance report match the generators. The job in [`.github/workflows/check.yml`](.github/workflows/check.yml) installs the same extra and runs pytest on Ubuntu with Python 3.12. That job does not start Triton and does not attach a GPU.

Print one local load record:

```bash
python -m hafactory.loadtest
```

Regenerate the two committed views after a generator change, then re-run pytest:

```bash
python -c "from pathlib import Path; from hafactory.economics import cost_per_inference, render_dashboard; from hafactory.acceptance_report import build_report, render_markdown; Path('observability').mkdir(exist_ok=True); Path('acceptance').mkdir(exist_ok=True); Path('observability/gpu-economics.html').write_text(render_dashboard(cost_per_inference()), encoding='utf-8'); Path('acceptance/deployment-acceptance.md').write_text(render_markdown(build_report(True)), encoding='utf-8')"
```

## API

Create the app with an audit path. The default token map is a fixture for tests and local runs. A deployment replaces it with the site identity provider.

| Token | Tenant | Role |
| --- | --- | --- |
| `dev-reader-a` | `tenant-a` | reader |
| `dev-approver-a` | `tenant-a` | approver |
| `dev-reader-b` | `tenant-b` | reader |

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/healthz` | Process up, `patient_records` 0 |
| GET | `/metrics` | Local process counters. The text says they are not GPU telemetry |
| POST | `/v1/inference` | Multipart field `dicom`. De-identification failure is 422. Triton URL set, or fault injection, is 503 |
| POST | `/v1/ask` | Extractive answer with citations, or `refused`, or `approval-required` |
| POST | `/v1/approvals` | Approver role only, `SYN-` study ids only |
| POST | `/v1/actions/export` | Same tenant and same study as the approval. No external system is called |

Unknown tokens return 401. The tenant comes from the token. Over quota returns 429.

## What each control is

| Capability | Where it lives |
| --- | --- |
| Threat model, PHI isolation, audit, policy | [`docs/threat-model.md`](docs/threat-model.md), `hafactory.deid`, `hafactory.audit` |
| Medical imaging | `hafactory.dicom_synth`, `hafactory.preprocess`, `tests/test_pipeline.py` |
| GPU deployment shape | [`deploy/kubernetes`](deploy/kubernetes), [`deploy/helm/healthcare-ai-factory`](deploy/helm/healthcare-ai-factory) |
| Model serving | [`models/phantom-detector`](models/phantom-detector), local load record in `hafactory.loadtest` |
| GPU economics | [`observability/gpu-economics.html`](observability/gpu-economics.html) |
| Groundedness and fail-closed release | `hafactory.rag`, [`acceptance/deployment-acceptance.md`](acceptance/deployment-acceptance.md) |
| Multi-tenancy | Namespace, RBAC, quota, and the token-to-tenant map |
| Reliability | [`docs/slos.md`](docs/slos.md), [`deploy/prometheus/rules.yaml`](deploy/prometheus/rules.yaml), [`docs/runbooks`](docs/runbooks) |
| Production delivery | [`.github/workflows/check.yml`](.github/workflows/check.yml) and the acceptance report |

The picture of the path is [`docs/architecture.md`](docs/architecture.md).

## Cluster apply

The manifests are the isolation shape: restricted Pod Security, a service account that can `get` configmaps, a GPU quota of one, default-deny networking, and an API container with a read-only root filesystem. Applying them does not start a GPU workload. Triton is at zero replicas. NIM is disabled. The container image is `registry.example.invalid/healthcare-ai-factory-api:unreleased`.

Raise Triton replicas only in an acceptance window, mount the model repository read-only, and file serials plus the Triton log under `benchmarks/`. Until that file exists, the decision in the acceptance report stays NO-GO.

## License

MIT. Copyright 2026 Jatinder Singh.
