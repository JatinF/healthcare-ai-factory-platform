# Runbook: inference API down

Alert: `InferenceApiDown` (page) or `InferenceBackendErrors` (ticket).

The page means Prometheus has seen `up == 0` for the `healthcare-inference-api` job for five minutes. The ticket means the API target is up and has recorded no inference requests for thirty minutes during a staffed window. Neither alert ships with sample series in this repository.

## Confirm

1. Read the alert annotation and open this runbook. Do not restart Triton as the first step. The published Triton Deployment has `replicas: 0`.
2. Check the API pod in `clinical-inference` and its readiness probe, `GET /healthz` on port 8080. A healthy body is `{"status":"ok","patient_records":0}`.
3. Read the last audit lines. They contain action, result, tenant, actor, and sometimes a study id. They do not contain pixels or question text. Treat study ids as operational data.
4. Read `HAF_TRITON_URL` on the pod. An empty value selects the local detector and the response runtime is `local-cpu`. A non-empty value returns HTTP 503 with the detail that this process has no measured Triton client result. That 503 is the fail-closed path. Do not clear the failure by pointing the URL at a server and expecting the CPU detector to answer in its place.

## Fault injection

Tests construct the app with `force_triton_down=True`. The inference route then returns 503 and writes an audit denial `triton unavailable`. Use that switch in a staging process to rehearse the page. Do not use it as a way to mark the GPU gate passed.

## Restore

1. If the pod is crash-looping, roll back the API image using `docs/runbooks/rollback.md`. The image name in the manifest is `registry.example.invalid/healthcare-ai-factory-api:unreleased` until a site registry exists.
2. If the URL was set before a Triton measurement existed, set `HAF_TRITON_URL` back to empty, restart the API, and confirm `/v1/inference` on a synthetic `SYN-` slice returns `phantom-present`.
3. Raise Triton replicas above zero only inside an acceptance window, with the model repository mounted read-only and a GPU on the node. Then capture serials and the Triton log into `benchmarks/gpu-evidence.json` on that chassis. Until those rows are `MEASURED` passes, `acceptance/deployment-acceptance.md` stays **NO-GO**.
4. Confirm the network policy still denies egress except for a rule you added on purpose. This path does not call a hosted model.

## Do not

- Copy `python -m hafactory.loadtest` timings into the GPU evidence file.
- Copy the modeled cost page into a utilization report. Utilization on that page is none.
- Enable `nim.enabled` to bypass a down API.
