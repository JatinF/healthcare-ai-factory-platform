# Runbook: rollback

Use this when the API image or the Helm values are the cause of `InferenceApiDown`, or when a change set the Triton URL without a measured client.

## Steps

1. Identify the last known image. In this repository the manifest and the Helm values both name `registry.example.invalid/healthcare-ai-factory-api:unreleased`. A site replaces that name. Roll back to the previous digest in the site registry, not to a tag that was never pushed.
2. With Helm, roll back the release in `clinical-inference`:

   ```bash
   helm history healthcare-ai-factory --namespace clinical-inference
   helm rollback healthcare-ai-factory <REVISION> --namespace clinical-inference
   ```

   Chart version in this repo is `0.1.0`. `triton.enabled` and `nim.enabled` are false. A rollback that flips either flag on is not a rollback.
3. Confirm the API Deployment still has `readOnlyRootFilesystem`, `runAsUser: 65534`, `drop: ["ALL"]`, an empty `HAF_TRITON_URL`, and no GPU request. Confirm the Triton Deployment is at `replicas: 0` unless an acceptance window is open.
4. Probe `GET /healthz`. Send one synthetic slice with a site token and confirm the label is `phantom-present` and the runtime is `local-cpu`.
5. Re-run the software slice before calling the rollback done:

   ```bash
   python -m pytest
   ```

   The committed acceptance report must still render as **NO-GO** when `benchmarks/gpu-evidence.json` is absent. A green unit test does not change that decision.

## Afterward

Write the audit of the rollback in the change record: who rolled back, which revision, and which alert cleared. Do not attach local CPU timings as GPU evidence. If the failure was a de-identification reject, that is a 422 by design. Rolling forward by weakening `hafactory.deid` is outside this runbook.
