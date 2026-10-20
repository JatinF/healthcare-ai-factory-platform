# Threat model

This document is the privacy and isolation model for the reference implementation. It is a repository control. It is not a HIPAA certification, a penetration test, or a statement that a file is de-identified under any regulation.

## Assets

| Asset | Where it lives | What leaves the process |
| --- | --- | --- |
| Synthetic DICOM | Temp file, deleted after the request | Study id and label in the audit line |
| Pixel data | Memory, then the temp file | Nothing in the audit log or the metrics text |
| Synthetic notes | `data/synthetic_notes.json` | Cited note ids in the API response |
| Question text | Request body | Not written to the audit log |
| Bearer tokens | In-memory map supplied to `create_app` | Token name, tenant, and role on the principal |
| Approvals | In-memory `ApprovalBook` for the process lifetime | Approval id and study id |
| Audit log | JSONL path chosen at startup | Action, result, tenant, actor, study id, label |

The corpus declares `patient_records: 0`. Study ids in this reference start with `SYN-`. The phantom is a drawn sphere, not a scan of a person.

## Actors

| Actor | Intended power |
| --- | --- |
| Unauthenticated caller | No route that reads a study or a note |
| Reader | Inference and questions inside the tenant on the token |
| Approver | Grant an approval for one `SYN-` study in that tenant |
| Other tenant | A different token, even with the same approval id |
| Cluster operator | Apply the namespace, quota, and network policy |

The tenant is read from the token. A field on the JSON body cannot change it.

## Boundaries

1. **De-identification gate.** `assert_deidentified` runs before preprocessing. A non-empty accession number, birth date, sex, address, institution, referring physician, or medical-record value fails the request. `PatientName` must be `SYNTHETIC` when present. `PatientID` and `StudyID` must start with `SYN-` when present. Missing pixel data fails the request. The API returns 422 and the audit line says `de-identification gate`. The uploaded bytes are unlinked in a `finally` block.
2. **Authentication.** A missing or unknown bearer token returns 401 and an audit line. Roles are `reader` or `approver`.
3. **Quota.** Each accepted authenticated call increments the tenant counter. Over the quota returns 429. A rejected study still consumes a unit, so identifier probing spends the same budget as a successful call.
4. **Assistant.** Answers are sentences copied from retrieved notes. `assert_grounded` fail-closes on a sentence that is not in a cited note. No supporting note returns `refused`. Prescribe, diagnose, and EHR wording returns `approval-required` and an empty answer. The implementation makes no network call. A hosted model API is out of policy for note text and for questions.
5. **Export.** `POST /v1/actions/export` requires an approval with the same id, the same tenant, and the same study id. Readers cannot grant. The destination string is `site-record-system-placeholder`.
6. **Cluster.** Namespace `clinical-inference` enforces restricted Pod Security. The API service account can `get` configmaps and nothing else, and it does not mount a token. Default-deny NetworkPolicy, plus TCP 8080 ingress to the API pods. The API root filesystem is read-only, capabilities are dropped, and the process uid is 65534. Triton stays at zero replicas. NIM stays disabled. The API container does not request a GPU.

## What an operator must replace

| Fixture in this repo | Replacement before any site use |
| --- | --- |
| `dev-reader-a`, `dev-approver-a`, `dev-reader-b` | Site identity provider. These names are test fixtures. |
| In-memory approval book | Durable approval store scoped to the tenant |
| `registry.example.invalid/healthcare-ai-factory-api:unreleased` | Site registry and a built image |
| Empty Triton model volume | Read-only mount of `models/phantom-detector` during an acceptance window |
| `HAF_TRITON_URL=""` | Set only after Triton is up and a measured client result exists. Until then the API returns 503 when the variable is non-empty. |

## Residual risks

- The approval book disappears when the process exits. A restart drops approvals. That is acceptable for the test slice and insufficient for a staffed service.
- Dev tokens are in source so the tests can run. They authorize nothing outside this process, and they are the wrong credential for a cluster.
- The de-identification gate checks a fixed keyword list. It is not a full DICOM confidential-attribute profile.
- Audit lines include the study id. They do not include pixels or the question. An operator who points the audit file at a shared disk should still treat study ids as sensitive operational data.
- Network egress is denied by the default policy, including from the API pod. That matches the rule that this path does not call a hosted model. A future Triton client needs an explicit egress rule to the Triton pod, written in the same change that sets `HAF_TRITON_URL`.
- No GPU measurement exists. Publishing the modeled cost or the local CPU load test as utilization would misstate the system. The acceptance report stays NO-GO to prevent that.

## Evidence classes used when a number appears

| Class | Use here |
| --- | --- |
| `ASSUMPTION` | $3.50 per GPU-hour and 18,000 inferences per GPU-hour, dated 2026-10-19 |
| `EXAMPLE` | The three synthetic notes, and the local CPU load-test record |
| `GATE` | Triton latency, throughput, utilization, and accelerator parity |
| `MEASURED` | Only a row in `benchmarks/gpu-evidence.json` with serials and a log |
| `SPEC` | Not claimed by this repository |
| `DERIVED` | The per-inference dollar figure is the assumption divided by the assumption |
