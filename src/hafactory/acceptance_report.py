"""Deployment acceptance report.

Software gates are checks this module can execute. GPU serving gates stay
not-run until an evidence file names a host, a serial, and a Triton log.
The production decision is NO-GO while any required GPU row is not a measured pass.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GPU_EVIDENCE = ROOT / "benchmarks" / "gpu-evidence.json"

GPU_GATES = (
    "triton-latency",
    "triton-throughput",
    "gpu-utilization",
    "tensorrt-or-onnx-parity",
)


def gpu_rows(evidence_path: Path | None = None) -> list[dict]:
    path = evidence_path if evidence_path is not None else GPU_EVIDENCE
    records = []
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
    by_id = {record.get("test_id"): record for record in records}
    rows = []
    for test_id in GPU_GATES:
        record = by_id.get(test_id)
        if (
            record
            and record.get("evidence_class") == "MEASURED"
            and record.get("result") == "pass"
            and record.get("serials")
            and record.get("log")
        ):
            rows.append({"test_id": test_id, "status": "pass"})
        elif record:
            rows.append({"test_id": test_id, "status": "fail"})
        else:
            rows.append({"test_id": test_id, "status": "not-run"})
    return rows


def software_checks() -> bool:
    """Execute the local slice. Returns false when a gate inside this process fails."""
    import tempfile

    from hafactory.deid import DeidError, assert_deidentified
    from hafactory.dicom_synth import write_synthetic_slice
    from hafactory.economics import cost_per_inference
    from hafactory.pipeline import run_slice
    from hafactory.rag import assert_grounded, grounded_answer, load_notes
    from pydicom import dcmread

    with tempfile.TemporaryDirectory() as tmp:
        clean = write_synthetic_slice(Path(tmp) / "clean.dcm")
        poisoned = write_synthetic_slice(
            Path(tmp) / "poison.dcm",
            poison={"AccessionNumber": "ACC-SHOULD-FAIL"},
        )
        result = run_slice(clean)
        if result["label"] != "phantom-present" or result["patient_records"] != 0:
            return False
        try:
            assert_deidentified(dcmread(poisoned))
        except DeidError:
            pass
        else:
            return False
    notes = load_notes()
    answer = grounded_answer("What phantom threshold does the synthetic study use?", notes)
    if answer["status"] != "answered" or not answer["citations"]:
        return False
    try:
        assert_grounded("This sentence was not in the corpus.", {"SYN-NOTE-001": notes[0]["text"]})
    except Exception:
        pass
    else:
        return False
    model = cost_per_inference()
    return model["evidence_class"] == "ASSUMPTION" and model["measured_gpu_results"] == 0


def build_report(software_ok: bool, evidence_path: Path | None = None) -> dict:
    rows = gpu_rows(evidence_path)
    gpu_go = all(row["status"] == "pass" for row in rows)
    if software_ok and gpu_go:
        decision = "GO"
    elif software_ok:
        decision = "NO-GO"
    else:
        decision = "NO-GO"
    return {
        "decision": decision,
        "software_slice": "pass" if software_ok else "fail",
        "gpu_rows": rows,
        "reason": (
            "Software slice passed and GPU evidence is filed."
            if decision == "GO"
            else "Production serving stays NO-GO until every GPU gate is a MEASURED pass on the chassis that will run it. A passing unit test is not that evidence."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Deployment acceptance report",
        "",
        f"**Decision: {report['decision']}**",
        "",
        f"Software vertical slice: {report['software_slice']}.",
        "",
        report["reason"],
        "",
        "| GPU gate | Status |",
        "| --- | --- |",
    ]
    for row in report["gpu_rows"]:
        lines.append(f"| {row['test_id']} | {row['status']} |")
    lines.extend(
        [
            "",
            "Latency, throughput, and GPU utilization for Triton are rows in this table. They are not copied from the modeled cost page.",
            "",
        ]
    )
    return "\n".join(lines)
