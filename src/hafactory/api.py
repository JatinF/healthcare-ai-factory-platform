"""Authenticated API for the synthetic imaging slice and the private assistant.

Triton is used when HAF_TRITON_URL is set and the client library is installed.
Otherwise the local detector runs and the response says runtime=local-cpu.
A Triton outage fail-closes with HTTP 503. It does not fall through silently
when the caller asked for Triton.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from hafactory.approval import Approval, ApprovalBook, ApprovalError
from hafactory.audit import AuditLog
from hafactory.auth import AuthError, Principal, authenticate, require_role
from hafactory.deid import DeidError
from hafactory.pipeline import run_slice
from hafactory.rag import grounded_answer, load_notes

DEV_TOKENS = {
    "dev-reader-a": {"name": "dev-reader-a", "tenant": "tenant-a", "role": "reader"},
    "dev-approver-a": {"name": "dev-approver-a", "tenant": "tenant-a", "role": "approver"},
    "dev-reader-b": {"name": "dev-reader-b", "tenant": "tenant-b", "role": "reader"},
}


class Question(BaseModel):
    question: str


class ExportRequest(BaseModel):
    study_id: str
    approval_id: str


class ApprovalRequest(BaseModel):
    study_id: str
    approval_id: str


def create_app(
    audit_path: Path,
    tokens: dict[str, dict[str, str]] | None = None,
    quota: int = 100,
    triton_url: str | None = None,
    force_triton_down: bool = False,
) -> FastAPI:
    app = FastAPI(title="Healthcare AI factory", version="0.1.0")
    audit = AuditLog(audit_path)
    approvals = ApprovalBook()
    token_map = tokens if tokens is not None else dict(DEV_TOKENS)
    counts: dict[str, int] = {}
    metrics = {"inference_requests": 0, "inference_latency_sum_seconds": 0.0}
    notes = load_notes()
    configured_triton = triton_url if triton_url is not None else os.environ.get("HAF_TRITON_URL", "")

    def principal(authorization: str | None) -> Principal:
        try:
            return authenticate(authorization, token_map)
        except AuthError as exc:
            audit.write(action="auth", result="deny", detail=str(exc))
            raise HTTPException(status_code=401, detail="unauthorized") from exc

    def charge(who: Principal) -> None:
        used = counts.get(who.tenant, 0) + 1
        if used > quota:
            audit.write(action="quota", result="deny", tenant=who.tenant, actor=who.token_name)
            raise HTTPException(status_code=429, detail="tenant quota exceeded")
        counts[who.tenant] = used

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "patient_records": 0}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_text() -> str:
        count = metrics["inference_requests"]
        total = metrics["inference_latency_sum_seconds"]
        return (
            "# Local API process counters. Not GPU telemetry.\n"
            f"hafactory_inference_requests_total {count}\n"
            f"hafactory_inference_latency_seconds_sum {total}\n"
        )

    @app.post("/v1/inference")
    async def inference(
        dicom: UploadFile = File(...),
        authorization: str | None = Header(default=None),
    ) -> dict:
        who = principal(authorization)
        charge(who)
        if configured_triton or force_triton_down:
            if force_triton_down or not configured_triton:
                audit.write(action="inference", result="deny", tenant=who.tenant, actor=who.token_name, detail="triton unavailable")
                raise HTTPException(status_code=503, detail="inference backend unavailable")
            audit.write(action="inference", result="deny", tenant=who.tenant, actor=who.token_name, detail="triton client not executed in this process")
            raise HTTPException(status_code=503, detail="Triton URL is set but this process has no measured Triton client result")
        payload = await dicom.read()
        started = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as handle:
            handle.write(payload)
            destination = Path(handle.name)
        try:
            result = run_slice(destination)
        except DeidError as exc:
            audit.write(action="inference", result="deny", tenant=who.tenant, actor=who.token_name, detail="de-identification gate")
            raise HTTPException(status_code=422, detail={"deid": exc.problems}) from exc
        finally:
            destination.unlink(missing_ok=True)
        elapsed = time.perf_counter() - started
        metrics["inference_requests"] += 1
        metrics["inference_latency_sum_seconds"] += elapsed
        audit.write(
            action="inference",
            result="allow",
            tenant=who.tenant,
            actor=who.token_name,
            study_id=result["study_id"],
            label=result["label"],
        )
        result["tenant"] = who.tenant
        result["latency_seconds"] = elapsed
        return result

    @app.post("/v1/ask")
    def ask(body: Question, authorization: str | None = Header(default=None)) -> dict:
        who = principal(authorization)
        charge(who)
        try:
            result = grounded_answer(body.question, notes)
        except Exception as exc:
            audit.write(action="ask", result="deny", tenant=who.tenant, actor=who.token_name, detail="groundedness")
            raise HTTPException(status_code=422, detail="answer failed the groundedness gate") from exc
        audit.write(action="ask", result=result["status"], tenant=who.tenant, actor=who.token_name)
        return result

    @app.post("/v1/approvals")
    def grant(body: ApprovalRequest, authorization: str | None = Header(default=None)) -> dict:
        who = principal(authorization)
        try:
            require_role(who, "approver")
            approvals.grant(
                Approval(
                    approval_id=body.approval_id,
                    tenant=who.tenant,
                    study_id=body.study_id,
                    approver=who.token_name,
                )
            )
        except (AuthError, ApprovalError) as exc:
            audit.write(action="approve", result="deny", tenant=who.tenant, actor=who.token_name, detail=str(exc))
            status = 403
            raise HTTPException(status_code=status, detail="approval denied") from exc
        audit.write(action="approve", result="allow", tenant=who.tenant, actor=who.token_name, study_id=body.study_id)
        return {"approval_id": body.approval_id, "study_id": body.study_id}

    @app.post("/v1/actions/export")
    def export(body: ExportRequest, authorization: str | None = Header(default=None)) -> dict:
        who = principal(authorization)
        try:
            approvals.require(body.approval_id, who.tenant, body.study_id)
        except ApprovalError as exc:
            audit.write(action="export", result="deny", tenant=who.tenant, actor=who.token_name, study_id=body.study_id)
            raise HTTPException(status_code=403, detail="export denied without a matching approval") from exc
        audit.write(action="export", result="allow", tenant=who.tenant, actor=who.token_name, study_id=body.study_id)
        return {
            "status": "accepted",
            "study_id": body.study_id,
            "destination": "site-record-system-placeholder",
            "note": "No external system was called. The approval gate is what this endpoint demonstrates.",
        }

    return app
