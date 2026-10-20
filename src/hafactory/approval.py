"""Human approval records for high-risk actions.

An approval is bound to one tenant and one study. The reader role cannot
create one. Export without a matching approval fail-closes.
"""

from __future__ import annotations

from dataclasses import dataclass


class ApprovalError(PermissionError):
    pass


@dataclass
class Approval:
    approval_id: str
    tenant: str
    study_id: str
    approver: str


class ApprovalBook:
    def __init__(self) -> None:
        self._items: dict[str, Approval] = {}

    def grant(self, approval: Approval) -> Approval:
        if not approval.study_id.startswith("SYN-"):
            raise ApprovalError("approvals in this reference are limited to SYN- studies")
        self._items[approval.approval_id] = approval
        return approval

    def require(self, approval_id: str, tenant: str, study_id: str) -> Approval:
        approval = self._items.get(approval_id)
        if approval is None:
            raise ApprovalError("unknown approval")
        if approval.tenant != tenant or approval.study_id != study_id:
            raise ApprovalError("approval does not match this tenant and study")
        return approval
