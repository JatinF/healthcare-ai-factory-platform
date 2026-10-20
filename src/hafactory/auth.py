"""Token map for development and tests.

Production replaces this map with the site identity provider. Tokens are
looked up by exact value. An unknown token fail-closes. The tenant comes
from the token, not from the request body.
"""

from __future__ import annotations

from dataclasses import dataclass


class AuthError(PermissionError):
    pass


@dataclass(frozen=True)
class Principal:
    token_name: str
    tenant: str
    role: str


def authenticate(header: str | None, tokens: dict[str, dict[str, str]]) -> Principal:
    if not header or not header.lower().startswith("bearer "):
        raise AuthError("missing bearer token")
    token = header.split(" ", 1)[1].strip()
    record = tokens.get(token)
    if not record:
        raise AuthError("unknown token")
    role = record.get("role", "")
    tenant = record.get("tenant", "")
    if role not in {"reader", "approver"} or not tenant:
        raise AuthError("token is missing a tenant or a role")
    return Principal(token_name=record.get("name", "unnamed"), tenant=tenant, role=role)


def require_role(principal: Principal, role: str) -> None:
    if principal.role != role:
        raise AuthError(f"role {role} is required")
