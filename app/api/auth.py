"""Auth utilities for user-scoped API endpoints.

Priority:
1) Bearer JWT (Supabase-compatible)
2) Optional `X-User-Id` fallback (dev only by default)
"""

from __future__ import annotations

import os
from functools import lru_cache

import jwt
from fastapi import Header, HTTPException, status
from pydantic import BaseModel


class AuthenticatedUser(BaseModel):
    id: str


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_production() -> bool:
    if _truthy_env("CODER_REQUIRE_PHI_REVIEW"):
        return True
    return os.getenv("PROCSUITE_ENV", "").strip().lower() == "production"


def _allow_x_user_id_fallback() -> bool:
    if "VAULT_AUTH_ALLOW_X_USER_ID" in os.environ:
        return _truthy_env("VAULT_AUTH_ALLOW_X_USER_ID")
    return not _is_production()


def _jwt_algorithms() -> list[str]:
    raw = os.getenv("SUPABASE_JWT_ALGORITHMS", "HS256,RS256").strip()
    algs = [part.strip() for part in raw.split(",") if part.strip()]
    return algs or ["HS256", "RS256"]


def _resolve_jwks_url() -> str | None:
    explicit = os.getenv("SUPABASE_JWKS_URL", "").strip()
    if explicit:
        return explicit
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    if not supabase_url:
        return None
    return f"{supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"


@lru_cache(maxsize=4)
def _jwk_client(jwks_url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(jwks_url)


def _extract_user_id_from_claims(claims: dict) -> str:
    candidate = str(claims.get("sub") or claims.get("user_id") or "").strip()
    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT missing subject claim"
        )
    if len(candidate) > 255:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user id")
    return candidate


def _decode_bearer_token(token: str) -> str:
    algorithms = _jwt_algorithms()
    secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    options = {"verify_aud": False}

    try:
        if secret:
            claims = jwt.decode(token, secret, algorithms=algorithms, options=options)
            return _extract_user_id_from_claims(claims)

        jwks_url = _resolve_jwks_url()
        if jwks_url:
            client = _jwk_client(jwks_url)
            signing_key = client.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, signing_key, algorithms=algorithms, options=options)
            return _extract_user_id_from_claims(claims)
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Bearer auth unavailable: set SUPABASE_JWT_SECRET or SUPABASE_URL/SUPABASE_JWKS_URL",
    )


def _extract_bearer_token(authorization: str | None) -> str | None:
    value = (authorization or "").strip()
    if not value:
        return None
    if not value.lower().startswith("bearer "):
        return None
    token = value[7:].strip()
    return token or None


def get_current_user(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> AuthenticatedUser:
    """Resolve current user from request headers.

    - Prefer Bearer JWT (Supabase-compatible verification).
    - Optionally accept `X-User-Id` in non-production/dev contexts.
    """
    token = _extract_bearer_token(authorization)
    if token:
        return AuthenticatedUser(id=_decode_bearer_token(token))

    user_id = (x_user_id or "").strip()
    if user_id and _allow_x_user_id_fallback():
        if len(user_id) > 255:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user id")
        return AuthenticatedUser(id=user_id)

    if user_id and not _allow_x_user_id_fallback():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-User-Id auth disabled; use Bearer token",
        )

    if _allow_x_user_id_fallback():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication (Bearer token or X-User-Id)",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing Bearer token",
    )


def build_auth_headers_for_user_id(user_id: str) -> dict[str, str]:
    """Test utility helper for legacy/dev auth calls."""
    value = str(user_id or "").strip()
    if not value:
        return {}
    return {"X-User-Id": value}


def build_bearer_header(token: str) -> dict[str, str]:
    value = str(token or "").strip()
    if not value:
        return {}
    return {"Authorization": f"Bearer {value}"}


__all__ = [
    "AuthenticatedUser",
    "build_auth_headers_for_user_id",
    "build_bearer_header",
    "get_current_user",
]
