"""Schemas for ciphertext-only client vault APIs."""

from __future__ import annotations

import base64
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


CRYPTO_VERSION_V1: Literal[1] = 1


def _decode_base64(value: str, *, field_name: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{field_name} must be valid base64") from exc


class VaultSettingsUpsert(BaseModel):
    wrapped_vmk_b64: str = Field(..., min_length=16, max_length=4096)
    wrap_iv_b64: str = Field(..., min_length=8, max_length=64)
    kdf_salt_b64: str = Field(..., min_length=16, max_length=128)
    kdf_iterations: int = Field(..., ge=100_000, le=2_000_000)
    kdf_hash: str = Field(default="PBKDF2-SHA256", min_length=3, max_length=32)
    crypto_version: Literal[1] = CRYPTO_VERSION_V1

    @field_validator("wrapped_vmk_b64")
    @classmethod
    def _validate_wrapped_vmk_b64(cls, value: str) -> str:
        _decode_base64(value, field_name="wrapped_vmk_b64")
        return value

    @field_validator("kdf_salt_b64")
    @classmethod
    def _validate_kdf_salt_b64(cls, value: str) -> str:
        _decode_base64(value, field_name="kdf_salt_b64")
        return value

    @field_validator("wrap_iv_b64")
    @classmethod
    def _validate_wrap_iv(cls, value: str) -> str:
        decoded = _decode_base64(value, field_name="wrap_iv_b64")
        if len(decoded) != 12:
            raise ValueError("wrap_iv_b64 must decode to 12 bytes")
        return value


class VaultSettingsOut(VaultSettingsUpsert):
    user_id: str
    created_at: datetime
    updated_at: datetime


class VaultRecordUpsert(BaseModel):
    registry_uuid: uuid.UUID
    ciphertext_b64: str = Field(..., min_length=16, max_length=32_768)
    iv_b64: str = Field(..., min_length=8, max_length=64)
    crypto_version: Literal[1] = CRYPTO_VERSION_V1

    @field_validator("ciphertext_b64")
    @classmethod
    def _validate_ciphertext_b64(cls, value: str) -> str:
        _decode_base64(value, field_name="ciphertext_b64")
        return value

    @field_validator("iv_b64")
    @classmethod
    def _validate_iv(cls, value: str) -> str:
        decoded = _decode_base64(value, field_name="iv_b64")
        if len(decoded) != 12:
            raise ValueError("iv_b64 must decode to 12 bytes")
        return value


class VaultRecordOut(VaultRecordUpsert):
    user_id: str
    created_at: datetime
    updated_at: datetime


class VaultDeleteResponse(BaseModel):
    ok: bool = True
    registry_uuid: uuid.UUID


__all__ = [
    "CRYPTO_VERSION_V1",
    "VaultDeleteResponse",
    "VaultRecordOut",
    "VaultRecordUpsert",
    "VaultSettingsOut",
    "VaultSettingsUpsert",
]

