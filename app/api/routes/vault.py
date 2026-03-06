"""User-scoped ciphertext vault APIs."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.api.auth import AuthenticatedUser, get_current_user
from app.api.schemas.vault import (
    VaultDeleteResponse,
    VaultRecordOut,
    VaultRecordUpsert,
    VaultSettingsOut,
    VaultSettingsUpsert,
)
from app.registry_store.dependencies import get_registry_store_db
from app.vault.models import UserPatientVault, UserVaultSettings

router = APIRouter(prefix="/v1/vault", tags=["vault"])

_current_user_dep = Depends(get_current_user)
_db_dep = Depends(get_registry_store_db)

MAX_RECORD_CIPHERTEXT_B64_LEN = int(
    os.getenv("VAULT_RECORD_MAX_B64_LEN", "16384").strip() or "16384"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _settings_to_out(row: UserVaultSettings) -> VaultSettingsOut:
    return VaultSettingsOut(
        user_id=str(row.user_id),
        wrapped_vmk_b64=str(row.wrapped_vmk_b64),
        wrap_iv_b64=str(row.wrap_iv_b64),
        kdf_salt_b64=str(row.kdf_salt_b64),
        kdf_iterations=int(row.kdf_iterations),
        kdf_hash=str(row.kdf_hash),
        crypto_version=int(row.crypto_version),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _record_to_out(row: UserPatientVault) -> VaultRecordOut:
    return VaultRecordOut(
        user_id=str(row.user_id),
        registry_uuid=row.registry_uuid,
        ciphertext_b64=str(row.ciphertext_b64),
        iv_b64=str(row.iv_b64),
        crypto_version=int(row.crypto_version),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("/settings", response_model=VaultSettingsOut)
def get_vault_settings(
    current_user: Annotated[AuthenticatedUser, _current_user_dep],
    db: Annotated[Session, _db_dep],
) -> VaultSettingsOut:
    stmt: Select[tuple[UserVaultSettings]] = select(UserVaultSettings).where(
        UserVaultSettings.user_id == current_user.id
    )
    row = db.execute(stmt).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Vault settings not found"
        )
    return _settings_to_out(row)


@router.put("/settings", response_model=VaultSettingsOut)
def put_vault_settings(
    payload: VaultSettingsUpsert,
    current_user: Annotated[AuthenticatedUser, _current_user_dep],
    db: Annotated[Session, _db_dep],
) -> VaultSettingsOut:
    stmt: Select[tuple[UserVaultSettings]] = select(UserVaultSettings).where(
        UserVaultSettings.user_id == current_user.id
    )
    row = db.execute(stmt).scalar_one_or_none()
    now = _utcnow()

    if row is None:
        row = UserVaultSettings(
            user_id=current_user.id,
            wrapped_vmk_b64=payload.wrapped_vmk_b64,
            wrap_iv_b64=payload.wrap_iv_b64,
            kdf_salt_b64=payload.kdf_salt_b64,
            kdf_iterations=payload.kdf_iterations,
            kdf_hash=payload.kdf_hash,
            crypto_version=payload.crypto_version,
            created_at=now,
            updated_at=now,
        )
    else:
        row.wrapped_vmk_b64 = payload.wrapped_vmk_b64
        row.wrap_iv_b64 = payload.wrap_iv_b64
        row.kdf_salt_b64 = payload.kdf_salt_b64
        row.kdf_iterations = payload.kdf_iterations
        row.kdf_hash = payload.kdf_hash
        row.crypto_version = payload.crypto_version
        row.updated_at = now

    db.add(row)
    db.commit()
    db.refresh(row)
    return _settings_to_out(row)


@router.get("/records", response_model=list[VaultRecordOut])
def get_vault_records(
    current_user: Annotated[AuthenticatedUser, _current_user_dep],
    db: Annotated[Session, _db_dep],
) -> list[VaultRecordOut]:
    stmt: Select[tuple[UserPatientVault]] = (
        select(UserPatientVault)
        .where(UserPatientVault.user_id == current_user.id)
        .order_by(UserPatientVault.updated_at.desc())
    )
    rows = db.execute(stmt).scalars().all()
    return [_record_to_out(row) for row in rows]


@router.put("/record", response_model=VaultRecordOut)
def put_vault_record(
    payload: VaultRecordUpsert,
    response: Response,
    current_user: Annotated[AuthenticatedUser, _current_user_dep],
    db: Annotated[Session, _db_dep],
) -> VaultRecordOut:
    if len(payload.ciphertext_b64) > MAX_RECORD_CIPHERTEXT_B64_LEN:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"ciphertext_b64 exceeds {MAX_RECORD_CIPHERTEXT_B64_LEN} characters",
        )

    stmt: Select[tuple[UserPatientVault]] = select(UserPatientVault).where(
        UserPatientVault.user_id == current_user.id,
        UserPatientVault.registry_uuid == payload.registry_uuid,
    )
    row = db.execute(stmt).scalar_one_or_none()
    now = _utcnow()

    if row is None:
        row = UserPatientVault(
            user_id=current_user.id,
            registry_uuid=payload.registry_uuid,
            ciphertext_b64=payload.ciphertext_b64,
            iv_b64=payload.iv_b64,
            crypto_version=payload.crypto_version,
            created_at=now,
            updated_at=now,
        )
        response.status_code = status.HTTP_201_CREATED
    else:
        row.ciphertext_b64 = payload.ciphertext_b64
        row.iv_b64 = payload.iv_b64
        row.crypto_version = payload.crypto_version
        row.updated_at = now

    db.add(row)
    db.commit()
    db.refresh(row)
    return _record_to_out(row)


@router.delete("/records/{registry_uuid}", response_model=VaultDeleteResponse)
def delete_vault_record(
    registry_uuid: uuid.UUID,
    current_user: Annotated[AuthenticatedUser, _current_user_dep],
    db: Annotated[Session, _db_dep],
) -> VaultDeleteResponse:
    stmt: Select[tuple[UserPatientVault]] = select(UserPatientVault).where(
        UserPatientVault.user_id == current_user.id,
        UserPatientVault.registry_uuid == registry_uuid,
    )
    row = db.execute(stmt).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vault record not found")

    db.delete(row)
    db.commit()
    return VaultDeleteResponse(ok=True, registry_uuid=registry_uuid)


__all__ = ["router"]
