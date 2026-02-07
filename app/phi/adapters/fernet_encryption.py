"""Fernet-based PHI encryption adapter.

Closer to production behavior but still intended for synthetic/demo PHI here.
Never log keys or plaintext.
"""

from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken

from app.phi.ports import PHIEncryptionPort


class FernetEncryptionAdapter(PHIEncryptionPort):
    """Encrypt/decrypt PHI payloads using Fernet symmetric keys."""

    def __init__(self, key: str | bytes | None = None):
        key = key or os.getenv("PHI_ENCRYPTION_KEY")
        if not key:
            raise RuntimeError("PHI_ENCRYPTION_KEY is not set")
        if isinstance(key, str):
            key = key.encode("utf-8")
        self._fernet = Fernet(key)

    def encrypt(self, plaintext: str) -> tuple[bytes, str, int]:
        ciphertext = self._fernet.encrypt(plaintext.encode("utf-8"))
        return ciphertext, "FERNET", 1

    def decrypt(self, ciphertext: bytes, algorithm: str, key_version: int) -> str:
        if algorithm != "FERNET":
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        try:
            plaintext_bytes = self._fernet.decrypt(ciphertext)
        except InvalidToken as exc:
            raise ValueError("Failed to decrypt PHI vault data") from exc
        return plaintext_bytes.decode("utf-8")


__all__ = ["FernetEncryptionAdapter"]
