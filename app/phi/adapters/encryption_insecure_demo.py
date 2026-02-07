"""Demo-only PHI encryption adapter.

NOT FOR PRODUCTION. This is intentionally reversible and only suitable
for synthetic PHI in the demo environment.
"""

from __future__ import annotations

from app.phi.ports import PHIEncryptionPort


class InsecureDemoEncryptionAdapter(PHIEncryptionPort):
    """A trivial reversible adapter for synthetic/demo data."""

    def encrypt(self, plaintext: str) -> tuple[bytes, str, int]:
        ciphertext = b"demo-" + plaintext.encode("utf-8")
        return ciphertext, "PLAINTEXT_DEMO", 1

    def decrypt(self, ciphertext: bytes, algorithm: str, key_version: int) -> str:
        prefix = b"demo-"
        data = ciphertext[len(prefix) :] if ciphertext.startswith(prefix) else ciphertext
        return data.decode("utf-8")


__all__ = ["InsecureDemoEncryptionAdapter"]
