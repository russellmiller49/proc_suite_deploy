"""Add user vault settings and encrypted patient vault tables.

These tables store only client-encrypted payloads (no plaintext PHI).
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "d7e8f9a0b1c2"
down_revision = "c1d2e3f4a5b6"
branch_labels = None
depends_on = None


UUIDType = postgresql.UUID(as_uuid=True).with_variant(sa.String(length=36), "sqlite")


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _enable_rls_policies() -> None:
    if not _is_postgres():
        return

    op.execute("ALTER TABLE user_vault_settings ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE user_patient_vault ENABLE ROW LEVEL SECURITY")

    op.execute("DROP POLICY IF EXISTS user_vault_settings_owner_policy ON user_vault_settings")
    op.execute(
        """
        CREATE POLICY user_vault_settings_owner_policy
        ON user_vault_settings
        USING (user_id::text = auth.uid()::text)
        WITH CHECK (user_id::text = auth.uid()::text)
        """
    )

    op.execute("DROP POLICY IF EXISTS user_patient_vault_owner_policy ON user_patient_vault")
    op.execute(
        """
        CREATE POLICY user_patient_vault_owner_policy
        ON user_patient_vault
        USING (user_id::text = auth.uid()::text)
        WITH CHECK (user_id::text = auth.uid()::text)
        """
    )


def _drop_rls_policies() -> None:
    if not _is_postgres():
        return
    op.execute("DROP POLICY IF EXISTS user_patient_vault_owner_policy ON user_patient_vault")
    op.execute("DROP POLICY IF EXISTS user_vault_settings_owner_policy ON user_vault_settings")


def upgrade() -> None:
    op.create_table(
        "user_vault_settings",
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("wrapped_vmk_b64", sa.Text(), nullable=False),
        sa.Column("wrap_iv_b64", sa.String(length=64), nullable=False),
        sa.Column("kdf_salt_b64", sa.String(length=128), nullable=False),
        sa.Column("kdf_iterations", sa.Integer(), nullable=False),
        sa.Column("kdf_hash", sa.String(length=32), nullable=False, server_default="PBKDF2-SHA256"),
        sa.Column("crypto_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("user_id", name="pk_user_vault_settings"),
    )
    op.create_index(
        "ix_user_vault_settings_user_id",
        "user_vault_settings",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "user_patient_vault",
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("registry_uuid", UUIDType, nullable=False),
        sa.Column("ciphertext_b64", sa.Text(), nullable=False),
        sa.Column("iv_b64", sa.String(length=64), nullable=False),
        sa.Column("crypto_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("user_id", "registry_uuid", name="pk_user_patient_vault"),
    )
    op.create_index(
        "ix_user_patient_vault_user_id",
        "user_patient_vault",
        ["user_id"],
        unique=False,
    )

    _enable_rls_policies()


def downgrade() -> None:
    _drop_rls_policies()
    op.drop_index("ix_user_patient_vault_user_id", table_name="user_patient_vault")
    op.drop_table("user_patient_vault")
    op.drop_index("ix_user_vault_settings_user_id", table_name="user_vault_settings")
    op.drop_table("user_vault_settings")

