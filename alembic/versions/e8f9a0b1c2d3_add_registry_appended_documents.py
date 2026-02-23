"""Add registry_appended_documents table for case-level append workflow."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "e8f9a0b1c2d3"
down_revision = "d7e8f9a0b1c2"
branch_labels = None
depends_on = None


UUIDType = postgresql.UUID(as_uuid=True).with_variant(sa.String(length=36), "sqlite")
JSONType = postgresql.JSONB().with_variant(sa.JSON(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "registry_appended_documents",
        sa.Column("id", UUIDType, nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("registry_uuid", UUIDType, nullable=False),
        sa.Column("note_text", sa.Text(), nullable=False),
        sa.Column("note_sha256", sa.String(length=64), nullable=False),
        sa.Column("document_kind", sa.String(length=64), nullable=False, server_default="pathology"),
        sa.Column("source_type", sa.String(length=64), nullable=True),
        sa.Column(
            "ocr_correction_applied",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("metadata", JSONType, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_registry_appended_documents_user_id",
        "registry_appended_documents",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_registry_appended_documents_registry_uuid",
        "registry_appended_documents",
        ["registry_uuid"],
        unique=False,
    )
    op.create_index(
        "ix_registry_appended_documents_note_sha256",
        "registry_appended_documents",
        ["note_sha256"],
        unique=False,
    )
    op.create_index(
        "ix_registry_appended_documents_created_at",
        "registry_appended_documents",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_registry_appended_documents_created_at",
        table_name="registry_appended_documents",
    )
    op.drop_index(
        "ix_registry_appended_documents_note_sha256",
        table_name="registry_appended_documents",
    )
    op.drop_index(
        "ix_registry_appended_documents_registry_uuid",
        table_name="registry_appended_documents",
    )
    op.drop_index(
        "ix_registry_appended_documents_user_id",
        table_name="registry_appended_documents",
    )
    op.drop_table("registry_appended_documents")

