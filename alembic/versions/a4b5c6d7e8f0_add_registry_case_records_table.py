"""Add registry_case_records table for canonical case-level registry state."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "a4b5c6d7e8f0"
down_revision = "f9a1b2c3d4e5"
branch_labels = None
depends_on = None


UUIDType = postgresql.UUID(as_uuid=True).with_variant(sa.String(length=36), "sqlite")
JSONType = postgresql.JSONB().with_variant(sa.JSON(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "registry_case_records",
        sa.Column("registry_uuid", UUIDType, nullable=False),
        sa.Column("registry_json", JSONType, nullable=False),
        sa.Column("schema_version", sa.String(length=32), nullable=False, server_default="v3"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("source_run_id", UUIDType, nullable=True),
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
        sa.PrimaryKeyConstraint("registry_uuid"),
    )
    op.create_index(
        "ix_registry_case_records_source_run_id",
        "registry_case_records",
        ["source_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_registry_case_records_created_at",
        "registry_case_records",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_registry_case_records_updated_at",
        "registry_case_records",
        ["updated_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_registry_case_records_updated_at", table_name="registry_case_records")
    op.drop_index("ix_registry_case_records_created_at", table_name="registry_case_records")
    op.drop_index("ix_registry_case_records_source_run_id", table_name="registry_case_records")
    op.drop_table("registry_case_records")
