"""Add event_type and relative_day_offset to registry_appended_documents."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "f9a1b2c3d4e5"
down_revision = "e8f9a0b1c2d3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "registry_appended_documents",
        sa.Column(
            "event_type",
            sa.String(length=64),
            nullable=False,
            server_default="pathology",
        ),
    )
    op.add_column(
        "registry_appended_documents",
        sa.Column("relative_day_offset", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_registry_appended_documents_event_type",
        "registry_appended_documents",
        ["event_type"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_registry_appended_documents_event_type",
        table_name="registry_appended_documents",
    )
    op.drop_column("registry_appended_documents", "relative_day_offset")
    op.drop_column("registry_appended_documents", "event_type")
