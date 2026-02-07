"""Add registry_runs table for scrubbed-only registry run persistence.

This table stores:
- scrubbed note text (never raw PHI)
- raw unified pipeline response JSON
- optional UI corrections payloads
- a single feedback submission per run
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "c1d2e3f4a5b6"
down_revision = "b4c5d6e7f8a9"
branch_labels = None
depends_on = None


UUIDType = postgresql.UUID(as_uuid=True).with_variant(sa.String(length=36), "sqlite")
JSONType = postgresql.JSONB().with_variant(sa.JSON(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "registry_runs",
        sa.Column("id", UUIDType, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("submitter_name", sa.String(length=255), nullable=True),
        sa.Column("note_text", sa.Text(), nullable=False),
        sa.Column("note_sha256", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.String(length=32), nullable=False),
        sa.Column("pipeline_config", JSONType, nullable=False),
        sa.Column("raw_response_json", JSONType, nullable=False),
        sa.Column("corrected_response_json", JSONType, nullable=True),
        sa.Column("edited_tables_json", JSONType, nullable=True),
        sa.Column("correction_editor_name", sa.String(length=255), nullable=True),
        sa.Column("corrected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("feedback_reviewer_name", sa.String(length=255), nullable=True),
        sa.Column("feedback_rating", sa.Integer(), nullable=True),
        sa.Column("feedback_comment", sa.Text(), nullable=True),
        sa.Column("feedback_submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("needs_manual_review", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("review_status", sa.String(length=50), server_default="new", nullable=False),
        sa.Column("kb_version", sa.String(length=64), nullable=True),
        sa.Column("kb_hash", sa.String(length=64), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(op.f("ix_registry_runs_created_at"), "registry_runs", ["created_at"], unique=False)
    op.create_index(op.f("ix_registry_runs_note_sha256"), "registry_runs", ["note_sha256"], unique=False)
    op.create_index(op.f("ix_registry_runs_submitter_name"), "registry_runs", ["submitter_name"], unique=False)
    op.create_index(op.f("ix_registry_runs_needs_manual_review"), "registry_runs", ["needs_manual_review"], unique=False)
    op.create_index(op.f("ix_registry_runs_review_status"), "registry_runs", ["review_status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_registry_runs_review_status"), table_name="registry_runs")
    op.drop_index(op.f("ix_registry_runs_needs_manual_review"), table_name="registry_runs")
    op.drop_index(op.f("ix_registry_runs_submitter_name"), table_name="registry_runs")
    op.drop_index(op.f("ix_registry_runs_note_sha256"), table_name="registry_runs")
    op.drop_index(op.f("ix_registry_runs_created_at"), table_name="registry_runs")
    op.drop_table("registry_runs")

