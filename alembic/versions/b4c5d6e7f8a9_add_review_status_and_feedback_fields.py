"""Add review status/action and feedback reviewer fields.

This supports the PHI review workflow by:
- Adding a PHI_REVIEWED status to processingstatus enum.
- Adding SCRUBBING_FEEDBACK_APPLIED to auditaction enum.
- Extending scrubbing_feedback with reviewer metadata and updated scrub outputs.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b4c5d6e7f8a9"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def _add_enum_value(enum_name: str, new_value: str) -> None:
    """Safely add an enum value for Postgres; no-op for SQLite."""

    conn = op.get_bind()
    if conn.dialect.name == "postgresql":
        op.execute(sa.text(f"ALTER TYPE {enum_name} ADD VALUE IF NOT EXISTS :value").bindparams(value=new_value))
    # SQLite uses CHECK constraints on strings; no action needed


def upgrade() -> None:
    _add_enum_value("processingstatus", "phi_reviewed")
    _add_enum_value("auditaction", "scrubbing_feedback_applied")

    op.add_column(
        "scrubbing_feedback",
        sa.Column("reviewer_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "scrubbing_feedback",
        sa.Column("reviewer_email", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "scrubbing_feedback",
        sa.Column("reviewer_role", sa.String(length=100), nullable=True),
    )
    op.add_column("scrubbing_feedback", sa.Column("comment", sa.Text(), nullable=True))
    op.add_column("scrubbing_feedback", sa.Column("updated_scrubbed_text", sa.Text(), nullable=True))
    op.add_column("scrubbing_feedback", sa.Column("updated_entity_map", sa.JSON(), nullable=True))


def downgrade() -> None:
    # Enum value removal is not safe in Postgres; note the limitation.
    op.drop_column("scrubbing_feedback", "updated_entity_map")
    op.drop_column("scrubbing_feedback", "updated_scrubbed_text")
    op.drop_column("scrubbing_feedback", "comment")
    op.drop_column("scrubbing_feedback", "reviewer_role")
    op.drop_column("scrubbing_feedback", "reviewer_email")
    op.drop_column("scrubbing_feedback", "reviewer_id")
