"""Add PHI vault, procedure_data, audit_log, scrubbing_feedback tables.

This migration introduces the PHI vault schema for synthetic/demo data.
Future HIPAA deployments can point Alembic at a compliant vault backend via
`DATABASE_URL` without exposing raw PHI to non-vault tables.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = None
branch_labels = None
depends_on = None


UUIDType = postgresql.UUID(as_uuid=True).with_variant(sa.String(length=36), "sqlite")
JSONType = postgresql.JSONB().with_variant(sa.JSON(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "phi_vault",
        sa.Column("id", UUIDType, nullable=False),
        sa.Column("encrypted_data", sa.LargeBinary(), nullable=False),
        sa.Column("data_hash", sa.String(length=64), nullable=False),
        sa.Column("encryption_algorithm", sa.String(length=50), server_default="FERNET", nullable=True),
        sa.Column("key_version", sa.Integer(), server_default="1", nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("false"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    status_enum = sa.Enum(
        "pending_review",
        "phi_confirmed",
        "processing",
        "completed",
        "failed",
        name="processingstatus",
    )

    op.create_table(
        "procedure_data",
        sa.Column("id", UUIDType, nullable=False),
        sa.Column("phi_vault_id", UUIDType, nullable=False),
        sa.Column("scrubbed_text", sa.Text(), nullable=False),
        sa.Column("original_text_hash", sa.String(length=64), nullable=False),
        sa.Column("entity_map", JSONType, nullable=False),
        sa.Column("status", status_enum, nullable=True),
        sa.Column("coding_results", JSONType, nullable=True),
        sa.Column("document_type", sa.String(length=100), nullable=True),
        sa.Column("specialty", sa.String(length=100), nullable=True),
        sa.Column("submitted_by", sa.String(length=255), nullable=False),
        sa.Column("reviewed_by", sa.String(length=255), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("processed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["phi_vault_id"], ["phi_vault.id"], ondelete=None),
        sa.PrimaryKeyConstraint("id"),
    )

    audit_action_enum = sa.Enum(
        "phi_created",
        "phi_accessed",
        "phi_decrypted",
        "review_started",
        "entity_confirmed",
        "entity_unflagged",
        "entity_added",
        "review_completed",
        "llm_called",
        "reidentified",
        name="auditaction",
    )

    op.create_table(
        "audit_log",
        sa.Column("id", UUIDType, nullable=False),
        sa.Column("phi_vault_id", UUIDType, nullable=True),
        sa.Column("procedure_data_id", UUIDType, nullable=True),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("user_email", sa.String(length=255), nullable=True),
        sa.Column("user_role", sa.String(length=100), nullable=True),
        sa.Column("action", audit_action_enum, nullable=False),
        sa.Column("action_detail", sa.Text(), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("request_id", sa.String(length=255), nullable=True),
        sa.Column("metadata", JSONType, nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["phi_vault_id"], ["phi_vault.id"], ondelete=None),
        sa.ForeignKeyConstraint(["procedure_data_id"], ["procedure_data.id"], ondelete=None),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "scrubbing_feedback",
        sa.Column("id", UUIDType, nullable=False),
        sa.Column("procedure_data_id", UUIDType, nullable=True),
        sa.Column("presidio_entities", JSONType, nullable=False),
        sa.Column("confirmed_entities", JSONType, nullable=False),
        sa.Column("false_positives", JSONType, nullable=True),
        sa.Column("false_negatives", JSONType, nullable=True),
        sa.Column("true_positives", sa.Integer(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1_score", sa.Float(), nullable=True),
        sa.Column("document_type", sa.String(length=100), nullable=True),
        sa.Column("specialty", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["procedure_data_id"], ["procedure_data.id"], ondelete=None),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("scrubbing_feedback")
    op.drop_table("audit_log")
    op.drop_table("procedure_data")
    op.drop_table("phi_vault")
    op.execute("DROP TYPE IF EXISTS auditaction")
    op.execute("DROP TYPE IF EXISTS processingstatus")

