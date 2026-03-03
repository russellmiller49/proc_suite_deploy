"""Add case event metadata/cache columns and manual override locks.

Revision ID: c5d6e7f8a9b0
Revises: a4b5c6d7e8f0
Create Date: 2026-02-24
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "c5d6e7f8a9b0"
down_revision = "a4b5c6d7e8f0"
branch_labels = None
depends_on = None


JSONType = postgresql.JSONB().with_variant(sa.JSON(), "sqlite")


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {col["name"] for col in inspector.get_columns(table_name)}


def _index_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {idx["name"] for idx in inspector.get_indexes(table_name)}


def _json_empty_object_server_default() -> sa.TextClause:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        return sa.text("'{}'::jsonb")
    return sa.text("'{}'")


def upgrade() -> None:
    appended_cols = _column_names("registry_appended_documents")
    if "source_modality" not in appended_cols:
        op.add_column("registry_appended_documents", sa.Column("source_modality", sa.Text(), nullable=True))
    if "event_subtype" not in appended_cols:
        op.add_column("registry_appended_documents", sa.Column("event_subtype", sa.Text(), nullable=True))
    if "event_title" not in appended_cols:
        op.add_column("registry_appended_documents", sa.Column("event_title", sa.Text(), nullable=True))
    if "extracted_json" not in appended_cols:
        op.add_column("registry_appended_documents", sa.Column("extracted_json", JSONType, nullable=True))
    if "aggregated_at" not in appended_cols:
        op.add_column(
            "registry_appended_documents",
            sa.Column("aggregated_at", sa.DateTime(timezone=True), nullable=True),
        )
    if "aggregation_version" not in appended_cols:
        op.add_column("registry_appended_documents", sa.Column("aggregation_version", sa.Integer(), nullable=True))

    appended_indexes = _index_names("registry_appended_documents")
    if "ix_registry_appended_documents_registry_uuid_created_at" not in appended_indexes:
        op.create_index(
            "ix_registry_appended_documents_registry_uuid_created_at",
            "registry_appended_documents",
            ["registry_uuid", "created_at"],
            unique=False,
        )
    if "ix_registry_appended_documents_registry_uuid_event_type" not in appended_indexes:
        op.create_index(
            "ix_registry_appended_documents_registry_uuid_event_type",
            "registry_appended_documents",
            ["registry_uuid", "event_type"],
            unique=False,
        )

    case_cols = _column_names("registry_case_records")
    if "manual_overrides" not in case_cols:
        op.add_column(
            "registry_case_records",
            sa.Column(
                "manual_overrides",
                JSONType,
                nullable=False,
                server_default=_json_empty_object_server_default(),
            ),
        )
        op.alter_column("registry_case_records", "manual_overrides", server_default=None)


def downgrade() -> None:
    case_cols = _column_names("registry_case_records")
    if "manual_overrides" in case_cols:
        op.drop_column("registry_case_records", "manual_overrides")

    appended_indexes = _index_names("registry_appended_documents")
    if "ix_registry_appended_documents_registry_uuid_event_type" in appended_indexes:
        op.drop_index(
            "ix_registry_appended_documents_registry_uuid_event_type",
            table_name="registry_appended_documents",
        )
    if "ix_registry_appended_documents_registry_uuid_created_at" in appended_indexes:
        op.drop_index(
            "ix_registry_appended_documents_registry_uuid_created_at",
            table_name="registry_appended_documents",
        )

    appended_cols = _column_names("registry_appended_documents")
    if "aggregation_version" in appended_cols:
        op.drop_column("registry_appended_documents", "aggregation_version")
    if "aggregated_at" in appended_cols:
        op.drop_column("registry_appended_documents", "aggregated_at")
    if "extracted_json" in appended_cols:
        op.drop_column("registry_appended_documents", "extracted_json")
    if "event_title" in appended_cols:
        op.drop_column("registry_appended_documents", "event_title")
    if "event_subtype" in appended_cols:
        op.drop_column("registry_appended_documents", "event_subtype")
    if "source_modality" in appended_cols:
        op.drop_column("registry_appended_documents", "source_modality")
