"""Alembic environment configured for PHI vault tables.

Default URL uses local SQLite (`sqlite:///./phi_demo.db`) for synthetic/demo data.
Override with `DATABASE_URL` for Postgres/Supabase in HIPAA-ready deployments.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.phi.db import metadata as target_metadata
# Import models so metadata is populated
import app.phi.models  # noqa: F401
import app.registry_store.models  # noqa: F401


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def get_url() -> str:
    """Resolve database URL with a safe default for local runs."""
    default_url = config.get_main_option("sqlalchemy.url")
    return os.getenv("DATABASE_URL", default_url)


def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section)
    if configuration is None:
        configuration = {}
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
