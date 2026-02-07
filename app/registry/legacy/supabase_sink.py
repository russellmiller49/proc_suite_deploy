"""Typed Supabase/Postgres upsert helpers."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterable

import psycopg
from psycopg import sql

_TABLE_MAP = {
    "bronchoscopy_procedure": "bronchoscopy_procedure",
    "specimens": "bronchoscopy_specimens",
    "devices": "bronchoscopy_devices",
    "complications": "bronchoscopy_complications",
    "billing_lines": "bronchoscopy_billing_lines",
}


def upsert_bundle(bundle: Dict[str, Any]) -> None:
    if not bundle:
        return
    with _get_conn() as conn, conn.cursor() as cur:
        proc = bundle.get("bronchoscopy_procedure")
        if proc:
            _upsert_record(cur, _TABLE_MAP["bronchoscopy_procedure"], proc)
        for key in ("specimens", "devices", "complications", "billing_lines"):
            for row in bundle.get(key, []) or []:
                _upsert_record(cur, _TABLE_MAP[key], row)
        conn.commit()


@contextmanager
def _get_conn():
    conn_str = os.getenv("SUPABASE_DB_URL")
    if not conn_str:
        raise RuntimeError("SUPABASE_DB_URL missing; copy .env.sample -> .env")
    conn = psycopg.connect(conn_str, autocommit=False)
    try:
        yield conn
    finally:
        conn.close()


def _upsert_record(cur: psycopg.Cursor, table_name: str, payload: Dict[str, Any]):
    record = dict(payload)
    record.setdefault("lineage", {})
    if "external_id" not in record:
        record["external_id"] = _stable_id(record)
    stmt = sql.SQL(
        """
        INSERT INTO {table} (external_id, payload)
        VALUES (%s, %s::jsonb)
        ON CONFLICT (external_id)
        DO UPDATE SET payload = EXCLUDED.payload
        """
    ).format(table=sql.Identifier(table_name))
    cur.execute(stmt, (record["external_id"], json.dumps(record)))


def _stable_id(payload: Dict[str, Any]) -> str:
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest
