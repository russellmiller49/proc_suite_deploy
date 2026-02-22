from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from filelock import FileLock

from config.settings import UmlsSettings


_logger = logging.getLogger(__name__)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3://bucket/key URI, got: {uri!r}")
    bucket = parsed.netloc.strip()
    key = (parsed.path or "").lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Expected s3://bucket/key URI, got: {uri!r}")
    return bucket, key


def download_s3_to_path(bucket: str, key: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest_path.name}.",
        suffix=".tmp",
        dir=str(dest_path.parent),
        text=False,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        import boto3

        boto3.client("s3").download_file(bucket, key, str(tmp_path))
        os.replace(tmp_path, dest_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        raise


def ensure_ip_umls_map_path(settings: UmlsSettings) -> Path:
    if settings.ip_umls_map_local_path is not None:
        path = settings.ip_umls_map_local_path
        if not path.exists():
            raise FileNotFoundError(f"UMLS local map not found: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"UMLS local map must be a file: {path}")
        return path

    if settings.ip_umls_map_s3_uri:
        bucket, key = parse_s3_uri(settings.ip_umls_map_s3_uri)
        dest = settings.ip_umls_map_cache_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(f"{dest}.lock")
        with lock:
            if dest.exists() and not settings.force_refresh:
                return dest
            try:
                download_s3_to_path(bucket=bucket, key=key, dest_path=dest)
                return dest
            except Exception as exc:  # noqa: BLE001
                if dest.exists():
                    _logger.warning("UMLS S3 download failed; using existing cache: %s", exc)
                    return dest
                raise

    repo_relative = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "ip_umls_map.json"
    if repo_relative.exists():
        return repo_relative

    raise FileNotFoundError("UMLS map unavailable (no local file, no S3 URI, and no repo fallback present)")


def normalize_strict(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().casefold())


def normalize_loose(text: str) -> str:
    lowered = (text or "").strip().casefold()
    lowered = re.sub(r"[^0-9a-z]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


class DistilledUmlsStore:
    def __init__(self, payload: dict[str, Any]) -> None:
        concepts = payload.get("concepts", {}) or {}
        term_index = payload.get("term_index", {}) or {}
        if not isinstance(concepts, dict):
            concepts = {}
        if not isinstance(term_index, dict):
            term_index = {}

        self.concepts: dict[str, dict[str, Any]] = {
            str(cui): (concept if isinstance(concept, dict) else {})
            for cui, concept in concepts.items()
            if str(cui)
        }

        self.term_index_strict: dict[str, list[str]] = {}
        for raw_term, cuis in term_index.items():
            term = normalize_strict(str(raw_term))
            if not term:
                continue
            if not isinstance(cuis, list):
                continue
            cleaned_cuis = [str(cui) for cui in cuis if str(cui)]
            if not cleaned_cuis:
                continue
            existing = self.term_index_strict.setdefault(term, [])
            for cui in cleaned_cuis:
                if cui not in existing:
                    existing.append(cui)

        self.term_index_loose: dict[str, list[str]] = {}
        for strict_term, cuis in self.term_index_strict.items():
            loose_term = normalize_loose(strict_term)
            if not loose_term:
                continue
            existing = self.term_index_loose.setdefault(loose_term, [])
            for cui in cuis:
                if cui not in existing:
                    existing.append(cui)

        self.sorted_terms: list[str] = sorted(self.term_index_strict.keys())
        self.semtype_legend: dict[str, str] | None = payload.get("semtype_legend")
        self.meta: dict[str, Any] | None = payload.get("meta")

    def _choose_cui(self, cuis: list[str], category: str | None) -> str | None:
        if not cuis:
            return None
        if category:
            for candidate in cuis:
                concept = self.concepts.get(candidate, {})
                categories = concept.get("categories", []) or []
                if isinstance(categories, list) and category in categories:
                    return candidate
        return cuis[0]

    def match(self, term: str, category: str | None = None) -> dict[str, Any] | None:
        raw = (term or "").strip()
        if not raw:
            return None

        key_strict = normalize_strict(raw)
        matched_term = None
        match_type = None
        cuis: list[str] | None = None

        strict_hit = self.term_index_strict.get(key_strict)
        if strict_hit:
            matched_term = key_strict
            match_type = "exact"
            cuis = list(strict_hit)
        else:
            key_loose = normalize_loose(raw)
            loose_hit = self.term_index_loose.get(key_loose)
            if loose_hit:
                matched_term = key_loose
                match_type = "loose"
                cuis = list(loose_hit)

        if not matched_term or not match_type or not cuis:
            return None

        chosen_cui = self._choose_cui(cuis, category)
        if not chosen_cui:
            return None

        concept = self.concepts.get(chosen_cui, {}) or {}
        preferred_name = concept.get("preferred_name") or concept.get("name") or ""
        categories = concept.get("categories", []) or []
        semtypes = concept.get("semtypes", []) or []
        if not isinstance(categories, list):
            categories = []
        if not isinstance(semtypes, list):
            semtypes = []

        return {
            "raw_text": raw,
            "matched_term": matched_term,
            "match_type": match_type,
            "cuis": cuis,
            "chosen_cui": chosen_cui,
            "preferred_name": str(preferred_name),
            "categories": categories,
            "semtypes": semtypes,
        }

    def suggest(self, prefix: str, category: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        raw_prefix = (prefix or "").strip()
        if not raw_prefix or limit <= 0:
            return []

        prefix_norm = normalize_strict(raw_prefix)
        if not prefix_norm:
            return []

        start_idx = bisect_left(self.sorted_terms, prefix_norm)
        results: list[dict[str, Any]] = []

        for term in self.sorted_terms[start_idx:]:
            if len(results) >= limit:
                break
            if not term.startswith(prefix_norm):
                break
            cuis = self.term_index_strict.get(term) or []
            chosen_cui = self._choose_cui(cuis, category)
            if not chosen_cui:
                continue
            concept = self.concepts.get(chosen_cui, {}) or {}
            categories = concept.get("categories", []) or []
            if not isinstance(categories, list):
                categories = []
            if category and category not in categories:
                continue

            results.append(
                {
                    "term": term,
                    "preferred_name": str(concept.get("preferred_name") or concept.get("name") or ""),
                    "cui": chosen_cui,
                    "categories": categories,
                }
            )

        return results


@lru_cache(maxsize=1)
def get_ip_umls_store() -> DistilledUmlsStore:
    settings = UmlsSettings()
    path = ensure_ip_umls_map_path(settings)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"UMLS map payload must be a JSON object: {path}")
    return DistilledUmlsStore(payload)


__all__ = [
    "DistilledUmlsStore",
    "download_s3_to_path",
    "ensure_ip_umls_map_path",
    "get_ip_umls_store",
    "normalize_loose",
    "normalize_strict",
    "parse_s3_uri",
]

