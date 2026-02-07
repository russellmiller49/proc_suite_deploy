from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen


def _read_terms_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    terms: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        terms.append(line)
    return terms


def _try_fetch_openfda_terms(limit: int, timeout_s: float) -> list[str]:
    # Best-effort: keep small and safe. OpenFDA supports aggregation counts.
    url = (
        "https://api.fda.gov/device/510k.json?"
        "count=device_name.exact&limit="
        + str(limit)
    )
    try:
        with urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 - explicit small fetch
            payload = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError):
        return []

    out: list[str] = []
    for row in payload.get("results", []) or []:
        term = str(row.get("term", "")).strip()
        if term:
            out.append(term)
    return out


_SPACE_RE = re.compile(r"\s+")
_TRIM_RE = re.compile(r"^[\W_]+|[\W_]+$")


def _normalize_term(term: str) -> str | None:
    t = term.strip().lower()
    if not t:
        return None
    t = _SPACE_RE.sub(" ", t)
    t = _TRIM_RE.sub("", t)
    if len(t) < 2:
        return None
    return t


def _build_trie(terms: Iterable[str]) -> dict:
    root: dict = {}
    for term in terms:
        node = root
        for ch in term:
            node = node.setdefault(ch, {})
        node["$"] = 1
    return root


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Build a compact allowlist trie JSON for the client-side PHI redactor."
    )
    parser.add_argument(
        "--terms-file",
        default=str(repo_root / "data" / "allowlist_terms.txt"),
        help="Primary repo-local allowlist terms file (one term per line).",
    )
    parser.add_argument(
        "--private-umls-file",
        default=str(repo_root / "data" / "private" / "umls_abbreviations.txt"),
        help="Optional local-only file (NOT committed).",
    )
    parser.add_argument(
        "--out",
        default=str(
            repo_root
            / "modules"
            / "api"
            / "static"
            / "phi_redactor"
            / "allowlist_trie.json"
        ),
        help="Output path for allowlist_trie.json.",
    )
    parser.add_argument(
        "--max-term-len",
        type=int,
        default=48,
        help="Maximum term length to include (trims overly long entries).",
    )
    parser.add_argument(
        "--openfda",
        action="store_true",
        help="Best-effort: fetch a small set of OpenFDA device terms.",
    )
    parser.add_argument(
        "--openfda-limit",
        type=int,
        default=500,
        help="Max number of OpenFDA device terms to include (if --openfda).",
    )
    args = parser.parse_args(argv)

    terms_file = Path(args.terms_file)
    private_umls_file = Path(args.private_umls_file)
    out_path = Path(args.out)
    max_len = int(args.max_term_len)

    raw_terms: list[str] = []
    raw_terms.extend(_read_terms_file(terms_file))
    raw_terms.extend(_read_terms_file(private_umls_file))

    if args.openfda and os.getenv("PROCSUITE_SKIP_NETWORK", "0") not in ("1", "true", "yes"):
        raw_terms.extend(_try_fetch_openfda_terms(args.openfda_limit, timeout_s=5.0))

    normalized: list[str] = []
    seen: set[str] = set()
    for t in raw_terms:
        nt = _normalize_term(t)
        if not nt:
            continue
        if len(nt) > max_len:
            continue
        if nt in seen:
            continue
        seen.add(nt)
        normalized.append(nt)

    normalized.sort()

    trie = _build_trie(normalized)
    payload = {"v": 1, "max_term_len": max_len, "count": len(normalized), "trie": trie}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    print(f"Wrote {out_path} ({len(normalized)} terms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

