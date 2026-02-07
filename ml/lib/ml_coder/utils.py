"""Shared helpers for CPT data cleaning and utilities."""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd

__all__ = ["clean_cpt_codes", "join_codes"]


def clean_cpt_codes(raw_val: object) -> list[str]:
    """
    Normalize CPT codes from concatenated digit strings into 5-digit chunks.

    Examples
    --------
    "316,273,165,431,628" -> ["31627", "31654", "31628"]
    "3,260,932,650"       -> ["32609", "32650"]
    """

    if pd.isna(raw_val):
        return []

    clean_digits = str(raw_val).replace(",", "").replace(" ", "").replace(".", "")
    codes = [clean_digits[i : i + 5] for i in range(0, len(clean_digits), 5)]
    return [c for c in codes if len(c) == 5]


def join_codes(codes: Iterable[str]) -> str:
    """Join a collection of CPT codes into a comma-separated string."""

    return ",".join(code.strip() for code in codes if code and str(code).strip())
