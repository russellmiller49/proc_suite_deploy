"""IO helpers for teacher-student distillation artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TeacherLogits:
    ids: list[str]
    logits: np.ndarray  # shape [N, L]
    label_fields: list[str]  # length L, ordered


def load_label_fields_json(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) and x.strip() for x in data):
        raise ValueError(f"Invalid label_fields JSON: {path}")
    return [str(x) for x in data]


def validate_label_fields_match(teacher_fields: list[str], student_fields: list[str]) -> None:
    if teacher_fields != student_fields:
        raise ValueError(
            "Label order mismatch between teacher and student.\n"
            f"teacher={teacher_fields}\n"
            f"student={student_fields}\n"
        )


def load_teacher_logits_npz(path: str | Path) -> TeacherLogits:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "ids" not in data or "logits" not in data:
            raise ValueError(f"Missing required arrays in {path} (need: ids, logits)")
        ids = data["ids"]
        logits = data["logits"]
        label_fields = data["label_fields"] if "label_fields" in data else None

    ids_list = [str(x) for x in ids.tolist()] if hasattr(ids, "tolist") else [str(x) for x in ids]
    logits_arr = np.asarray(logits)
    if logits_arr.ndim != 2:
        raise ValueError(f"Expected logits to be rank-2 [N,L], got shape={logits_arr.shape} in {path}")
    if label_fields is None:
        raise ValueError(f"Missing label_fields in {path}")

    label_fields_list = (
        [str(x) for x in label_fields.tolist()] if hasattr(label_fields, "tolist") else [str(x) for x in label_fields]
    )
    if len(label_fields_list) != int(logits_arr.shape[1]):
        raise ValueError(
            f"label_fields length {len(label_fields_list)} != logits L {int(logits_arr.shape[1])} in {path}"
        )
    if len(ids_list) != int(logits_arr.shape[0]):
        raise ValueError(f"ids length {len(ids_list)} != logits N {int(logits_arr.shape[0])} in {path}")

    return TeacherLogits(ids=ids_list, logits=logits_arr, label_fields=label_fields_list)


def align_teacher_logits(
    student_ids: list[str],
    teacher: TeacherLogits,
    *,
    dtype: Any = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Align teacher logits to a list of student IDs.

    Returns:
        aligned_logits: [N, L] (zeros where missing)
        has_teacher: [N] mask (0/1)
    """
    id_to_idx = {str(i): n for n, i in enumerate(teacher.ids)}
    n = len(student_ids)
    l = int(teacher.logits.shape[1])
    aligned = np.zeros((n, l), dtype=dtype)
    mask = np.zeros((n,), dtype=np.float32)
    for i, sid in enumerate(student_ids):
        t_idx = id_to_idx.get(str(sid))
        if t_idx is None:
            continue
        aligned[i] = teacher.logits[t_idx].astype(dtype, copy=False)
        mask[i] = 1.0
    return aligned, mask

