"""JSON Patch (RFC6902) helpers for ProcedureBundle updates."""

from __future__ import annotations

import copy
from typing import Any

from proc_schemas.clinical.common import ProcedureBundle


class BundleJsonPatchError(ValueError):
    """Raised when a bundle JSON patch operation is invalid."""


def apply_bundle_json_patch(bundle: ProcedureBundle, patch: list[dict[str, Any]]) -> ProcedureBundle:
    payload = bundle.model_dump(exclude_none=False)
    patched = copy.deepcopy(payload)

    try:
        for idx, op in enumerate(patch):
            _apply_op(patched, op, idx=idx)
    except BundleJsonPatchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise BundleJsonPatchError(f"Failed applying JSON Patch: {exc}") from exc

    try:
        return ProcedureBundle.model_validate(patched)
    except Exception as exc:  # noqa: BLE001
        raise BundleJsonPatchError(f"Patched bundle failed validation: {exc}") from exc


def _apply_op(doc: object, op: dict[str, Any], *, idx: int) -> None:
    verb = op.get("op")
    if verb not in {"add", "replace", "remove"}:
        raise BundleJsonPatchError(f"patch[{idx}].op='{verb}' is not supported")

    path = op.get("path")
    if not isinstance(path, str) or not path.startswith("/"):
        raise BundleJsonPatchError(f"patch[{idx}].path is invalid: {path!r}")

    tokens = _parse_pointer(path)
    if not tokens:
        raise BundleJsonPatchError(f"patch[{idx}] cannot target document root")

    if verb in {"replace", "remove"} and not _pointer_exists(doc, tokens):
        raise BundleJsonPatchError(f"patch[{idx}] path does not exist for {verb}: {path}")

    parent = _traverse(doc, tokens[:-1], create=(verb == "add"))
    token = tokens[-1]

    if verb == "remove":
        _remove_child(parent, token)
        return

    if "value" not in op:
        raise BundleJsonPatchError(f"patch[{idx}] missing required 'value'")
    _set_child(parent, token, op.get("value"), verb=verb)


def _parse_pointer(path: str) -> list[str]:
    parts = path.split("/")[1:]
    return [_unescape(part) for part in parts]


def _unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _pointer_exists(doc: object, tokens: list[str]) -> bool:
    current: object = doc
    for token in tokens:
        if isinstance(current, dict):
            if token not in current:
                return False
            current = current[token]
        elif isinstance(current, list):
            if token == "-":
                return False
            try:
                index = int(token)
            except Exception:
                return False
            if index < 0 or index >= len(current):
                return False
            current = current[index]
        else:
            return False
    return True


def _traverse(doc: object, tokens: list[str], *, create: bool) -> object:
    current: object = doc
    for token in tokens:
        if isinstance(current, dict):
            child = current.get(token)
            if child is None:
                if not create:
                    raise BundleJsonPatchError(f"Missing object at '{token}'")
                child = {}
                current[token] = child
            current = child
            continue
        if isinstance(current, list):
            if token == "-":
                raise BundleJsonPatchError("'-' token is only valid in the final path segment for add")
            try:
                index = int(token)
            except Exception as exc:
                raise BundleJsonPatchError(f"Invalid list index '{token}'") from exc
            if index < 0 or index >= len(current):
                raise BundleJsonPatchError(f"List index out of range at '{token}'")
            current = current[index]
            continue
        raise BundleJsonPatchError(f"Cannot traverse non-container at '{token}'")
    return current


def _set_child(parent: object, token: str, value: object, *, verb: str) -> None:
    if isinstance(parent, dict):
        if verb == "replace" and token not in parent:
            raise BundleJsonPatchError(f"replace target '{token}' does not exist")
        parent[token] = value
        return

    if isinstance(parent, list):
        if token == "-" and verb == "add":
            parent.append(value)
            return
        try:
            index = int(token)
        except Exception as exc:
            raise BundleJsonPatchError(f"Invalid list index '{token}'") from exc

        if verb == "replace":
            if index < 0 or index >= len(parent):
                raise BundleJsonPatchError("replace list index out of range")
            parent[index] = value
            return

        if verb == "add":
            if index < 0 or index > len(parent):
                raise BundleJsonPatchError("add list index out of range")
            parent.insert(index, value)
            return

    raise BundleJsonPatchError(f"Cannot set child on non-container type {type(parent).__name__}")


def _remove_child(parent: object, token: str) -> None:
    if isinstance(parent, dict):
        if token not in parent:
            raise BundleJsonPatchError(f"remove target '{token}' does not exist")
        del parent[token]
        return

    if isinstance(parent, list):
        try:
            index = int(token)
        except Exception as exc:
            raise BundleJsonPatchError(f"Invalid list index '{token}'") from exc
        if index < 0 or index >= len(parent):
            raise BundleJsonPatchError("remove list index out of range")
        del parent[index]
        return

    raise BundleJsonPatchError(f"Cannot remove child from non-container type {type(parent).__name__}")


__all__ = ["BundleJsonPatchError", "apply_bundle_json_patch"]
