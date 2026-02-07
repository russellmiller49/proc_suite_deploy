"""Phase 6 patch application (safe dict->rebuild, no in-place mutation)."""

from __future__ import annotations

import copy

from app.registry.schema import RegistryRecord


class SelfCorrectionApplyError(RuntimeError):
    pass


def apply_patch_to_record(*, record: RegistryRecord, patch: list[dict]) -> RegistryRecord:
    record_dict = record.model_dump()
    patched = copy.deepcopy(record_dict)

    try:
        for idx, op in enumerate(patch):
            _apply_op(patched, op, idx=idx)
        _apply_semantic_shims(patched, patch)
    except Exception as exc:  # noqa: BLE001
        raise SelfCorrectionApplyError(f"Failed applying JSON Patch: {exc}") from exc

    try:
        return RegistryRecord(**patched)
    except Exception as exc:  # noqa: BLE001
        raise SelfCorrectionApplyError(f"Patched record failed validation: {exc}") from exc


def _apply_op(doc: object, op: dict, *, idx: int) -> None:
    verb = op.get("op")
    if verb not in {"add", "replace"}:
        raise SelfCorrectionApplyError(f"patch[{idx}].op='{verb}' is not supported")

    path = op.get("path")
    if not isinstance(path, str) or not path.startswith("/"):
        raise SelfCorrectionApplyError(f"patch[{idx}].path is invalid: {path!r}")

    if "value" not in op:
        raise SelfCorrectionApplyError(f"patch[{idx}] missing required 'value'")
    value = op.get("value")

    tokens = _parse_pointer(path)
    if not tokens:
        raise SelfCorrectionApplyError(f"patch[{idx}].path points to document root, which is forbidden")

    if verb == "replace" and not _pointer_exists(doc, tokens):
        raise SelfCorrectionApplyError(f"patch[{idx}].path does not exist for replace: {path}")

    parent = _traverse(doc, tokens[:-1], create=(verb == "add"))
    _set_child(parent, tokens[-1], value, verb=verb)


def _parse_pointer(path: str) -> list[str]:
    parts = path.split("/")[1:]
    return [_unescape(p) for p in parts]


def _unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _pointer_exists(doc: object, tokens: list[str]) -> bool:
    cur: object = doc
    for token in tokens:
        if isinstance(cur, dict):
            if token not in cur:
                return False
            cur = cur[token]
        elif isinstance(cur, list):
            try:
                index = int(token)
            except ValueError:
                return False
            if index < 0 or index >= len(cur):
                return False
            cur = cur[index]
        else:
            return False
    return True


def _traverse(doc: object, tokens: list[str], *, create: bool) -> object:
    cur: object = doc
    for token in tokens:
        if isinstance(cur, dict):
            nxt = cur.get(token)
            if nxt is None:
                if not create:
                    raise SelfCorrectionApplyError(f"Missing object at '{token}'")
                nxt = {}
                cur[token] = nxt
            cur = nxt
        elif isinstance(cur, list):
            index = int(token)
            if index < 0 or index >= len(cur):
                raise SelfCorrectionApplyError(f"List index out of range at '{token}'")
            cur = cur[index]
        else:
            raise SelfCorrectionApplyError(f"Cannot traverse non-container at '{token}'")
    return cur


def _set_child(parent: object, token: str, value: object, *, verb: str) -> None:
    if isinstance(parent, dict):
        if verb == "replace" and token not in parent:
            raise SelfCorrectionApplyError(f"replace target '{token}' does not exist")
        parent[token] = value
        return

    if isinstance(parent, list):
        if token == "-" and verb == "add":
            parent.append(value)
            return
        index = int(token)
        if verb == "replace":
            if index < 0 or index >= len(parent):
                raise SelfCorrectionApplyError("replace list index out of range")
            parent[index] = value
            return
        if verb == "add":
            if index < 0 or index > len(parent):
                raise SelfCorrectionApplyError("add list index out of range")
            parent.insert(index, value)
            return

    raise SelfCorrectionApplyError(f"Cannot set child on non-container type {type(parent).__name__}")


def _normalize_foreign_body_retrieval_tool(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    canonical = {
        "forceps": "Forceps",
        "basket": "Basket",
        "cryoprobe": "Cryoprobe",
        "snare": "Snare",
        "other": "Other",
    }
    lowered = text.lower()
    for key, out in canonical.items():
        if lowered == key:
            return out
    if "forceps" in lowered:
        return "Forceps"
    if "basket" in lowered:
        return "Basket"
    if "cryo" in lowered:
        return "Cryoprobe"
    if "snare" in lowered:
        return "Snare"
    if lowered in {"unknown", "n/a", "na"}:
        return "Other"
    return "Other"


def _apply_semantic_shims(doc: dict, patch: list[dict]) -> None:
    """Small, safe post-patch shims to keep downstream derivation consistent.

    Self-correction patches are intentionally conservative and often set only
    `.performed=true`. Some downstream CPT derivations are action-gated to avoid
    false positives, so we fill minimal action defaults when the patch intent is
    unambiguous.
    """
    ipc_enabled = any(
        isinstance(op, dict)
        and op.get("path") in {"/pleural_procedures/ipc/performed", "/pleural_procedures/tunneled_pleural_catheter/performed"}
        and op.get("value") is True
        for op in patch
    )
    if ipc_enabled:
        pleural = doc.get("pleural_procedures")
        if not isinstance(pleural, dict):
            pleural = {}
            doc["pleural_procedures"] = pleural
        ipc = pleural.get("ipc")
        if not isinstance(ipc, dict):
            ipc = {}
            pleural["ipc"] = ipc
        if ipc.get("performed") is True and ipc.get("action") in (None, "", "Unknown"):
            ipc["action"] = "Insertion"

    procs = doc.get("procedures_performed")
    if not isinstance(procs, dict):
        return

    foreign_body = procs.get("foreign_body_removal")
    if not isinstance(foreign_body, dict):
        return

    # Legacy judge outputs may use tool/tool_used; canonical schema field is retrieval_tool.
    if "retrieval_tool" not in foreign_body:
        legacy_value = foreign_body.pop("tool_used", None)
        if legacy_value in (None, ""):
            legacy_value = foreign_body.pop("tool", None)
        if legacy_value not in (None, ""):
            foreign_body["retrieval_tool"] = legacy_value

    normalized_tool = _normalize_foreign_body_retrieval_tool(foreign_body.get("retrieval_tool"))
    if normalized_tool is not None:
        foreign_body["retrieval_tool"] = normalized_tool


__all__ = ["apply_patch_to_record", "SelfCorrectionApplyError"]
