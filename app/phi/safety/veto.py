"""
PHI Redactor Veto Module - Post-processing to prevent false positives.

This module implements the "drop whole entity on veto hit" pattern:
- If ANY protected term is found inside a predicted entity span, the ENTIRE entity is dropped
- This prevents dangling leftover tokens after IOB repair (e.g., "a" from "a chartis")

Key changes (Dec 2025):
- Drop-whole-entity logic: veto clears entire contiguous entity spans, not just matched tokens
- Atomic numeric spans: CPT/ICD-like codes are treated as units
- Enhanced CBCT/coding context detection for numeric allowlist
"""

import re
from typing import List, Tuple

from app.phi.safety.protected_terms import (
    LN_CONTEXT_WORDS,
    is_ln_station,
    is_protected_anatomy_phrase,
    is_protected_device,
    normalize,
    reconstruct_wordpiece,
)

# Stopwords that should never be standalone entity spans
STOPWORDS = {
    "a", "an", "the", "of", "in", "and", "with", "to", "for", "or", "by", "at",
    "is", "was", "are", "were", "be", "been", "being", "has", "had", "have",
    "did", "does", "do", "will", "would", "could", "should", "may", "might",
    "on", "as", "from", "but", "not", "no", "yes", "so", "if", "then",
}

ID_LABELS = {"ID", "MRN", "SSN"}
# "ID" is frequently used by models for CPT-like numeric codes and even device tokens;
# do not exempt it from veto logic. Only preserve explicitly sensitive IDs.
SENSITIVE_ID_LABELS = {"MRN", "SSN"}

AMBIGUOUS_DEVICE_TERMS = {"cook", "king", "edwards", "young", "wang", "mark"}
AMBIGUOUS_DEVICE_CONTEXT_PATTERNS = {
    "cook": re.compile(r"\bcook\s+(medical|catheter|guide|stent)\b", re.IGNORECASE),
    "king": re.compile(r"\bking\s+(airway|tube|system)\b", re.IGNORECASE),
    "edwards": re.compile(r"\bedwards\s+(lifesciences|valve)\b", re.IGNORECASE),
    "wang": re.compile(r"\bwang\s+(needle|aspirat)\b", re.IGNORECASE),
}
AMBIGUOUS_DEVICE_NAME_ONLY = {"young", "mark"}
AMBIGUOUS_CONTEXT_WINDOW = 4

CPT_CONTEXT_WORDS = {
    "cpt",
    "code",
    "codes",
    "billing",
    "submitted",
    "justification",
    "rvu",
    "coding",
    "radiology",
    "guidance",
    "ct",
    "cbct",
    "fluoro",
    "fluoroscopy",
    "localization",
    "rationale",
}
CPT_PUNCT_TOKENS = {",", ";", ":", "/", "(", ")", "[", "]"}
UNIT_TOKENS = {"l", "liter", "liters", "ml", "cc"}
VOLUME_VERBS = {"drained", "output", "removed"}


def _normalize_token(token: str) -> str:
    if token.startswith("##"):
        token = token[2:]
    return normalize(token)


def _normalize_label(label: str) -> str:
    label = (label or "").upper()
    if label.startswith(("B-", "I-")) and "-" in label:
        _, label = label.split("-", 1)
    return label


def _is_id_label(label: str) -> bool:
    return _normalize_label(label) in ID_LABELS


def _is_sensitive_id_label(label: str) -> bool:
    return _normalize_label(label) in SENSITIVE_ID_LABELS


def _extract_entity_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    """Extract contiguous entity spans from BIO tags.

    Returns list of (start_idx, end_idx_inclusive, entity_type).
    A span is B-X followed by any number of I-X (same X).
    """
    spans: List[Tuple[int, int, str]] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if not tag or tag == "O" or "-" not in tag:
            i += 1
            continue
        prefix, label = tag.split("-", 1)
        if prefix != "B":
            # Orphan I- tag, treat as start of span
            start = i
            end = i
            while end + 1 < len(tags):
                next_tag = tags[end + 1]
                if next_tag == f"I-{label}":
                    end += 1
                else:
                    break
            spans.append((start, end, label))
            i = end + 1
            continue
        # B- tag starts a span
        start = i
        end = i
        while end + 1 < len(tags):
            next_tag = tags[end + 1]
            if next_tag == f"I-{label}":
                end += 1
            else:
                break
        spans.append((start, end, label))
        i = end + 1
    return spans


def _span_has_any_label(tags: List[str], start: int, end: int, labels: set[str]) -> bool:
    for i in range(start, end + 1):
        tag = tags[i]
        if not tag or tag == "O" or "-" not in tag:
            continue
        _, tag_label = tag.split("-", 1)
        if tag_label.upper() in labels:
            return True
    return False


def _reconstruct_span_text(tokens: List[str], start: int, end: int) -> str:
    """Reconstruct surface text from a span of tokens, handling wordpieces."""
    result = []
    for i in range(start, end + 1):
        tok = tokens[i]
        if tok.startswith("##"):
            if result:
                result[-1] += tok[2:]
            else:
                result.append(tok[2:])
        else:
            result.append(tok)
    return " ".join(result)


def _get_context_text(tokens: List[str], start: int, end: int, window: int) -> str:
    ctx_start = max(0, start - window)
    ctx_end = min(len(tokens) - 1, end + window)
    return _reconstruct_span_text(tokens, ctx_start, ctx_end)


def _is_protected_device_with_context(norm_word: str, tokens: List[str], start: int, end: int) -> bool:
    if not is_protected_device(norm_word):
        return False
    if norm_word not in AMBIGUOUS_DEVICE_TERMS:
        return True
    if norm_word in AMBIGUOUS_DEVICE_NAME_ONLY:
        return False
    pattern = AMBIGUOUS_DEVICE_CONTEXT_PATTERNS.get(norm_word)
    if not pattern:
        return False
    context = normalize(_get_context_text(tokens, start, end, AMBIGUOUS_CONTEXT_WINDOW))
    return bool(pattern.search(context))


def _is_protected_in_span(tokens: List[str], start: int, end: int) -> bool:
    """Check if any reconstructed word in the span matches a protected term."""
    idx = start
    while idx <= end:
        word, word_end = reconstruct_wordpiece(tokens, idx)
        word_end = min(word_end, end)  # Don't extend beyond span
        norm_word = normalize(word)
        if _is_protected_device_with_context(norm_word, tokens, idx, word_end) or is_protected_anatomy_phrase(norm_word):
            return True
        idx = word_end + 1
    return False


def _is_stopword_only_span(tokens: List[str], start: int, end: int) -> bool:
    """Check if the span consists only of stopwords or very short tokens."""
    text = _reconstruct_span_text(tokens, start, end)
    norm = normalize(text)
    # Single word that's a stopword
    if norm in STOPWORDS:
        return True
    # Very short span (< 2 chars after normalization)
    if len(norm) < 2:
        if norm == "o":
            if "'" in text or _next_token_is_apostrophe(tokens, end):
                return False
        return True
    # Check if all words are stopwords
    words = norm.split()
    if all(w in STOPWORDS for w in words):
        return True
    return False


def _is_punctuation_token(token: str) -> bool:
    """Check if token is purely punctuation."""
    return all(c in "()[]{},.;:!?/\\-_\"'" for c in token)


def _next_token_is_apostrophe(tokens: List[str], end: int) -> bool:
    if end + 1 >= len(tokens):
        return False
    token = tokens[end + 1]
    return token == "'" or token.startswith("##'") or token.startswith("'")


def _span_starts_with_punct(tokens: List[str], start: int, label: str) -> bool:
    """Check if span starts with punctuation token."""
    if start < len(tokens):
        token = tokens[start]
        if not _is_punctuation_token(token):
            return False
        if token == "(" and label in ("CONTACT", "PHONE"):
            return False
        return True
    return False


def _span_is_all_punct(tokens: List[str], start: int, end: int) -> bool:
    """Check if span consists entirely of punctuation tokens.

    This catches cases like standalone "(", ",", ")" being tagged as entities.
    These are NEVER valid PHI and should always be vetoed.
    """
    for i in range(start, end + 1):
        if not _is_punctuation_token(tokens[i]):
            return False
    return True


def _is_stable_cpt_split(tokens: List[str], i: int) -> str | None:
    if i + 1 >= len(tokens):
        return None
    if tokens[i].isdigit() and len(tokens[i]) == 3 and tokens[i + 1].startswith("##"):
        suffix = tokens[i + 1][2:]
        if suffix.isdigit() and len(suffix) == 2:
            return tokens[i] + suffix
    return None


def _reconstruct_numeric_code(tokens: List[str], start: int) -> Tuple[str, int]:
    """Reconstruct a numeric code from wordpieces.

    Returns (code_string, end_index).
    Handles patterns like "760" + "##00" => "76000"
    """
    if start >= len(tokens):
        return "", start
    code = tokens[start]
    end = start
    while end + 1 < len(tokens) and tokens[end + 1].startswith("##"):
        piece = tokens[end + 1][2:]
        code += piece
        end += 1
    return code, end


def _is_numeric_code(code: str) -> bool:
    """Check if string is a 4-6 digit numeric code (CPT/ICD-like)."""
    return code.isdigit() and 4 <= len(code) <= 6


def _has_cpt_context(tokens: List[str], i: int, j: int, text: str | None) -> bool:
    """Check if numeric code appears in CPT/billing context.

    Returns True if:
    - CPT context words are nearby (cpt, coding, cbct, etc.)
    - Slash-separated in parentheses pattern: "(76000/77002)"
    - Text contains CPT context words
    """
    start = max(0, i - 10)
    end = min(len(tokens), j + 11)
    context_tokens = tokens[start:end]
    norm_context = {_normalize_token(tok) for tok in context_tokens}

    # Primary: CPT context words nearby
    if any(word in norm_context for word in CPT_CONTEXT_WORDS):
        return True

    # Secondary: Slash-separated in parentheses pattern - needs both ( and /
    has_parens = "(" in context_tokens or ")" in context_tokens
    has_slash = "/" in context_tokens
    if has_parens and has_slash:
        return True

    # Tertiary: Text contains CPT context words
    if text:
        text_norm = normalize(text)
        return any(word in text_norm.split() for word in CPT_CONTEXT_WORDS)

    return False


def _is_volume_context(tokens: List[str], idx: int) -> bool:
    unit_window = tokens[max(0, idx - 3) : min(len(tokens), idx + 4)]
    verb_window = tokens[max(0, idx - 6) : min(len(tokens), idx + 7)]
    if not any(_normalize_token(tok) in UNIT_TOKENS for tok in unit_window):
        return False
    return any(_normalize_token(tok) in VOLUME_VERBS for tok in verb_window)


def _repair_bio(tags: List[str]) -> List[str]:
    corrected = tags[:]
    prev_type = "O"
    for i, tag in enumerate(corrected):
        if not tag or tag == "O":
            prev_type = "O"
            corrected[i] = "O"
            continue
        if "-" not in tag:
            corrected[i] = "O"
            prev_type = "O"
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B":
            prev_type = label
            continue
        if prefix == "I":
            if prev_type != label:
                corrected[i] = f"B-{label}"
                prev_type = label
            else:
                prev_type = label
            continue
        corrected[i] = "O"
        prev_type = "O"
    return corrected


def apply_protected_veto(
    tokens: List[str],
    pred_tags: List[str],
    text: str | None = None,
) -> List[str]:
    """Apply protected term veto to predicted tags.

    Key behavior: If ANY protected term is found INSIDE a predicted entity span,
    the ENTIRE contiguous entity span is dropped (set to "O"). This prevents
    dangling leftover tokens after IOB repair.

    Example:
        tokens: ["did", "a", "chart", "##is", "on", "gloria", "ortiz"]
        preds:  ["O", "I-PATIENT", "I-PATIENT", "I-PATIENT", "O", "B-PATIENT", "I-PATIENT"]

        The span ["a", "chart", "##is"] contains protected device "chartis".
        After veto: ["O", "O", "O", "O", "O", "B-PATIENT", "I-PATIENT"]

        Without drop-whole-entity, IOB repair would turn "a" into B-PATIENT (wrong!).
    """
    if len(tokens) != len(pred_tags):
        raise ValueError("Tokens and predicted tags must be the same length.")

    corrected = pred_tags[:]

    # PHASE 1: Drop entire entity spans if they contain protected terms
    # or are stopword-only, or start with punctuation
    entity_spans = _extract_entity_spans(pred_tags)

    for start, end, label in entity_spans:
        if _is_sensitive_id_label(label):
            continue
        should_drop = False

        # Check if span is entirely punctuation (e.g., "(", ",", ")")
        # This is the FIRST check - punctuation-only spans are NEVER valid PHI
        if _span_is_all_punct(tokens, start, end):
            should_drop = True

        # Check if span contains protected device/anatomy term
        elif _is_protected_in_span(tokens, start, end):
            should_drop = True

        # Check if span is stopword-only (e.g., just "a")
        elif _is_stopword_only_span(tokens, start, end):
            should_drop = True

        # Check if span starts with punctuation (e.g., "(")
        elif _span_starts_with_punct(tokens, start, label):
            should_drop = True

        if should_drop:
            for j in range(start, end + 1):
                corrected[j] = "O"

    # PHASE 2: Per-token veto rules (for tokens not already cleared)

    # Wordpiece reconstruction for device/anatomy terms (catch any missed)
    idx = 0
    while idx < len(tokens):
        word, end_idx = reconstruct_wordpiece(tokens, idx)
        if _span_has_any_label(pred_tags, idx, end_idx, SENSITIVE_ID_LABELS):
            idx = end_idx + 1
            continue
        norm_word = normalize(word)
        if _is_protected_device_with_context(norm_word, tokens, idx, end_idx) or is_protected_anatomy_phrase(norm_word):
            for j in range(idx, end_idx + 1):
                corrected[j] = "O"
        idx = end_idx + 1

    # Anatomy phrase scan: left/right + upper/lower/middle + lobe
    words: List[str] = []
    word_spans: List[List[int]] = []
    idx = 0
    while idx < len(tokens):
        word, end_idx = reconstruct_wordpiece(tokens, idx)
        words.append(normalize(word))
        word_spans.append(list(range(idx, end_idx + 1)))
        idx = end_idx + 1
    for i in range(len(words) - 2):
        if words[i] in ("left", "right") and words[i + 1] in ("upper", "lower", "middle") and words[i + 2] == "lobe":
            span_indices = word_spans[i] + word_spans[i + 1] + word_spans[i + 2]
            if _span_has_any_label(pred_tags, span_indices[0], span_indices[-1], SENSITIVE_ID_LABELS):
                continue
            for j in span_indices:
                corrected[j] = "O"

    # CPT codes via stable split with context cues
    for i in range(len(tokens) - 1):
        cpt = _is_stable_cpt_split(tokens, i)
        if not cpt:
            continue
        if _span_has_any_label(pred_tags, i, i + 1, SENSITIVE_ID_LABELS):
            continue
        if _has_cpt_context(tokens, i, i + 1, text):
            corrected[i] = "O"
            corrected[i + 1] = "O"

    # Numeric codes with CPT/CBCT context (atomic spans)
    i = 0
    while i < len(tokens):
        code, end_i = _reconstruct_numeric_code(tokens, i)
        if _span_has_any_label(pred_tags, i, end_i, SENSITIVE_ID_LABELS):
            i = end_i + 1
            continue
        if _is_numeric_code(code) and _has_cpt_context(tokens, i, end_i, text):
            for j in range(i, end_i + 1):
                corrected[j] = "O"
            i = end_i + 1
        else:
            i += 1

    # LN stations via digit+side (+ optional i/s) and station 7 context
    for i in range(len(tokens) - 1):
        if tokens[i].isdigit() and len(tokens[i]) in (1, 2) and tokens[i + 1].startswith("##"):
            if _span_has_any_label(pred_tags, i, i + 1, SENSITIVE_ID_LABELS):
                continue
            side = tokens[i + 1][2:].lower()
            if side in ("r", "l") and not _is_volume_context(tokens, i):
                station = tokens[i] + side
                indices = [i, i + 1]
                if i + 2 < len(tokens) and tokens[i + 2].startswith("##"):
                    suffix = tokens[i + 2][2:].lower()
                    if suffix in ("i", "s"):
                        station += suffix
                        indices.append(i + 2)
                if is_ln_station(station):
                    for idx in indices:
                        corrected[idx] = "O"

    for i, tok in enumerate(tokens):
        if tok == "7" and not _is_volume_context(tokens, i):
            if _span_has_any_label(pred_tags, i, i, SENSITIVE_ID_LABELS):
                continue
            start = max(0, i - 6)
            end = min(len(tokens), i + 7)
            context = {_normalize_token(t) for t in tokens[start:end]}
            if any(word in context for word in LN_CONTEXT_WORDS):
                corrected[i] = "O"

    return _repair_bio(corrected)
