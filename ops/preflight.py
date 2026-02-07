"""Basic environment + asset validation ahead of CI."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def check_spacy_stack() -> None:
    import scispacy  # type: ignore
    import spacy  # type: ignore

    print(f"spaCy {spacy.__version__} detected")
    print(f"scispaCy {scispacy.__version__} detected")


def check_sklearn() -> None:
    import sklearn

    version = sklearn.__version__
    if not version.startswith("1.7"):
        raise RuntimeError(f"sklearn version must be 1.7.x, found {version}")
    print(f"sklearn {version} pinned OK")


def check_rules_and_templates() -> None:
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs"
    for path in config_dir.rglob("*.yaml"):
        yaml.safe_load(path.read_text())
    print("YAML configs parsed")

    from app.reporting.engine import compose_report_from_text

    text = "EBUS-TBNA of stations 7 and 4R; 3 FNA passes at each."
    report, note = compose_report_from_text(text, {"plan": "Recover in PACU"})
    assert "Targets & Specimens" in note
    from app.autocode.engine import autocode

    billing = autocode(report)
    assert billing.codes, "Autocoder produced zero codes for smoke test"
    print("Templates render and coder returns codes")


def main() -> None:
    check_spacy_stack()
    check_sklearn()
    check_rules_and_templates()
    print("Preflight completed ✅")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Preflight failed: {exc}", file=sys.stderr)
        raise
