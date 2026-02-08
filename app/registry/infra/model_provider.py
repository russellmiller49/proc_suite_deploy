"""Registry ML predictor provider with backend-aware lazy initialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.common.logger import get_logger
from app.registry.model_runtime import get_registry_runtime_dir, resolve_model_backend
from ml.lib.ml_coder.registry_predictor import RegistryMLPredictor

logger = get_logger("registry_model_provider")


class RegistryModelProvider:
    """Lazily resolve and cache the configured registry predictor backend."""

    def __init__(self) -> None:
        self._predictor: Any | None = None
        self._init_attempted = False

    def get_predictor(self) -> Any | None:
        """Get a registry predictor instance or None when artifacts are unavailable."""
        if self._init_attempted:
            return self._predictor

        self._init_attempted = True

        backend = resolve_model_backend()
        runtime_dir = get_registry_runtime_dir()

        def _try_pytorch() -> Any | None:
            try:
                from app.registry.inference_pytorch import TorchRegistryPredictor

                predictor = TorchRegistryPredictor(bundle_dir=runtime_dir)
                if predictor.available:
                    logger.info(
                        "Using TorchRegistryPredictor with %d labels",
                        len(getattr(predictor, "labels", [])),
                    )
                    return predictor
                logger.debug("Torch predictor initialized but not available")
            except ImportError as exc:
                logger.debug("PyTorch/Transformers not available (%s)", exc)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Torch predictor init failed (%s)", exc)
            return None

        def _try_onnx() -> Any | None:
            try:
                from app.registry.inference_onnx import ONNXRegistryPredictor

                model_path: Path | None = None
                for candidate in ("registry_model_int8.onnx", "registry_model.onnx"):
                    candidate_path = runtime_dir / candidate
                    if candidate_path.exists():
                        model_path = candidate_path
                        break

                tokenizer_path: Path | None = None
                for candidate in ("tokenizer", "roberta_registry_tokenizer"):
                    candidate_path = runtime_dir / candidate
                    if candidate_path.exists():
                        tokenizer_path = candidate_path
                        break

                thresholds_path: Path | None = None
                for candidate in (
                    "thresholds.json",
                    "registry_thresholds.json",
                    "roberta_registry_thresholds.json",
                ):
                    candidate_path = runtime_dir / candidate
                    if candidate_path.exists():
                        thresholds_path = candidate_path
                        break

                label_fields_path: Path | None = None
                candidate_label_fields = runtime_dir / "registry_label_fields.json"
                if candidate_label_fields.exists():
                    label_fields_path = candidate_label_fields
                    try:
                        thresholds_payload = (
                            json.loads(thresholds_path.read_text())
                            if thresholds_path and thresholds_path.exists()
                            else None
                        )
                        labels_payload = json.loads(candidate_label_fields.read_text())
                        if isinstance(thresholds_payload, dict) and thresholds_payload:
                            threshold_keys = {
                                key for key in thresholds_payload.keys() if isinstance(key, str)
                            }
                            label_keys = (
                                {x for x in labels_payload if isinstance(x, str)}
                                if isinstance(labels_payload, list)
                                else set()
                            )
                            if threshold_keys and label_keys != threshold_keys:
                                label_fields_path = None
                    except Exception:  # noqa: BLE001
                        label_fields_path = None

                predictor = ONNXRegistryPredictor(
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    thresholds_path=thresholds_path,
                    label_fields_path=label_fields_path,
                )
                if predictor.available:
                    logger.info(
                        "Using ONNXRegistryPredictor with %d labels",
                        len(getattr(predictor, "labels", [])),
                    )
                    return predictor
                logger.debug("ONNX model not available")
            except ImportError:
                logger.debug("ONNX runtime not available")
            except Exception as exc:  # noqa: BLE001
                logger.debug("ONNX predictor init failed (%s)", exc)
            return None

        if backend == "pytorch":
            predictor = _try_pytorch()
            if predictor is not None:
                self._predictor = predictor
                return self._predictor
        elif backend == "onnx":
            predictor = _try_onnx()
            if predictor is None:
                model_path = runtime_dir / "registry_model_int8.onnx"
                raise RuntimeError(
                    "MODEL_BACKEND=onnx but ONNXRegistryPredictor failed to initialize. "
                    f"Expected model at {model_path}."
                )
            self._predictor = predictor
            return self._predictor
        else:
            predictor = _try_onnx()
            if predictor is not None:
                self._predictor = predictor
                return self._predictor

        try:
            self._predictor = RegistryMLPredictor()
            if not self._predictor.available:
                logger.warning(
                    "RegistryMLPredictor initialized but not available "
                    "(model artifacts missing). ML hybrid audit disabled."
                )
                self._predictor = None
            else:
                logger.info(
                    "Using RegistryMLPredictor (TF-IDF) with %d labels",
                    len(self._predictor.labels),
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to initialize RegistryMLPredictor; ML hybrid audit disabled."
            )
            self._predictor = None

        return self._predictor

    def set_predictor_for_testing(self, predictor: Any | None) -> None:
        """Override predictor cache (test helper)."""
        self._predictor = predictor
        self._init_attempted = True

    @property
    def init_attempted(self) -> bool:
        return self._init_attempted


__all__ = ["RegistryModelProvider"]
