from __future__ import annotations

from typing import Any, ClassVar, List, Optional, Type

from pydantic import BaseModel


class AdapterRegistry:
    _adapters: List[Type["ExtractionAdapter"]] = []

    @classmethod
    def register(cls, adapter_cls: Type["ExtractionAdapter"]) -> None:
        if adapter_cls not in cls._adapters:
            cls._adapters.append(adapter_cls)

    @classmethod
    def all(cls) -> list[Type["ExtractionAdapter"]]:
        return list(cls._adapters)


class ExtractionAdapter:
    proc_type: ClassVar[str]
    schema_model: ClassVar[Type[BaseModel]]
    schema_id: ClassVar[Optional[str]] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is ExtractionAdapter:
            return
        if not getattr(cls, "proc_type", None) or not getattr(cls, "schema_model", None):
            return
        AdapterRegistry.register(cls)

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        raise NotImplementedError

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any] | BaseModel | None:
        raise NotImplementedError

    @classmethod
    def get_schema_id(cls) -> str:
        if cls.schema_id:
            return cls.schema_id
        return f"{cls.proc_type}_v1"

    @classmethod
    def extract(cls, source: dict[str, Any]) -> BaseModel | None:
        if not cls.matches(source):
            return None
        payload = cls.build_payload(source)
        if payload is None:
            return None
        if isinstance(payload, cls.schema_model):
            return payload
        try:
            return cls.schema_model.model_validate(payload)
        except Exception:
            return payload


class DictPayloadAdapter(ExtractionAdapter):
    source_key: ClassVar[str]

    @classmethod
    def matches(cls, source: dict[str, Any]) -> bool:
        value = source.get(cls.source_key)
        return isinstance(value, (dict, BaseModel))

    @classmethod
    def build_payload(cls, source: dict[str, Any]) -> dict[str, Any] | BaseModel | None:
        return source.get(cls.source_key)
