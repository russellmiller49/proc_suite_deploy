"""Non-PHI demo case storage for linking Supabase metadata to PHI procedures.

Stores only synthetic/non-PHI labels and metadata. Attempts to use Supabase if
configured and supabase-py is available; otherwise falls back to in-memory.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List


@dataclass
class PhiDemoCase:
    id: uuid.UUID
    procedure_id: uuid.UUID | None = None
    synthetic_patient_label: str | None = None
    procedure_date: str | None = None
    operator_name: str | None = None
    scenario_label: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["id"] = str(self.id)
        data["procedure_id"] = str(self.procedure_id) if self.procedure_id else None
        data["created_at"] = self.created_at.isoformat()
        return data


class InMemoryPhiDemoStore:
    def __init__(self):
        self._cases: dict[uuid.UUID, PhiDemoCase] = {}

    def list_cases(self) -> List[PhiDemoCase]:
        return list(self._cases.values())

    def create_case(
        self,
        *,
        synthetic_patient_label: str | None,
        procedure_date: str | None,
        operator_name: str | None,
        scenario_label: str | None,
        procedure_id: uuid.UUID | None = None,
    ) -> PhiDemoCase:
        case = PhiDemoCase(
            id=uuid.uuid4(),
            procedure_id=procedure_id,
            synthetic_patient_label=synthetic_patient_label,
            procedure_date=procedure_date,
            operator_name=operator_name,
            scenario_label=scenario_label,
        )
        self._cases[case.id] = case
        return case

    def attach_procedure(self, case_id: uuid.UUID, procedure_id: uuid.UUID) -> PhiDemoCase:
        case = self._cases.get(case_id)
        if case is None:
            raise KeyError("Case not found")
        case.procedure_id = procedure_id
        self._cases[case_id] = case
        return case


def _build_supabase_store():
    """Attempt to build a supabase-backed store if available."""

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not (url and key):
        return None

    try:
        from supabase import create_client  # type: ignore
    except Exception:
        return None

    client = create_client(url, key)
    table = "phi_demo_cases"

    class SupabasePhiDemoStore(InMemoryPhiDemoStore):
        def list_cases(self) -> List[PhiDemoCase]:
            resp = client.table(table).select("*").execute()
            cases: List[PhiDemoCase] = []
            for row in resp.data or []:
                cases.append(
                    PhiDemoCase(
                        id=uuid.UUID(row["id"]),
                        procedure_id=(
                            uuid.UUID(row["procedure_id"]) if row.get("procedure_id") else None
                        ),
                        synthetic_patient_label=row.get("synthetic_patient_label"),
                        procedure_date=row.get("procedure_date"),
                        operator_name=row.get("operator_name"),
                        scenario_label=row.get("scenario_label"),
                        created_at=datetime.fromisoformat(
                            row.get("created_at") or datetime.utcnow().isoformat()
                        ),
                    )
                )
            return cases

        def create_case(
            self,
            *,
            synthetic_patient_label: str | None,
            procedure_date: str | None,
            operator_name: str | None,
            scenario_label: str | None,
            procedure_id: uuid.UUID | None = None,
        ) -> PhiDemoCase:
            case = super().create_case(
                synthetic_patient_label=synthetic_patient_label,
                procedure_date=procedure_date,
                operator_name=operator_name,
                scenario_label=scenario_label,
                procedure_id=procedure_id,
            )
            client.table(table).insert(
                {
                    "id": str(case.id),
                    "procedure_id": str(case.procedure_id) if case.procedure_id else None,
                    "synthetic_patient_label": case.synthetic_patient_label,
                    "procedure_date": case.procedure_date,
                    "operator_name": case.operator_name,
                    "scenario_label": case.scenario_label,
                    "created_at": case.created_at.isoformat(),
                }
            ).execute()
            return case

        def attach_procedure(self, case_id: uuid.UUID, procedure_id: uuid.UUID) -> PhiDemoCase:
            case = super().attach_procedure(case_id, procedure_id)
            (
                client.table(table)
                .update({"procedure_id": str(procedure_id)})
                .eq("id", str(case_id))
                .execute()
            )
            return case

    return SupabasePhiDemoStore()


_store = _build_supabase_store() or InMemoryPhiDemoStore()


def get_phi_demo_store():
    return _store


__all__ = ["PhiDemoCase", "InMemoryPhiDemoStore", "get_phi_demo_store"]
