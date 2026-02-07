from proc_schemas.procedure_report import ProcedureReport


def enforce_laterality(report: ProcedureReport) -> ProcedureReport:
    """Fallback laterality if missing but targets imply a side."""
    if report.procedure_core.laterality:
        return report
    inferred = None
    for target in report.procedure_core.targets:
        segment = (target.segment or "").upper()
        if segment.endswith("R"):
            inferred = "right"
            break
        if segment.endswith("L"):
            inferred = "left"
            break
    if inferred:
        new_core = report.procedure_core.model_copy(update={"laterality": inferred})
        return report.model_copy(update={"procedure_core": new_core})
    return report
