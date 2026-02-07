from proc_schemas.procedure_report import ProcedureReport


def backfill_specimen_counts(report: ProcedureReport) -> ProcedureReport:
    """Return a copy of report w/ implied specimen counts inserted."""
    updated_targets = []
    for target in report.procedure_core.targets:
        specimens = dict(target.specimens)
        if not specimens:
            specimens["fna"] = 1
        updated_targets.append(target.model_copy(update={"specimens": specimens}))
    new_core = report.procedure_core.model_copy(update={"targets": updated_targets})
    return report.model_copy(update={"procedure_core": new_core})
