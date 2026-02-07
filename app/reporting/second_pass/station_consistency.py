from proc_schemas.procedure_report import ProcedureReport


def align_station_metadata(report: ProcedureReport) -> ProcedureReport:
    """Ensure intraop metadata references the same stations list."""
    stations = report.procedure_core.stations_sampled
    intraop = dict(report.intraop)
    intraop.setdefault("stations", stations)
    return report.model_copy(update={"intraop": intraop})
