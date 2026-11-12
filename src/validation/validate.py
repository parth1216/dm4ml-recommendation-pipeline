# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Stage 2 - Data validation.

Automated quality gates on the raw entities using Pandera schemas plus
custom referential-integrity checks:

  * missing values in required columns
  * wrong / un-coercible data types
  * duplicate primary keys and duplicate interaction events
  * rating outside [1, 5]
  * timestamps before 1995-01-01 or in the future
  * interactions that reference unknown users or items
  * categorical domains (gender, genre flags)

Invalid records are QUARANTINED to data/errors/<entity>_errors.csv with
an ``error_reasons`` column; clean records continue to
data/processed/validated_<entity>.parquet. A machine-readable report is
written to reports/validation_report.json and a human-readable one to
reports/validation_report.md.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandera as pa

from src.config import load_config
from src.ingestion.ingest import GENRES
from src.utils import get_logger, utc_now_iso, write_json


# --------------------------------------------------------------------- #
# Schema builders (ranges come from configs/pipeline.yaml)
# --------------------------------------------------------------------- #
def _schemas(v: dict, ts_min: int, ts_max: int) -> dict[str, pa.DataFrameSchema]:
    return {
        "users": pa.DataFrameSchema(
            {
                "user_id": pa.Column(int, pa.Check.ge(1), unique=True, coerce=True),
                "age": pa.Column(int, pa.Check.in_range(v["age_min"], v["age_max"]), coerce=True),
                "gender": pa.Column(str, pa.Check.isin(v["genders"])),
                "occupation": pa.Column(str, nullable=False),
                "zip_code": pa.Column(str, nullable=False),
            },
        ),
        "items": pa.DataFrameSchema(
            {
                "movie_id": pa.Column(int, pa.Check.ge(1), unique=True, coerce=True),
                "title": pa.Column(str, nullable=False),
                # release_date is nullable; format is checked in preparation
                **{g: pa.Column(int, pa.Check.isin([0, 1]), coerce=True) for g in GENRES},
            },
        ),
        "interactions": pa.DataFrameSchema(
            {
                "user_id": pa.Column(int, pa.Check.ge(1), coerce=True),
                "movie_id": pa.Column(int, pa.Check.ge(1), coerce=True),
                "rating": pa.Column(
                    float, pa.Check.in_range(v["rating_min"], v["rating_max"]), coerce=True
                ),
                "timestamp": pa.Column(int, pa.Check.in_range(ts_min, ts_max), coerce=True),
            },
            checks=pa.Check(
                lambda df: ~df.duplicated(subset=["user_id", "movie_id", "timestamp"], keep="first"),
                name="unique_interaction_event",
                error="duplicate (user_id, movie_id, timestamp) event",
            ),
        ),
    }


def _apply_schema(name: str, df: pd.DataFrame, schema: pa.DataFrameSchema):
    """Validate lazily; return (bad_index -> [reasons]) and check records."""
    reasons: dict[int, list[str]] = {}
    checks: list[dict] = []
    try:
        schema.validate(df, lazy=True)
        failure_cases = pd.DataFrame(columns=["column", "check", "index"])
    except pa.errors.SchemaErrors as err:
        failure_cases = err.failure_cases

    for (col, check), grp in failure_cases.groupby(["column", "check"], dropna=False):
        idx = [i for i in grp["index"].tolist() if i is not None]
        checks.append({"entity": name, "check": f"{col}: {check}",
                       "status": "FAIL", "n_failed": len(idx)})
        for i in idx:
            reasons.setdefault(i, []).append(f"{col}: {check}")

    # record the passed checks too, so the report shows full coverage
    for col, col_schema in schema.columns.items():
        for chk in col_schema.checks:
            label = f"{col}: {chk.error or chk.name}"
            if not any(c["check"].startswith(f"{col}:") and str(chk.name) in c["check"]
                       for c in checks):
                checks.append({"entity": name, "check": label, "status": "PASS", "n_failed": 0})
    for chk in schema.checks:  # dataframe-level checks (e.g. duplicate events)
        label = f"dataframe: {chk.error or chk.name}"
        if not any(str(chk.name) in c["check"] for c in checks):
            checks.append({"entity": name, "check": label, "status": "PASS", "n_failed": 0})
    return reasons, checks


def run() -> dict:
    cfg = load_config()
    logger = get_logger("validation", cfg["paths"]["logs"])
    raw, proc, errs = (cfg["paths"]["data_raw"], cfg["paths"]["data_processed"],
                       cfg["paths"]["data_errors"])
    v = cfg["validation"]

    ts_min = int(datetime.fromisoformat(v["timestamp_min"])
                 .replace(tzinfo=timezone.utc).timestamp())
    ts_max = int(time.time())

    frames = {
        "users": pd.read_csv(raw / "users.csv", dtype={"zip_code": str}),
        "items": pd.read_csv(raw / "items.csv"),
        "interactions": pd.read_csv(raw / "interactions.csv"),
    }
    schemas = _schemas(v, ts_min, ts_max)

    report = {"stage": "validation", "executed_at_utc": utc_now_iso(),
              "entities": {}, "checks": []}

    # ---- schema + domain checks per entity ---------------------------
    quarantined: dict[str, pd.DataFrame] = {}
    for name, df in frames.items():
        # missing values in required columns are a dedicated check
        required = list(schemas[name].columns.keys())
        null_mask = df[required].isna().any(axis=1)
        reasons = {i: ["missing value in required column"] for i in df.index[null_mask]}
        report["checks"].append({"entity": name, "check": "no missing values in required columns",
                                 "status": "FAIL" if null_mask.any() else "PASS",
                                 "n_failed": int(null_mask.sum())})

        schema_reasons, checks = _apply_schema(name, df[~null_mask], schemas[name])
        for i, r in schema_reasons.items():
            reasons.setdefault(i, []).extend(r)
        report["checks"].extend(checks)

        bad_idx = sorted(reasons)
        bad = df.loc[bad_idx].copy()
        if len(bad):
            bad["error_reasons"] = ["; ".join(reasons[i]) for i in bad_idx]
        quarantined[name] = bad
        frames[name] = df.drop(index=bad_idx)

    # ---- referential integrity (after entity-level cleaning) ---------
    valid_users = set(frames["users"]["user_id"])
    valid_items = set(frames["items"]["movie_id"])
    inter = frames["interactions"]
    orphan_mask = ~(inter["user_id"].isin(valid_users) & inter["movie_id"].isin(valid_items))
    report["checks"].append({"entity": "interactions",
                             "check": "referential integrity (user_id in users, movie_id in items)",
                             "status": "FAIL" if orphan_mask.any() else "PASS",
                             "n_failed": int(orphan_mask.sum())})
    if orphan_mask.any():
        orphans = inter[orphan_mask].copy()
        orphans["error_reasons"] = "referential integrity violation"
        quarantined["interactions"] = pd.concat([quarantined["interactions"], orphans])
        frames["interactions"] = inter[~orphan_mask]

    # ---- persist clean + quarantined data ----------------------------
    for name, df in frames.items():
        df.to_parquet(proc / f"validated_{name}.parquet", index=False)
    for name, bad in quarantined.items():
        err_file = errs / f"{name}_errors.csv"
        bad.to_csv(err_file, index=False)
        report["entities"][name] = {
            "rows_in": int(len(frames[name]) + len(bad)),
            "rows_valid": int(len(frames[name])),
            "rows_quarantined": int(len(bad)),
            "error_file": str(err_file.relative_to(cfg["project_root"])),
        }
        logger.info("%-13s rows_in=%-7d valid=%-7d quarantined=%d",
                    name, len(frames[name]) + len(bad), len(frames[name]), len(bad))

    write_json(report, cfg["paths"]["reports"] / "validation_report.json")
    _write_md(report, cfg["paths"]["reports"] / "validation_report.md")
    logger.info("Validation complete - %d checks executed, report written.",
                len(report["checks"]))
    return report


def _write_md(report: dict, path: Path) -> None:
    lines = [
        "# Data Validation Report",
        "",
        "Roll No: 2025AG05429 | DM4ML Assignment 2",
        "",
        f"Executed (UTC): {report['executed_at_utc']}",
        "",
        "## Entity summary",
        "",
        "| Entity | Rows in | Valid | Quarantined | Error file |",
        "|---|---:|---:|---:|---|",
    ]
    for name, e in report["entities"].items():
        lines.append(f"| {name} | {e['rows_in']} | {e['rows_valid']} | "
                     f"{e['rows_quarantined']} | `{e['error_file']}` |")
    lines += ["", "## Checks executed", "",
              "| Entity | Check | Status | Failed rows |", "|---|---|---|---:|"]
    for c in report["checks"]:
        lines.append(f"| {c['entity']} | {c['check']} | {c['status']} | {c['n_failed']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
