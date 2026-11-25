"""Fresh-process benchmark worker (Linux/Colab)."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from .contracts import read_json
from .storage import query, signature

CSV_COLUMNS = ["CERT", "REPDTE", "ASSET", "DEP"]


def read_engine(path, engine):
    """Supplementary CSV reader study with identical selected columns and types."""
    if engine == "pandas":
        import pandas as pd
        start = time.perf_counter()
        frame = pd.read_csv(path, usecols=CSV_COLUMNS,
                            dtype={"CERT": "int64", "REPDTE": "string", "ASSET": "float64", "DEP": "float64"},
                            keep_default_na=False, na_values=[""], float_precision="round_trip")
        elapsed = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        result = {"rows": len(frame), "cert_sum": int(frame.CERT.sum()), "asset_sum": float(frame.ASSET.sum()),
                  "dep_sum": float(frame.DEP.sum()), "asset_nulls": int(frame.ASSET.isna().sum()),
                  "dep_nulls": int(frame.DEP.isna().sum()), "date_min": str(frame.REPDTE.min()), "date_max": str(frame.REPDTE.max())}
    elif engine == "polars":
        import polars as pl
        start = time.perf_counter()
        frame = pl.read_csv(path, columns=CSV_COLUMNS, schema_overrides={"CERT": pl.Int64, "REPDTE": pl.String,
                            "ASSET": pl.Float64, "DEP": pl.Float64}, null_values=[""])
        elapsed = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        result = {"rows": frame.height, "cert_sum": int(frame["CERT"].sum()), "asset_sum": float(frame["ASSET"].sum() or 0),
                  "dep_sum": float(frame["DEP"].sum() or 0), "asset_nulls": frame["ASSET"].null_count(),
                  "dep_nulls": frame["DEP"].null_count(), "date_min": str(frame["REPDTE"].min()), "date_max": str(frame["REPDTE"].max())}
    else:
        raise ValueError("Unknown CSV engine")
    return {"seconds": elapsed, "peak_rss_mib": rss, "signature": result}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--format", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--threads", type=int, required=True)
    args = p.parse_args()
    manifest = read_json(args.work / "preparation.json")
    if args.format in ("pandas", "polars"):
        result = read_engine(args.work / manifest["formats"]["csv"]["path"], args.format)
    else:
        import pyarrow as pa
        import pyarrow.compute
        import pyarrow.csv
        import pyarrow.dataset
        pa.set_cpu_count(args.threads)
        pa.set_io_thread_count(args.threads)
        start = time.perf_counter()
        table, scan = query(args.work, manifest, args.format, args.workload)
        elapsed = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        result = {"seconds": elapsed, "peak_rss_mib": rss, "signature": signature(table), **scan}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
