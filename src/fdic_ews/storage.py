"""Controlled storage layouts and query-equivalence checks."""
from __future__ import annotations

import math
import shutil
from datetime import date
from pathlib import Path

from .acquisition import verify
from .contracts import (IntegrityError, schema, restore_schema, read_json, write_json, sha256, utc_now)

FORMATS = ("csv", "flat_snappy", "flat_zstd", "partitioned_snappy", "partitioned_zstd")
WORKLOADS = ("full", "projection", "year_2020", "quarter_2020q1")
PROJECTION = ["CERT", "REPDTE", "ASSET", "DEP", "STNAME", "ROA"]


def read_csv(path, columns=None):
    import pyarrow as pa
    import pyarrow.csv as csv
    full = schema("financials")
    if columns is None:
        import csv as stdcsv
        with Path(path).open(newline="", encoding="utf-8") as handle:
            if next(stdcsv.reader(handle), None) != full.names:
                raise IntegrityError("CSV header differs from Financials contract")
    expected = full if columns is None else pa.schema([full.field(c) for c in columns])
    # Arrow's CSV reader infers all fields as nullable, even when the source has
    # required identifiers. Restore the schema after checking required values.
    table = csv.read_csv(path, read_options=csv.ReadOptions(use_threads=True),
                         parse_options=csv.ParseOptions(newlines_in_values=True),
                         convert_options=csv.ConvertOptions(
                             column_types={f.name: f.type for f in full},
                             include_columns=expected.names,
                             strings_can_be_null=True,
                             quoted_strings_can_be_null=False,
                             null_values=[""]))
    return restore_schema(table, expected)


def compare_tables(a, b):
    """Full value/null-mask comparison in canonical key order (float tolerance for CSV)."""
    import numpy as np
    import pyarrow as pa
    if a.schema != b.schema or a.num_rows != b.num_rows:
        raise IntegrityError("Table shape/schema mismatch")
    for field in a.schema:
        x, y = a[field.name], b[field.name]
        if not x.is_null().combine_chunks().equals(y.is_null().combine_chunks()):
            raise IntegrityError(f"Null mask mismatch: {field.name}")
        if pa.types.is_floating(field.type):
            if not np.allclose(x.to_numpy(zero_copy_only=False), y.to_numpy(zero_copy_only=False),
                               rtol=1e-12, atol=1e-9, equal_nan=True):
                raise IntegrityError(f"Numeric mismatch: {field.name}")
        elif not x.combine_chunks().equals(y.combine_chunks()):
            raise IntegrityError(f"Value mismatch: {field.name}")


def _entry(work, path):
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    if not files:
        raise IntegrityError(f"Empty format: {path}")
    return {"path": path.relative_to(work).as_posix(), "bytes": sum(p.stat().st_size for p in files),
            "files": [{"path": p.relative_to(work).as_posix(), "bytes": p.stat().st_size,
                       "sha256": sha256(p)} for p in files]}


def prepare(source, work):
    """Stage every format on one local filesystem; do not mutate the source."""
    import pyarrow as pa
    import pyarrow.csv as csv
    import pyarrow.parquet as pq
    source, work = Path(source).resolve(), Path(work).resolve()
    if work.exists() or work in source.parents or source in work.parents:
        raise IntegrityError("Use a new, separate benchmark directory")
    evidence = verify(source)
    work.mkdir(parents=True)
    files = read_json(source / "manifest.json")["financials"]
    paths = {"csv": work / "financials.csv", "flat_snappy": work / "flat_snappy.parquet",
             "flat_zstd": work / "flat_zstd.parquet", "partitioned_snappy": work / "snappy",
             "partitioned_zstd": work / "zstd"}
    # Read a single quarter at a time; use the original Snappy bytes for the
    # production layout, and write the alternative codec with identical options.
    parts = []
    for q in sorted(files):
        original = source / files[q]["path"]
        relative = Path(files[q]["path"]).relative_to("financials")
        a, b = paths["partitioned_snappy"] / relative, paths["partitioned_zstd"] / relative
        a.parent.mkdir(parents=True, exist_ok=True)
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, a)
        if sha256(a) != files[q]["sha256"]:
            raise IntegrityError("Local staging checksum mismatch")
        t = pq.ParquetFile(a).read()
        pq.write_table(t, b, compression="zstd", compression_level=3, row_group_size=65536,
                       version="2.6", use_dictionary=True)
        compare_tables(t, pq.ParquetFile(b).read())
        parts.append(t)
    full = pa.concat_tables(parts)
    del parts
    if full.num_rows != evidence["row_counts"]["financials"]:
        raise IntegrityError("Prepared data differs from source count")
    csv.write_csv(full, paths["csv"])
    for name, codec in (("flat_snappy", "snappy"), ("flat_zstd", "zstd")):
        extra = {"compression_level": 3} if codec == "zstd" else {}
        pq.write_table(full, paths[name], compression=codec, row_group_size=65536,
                       version="2.6", use_dictionary=True, **extra)
        compare_tables(full, pq.ParquetFile(paths[name]).read())
    compare_tables(full, read_csv(paths["csv"]))
    # Compare all query outputs, including filters, rather than just file totals.
    manifest = {"status": "PREPARED", "created_utc": utc_now(),
                "source_manifest_sha256": evidence["manifest_sha256"],
                "financial_rows": full.num_rows, "formats": {k: _entry(work, p) for k, p in paths.items()},
                "compression": {"snappy": "default", "zstd_level": 3}, "row_group_size": 65536}
    del full
    for workload in WORKLOADS:
        ref, _ = query(work, manifest, "csv", workload)
        ref = ref.sort_by([("REPDTE", "ascending"), ("CERT", "ascending")])
        for fmt in FORMATS[1:]:
            actual, _ = query(work, manifest, fmt, workload)
            compare_tables(ref, actual.sort_by([("REPDTE", "ascending"), ("CERT", "ascending")]))
    manifest["equivalence"] = {"full_value_roundtrip": True, "query_workloads": list(WORKLOADS),
                               "formats": list(FORMATS), "status": "PASS"}
    write_json(work / "preparation.json", manifest)
    return manifest


def verify_preparation(work):
    work = Path(work).resolve()
    m = read_json(work / "preparation.json")
    if m["status"] != "PREPARED" or m["equivalence"]["status"] != "PASS" or set(m["formats"]) != set(FORMATS):
        raise IntegrityError("Unverified format preparation")
    for e in m["formats"].values():
        path = work / e["path"]
        if _entry(work, path) != e:
            raise IntegrityError(f"Prepared file identity changed: {path}")
    return m


def query(work, manifest, fmt, workload):
    """Return materialized data plus scan metadata. Hive pruning uses explicit keys."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    if fmt not in FORMATS or workload not in WORKLOADS:
        raise ValueError("Unknown format or workload")
    columns = schema("financials").names if workload == "full" else PROJECTION
    expected = pa.schema([schema("financials").field(c) for c in columns])
    path = Path(work) / manifest["formats"][fmt]["path"]
    part = fmt.startswith("partitioned")
    date_filter = None
    partition_filter = None
    if workload == "year_2020":
        date_filter = (pc.field("REPDTE") >= pa.scalar(date(2020, 1, 1))) & (pc.field("REPDTE") <= pa.scalar(date(2020, 12, 31)))
        partition_filter = pc.field("year") == 2020
    elif workload == "quarter_2020q1":
        date_filter = pc.field("REPDTE") == pa.scalar(date(2020, 3, 31))
        partition_filter = (pc.field("year") == 2020) & (pc.field("quarter") == 1)
    if fmt == "csv":
        table = read_csv(path, columns)
        if date_filter is not None:
            table = table.filter(date_filter)
        return table, {"total_fragments": 1, "candidate_fragments": 1}
    partitioning = ds.partitioning(pa.schema([("year", pa.int32()), ("quarter", pa.int32())]), flavor="hive") if part else None
    dataset = ds.dataset(path, format="parquet", partitioning=partitioning)
    pred = (date_filter & partition_filter) if part and date_filter is not None else date_filter
    table = restore_schema(dataset.to_table(columns=columns, filter=pred), expected)
    # Candidate fragments are files surviving partition predicates; not bytes read.
    count = sum(1 for _ in dataset.get_fragments(filter=pred))
    total = len(dataset.files)
    return table, {"total_fragments": total, "candidate_fragments": count}


def signature(table):
    """Small order-independent check for each timed read; full parity is checked before timing."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import hashlib
    import numpy as np
    k = table.select(["REPDTE", "CERT"]).sort_by([("REPDTE", "ascending"), ("CERT", "ascending")])
    dates = pc.cast(k["REPDTE"], pa.int32()).to_numpy(zero_copy_only=False)
    certs = k["CERT"].to_numpy(zero_copy_only=False)
    h = hashlib.sha256(np.asarray(dates, dtype="<i4").tobytes() + np.asarray(certs, dtype="<i8").tobytes()).hexdigest()
    sums = {}
    for f in table.schema:
        if pa.types.is_floating(f.type):
            value = pc.sum(table[f.name]).as_py()
            sums[f.name] = None if value is None else float(value)
    return {"rows": table.num_rows, "keys_sha256": h, "null_counts": {c: table[c].null_count for c in table.column_names},
            "numeric_sums": sums}


def compare_signatures(a, b):
    for k in ("rows", "keys_sha256", "null_counts"):
        if a[k] != b[k]:
            raise IntegrityError(f"Benchmark result mismatch: {k}")
    if set(a["numeric_sums"]) != set(b["numeric_sums"]):
        raise IntegrityError("Benchmark numeric field mismatch")
    for k, v in a["numeric_sums"].items():
        other = b["numeric_sums"][k]
        if v is None or other is None:
            if v != other:
                raise IntegrityError(f"Benchmark null sum mismatch: {k}")
        elif not math.isclose(v, other, rel_tol=1e-10, abs_tol=1e-6):
            raise IntegrityError(f"Benchmark sum mismatch: {k}")
