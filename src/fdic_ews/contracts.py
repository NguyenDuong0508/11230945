"""Selected source fields, loss-aware conversion and integrity contracts."""
from __future__ import annotations

import calendar
import hashlib
import json
import math
import os
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import NamedTemporaryFile

START, END = date(1984, 3, 31), date(2026, 6, 30)
FIELDS = {
    "financials": tuple("CERT REPDTE STNAME ASSET DEP DEPDOM EQ BRO CHBAL SC LNLSGR LNLSNET LNRE LNCI LNCON LNATRES NCLNLS P3ASSET P9ASSET NAASSET NTLNLS ORE INTAN NETINC INTINC EINTEXP NONII NONIX ELNATR ROA ROE NIMY RBC1AAJ RBCRWAJ".split()),
    "failures": tuple("CERT FIN NAME CITYST FAILDATE FAILYR COST QBFASSET QBFDEP RESTYPE RESTYPE1 COSTMOSTRECENTASOF".split()),
    "institutions": tuple("CERT NAME STNAME BKCLASS ACTIVE ESTYMD ENDEFYMD CHANGEC1 INSDATE".split()),
}
TEXT = {
    "financials": {"STNAME"},
    "failures": {"NAME", "CITYST", "FAILDATE", "RESTYPE", "RESTYPE1", "COSTMOSTRECENTASOF"},
    "institutions": {"NAME", "STNAME", "BKCLASS", "ACTIVE", "ESTYMD", "ENDEFYMD", "CHANGEC1", "INSDATE"},
}
INTEGER = {"financials": {"CERT"}, "failures": {"CERT", "FIN", "FAILYR"}, "institutions": {"CERT"}}
REQUIRED = {"financials": {"CERT", "REPDTE"}, "failures": set(), "institutions": {"CERT"}}


class IntegrityError(RuntimeError):
    """An input or output did not satisfy the acquisition contract."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def quarter_ends(start=START, end=END):
    def valid(d):
        return d.month in (3, 6, 9, 12) and d.day == calendar.monthrange(d.year, d.month)[1]
    if not valid(start) or not valid(end) or start > end:
        raise ValueError("Expected ordered calendar quarter ends")
    return [d for y in range(start.year, end.year + 1) for m in (3, 6, 9, 12)
            if start <= (d := date(y, m, calendar.monthrange(y, m)[1])) <= end]


def report_date(value):
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except ValueError:
            continue
    # Accept ISO timestamp representations only when they denote midnight.
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.hour == dt.minute == dt.second == dt.microsecond == 0:
            return dt.date()
    except ValueError:
        pass
    raise IntegrityError(f"Invalid reporting date: {value!r}")


def number(value, integer=False):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise IntegrityError("Boolean in numeric field")
    try:
        n = Decimal(str(value))
        if not n.is_finite():
            raise ValueError("Nonfinite value")
        if integer:
            if n != n.to_integral_value() or not -(2**63) <= n < 2**63:
                raise ValueError("Not an int64")
            return int(n)
        v = float(n)
        if not math.isfinite(v):
            raise ValueError("Outside float64 range")
        return v
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise IntegrityError(f"Invalid numeric field: {value!r}") from exc


def normalize(endpoint, records, quarter=None):
    """Keep all rows; distinguish absent fields from explicit source nulls."""
    fields = FIELDS[endpoint]
    absent = dict.fromkeys(fields, 0)
    nulls = dict.fromkeys(fields, 0)
    rows = []
    for raw in records:
        if not isinstance(raw, dict):
            raise IntegrityError("Source row must be an object")
        row = {}
        for f in fields:
            absent[f] += int(f not in raw)
            v = raw.get(f)
            if f == "REPDTE":
                v = report_date(v)
            elif f in INTEGER[endpoint]:
                v = number(v, integer=True)
            elif f in TEXT[endpoint]:
                if v is not None and not isinstance(v, (str, int, float, bool)):
                    raise IntegrityError(f"Unexpected text field value: {f}")
                v = None if v is None else str(v)
            else:
                v = number(v)
            if f in REQUIRED[endpoint] and v is None:
                raise IntegrityError(f"Missing required field: {f}")
            nulls[f] += int(v is None)
            row[f] = v
        rows.append(row)
    quality = validate_records(endpoint, rows, quarter)
    return rows, {**quality, "absent_key_counts": absent, "null_counts": nulls}


def validate_records(endpoint, rows, quarter=None):
    if not rows:
        raise IntegrityError(f"Empty {endpoint} response")
    certs = [r["CERT"] for r in rows]
    if endpoint != "failures":
        if any(c is None or c <= 0 for c in certs):
            raise IntegrityError(f"Invalid {endpoint} CERT")
        if len(certs) != len(set(certs)):
            raise IntegrityError(f"Duplicate {endpoint} CERT within requested scope")
    if endpoint == "financials" and (quarter is None or any(r["REPDTE"] != quarter for r in rows)):
        raise IntegrityError("Financial records outside requested quarter")
    return {"rows": len(rows), "distinct_certs": len({c for c in certs if c is not None}),
            "null_cert": certs.count(None)}


def schema(endpoint):
    import pyarrow as pa
    return pa.schema([pa.field(f, pa.date32() if f == "REPDTE" else
                               pa.int64() if f in INTEGER[endpoint] else
                               pa.string() if f in TEXT[endpoint] else pa.float64(),
                               nullable=f not in REQUIRED[endpoint]) for f in FIELDS[endpoint]])


def restore_schema(table, expected):
    """CSV does not encode Arrow nullability; validate data before restoring it."""
    if table.column_names != expected.names:
        raise IntegrityError("Column order/names differ from schema")
    for field in expected:
        col = table[field.name]
        if col.type != field.type:
            raise IntegrityError(f"Unexpected type for {field.name}: {col.type}")
        if not field.nullable and col.null_count:
            raise IntegrityError(f"Null values in required field: {field.name}")
    return table.cast(expected, safe=True)


def partition_path(q):
    return f"financials/year={q.year}/quarter={q.month // 3}/part.parquet"


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temp = Path(f.name)
            json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        os.replace(temp, path)
    finally:
        if temp:
            temp.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
