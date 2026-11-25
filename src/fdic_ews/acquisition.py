"""Checkpointed acquisition and independent verification of Bronze artifacts."""
from __future__ import annotations

import os
import platform
import subprocess
from collections import Counter
from pathlib import Path
from uuid import uuid4

from .api import BASE, FDICClient
from .contracts import (START, END, FIELDS, IntegrityError, normalize, validate_records,
                        schema, quarter_ends, partition_path, read_json, write_json, sha256, utc_now)

VERSION = 1


def code_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_id():
    return utc_now().replace(":", "").replace("+0000", "Z").replace(".", "_") + "_" + uuid4().hex[:8]


def contract():
    return {"version": VERSION, "start": str(START), "end": str(END),
            "source": BASE, "fields": {e: list(f) for e, f in FIELDS.items()}}


def artifact_path(endpoint, q=None):
    return partition_path(q) if endpoint == "financials" else f"{endpoint}.parquet"


def parquet_commit(table, target):
    import pyarrow.parquet as pq
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + "." + uuid4().hex + ".tmp")
    try:
        pq.write_table(table, tmp, compression="snappy", row_group_size=65536,
                       version="2.6", use_dictionary=True)
        restored = pq.ParquetFile(tmp).read()
        if restored.schema != table.schema or not restored.equals(table):
            raise IntegrityError(f"Parquet round-trip failed: {target.name}")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def inspect_artifact(root, entry, endpoint, q=None):
    import pyarrow.parquet as pq
    rel = artifact_path(endpoint, q)
    if entry["path"] != rel or entry["endpoint"] != endpoint:
        raise IntegrityError("Artifact identity differs from contract")
    path = root / rel
    if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise IntegrityError(f"Missing or modified artifact: {rel}")
    table = pq.ParquetFile(path).read()
    if table.schema != schema(endpoint):
        raise IntegrityError(f"Schema mismatch: {rel}")
    keys = ["CERT", "REPDTE"] if q is not None else ["CERT"]
    quality = validate_records(endpoint, table.select(keys).to_pylist(), q)
    if quality != {k: entry["quality"][k] for k in quality} or quality["rows"] != entry["api"]["api_total"]:
        raise IntegrityError(f"Counts differ from API or manifest: {rel}")
    pages = entry["api"]["pages"]
    expected_offset = 0
    for p in pages:
        if p["offset"] != expected_offset or p["api_total"] != quality["rows"] or p["rows"] <= 0:
            raise IntegrityError(f"Invalid pagination evidence: {rel}")
        expected_offset += p["rows"]
    if expected_offset != quality["rows"]:
        raise IntegrityError(f"Incomplete pagination evidence: {rel}")
    if entry["api"]["endpoint"] != endpoint:
        raise IntegrityError("Unexpected API endpoint")
    expected_filter = f"REPDTE:{q:%Y%m%d}" if q else None
    if entry["api"]["parameters"].get("filters") != expected_filter:
        raise IntegrityError("Unexpected API filter")
    if entry["api"]["parameters"]["fields"] != ",".join(FIELDS[endpoint]):
        raise IntegrityError("Unexpected API field list")
    if {name: table[name].null_count for name in table.column_names} != entry["quality"]["null_counts"]:
        raise IntegrityError(f"Null counts differ: {rel}")
    return table


def verify(root, require_sealed=True):
    """Verify files, keys, API page counts and calendar, without changing the run."""
    root = Path(root).resolve()
    m = read_json(root / "manifest.json")
    if m["contract"] != contract():
        raise IntegrityError("Source contract mismatch")
    if require_sealed and m["state"] != "VERIFIED":
        raise IntegrityError("Source run is not sealed")
    expected = {str(q) for q in quarter_ends()}
    if set(m["financials"]) != expected or set(m["references"]) != {"failures", "institutions"}:
        raise IntegrityError("Incomplete quarter or reference index")
    wanted = {partition_path(q) for q in quarter_ends()} | {"failures.parquet", "institutions.parquet"}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*.parquet")}
    if wanted != actual:
        raise IntegrityError(f"Unexpected/missing files: {sorted(wanted ^ actual)}")
    certs, totals, quarter_stats = set(), {}, []
    absent = Counter()
    for q in quarter_ends():
        entry = m["financials"][str(q)]
        table = inspect_artifact(root, entry, "financials", q)
        certs.update(table["CERT"].to_pylist())
        absent.update(entry["quality"]["absent_key_counts"])
        quarter_stats.append({"quarter": str(q), "rows": table.num_rows, "api_total": entry["api"]["api_total"],
                              "bytes": entry["bytes"], "sha256": entry["sha256"]})
    totals["financials"] = sum(x["rows"] for x in quarter_stats)
    restypes, missing_cert, nonpositive_cert = [], 0, 0
    for e in ("failures", "institutions"):
        table = inspect_artifact(root, m["references"][e], e)
        totals[e] = table.num_rows
        if e == "failures":
            counts = Counter(table["RESTYPE"].to_pylist())
            restypes = [{"source_value": v, "rows": n} for v, n in sorted(counts.items(), key=lambda p: str(p[0]))]
            missing_cert = table["CERT"].null_count
            nonpositive_cert = sum(c is not None and c <= 0 for c in table["CERT"].to_pylist())
    # A field absent from every response may be a misspelled or unsupported API field.
    absent_all = [f for f in FIELDS["financials"] if absent[f] == totals["financials"]]
    if absent_all:
        raise IntegrityError(f"Financial fields absent from all records: {absent_all}")
    for e in ("failures", "institutions"):
        q = m["references"][e]["quality"]
        missing_fields = [f for f in FIELDS[e] if q["absent_key_counts"][f] == totals[e]]
        if missing_fields:
            raise IntegrityError(f"Reference fields absent from all {e} records: {missing_fields}")
    return {"status": "PASS", "source_state": m["state"], "manifest_sha256": sha256(root / "manifest.json"),
            "quarter_count": len(quarter_stats), "row_counts": totals, "distinct_financial_certs": len(certs),
            "failure_null_cert": missing_cert, "failure_nonpositive_cert": nonpositive_cert,
            "failure_restype_counts": restypes, "quarter_statistics": quarter_stats,
            "financial_absent_key_counts": dict(absent)}


def acquire(root, client=None, max_new_quarters=None):
    """Resume an unsealed run. A sealed run can only be verified, not modified."""
    import pyarrow as pa
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    if path.exists():
        m = read_json(path)
        if m["contract"] != contract():
            raise IntegrityError("Cannot resume a different source contract")
        if m["state"] == "VERIFIED":
            return verify(root)
        if m["state"] != "IN_PROGRESS":
            raise IntegrityError("Unknown acquisition state")
    else:
        if any(root.iterdir()):
            raise IntegrityError("New source directory is not empty")
        m = {"contract": contract(), "state": "IN_PROGRESS", "created_utc": utc_now(),
             "code_revision": code_revision(), "python": platform.python_version(), "pyarrow": pa.__version__,
             "financials": {}, "references": {}}
        write_json(path, m)
    client = client if client is not None else FDICClient()
    if max_new_quarters is not None and max_new_quarters < 1:
        raise ValueError("max_new_quarters must be positive")
    new = 0
    for endpoint, q in [("financials", q) for q in quarter_ends()] + [("failures", None), ("institutions", None)]:
        index = m["financials"] if q is not None else m["references"]
        key = str(q) if q is not None else endpoint
        if key in index:
            try:
                inspect_artifact(root, index[key], endpoint, q)
                continue
            except (IntegrityError, OSError, ValueError, KeyError, pa.ArrowException) as exc:
                print(f"Reacquiring invalid checkpoint {key}: {exc}", flush=True)
        if max_new_quarters is not None and (new >= max_new_quarters or q is None):
            return {"status": "IN_PROGRESS", "quarters": len(m["financials"])}
        raw, api = client.fetch(endpoint, q)
        rows, quality = normalize(endpoint, raw, q)
        if len(rows) != api["api_total"]:
            raise IntegrityError("API total does not match received rows")
        table = pa.Table.from_pylist(rows, schema=schema(endpoint))
        rel = artifact_path(endpoint, q)
        target = root / rel
        parquet_commit(table, target)
        index[key] = {"endpoint": endpoint, "path": rel, "bytes": target.stat().st_size,
                      "sha256": sha256(target), "quality": quality, "api": api, "completed_utc": utc_now(),
                      "code_revision": code_revision()}
        write_json(path, m)
        if q is not None:
            new += 1
        print(f"{endpoint} {key}: {len(rows):,} rows; quarters {len(m['financials'])}/170", flush=True)
    verify(root, require_sealed=False)
    m["state"], m["sealed_utc"] = "VERIFIED", utc_now()
    write_json(path, m)
    return verify(root)


def source_run(workspace, fresh=False):
    """Resolve or create a pending source run; promote only after full verification."""
    workspace = Path(workspace).resolve()
    pending, active = workspace / "pending_source.json", workspace / "active_source.json"
    if not fresh and pending.is_file():
        root = resolve_pointer(workspace, pending, sealed=False)
    elif not fresh and active.is_file():
        return resolve_pointer(workspace, active)
    else:
        root = workspace / "bronze" / run_id()
        root.mkdir(parents=True, exist_ok=False)
        write_json(pending, {"path": root.relative_to(workspace).as_posix()})
    return root


def resolve_pointer(workspace, pointer=None, sealed=True):
    workspace = Path(workspace).resolve()
    pointer = Path(pointer) if pointer else workspace / "active_source.json"
    p = read_json(pointer)
    root = (workspace / p["path"]).resolve()
    if workspace / "bronze" != root.parent or not root.is_dir():
        raise IntegrityError("Invalid source pointer")
    if sealed and sha256(root / "manifest.json") != p["manifest_sha256"]:
        raise IntegrityError("Source pointer manifest hash mismatch")
    if sealed and read_json(root / "manifest.json")["state"] != "VERIFIED":
        raise IntegrityError("Source pointer does not refer to a verified run")
    return root


def publish_source(workspace, root, report):
    workspace, root = Path(workspace).resolve(), Path(root).resolve()
    if root.parent != workspace / "bronze" or report["status"] != "PASS" or report["source_state"] != "VERIFIED":
        raise IntegrityError("Only a verified Bronze run can be published")
    if report["manifest_sha256"] != sha256(root / "manifest.json"):
        raise IntegrityError("Report does not match source")
    write_json(root / "verification.json", report)
    write_json(workspace / "active_source.json", {"path": root.relative_to(workspace).as_posix(),
                                                   "manifest_sha256": report["manifest_sha256"]})
    pending = workspace / "pending_source.json"
    if pending.exists() and read_json(pending)["path"] == root.relative_to(workspace).as_posix():
        pending.unlink()
