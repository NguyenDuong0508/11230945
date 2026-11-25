"""Randomized, repeated storage experiments with independently checked outputs."""
from __future__ import annotations

import csv
import importlib.metadata as im
import json
import math
import os
import platform
import random
import statistics as st
import subprocess
import sys
from pathlib import Path

from .acquisition import code_revision
from .contracts import IntegrityError, sha256, write_json, utc_now
from .storage import FORMATS, WORKLOADS, compare_signatures, verify_preparation

SEED = 11230945


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def schedule(warmups=1, repetitions=5, seed=SEED):
    if warmups < 1 or repetitions < 3:
        raise ValueError("At least one warm-up and three measurements")
    rng = random.Random(seed)
    jobs = []
    pairs = [(f, w) for f in FORMATS for w in WORKLOADS] + [(e, "csv_projection") for e in ("pandas", "polars")]
    for phase, n in (("warmup", warmups), ("measured", repetitions)):
        for rep in range(n):
            block = pairs.copy()
            rng.shuffle(block)
            jobs.extend((phase, rep + 1, f, w) for f, w in block)
    return jobs


def compare_engines(a, b):
    for k in a:
        if k in {"asset_sum", "dep_sum"}:
            if not math.isclose(a[k], b[k], rel_tol=1e-10, abs_tol=1e-6):
                raise IntegrityError(f"CSV reader discrepancy: {k}")
        elif a[k] != b[k]:
            raise IntegrityError(f"CSV reader discrepancy: {k}")


def summarize(rows):
    out = []
    for f, w in sorted({(x["format"], x["workload"]) for x in rows}):
        group = [r for r in rows if r["format"] == f and r["workload"] == w and r["phase"] == "measured"]
        t = [r["seconds"] for r in group]
        memory = [r["peak_rss_mib"] for r in group]
        out.append({"format": f, "workload": w, "n": len(t), "median_seconds": st.median(t),
                    "min_seconds": min(t), "max_seconds": max(t), "iqr_seconds": st.quantiles(t, n=4, method="inclusive")[2] - st.quantiles(t, n=4, method="inclusive")[0],
                    "median_peak_rss_mib": st.median(memory), "max_peak_rss_mib": max(memory),
                    "rows": group[0]["rows"], "candidate_fragments": group[0]["candidate_fragments"],
                    "total_fragments": group[0]["total_fragments"]})
    return out


def figures(result, sizes, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["CSV", "Flat Snappy", "Flat ZSTD", "Partitioned Snappy", "Partitioned ZSTD"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(labels[::-1], [sizes[f] / 2**20 for f in FORMATS[::-1]], color="#3D758C")
    ax.set_xlabel("Storage size (MiB)")
    fig.tight_layout(); fig.savefig(result / "storage_size.png", dpi=180); plt.close(fig)
    for metric, title, filename in (("median_seconds", "Median query time (s)", "latency.png"),
                                   ("median_peak_rss_mib", "Median peak worker RSS (MiB)", "memory.png")):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        for ax, workload in zip(axes.flat, WORKLOADS):
            lookup = {x["format"]: x for x in summary if x["workload"] == workload}
            vals = [lookup[f][metric] for f in FORMATS[::-1]]
            ax.barh(labels[::-1], vals, color="#3D758C")
            ax.set_title(workload.replace("_", " ")); ax.set_xlabel(title)
        fig.savefig(result / filename, dpi=180); plt.close(fig)


def run(work, output, warmups=1, repetitions=5, threads=2, seed=SEED):
    import psutil
    work, output = Path(work).resolve(), Path(output).resolve()
    if output.exists() or output == work or output in work.parents or work in output.parents:
        raise IntegrityError("Use a fresh evidence directory outside the staged formats")
    if threads < 1:
        raise ValueError("threads must be positive")
    manifest = verify_preparation(work)
    output.mkdir(parents=True)
    env = dict(os.environ, POLARS_MAX_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    # Worker imports are outside the timed read. Fresh workers isolate peak RSS.
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    signatures, rows, checks = {}, [], []
    jobs = schedule(warmups, repetitions, seed)
    for i, (phase, repeat, fmt, workload) in enumerate(jobs, 1):
        cmd = [sys.executable, "-m", "fdic_ews.worker", "--work", str(work), "--format", fmt,
               "--workload", workload, "--threads", str(threads)]
        proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=1800)
        if proc.returncode:
            raise IntegrityError(f"Worker failed for {fmt}/{workload}:\n{proc.stdout}\n{proc.stderr}")
        record = json.loads(proc.stdout)
        sig = record.pop("signature")
        if workload in signatures:
            (compare_engines if workload == "csv_projection" else compare_signatures)(signatures[workload], sig)
        else:
            signatures[workload] = sig
        if fmt.startswith("partitioned"):
            expected = {"year_2020": 4, "quarter_2020q1": 1}.get(workload, 170)
            if record["candidate_fragments"] != expected or record["total_fragments"] != 170:
                raise IntegrityError(f"Partition pruning contract failed: {fmt}/{workload}")
        checks.append({"trial": i, "workload": workload, "format": fmt, "signature": sig})
        rows.append({"trial": i, "phase": phase, "repeat": repeat, "format": fmt, "workload": workload,
                     "seconds": record["seconds"], "peak_rss_mib": record["peak_rss_mib"], "rows": sig["rows"],
                     "candidate_fragments": record.get("candidate_fragments"), "total_fragments": record.get("total_fragments")})
        print(f"{i}/{len(jobs)} {phase} {fmt} {workload}: {record['seconds']:.4f}s", flush=True)
    verify_preparation(work)
    summary = summarize(rows)
    sizes = {k: manifest["formats"][k]["bytes"] for k in FORMATS}
    write_csv(output / "trials.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "sizes.csv", [{"format": f, "bytes": sizes[f], "mib": sizes[f] / 2**20,
                "reduction_vs_csv_pct": (1 - sizes[f] / sizes["csv"]) * 100} for f in FORMATS])
    write_json(output / "checks.json", checks)
    write_json(output / "preparation.json", manifest)
    figures(output, sizes, summary)
    evidence = {"status": "PASS", "created_utc": utc_now(), "source_manifest_sha256": manifest["source_manifest_sha256"],
                "seed": seed, "threads": threads, "warmups": warmups, "repetitions": repetitions,
                "code_revision": code_revision(),
                "measured_trials": sum(x["phase"] == "measured" for x in rows), "total_trials": len(rows),
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "logical_cpus": os.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
                                "packages": {p: im.version(p) for p in ("pyarrow", "pandas", "polars", "numpy", "matplotlib", "psutil")}},
                "method": "Randomized blocks, warmups before measurements, fresh worker processes, same local filesystem; OS cache not flushed. Timing includes reading, materialization, query metadata and schema restoration, excludes process startup and signature validation. Peak RSS is whole worker, including imports.",
                "files": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "evidence.json", evidence)
    return evidence
