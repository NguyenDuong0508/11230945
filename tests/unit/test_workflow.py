import csv
from pathlib import Path

import pytest

from fdic_ews.acquisition import source_run, resolve_pointer
from fdic_ews.benchmark import schedule, compare_engines, summarize
from fdic_ews.contracts import IntegrityError, write_json
from fdic_ews.storage import compare_signatures
from fdic_ews.worker import read_engine


def test_run_pointer_resume(tmp_path):
    p = source_run(tmp_path)
    assert p == source_run(tmp_path)
    new = source_run(tmp_path, fresh=True)
    assert new != p and new == source_run(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_pointer(tmp_path)


def test_path_traversal(tmp_path):
    write_json(tmp_path / "pending_source.json", {"path": "../"})
    with pytest.raises(IntegrityError):
        source_run(tmp_path)


def test_schedule():
    jobs = schedule()
    assert len(jobs) == 132 and len([x for x in jobs if x[0] == "measured"]) == 110
    assert jobs == schedule() and jobs != schedule(seed=1)
    assert all(x[0] == "warmup" for x in jobs[:22])
    assert len(set(jobs[:22])) == 22


def test_signature_validation():
    a = {"rows": 2, "keys_sha256": "a", "null_counts": {"ASSET": 1}, "numeric_sums": {"ASSET": 2.0}}
    compare_signatures(a, {**a, "numeric_sums": {"ASSET": 2.00000000001}})
    with pytest.raises(IntegrityError):
        compare_signatures(a, {**a, "keys_sha256": "b"})
    with pytest.raises(IntegrityError):
        compare_signatures(a, {**a, "numeric_sums": {"ASSET": 3.0}})


def test_summary_uses_measurements_only():
    rows = [{"format": "csv", "workload": "full", "phase": p, "seconds": s,
             "peak_rss_mib": 100, "rows": 2, "candidate_fragments": 1, "total_fragments": 1}
            for p,s in [("warmup",99),("measured",1),("measured",2),("measured",3)]]
    assert summarize(rows)[0]["median_seconds"] == 2 and summarize(rows)[0]["n"] == 3


def test_pandas_csv_reader(tmp_path):
    p = tmp_path / "fixture.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f); w.writerows([["CERT","REPDTE","ASSET","DEP","STNAME"],
            [1,"2006-12-31",0,50,"A, Inc."], [2,"2006-12-31",None,70,""], [3,"2007-03-31",-5,None,"NA"]])
    result = read_engine(p, "pandas")
    assert result["signature"] == {"rows":3,"cert_sum":6,"asset_sum":-5.,"dep_sum":120.,
        "asset_nulls":1,"dep_nulls":1,"date_min":"2006-12-31","date_max":"2007-03-31"}
    assert result["seconds"] >= 0
    compare_engines(result["signature"], result["signature"])


def test_cli_help():
    from fdic_ews.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_benchmark_evidence_flow(tmp_path, monkeypatch):
    """Exercise orchestration and artifact hashes with explicitly mocked worker results."""
    import json
    from types import SimpleNamespace
    from fdic_ews import benchmark as b
    from fdic_ews.contracts import read_json, sha256
    from fdic_ews.storage import FORMATS, WORKLOADS

    work, output = tmp_path / "formats", tmp_path / "evidence"
    work.mkdir()
    preparation = {"source_manifest_sha256":"fixture-source", "formats":{f:{"bytes":100+i} for i,f in enumerate(FORMATS)}}
    monkeypatch.setattr(b,"verify_preparation",lambda _:preparation)
    monkeypatch.setattr(b, "im", SimpleNamespace(version=lambda _: "fixture"))
    original_run = b.subprocess.run
    def fake_worker(cmd,**kwargs):
        if "fdic_ews.worker" not in cmd:
            return original_run(cmd, **kwargs)
        f, w = cmd[cmd.index("--format")+1], cmd[cmd.index("--workload")+1]
        if w == "csv_projection":
            sig={"rows":2,"asset_sum":10.0,"dep_sum":5.0}
            scan={}
        else:
            sig={"rows":2,"keys_sha256":"fixture","null_counts":{"ASSET":0},"numeric_sums":{"ASSET":10.0}}
            total=170 if f.startswith("partitioned") else 1
            selected={"year_2020":4,"quarter_2020q1":1}.get(w,170) if total==170 else 1
            scan={"total_fragments":total,"candidate_fragments":selected}
        return SimpleNamespace(returncode=0,stdout=json.dumps({"seconds":0.1,"peak_rss_mib":10,"signature":sig,**scan}),stderr="")
    monkeypatch.setattr(b.subprocess,"run",fake_worker)
    monkeypatch.setattr(b,"code_revision",lambda:None)
    result=b.run(work,output,warmups=1,repetitions=3)
    assert result["total_trials"]==88 and result["measured_trials"]==66
    assert result["status"]=="PASS"
    assert read_json(output/"evidence.json")==result
    assert all(sha256(output/name)==digest for name,digest in result["files"].items())
    assert {"summary.csv","sizes.csv","trials.csv","storage_size.png","latency.png","memory.png"} <= set(result["files"])
