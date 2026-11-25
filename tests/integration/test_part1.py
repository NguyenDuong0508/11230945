"""Real Arrow integration. Missing dependencies are errors, not skipped tests."""
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pytest

from fdic_ews.acquisition import acquire, publish_source, resolve_pointer, source_run, verify
from fdic_ews.contracts import FIELDS, IntegrityError, schema, partition_path, read_json
from fdic_ews.storage import (prepare, verify_preparation, read_csv, query, FORMATS, WORKLOADS,
                              compare_tables, signature, compare_signatures)
from fdic_ews.worker import read_engine
from fdic_ews.benchmark import compare_engines


class FixtureClient:
    def __init__(self):
        self.calls = []
    def fetch(self, endpoint, q=None):
        self.calls.append((endpoint,q))
        def row(**kw):
            return {**dict.fromkeys(FIELDS[endpoint]), **kw}
        if endpoint == "financials":
            rows = [row(CERT=10, REPDTE=str(q), STNAME="", ASSET=1.23456789012345, DEP=50., ROA=-.5),
                    row(CERT=20, REPDTE=str(q), STNAME=None, ASSET=None, DEP=80., ROA=0.),
                    row(CERT=30, REPDTE=str(q), STNAME='A, "B"', ASSET=-1., DEP=None)]
        elif endpoint == "institutions":
            rows = [row(CERT=i, NAME="Fixture") for i in (10,20,30)]
        else:
            rows = [row(CERT=10, FIN=1, RESTYPE="ASSISTANCE", FAILDATE="20081123"),
                    row(CERT=None, FIN=2, RESTYPE="FAILURE", FAILDATE="19340101")]
        return rows, {"endpoint":endpoint,"parameters":{"fields":",".join(FIELDS[endpoint]),
                  **({"filters":f"REPDTE:{q:%Y%m%d}"} if q else {})}, "api_total":len(rows),
                  "pages":[{"offset":0,"rows":len(rows),"api_total":len(rows)}]}


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    ws = tmp_path_factory.mktemp("workspace")
    root = source_run(ws)
    c = FixtureClient()
    assert acquire(root,c,max_new_quarters=1)["status"] == "IN_PROGRESS"
    assert len(c.calls) == 1
    c2 = FixtureClient()
    result = acquire(root,c2)
    assert len(c2.calls) == 171  # 169 remaining quarters + 2 references
    assert result["quarter_count"] == 170 and result["row_counts"] == {"financials":510,"failures":2,"institutions":3}
    publish_source(ws,root,result)
    assert resolve_pointer(ws) == root and not (ws/"pending_source.json").exists()
    assert verify(root) == result
    return root


def test_source_and_sealed_run(source):
    c = FixtureClient()
    assert acquire(source,c)["status"] == "PASS" and not c.calls
    assert verify(source)["failure_null_cert"] == 1
    assert {x["source_value"] for x in verify(source)["failure_restype_counts"]} == {"FAILURE","ASSISTANCE"}


def test_corrupt_checkpoint_recovery(tmp_path):
    root = tmp_path / "source"
    c = FixtureClient()
    acquire(root,c,max_new_quarters=1)
    p = root / partition_path(date(1984,3,31))
    p.write_bytes(b"corrupt")
    c2 = FixtureClient()
    acquire(root,c2,max_new_quarters=1)
    assert c2.calls == [("financials",date(1984,3,31))]
    assert read_json(root/"manifest.json")["state"] == "IN_PROGRESS"


def test_csv_roundtrip_schema_and_nullability(tmp_path):
    s = schema("financials")
    records = [{**dict.fromkeys(s.names), "CERT":i, "REPDTE":date(2006,12,31), "STNAME":v, "ASSET":a}
               for i,v,a in [(1,"",0.),(2,None,None),(3,'A, "quoted"\nline',-2.123456789012345)]]
    table = pa.Table.from_pylist(records,schema=s)
    p = tmp_path / "test.csv"
    pacsv.write_csv(table,p)
    restored = read_csv(p)
    assert restored.schema == s and not restored.schema.field("CERT").nullable
    assert not restored.schema.field("REPDTE").nullable
    compare_tables(table,restored)
    assert restored["STNAME"].to_pylist() == ["",None,'A, "quoted"\nline']


def test_csv_missing_required_value(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("CERT,REPDTE\n,2006-12-31\n")
    with pytest.raises(IntegrityError,match="required field"):
        read_csv(p,["CERT","REPDTE"])


def test_all_storage_representations(source,tmp_path):
    work = tmp_path / "staged"
    m = prepare(source,work)
    assert set(m["formats"]) == set(FORMATS) and verify_preparation(work)==m
    for w in WORKLOADS:
        ref,_=query(work,m,"csv",w)
        expected=signature(ref)
        for f in FORMATS[1:]:
            t,stats=query(work,m,f,w)
            compare_signatures(expected,signature(t))
            if f.startswith("partitioned"):
                assert stats["total_fragments"]==170
                assert stats["candidate_fragments"]==({"year_2020":4,"quarter_2020q1":1}.get(w,170))
    p = work / m["formats"]["csv"]["path"]
    compare_engines(read_engine(p,"pandas")["signature"],read_engine(p,"polars")["signature"])
    with p.open("ab") as f:
        f.write(b"modified")
    with pytest.raises(IntegrityError,match="identity changed"):
        verify_preparation(work)
