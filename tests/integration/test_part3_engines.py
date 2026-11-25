"""Mandatory real Parquet and local Spark integration; no dependency skips."""
from datetime import date
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fdic_ews.contracts import sha256
from fdic_ews.processing.engines import collect, compare, spark_session, WORKLOADS, COLUMNS
from fdic_ews.processing.benchmark import stage


def test_projected_stage_and_all_engines(tmp_path,monkeypatch):
    monkeypatch.setenv('SPARK_LOCAL_IP','127.0.0.1')
    monkeypatch.setenv('PYSPARK_PYTHON',__import__('sys').executable)
    silver=tmp_path/'silver'; silver.mkdir(); panel={}
    spec=pa.schema([('CERT',pa.int64()),('REPDTE',pa.date32()),('ASSET',pa.float64()),('DEP',pa.float64()),('ROA',pa.float64()),('STNAME',pa.string()),('DQ_ROE_ABS_GT_100',pa.bool_()),('EQ',pa.float64())])
    for q,certs,assets,deps,roa,states in [(date(1999,12,31),[1],[80.],[60.],[1.],['NY']),(date(2000,3,31),[1,2,3],[100.,50.,None],[80.,None,None],[1.,3.,None],['NY','NY',None]),(date(2000,6,30),[1,3],[120.,40.],[90.,None],[2.,None],['NY','']),(date(2000,9,30),[2],[100.],[None],[4.],['NY'])]:
        n=len(certs); path=silver/(str(q)+'.parquet'); table=pa.Table.from_arrays([pa.array(v,type=f.type) for f,v in zip(spec,[certs,[q]*n,assets,deps,roa,states,[False]*n,[None]*n])],schema=spec); pq.write_table(table,path)
        panel[str(q)]={'path':path.name,'rows':n,'sha256':sha256(path)}
    root=tmp_path/'stage'; meta=stage(silver,{'panel':panel,'summary':{'panel_rows':7}},root)
    assert [meta[s]['rows'] for s in (1,5,10)]==[7,35,70]
    spark=spark_session()
    try:
        for w in WORKLOADS:
            ref,_=collect('pandas',root/'1',w)
            for engine in ('polars','pyspark'):
                rows,plan=collect(engine,root/'1',w,spark,plan=True); compare(ref,rows,w); assert plan
            for s in (5,10):
                rows,_=collect('polars',root/str(s),w); compare(ref,rows,w,scale=s)
                pandas_rows,_=collect('pandas',root/str(s),w); compare(rows,pandas_rows,w)
    finally: spark.stop()
