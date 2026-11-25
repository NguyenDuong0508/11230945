"""Full 170-quarter synthetic Bronze-to-Silver integration with real Parquet."""
from datetime import date
from pathlib import Path
import pyarrow.parquet as pq
import pytest
from fdic_ews.acquisition import acquire, publish_source, source_run
from fdic_ews.contracts import FIELDS, IntegrityError
from fdic_ews.quality.panel import run, resolve_panel, panel_schema

class Client:
    def fetch(self,endpoint,q=None):
        def r(**kw): return {**dict.fromkeys(FIELDS[endpoint]),**kw}
        if endpoint=='financials':
            rows=[r(CERT=7213,REPDTE=str(q),ASSET=100,DEP=90,NETINC=q.month,STNAME='NY',RBCRWAJ=0),
                  r(CERT=2,REPDTE=str(q),ASSET=-1,DEP=1,NETINC=-q.month)]
        elif endpoint=='failures':
            rows=[r(CERT=7213,FIN=1,RESTYPE='ASSISTANCE',FAILDATE='20081123'),r(CERT=None,FIN=2,RESTYPE='FAILURE',FAILDATE='19340101')]
        else:
            rows=[r(CERT=7213,ACTIVE='1'),r(CERT=2,ACTIVE='1')]
        return rows,dict(endpoint=endpoint,parameters=dict(fields=','.join(FIELDS[endpoint]),**({'filters':f'REPDTE:{q:%Y%m%d}'} if q else {})),
                         api_total=len(rows),pages=[dict(offset=0,rows=len(rows),api_total=len(rows))])

def test_bronze_silver_roundtrip(tmp_path):
    source=source_run(tmp_path); report=acquire(source,Client()); publish_source(tmp_path,source,report)
    root,m=run(tmp_path)
    assert m['summary']['panel_rows']==340 and m['summary']['asset_to_null']==170 and m['summary']['rows_removed']==0
    assert m['summary']['horizon_case']['full_result']==[0,'observed_at_horizon']
    assert resolve_panel(tmp_path)==(root,m)
    p=root/m['panel']['2006-12-31']['path']; t=pq.ParquetFile(p).read()
    assert t.schema==panel_schema() and t['ASSET'].null_count==1
    assert 'TERMINAL_CLASS' not in t.column_names and 'FAILDATE' not in t.column_names
    with (root/'quality_flags.csv').open('ab') as f: f.write(b'tamper')
    with pytest.raises(IntegrityError,match='Modified Silver'):
        resolve_panel(tmp_path)
