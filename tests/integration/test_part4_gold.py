"""Full-calendar synthetic acquisition, Silver and Gold integration with real Parquet."""
from datetime import date
import pyarrow.parquet as pq
import pytest
from fdic_ews.acquisition import acquire, publish_source, source_run
from fdic_ews.contracts import FIELDS, IntegrityError
from fdic_ews.quality.panel import run as silver_run
from fdic_ews.features.pipeline import run, resolve_features, gold_schema
from fdic_ews.features.financial import FEATURES

FAILURES={1:date(1995,1,15),2:date(2004,1,15),**{i:date(2009,1,15) for i in range(10,61)}}
CERTS=[*FAILURES,7213,9999]

class Client:
    def fetch(self,endpoint,q=None):
        def r(**kw): return {**dict.fromkeys(FIELDS[endpoint]),**kw}
        if endpoint=='financials':
            rows=[r(CERT=c,REPDTE=str(q),STNAME='NY',ASSET=100+c,DEP=80,EQ=10,INTAN=2,NCLNLS=3,ORE=1,LNATRES=2,LNLSGR=50,ROA=1.,ROE=10.,RBC1AAJ=0.,RBCRWAJ=12.) for c in CERTS if c not in FAILURES or q<FAILURES[c]]
        elif endpoint=='failures':
            rows=[r(CERT=c,FIN=c,RESTYPE='FAILURE',FAILDATE=str(d)) for c,d in FAILURES.items()]+[r(CERT=7213,FIN=7213,RESTYPE='ASSISTANCE',FAILDATE='20081123')]
        else: rows=[r(CERT=c,ACTIVE='0' if c in FAILURES else '1',ENDEFYMD=str(FAILURES[c]) if c in FAILURES else '9999-12-31') for c in CERTS]
        return rows,dict(endpoint=endpoint,parameters=dict(fields=','.join(FIELDS[endpoint]),**({'filters':f'REPDTE:{q:%Y%m%d}'} if q else {})),api_total=len(rows),pages=[dict(offset=0,rows=len(rows),api_total=len(rows))])

def test_bronze_silver_gold_roundtrip(tmp_path):
    source=source_run(tmp_path); report=acquire(source,Client()); publish_source(tmp_path,source,report)
    silver_run(tmp_path)
    root,m=run(tmp_path)
    assert m['summary']['test_distinct_failure_certs_in_calendar_period']==51
    assert m['summary']['decision_quarters']==166 and m['summary']['feature_count']==len(FEATURES)
    e=m['panel']['2008-12-31']; t=pq.ParquetFile(root/e['path']).read(); assert t.schema==gold_schema()
    f=t.to_pandas(); assert f.Y_FAIL_4Q.eq(1).sum()==51; assert f.loc[f.CERT.eq(7213),'Y_FAIL_4Q'].iloc[0]==0
    assert resolve_features(tmp_path)==(root,m)
    with (root/'training_feature_quality.csv').open('ab') as f: f.write(b'tampered')
    with pytest.raises(IntegrityError,match='Gold artifact integrity'): resolve_features(tmp_path)
