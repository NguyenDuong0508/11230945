"""Synthetic Bronze-to-Gold-to-model workflow with actual Arrow and Spark MLlib."""
from datetime import date
import pytest
from fdic_ews.contracts import FIELDS,IntegrityError
from fdic_ews.acquisition import acquire,publish_source,source_run
from fdic_ews.quality.panel import run as silver_run
from fdic_ews.features.pipeline import run as gold_run
from fdic_ews.modelling.pipeline import develop,evaluate,verify
from fdic_ews.contracts import read_json

FAILURES={1:date(1987,1,15),2:date(1990,1,15),3:date(1995,1,15),4:date(2000,1,15),5:date(2004,1,15),**{i:date(2009,1,15) for i in range(10,61)}}
CERTS=[*FAILURES,7213,9999,10000]

class Client:
    def fetch(self,endpoint,q=None):
        def r(**kw): return {**dict.fromkeys(FIELDS[endpoint]),**kw}
        if endpoint=='financials':
            rows=[r(CERT=c,REPDTE=str(q),STNAME='NY',ASSET=100+c,DEP=80,EQ=10,INTAN=2,NCLNLS=3,ORE=1,LNATRES=2,LNLSGR=50,ROA=float(c%5),ROE=10.,RBC1AAJ=0.,RBCRWAJ=12.) for c in CERTS if c not in FAILURES or q<FAILURES[c]]
        elif endpoint=='failures':
            rows=[r(CERT=c,FIN=c,RESTYPE='FAILURE',FAILDATE=str(d),COST=100+c,COSTMOSTRECENTASOF=str(d)) for c,d in FAILURES.items()]
        else:
            rows=[r(CERT=c,ACTIVE='0' if c in FAILURES else '1',ENDEFYMD=str(FAILURES[c]) if c in FAILURES else '9999-12-31') for c in CERTS]
        return rows,dict(endpoint=endpoint,parameters=dict(fields=','.join(FIELDS[endpoint]),**({'filters':f'REPDTE:{q:%Y%m%d}'} if q else {})),api_total=len(rows),pages=[dict(offset=0,rows=len(rows),api_total=len(rows))])


def test_full_model_evidence_and_single_attempt(tmp_path):
    source=source_run(tmp_path); report=acquire(source,Client()); publish_source(tmp_path,source,report)
    silver_run(tmp_path); gold_run(tmp_path)
    root,m=develop(tmp_path)
    assert m['state']=='SEALED' and m['summary']['candidate_count']==8 and m['summary']['fold_count']==4
    assert read_json(root/'spark_demo.json')['validation_sampled'] is False
    assert not (root.parent/'test_attempt.json').exists()
    result,t=evaluate(tmp_path); assert result==root and t['state']=='VERIFIED'
    assert verify(tmp_path)==(root,t)
    with pytest.raises(IntegrityError,match='already attempted'): evaluate(tmp_path)
    with pytest.raises(IntegrityError,match='already attempted'): develop(tmp_path)
    with (root/'test'/'test_metrics.csv').open('a') as f: f.write('tampered')
    with pytest.raises(IntegrityError,match='seal'): verify(tmp_path)
