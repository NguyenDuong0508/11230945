from datetime import date
import pandas as pd
import pytest
from fdic_ews.processing.engines import pandas_transform, compare, normalize, WORKLOADS
from fdic_ews.processing.benchmark import scenarios, validate_trials, summarize, comparisons
from fdic_ews.processing.worker import measured


def fixture():
    # Null aggregate, two-bank peer group, missing predecessor, invalid denominator.
    return pd.DataFrame([
        (1,date(2000,3,31),100.,80.,1.,'NY',False,20.),
        (2,date(2000,3,31),50.,None,3.,'NY',True,10.),
        (3,date(2000,3,31),None,None,None,None,False,None),
        (1,date(2000,6,30),120.,90.,2.,'NY',False,30.),
        (3,date(2000,6,30),40.,None,None,'',False,10.),
        (2,date(2000,9,30),100.,None,4.,'NY',False,10.),
    ],columns=['CERT','REPDTE','ASSET','DEP','ROA','STNAME','DQ_ROE_ABS_GT_100','EQ'])


def test_summary():
    r=pandas_transform(fixture(),'quarter_summary')
    assert r[0]==dict(REPDTE='2000-03-31',ROWS=3,ASSETS=150.,DEPOSITS=80.,MEAN_ROA=2.,EXTREME_ROE=1)
    assert r[2]['DEPOSITS'] is None


def test_growth():
    r=pandas_transform(fixture(),'adjacent_growth')
    assert r[0]['MEAN_GROWTH'] is None
    assert r[1]['ADJACENT']==2 and r[1]['VALID_GROWTH']==1
    assert r[1]['MEAN_GROWTH']==pytest.approx(20.)
    assert r[2]['ADJACENT']==0 and r[2]['MEAN_GROWTH'] is None


def test_join():
    r=pandas_transform(fixture(),'peer_join')
    assert r[0]['MATCHED']==2 and r[0]['MEAN_ABS_DEVIATION']==1.
    assert r[1]['ROWS']==2 and r[1]['MATCHED']==1 and r[1]['MEAN_ABS_DEVIATION']==0.


@pytest.mark.parametrize('workload',WORKLOADS)
def test_replica_invariance(workload):
    frame=fixture(); larger=pd.concat([frame,frame.assign(CERT=frame.CERT+3)],ignore_index=True)
    compare(pandas_transform(frame,workload),pandas_transform(larger,workload),workload,scale=2)


def test_reject_bad_count():
    r=pandas_transform(fixture(),'quarter_summary'); bad=[dict(x) for x in r]; bad[0]['ROWS']+=1
    with pytest.raises(ValueError,match='Parity mismatch'): compare(r,bad,'quarter_summary')


def test_reject_null_mismatch():
    r=pandas_transform(fixture(),'adjacent_growth'); bad=[dict(x) for x in r]; bad[0]['MEAN_GROWTH']=0.
    with pytest.raises(ValueError,match='Parity mismatch'): compare(r,bad,'adjacent_growth')


def test_schedule_and_evidence():
    jobs=scenarios(); assert len(jobs)==len(set(jobs))==24 and jobs==scenarios()
    results=[]; meta={s:dict(rows=6*s,filtered_rows=6*s) for s in (1,5,10)}
    for s,e,w,t in jobs:
        f=fixture(); f=pd.concat([f.assign(CERT=f.CERT+3*i) for i in range(s)])
        rows=pandas_transform(f,w)
        results.append(((s,e,w,t),dict(plan='test plan',trials=[dict(phase='warmup' if i==0 else 'measured',repeat=i,seconds=1.,baseline_rss_mib=20.,peak_rss_mib=21.,incremental_rss_mib=1.,samples=2,result=rows) for i in range(4)])))
    trials=validate_trials(results,meta); assert len(trials)==96
    summary=summarize(trials); assert len(summary)==24 and all(r['n']==3 for r in summary)
    scales,parallel=comparisons(summary); assert len(scales)==18 and len(parallel)==3
    with pytest.raises(Exception,match='Missing scenario'): validate_trials(results[:-1],meta)


def test_memory_sampling():
    value,metrics=measured(lambda: 42)
    assert value==42 and metrics['seconds']>0 and metrics['samples']>=2 and metrics['peak_rss_mib']>=metrics['baseline_rss_mib']
