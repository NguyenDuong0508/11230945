"""Regression tests for chronological isolation, actual model fits and event economics."""
from datetime import date
import numpy as np
import pandas as pd
import pytest
from fdic_ews.features.financial import FEATURES,CAPITAL
from fdic_ews.features.policy import split_for,TEST_START
from fdic_ews.quality.policy import shift_quarter
from fdic_ews.contracts import quarter_ends,IntegrityError
from fdic_ews.modelling.design import (FOLDS,SPECS,validate_design,validate_rows,split_fold,matrix,fit_model,score_model)
from fdic_ews.modelling.metrics import (event_scores,threshold,operating,ranking,calibration,event_detail)
from fdic_ews.modelling.costs import estimate,parse_asof
from fdic_ews.modelling.pipeline import mark_attempt,model_summary,test_tables as write_test_tables


def fixture_dev(n=40):
    rng=np.random.default_rng(11230945); parts=[]
    for q in quarter_ends(date(1984,6,30),date(2006,12,31)):
        s=split_for(q)
        if s not in ('train','validation'): continue
        x=pd.DataFrame(rng.normal(size=(n,len(FEATURES))),columns=FEATURES)
        x['CERT']=np.arange(n)+10000*(q.year*4+q.month//3)
        x['DECISION_DATE']=pd.Timestamp(q); x['HORIZON_END']=pd.Timestamp(shift_quarter(q,4)); x['SPLIT']=s
        x['Y_FAIL_4Q']=np.r_[np.ones(2,dtype='int8'),np.zeros(n-2,dtype='int8')]
        x['FAILURE_DATE']=pd.Series(pd.NaT,index=x.index,dtype='datetime64[ns]'); x.loc[:1,'FAILURE_DATE']=pd.Timestamp(shift_quarter(q,1))
        x['LOG_ASSET']=np.log1p(1000+np.arange(n)); x.loc[:1,'ROA_PCT']=-5.; x.loc[2,'ROA_PCT']=np.nan; x.loc[3,CAPITAL[0]]=0.
        parts.append(x)
    return pd.concat(parts,ignore_index=True)


def small_pred():
    # Same failure represented by two decisions; cost counts it once.
    return pd.DataFrame({'CERT':[1,1,2,3,4], 'DECISION_DATE':pd.to_datetime(['2008-03-31','2008-06-30','2008-03-31','2008-03-31','2008-06-30']),
        'FAILURE_DATE':pd.to_datetime(['2008-09-30','2008-09-30','2008-12-31',None,None]),'Y_FAIL_4Q':[1,1,1,0,0],'score':[.4,.8,.1,.6,.2]})


def test_fold_design_and_maturity():
    validate_design(); f=fixture_dev(); validate_rows(f,'development')
    for fold in FOLDS:
        t,v=split_fold(f,fold)
        assert t.HORIZON_END.max()<v.DECISION_DATE.min() and t.DECISION_DATE.max()<=pd.Timestamp(fold.train_end)
        assert set(t.SPLIT)=={'train'}


def test_duplicate_and_mismatched_split_rejected():
    f=fixture_dev()
    with pytest.raises(ValueError,match='Duplicate'): validate_rows(pd.concat([f,f.iloc[:1]],ignore_index=True),'development')
    with pytest.raises(ValueError,match='Split/date'): validate_rows(f.assign(SPLIT='validation'),'development')
    with pytest.raises(ValueError,match='Invalid outcome horizon'): validate_rows(f.assign(HORIZON_END=f.DECISION_DATE),'development')


def test_allowlist_and_alternative():
    f=fixture_dev(); x=matrix(f,'lightgbm_full'); z=matrix(f,'lightgbm_zero_missing')
    assert list(x)==list(FEATURES) and 'CERT' not in x and 'Y_FAIL_4Q' not in x
    assert x[CAPITAL[0]].eq(0).sum()>0 and z[CAPITAL[0]].isna().sum()>x[CAPITAL[0]].isna().sum()
    assert len(matrix(f,'lightgbm_core').columns)==21

@pytest.mark.parametrize('name',list(SPECS))
def test_real_model_fit_and_fresh_fold_preprocessing(name):
    f=fixture_dev(); t,v=split_fold(f,FOLDS[0]); m,_=fit_model(t,name)
    p=score_model(m,v,name); assert len(p)==len(v) and p.score.between(0,1).all()
    if name!='prior':
        med=m.named_steps['imputer'].statistics_.copy()
        score_model(m,v.assign(LOG_ASSET=1e8),name)
        np.testing.assert_array_equal(med,m.named_steps['imputer'].statistics_)
    else: assert np.allclose(p.score,t.Y_FAIL_4Q.mean())


def test_test_fit_rejected():
    f=fixture_dev().assign(SPLIT='test')
    with pytest.raises(ValueError,match='development'): fit_model(f,'logistic')


def test_exact_threshold_matches_brute_force():
    p=small_pred(); a=threshold(p,2,5)
    candidates=[2.,*sorted(p.score.unique(),reverse=True)]
    expected=min(candidates,key=lambda t:operating(p,t,2,5)['loss_usd_thousands'])
    assert a['threshold']==expected and a['loss']==operating(p,expected,2,5)['loss_usd_thousands']
    assert len(event_scores(p))==2 and operating(p,2,2,5)['loss_usd_thousands']==10


def test_ties_and_no_alarm():
    p=small_pred().assign(score=.5)
    assert threshold(p,100,1)['threshold']==2.
    assert threshold(p,1,1)['threshold']==2. # equal loss; choose fewer alarms
    assert threshold(p,1,100)['threshold']==.5


def test_event_lead_and_row_metrics_differ():
    p=small_pred(); m=operating(p,.4,2,5)
    assert (m['tp'],m['fp'],m['fn'],m['tn'])==(2,1,1,1)
    assert (m['events'],m['detected_events'],m['missed_events'])==(2,1,1)
    assert m['median_lead_days']==183 and m['loss_usd_thousands']==7
    assert len(event_detail(p,.4))==2 and calibration(p).rows.sum()==5
    assert ranking(p)['ap']>0


def test_no_positive_period_metrics():
    p=small_pred().query('Y_FAIL_4Q==0')
    assert ranking(p)['ap'] is None and ranking(p)['roc_auc'] is None
    assert operating(p,.5,2,5)['event_recall'] is None


def test_cost_vintage_duplicate_and_fallback():
    f=fixture_dev(); pos=f.loc[f.Y_FAIL_4Q.eq(1)].iloc[:25]
    rec=[dict(CERT=int(r.CERT),event_date=r.FAILURE_DATE.date(),event_type='failure',linked=True,date_status='valid',COST=10+i,COSTMOSTRECENTASOF='2007-12-31') for i,r in enumerate(pos.itertuples())]
    a=estimate(f,rec+[rec[0]])
    assert a['eligible_dated_cost_events']==25 and a['c_fn']==22 and a['missed_cost_source']=='dated_fdic_cost'
    b=estimate(f,[{**r,'COSTMOSTRECENTASOF':'2026-01-01'} for r in rec])
    assert b['eligible_dated_cost_events']==0 and b['c_fn']==pytest.approx(.1*b['median_failed_event_assets'])
    assert parse_asof('20071231')==date(2007,12,31) and parse_asof('bad') is None
    # Future-vintage cost cannot change the primary cost estimate.
    assert estimate(f,rec+[{**rec[0],'COST':1e9,'COSTMOSTRECENTASOF':'2026-01-01'}])['c_fn']==a['c_fn']


def test_cost_conflicting_vintages():
    f=fixture_dev(); r=f.loc[f.Y_FAIL_4Q.eq(1)].iloc[0]
    a=dict(CERT=int(r.CERT),event_date=r.FAILURE_DATE.date(),event_type='failure',linked=True,date_status='valid',COST=10,COSTMOSTRECENTASOF='2007-12-31')
    out=estimate(f,[a,{**a,'COST':11}]); assert out['conflicting_dated_cost_events']==1 and out['eligible_dated_cost_events']==0


def test_exclusive_attempt(tmp_path):
    mark_attempt(tmp_path,'seal'); old=(tmp_path/'test_attempt.json').read_bytes()
    with pytest.raises(IntegrityError,match='already attempted'): mark_attempt(tmp_path,'seal')
    assert (tmp_path/'test_attempt.json').read_bytes()==old


def test_macro_selection_and_tie_order():
    rows=[dict(model=n,fold=f.name,ap=.5 if n!='prior' else .01,roc_auc=.5,brier=.1,fit_seconds=1) for n in SPECS for f in FOLDS]
    a,chosen=model_summary(rows); assert chosen=='logistic' and a.selected.sum()==1
    with pytest.raises(IntegrityError): model_summary(rows[:-1])


def test_readonly_test_table_recomputation(tmp_path):
    p=small_pred(); p['HORIZON_END']=p.DECISION_DATE.map(lambda x:x+pd.DateOffset(years=1)); p['SPLIT']='test'
    summary={'selected_model':'logistic','thresholds':{'logistic':.4,'prior':2.},'cost_policy':{'c_fp':2.,'c_fn':5.}}
    preds={'logistic':p,'prior':p.assign(score=.1)}
    write_test_tables(preds,summary,tmp_path); write_test_tables(preds,summary,tmp_path,write=False)
    with (tmp_path/'test_metrics.csv').open('a') as f: f.write('tampered')
    with pytest.raises(IntegrityError): write_test_tables(preds,summary,tmp_path,write=False)
